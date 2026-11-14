// A replica's pruned projections multiplied compactly on tensor cores (p2p.compact.Tiles; built at first use).
//
// The weights are held in 8 x 8 pieces, each a 64-bit mask (bit r * 8 + c: row r, column c) and its kept values in row
// order, in the order the kernel uses them: a block of RT rows (WARPS warps of 32) takes the columns KCOLS at a time
// (a chunk), and a warp's pieces of a chunk lie together, by step of 16 columns, its two 16-row halves, then the four
// pieces of a tensor-core fragment. woff holds where each warp's values of each chunk start.
//
// For each chunk a block copies the masks, the values and the pass's tokens into shared memory, STAGES chunks ahead (on
// cp.async); each warp counts its pieces' kept weights to find where their values start, then builds its fragments
// straight from the masks: the 32 lanes of a warp read one piece's values at once, which lie within 128 bytes, so they
// don't clash. The tokens come in by ldmatrix, once for both halves. bfloat16 or float16, float32 sums, up to 64 tokens a
// launch; a matrix of few blocks is split along its columns too, the parts added in a fixed order.
// P2P_WARPS, P2P_KCOLS and P2P_STAGES set the shape at build time.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

#ifndef P2P_WARPS
#define P2P_WARPS 4
#endif
#ifndef P2P_KCOLS
#define P2P_KCOLS 64
#endif
#ifndef P2P_STAGES
#define P2P_STAGES 2
#endif
constexpr int WARPS = P2P_WARPS, RT = 32 * WARPS;   // rows a block takes
constexpr int KCOLS = P2P_KCOLS;            // columns a chunk holds
constexpr int KS = KCOLS / 16;              // steps of 16 columns a chunk takes
constexpr int SUB = KS * 2 * 4;             // pieces a warp holds in a chunk
constexpr int XSTRIDE = KCOLS + 8;          // a token's row in shared memory, in elements: padded so ldmatrix doesn't clash
constexpr int STAGES = P2P_STAGES;

__device__ __forceinline__ void cp16(void* smem, const void* gmem, int bytes) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait_for() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4(uint32_t* r, const void* p) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldmatrix2(uint32_t* r, const void* p) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}

template <typename T> __device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b);
template <> __device__ __forceinline__ void mma<__nv_bfloat16>(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <> __device__ __forceinline__ void mma<__half>(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <typename T> __device__ __forceinline__ T cast(float v);
template <> __device__ __forceinline__ __nv_bfloat16 cast<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half cast<__half>(float v) { return __float2half(v); }

// A lane's two adjacent weights of an 8 x 8 piece (row g, columns 2t and 2t + 1: bits q and q + 1 of the mask's word
// that holds row g, whose lower bits are below) as a fragment register; vs + at is where that word's values start.
__device__ __forceinline__ uint32_t pair(const uint16_t* vs, uint32_t word, uint32_t below, int q, int at) {
    const int i = at + __popc(word & below);
    const uint32_t bits = word >> q;
    const uint32_t a = vs[i], b = vs[i + (bits & 1u)];
    return ((bits & 1u) ? a : 0u) | ((bits & 2u) ? b << 16 : 0u);
}

template <typename T, int NT>
__global__ void __launch_bounds__(32 * WARPS) spmm_kernel(
    const T* __restrict__ x, int ntok, int cols, const T* __restrict__ vals, const int64_t* __restrict__ woff,
    const uint64_t* __restrict__ masks, int rows, int nchunks, int per_split, int vcap, T* __restrict__ y,
    float* __restrict__ part) {
    constexpr int THREADS = 32 * WARPS, TOK = NT * 8;
    constexpr int XBYTES = TOK * XSTRIDE * 2, MBYTES = WARPS * SUB * 8;
    extern __shared__ __align__(16) unsigned char smem[];
    const int stage_bytes = XBYTES + MBYTES + vcap * 2;
    uint4* pieces = reinterpret_cast<uint4*>(smem + STAGES * stage_bytes);     // [WARPS][SUB]: a piece's lane word
    int64_t* offs = reinterpret_cast<int64_t*>(pieces + WARPS * SUB);          // and where its values start; woff
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x, j0 = blockIdx.y * per_split, j1 = min(nchunks, j0 + per_split);
    const int n_off = (j1 - j0) * WARPS + 1;
    for (int i = tid; i < n_off; i += THREADS) offs[i] = woff[((size_t)tile * nchunks + j0) * WARPS + i];
    __syncthreads();

    auto load = [&](int j, int s) {
        unsigned char* base = smem + s * stage_bytes;
        uint16_t* xs = reinterpret_cast<uint16_t*>(base);
        uint64_t* ms = reinterpret_cast<uint64_t*>(base + XBYTES);
        uint16_t* vs = reinterpret_cast<uint16_t*>(base + XBYTES + MBYTES);
        const int col0 = j * KCOLS;
        for (int i = tid; i < TOK * (KCOLS / 8); i += THREADS) {          // the tokens, 16 bytes at a time
            const int n = i / (KCOLS / 8), c = col0 + (i % (KCOLS / 8)) * 8;
            const bool ok = n < ntok && c < cols;
            cp16(xs + n * XSTRIDE + (c - col0), ok ? x + (size_t)n * cols + c : x, ok ? 16 : 0);
        }
        const uint64_t* mg = masks + ((size_t)tile * nchunks + j) * WARPS * SUB;
        for (int i = tid; i < WARPS * SUB / 2; i += THREADS) cp16(ms + i * 2, mg + i * 2, 16);
        const int64_t a = offs[(j - j0) * WARPS], b = offs[(j - j0 + 1) * WARPS];
        const int64_t lo = a & ~int64_t(7);
        const int n16 = (int)((b - lo + 7) >> 3);
        for (int i = tid; i < n16; i += THREADS) cp16(vs + i * 8, vals + lo + i * 8, 16);
    };

    float acc[2][NT][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[m][nt][q] = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (j0 + s < j1) load(j0 + s, s);
        commit();
    }
    const bool upper = g >= 4;                    // this lane's row of a piece is in the mask's upper word
    const int q = (g & 3) * 8 + 2 * t;
    const uint32_t below = (1u << q) - 1u;
    for (int j = j0; j < j1; ++j) {
        wait_for<STAGES - 2>();                   // this chunk is in, and every warp is done with the last one's stage
        __syncthreads();
        const int jn = j + STAGES - 1;
        if (jn < j1) load(jn, (jn - j0) % STAGES);
        commit();
        const unsigned char* base = smem + ((j - j0) % STAGES) * stage_bytes;
        const uint16_t* xs = reinterpret_cast<const uint16_t*>(base);
        const uint2* ms = reinterpret_cast<const uint2*>(base + XBYTES) + warp * SUB;
        const uint16_t* vs = reinterpret_cast<const uint16_t*>(base + XBYTES + MBYTES);
        const int64_t lo = offs[(j - j0) * WARPS] & ~int64_t(7);
        uint4* mine = pieces + warp * SUB;
        {                                         // where each of this warp's pieces' values start: a scan of their counts
            constexpr int PER = SUB / 32;
            uint2 m[PER];
            int c[PER], sum = 0;
#pragma unroll
            for (int k = 0; k < PER; ++k) {
                m[k] = ms[lane * PER + k];
                c[k] = sum;
                sum += __popc(m[k].x) + __popc(m[k].y);
            }
            int incl = sum;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, incl, d);
                if (lane >= d) incl += v;
            }
            const int start = (int)(offs[(j - j0) * WARPS + warp] - lo) + incl - sum;
#pragma unroll
            for (int k = 0; k < PER; ++k)
                mine[lane * PER + k] = make_uint4(m[k].x, m[k].y, start + c[k], start + c[k] + __popc(m[k].x));
            __syncwarp();
        }
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            uint32_t b[NT][2];
            const int k = ks * 16 + ((lane >> 3) & 1) * 8;
            if constexpr (NT == 1) {
                ldmatrix2(b[0], xs + (lane & 7) * XSTRIDE + k);
            } else {
#pragma unroll
                for (int np = 0; np < NT / 2; ++np) {
                    uint32_t r[4];
                    ldmatrix4(r, xs + ((2 * np + (lane >> 4)) * 8 + (lane & 7)) * XSTRIDE + k);
                    b[2 * np][0] = r[0];
                    b[2 * np][1] = r[1];
                    b[2 * np + 1][0] = r[2];
                    b[2 * np + 1][1] = r[3];
                }
            }
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                uint32_t a[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint4 pc = mine[(ks * 2 + m) * 4 + k];
                    a[k] = pair(vs, upper ? pc.y : pc.x, below, q, upper ? pc.w : pc.z);
                }
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) mma<T>(acc[m][nt], a, b[nt]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int r = tile * RT + warp * 32 + m * 16 + g;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int n = nt * 8 + 2 * t + (q & 1), rr = r + (q >> 1) * 8;
                if (n < ntok && rr < rows) {
                    if (part) part[((size_t)blockIdx.y * ntok + n) * rows + rr] = acc[m][nt][q];
                    else y[(size_t)n * rows + rr] = cast<T>(acc[m][nt][q]);
                }
            }
        }
    }
}

template <typename T>
__global__ void sum_parts(const float* __restrict__ part, int splits, int n, T* __restrict__ y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.f;
    for (int k = 0; k < splits; ++k) s += part[(size_t)k * n + i];
    y[i] = cast<T>(s);
}

template <typename T, int NT>
int launch(const torch::Tensor& x, const torch::Tensor& vals, const torch::Tensor& woff, const torch::Tensor& masks,
           int rows, int cols, int vcap, int splits, torch::Tensor* y) {
    const int ntok = x.size(0), nchunks = (cols + KCOLS - 1) / KCOLS, tiles = (rows + RT - 1) / RT;
    const int per_split = (nchunks + splits - 1) / splits, nsplit = (nchunks + per_split - 1) / per_split;
    const int stage = NT * 8 * XSTRIDE * 2 + WARPS * SUB * 8 + vcap * 2;
    const int bytes = STAGES * stage + WARPS * SUB * 16 + (per_split * WARPS + 1) * 8;
    if (!y) return bytes;                                    // the shared memory it takes
    auto kernel = spmm_kernel<T, NT>;
    if (bytes > 48 * 1024)
        TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess,
                    "p2p_compact: ", bytes, " bytes of shared memory");
    auto stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor part;
    if (nsplit > 1) part = torch::empty({nsplit, ntok, rows}, x.options().dtype(torch::kFloat32));
    kernel<<<dim3(tiles, nsplit), 32 * WARPS, bytes, stream>>>(
        reinterpret_cast<const T*>(x.data_ptr()), ntok, cols, reinterpret_cast<const T*>(vals.data_ptr()),
        woff.data_ptr<int64_t>(), reinterpret_cast<const uint64_t*>(masks.data_ptr()), rows, nchunks, per_split, vcap,
        reinterpret_cast<T*>(y->data_ptr()), nsplit > 1 ? part.data_ptr<float>() : nullptr);
    if (nsplit > 1) {
        const int n = ntok * rows;
        sum_parts<T><<<(n + 255) / 256, 256, 0, stream>>>(part.data_ptr<float>(), nsplit, n,
                                                          reinterpret_cast<T*>(y->data_ptr()));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return bytes;
}

template <typename T>
int by_tokens(const torch::Tensor& x, const torch::Tensor& vals, const torch::Tensor& woff, const torch::Tensor& masks,
              int rows, int cols, int vcap, int splits, torch::Tensor* y) {
    const int n = x.size(0);
    if (n <= 8) return launch<T, 1>(x, vals, woff, masks, rows, cols, vcap, splits, y);
    if (n <= 16) return launch<T, 2>(x, vals, woff, masks, rows, cols, vcap, splits, y);
    if (n <= 32) return launch<T, 4>(x, vals, woff, masks, rows, cols, vcap, splits, y);
    return launch<T, 8>(x, vals, woff, masks, rows, cols, vcap, splits, y);
}

}  // namespace

// y [T, rows] = x [T, cols] @ W.T for the W the pieces hold; T <= 64.
torch::Tensor spmm(torch::Tensor x, torch::Tensor vals, torch::Tensor woff, torch::Tensor masks, int64_t rows,
                   int64_t cols, int64_t vcap, int64_t splits) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.size(1) == cols && x.size(0) <= 64);
    TORCH_CHECK(x.scalar_type() == vals.scalar_type() && cols % 8 == 0);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(vals.data_ptr()) % 16 == 0);
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto y = torch::empty({x.size(0), rows}, x.options());
    if (x.size(0) == 0) return y;
    if (x.scalar_type() == torch::kBFloat16) by_tokens<__nv_bfloat16>(x, vals, woff, masks, rows, cols, vcap, splits, &y);
    else if (x.scalar_type() == torch::kFloat16) by_tokens<__half>(x, vals, woff, masks, rows, cols, vcap, splits, &y);
    else TORCH_CHECK(false, "p2p_compact: bfloat16 or float16");
    return y;
}

// The shared memory a launch for this many tokens takes.
int64_t shared_bytes(int64_t tokens, int64_t cols, int64_t vcap, int64_t splits) {
    const int nchunks = (cols + KCOLS - 1) / KCOLS, per = (nchunks + splits - 1) / splits;
    const int nt = tokens <= 8 ? 1 : tokens <= 16 ? 2 : tokens <= 32 ? 4 : 8;
    return STAGES * (nt * 8 * XSTRIDE * 2 + WARPS * SUB * 8 + vcap * 2) + WARPS * SUB * 16 + (per * WARPS + 1) * 8;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spmm", &spmm, "compact pieces times tokens");
    m.def("shared_bytes", &shared_bytes, "shared memory a launch takes");
    m.attr("RT") = RT;
    m.attr("KCOLS") = KCOLS;
    m.attr("WARPS") = WARPS;
}
