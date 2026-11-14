#!/bin/bash
# netns.sh D cmd...: run cmd in a throwaway network namespace whose loopback delays every packet by D ms, one way, with
# NCCL and gloo forced onto sockets over it, so processes on one machine talk as nodes on a network would.
D=$1; shift
export P2P_NET_DELAY_MS=$D        # for the runs' records
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
exec unshare -rn sh -c 'ip link set lo up && { [ "$0" = 0 ] || tc qdisc add dev lo root netem delay "$0"ms limit 100000; } && exec "$@"' "$D" "$@"
