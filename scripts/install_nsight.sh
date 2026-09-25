#!/usr/bin/env bash
# Install Nsight Compute (ncu) and Nsight Systems (nsys) CLIs if they are missing.
# Uses NVIDIA's CUDA apt repository for this Ubuntu release. Needs root (Colab/Kaggle are root).
set -uo pipefail
have() { command -v "$1" >/dev/null 2>&1 || ls /usr/local/cuda*/bin/"$1" /opt/nvidia/nsight-*/*/"$1" /opt/nvidia/nsight-*/*/bin/"$1" >/dev/null 2>&1; }
if have ncu && have nsys; then echo "ncu and nsys already installed"; exit 0; fi

. /etc/os-release
REPO="ubuntu${VERSION_ID//./}"
cd /tmp
# Only add NVIDIA's repo if it isn't configured already (Colab ships one; adding a
# second entry with a different keyring breaks apt with "Conflicting values set for option Signed-By").
if ! grep -rqs "developer.download.nvidia.com/compute/cuda/repos" /etc/apt/sources.list /etc/apt/sources.list.d/; then
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/cuda-keyring_1.1-1_all.deb" \
    && dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
fi
apt-get update -qq
# newest versioned packages available in the repo
NC=$(apt-cache pkgnames | grep -E '^nsight-compute-20[0-9.]+$' | sort -V | tail -1)
NS=$(apt-cache pkgnames | grep -E '^nsight-systems-20[0-9.]+$' | sort -V | tail -1)
echo "installing: ${NC:-none} ${NS:-none}"
[ -n "$NC" ] && apt-get install -y -qq "$NC" >/dev/null
[ -n "$NS" ] && apt-get install -y -qq "$NS" >/dev/null

NCU=$(ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)
NSYS=$(ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | sort -V | tail -1)
[ -n "$NCU" ] && ln -sf "$NCU" /usr/local/bin/ncu
[ -n "$NSYS" ] && ln -sf "$NSYS" /usr/local/bin/nsys
echo "ncu:  $(command -v ncu || echo MISSING)"
echo "nsys: $(command -v nsys || echo MISSING)"
