#!/usr/bin/env bash
# This script installs MKL
export MKL_VERSION=2023.0.0
# structure
export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )/mkl"
export DIR_SRC="$ROOT/src"
export DIR_INSTALL="$ROOT/install"
# get files
export DOWNLOAD_URL="https://registrationcenter-download.intel.com/akdlm/irc_nas/19138/l_onemkl_p_${MKL_VERSION}.25398_offline.sh"

if [[ ! -d ${DIR_SRC} ]]; then
    (
        mkdir -p ${DIR_SRC}
        cd ${DIR_SRC}
        wget ${DOWNLOAD_URL}
    )
fi

(
    cd ${DIR_SRC}
    chmod +x ./l_onemkl_p_${MKL_VERSION}.25398_offline.sh
    ./l_onemkl_p_${MKL_VERSION}.25398_offline.sh -a --silent --install-dir=${DIR_INSTALL} --eula=accept --intel-sw-improvement-program-consent=decline
)