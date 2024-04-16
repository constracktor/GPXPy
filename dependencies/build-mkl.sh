#!/usr/bin/env bash
# This script installs MKL
export FILE_NAME=l_onemkl_p_2024.1.0.695_offline.sh
# structure
export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )/mkl"
export DIR_SRC="$ROOT/src"
export DIR_INSTALL="$ROOT/install"
# get files
export DOWNLOAD_URL="wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/2f3a5785-1c41-4f65-a2f9-ddf9e0db3ea0/${FILE_NAME}"

if [[ ! -d ${DIR_SRC} ]]; then
    (
        mkdir -p ${DIR_SRC}
        cd ${DIR_SRC}
        wget ${DOWNLOAD_URL}
    )
fi

(
    cd ${DIR_SRC}
    chmod +x ${FILE_NAME}
    ./${FILE_NAME} -a --silent --install-dir=${DIR_INSTALL} --eula=accept --intel-sw-improvement-program-consent=decline
)
