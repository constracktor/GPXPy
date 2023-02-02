#!/bin/bash
: ${SOURCE_ROOT:?} ${INSTALL_ROOT:?} ${MKL_VERSION:?}

DIR_SRC=${SOURCE_ROOT}/mkl
DIR_INSTALL=${INSTALL_ROOT}/mkl

DOWNLOAD_URL="https://registrationcenter-download.intel.com/akdlm/irc_nas/19138/l_onemkl_p_${MKL_VERSION}.25398_offline.sh"

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