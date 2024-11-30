  config:
    install_tree:
      root: /opt/spack
      padded_length: 128

  mirrors:
    local-buildcache:
      url: oci://ghcr.io/timniederhausen/spack-buildcache
      signed: false
