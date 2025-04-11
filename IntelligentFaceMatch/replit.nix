{pkgs}: {
  deps = [
    pkgs.libxcrypt
    pkgs.libGLU
    pkgs.libGL
    pkgs.zlib
    pkgs.pkg-config
    pkgs.grpc
    pkgs.c-ares
    pkgs.postgresql
    pkgs.openssl
  ];
}
