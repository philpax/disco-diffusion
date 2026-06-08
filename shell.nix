{ pkgs ? import <nixpkgs> {} }:

let
  electronDeps = with pkgs; [
    glib
    nss
    nspr
    atk
    cups
    dbus
    libdrm
    gtk3
    pango
    cairo
    libx11
    libxcomposite
    libxdamage
    libxext
    libxfixes
    libxrandr
    libxcb
    mesa
    libgbm
    libGL
    libxkbcommon
    expat
    alsa-lib
    at-spi2-atk
    at-spi2-core
    libxshmfence
  ];
in
pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    pkg-config
    zenity  # native file dialogs for the studio (crossfiledialog drives it on Linux)
  ];

  buildInputs = electronDeps;

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (electronDeps ++ [
    pkgs.stdenv.cc.cc.lib    # libstdc++.so.6 (needed by PyTorch native extensions)
    pkgs.zlib                # libz.so.1 (needed by numpy, opencv, etc.)
    "/run/opengl-driver"     # libcuda.so, libnvidia-ml.so (NVIDIA driver)
  ]);

  # Triton calls /sbin/ldconfig to find libcuda.so, which doesn't exist on NixOS.
  # Point it directly at the driver library path instead.
  TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";

  # The UV-managed venv's sysconfig reports /run/current-system/sw/include/python3.13
  # as the include path, but on NixOS the actual Python.h lives in the nix store.
  # Triton needs this to compile CUDA utility C extensions with gcc.
  C_INCLUDE_PATH = "${pkgs.python313}/include/python3.13";
}
