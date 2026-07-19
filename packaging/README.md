# libtriton_jit Packaging

This directory contains packaging configurations for building Debian (.deb) and RPM packages for libtriton_jit.

## Prerequisites

- Docker
- Docker Buildx (for multi-platform builds)

## Building Debian Packages

### Using the build script

```bash
cd packaging/debian/build-helpers
./build-libtriton-jit.sh --base-image nvidia/cuda:12.8.0-devel-ubuntu22.04 --output-dir ./output
```

### Manual build

```bash
cd packaging/debian
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04 -f Dockerfile.deb -t libtriton-jit-builder ../../
```

## Building RPM Packages

### Using the build script

```bash
cd packaging/rpm
./build-rpm.sh --base-image nvidia/cuda:12.6.0-devel-rockylinux9 --output-dir ./output
```

### Manual build

```bash
cd packaging/rpm
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.6.0-devel-rockylinux9 -f Dockerfile.rpm -t libtriton-jit-rpm-builder ../..
```

## Package Contents

### libtriton-jit-nvidia (Runtime Package)
- `/usr/lib/*/libtriton_jit.so` - Shared library (not soname-versioned)
- `/usr/share/triton_jit/scripts/*.py` - Python helper scripts

### libtriton-jit-nvidia-dev (Development Package)
- `/usr/include/triton_jit/` - Header files
- `/usr/lib/*/cmake/TritonJIT/` - CMake configuration files
- depends on the distro `libfmt-dev` (>= 8.1.1) for the fmt headers the
  exported CMake target references (no more bundled fmt or
  `Conflicts: libfmt-dev`; the RPM package still bundles fmt because the
  Rocky Linux fmt is too old)

## GitHub Actions

The `.github/workflows/build-deb.yml` and `build-rpm.yml` workflows build
packages on tag push (`v*`) and on PRs that touch packaging, targeting the
FlagOS NVIDIA environment:
- Debian packages on Ubuntu 22.04 + CUDA 12.8
- RPM packages on Rocky Linux 9 + CUDA 12.6

## Dependencies

### Build Dependencies
- CMake >= 3.26
- Ninja build system
- CUDA Toolkit
- Python 3 development files
- PyTorch >= 2.5.0
- Triton >= 3.1.0
- pybind11
- nlohmann-json
- fmt >= 10.2.1

### Runtime Dependencies
- PyTorch
- Triton
- CUDA runtime

## Notes

- pybind11 is supplied externally (via pip); fmt and nlohmann-json come from the distro packages (libfmt-dev >= 8.1.1, nlohmann-json3-dev >= 3.10.5) for the deb build; the RPM build still fetches them via FetchContent (the Rocky Linux base repos do not provide suitable versions)
- RPATH is removed from the shared libraries during packaging
- Examples are not built in the packages to reduce build time
