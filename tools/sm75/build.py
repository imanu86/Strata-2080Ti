"""Explicit CMake build of SM75 daily source; never starts the engine."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--arch', default='75;86')
    ap.add_argument('--ggml-dir', type=Path)
    ap.add_argument('--build-dir', type=Path, default=ROOT/'build-sm75')
    a = ap.parse_args()
    if a.jobs < 1:
        ap.error('jobs must be positive')
    command = ['cmake', '-S', str(ROOT), '-B', str(a.build_dir.resolve()), '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES='+a.arch,
        '-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared', '-DSTRATA_PORTABLE=OFF',
        '-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_ENABLE_HIP=OFF', '-DSTRATA_BUILD_TESTS=OFF',
        '-DSTRATA_NATIVE_EXPERTS=ON']
    if os.name == 'nt':
        command.append('-DCMAKE_CUDA_FLAGS=-D_WINDOWS -Xcompiler=/EHsc')
    if a.ggml_dir:
        command.append('-DSTRATA_GGML_DIR='+str(a.ggml_dir.resolve(strict=True)))
    subprocess.run(command, check=True)
    subprocess.run(['cmake', '--build', str(a.build_dir.resolve()), '--parallel', str(a.jobs),
                    '--target', 'strata'], check=True)


if __name__ == '__main__':
    main()
