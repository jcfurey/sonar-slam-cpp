#!/usr/bin/env bash
# Lyrical source fallback; pins match nautilus_ws/scripts/docker/install_libpointmatcher.sh.
set -euo pipefail
apt-get install -y build-essential cmake curl ca-certificates ninja-build \
  libboost-date-time-dev libboost-program-options-dev libboost-thread-dev \
  libeigen3-dev libyaml-cpp-dev
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
for specification in \
  libnabo:cd95c8b95e8620e6c4ba722ed0f56974b4740873 \
  libpointmatcher:3eb448330b2c35f7718e690e66f0c1057115e877; do
  library=${specification%%:*}
  revision=${specification#*:}
  mkdir "$build_dir/$library"
  curl --fail --silent --show-error --location --retry 3 \
    "https://github.com/norlab-ulaval/$library/archive/$revision.tar.gz" \
    --output "$build_dir/$library.tar.gz"
  tar -xzf "$build_dir/$library.tar.gz" --strip-components=1 -C "$build_dir/$library"
done
cmake -S "$build_dir/libnabo" -B "$build_dir/libnabo/build" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DSHARED_LIBS=ON \
  -DLIBNABO_BUILD_EXAMPLES=OFF -DLIBNABO_BUILD_TESTS=OFF \
  -DLIBNABO_BUILD_PYTHON=OFF -DLIBNABO_BUILD_DOXYGEN=OFF
cmake --build "$build_dir/libnabo/build" --parallel 2
cmake --install "$build_dir/libnabo/build"
sed -i 's/COMPONENTS thread system /COMPONENTS thread /g' \
  "$build_dir/libpointmatcher/CMakeLists.txt" "$build_dir/libpointmatcher/libpointmatcherConfig.cmake.in"
cmake -S "$build_dir/libpointmatcher" -B "$build_dir/libpointmatcher/build" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=ON \
  -DBUILD_EXAMPLES=OFF -DBUILD_EVALUATIONS=OFF -DBUILD_TESTS=OFF -DBUILD_PYTHON_MODULE=OFF
cmake --build "$build_dir/libpointmatcher/build" --parallel 2
cmake --install "$build_dir/libpointmatcher/build"
ldconfig
