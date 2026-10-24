#!/bin/sh
# Ubuntu has not published a CVE-2026-111497 backport yet. Build checksum-pinned
# upstream 1.2.1 and upgrade the actual distro packages, including static libs,
# so curl, the worker, and later dependency builds all use the patched decoder.
set -eu

brotli_version=1.2.1
brotli_package_version=1.2.1-0devxt1
brotli_sha256=98d35d11a78dc453aebb7d10c3748f3f73ff78c8267c0dc697b8631f67bf7e8f
brotli_work=$(mktemp -d)
trap 'rm -rf "$brotli_work"' EXIT HUP INT TERM
brotli_arch=$(dpkg --print-architecture)
brotli_multiarch=$(gcc -print-multiarch)

curl --fail --location --silent --show-error \
    "https://github.com/google/brotli/archive/refs/tags/v${brotli_version}.tar.gz" \
    -o "$brotli_work/source.tar.gz"
printf '%s  %s\n' "$brotli_sha256" "$brotli_work/source.tar.gz" | sha256sum -c -
tar -xzf "$brotli_work/source.tar.gz" -C "$brotli_work"
brotli_source="$brotli_work/brotli-$brotli_version"
cmake -S "$brotli_source" -B "$brotli_work/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr \
    -DCMAKE_INSTALL_LIBDIR="lib/$brotli_multiarch" \
    -DBROTLI_BUILD_FOR_PACKAGE=ON
cmake --build "$brotli_work/build" --parallel 2
ctest --test-dir "$brotli_work/build" --output-on-failure
DESTDIR="$brotli_work/install" cmake --install "$brotli_work/build"

brotli_lib="$brotli_work/libbrotli1"
brotli_dev="$brotli_work/libbrotli-dev"
mkdir -p "$brotli_lib/DEBIAN" "$brotli_lib/usr/lib/$brotli_multiarch" \
    "$brotli_lib/usr/share/doc/libbrotli1" "$brotli_dev/DEBIAN" \
    "$brotli_dev/usr/lib/$brotli_multiarch/pkgconfig" "$brotli_dev/usr/include" \
    "$brotli_dev/usr/share/doc/libbrotli-dev"
cp -a "$brotli_work/install/usr/include/brotli" "$brotli_dev/usr/include/"
cp -a "$brotli_work/install/usr/lib/$brotli_multiarch/pkgconfig/." \
    "$brotli_dev/usr/lib/$brotli_multiarch/pkgconfig/"
for brotli_name in common dec enc; do
    cp -a "$brotli_work/install/usr/lib/$brotli_multiarch/libbrotli$brotli_name.so."* \
        "$brotli_lib/usr/lib/$brotli_multiarch/"
    cp -a "$brotli_work/install/usr/lib/$brotli_multiarch/libbrotli$brotli_name.so" \
        "$brotli_dev/usr/lib/$brotli_multiarch/"
    cp "$brotli_work/install/usr/lib/$brotli_multiarch/libbrotli$brotli_name-static.a" \
        "$brotli_dev/usr/lib/$brotli_multiarch/libbrotli$brotli_name.a"
done
cp "$brotli_source/LICENSE" "$brotli_lib/usr/share/doc/libbrotli1/copyright"
cp "$brotli_source/LICENSE" "$brotli_dev/usr/share/doc/libbrotli-dev/copyright"
cat > "$brotli_lib/DEBIAN/control" <<EOF
Package: libbrotli1
Source: brotli
Version: $brotli_package_version
Architecture: $brotli_arch
Multi-Arch: same
Section: libs
Priority: optional
Maintainer: DevXT <support@devxt.com>
Depends: libc6
Description: Brotli shared libraries with the CVE-2026-111497 decoder fix
 Built from checksum-verified upstream Brotli $brotli_version.
EOF
cat > "$brotli_dev/DEBIAN/control" <<EOF
Package: libbrotli-dev
Source: brotli
Version: $brotli_package_version
Architecture: $brotli_arch
Multi-Arch: same
Section: libdevel
Priority: optional
Maintainer: DevXT <support@devxt.com>
Depends: libbrotli1 (= $brotli_package_version)
Description: Brotli headers and static libraries with CVE-2026-111497 fixed
 Built from checksum-verified upstream Brotli $brotli_version.
EOF
dpkg-deb --build --root-owner-group "$brotli_lib" "$brotli_work/libbrotli1.deb"
dpkg-deb --build --root-owner-group "$brotli_dev" "$brotli_work/libbrotli-dev.deb"
dpkg -i "$brotli_work/libbrotli1.deb" "$brotli_work/libbrotli-dev.deb"
ldconfig
[ "$(pkg-config --modversion libbrotlidec)" = "$brotli_version" ]
dpkg-query -W libbrotli1 libbrotli-dev
