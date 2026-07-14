#!/usr/bin/env bash
# Install Nocturne to the desktop: binary, icons, launcher.
#
# Everything lands under $HOME (~/.local), so this needs no root and touches nothing the system
# package manager owns. Re-running it is an upgrade — every step overwrites in place.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${PREFIX:-$HOME/.local}"
bindir="$prefix/bin"
appdir="$prefix/share/applications"
icons="$prefix/share/icons/hicolor"

echo "==> building (release)"
cargo build --release --manifest-path "$root/Cargo.toml"

echo "==> installing binary to $bindir"
install -Dm755 "$root/target/release/nocturne" "$bindir/nocturne"

# One PNG per size, in the hicolor theme where the desktop actually looks for them. The .desktop
# says `Icon=nocturne`, which is a *theme lookup*, not a path — miss these and you get a blank tile.
echo "==> installing icons to $icons"
for size in 32 48 64 128 256; do
    install -Dm644 "$root/dist/nocturne-$size.png" \
        "$icons/${size}x${size}/apps/nocturne.png"
done

echo "==> installing launcher to $appdir"
install -Dm644 "$root/dist/nocturne.desktop" "$appdir/nocturne.desktop"

# Best-effort: without these the launcher still works, it just may not appear until the next login.
if command -v update-desktop-database >/dev/null; then
    update-desktop-database "$appdir" || true
fi
# Only meaningful if this icon tree is a real theme. A user-local hicolor dir usually has no
# index.theme, and gtk-update-icon-cache refuses to build a cache without one — desktops fall back
# to scanning the directory, which works fine, so don't cry wolf about it.
if command -v gtk-update-icon-cache >/dev/null && [ -f "$icons/index.theme" ]; then
    gtk-update-icon-cache -qtf "$icons" || true
fi

echo
echo "Nocturne installed."
echo "  binary:   $bindir/nocturne"
echo "  launcher: $appdir/nocturne.desktop"
case ":$PATH:" in
    *":$bindir:"*) ;;
    *) echo "  note: $bindir is not on your PATH" ;;
esac
