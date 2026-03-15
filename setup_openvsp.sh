#!/usr/bin/env bash
# Link an existing OpenVSP installation into the active Python environment.
#
# Usage:
#   ./setup_openvsp.sh                     # defaults to /opt/OpenVSP
#   ./setup_openvsp.sh /path/to/OpenVSP    # custom install location
#
# This creates a .pth file in your site-packages so that
# `import openvsp` works without modifying PYTHONPATH.

set -euo pipefail

VSP_ROOT="${1:-/opt/OpenVSP}"
VSP_PYTHON="$VSP_ROOT/python"

if [ ! -d "$VSP_PYTHON/openvsp" ]; then
    echo "ERROR: OpenVSP Python API not found at $VSP_PYTHON/openvsp"
    echo ""
    echo "Install OpenVSP first:"
    echo "  https://openvsp.org/download.php"
    echo ""
    echo "Then re-run:  ./setup_openvsp.sh /path/to/OpenVSP"
    exit 1
fi

# Find the site-packages directory for the active Python
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

PTH_FILE="$SITE_PACKAGES/openvsp.pth"

cat > "$PTH_FILE" <<EOF
$VSP_PYTHON/openvsp
$VSP_PYTHON/degen_geom
$VSP_PYTHON/utilities
EOF

echo "Created $PTH_FILE pointing to $VSP_PYTHON"
echo ""
python3 -c "import openvsp; print(f'openvsp imported OK from {openvsp.__file__}')"