#!/bin/bash
# Install cuda-freeze watchdog + llama-server + systemd service.
# Usage: sudo bash install-cuda-freeze.sh [build_dir]
set -euo pipefail

BUILD="${1:-build}"

if [ ! -f "$BUILD/bin/llama-server" ] || [ ! -f "$BUILD/bin/llama-freeze-watchdog" ]; then
    echo "Error: binaries not found in $BUILD/bin/"
    echo "Build first: cmake -B build -DGGML_CUDA=ON && cmake --build build -j\$(nproc)"
    exit 1
fi

echo "Installing binaries to /usr/local/bin/..."
cp "$BUILD/bin/llama-server" /usr/local/bin/
cp "$BUILD/bin/llama-freeze-watchdog" /usr/local/bin/
chmod +x /usr/local/bin/llama-server /usr/local/bin/llama-freeze-watchdog

echo "Installing systemd service..."
cp scripts/llama-server.service /etc/systemd/system/
systemctl daemon-reload

echo ""
echo "=== Installed ==="
echo ""
echo "1. Edit model path in /etc/systemd/system/llama-server.service"
echo "2. sudo systemctl enable --now llama-server"
echo "3. curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\":\"gemma\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'"
echo ""
echo "Logs: journalctl -u llama-server -f"
