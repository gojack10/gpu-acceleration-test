#!/bin/bash

# Exit on error
set -e

echo "Building WebAssembly module..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack is not installed. Installing..."
    cargo install wasm-pack
fi

# Build the WebAssembly module
wasm-pack build --target web --out-dir pkg

echo "WebAssembly module built successfully!"
echo "The output is in the 'pkg' directory."
echo ""
echo "To serve the application locally, you can use a simple HTTP server:"
echo "  python -m http.server"
echo "  or"
echo "  npx serve"
echo ""
echo "Then open http://localhost:8000 (or the port shown) in your browser." 