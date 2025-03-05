# GPU Acceleration Test - WebGL Integration

This project demonstrates GPU acceleration using WebGL through WebAssembly. The application can run both as a native desktop application and as a web application in the browser.

## Features

- GPU-accelerated rendering
- WebGL integration for browser support
- Cross-platform compatibility (desktop and web)
- System information display (CPU/GPU)

## Prerequisites

- Rust (https://www.rust-lang.org/tools/install)
- wasm-pack (installed automatically by the build script)
- A modern web browser with WebGL support

## Building for Web

To build the WebAssembly module for web deployment:

```bash
./build_wasm.sh
```

This script will:
1. Check if wasm-pack is installed and install it if needed
2. Build the WebAssembly module with the web target
3. Output the compiled files to the `pkg` directory

## Running the Web Application

After building, you can serve the application using the provided script:

```bash
./serve.sh
```

This will start a local HTTP server. Open your browser and navigate to http://localhost:8000.

Alternatively, you can use any of these methods:

```bash
# Using Python's built-in HTTP server
python -m http.server

# Or using Node.js serve package
npx serve
```

## Building for Desktop

To build and run the native desktop application:

```bash
cargo run --release
```

## Project Structure

- `src/` - Main source code
  - `src/web.rs` - WebAssembly bindings and web-specific code
  - `src/state.rs` - Rendering state management
  - `src/system_info.rs` - System information gathering
  - `src/debug.rs` - Debug overlay and performance monitoring
- `index.html` - HTML template for the web application
- `build_wasm.sh` - Build script for WebAssembly
- `serve.sh` - Script to start a local HTTP server
- `pkg/` - Output directory for the WebAssembly build (created by build script)

## How It Works

The application uses wgpu with the WebGL backend to render graphics in the browser. The Rust code is compiled to WebAssembly and exposed to JavaScript through wasm-bindgen. The web page creates a canvas element and passes it to the WebAssembly module, which then initializes the renderer and starts the render loop.

For the desktop version, the application uses the native wgpu backend appropriate for the platform (Vulkan, Metal, DirectX, etc.).

## WebAssembly Integration

The WebAssembly integration is implemented through conditional compilation:

```rust
#[cfg(target_arch = "wasm32")]
// Web-specific code

#[cfg(not(target_arch = "wasm32"))]
// Desktop-specific code
```

This allows the codebase to be shared between the web and desktop versions while handling platform-specific differences.

## Customization

You can customize the appearance and behavior of the web application by:

- Modifying the CSS styles in `index.html`
- Adjusting the canvas size in `index.html`
- Changing the rendering parameters in `src/web.rs`

## Troubleshooting

- If you see a blank canvas, check the browser console for errors
- Make sure your browser supports WebGL 2
- If you get a "WebAssembly module not found" error, make sure you've built the module with `./build_wasm.sh`
- For performance issues, try reducing the canvas size or simplifying the rendering 