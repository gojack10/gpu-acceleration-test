# GPU Acceleration Test - WebGL Integration

This project demonstrates a Minecraft block renderer using WebGL through WebAssembly. The application can run both as a native desktop application and as a web application in the browser.

## Features

- 3D rendering of Minecraft-style blocks
- WebGL integration for browser support
- Responsive canvas that adapts to the browser window
- Simple UI controls for starting and stopping rendering

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

After building, you can serve the application using a simple HTTP server:

```bash
# Using Python's built-in HTTP server
python -m http.server

# Or using Node.js serve package
npx serve
```

Then open your browser and navigate to http://localhost:8000 (or the port shown in the terminal).

## Usage

1. Open the web page in your browser
2. Wait for the WebAssembly module to load
3. Click the "Start Rendering" button to begin rendering
4. Click the "Stop Rendering" button to stop the render loop

## Building for Desktop

To build and run the native desktop application:

```bash
cargo run --release
```

## Project Structure

- `src/web.rs` - WebAssembly bindings and web-specific code
- `index.html` - HTML template for the web application
- `build_wasm.sh` - Build script for WebAssembly
- `pkg/` - Output directory for the WebAssembly build (created by build script)

## How It Works

The application uses wgpu with the WebGL backend to render 3D graphics in the browser. The Rust code is compiled to WebAssembly and exposed to JavaScript through wasm-bindgen. The web page creates a canvas element and passes it to the WebAssembly module, which then initializes the renderer and starts the render loop.

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