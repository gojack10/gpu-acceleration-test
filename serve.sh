#!/bin/bash

echo "Starting HTTP server on port 8000..."
echo "Open http://localhost:8000 in your browser"
echo "Press Ctrl+C to stop the server"

# Check if python3 is available
if command -v python3 &> /dev/null; then
    python3 -m http.server
# Check if python is available
elif command -v python &> /dev/null; then
    python -m http.server
# Check if npx is available
elif command -v npx &> /dev/null; then
    npx serve
else
    echo "Error: Neither python, python3, nor npx is available."
    echo "Please install one of them to run a local server."
    exit 1
fi 