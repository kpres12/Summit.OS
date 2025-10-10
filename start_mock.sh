#!/bin/bash
# Summit.OS Mock Server Startup Script

echo "🚀 Starting Summit.OS Mock Server..."
echo "=================================="

# Set environment variables
export HTTP_PORT=8000
export SUMMIT_API_KEY=dev_key_placeholder
export MQTT_WS_URL=ws://localhost:1883

echo "Configuration:"
echo "  HTTP_PORT: $HTTP_PORT"
echo "  SUMMIT_API_KEY: $SUMMIT_API_KEY"
echo "  MQTT_WS_URL: $MQTT_WS_URL"
echo ""

# Install requirements if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing requirements..."
pip install -r requirements_mock.txt

echo ""
echo "Starting Summit.OS Mock Server..."
echo "=================================="
echo "API: http://localhost:$HTTP_PORT/api/v1"
echo "Docs: http://localhost:$HTTP_PORT/docs"
echo "MQTT: ws://localhost:1883"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server
python summit_mock.py
