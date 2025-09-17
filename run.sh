#!/bin/bash

# Life Expectancy Dashboard Startup Script

echo "🌍 Starting Life Expectancy Dashboard..."
echo "📦 Installing dependencies..."

# Install dependencies if not already installed
pip install -r requirements.txt

echo "🚀 Starting Streamlit application..."
echo "📊 Dashboard will be available at: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the server"

# Start the Streamlit app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0