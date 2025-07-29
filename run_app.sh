#!/bin/bash

# Agentic Chat App Startup Script

echo "🚀 Starting Agentic Chat App with LangGraph & AWS Bedrock"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your AWS credentials"
    echo "   You can run the app in fallback mode without Bedrock configuration"
fi

echo "🌐 Starting Streamlit server..."
echo "📱 App will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Run the Streamlit app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0