#!/bin/zsh

if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Error: Ollama is not running. Please start Ollama first."
    exit 1
fi

echo "🚀 Launching LegalRAG Agent System..."

echo "Starting Backend Server on port 8000..."
python3 server.py & 
BACKEND_PID=$!

echo "Installing frontend dependencies and starting..."
cd frontend && npm install && npm run dev &
FRONTEND_PID=$!

echo "--------------------------------------------------"
echo "✅ System is launching!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Frontend should be available at http://localhost:5173"
echo "--------------------------------------------------"

wait
