#!/bin/bash
# Setup Ollama Models for Basketball LLM Ensemble
# Downloads and configures 5 models with ~20GB total size

set -e

echo "🤖 Basketball LLM Ensemble - Model Setup"
echo "========================================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not installed!"
    echo ""
    echo "Install Ollama first:"
    echo "  Linux:   curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Mac:     brew install ollama"
    echo "  Windows: Download from https://ollama.com/download"
    echo ""
    exit 1
fi

echo "✅ Ollama is installed"
echo ""

# Check if Ollama service is running
if ! ollama list &> /dev/null; then
    echo "❌ Ollama service not running!"
    echo ""
    echo "Start Ollama service:"
    echo "  Linux/Mac: ollama serve &"
    echo "  Or run in separate terminal: ollama serve"
    echo ""
    exit 1
fi

echo "✅ Ollama service is running"
echo ""

# Function to pull model with progress
pull_model() {
    local model=$1
    local name=$2
    local weight=$3
    local size=$4

    echo "📥 Downloading: $name ($size)"
    echo "   Model: $model"
    echo "   Weight: $weight"

    if ollama list | grep -q "$model"; then
        echo "   ✅ Already downloaded"
    else
        ollama pull $model
        echo "   ✅ Downloaded successfully"
    fi
    echo ""
}

echo "Total download size: ~20GB"
echo "This will take 10-30 minutes depending on your internet speed"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled"
    exit 0
fi

echo ""
echo "Starting downloads..."
echo ""

# Download all 5 models
pull_model "mistral:7b-instruct" "Mistral-7B Instruct" "4.2" "4.1GB"
pull_model "openchat:7b" "OpenChat-7B" "4.1" "4.1GB"
pull_model "dolphin-mistral:7b" "Dolphin-Mistral-7B" "4.0" "4.1GB"
pull_model "codellama:7b-instruct" "CodeLlama-7B Instruct" "4.0" "4.1GB"
pull_model "neural-chat:7b" "Neural-Chat-7B" "3.9" "3.9GB"

echo "========================================="
echo "✅ All models downloaded!"
echo "========================================="
echo ""
echo "📋 Installed Models:"
ollama list
echo ""
echo "🧪 Test the ensemble:"
echo "   python basketball_llm_ensemble.py"
echo ""
echo "🚀 Use in predictions:"
echo "   python basketball_main.py --llm-predict"
echo ""
