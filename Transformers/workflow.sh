#!/bin/bash
# Transformer NER Training Workflow Scripts
# ==========================================
# Collection of utility scripts for easy execution

# Generate small test dataset (for quick validation)
generate_test_data() {
    echo "🧪 Generating test dataset..."
    python transformer_data_generator.py \
        --train-size 1000 \
        --dev-size 200 \
        --countries chile mexico brazil \
        --noise-level 0.2 \
        --output-dir output
}

# Generate production dataset (recommended size)
generate_production_data() {
    echo "🏭 Generating production dataset..."
    python transformer_data_generator.py \
        --train-size 50000 \
        --dev-size 10000 \
        --countries chile mexico brazil uruguay \
        --noise-level 0.3 \
        --output-dir output
}

# Generate large dataset (for maximum accuracy)
generate_large_data() {
    echo "🚀 Generating large dataset..."
    python transformer_data_generator.py \
        --train-size 200000 \
        --dev-size 40000 \
        --countries chile mexico brazil uruguay \
        --noise-level 0.3 \
        --output-dir output
}

# Train test model (1 epoch, fast)
train_test_model() {
    echo "🧪 Training test model..."
    python train_transformer_ner.py \
        --train-file output/train_transformer_1000.json \
        --dev-file output/dev_transformer_200.json \
        --epochs 1 \
        --batch-size 8 \
        --learning-rate 2e-5 \
        --output-dir models
}

# Train production model (recommended settings)
train_production_model() {
    echo "🏭 Training production model..."
    python train_transformer_ner.py \
        --train-file output/train_transformer_50000.json \
        --dev-file output/dev_transformer_10000.json \
        --epochs 5 \
        --batch-size 16 \
        --learning-rate 2e-5 \
        --output-dir models
}

# Train large model (maximum accuracy)
train_large_model() {
    echo "🚀 Training large model..."
    python train_transformer_ner.py \
        --train-file output/train_transformer_200000.json \
        --dev-file output/dev_transformer_40000.json \
        --epochs 8 \
        --batch-size 16 \
        --learning-rate 1e-5 \
        --output-dir models
}

# Complete quick test workflow
quick_test() {
    echo "🧪 Running complete quick test..."
    python quick_test.py
}

# Complete production workflow
production_workflow() {
    echo "🏭 Running complete production workflow..."
    echo "Step 1: Generating production dataset..."
    generate_production_data
    
    if [ $? -eq 0 ]; then
        echo "Step 2: Training production model..."
        train_production_model
    else
        echo "❌ Dataset generation failed"
        return 1
    fi
}

# Setup environment
setup_environment() {
    echo "⚙️ Setting up environment..."
    
    # Create directories
    mkdir -p output
    mkdir -p models
    
    # Install requirements
    echo "📦 Installing Python requirements..."
    pip install -r requirements.txt
    
    echo "✅ Environment setup complete"
}

# Clean outputs
clean_outputs() {
    echo "🧹 Cleaning output files..."
    read -p "Are you sure you want to delete all output files? (y/N): " confirm
    if [[ $confirm == [yY] ]]; then
        rm -rf output/*
        rm -rf models/*
        echo "✅ Output files cleaned"
    else
        echo "ℹ️ Clean operation cancelled"
    fi
}

# Show help
show_help() {
    echo "🤖 Transformer NER Training Workflow"
    echo "===================================="
    echo ""
    echo "Usage: source workflow.sh"
    echo "Then call any of these functions:"
    echo ""
    echo "📦 Setup:"
    echo "  setup_environment         - Install dependencies and create directories"
    echo ""
    echo "🧪 Quick Testing:"
    echo "  quick_test                - Run complete quick test (1K examples, 1 epoch)"
    echo "  generate_test_data        - Generate small test dataset"
    echo "  train_test_model          - Train quick test model"
    echo ""
    echo "🏭 Production:"
    echo "  production_workflow       - Complete production pipeline (50K examples, 5 epochs)"
    echo "  generate_production_data  - Generate production dataset"
    echo "  train_production_model    - Train production model"
    echo ""
    echo "🚀 Large Scale:"
    echo "  generate_large_data       - Generate large dataset (200K examples)"
    echo "  train_large_model         - Train large model (8 epochs)"
    echo ""
    echo "🧹 Utilities:"
    echo "  clean_outputs             - Remove all generated files"
    echo "  show_help                 - Show this help message"
    echo ""
    echo "💡 Examples:"
    echo "  setup_environment && quick_test"
    echo "  production_workflow"
    echo "  generate_large_data && train_large_model"
}

# Show help by default
show_help
