#!/usr/bin/env python3
"""
Quick Test Script for Transformer NER Pipeline
==============================================

This script performs a quick test of the transformer NER pipeline
with a small dataset to verify everything works correctly.

Usage:
    python quick_test.py
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        else:
            print(f"✅ {description} completed")
            if result.stdout.strip():
                print(f"📄 Output: {result.stdout.strip()}")
            return True
    except Exception as e:
        print(f"❌ Exception during {description}: {e}")
        return False

def main():
    print("🧪 Transformer NER Quick Test")
    print("=============================")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if we're in the right directory
    if not os.path.exists("transformer_data_generator.py"):
        print("❌ Error: transformer_data_generator.py not found")
        print("💡 Please run this script from the Transformers directory")
        return False
    
    # Test parameters
    train_size = 1000
    dev_size = 200
    
    print(f"\n📊 Test Configuration:")
    print(f"   - Training examples: {train_size:,}")
    print(f"   - Dev examples: {dev_size:,}")
    print(f"   - Countries: Chile, Mexico, Brazil")
    print(f"   - Noise level: 0.2 (reduced for testing)")
    
    # Step 1: Generate small test dataset
    print(f"\n1️⃣ Generating test dataset...")
    data_command = f"python transformer_data_generator.py --train-size {train_size} --dev-size {dev_size} --countries chile mexico brazil --noise-level 0.2"
    
    if not run_command(data_command, "Dataset generation"):
        return False
    
    # Check if files were created
    train_file = f"output/train_transformer_{train_size}.json"
    dev_file = f"output/dev_transformer_{dev_size}.json"
    
    if not os.path.exists(train_file):
        print(f"❌ Training file not created: {train_file}")
        return False
    
    if not os.path.exists(dev_file):
        print(f"❌ Dev file not created: {dev_file}")
        return False
    
    print(f"✅ Dataset files created successfully")
    
    # Step 2: Quick data validation
    print(f"\n2️⃣ Validating dataset...")
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(dev_file, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
        
        print(f"📊 Training examples loaded: {len(train_data):,}")
        print(f"📊 Dev examples loaded: {len(dev_data):,}")
        
        # Check first example structure
        if len(train_data) > 0:
            example = train_data[0]
            required_keys = ["id", "text", "entities", "country"]
            missing_keys = [key for key in required_keys if key not in example]
            
            if missing_keys:
                print(f"❌ Missing keys in example: {missing_keys}")
                return False
            
            print(f"✅ Dataset structure validation passed")
            print(f"📝 Sample text: {example['text'][:100]}...")
            print(f"🏷️  Sample entities: {len(example['entities'])} entities")
        
    except Exception as e:
        print(f"❌ Dataset validation error: {e}")
        return False
    
    # Step 3: Test training (1 epoch only)
    print(f"\n3️⃣ Testing model training (1 epoch)...")
    train_command = f"python train_transformer_ner.py --train-file {train_file} --dev-file {dev_file} --epochs 1 --batch-size 8"
    
    if not run_command(train_command, "Model training test"):
        print("⚠️  Training test failed - this might be due to missing dependencies")
        print("💡 Install requirements: pip install -r requirements.txt")
        return False
    
    # Step 4: Check model output
    print(f"\n4️⃣ Checking training output...")
    models_dir = "models"
    if os.path.exists(models_dir):
        model_dirs = [d for d in os.listdir(models_dir) if d.startswith("transformer_ner_")]
        if model_dirs:
            latest_model = sorted(model_dirs)[-1]
            model_path = os.path.join(models_dir, latest_model)
            print(f"✅ Model created: {model_path}")
            
            # Check model files
            final_model_path = os.path.join(model_path, "final_model")
            if os.path.exists(final_model_path):
                model_files = os.listdir(final_model_path)
                print(f"📁 Model files: {', '.join(model_files)}")
                
                # Check if key files exist
                required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
                present_files = [f for f in required_files if f in model_files]
                print(f"✅ Required files present: {len(present_files)}/{len(required_files)}")
        else:
            print("⚠️  No model directories found")
    else:
        print("⚠️  Models directory not found")
    
    print(f"\n🎉 Quick test completed!")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n💡 Next steps:")
    print(f"   1. Generate full dataset: python transformer_data_generator.py --train-size 50000 --dev-size 10000")
    print(f"   2. Train production model: python train_transformer_ner.py --epochs 5")
    print(f"   3. Monitor training progress in the models/ directory")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
