#!/bin/bash
# Dataset Creation Script with Custom Weights
# Fixed the 'last_names' error

echo "🎯 Creating Custom Weighted Dataset (Fixed)"
echo "============================================"
echo "📊 Training: 100,000 examples"
echo "📊 Development: 20,000 examples"
echo "📊 Countries: All (Chile, Mexico, Brazil, Uruguay)"
echo "📊 Custom Weights: ADDRESS & ID_NUMBER boosted"
echo ""

cd /Users/andresverafigueroa/Documents/GitHub/DATA_GENERATION/Spacy

echo "🚀 Starting dataset creation..."
python3 data_generation_noisy.py \
  --mode create-dataset \
  --country all \
  --train-size 100000 \
  --dev-size 20000 \
  --custom-weights '{"personal_id":30,"address_focused":30,"contact_only":20,"full":15,"financial_heavy":5}' \
  --noise \
  --noise-level 0.3

echo ""
echo "✅ Dataset creation completed!"
echo "📁 Check the 'output' directory for your files."
