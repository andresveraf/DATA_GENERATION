# Transformer-Based NER Training for Latin American PII Data

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)

A complete pipeline for training BERT-based multilingual Named Entity Recognition (NER) models specifically designed for Latin American Personally Identifiable Information (PII) data in Spanish and Portuguese.

## 🎯 Overview

This project provides a production-ready alternative to spaCy-based NER training, leveraging transformer models (BERT multilingual) to achieve superior accuracy in multilingual PII entity recognition across Chile, Mexico, Brazil, and Uruguay.

### Key Features

- **🌍 Multilingual Support**: Native Spanish and Portuguese processing
- **🏛️ Multi-Country Data**: Realistic data patterns for 4 Latin American countries
- **🏷️ 7 Entity Types**: Comprehensive PII coverage
- **🎭 OCR Noise Simulation**: Robust training with realistic character corruptions
- **⚡ Production Ready**: Complete pipeline with evaluation, checkpointing, and deployment
- **🔧 Easy to Use**: One-command workflows and comprehensive documentation

## 📊 Entity Types Supported

| Entity Type | Description | Example |
|-------------|-------------|---------|
| `CUSTOMER_NAME` | Personal names | JOSÉ GONZÁLEZ, JOÃO SILVA |
| `ID_NUMBER` | National ID numbers | RUT: 12.345.678-9, CPF: 123.456.789-01 |
| `ADDRESS` | Physical addresses | Av. Providencia 123, Av. Paulista 456 |
| `PHONE_NUMBER` | Phone numbers | +56 912345678, +55 11 987654321 |
| `EMAIL` | Email addresses | jose.gonzalez@empresa.cl |
| `AMOUNT` | Monetary amounts | $50,000, R$ 25.000,00 |
| `SEQ_NUMBER` | Reference numbers | CL12345, BR78901, A123456 |

## 🌎 Supported Countries

- **🇨🇱 Chile**: RUT format, Chilean addresses, +56 phone codes
- **🇲🇽 Mexico**: CURP format, Mexican addresses, +52 phone codes  
- **🇧🇷 Brazil**: CPF format, Brazilian addresses, +55 phone codes
- **🇺🇾 Uruguay**: CI format, Uruguayan addresses, +598 phone codes

## 📁 Project Structure

```
Transformers/
├── 📄 README.md                    # This documentation
├── 🐍 transformer_data_generator.py # Dataset generation for transformers
├── 🤖 train_transformer_ner.py     # BERT model training script
├── 🧪 quick_test.py                # Quick validation script
├── ⚙️ workflow.sh                  # Convenient workflow commands
├── 📦 requirements.txt             # Python dependencies
├── 💻 inference_example.py         # Model inference examples
├── 📝 transformer_notes.txt        # Detailed technical documentation
├── 📂 output/                      # Generated datasets
│   ├── train_transformer_*.json    # Training datasets
│   ├── dev_transformer_*.json      # Development datasets
│   └── transformer_dataset_stats_*.json # Dataset statistics
└── 📂 models/                      # Trained models
    └── transformer_ner_*/          # Model directories (timestamped)
        ├── final_model/             # Best model checkpoint
        ├── training_config.json     # Training configuration
        └── evaluation_results.json  # Final evaluation metrics
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the project
cd /path/to/DATA_GENERATION/Transformers

# Install dependencies
pip install -r requirements.txt

# Create output directories (automatic, but good to verify)
mkdir -p output models
```

### 2. Quick Test (Recommended First Step)

```bash
# Run complete validation pipeline (1K examples, 1 epoch)
python quick_test.py
```

This will:
- Generate a small test dataset (1,000 train + 200 dev examples)
- Validate data format and entity detection
- Train a quick test model (1 epoch)
- Verify model creation and output files

### 3. Production Training

```bash
# Generate production dataset (50K train, 10K dev)
python transformer_data_generator.py \
    --train-size 50000 \
    --dev-size 10000 \
    --countries chile mexico brazil uruguay \
    --noise-level 0.3

# Train production model (5 epochs)
python train_transformer_ner.py \
    --train-file output/train_transformer_50000.json \
    --dev-file output/dev_transformer_10000.json \
    --epochs 5 \
    --batch-size 16
```

### 4. Use Trained Model

```bash
# Interactive inference
python inference_example.py --model-path models/transformer_ner_*/final_model

# Demo predictions
python inference_example.py --model-path models/transformer_ner_*/final_model --demo

# Single text prediction
python inference_example.py \
    --model-path models/transformer_ner_*/final_model \
    --text "Cliente: JOSÉ GONZÁLEZ - RUT 12.345.678-9 - Tel: +56 912345678"
```

## 🔧 Workflow Commands

For convenience, use the pre-defined workflow commands:

```bash
# Source workflow functions (run once per session)
source workflow.sh

# Quick validation
quick_test

# Complete production pipeline
production_workflow

# High-accuracy training
generate_large_data && train_large_model

# Environment setup
setup_environment

# Clean all outputs
clean_outputs
```

### Available Workflow Functions

| Function | Description |
|----------|-------------|
| `quick_test` | Complete quick test (1K examples, 1 epoch) |
| `production_workflow` | Full production pipeline (50K examples, 5 epochs) |
| `generate_test_data` | Generate small test dataset |
| `generate_production_data` | Generate production dataset (50K examples) |
| `generate_large_data` | Generate large dataset (200K examples) |
| `train_test_model` | Quick test training (1 epoch) |
| `train_production_model` | Production training (5 epochs) |
| `train_large_model` | High-accuracy training (8 epochs) |
| `setup_environment` | Install dependencies and create directories |
| `clean_outputs` | Remove all generated files |

## 📊 Dataset Generation

### Basic Usage

```bash
python transformer_data_generator.py [OPTIONS]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--countries` | `chile mexico brazil uruguay` | Countries to include |
| `--train-size` | `50000` | Number of training examples |
| `--dev-size` | `10000` | Number of development examples |
| `--noise-level` | `0.3` | OCR noise level (0.0-1.0) |
| `--output-dir` | `output` | Output directory |

### Example Commands

```bash
# Quick test dataset
python transformer_data_generator.py \
    --train-size 1000 \
    --dev-size 200 \
    --noise-level 0.2

# Production dataset
python transformer_data_generator.py \
    --train-size 50000 \
    --dev-size 10000 \
    --countries chile mexico brazil uruguay

# Large dataset for maximum accuracy
python transformer_data_generator.py \
    --train-size 200000 \
    --dev-size 40000 \
    --noise-level 0.25
```

### Dataset Sizes Recommendations

| Use Case | Train Size | Dev Size | Training Time | Accuracy |
|----------|------------|----------|---------------|----------|
| **Quick Test** | 1,000 | 200 | 2-5 min | Basic validation |
| **Development** | 10,000 | 2,000 | 30-60 min | Good for prototyping |
| **Production** | 50,000 | 10,000 | 2-4 hours | Balanced quality/time |
| **High Accuracy** | 200,000 | 40,000 | 8-16 hours | Maximum quality |

## 🤖 Model Training

### Basic Usage

```bash
python train_transformer_ner.py [OPTIONS]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-name` | `bert-base-multilingual-cased` | Pre-trained model |
| `--train-file` | `output/train_transformer_50000.json` | Training dataset |
| `--dev-file` | `output/dev_transformer_10000.json` | Development dataset |
| `--epochs` | `5` | Number of training epochs |
| `--batch-size` | `16` | Training batch size |
| `--learning-rate` | `2e-5` | Learning rate |
| `--max-length` | `128` | Maximum sequence length |
| `--output-dir` | `models` | Model output directory |

### Training Configurations

#### Quick Test
```bash
python train_transformer_ner.py \
    --epochs 1 \
    --batch-size 8 \
    --learning-rate 2e-5
```

#### Production (Recommended)
```bash
python train_transformer_ner.py \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 2e-5
```

#### High Accuracy
```bash
python train_transformer_ner.py \
    --epochs 8 \
    --batch-size 16 \
    --learning-rate 1e-5 \
    --train-file output/train_transformer_200000.json \
    --dev-file output/dev_transformer_40000.json
```

## 📈 Performance Expectations

### Entity Success Rates

| Entity Type | Expected Success Rate | Accuracy Level |
|-------------|---------------------|----------------|
| **CUSTOMER_NAME** | >95% | Excellent |
| **EMAIL** | >95% | Excellent |
| **PHONE_NUMBER** | >95% | Excellent |
| **ADDRESS** | >90% | Very Good |
| **AMOUNT** | >85% | Good |
| **ID_NUMBER** | >80% | Good (complex formats) |
| **SEQ_NUMBER** | >75% | Fair (high variability) |

### Overall Performance

| Metric | Quick Test | Production | High Accuracy |
|--------|------------|------------|---------------|
| **Overall F1 Score** | 85-90% | 92-95% | 95-97% |
| **Training Time** | 2-5 min | 2-4 hours | 8-16 hours |
| **Model Size** | ~500MB | ~500MB | ~500MB |
| **Inference Speed** | Fast | Fast | Fast |

### Comparison with spaCy Version

| Aspect | spaCy (Optimized) | Transformer | Winner |
|--------|------------------|-------------|---------|
| **F1 Score** | 88-92% | 92-96% | 🏆 Transformer |
| **Multilingual** | Good | Excellent | 🏆 Transformer |
| **Inference Speed** | Very Fast | Fast | 🏆 spaCy |
| **Model Size** | 50MB | 500MB | 🏆 spaCy |
| **Training Time** | Fast | Moderate | 🏆 spaCy |
| **Memory Usage** | Low | Moderate | 🏆 spaCy |
| **Deployment** | Simple | Standard | 🏆 spaCy |
| **Accuracy** | Good | Excellent | 🏆 Transformer |

## 🔍 Model Inference

### Using the Inference Script

```bash
# Interactive mode
python inference_example.py --model-path models/transformer_ner_*/final_model

# Demo with sample texts
python inference_example.py --model-path models/transformer_ner_*/final_model --demo

# Analyze specific text
python inference_example.py \
    --model-path models/transformer_ner_*/final_model \
    --text "Cliente: MARÍA SILVA - CPF 123.456.789-01 - Email: maria@empresa.com.br"
```

### Python Integration

```python
from inference_example import TransformerNERInference

# Initialize model
model = TransformerNERInference("models/transformer_ner_*/final_model")

# Predict entities
text = "Cliente: JOSÉ GONZÁLEZ - RUT 12.345.678-9 - Tel: +56 912345678"
entities = model.predict(text)

for entity in entities:
    print(f"{entity['label']}: {entity['text']} (confidence: {entity['confidence']})")
```

### Expected Output

```
CUSTOMER_NAME: JOSÉ GONZÁLEZ (confidence: 0.9956)
ID_NUMBER: 12.345.678-9 (confidence: 0.9834)
PHONE_NUMBER: +56 912345678 (confidence: 0.9912)
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. ImportError: No module named 'transformers'
```bash
# Solution: Install requirements
pip install -r requirements.txt
```

#### 2. CUDA out of memory
```bash
# Solution: Reduce batch size
python train_transformer_ner.py --batch-size 8

# Or disable FP16
python train_transformer_ner.py --fp16 False
```

#### 3. Dataset files not found
```bash
# Solution: Generate dataset first
python transformer_data_generator.py --train-size 1000 --dev-size 200
```

#### 4. Low entity success rates
```bash
# Solutions:
# - Reduce noise level
python transformer_data_generator.py --noise-level 0.2

# - Increase dataset size
python transformer_data_generator.py --train-size 100000

# - Check generated examples
head -5 output/train_transformer_*.json
```

#### 5. Training is too slow
```bash
# Solutions:
# - Use GPU if available (automatic detection)
# - Reduce batch size if memory constrained
python train_transformer_ner.py --batch-size 8

# - Use fewer epochs for testing
python train_transformer_ner.py --epochs 3
```

### Performance Optimization

#### For Faster Training
- ✅ Use GPU if available (automatic)
- ✅ Enable FP16 precision (enabled by default)
- ✅ Increase batch size if memory allows
- ✅ Use gradient accumulation for large batches
- ✅ Start with fewer epochs for initial testing

#### For Better Accuracy
- ✅ Increase dataset size (200K+ examples)
- ✅ Use more training epochs (8-10)
- ✅ Lower learning rate (1e-5)
- ✅ Reduce noise level (0.2-0.25)
- ✅ Use larger pre-trained models (BERT-large)

## 🔄 Migration from spaCy

If you're migrating from the existing spaCy-based solution:

### Key Differences

| Aspect | spaCy Version | Transformer Version |
|--------|---------------|-------------------|
| **Data Format** | `.spacy` binary | `.json` with entities |
| **Model Type** | spaCy CNN | BERT multilingual |
| **Labeling** | spaCy format | BIO tagging scheme |
| **Training** | spaCy CLI | Hugging Face Trainer |
| **Inference** | spaCy NLP | Transformers pipeline |

### Migration Steps

1. **Use Same Entity Types**: Both versions use identical entity labels
2. **Generate New Data**: Use `transformer_data_generator.py` instead of `data_generation_noisy.py`
3. **Train New Model**: Use `train_transformer_ner.py` instead of spaCy CLI
4. **Update Inference**: Use `inference_example.py` or transformers pipeline

### Side-by-Side Comparison

```bash
# spaCy approach
python data_generation_noisy.py --examples 50000
python -m spacy train config.cfg --output models

# Transformer approach  
python transformer_data_generator.py --train-size 50000 --dev-size 10000
python train_transformer_ner.py --epochs 5
```

## 📚 Advanced Usage

### Custom Model Configuration

```python
# Use different pre-trained models
python train_transformer_ner.py --model-name "distilbert-base-multilingual-cased"
python train_transformer_ner.py --model-name "xlm-roberta-base"
```

### Hyperparameter Tuning

```bash
# Experiment with learning rates
python train_transformer_ner.py --learning-rate 1e-5  # Lower for stability
python train_transformer_ner.py --learning-rate 3e-5  # Higher for faster learning

# Adjust sequence length
python train_transformer_ner.py --max-length 256  # Longer sequences
python train_transformer_ner.py --max-length 64   # Shorter for speed
```

### Custom Dataset Configuration

```bash
# Focus on specific countries
python transformer_data_generator.py --countries chile brazil

# Adjust noise for your use case
python transformer_data_generator.py --noise-level 0.1  # Less noise
python transformer_data_generator.py --noise-level 0.5  # More noise
```

## 📊 Monitoring and Evaluation

### Training Logs

Monitor training progress in the model output directory:

```
models/transformer_ner_YYYYMMDD_HHMMSS/
├── checkpoint-*/          # Training checkpoints
├── final_model/           # Best model
├── training_config.json   # Configuration used
└── evaluation_results.json # Final metrics
```

### Key Metrics to Watch

- **eval_f1**: Overall F1 score (target: >0.92)
- **eval_loss**: Validation loss (should decrease)
- **f1_B-CUSTOMER_NAME**: Entity-specific F1 scores
- **train_loss**: Training loss (should decrease smoothly)

### Model Comparison

```bash
# Compare multiple models
ls -la models/
cat models/transformer_ner_*/evaluation_results.json | grep '"eval_f1"'
```

## 🎯 Production Deployment

### Model Packaging

```python
# Save model for production
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("models/transformer_ner_*/final_model")
model = AutoModelForTokenClassification.from_pretrained("models/transformer_ner_*/final_model")

# Save to specific location
tokenizer.save_pretrained("production_model/")
model.save_pretrained("production_model/")
```

### API Integration

```python
from transformers import pipeline

# Create production pipeline
ner_pipeline = pipeline(
    "ner",
    model="production_model/",
    tokenizer="production_model/",
    aggregation_strategy="simple"
)

# Use in your application
def extract_entities(text):
    return ner_pipeline(text)
```

## 🤝 Contributing

### Development Setup

```bash
# Clone the repository
git clone https://github.com/andresveraf/DATA_GENERATION.git
cd DATA_GENERATION/Transformers

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python quick_test.py
pytest tests/ (if test directory exists)
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to functions
- Run `black` for code formatting

## 📝 Changelog

### Version 1.0.0 (August 2025)
- ✅ Initial release with complete transformer pipeline
- ✅ BERT multilingual base model support
- ✅ 7 entity types for Latin American PII
- ✅ Multi-country support (Chile, Mexico, Brazil, Uruguay)
- ✅ OCR noise simulation
- ✅ Production-ready training and inference
- ✅ Comprehensive documentation and workflows

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Andrés Vera Figueroa**
- Purpose: Transformer-based NER model training for multilingual PII data
- Date: August 2025
- Context: Production-ready alternative to spaCy-based approach

## 🙏 Acknowledgments

- Hugging Face for the transformers library
- Google for BERT multilingual models
- spaCy team for inspiration and baseline comparison
- Latin American data formatting standards

---

## 🎉 Ready to Get Started?

1. **Quick Start**: Run `python quick_test.py`
2. **Production**: Run `source workflow.sh && production_workflow`
3. **Questions**: Check the troubleshooting section or review `transformer_notes.txt`

**Happy training! 🚀**
