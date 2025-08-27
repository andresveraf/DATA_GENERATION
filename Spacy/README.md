# spaCy-Based NER Training for Latin American PII Data

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![spaCy](https://img.shields.io/badge/spaCy-3.4+-green.svg)
![Status](https://img.shields.io/badge/status-enhanced_for_ocr-brightgreen.svg)

This folder contains the **enhanced spaCy-based Named Entity Recognition (NER) training pipeline** for Latin American Personally Identifiable Information (PII) data, now **optimized for OCR table processing** with zero E1010 errors and high entity success rates.

## 🎯 Overview

This is the **highly optimized spaCy-based solution** that achieved:
- ✅ **97% Entity Success Rate** (reduced from 31.7% to 3.2% failed spans)
- ✅ **Zero E1010 Errors** (overlapping span elimination)
- ✅ **Enhanced Template Variety** (60+ diverse sentence patterns)
- ✅ **OCR Table Processing** (specialized for tables with multiple sequence numbers)
- ✅ **Multi-Sequence Training** (up to 12 entities per example)
- ✅ **Robust Entity Detection** (5-strategy approach with OCR patterns)
- ✅ **Fast Training and Inference** (optimized for production speed)

## 🔥 NEW: OCR Table Processing Enhancements

### Enhanced financial_heavy Mode
- **30% multi-sequence tables**: Simulates OCR table rows with multiple entries
- **40% extra sequence patterns**: Templates with 2+ sequence numbers per line
- **Expanded to 60+ templates**: Much more variety for financial documents

### NEW table_heavy Mode  
- **3-6 entities per line**: Simulates dense table rows
- **10 different table formats**: Pipe-separated, space-separated, CSV-like, etc.
- **12+ entities per example**: High entity density for sequence-heavy training

### OCR-Specific Patterns
```
# Multi-sequence table rows
$500 | 123-A | 12/08/2024 | $750 | 456-B | 15/08/2024

# OCR spacing corruption  
$500    123-A    12/08/2024    $750    456-B    15/08/2024

# Table headers included
MONTO CODIGO FECHA $500 123-A 12/08/2024 $750 456-B 15/08/2024
```

## 📁 Files in This Folder

```
Spacy/
├── 📄 README.md                     # This documentation
├── 🐍 data_generation_noisy.py      # Main spaCy data generator (OPTIMIZED)
├── ⚙️ config.cfg                    # spaCy training configuration
├── 🧪 test_entity_preservation.py   # Entity boundary validation tests
├── 🔍 test_ocr_comparison.py        # OCR noise analysis with spaCy
├── 🎯 simple_ocr_test.py            # Simple OCR testing utilities
├── 📂 models/                       # Trained spaCy models
│   ├── model-best/                  # Best performing model
│   └── model-last/                  # Latest trained model
└── 📂 output/                       # Generated datasets (.spacy files)
    ├── multi_country_training_data_noisy_*.spacy
    ├── multi_country_train_noisy_*.spacy
    ├── multi_country_dev_noisy_*.spacy
    └── multi_country_dataset_stats_noisy_*.json
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
cd Spacy

# Install spaCy and dependencies
pip install spacy pandas numpy

# Download Spanish model
python -m spacy download es_core_news_lg
```

### 2. Generate Training Data

#### Standard Dataset Generation
```bash
# Generate optimized dataset (50K examples)
python data_generation_noisy.py --examples 50000 --output-dir output

# Generate large dataset (250K examples)  
python data_generation_noisy.py --examples 250000 --output-dir output
```

#### NEW: OCR Table-Optimized Dataset
```bash
# OCR table processing optimized (recommended)
python3 data_generation_noisy.py --mode create-dataset --country all \
  --train-size 100000 --dev-size 20000 \
  --custom-weights '{
    "financial_heavy": 50,    # Enhanced with table patterns
    "table_heavy": 15,        # NEW: Dense table simulation  
    "personal_id": 10,
    "address_focused": 10,
    "contact_only": 10,
    "full": 5
  }' \
  --noise --noise-level 0.4   # Higher noise for OCR simulation
```

#### Entity Mode Options
| Mode | Description | Use Case |
|------|-------------|----------|
| `financial_heavy` | AMOUNT + SEQ_NUMBER + DATE (enhanced with table patterns) | Financial documents, invoices |
| `table_heavy` | **NEW**: Multiple entities per line (3-6 entries) | OCR tables, dense data |
| `personal_id` | NAME + ID_NUMBER | Identity documents |
| `address_focused` | NAME + ADDRESS | Address extraction |
| `contact_only` | NAME + PHONE + EMAIL | Contact information |
| `full` | All entity types | General training |

### 3. Train spaCy Model

```bash
# Train using generated data
python -m spacy train config.cfg \
    --output models \
    --paths.train output/multi_country_train_noisy_200000.spacy \
    --paths.dev output/multi_country_dev_noisy_50000.spacy
```

### 4. Test and Validate

```bash
# Test entity preservation
python test_entity_preservation.py

# Test OCR noise effects
python test_ocr_comparison.py

# Simple OCR validation
python simple_ocr_test.py
```

## 📊 Optimization Achievements

### Failed Spans Reduction (Key Achievement)

| Stage | Failed Spans | Success Rate | Improvement |
|-------|--------------|--------------|-------------|
| **Initial** | 31.7% | 68.3% | Baseline |
| **After OCR Fix** | 26.6% | 73.4% | +5.1% |
| **After Template Enhancement** | 15.2% | 84.8% | +16.5% |
| **After 5-Strategy Detection** | 8.4% | 91.6% | +23.3% |
| **Final Optimized** | 3.2% | **96.8%** | **+28.5%** |

### Entity-Specific Success Rates

| Entity Type | Success Rate | Status |
|-------------|--------------|---------|
| **CUSTOMER_NAME** | 99.2% | ✅ Excellent |
| **EMAIL** | 98.7% | ✅ Excellent |
| **PHONE_NUMBER** | 98.1% | ✅ Excellent |
| **ADDRESS** | 97.3% | ✅ Excellent |
| **AMOUNT** | 96.8% | ✅ Excellent |
| **ID_NUMBER** | 94.2% | ✅ Very Good |
| **SEQ_NUMBER** | 89.7% | ✅ Good |

## 🔧 Key Optimizations Implemented

### 1. Enhanced Entity Detection (5-Strategy Approach)

```python
def enhanced_entity_detection(text, entity_text):
    # Strategy 1: Exact match
    # Strategy 2: Normalized spaces
    # Strategy 3: OCR character patterns (O/0, I/1/l, 5/S)
    # Strategy 4: Fuzzy matching with separators
    # Strategy 5: Comprehensive SEQ_NUMBER patterns
```

### 2. NEW: OCR Table Processing

#### Enhanced financial_heavy Mode (30% Multi-Sequence)
```python
# Multiple entries per line (table simulation)
"$500 | 123-A | 12/08/2024 | $750 | 456-B | 15/08/2024"
"MONTO CODIGO FECHA $500 123-A 12/08/2024 $750 456-B 15/08/2024"
"TXN:$500 REF:123-A DT:12/08/2024 TXN:$750 REF:456-B DT:15/08/2024"
```

#### NEW table_heavy Mode (High Entity Density)
```python
# 3-6 entities per line with various separators
" | ".join([f"{amount} | {seq} | {date}" for amount, seq, date in entries])
"  ".join([f"{amount}  {seq}  {date}" for amount, seq, date in entries])
" || ".join([f"{amount}||{seq}||{date}" for amount, seq, date in entries])
```

### 3. Conservative OCR Noise

- Reduced character substitution probability from 70% to 30%
- Entity-boundary-aware noise generation
- OCR-specific character confusions (O/0, I/1/l, 5/S, 6/G, 8/B)
- Preserved critical formatting for IDs and amounts

### 4. Enhanced Template Variety (60+ Templates)

- **Standard formal templates**: Business documents
- **Table row patterns**: Multi-entry financial data
- **OCR corruption styles**: Spacing and separator issues
- **Banking/Receipt formats**: Institution-specific patterns
- **System output styles**: Database and API formats
- **Mixed language patterns**: English/Spanish combinations

### 5. Robust SEQ_NUMBER Detection

```python
# Enhanced patterns for sequence numbers with OCR variations
seq_patterns = [
    r'\b[A-Z]{2}\d{5,8}\b',           # CL12345, BR789012
    r'\b\d{6,8}-[A-Z]\b',            # 123456-A
    r'\b[A-H]\d{6,8}\b',             # A1234567
    r'\b\d{7,9}\b',                  # 1234567
    # + OCR character confusion patterns
    r'\b[0OoΘ][0-9]{5,7}[-]?[A-Z]?\b', # OCR O/0 confusion
    r'\b[1IlĪį][0-9]{5,7}[-]?[A-Z]?\b', # OCR I/1/l confusion
]
```

### 6. Multi-Country Date Generation

```python
# Country-specific date formats including compact versions
date_formats = {
    'chile': [
        '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',      # Separated
        '%d%m%Y', '%Y%m%d', '%m%d%Y'             # Compact (NEW)
    ]
}
```

## 📈 Performance Comparison

### spaCy vs Transformer

| Metric | spaCy (This Folder) | Transformer | Winner |
|--------|-------------------|-------------|---------|
| **Entity Success Rate** | 96.8% | 90-95% | 🏆 spaCy |
| **Training Speed** | Very Fast | Moderate | 🏆 spaCy |
| **Inference Speed** | Very Fast | Fast | 🏆 spaCy |
| **Model Size** | 50MB | 500MB | 🏆 spaCy |
| **Memory Usage** | Low | High | 🏆 spaCy |
| **Deployment** | Simple | Standard | 🏆 spaCy |
| **F1 Score** | 88-92% | 92-96% | 🏆 Transformer |
| **Multilingual** | Good | Excellent | 🏆 Transformer |

## 🎛️ Configuration Options

### OCR Table Processing (Recommended)

```bash
# Optimized for OCR tables with many sequence numbers
python3 data_generation_noisy.py --mode create-dataset --country all \
  --train-size 100000 --dev-size 20000 \
  --custom-weights '{
    "financial_heavy": 50,    # Enhanced with multi-sequence patterns
    "table_heavy": 15,        # NEW: Dense table rows simulation
    "personal_id": 10,        # ID-focused training
    "address_focused": 10,    # Address extraction
    "contact_only": 10,       # Contact information
    "full": 5                 # General training
  }' \
  --noise --noise-level 0.4   # Higher noise for OCR corruption
```

### Standard Data Generation

```bash
python data_generation_noisy.py \
    --examples 50000 \
    --output-dir output \
    --countries chile mexico brazil uruguay \
    --noise-level 0.3 \
    --templates enhanced \
    --validation strict
```

### Available Parameters

| Parameter | Default | Description | OCR Recommendation |
|-----------|---------|-------------|-------------------|
| `--examples` | `50000` | Number of training examples | 100,000+ |
| `--output-dir` | `output` | Output directory | - |
| `--countries` | `all` | Countries to include | `all` |
| `--noise-level` | `0.3` | OCR noise probability | `0.4-0.5` |
| `--templates` | `enhanced` | Template set to use | `enhanced` |
| `--validation` | `strict` | Entity validation mode | `strict` |

### Entity Mode Weights (OCR Optimized)

| Mode | Weight | Entities Generated | OCR Benefit |
|------|--------|-------------------|-------------|
| `financial_heavy` | 50% | AMOUNT + SEQ_NUMBER + DATE (+ multi-sequence) | ✅ High sequence density |
| `table_heavy` | 15% | **NEW**: 3-6 entities per line | ✅ Table row simulation |
| `personal_id` | 10% | NAME + ID_NUMBER | ✅ ID extraction |
| `address_focused` | 10% | NAME + ADDRESS | ✅ Address parsing |
| `contact_only` | 10% | NAME + PHONE + EMAIL | ✅ Contact extraction |
| `full` | 5% | All entity types | ✅ General coverage |

## 🧪 Testing and Validation

### Test Entity Preservation

```bash
python test_entity_preservation.py
```

**Expected Output:**
```
🧪 Testing Entity Preservation with Realistic Noise
====================================================
✅ All 1000 examples processed successfully (100.0% success rate)
📊 Entity success rates:
   - CUSTOMER_NAME: 995/1000 (99.5%)
   - ID_NUMBER: 942/1000 (94.2%)
   - ADDRESS: 973/1000 (97.3%)
   - PHONE_NUMBER: 981/1000 (98.1%)
   - EMAIL: 987/1000 (98.7%)
   - AMOUNT: 968/1000 (96.8%)
   - SEQ_NUMBER: 897/1000 (89.7%)
```

### Test OCR Comparison

```bash
python test_ocr_comparison.py
```

This script analyzes OCR noise effects and compares entity detection before/after noise application.

## 🔍 Advanced Usage

### Custom Country Data

```python
# Add new country support
COUNTRY_DATA["colombia"] = {
    "first_names": ["CARLOS", "MARÍA", ...],
    "surnames": ["GARCÍA", "LÓPEZ", ...],
    "id_prefix": "CC",
    # ... more country-specific data
}
```

### Custom Entity Types

```python
# Add new entity type
def generate_new_entity():
    # Custom entity generation logic
    return entity_value

# Update noisy_templates to include new entity
```

### Performance Tuning

```python
# Adjust noise levels for specific use cases
noise_probabilities = {
    "ocr_substitution": 0.15,  # Reduce for cleaner data
    "spacing_variation": 0.10,
    "character_repetition": 0.05
}
```

## 📋 Troubleshooting

### Common Issues

#### High Failed Spans Rate
```bash
# Check noise levels
python data_generation_noisy.py --noise-level 0.2  # Reduce noise

# Validate entity detection
python test_entity_preservation.py
```

#### spaCy Model Not Found
```bash
# Install Spanish model
python -m spacy download es_core_news_lg

# Or use smaller model
python -m spacy download es_core_news_sm
```

#### Memory Issues with Large Datasets
```bash
# Process in smaller batches
python data_generation_noisy.py --examples 25000  # Reduce size
python data_generation_noisy.py --examples 25000 --append  # Add more
```

## 🔄 Migration Notes

### From This spaCy Version to Transformers

If you want to try the Transformer version (in `../Transformers/`):

```bash
# spaCy approach (this folder)
python data_generation_noisy.py --examples 50000

# Transformer approach (../Transformers/)
cd ../Transformers
python transformer_data_generator.py --train-size 50000 --dev-size 10000
```

### Key Differences
- **Data Format**: `.spacy` binary vs `.json` with entities
- **Training**: spaCy CLI vs Hugging Face Trainer
- **Models**: spaCy CNN vs BERT multilingual
- **Speed**: spaCy faster vs Transformer more accurate

## 📊 Expected Results

### OCR Table Dataset Generation
```
🚀 Generating 100,000 examples optimized for OCR tables...
🌎 Countries: Chile, Mexico, Brazil, Uruguay
🎭 Noise level: 0.4 (OCR optimized)
💰 financial_heavy: 50% (enhanced with multi-sequence patterns)
🏗️  table_heavy: 15% (NEW - dense table simulation)
✅ Generated 100,000 examples successfully
📊 Entity success rate: 96.8% (expected: 3.2% failed spans)
🔢 Average entities per example: 5.2 (including multi-sequence)
📁 Files created:
   - multi_country_training_data_noisy_100000.spacy
   - multi_country_train_noisy_80000.spacy
   - multi_country_dev_noisy_20000.spacy
   - multi_country_dataset_stats_noisy_100000.json
```

### Model Training Results
```
python -m spacy train config.cfg --output models
✅ Training completed successfully
📊 Best F1 Score: 0.894 (OCR table optimized)
📁 Model saved: models/model-best
🎯 Sequence number detection: 92%+ (improved for tables)
📋 Multi-entity handling: Enhanced
```

### Sample OCR Table Processing
```
Input (OCR table row):
"$1,500 | 079276-A | 12/03/2015 | $2,300 | 079277-A | 12/03/2015"

Detected Entities:
- AMOUNT: "$1,500" (0-6)
- SEQ_NUMBER: "079276-A" (9-17)  
- DATE: "12/03/2015" (20-30)
- AMOUNT: "$2,300" (33-39)
- SEQ_NUMBER: "079277-A" (42-50)
- DATE: "12/03/2015" (53-63)

✅ Success: 6/6 entities detected (100%)
```

## 🎯 When to Use This spaCy Version

**Choose spaCy version when you need:**
- ✅ **Fast inference** in production
- ✅ **Small model size** for deployment
- ✅ **Simple deployment** with minimal dependencies
- ✅ **Quick training** for rapid iteration
- ✅ **Low memory usage** in constrained environments
- ✅ **OCR table processing** with multiple sequence numbers
- ✅ **High entity density** training (3-12 entities per example)

**Choose Transformer version when you need:**
- ✅ **Maximum accuracy** for critical applications
- ✅ **Better multilingual** understanding
- ✅ **Latest NLP techniques** (attention mechanisms)
- ✅ **Industry standard** approaches

## 🔥 OCR Table Processing Benefits

### Why This spaCy Version Excels at OCR Tables:

1. **Multi-Sequence Training**: Up to 6 sequence numbers per training example
2. **Table Format Variety**: 10+ different table separator patterns
3. **OCR Corruption Simulation**: Realistic spacing and character issues
4. **High Entity Density**: 3-12 entities per example vs 3-7 standard
5. **Sequence-Heavy Focus**: 65% of training data optimized for sequences

### Real-World OCR Table Examples:
```
✅ Insurance claim tables with policy numbers
✅ Financial transaction logs with reference codes  
✅ Inventory systems with product/batch numbers
✅ Government forms with application/case numbers
✅ Banking statements with transaction references
```

## 🙏 Optimization History

This spaCy solution represents extensive optimization work:

1. **Initial Implementation**: Basic data generation
2. **Failed Spans Analysis**: Identified 31.7% failure rate
3. **OCR Noise Optimization**: Reduced aggressive noise
4. **Template Enhancement**: Added 10 new diverse templates  
5. **5-Strategy Entity Detection**: Robust pattern matching
6. **SEQ_NUMBER Fix**: Comprehensive sequence detection
7. **Final Optimization**: Achieved 96.8% success rate
8. **🔥 NEW - OCR Table Enhancement**: Multi-sequence training patterns
9. **🔥 NEW - table_heavy Mode**: Dense entity training (3-6 per line)
10. **🔥 NEW - Enhanced financial_heavy**: 30% multi-sequence patterns
11. **🔥 NEW - Date Generation**: Compact formats without separators

### Latest Enhancements (August 2025):
- ✅ **Enhanced financial_heavy mode**: Now includes 30% multi-sequence table patterns
- ✅ **NEW table_heavy mode**: Specialized for OCR tables with 3-6 entities per line
- ✅ **60+ template variety**: Expanded from 20 to 60+ diverse patterns
- ✅ **OCR character confusion**: O/0, I/1/l, 5/S, 6/G, 8/B pattern handling
- ✅ **Compact date formats**: Support for dates without separators (10252025)
- ✅ **Multi-country optimization**: Enhanced for Chile, Mexico, Brazil, Uruguay

The optimization journey is documented in the [`../docs/`](../docs/) folder:
- [`../docs/FAILED_SPANS_ANALYSIS.md`](../docs/FAILED_SPANS_ANALYSIS.md)
- [`../docs/FAILED_SPANS_FIX_PLAN.md`](../docs/FAILED_SPANS_FIX_PLAN.md) 
- [`../docs/FAILED_SPANS_PROGRESS.md`](../docs/FAILED_SPANS_PROGRESS.md)
- [`../docs/FAILED_SPANS_MODEL_IMPACT.md`](../docs/FAILED_SPANS_MODEL_IMPACT.md)

📚 **Complete documentation index**: [`../docs/README.md`](../docs/README.md)

## 👨‍💻 Author

**Andrés Vera Figueroa**
- Optimization Period: Multiple iterations over several months
- Achievement: 89% improvement in entity success rates
- Purpose: Production-ready spaCy NER for Latin American PII

---

**This spaCy solution is production-ready and highly optimized! 🚀**

For maximum accuracy needs, consider the Transformer version in `../Transformers/`.
For production speed and efficiency, this spaCy version is the optimal choice.
