# Latin American Customer Data Generator - GitHub Copilot Instructions

**ALWAYS follow these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Project Overview

Latin American Customer Data Generator is a Python-based NLP training dataset generator that creates realistic customer data for Named Entity Recognition (NER) training. It supports 5 Latin American countries and generates spaCy-compatible training datasets with 7 entity types.

## Working Effectively

### Initial Setup (One-time, ~5 minutes total)

**NEVER CANCEL** - Setup commands may take several minutes. Always wait for completion.

```bash
# Check Python version (requires 3.8+)
python3 --version

# Install required dependencies (~2 minutes)
pip install spacy pandas openpyxl

# Download Spanish language model (~3 minutes, 570MB download)
python3 -m spacy download es_core_news_lg
```

### Quick Validation

**Run these commands to verify everything works correctly:**

```bash
# Test basic functionality (should complete in <5 seconds)
python3 data_generation.py

# Test pre-trained model (should complete in <5 seconds)
python3 quick_test.py

# Verify spaCy installation
python3 -c "import spacy; print('spaCy version:', spacy.__version__)"
python3 -c "import spacy; nlp = spacy.load('es_core_news_lg'); print('✅ Spanish model loaded successfully')"
```

**Expected Output from quick_test.py:**
```
✅ Trained model loaded successfully from ./precise_model/model-best!
📝 Testing: El cliente Juan Carlos González con RUT 15.234.567-8 vive en Av. Providencia 123, Santiago.
🎯 Found 4 entities:
   CUSTOMER_NAME | Juan Carlos González
   ID_NUMBER    | 15.234.567-8
   ADDRESS      | Av. Providencia 123
   ADDRESS      | Santiago
```

### Core Data Generation Workflows

**All dataset generation commands complete in under 10 seconds unless noted otherwise.**

```bash
# Create small test dataset (125 examples, ~3 seconds)
python3 data_generation.py --mode create-dataset --train-size 100 --dev-size 25

# Create development dataset (1,250 examples, ~3 seconds)
python3 data_generation.py --mode create-dataset --train-size 1000 --dev-size 250

# Create medium production dataset (6,000 examples, ~5 seconds)
python3 data_generation.py --mode create-dataset --train-size 5000 --dev-size 1000

# Create large production dataset (60,000 examples, ~30 seconds)
python3 data_generation.py --mode create-dataset --train-size 50000 --dev-size 10000

# Custom output directory
python3 data_generation.py --mode create-dataset --train-size 1000 --dev-size 250 --output-dir ./my_datasets
```

### Excel Export for Data Review

```bash
# Create Excel review file (25 examples, ~1 second)
python3 data_generation.py --mode excel-export --excel-examples 25

# Create comprehensive Excel file (200 examples, ~2 seconds)
python3 data_generation.py --mode excel-export --excel-examples 200 --excel-file full_review.xlsx
```

### Model Training Pipeline

**NEVER CANCEL** - Model training can take 1-30 minutes depending on dataset size. Set timeout to 60+ minutes.

```bash
# Create spaCy training configuration (~1 second)
python3 -m spacy init config config.cfg --lang es --pipeline ner

# Train model with small dataset (~1-2 minutes)
python3 -m spacy train config.cfg --output ./model --paths.train train.spacy --paths.dev dev.spacy --training.max_epochs 5

# Train model with large dataset (~10-30 minutes) - NEVER CANCEL
python3 -m spacy train config.cfg --output ./model --paths.train train.spacy --paths.dev dev.spacy --training.max_epochs 10
```

**Timing expectations:**
- Small dataset (1K examples): 1-2 minutes
- Medium dataset (10K examples): 5-10 minutes  
- Large dataset (50K+ examples): 15-30 minutes

## Validation

**ALWAYS run these validation steps after making changes:**

### Manual Testing Scenarios

```bash
# 1. Test basic data generation
python3 data_generation.py

# 2. Test model loading and inference
python3 quick_test.py

# 3. Test dataset creation with small size
python3 data_generation.py --mode create-dataset --train-size 10 --dev-size 5

# 4. Verify generated files exist and are valid
ls -la *.spacy *.json
python3 -c "
import spacy
from spacy.tokens import DocBin
nlp = spacy.load('es_core_news_lg')
docbin = DocBin().from_disk('train.spacy')
docs = list(docbin.get_docs(nlp.vocab))
print(f'✅ Loaded {len(docs)} training examples')
print(f'✅ First example: {docs[0].text[:100]}...')
"
```

### Comprehensive Testing

```bash
# Interactive testing with custom text (requires trained model or uses base model)
python3 test_pii_ner.py
# If no trained model found, select 'y' to use base spaCy model
# Choose option 2 for interactive mode
# Test with: "El cliente María González con RUT 12.345.678-9 vive en Santiago"

# Model comparison (if multiple models available)
python3 compare_models.py
```

### Performance Validation

**Always test end-to-end workflow:**

1. Generate dataset: `python3 data_generation.py --mode create-dataset --train-size 100 --dev-size 25`
2. Train model: `python3 -m spacy train config.cfg --output ./test_model --paths.train output/train.spacy --paths.dev output/dev.spacy --training.max_epochs 3`
3. Test model: Modify `quick_test.py` to load `./test_model/model-best` and run

**Test different entity generation modes:**

```bash
# Test all supported countries and modes
for country in CL AR BR UY MX; do
  echo "Testing $country:"
  python3 -c "from data_generation import generate_example_with_mode; print(generate_example_with_mode('$country', 'full')[0])"
done
```

**Validate timing expectations:**

```bash
# All these should complete in under 10 seconds
time python3 data_generation.py --mode create-dataset --train-size 100 --dev-size 25
time python3 data_generation.py --mode excel-export --excel-examples 25
time python3 quick_test.py
```

## Repository Structure

```
DATA_GENERATION/
├── data_generation.py           # Main data generator (primary script)
├── quick_test.py               # Simple model testing
├── test_pii_ner.py            # Comprehensive testing
├── compare_models.py          # Model comparison tool
├── precise_config.cfg         # Optimized training configuration
├── precise_model/             # Pre-trained best model (100% accuracy)
│   ├── model-best/           # Primary trained model
│   └── model-last/           # Backup model
├── large_dataset/            # Large pre-generated datasets
├── documentation/            # Comprehensive documentation
│   ├── QUICK_START.md       # Getting started guide
│   ├── README.md            # Technical overview
│   ├── TECHNICAL_DOCS.md    # Detailed specifications
│   └── EXCEL_EXPORT_GUIDE.md # Export functionality
└── README.md                # Main project documentation
```

## Common Tasks

### Generate Different Data Types

```bash
# Full customer records (all 7 entity types)
python3 -c "from data_generation import generate_example_with_mode; print(generate_example_with_mode('CL', 'full')[0])"

# Address only
python3 -c "from data_generation import generate_example_with_mode; print(generate_example_with_mode('CL', 'addr_only')[0])"

# ID numbers only  
python3 -c "from data_generation import generate_example_with_mode; print(generate_example_with_mode('CL', 'id_only')[0])"

# Contact information
python3 -c "from data_generation import generate_example_with_mode; print(generate_example_with_mode('CL', 'contact_only')[0])"

# Financial data
python3 -c "from data_generation import generate_example_with_mode; print(generate_example_with_mode('CL', 'financial_only')[0])"
```

### Supported Countries and Formats

- **Chile (CL)**: RUT format (XX.XXX.XXX-X), +56 phone numbers, CLP currency
- **Argentina (AR)**: DNI format (XXXXXXXX), +54 phone numbers, ARS currency  
- **Brazil (BR)**: CPF format (XXX.XXX.XXX-XX), +55 phone numbers, BRL currency
- **Uruguay (UY)**: CI format (X.XXX.XXX-X), +598 phone numbers, UYU currency
- **Mexico (MX)**: CURP/RFC format, +52 phone numbers, MXN currency

### Entity Types Generated

1. **CUSTOMER_NAME**: Full names with Latin American naming patterns
2. **ID_NUMBER**: Country-specific government IDs
3. **ADDRESS**: Street addresses and cities
4. **PHONE_NUMBER**: Country-specific phone formats
5. **EMAIL**: Realistic email addresses
6. **AMOUNT**: Monetary amounts with proper currency formatting
7. **SEQ_NUMBER**: Sequential reference numbers

## Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'spacy'**
```bash
pip install spacy pandas openpyxl
```

**Can't find model 'es_core_news_lg'**
```bash
python3 -m spacy download es_core_news_lg
```

**Permission denied writing files**
```bash
python3 data_generation.py --mode create-dataset --output-dir ~/datasets
```

**Memory errors with large datasets**
```bash
# Reduce dataset size
python3 data_generation.py --mode create-dataset --train-size 1000 --dev-size 250
```

### Verification Commands

```bash
# Check all dependencies
python3 -c "
import spacy, pandas, openpyxl
print('✅ All dependencies installed')
nlp = spacy.load('es_core_news_lg')
print('✅ Spanish model available')
"

# Test core functionality
python3 -c "
from data_generation import quick_test
quick_test()
print('✅ Core functionality working')
"
```

## Performance Expectations

**NEVER CANCEL these operations - they complete quickly:**

- Basic data generation: <1 second
- Small dataset (100 examples): 3 seconds
- Medium dataset (1K examples): 3 seconds
- Large dataset (5K examples): 5 seconds
- Very large dataset (50K examples): 30 seconds
- Excel export (25 examples): 1 second
- Excel export (200 examples): 2 seconds

**Model training times (NEVER CANCEL):**
- Small dataset (10-100 examples, 3 epochs): 30 seconds - 1 minute
- Medium dataset (1K examples, 5 epochs): 1-2 minutes
- Large dataset (10K examples, 10 epochs): 5-10 minutes
- Production dataset (50K+ examples): 15-30 minutes

**Set explicit timeouts:**
- Dataset generation: 60 seconds minimum
- Model training: 60+ minutes (3600+ seconds)
- Dependency installation: 300 seconds

## Key Files Reference

### Most Frequently Used
- `data_generation.py` - Main script for all generation tasks
- `quick_test.py` - Fast model validation
- `documentation/QUICK_START.md` - Complete setup guide

### Configuration Files
- `precise_config.cfg` - Optimized spaCy training configuration
- `.gitignore` - Excludes test files and temporary artifacts

### Pre-trained Assets
- `precise_model/model-best/` - Best performing NER model (100% accuracy)
- `large_dataset/` - Pre-generated large datasets for immediate use

**Always reference documentation/QUICK_START.md for detailed examples and troubleshooting.**

## Agent Validation Checklist

**Before making any changes, ALWAYS run this validation checklist:**

1. **Basic functionality**: `python3 data_generation.py` (should show examples)
2. **Model testing**: `python3 quick_test.py` (should show 4 entities found)
3. **Small dataset**: `python3 data_generation.py --mode create-dataset --train-size 10 --dev-size 5` (~3 seconds)
4. **Excel export**: `python3 data_generation.py --mode excel-export --excel-examples 25` (~1 second)
5. **All countries**: Test the country validation loop shown above

**After making changes, ALWAYS run:**

1. All the validation steps above
2. **End-to-end test**: Create dataset → Train model → Test model
3. **Performance check**: Ensure operations complete within expected timeframes

**If ANY validation step fails:**
- Check dependencies: `python3 -c "import spacy, pandas, openpyxl; print('✅ All dependencies OK')"`
- Check Spanish model: `python3 -c "import spacy; spacy.load('es_core_news_lg'); print('✅ Spanish model OK')"`
- Review error messages and consult documentation/QUICK_START.md