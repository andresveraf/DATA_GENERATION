# Quick Setup Guide - Latin American NLP Data Generator

## 🚀 5-Minute Quick Start

### Step 1: Install Dependencies
```bash
# Install spaCy
pip install spacy

# Download Spanish language model (570MB)
python3 -m spacy download es_core_news_lg
```

### Step 2: Clone and Test
```bash
# Navigate to your project
cd /Users/andresverafigueroa/Documents/GitHub/DATA_GENERATION

# Test the installation
python3 data_generation.py
```

### Step 3: Generate Your First Dataset
```bash
# Create a small test dataset (1000 training, 250 dev examples)
python3 data_generation.py --mode create-dataset --train-size 1000 --dev-size 250

# Output files will be created:
# - train.spacy (training data)
# - dev.spacy (development data)
# - generation_stats.json (statistics)
```

## 📊 Dataset Size Quick Reference

### How to Modify Dataset Sizes

#### Method 1: Command Line (Recommended)
```bash
# Small dataset for testing
python3 data_generation.py --mode create-dataset --train-size 500 --dev-size 100

# Medium dataset for development  
python3 data_generation.py --mode create-dataset --train-size 5000 --dev-size 1000

# Large dataset for production
python3 data_generation.py --mode create-dataset --train-size 50000 --dev-size 10000

# Custom output directory
python3 data_generation.py --mode create-dataset --train-size 10000 --dev-size 2000 --output-dir ./my_datasets
```

#### Method 2: Edit Code Defaults
In `data_generation.py`, find these lines and modify:

```python
# Around line 877 - Command line defaults
parser.add_argument("--train-size", type=int, default=8000, help="Training set size")
parser.add_argument("--dev-size", type=int, default=2000, help="Development set size")

# Around line 655 - Function defaults  
def make_docbin(n_total: int = 100000, balance: bool = True, output_dir: str = "."):
```

#### Method 3: Direct Function Calls
```python
from data_generation import create_training_dataset, make_docbin

# Create custom-sized dataset
create_training_dataset(
    train_size=15000,
    dev_size=3000,
    output_dir="./datasets"
)

# Or use make_docbin directly
docbin, stats = make_docbin(n_total=25000, balance=True)
```

## 🎯 Size Recommendations

| Use Case | Train Size | Dev Size | Time | Memory |
|----------|------------|----------|------|--------|
| **Quick Test** | 100 | 25 | 30s | 1MB |
| **Development** | 1,000 | 250 | 2min | 10MB |
| **Small Model** | 5,000 | 1,000 | 5min | 50MB |
| **Medium Model** | 15,000 | 3,000 | 15min | 150MB |
| **Large Model** | 50,000 | 10,000 | 45min | 500MB |
| **Production** | 100,000+ | 20,000+ | 2h+ | 1GB+ |

## 🎛️ Generation Modes

Test different entity complexity levels:

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

## 🌍 Supported Countries

Test each country's data format:

```bash
# Chile
python3 -c "from data_generation import generate_example; print(generate_example('CL')[0])"

# Argentina  
python3 -c "from data_generation import generate_example; print(generate_example('AR')[0])"

# Brazil
python3 -c "from data_generation import generate_example; print(generate_example('BR')[0])"

# Uruguay
python3 -c "from data_generation import generate_example; print(generate_example('UY')[0])"

# Mexico
python3 -c "from data_generation import generate_example; print(generate_example('MX')[0])"
```

## 🔍 Verify Your Setup

### Quick Functionality Test
```bash
python3 -c "
from data_generation import quick_test
quick_test()
"
```

### Check spaCy Model
```bash
python3 -c "
import spacy
nlp = spacy.load('es_core_news_lg')
print(f'✅ spaCy model loaded: {nlp.meta[\"name\"]} v{nlp.meta[\"version\"]}')
"
```

### Test Dataset Creation
```bash
# Create tiny dataset to verify everything works
python3 data_generation.py --mode create-dataset --train-size 10 --dev-size 5
ls -la *.spacy *.json
```

## 📁 Output File Structure

After running the generator, you'll get:

```
├── train.spacy          # Training dataset (spaCy DocBin format)
├── dev.spacy           # Development dataset (spaCy DocBin format)  
└── generation_stats.json  # Statistics and metadata
```

### Using the Generated Files

```python
import spacy
from spacy.tokens import DocBin

# Load the spaCy model
nlp = spacy.load("es_core_news_lg")

# Load training data
train_docbin = DocBin().from_disk("train.spacy")
train_docs = list(train_docbin.get_docs(nlp.vocab))

print(f"Loaded {len(train_docs)} training examples")

# Examine first example
doc = train_docs[0]
print(f"Text: {doc.text}")
print("Entities:")
for ent in doc.ents:
    print(f"  {ent.label_}: '{ent.text}' ({ent.start_char}-{ent.end_char})")
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. spaCy Model Not Found
```bash
# Error: Can't find model 'es_core_news_lg'
# Solution:
python3 -m spacy download es_core_news_lg
```

#### 2. Memory Error
```bash
# Error: MemoryError during generation
# Solution: Reduce dataset size
python3 data_generation.py --mode create-dataset --train-size 1000 --dev-size 250
```

#### 3. Permission Denied
```bash
# Error: Permission denied writing files
# Solution: Change output directory
python3 data_generation.py --mode create-dataset --output-dir ~/datasets
```

#### 4. Import Error
```bash
# Error: ModuleNotFoundError: No module named 'spacy'
# Solution: Install spaCy
pip install spacy
```

### Verify Installation
```bash
# Check Python version (requires 3.8+)
python3 --version

# Check spaCy installation
python3 -c "import spacy; print(f'spaCy version: {spacy.__version__}')"

# Check available models
python3 -m spacy info
```

## ⚡ Performance Tips

### For Large Datasets (50K+ examples)
1. **Monitor memory usage**: Use Activity Monitor/Task Manager
2. **Use SSD storage**: Faster file I/O
3. **Close other applications**: Free up RAM
4. **Process in chunks**: Split into smaller batches

### Optimize Generation Speed
```python
# Generate in batches
for i in range(0, total_size, batch_size):
    batch_end = min(i + batch_size, total_size)
    create_training_dataset(
        train_size=batch_end - i,
        dev_size=0,
        output_dir=f"./batch_{i}"
    )
```

## 📊 Monitor Progress

### Check Generation Progress
```bash
# Run with verbose output
python3 data_generation.py --mode create-dataset --train-size 10000 --dev-size 2000 2>&1 | tee generation.log

# Monitor file sizes during generation
watch -n 5 'ls -lh *.spacy *.json 2>/dev/null || echo "No files yet..."'
```

### Analyze Generated Statistics
```bash
# View statistics after generation
python3 -c "
import json
with open('generation_stats.json') as f:
    stats = json.load(f)
    print(json.dumps(stats, indent=2))
"
```

## 🧪 Step 4: Train and Test Your Model

### Train the NER Model

```bash
# 1. Create spaCy configuration
python3 -m spacy init config config.cfg --lang es --pipeline ner

# 2. Train the model (using your generated data)
python3 -m spacy train config.cfg --output ./model --paths.train train.spacy --paths.dev dev.spacy
```

### Test Your Trained Model

#### Quick Test
```bash
# Run simple test with predefined example
python3 quick_test.py
```

**Expected Output:**
```
✅ Trained model loaded!
📝 Testing: El cliente Juan Carlos González con RUT 15.234.567-8 vive en Av. Providencia 123, Santiago.

🎯 Found 4 entities:
   CUSTOMER_NAME | Juan Carlos González
   ID_NUMBER     | 15.234.567-8  
   ADDRESS       | Av. Providencia 123
   ADDRESS       | Santiago
```

#### Interactive Testing
```bash
# Run full-featured test script
python3 test_pii_ner.py

# Choose option 2 for interactive mode
# Enter your own text to test PII detection
```

#### Test Your Own Text
```python
# Modify quick_test.py to test custom text
test_text = "Tu texto aquí con nombres, RUT y direcciones"
```

### Typical Test Results

| Input | Detected Entities |
|-------|------------------|
| "María José Silva RUT 12.345.678-9" | CUSTOMER_NAME, ID_NUMBER |
| "Av. Providencia 123, Santiago" | ADDRESS (2 entities) |
| "Tel: +56 9 1234 5678" | PHONE_NUMBER |
| "Reclamo 7009808" | SEQ_NUMBER |

## 🎯 Next Steps

1. **Test with real data**: Use your actual customer service text
2. **Fine-tune**: Adjust training parameters based on test results  
3. **Scale up**: Generate larger datasets for production use
4. **Deploy**: Integrate the trained model into your application

---

**Ready to start generating Latin American NLP datasets! 🚀**
