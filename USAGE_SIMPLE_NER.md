# Simple NER Dataset Generator - Usage Guide

## 🎯 Purpose

This script (`generate_ner_dataset_address_sex.py`) generates synthetic training data for Named Entity Recognition (NER) models focused on detecting:
- **ADDRESS** - Physical addresses
- **SEX/GENDER** - Gender/sex identifiers
- **NAME** - Person names (bonus)

Perfect for training transformer models like BERT, RoBERTa, DistilBERT, etc.

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements_ner_simple.txt

# Or install directly
pip install faker
```

## 🚀 Quick Start

### Basic Usage (1000 samples)
```bash
python generate_ner_dataset_address_sex.py
```

This will create a `ner_dataset/` folder with:
- `train.jsonl` (800 samples) - For transformers
- `val.jsonl` (200 samples) - For validation
- `train.conll` (800 samples) - BIO format
- `val.conll` (200 samples) - BIO format
- `train.csv` (800 samples) - Human-readable
- `val.csv` (200 samples) - Human-readable
- `dataset_stats.json` - Dataset metadata

### Generate Medium Dataset (5000 samples)
```bash
python generate_ner_dataset_address_sex.py --num-samples 5000
```

### Generate Small Dataset (500 samples)
```bash
python generate_ner_dataset_address_sex.py -n 500
```

### Custom Output Directory
```bash
python generate_ner_dataset_address_sex.py -n 2000 -o my_custom_dataset
```

### Custom Train/Val Split (85% train, 15% val)
```bash
python generate_ner_dataset_address_sex.py -n 3000 --train-ratio 0.85
```

### Set Random Seed for Reproducibility
```bash
python generate_ner_dataset_address_sex.py --seed 12345
```

## 📊 Output Formats

### 1. JSONL Format (for Transformers)
Each line is a JSON object:
```json
{
  "text": "El cliente Juan Pérez de género masculino vive en Calle Principal 123, Madrid.",
  "entities": [
    {"start": 11, "end": 21, "label": "NAME", "text": "Juan Pérez"},
    {"start": 32, "end": 41, "label": "SEX", "text": "masculino"},
    {"start": 51, "end": 81, "label": "ADDRESS", "text": "Calle Principal 123, Madrid"}
  ],
  "language": "es"
}
```

### 2. CONLL/BIO Format (Standard NER)
```
El          O
cliente     O
Juan        B-NAME
Pérez       I-NAME
de          O
género      O
masculino   B-SEX
vive        O
en          O
Calle       B-ADDRESS
Principal   I-ADDRESS
123         I-ADDRESS
,           I-ADDRESS
Madrid      I-ADDRESS
.           O

```

### 3. CSV Format (for Inspection)
Human-readable format with columns:
- `text` - The full sentence
- `language` - Language code (es/en/pt)
- `entities_json` - JSON string of entities
- `num_entities` - Count of entities

## 🌍 Multi-language Support

The script generates samples in 3 languages:
- **Spanish (es)** - "masculino", "femenino", "vive en"
- **English (en)** - "male", "female", "lives at"
- **Portuguese (pt)** - "masculino", "feminino", "mora em"

Languages are randomly mixed in the dataset for better model generalization.

## 🔧 Advanced Options

```bash
python generate_ner_dataset_address_sex.py \
  --num-samples 10000 \
  --train-ratio 0.9 \
  --output-dir large_dataset \
  --seed 2024
```

### All Arguments
- `-n, --num-samples` - Total samples (default: 1000)
- `-r, --train-ratio` - Train split ratio (default: 0.8)
- `-o, --output-dir` - Output folder (default: ner_dataset)
- `-s, --seed` - Random seed (default: 42)

## 💡 Usage with Transformers

### Loading JSONL Data
```python
import json

# Load training data
train_data = []
with open('ner_dataset/train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        train_data.append(json.loads(line))

# Inspect first sample
sample = train_data[0]
print(f"Text: {sample['text']}")
print(f"Entities: {sample['entities']}")
```

### Training with Hugging Face Transformers
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer

# Load model
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=7  # O, B-ADDRESS, I-ADDRESS, B-SEX, I-SEX, B-NAME, I-NAME
)

# Prepare your data with tokenization and alignment
# ... (data preparation code)

# Train
training_args = TrainingArguments(
    output_dir="./ner-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

## 📈 Example Output

```
🚀 Generating 1000 samples...
   Train: 800
   Val: 200
   Generated 100/1000 samples...
   Generated 200/1000 samples...
   ...
   Generated 1000/1000 samples...

💾 Saving datasets...
✅ Saved 800 samples to ner_dataset/train.jsonl
✅ Saved 200 samples to ner_dataset/val.jsonl
✅ Saved 800 samples to ner_dataset/train.conll (CONLL format)
✅ Saved 200 samples to ner_dataset/val.conll (CONLL format)
✅ Saved 800 samples to ner_dataset/train.csv
✅ Saved 200 samples to ner_dataset/val.csv

✨ Dataset generation complete!
📊 Statistics:
   Total samples: 1000
   Train: 800
   Val: 200
   Languages: ES=335, EN=332, PT=333
   Entity types: ADDRESS, SEX, NAME

📁 Files saved in: ner_dataset/
   - train.jsonl, val.jsonl (for transformers)
   - train.conll, val.conll (BIO format)
   - train.csv, val.csv (for inspection)
   - dataset_stats.json (metadata)

🎯 Next steps:
   1. Inspect the CSV files to verify data quality
   2. Use JSONL files for transformer fine-tuning
   3. Use CONLL files for spaCy or other NER frameworks
```

## 🎨 Customization

To customize the script for your needs, edit these sections:

### Add More Gender Terms
```python
GENDERS = {
    'es': ['masculino', 'femenino', 'no binario', 'otro'],
    # ... add more
}
```

### Add New Templates
```python
TEMPLATES_ES = [
    "El cliente {name} de género {sex} vive en {address}.",
    "Tu custom template here...",
]
```

### Change Entity Labels
In the `generate_sample()` function, modify:
```python
entities.append({
    'start': sex_start,
    'end': sex_start + len(sex),
    'label': 'GENDER',  # Change from 'SEX' to 'GENDER'
    'text': sex
})
```

## ⚠️ Important Notes

1. **Synthetic Data**: This is artificially generated data using Faker. For production models, consider mixing with real annotated data.

2. **Token Alignment**: The BIO tagging uses simple tokenization. For transformers, you'll need proper subword tokenization and alignment.

3. **Language Mixing**: Samples are randomly mixed across languages. Filter by language if needed:
   ```python
   spanish_samples = [s for s in samples if s['language'] == 'es']
   ```

4. **Entity Overlap**: The script ensures no overlapping entities by design.

## 🐛 Troubleshooting

### Import Error
```bash
# If you get "No module named 'faker'"
pip install faker
```

### Unicode Errors
The script uses UTF-8 encoding. If you encounter issues:
```python
# When reading files
with open('file.txt', 'r', encoding='utf-8') as f:
    data = f.read()
```

### Empty Entities
If some samples have no entities, it's because string matching failed. This is rare but can happen with special characters.

## 📚 References

- [Hugging Face NER Guide](https://huggingface.co/docs/transformers/tasks/token_classification)
- [BIO Tagging Format](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))
- [Faker Documentation](https://faker.readthedocs.io/)

## 🤝 Contributing

Feel free to modify and extend this script for your specific needs!

## 📄 License

MIT License - Free to use and modify.

