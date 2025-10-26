# SpaCy Training Data Directory

This directory contains training data formatted for spaCy NER model training using the **DocBin binary format**.

## 🎯 Quick Start

Generate spaCy format data:
```bash
python main_pipeline.py --mode mixed-dataset --size 10000 --export-formats spacy --output-dir data_spacy/
```

## Directory Structure

```
data_spacy/
├── train.spacy     # Binary training data (DocBin format)
├── dev.spacy       # Binary development data (DocBin format)
├── mixed_dataset.json  # Optional JSON export
└── README.md       # This file
```

## Data Formats

### 🔵 SpaCy Binary Format (.spacy) - **Primary Format**
The `.spacy` files use spaCy's DocBin format for efficient storage and loading:

- **train.spacy** - Binary training data optimized for spaCy v3+
- **dev.spacy** - Binary development/validation data
- **Format**: DocBin serialized Doc objects with entity spans
- **Advantages**: Fast loading, memory efficient, preserves tokenization

### 📄 JSON Format (.json) - **Optional**
- **mixed_dataset.json** - Human-readable JSON format (if requested)
- **Structure**: Documents with character-level entity spans

## JSON Format Structure

```json
{
  "text": "El cliente Juan Pérez con RUT 12.345.678-9 reside en Av. Providencia 123, Santiago.",
  "entities": [
    {"start": 11, "end": 21, "label": "CUSTOMER_NAME"},
    {"start": 26, "end": 39, "label": "ID_NUMBER"},
    {"start": 50, "end": 68, "label": "ADDRESS"},
    {"start": 70, "end": 78, "label": "ADDRESS"}
  ]
}
```

## Entity Labels

- **CUSTOMER_NAME**: Full customer names
- **ID_NUMBER**: Country-specific ID formats (RUT/CURP/CPF/Cédula)
- **ADDRESS**: Street addresses and cities
- **PHONE_NUMBER**: Phone numbers with country codes
- **EMAIL**: Email addresses
- **AMOUNT**: Monetary amounts with currency
- **SEQ_NUMBER**: Sequential reference numbers
- **DATE**: Date information in various formats
- **DIRECTION**: Direction/orientation information
- **LOCATION**: Specific location references
- **POSTAL_CODE**: Postal/zip codes
- **REGION**: Region/state information

## Usage

### Training a spaCy Model

```bash
# Using spaCy CLI
python -m spacy train config.cfg --output ./models --paths.train ./data_spacy/train.spacy --paths.dev ./data_spacy/dev.spacy

# Using the project pipeline
python main_pipeline.py --mode spacy-training --data-dir data_spacy/
```

### Loading Data in Python

```python
import spacy
from spacy.tokens import DocBin

# Load binary data
nlp = spacy.blank("es")
doc_bin = DocBin().from_disk("data_spacy/train.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

# Load JSON data
import json
with open("data_spacy/train.json", "r") as f:
    data = json.load(f)
```

## Data Quality

- All data is generated with country-specific localization
- Entity boundaries are guaranteed to be non-overlapping
- Text includes realistic noise and variations
- Balanced representation across all entity types
- Multiple countries supported (Chile, Mexico, Brazil, Uruguay)

## 🔧 Loading SpaCy Data

### Loading .spacy Files in Python

```python
import spacy
from spacy.tokens import DocBin

# Load the language model
nlp = spacy.blank("es")  # or spacy.load("es_core_news_sm")

# Load training data
doc_bin = DocBin().from_disk("data_spacy/train.spacy")
train_docs = list(doc_bin.get_docs(nlp.vocab))

# Access documents and entities
for doc in train_docs:
    print(f"Text: {doc.text}")
    for ent in doc.ents:
        print(f"  Entity: '{ent.text}' -> {ent.label_} ({ent.start_char}-{ent.end_char})")
```

### Using with spaCy Training

```bash
# Train a spaCy NER model
python -m spacy train config.cfg --output ./models --paths.train data_spacy/train.spacy --paths.dev data_spacy/dev.spacy
```

## 📊 Usage Examples

Generate spaCy training data:

```bash
# Generate balanced mixed dataset for spaCy
python main_pipeline.py --mode mixed-dataset --size 10000 --export-formats spacy --output-dir data_spacy/

# Generate with specific composition
python main_pipeline.py --mode mixed-dataset --size 5000 --composition balanced --export-formats spacy

# Generate both spaCy and JSON formats
python main_pipeline.py --mode mixed-dataset --size 15000 --export-formats spacy,json --output-dir data_spacy/
```
