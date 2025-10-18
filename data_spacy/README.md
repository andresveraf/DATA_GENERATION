# SpaCy Training Data Directory

This directory contains training data formatted for spaCy NER model training.

## Directory Structure

```
data_spacy/
├── train/          # Training data files
├── dev/            # Development/validation data files  
├── test/           # Test data files
└── README.md       # This file
```

## Data Formats

### SpaCy Binary Format (.spacy)
- `train.spacy` - Binary training data for spaCy
- `dev.spacy` - Binary development data for spaCy
- `test.spacy` - Binary test data for spaCy

### JSON Format (.json)
- `train.json` - JSON format training data
- `dev.json` - JSON format development data
- `test.json` - JSON format test data

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

## Configuration

The data generation can be configured through the main pipeline:

```python
# Generate spaCy training data
python main_pipeline.py --mode mixed-dataset --size 10000 --export-formats spacy --output-dir data_spacy/
```
