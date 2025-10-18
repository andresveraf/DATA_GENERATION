# Transformers Training Data Directory

This directory contains training data formatted for Transformer-based NER model training (BERT, RoBERTa, etc.).

## Directory Structure

```
data_transformers/
├── train/          # Training data files
├── dev/            # Development/validation data files  
├── test/           # Test data files
└── README.md       # This file
```

## Data Formats

### JSON Format (.json)
- `train.json` - JSON format training data for Hugging Face
- `dev.json` - JSON format development data
- `test.json` - JSON format test data

### CoNLL Format (.conll)
- `train.conll` - CoNLL-2003 format training data
- `dev.conll` - CoNLL-2003 format development data
- `test.conll` - CoNLL-2003 format test data

### CSV Format (.csv)
- `train.csv` - CSV format for easy inspection
- `dev.csv` - CSV format development data
- `test.csv` - CSV format test data

## JSON Format Structure (Hugging Face Compatible)

```json
{
  "id": "example_001",
  "tokens": ["El", "cliente", "Juan", "Pérez", "con", "RUT", "12.345.678-9", "reside", "en", "Av.", "Providencia", "123", ",", "Santiago", "."],
  "ner_tags": ["O", "O", "B-CUSTOMER_NAME", "I-CUSTOMER_NAME", "O", "O", "B-ID_NUMBER", "O", "O", "B-ADDRESS", "I-ADDRESS", "I-ADDRESS", "O", "B-ADDRESS", "O"]
}
```

## CoNLL Format Structure

```
El O
cliente O
Juan B-CUSTOMER_NAME
Pérez I-CUSTOMER_NAME
con O
RUT O
12.345.678-9 B-ID_NUMBER
reside O
en O
Av. B-ADDRESS
Providencia I-ADDRESS
123 I-ADDRESS
, O
Santiago B-ADDRESS
. O

```

## Entity Labels (BIO Format)

### B- (Beginning) and I- (Inside) Tags
- **B-CUSTOMER_NAME** / **I-CUSTOMER_NAME**: Full customer names
- **B-ID_NUMBER** / **I-ID_NUMBER**: Country-specific ID formats
- **B-ADDRESS** / **I-ADDRESS**: Street addresses and cities
- **B-PHONE_NUMBER** / **I-PHONE_NUMBER**: Phone numbers
- **B-EMAIL** / **I-EMAIL**: Email addresses
- **B-AMOUNT** / **I-AMOUNT**: Monetary amounts
- **B-SEQ_NUMBER** / **I-SEQ_NUMBER**: Sequential reference numbers
- **B-DATE** / **I-DATE**: Date information
- **B-DIRECTION** / **I-DIRECTION**: Direction/orientation information
- **B-LOCATION** / **I-LOCATION**: Specific location references
- **B-POSTAL_CODE** / **I-POSTAL_CODE**: Postal/zip codes
- **B-REGION** / **I-REGION**: Region/state information
- **O**: Outside any entity

## Usage

### Training with Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset('json', data_files={
    'train': 'data_transformers/train.json',
    'validation': 'data_transformers/dev.json',
    'test': 'data_transformers/test.json'
})

# Load model and tokenizer
model_name = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=25)  # Adjust based on label count

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
)

# Train
trainer.train()
```

### Using the Project Pipeline

```bash
# Generate transformer training data
python main_pipeline.py --mode mixed-dataset --size 10000 --export-formats json,conll --output-dir data_transformers/

# Train using the transformer module
cd Transformers/
python train_transformer_ner.py --data_dir ../data_transformers/ --model_name dccuchile/bert-base-spanish-wwm-uncased
```

### Loading CoNLL Data

```python
def load_conll_data(file_path):
    sentences = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        label = []
        
        for line in f:
            line = line.strip()
            if line == '':
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            else:
                token, tag = line.split()
                sentence.append(token)
                label.append(tag)
        
        if sentence:
            sentences.append(sentence)
            labels.append(label)
    
    return sentences, labels

# Usage
sentences, labels = load_conll_data('data_transformers/train.conll')
```

## Recommended Models

### Spanish Models
- `dccuchile/bert-base-spanish-wwm-uncased`
- `PlanTL-GOB-ES/roberta-base-bne`
- `BSC-TeMU/roberta-base-bne`

### Portuguese Models
- `neuralmind/bert-base-portuguese-cased`
- `pierreguillou/bert-base-cased-pt-lenerbr`

### Multilingual Models
- `bert-base-multilingual-cased`
- `xlm-roberta-base`

## Data Quality

- Tokenization compatible with transformer models
- Proper BIO tagging scheme
- Balanced entity distribution
- Country-specific localization
- Realistic noise and variations
- No overlapping entities
- Consistent label mapping

## Configuration

Generate data with specific parameters:

```python
# Generate balanced dataset
python main_pipeline.py --mode mixed-dataset --size 20000 --composition balanced --export-formats json,conll,csv --output-dir data_transformers/

# Generate with augmentation
python main_pipeline.py --mode mixed-dataset --size 15000 --augmentation-enabled --augmentation-rate 0.3 --output-dir data_transformers/
```

## Evaluation Metrics

When training, monitor these metrics:
- **Precision**: Correctly identified entities / Total identified entities
- **Recall**: Correctly identified entities / Total actual entities  
- **F1-Score**: Harmonic mean of precision and recall
- **Entity-level accuracy**: Exact match accuracy for complete entities

## Tips for Training

1. **Batch Size**: Start with 16-32 for base models
2. **Learning Rate**: 2e-5 to 5e-5 typically works well
3. **Epochs**: 3-5 epochs usually sufficient
4. **Warmup**: Use 10% of total steps for warmup
5. **Evaluation**: Evaluate every 500-1000 steps
6. **Early Stopping**: Monitor validation F1-score
