#!/usr/bin/env python3
"""
Script to generate a synthetic NER dataset for PII detection
Focused on ADDRESS and SEX/GENDER entities for transformer model training

Outputs:
- JSONL format (for transformer fine-tuning)
- CONLL/BIO format (standard NER format)
- CSV format (for inspection)
"""

import json
import random
import csv
from typing import List, Dict, Tuple
from faker import Faker

# Initialize Faker for multiple locales
fake_es = Faker('es_ES')  # Spanish
fake_en = Faker('en_US')  # English
fake_pt = Faker('pt_BR')  # Portuguese/Brazil

# Gender/Sex options
GENDERS = {
    'es': ['masculino', 'femenino', 'hombre', 'mujer', 'varón', 'dama', 'caballero', 'señor', 'señora'],
    'en': ['male', 'female', 'man', 'woman', 'gentleman', 'lady', 'sir', 'madam', 'boy', 'girl'],
    'pt': ['masculino', 'feminino', 'homem', 'mulher', 'senhor', 'senhora', 'rapaz', 'rapariga']
}

# Template sentences for context
TEMPLATES_ES = [
    "El cliente {name} de género {sex} vive en {address}.",
    "La persona {name} ({sex}) tiene domicilio en {address}.",
    "Contactar a {name}, {sex}, dirección: {address}.",
    "Paciente {name}, sexo {sex}, reside en {address}.",
    "Trabajador {name} de sexo {sex} registrado en {address}.",
    "Estudiante {name} ({sex}) con residencia en {address}.",
    "El/La {sex} {name} declaró vivir en {address}.",
    "Según el registro, {name} es de género {sex} y su dirección es {address}.",
]

TEMPLATES_EN = [
    "The customer {name} of gender {sex} lives at {address}.",
    "The person {name} ({sex}) has residence at {address}.",
    "Contact {name}, {sex}, address: {address}.",
    "Patient {name}, sex {sex}, resides at {address}.",
    "Employee {name} of sex {sex} registered at {address}.",
    "Student {name} ({sex}) with residence at {address}.",
    "The {sex} {name} declared living at {address}.",
    "According to records, {name} is of gender {sex} and address is {address}.",
]

TEMPLATES_PT = [
    "O cliente {name} de gênero {sex} mora em {address}.",
    "A pessoa {name} ({sex}) tem domicílio em {address}.",
    "Contatar {name}, {sex}, endereço: {address}.",
    "Paciente {name}, sexo {sex}, reside em {address}.",
    "Funcionário {name} de sexo {sex} registrado em {address}.",
    "Estudante {name} ({sex}) com residência em {address}.",
    "O/A {sex} {name} declarou morar em {address}.",
    "Segundo o registro, {name} é de gênero {sex} e endereço é {address}.",
]


def generate_sample(language: str = 'es') -> Dict:
    """
    Generate a single training sample with ADDRESS and SEX entities
    
    Args:
        language: 'es' (Spanish), 'en' (English), or 'pt' (Portuguese)
    
    Returns:
        Dictionary with text and entities in BIO format
    """
    # Select appropriate faker and templates
    if language == 'es':
        faker = fake_es
        templates = TEMPLATES_ES
        genders = GENDERS['es']
    elif language == 'en':
        faker = fake_en
        templates = TEMPLATES_EN
        genders = GENDERS['en']
    else:  # pt
        faker = fake_pt
        templates = TEMPLATES_PT
        genders = GENDERS['pt']
    
    # Generate data
    name = faker.name()
    sex = random.choice(genders)
    address = faker.address().replace('\n', ', ')
    
    # Select template and fill it
    template = random.choice(templates)
    text = template.format(name=name, sex=sex, address=address)
    
    # Find entity positions (simple string matching)
    entities = []
    
    # Find NAME positions (we'll track but focus on ADDRESS and SEX)
    name_start = text.find(name)
    if name_start != -1:
        entities.append({
            'start': name_start,
            'end': name_start + len(name),
            'label': 'NAME',
            'text': name
        })
    
    # Find SEX positions
    sex_start = text.find(sex)
    if sex_start != -1:
        entities.append({
            'start': sex_start,
            'end': sex_start + len(sex),
            'label': 'SEX',
            'text': sex
        })
    
    # Find ADDRESS positions
    address_start = text.find(address)
    if address_start != -1:
        entities.append({
            'start': address_start,
            'end': address_start + len(address),
            'label': 'ADDRESS',
            'text': address
        })
    
    # Sort entities by start position
    entities.sort(key=lambda x: x['start'])
    
    return {
        'text': text,
        'entities': entities,
        'language': language
    }


def convert_to_bio(text: str, entities: List[Dict]) -> List[Tuple[str, str]]:
    """
    Convert text and entities to BIO (Beginning, Inside, Outside) format
    
    Args:
        text: The input text
        entities: List of entity dictionaries
    
    Returns:
        List of (token, tag) tuples
    """
    # Simple tokenization (split by spaces and punctuation)
    tokens = []
    current_token = ""
    
    for char in text:
        if char in ' \n\t.,;:!?()[]{}\"\'':
            if current_token:
                tokens.append(current_token)
                current_token = ""
            if char.strip():  # Keep punctuation as tokens
                tokens.append(char)
        else:
            current_token += char
    
    if current_token:
        tokens.append(current_token)
    
    # Create character to token mapping
    char_to_token = []
    char_index = 0
    
    for token_idx, token in enumerate(tokens):
        token_start = text.find(token, char_index)
        token_end = token_start + len(token)
        
        for i in range(token_start, token_end):
            char_to_token.append(token_idx)
        
        char_index = token_end
    
    # Initialize all tokens as 'O' (Outside)
    bio_tags = ['O'] * len(tokens)
    
    # Assign BIO tags based on entities
    for entity in entities:
        start_char = entity['start']
        end_char = entity['end']
        label = entity['label']
        
        if start_char < len(char_to_token) and end_char <= len(text):
            # Find token indices
            start_token = char_to_token[start_char] if start_char < len(char_to_token) else None
            end_token = char_to_token[end_char - 1] if end_char - 1 < len(char_to_token) else None
            
            if start_token is not None and end_token is not None:
                # Tag first token as B- (Beginning)
                bio_tags[start_token] = f'B-{label}'
                
                # Tag remaining tokens as I- (Inside)
                for token_idx in range(start_token + 1, end_token + 1):
                    if token_idx < len(bio_tags):
                        bio_tags[token_idx] = f'I-{label}'
    
    return list(zip(tokens, bio_tags))


def save_jsonl(samples: List[Dict], filename: str):
    """Save samples in JSONL format (one JSON object per line)"""
    with open(filename, 'w', encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    print(f"✅ Saved {len(samples)} samples to {filename}")


def save_conll(samples: List[Dict], filename: str):
    """Save samples in CONLL/BIO format"""
    with open(filename, 'w', encoding='utf-8') as f:
        for sample in samples:
            bio_tokens = convert_to_bio(sample['text'], sample['entities'])
            
            for token, tag in bio_tokens:
                f.write(f"{token}\t{tag}\n")
            
            # Blank line between samples
            f.write("\n")
    
    print(f"✅ Saved {len(samples)} samples to {filename} (CONLL format)")


def save_csv(samples: List[Dict], filename: str):
    """Save samples in CSV format for inspection"""
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'language', 'entities_json', 'num_entities'])
        
        for sample in samples:
            writer.writerow([
                sample['text'],
                sample['language'],
                json.dumps(sample['entities'], ensure_ascii=False),
                len(sample['entities'])
            ])
    
    print(f"✅ Saved {len(samples)} samples to {filename}")


def generate_dataset(
    num_samples: int = 1000,
    train_ratio: float = 0.8,
    output_dir: str = 'ner_dataset'
):
    """
    Generate complete NER dataset for ADDRESS and SEX detection
    
    Args:
        num_samples: Total number of samples to generate
        train_ratio: Ratio of training samples (rest will be validation)
        output_dir: Directory to save output files
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Generating {num_samples} samples...")
    print(f"   Train: {int(num_samples * train_ratio)}")
    print(f"   Val: {int(num_samples * (1 - train_ratio))}")
    
    # Generate samples with language variety
    all_samples = []
    languages = ['es', 'en', 'pt']
    
    for i in range(num_samples):
        language = random.choice(languages)
        sample = generate_sample(language)
        all_samples.append(sample)
        
        if (i + 1) % 100 == 0:
            print(f"   Generated {i + 1}/{num_samples} samples...")
    
    # Shuffle samples
    random.shuffle(all_samples)
    
    # Split train/val
    split_idx = int(num_samples * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print("\n💾 Saving datasets...")
    
    # Save in multiple formats
    # JSONL format
    save_jsonl(train_samples, f'{output_dir}/train.jsonl')
    save_jsonl(val_samples, f'{output_dir}/val.jsonl')
    
    # CONLL/BIO format
    save_conll(train_samples, f'{output_dir}/train.conll')
    save_conll(val_samples, f'{output_dir}/val.conll')
    
    # CSV format for inspection
    save_csv(train_samples, f'{output_dir}/train.csv')
    save_csv(val_samples, f'{output_dir}/val.csv')
    
    # Save statistics
    stats = {
        'total_samples': num_samples,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'languages': {
            'es': sum(1 for s in all_samples if s['language'] == 'es'),
            'en': sum(1 for s in all_samples if s['language'] == 'en'),
            'pt': sum(1 for s in all_samples if s['language'] == 'pt')
        },
        'entity_types': ['ADDRESS', 'SEX', 'NAME'],
        'formats': ['JSONL', 'CONLL/BIO', 'CSV']
    }
    
    with open(f'{output_dir}/dataset_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✨ Dataset generation complete!")
    print(f"📊 Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Train: {stats['train_samples']}")
    print(f"   Val: {stats['val_samples']}")
    print(f"   Languages: ES={stats['languages']['es']}, EN={stats['languages']['en']}, PT={stats['languages']['pt']}")
    print(f"   Entity types: {', '.join(stats['entity_types'])}")
    print(f"\n📁 Files saved in: {output_dir}/")
    print(f"   - train.jsonl, val.jsonl (for transformers)")
    print(f"   - train.conll, val.conll (BIO format)")
    print(f"   - train.csv, val.csv (for inspection)")
    print(f"   - dataset_stats.json (metadata)")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate NER training dataset for ADDRESS and SEX/GENDER PII detection'
    )
    parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=1000,
        help='Total number of samples to generate (default: 1000)'
    )
    parser.add_argument(
        '--train-ratio', '-r',
        type=float,
        default=0.8,
        help='Ratio of training samples (default: 0.8)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='ner_dataset',
        help='Output directory for dataset files (default: ner_dataset)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    Faker.seed(args.seed)
    
    # Generate dataset
    generate_dataset(
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        output_dir=args.output_dir
    )
    
    print("\n🎯 Next steps:")
    print("   1. Inspect the CSV files to verify data quality")
    print("   2. Use JSONL files for transformer fine-tuning")
    print("   3. Use CONLL files for spaCy or other NER frameworks")
    print("\n💡 Example usage:")
    print("   python generate_ner_dataset_address_sex.py -n 5000 -o my_dataset")
    print("   python generate_ner_dataset_address_sex.py --num-samples 2000 --train-ratio 0.85")


if __name__ == '__main__':
    main()

