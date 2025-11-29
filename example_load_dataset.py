#!/usr/bin/env python3
"""
Example script demonstrating how to load and use the generated NER dataset
"""

import json
import csv
from collections import Counter
from typing import List, Dict


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_conll(filepath: str) -> List[List[tuple]]:
    """Load CONLL/BIO format file"""
    sentences = []
    current_sentence = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                token, tag = line.split('\t')
                current_sentence.append((token, tag))
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


def analyze_dataset(data: List[Dict]):
    """Analyze and print statistics about the dataset"""
    print("\n" + "="*60)
    print("📊 DATASET ANALYSIS")
    print("="*60)
    
    # Basic stats
    print(f"\n📈 Total Samples: {len(data)}")
    
    # Language distribution
    languages = Counter(sample['language'] for sample in data)
    print(f"\n🌍 Language Distribution:")
    for lang, count in languages.most_common():
        percentage = (count / len(data)) * 100
        print(f"   {lang.upper()}: {count} ({percentage:.1f}%)")
    
    # Entity type distribution
    entity_labels = []
    for sample in data:
        for entity in sample['entities']:
            entity_labels.append(entity['label'])
    
    label_counts = Counter(entity_labels)
    print(f"\n🏷️  Entity Type Distribution:")
    for label, count in label_counts.most_common():
        percentage = (count / len(entity_labels)) * 100
        print(f"   {label}: {count} ({percentage:.1f}%)")
    
    # Entities per sample
    entities_per_sample = [len(sample['entities']) for sample in data]
    avg_entities = sum(entities_per_sample) / len(entities_per_sample)
    print(f"\n📊 Entities per Sample:")
    print(f"   Average: {avg_entities:.2f}")
    print(f"   Min: {min(entities_per_sample)}")
    print(f"   Max: {max(entities_per_sample)}")
    
    # Text length distribution
    text_lengths = [len(sample['text']) for sample in data]
    avg_length = sum(text_lengths) / len(text_lengths)
    print(f"\n📏 Text Length (characters):")
    print(f"   Average: {avg_length:.1f}")
    print(f"   Min: {min(text_lengths)}")
    print(f"   Max: {max(text_lengths)}")


def show_examples(data: List[Dict], num_examples: int = 5):
    """Show example samples"""
    print("\n" + "="*60)
    print("📝 EXAMPLE SAMPLES")
    print("="*60)
    
    for i, sample in enumerate(data[:num_examples], 1):
        print(f"\n--- Example {i} ---")
        print(f"Language: {sample['language'].upper()}")
        print(f"Text: {sample['text']}")
        print(f"Entities ({len(sample['entities'])}):")
        for entity in sample['entities']:
            print(f"   [{entity['label']}] \"{entity['text']}\" (pos: {entity['start']}-{entity['end']})")


def show_bio_examples(sentences: List[List[tuple]], num_examples: int = 3):
    """Show example BIO-tagged sentences"""
    print("\n" + "="*60)
    print("🏷️  BIO FORMAT EXAMPLES")
    print("="*60)
    
    for i, sentence in enumerate(sentences[:num_examples], 1):
        print(f"\n--- Sentence {i} ---")
        for token, tag in sentence:
            print(f"{token:20s} {tag}")


def extract_entities_by_type(data: List[Dict], entity_type: str) -> List[str]:
    """Extract all entities of a specific type"""
    entities = set()
    for sample in data:
        for entity in sample['entities']:
            if entity['label'] == entity_type:
                entities.add(entity['text'])
    return sorted(list(entities))


def demo_transformer_format(data: List[Dict]):
    """Demonstrate how data looks for transformer training"""
    print("\n" + "="*60)
    print("🤖 TRANSFORMER FORMAT PREVIEW")
    print("="*60)
    
    sample = data[0]
    print(f"\nOriginal format (JSONL):")
    print(json.dumps(sample, indent=2, ensure_ascii=False))
    
    print(f"\n\nFor transformer training, you would:")
    print("1. Tokenize the text (e.g., using BertTokenizer)")
    print("2. Align entities with subword tokens")
    print("3. Convert labels to label IDs")
    print("4. Create attention masks and other inputs")
    
    print(f"\nExample pseudo-code:")
    print("""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Tokenize
    tokens = tokenizer.tokenize(sample['text'])
    
    # Align entities to tokens
    labels = align_labels_with_tokens(tokens, sample['entities'])
    
    # Encode
    inputs = tokenizer.encode_plus(
        sample['text'],
        return_tensors='pt',
        padding='max_length',
        max_length=128
    )
    """)


def main():
    """Main demo function"""
    import os
    
    dataset_dir = 'ner_dataset'
    
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        print(f"\n💡 First, generate the dataset by running:")
        print(f"   python generate_ner_dataset_address_sex.py")
        return
    
    print("="*60)
    print("🎯 NER DATASET LOADER - DEMO")
    print("="*60)
    
    # Load JSONL data
    print(f"\n📂 Loading JSONL data...")
    train_data = load_jsonl(f'{dataset_dir}/train.jsonl')
    val_data = load_jsonl(f'{dataset_dir}/val.jsonl')
    print(f"✅ Loaded {len(train_data)} training samples")
    print(f"✅ Loaded {len(val_data)} validation samples")
    
    # Analyze training data
    analyze_dataset(train_data)
    
    # Show examples
    show_examples(train_data, num_examples=3)
    
    # Load CONLL data
    print(f"\n📂 Loading CONLL/BIO data...")
    train_bio = load_conll(f'{dataset_dir}/train.conll')
    print(f"✅ Loaded {len(train_bio)} BIO-tagged sentences")
    
    # Show BIO examples
    show_bio_examples(train_bio, num_examples=2)
    
    # Extract specific entity types
    print("\n" + "="*60)
    print("🔍 ENTITY EXTRACTION")
    print("="*60)
    
    addresses = extract_entities_by_type(train_data, 'ADDRESS')
    print(f"\n📍 Sample Addresses ({min(5, len(addresses))} of {len(addresses)}):")
    for addr in addresses[:5]:
        print(f"   • {addr}")
    
    sexes = extract_entities_by_type(train_data, 'SEX')
    print(f"\n⚧️  Unique Sex/Gender Terms ({len(sexes)}):")
    for sex in sexes:
        print(f"   • {sex}")
    
    # Show transformer format
    demo_transformer_format(train_data)
    
    # Load stats
    print("\n" + "="*60)
    print("📋 DATASET STATISTICS")
    print("="*60)
    
    with open(f'{dataset_dir}/dataset_stats.json', 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE!")
    print("="*60)
    print("\n💡 Next steps:")
    print("   1. Use train_data for training your NER model")
    print("   2. Use val_data for validation and evaluation")
    print("   3. Check USAGE_SIMPLE_NER.md for training examples")


if __name__ == '__main__':
    main()

