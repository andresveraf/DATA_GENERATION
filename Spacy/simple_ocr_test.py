#!/usr/bin/env python3
"""
Simple OCR Noise Tester with Entity Detection
Copy and paste any text to see it with OCR noise applied and check entity preservation.
Usage: python3 simple_ocr_test.py
"""

import sys
import re
from data_generation_noisy import (_add_ocr_character_noise, _add_ocr_scanning_noise, 
                                   _add_severe_ocr_corruption, _add_ocr_symbol_corruption)
import random

def identify_pii_entities(text):
    """Identify PII entities using regex patterns."""
    
    entities = []
    
    # Regex patterns for common PII
    patterns = {
        'ID_NUMBER': [
            r'\b\d{1,2}[.,]?\d{3}[.,]?\d{3}[-]\d{1}[kK]?\b',  # Chilean RUT
            r'\b\d{3}[.,]?\d{3}[.,]?\d{3}[-]\d{2}\b',         # Brazilian CPF
            r'\b\d{1,2}[.,]?\d{3}[.,]?\d{3}[-]\d{1}\b',       # Uruguayan Cédula
            r'\b\d{7,8}[-]?\d{1}[kK]?\b',                     # Short format IDs
        ],
        'PHONE': [
            r'\+56\s*[29]\s*\d{4}\s*\d{4}',                  # Chilean phones
            r'\+\d{1,3}\s*\d{1,3}\s*\d{4}\s*\d{4}',         # International
            r'\(\d{3}\)\s*\d{3}[-\s]\d{4}',                 # US format
        ],
        'EMAIL': [
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
        ],
        'DATE': [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        ],
        'MONEY': [
            r'\$[\d.,]+\s*(CLP|USD|EUR|PEN|UYU|BRL)',
            r'[\d.,]+\s*(pesos|dólares|euros)',
        ]
    }
    
    # Find regex-based entities
    for entity_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'text': match.group(),
                    'label': entity_type,
                    'start': match.start(),
                    'end': match.end()
                })
    
    # Sort by start position
    entities.sort(key=lambda x: x['start'])
    
    return entities

def apply_ocr_noise(text, iterations=2):
    """Apply OCR noise to text with specified iterations."""
    result = text
    
    for i in range(iterations):
        # Apply each noise function with some probability
        if random.random() < 0.7:
            result = _add_ocr_character_noise(result)
        if random.random() < 0.6:
            result = _add_ocr_scanning_noise(result)
        if random.random() < 0.8:
            result = _add_severe_ocr_corruption(result)
        if random.random() < 0.5:
            result = _add_ocr_symbol_corruption(result)
    
    return result

def main():
    """Main function for testing."""
    
    if len(sys.argv) > 1:
        # Use command line arguments
        text = " ".join(sys.argv[1:])
    else:
        # Interactive input
        print("=== SIMPLE OCR NOISE TESTER WITH ENTITY DETECTION ===")
        print("Paste your text and press Enter:")
        print()
        
        try:
            text = input().strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            return
        
        if not text:
            print("No text provided.")
            return
    
    print("\n" + "="*60)
    print("ORIGINAL TEXT:")
    print("="*60)
    print(text)
    
    # Show entities in original text
    original_entities = identify_pii_entities(text)
    print(f"\nENTITIES DETECTED IN ORIGINAL ({len(original_entities)}):")
    if original_entities:
        for i, entity in enumerate(original_entities, 1):
            print(f"  {i}. {entity['label']}: '{entity['text']}'")
    else:
        print("  No PII entities detected")
    
    print("\n" + "="*60)
    print("WITH OCR NOISE (3 variations):")
    print("="*60)
    
    # Show 3 different variations
    for i in range(3):
        print(f"\nVARIATION {i+1}:")
        noisy = apply_ocr_noise(text, 2)
        print(noisy)
        
        # Check entity preservation
        noisy_entities = identify_pii_entities(noisy)
        print(f"Entities after noise ({len(noisy_entities)}):", end=" ")
        
        if len(noisy_entities) == len(original_entities):
            print("✅ All entities preserved")
        elif len(noisy_entities) < len(original_entities):
            lost = len(original_entities) - len(noisy_entities)
            print(f"⚠️  {lost} entity(ies) lost")
        else:
            extra = len(noisy_entities) - len(original_entities)
            print(f"⚠️  {extra} extra entity(ies) detected")
        
        # Show detected entities if different
        if len(noisy_entities) != len(original_entities) or any(
            ne['text'] != oe['text'] for ne, oe in zip(noisy_entities, original_entities)
        ):
            for j, entity in enumerate(noisy_entities, 1):
                print(f"    {j}. {entity['label']}: '{entity['text']}'")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
