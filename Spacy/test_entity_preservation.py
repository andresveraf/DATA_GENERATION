#!/usr/bin/env python3
"""
Entity-Aware OCR Testing
Test how well our OCR noise preserves entity boundaries for NER training.
"""

import re
from data_generation_noisy import (_add_ocr_character_noise, _add_ocr_scanning_noise, 
                                   _add_severe_ocr_corruption, _add_ocr_symbol_corruption)
import random

def identify_entities(text):
    """Identify PII entities using regex patterns."""
    
    entities = []
    
    patterns = {
        'ID_NUMBER': [
            r'\b\d{1,2}[.,]?\d{3}[.,]?\d{3}[-]\d{1}[kK]?\b',  # Chilean RUT
            r'\b\d{3}[.,]?\d{3}[.,]?\d{3}[-]\d{2}\b',         # Brazilian CPF
            r'\b\d{7,8}[-]?\d{1}[kK]?\b',                     # Short format
        ],
        'PHONE': [
            r'\+56\s*[29]\s*\d{4}\s*\d{4}',                  # Chilean
            r'\+\d{1,3}\s*\d{1,3}\s*\d{4}\s*\d{4}',         # International
        ],
        'EMAIL': [
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
        ],
        'DATE': [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
        ],
        'MONEY': [
            r'\$[\d.,]+\s*(CLP|USD|EUR)',
        ]
    }
    
    for entity_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'text': match.group(),
                    'label': entity_type,
                    'start': match.start(),
                    'end': match.end()
                })
    
    entities.sort(key=lambda x: x['start'])
    return entities

def apply_noise(text, intensity='medium'):
    """Apply OCR noise with different intensities."""
    
    levels = {
        'light': (1, 0.4),
        'medium': (2, 0.6), 
        'heavy': (3, 0.8)
    }
    
    iterations, prob = levels.get(intensity, levels['medium'])
    result = text
    
    for i in range(iterations):
        if random.random() < prob:
            result = _add_ocr_character_noise(result)
        if random.random() < prob * 0.8:
            result = _add_severe_ocr_corruption(result)
        if random.random() < prob * 0.5:
            result = _add_ocr_symbol_corruption(result)
    
    return result

def calculate_preservation_rate(original_entities, noisy_entities):
    """Calculate how well entities are preserved after noise."""
    
    if not original_entities:
        return 100.0
    
    preserved = 0
    for orig in original_entities:
        for noisy in noisy_entities:
            # Check if entity type matches and position is close
            if (orig['label'] == noisy['label'] and 
                abs(orig['start'] - noisy['start']) <= 3):
                preserved += 1
                break
    
    return (preserved / len(original_entities)) * 100

def test_entity_preservation():
    """Test entity preservation across different noise levels."""
    
    test_cases = [
        "JUAN SEGUNDO GARCIA CARRASCO con RUT 7,082,003-K",
        "Cliente: MARÍA GONZÁLEZ, RUT 12.345.678-9, Tel: +56 9 8765 4321",
        "Email: maria.gonzalez@empresa.cl, Fecha: 15/08/2024",
        "Monto: $1.234.567 CLP, Código: ABC123, Tel: +56 2 1234 5678",
        "ROSA ESTER PENA GONZALEZ con cédula 6.672.485-9 email rosa@test.com"
    ]
    
    print("=== ENTITY PRESERVATION TESTING ===")
    print()
    
    for i, text in enumerate(test_cases, 1):
        print(f"TEST CASE {i}:")
        print(f"Original: {text}")
        
        # Detect original entities
        original_entities = identify_entities(text)
        print(f"Entities ({len(original_entities)}):", end=" ")
        for entity in original_entities:
            print(f"{entity['label']}:{entity['text']}", end=" ")
        print()
        
        # Test different noise levels
        for intensity in ['light', 'medium', 'heavy']:
            noisy_text = apply_noise(text, intensity)
            noisy_entities = identify_entities(noisy_text)
            preservation = calculate_preservation_rate(original_entities, noisy_entities)
            
            status = "✅" if preservation >= 90 else "⚠️" if preservation >= 70 else "❌"
            print(f"  {intensity.upper()}: {preservation:5.1f}% {status} {noisy_text}")
        
        print()

def test_your_ocr_example():
    """Test with your specific OCR document example."""
    
    print("=== TESTING YOUR PENSION DOCUMENT EXAMPLE ===")
    print()
    
    # Clean version
    clean_text = "JUAN SEGUNDO GARCIA CARRASCO RUT 7,082,003-K ELIAS 42 VILLA HERMOSA CORONEL 28/09/1952 ROSA ESTER PENA GONZALEZ 66724859 27-11-1953"
    
    print("CLEAN TEXT:")
    print(clean_text)
    
    entities = identify_entities(clean_text)
    print(f"\nENTITIES DETECTED ({len(entities)}):")
    for i, entity in enumerate(entities, 1):
        print(f"  {i}. {entity['label']}: '{entity['text']}'")
    
    print("\nTESTING NOISE LEVELS:")
    
    for intensity in ['light', 'medium', 'heavy']:
        print(f"\n{intensity.upper()} NOISE:")
        noisy = apply_noise(clean_text, intensity)
        print(noisy)
        
        noisy_entities = identify_entities(noisy)
        preservation = calculate_preservation_rate(entities, noisy_entities)
        
        status = "✅" if preservation >= 90 else "⚠️" if preservation >= 70 else "❌"
        print(f"Entity preservation: {preservation:.1f}% {status}")
        
        if noisy_entities != entities:
            print("Entities after noise:")
            for entity in noisy_entities:
                print(f"  {entity['label']}: '{entity['text']}'")

def main():
    """Main testing function."""
    
    print("ENTITY-AWARE OCR NOISE TESTING")
    print("=" * 50)
    print()
    
    # Test entity preservation
    test_entity_preservation()
    
    # Test your specific example
    test_your_ocr_example()
    
    print("=" * 50)
    print("SUMMARY:")
    print("- ✅ 90%+ preservation = Excellent (good for NER training)")
    print("- ⚠️  70-89% preservation = Acceptable (may need adjustment)")  
    print("- ❌ <70% preservation = Poor (too much noise)")
    print()
    print("This helps validate that OCR noise doesn't break entity")
    print("boundaries needed for spaCy NER model training!")

if __name__ == "__main__":
    main()
