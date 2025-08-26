#!/usr/bin/env python3
"""
Test script for OCR noise functionality
"""
import sys
import random

# Test the basic OCR noise functions without dependencies
def _add_ocr_character_substitution_noise(text: str) -> str:
    """Add realistic OCR character substitution errors."""
    # Common OCR character substitutions
    ocr_substitutions = {
        '0': 'O',  # Zero to letter O
        'O': '0',  # Letter O to zero
        '1': 'l',  # One to lowercase L
        'l': '1',  # Lowercase L to one
        'I': '1',  # Uppercase I to one
        '6': 'G',  # Six to G
        '5': 'S',  # Five to S
        'rn': 'm', # rn combination to m
        'cl': 'd', # cl combination to d
        'nn': 'n', # double n to single n
        'B': '8',  # B to 8
        '8': 'B',  # 8 to B
        'Z': '2',  # Z to 2
        '2': 'Z',  # 2 to Z
        'u': 'o',  # u to o confusion
        'a': 'e',  # a to e confusion (slight)
    }
    
    # Apply substitution with low probability to preserve readability
    if random.random() < 0.8:  # High probability for testing
        for original, substitute in ocr_substitutions.items():
            if original in text and random.random() < 0.5:
                # Only substitute first occurrence to avoid over-noise
                text = text.replace(original, substitute, 1)
                break
    
    return text

def _add_ocr_insertion_noise(text: str) -> str:
    """Add realistic OCR character insertion errors."""
    # Common OCR insertion characters (often artifacts from scanning)
    insertion_chars = ['m', 'n', 'w', 'u', 'o', 'i', 'l', '1', '|', '.', ',', '-']
    
    if random.random() < 0.8:  # High probability for testing
        words = text.split()
        if words:
            # Choose a random word to add insertion to
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx]
            
            if len(word) > 3:  # Only add noise to longer words
                # Insert random character at random position (not at start/end to preserve entity boundaries)
                insert_pos = random.randint(1, len(word) - 1)
                insert_char = random.choice(insertion_chars)
                words[word_idx] = word[:insert_pos] + insert_char + word[insert_pos:]
        
        text = " ".join(words)
    
    return text

def _add_ocr_spacing_artifacts_noise(text: str) -> str:
    """Add OCR-specific spacing artifacts common in scanned documents."""
    if random.random() < 0.8:  # High probability for testing
        spacing_patterns = [
            lambda t: t.replace(" ", "  "),     # Double spaces
            lambda t: t.replace(" ", "   "),    # Triple spaces (OCR artifact)
            lambda t: t.replace(".", " . "),    # Space around periods
            lambda t: t.replace(",", " , "),    # Space around commas
            lambda t: t.replace(":", " : "),    # Space around colons
            lambda t: t.replace("-", " - "),    # Space around hyphens
        ]
        pattern = random.choice(spacing_patterns)
        text = pattern(text)
    
    return text

def test_ocr_noise():
    """Test OCR noise functions with sample text"""
    print("Testing OCR Noise Functions")
    print("=" * 50)
    
    # Sample text similar to the examples in the issue
    test_texts = [
        "El cliente JUAN GARCIA con RUT 12345678-9 registrado en Avenida Libertador 123, Santiago.",
        "Traspaso de Cierre Casos cerrados de Scomp + Numero Cotizacion",
        "JUAN SEGUNDO GARCIA CARRASCO RUT 7,082,003-K",
        "Cotizacion de Seguro de Renta Vitalicia",
        "VALOR UF A FECHA COTIZACION 24.308.17"
    ]
    
    print("\n1. Testing OCR Character Substitution Noise:")
    for i, text in enumerate(test_texts, 1):
        original = text
        noisy = _add_ocr_character_substitution_noise(text)
        print(f"   {i}. Original: {original}")
        print(f"      Noisy:    {noisy}")
        if original != noisy:
            print(f"      ✓ Applied character substitution noise")
        else:
            print(f"      - No noise applied this time")
        print()
    
    print("\n2. Testing OCR Insertion Noise:")
    for i, text in enumerate(test_texts, 1):
        original = text
        noisy = _add_ocr_insertion_noise(text)
        print(f"   {i}. Original: {original}")
        print(f"      Noisy:    {noisy}")
        if original != noisy:
            print(f"      ✓ Applied insertion noise")
        else:
            print(f"      - No noise applied this time")
        print()
    
    print("\n3. Testing OCR Spacing Artifacts Noise:")
    for i, text in enumerate(test_texts, 1):
        original = text
        noisy = _add_ocr_spacing_artifacts_noise(text)
        print(f"   {i}. Original: {original}")
        print(f"      Noisy:    {noisy}")
        if original != noisy:
            print(f"      ✓ Applied spacing artifacts noise")
        else:
            print(f"      - No noise applied this time")
        print()
    
    print("\n4. Testing Combined OCR Noise (multiple functions):")
    for i, text in enumerate(test_texts, 1):
        original = text
        # Apply multiple noise functions
        noisy = _add_ocr_character_substitution_noise(text)
        noisy = _add_ocr_insertion_noise(noisy)
        noisy = _add_ocr_spacing_artifacts_noise(noisy)
        
        print(f"   {i}. Original: {original}")
        print(f"      Noisy:    {noisy}")
        if original != noisy:
            print(f"      ✓ Applied combined OCR noise")
        else:
            print(f"      - No noise applied this time")
        print()

if __name__ == "__main__":
    test_ocr_noise()