#!/usr/bin/env python3
"""
Test integration of OCR noise in the main generation pipeline (without spaCy dependencies)
"""
import sys
import random

# Copy relevant functions for testing
def add_enhanced_ocr_noise(text: str, noise_probability: float = 0.25) -> str:
    """
    Add enhanced OCR-specific noise to simulate realistic OCR processing errors.
    """
    if random.random() > noise_probability:
        return text  # No OCR noise applied
    
    def _add_ocr_character_substitution_noise(text: str) -> str:
        ocr_substitutions = {
            '0': 'O', 'O': '0', '1': 'l', 'l': '1', 'I': '1',
            '6': 'G', '5': 'S', 'rn': 'm', 'cl': 'd', 'nn': 'n',
            'B': '8', '8': 'B', 'Z': '2', '2': 'Z', 'u': 'o', 'a': 'e',
        }
        
        if random.random() < 0.2:
            for original, substitute in ocr_substitutions.items():
                if original in text and random.random() < 0.3:
                    text = text.replace(original, substitute, 1)
                    break
        return text

    def _add_ocr_insertion_noise(text: str) -> str:
        insertion_chars = ['m', 'n', 'w', 'u', 'o', 'i', 'l', '1', '|', '.', ',', '-']
        
        if random.random() < 0.15:
            words = text.split()
            if words:
                word_idx = random.randint(0, len(words) - 1)
                word = words[word_idx]
                
                if len(word) > 3:
                    insert_pos = random.randint(1, len(word) - 1)
                    insert_char = random.choice(insertion_chars)
                    words[word_idx] = word[:insert_pos] + insert_char + word[insert_pos:]
            
            text = " ".join(words)
        return text

    def _add_ocr_spacing_artifacts_noise(text: str) -> str:
        if random.random() < 0.2:
            spacing_patterns = [
                lambda t: t.replace(" ", "  "),     # Double spaces
                lambda t: t.replace(" ", "   "),    # Triple spaces 
                lambda t: t.replace(".", " . "),    # Space around periods
                lambda t: t.replace(",", " , "),    # Space around commas
                lambda t: t.replace(":", " : "),    # Space around colons
                lambda t: t.replace("-", " - "),    # Space around hyphens
            ]
            pattern = random.choice(spacing_patterns)
            text = pattern(text)
        return text
    
    # OCR-specific noise functions with higher probability of application
    ocr_noise_types = [
        _add_ocr_character_substitution_noise,
        _add_ocr_spacing_artifacts_noise,
        _add_ocr_insertion_noise,
    ]
    
    # Apply 1-2 types of OCR noise for more realistic artifacts
    num_noise_types = random.choices([1, 2], weights=[0.7, 0.3])[0]
    selected_noise_types = random.sample(ocr_noise_types, min(num_noise_types, len(ocr_noise_types)))
    
    for noise_function in selected_noise_types:
        text = noise_function(text)
    
    return text

def test_integration():
    """Test the integration of OCR noise with sample customer data"""
    print("Testing OCR Noise Integration")
    print("=" * 60)
    
    # Sample customer data similar to what the main system generates
    test_sentences = [
        "El cliente JUAN CARLOS GARCÍA RODRÍGUEZ con RUT 15.234.567-8 registrado en Avenida Libertador 1234, Santiago. Teléfono: +56 9 8765 4321. Email: juan.garcia@gmail.com. Monto: $1.234.567 CLP. Operación: REF-10001.",
        
        "Usuario MARÍA JOSÉ HERNÁNDEZ SILVA identificada con RUT 18.765.432-1 domiciliada en Calle Providencia 567, Valparaíso. Tel: +56 2 2345 6789. Correo: maria.hernandez@hotmail.com. Saldo: $2.345.678 CLP. Código: TRX-10002.",
        
        "Cliente PEDRO ANTONIO LÓPEZ MIRANDA RUT 12.345.678-9 dirección registrada en Pasaje Los Álamos 89, Concepción. Contacto: +56 9 1234 5678. Email: pedro.lopez@outlook.cl. Transacción: $789.012 CLP. Referencia: OP-10003.",
        
        "Paciente CARMEN ROSA GONZÁLEZ TORRES cédula 20.987.654-3 residente en Avenida Brasil 345, La Serena. Teléfono: +56 51 234 567. Email: carmen.gonzalez@yahoo.cl. Copago: $156.789 CLP. Atención: ATN-10004.",
        
        "Beneficiario LUIS FERNANDO MORALES CASTRO con RUT 16.543.210-7 ubicado en Calle San Martín 123, Temuco. Fono: +56 45 789 012. Correo: luis.morales@gmail.cl. Monto: $3.456.789 CLP. Serie: SER-10005."
    ]
    
    print("\nTesting Normal vs OCR-Enhanced Noise:")
    print("-" * 60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nExample {i}:")
        print(f"Original:    {sentence}")
        
        # Test with enhanced OCR noise (high probability for demo)
        ocr_noisy = add_enhanced_ocr_noise(sentence, 1.0)  # 100% probability for demo
        print(f"OCR Noisy:   {ocr_noisy}")
        
        if sentence != ocr_noisy:
            print(f"✓ OCR noise successfully applied")
        else:
            print(f"- No OCR noise applied this time")
        
        print()

def test_realistic_ocr_levels():
    """Test different OCR noise levels"""
    print("\n" + "=" * 60)
    print("Testing Different OCR Noise Levels")
    print("=" * 60)
    
    sample_text = "El cliente JUAN GARCÍA con RUT 12.345.678-9 registrado en Avenida O'Higgins 123, Santiago. Teléfono: +56 9 8765 4321."
    
    noise_levels = [0.1, 0.25, 0.5, 0.75]
    
    for level in noise_levels:
        print(f"\nNoise Level: {level}")
        print(f"Original:  {sample_text}")
        
        # Apply OCR noise multiple times to show variation
        for attempt in range(3):
            noisy = add_enhanced_ocr_noise(sample_text, level)
            if noisy != sample_text:
                print(f"Attempt {attempt + 1}: {noisy}")
            else:
                print(f"Attempt {attempt + 1}: (no noise applied)")

if __name__ == "__main__":
    test_integration()
    test_realistic_ocr_levels()