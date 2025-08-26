#!/usr/bin/env python3
"""
OCR Comparison and Testing Script
Tests our noise generation against real OCR examples to validate realism.
Includes entity identification to validate PII detection after noise application.
"""

import re
import spacy
from data_generation_noisy import (
    _add_ocr_character_noise,
    _add_ocr_scanning_noise, 
    _add_ocr_line_break_noise,
    _add_severe_ocr_corruption,
    _add_ocr_symbol_corruption
)

# Load spaCy model for entity recognition
try:
    nlp = spacy.load("es_core_news_lg")
    NLP_AVAILABLE = True
except IOError:
    print("Warning: spaCy Spanish model not available. Entity detection disabled.")
    NLP_AVAILABLE = False

def identify_pii_entities(text):
    """Identify PII entities in text using both spaCy and regex patterns."""
    
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
                    'end': match.end(),
                    'source': 'regex'
                })
    
    # Use spaCy if available for additional entity detection
    if NLP_AVAILABLE:
        doc = nlp(text)
        for ent in doc.ents:
            # Map spaCy labels to our PII labels
            label_mapping = {
                'PER': 'PERSON',
                'PERSON': 'PERSON',
                'LOC': 'LOCATION',
                'ORG': 'ORGANIZATION',
                'DATE': 'DATE',
                'TIME': 'TIME',
                'MONEY': 'MONEY'
            }
            
            pii_label = label_mapping.get(ent.label_, ent.label_)
            
            entities.append({
                'text': ent.text,
                'label': pii_label,
                'start': ent.start_char,
                'end': ent.end_char,
                'source': 'spacy',
                'confidence': getattr(ent, 'confidence', 1.0)
            })
    
    # Remove duplicates (prefer regex matches for PII-specific patterns)
    unique_entities = []
    for entity in entities:
        # Check if this entity overlaps with existing ones
        overlap = False
        for existing in unique_entities:
            if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                # Overlapping entities - prefer regex for PII types
                if entity['source'] == 'regex' and entity['label'] in ['ID_NUMBER', 'PHONE', 'EMAIL']:
                    # Replace the existing entity with regex one
                    unique_entities.remove(existing)
                    break
                else:
                    overlap = True
                    break
        
        if not overlap:
            unique_entities.append(entity)
    
    # Sort by start position
    unique_entities.sort(key=lambda x: x['start'])
    
    return unique_entities

def compare_entity_detection(original_text, noisy_text):
    """Compare entity detection between original and noisy text."""
    
    print("=== ENTITY DETECTION COMPARISON ===")
    print()
    
    original_entities = identify_pii_entities(original_text)
    noisy_entities = identify_pii_entities(noisy_text)
    
    print("ORIGINAL TEXT ENTITIES:")
    if original_entities:
        for i, entity in enumerate(original_entities, 1):
            source_info = f" ({entity['source']})" if 'source' in entity else ""
            print(f"  {i}. {entity['label']}: '{entity['text']}'{source_info}")
    else:
        print("  No entities detected")
    
    print("\nNOISY TEXT ENTITIES:")
    if noisy_entities:
        for i, entity in enumerate(noisy_entities, 1):
            source_info = f" ({entity['source']})" if 'source' in entity else ""
            print(f"  {i}. {entity['label']}: '{entity['text']}'{source_info}")
    else:
        print("  No entities detected")
    
    # Analysis
    print(f"\nENTITY COUNT COMPARISON:")
    print(f"  Original: {len(original_entities)} entities")
    print(f"  Noisy: {len(noisy_entities)} entities")
    
    if len(original_entities) == len(noisy_entities):
        print("  ✅ Entity count preserved")
    elif len(noisy_entities) < len(original_entities):
        print("  ⚠️  Some entities lost due to noise")
    else:
        print("  ⚠️  Additional entities detected (possible false positives)")
    
    # Check for entity preservation
    preserved_entities = 0
    for orig_entity in original_entities:
        for noisy_entity in noisy_entities:
            if (orig_entity['label'] == noisy_entity['label'] and 
                abs(orig_entity['start'] - noisy_entity['start']) <= 5):  # Allow small position shifts
                preserved_entities += 1
                break
    
    if original_entities:
        preservation_rate = (preserved_entities / len(original_entities)) * 100
        print(f"  Entity preservation rate: {preservation_rate:.1f}%")
        
        if preservation_rate >= 90:
            print("  ✅ Excellent entity preservation")
        elif preservation_rate >= 70:
            print("  ⚠️  Good entity preservation")
        else:
            print("  ❌ Poor entity preservation - noise too aggressive")
    
    print("-" * 50)
    """Analyze the real OCR sample to understand corruption patterns."""
    
    real_ocr = """Preview X BORRADOR Cotización de Seguro de Renta Vitalicia mportante: No acepte ofrecimientos de dinero u otros incentivos para contratar su pensión.Esto estal N de Cotización externa: 6572492 prohibido por ley y perjudica a los pensionados.Denuncie ofrecimientos de este tipo a la Superintendenciall No Solicitud Oferta SCOMP: 54807101 de Valores y Seguros.Folio Oferta SCOMP: 69241814 ónN - 28/ . ; VALOR UF A FECHA COTIZACION 24.308.17 1.FECHA DE COTIZACION : 281072014 VALIDA HASTA: 12/11/2014 VALOR ........ AFECHA COTIZACIÓN oo 2.TIPO DE PENSIÓN 3.MODALIDAD DE PENSION [ Cláusula Alternativa 4.PERIODO GARANTIZADO Seguro de Vejez Anticipada Diferida a 24 Meses Art.6, póliza 180 Meses (Marcar dlo si comesponde) 6 Tasa de descuento 5.FECHA DE PAGO DE RENTA VITALICIA.B primer pago de Renta Vitalicia depende de la Techa de trasparo de la prima a la Cía.de seguros.En a10 de Renta Vitalicia Citerida erte pago 16 efectuara una vez trancurrido el ptazo convenido, contado de 1de el traiparo de la prima, _ 6.DATOS DEL AFILIADO 7.RENTA MENSUAL OFRECIDA Nombre : JUAN SEGUNDO GARCIA CARRASCO RUT - 7,082,003-K POR COMPAKIA DE SEGUROS AL Dirección : ELIAS 42 VILLA HERMOSA SWAGER AFILIADO. Comuna : CORONEL Ciudad : CORONEL Fija U.F. Fecha Nacimiento: 28/09/1952 Sexo : Masculino Estado Civil : Casado Ercedente Libre Cisposición AFP : CUPRUM Sistema Salud : FONASA F solleitado UF Máimo UF 8.DATOS DE LOS BENEFICIARIOS Y RENTA MENSUAL omecida porla Compañía de seguros alos beneTiclarios en aso de penston de sobrevivencia o de maulocimiento del amilado. - Nombre RUT Relación Sexo Invalidez _ Fecha Nacimiento Renta I 2 ROSA ESTER PENA GONZALEZ 66724859 CONYUGE SIN HIJOS FEMENINO NO 27-11-1953 5.44 9.CERTIFICADO DE SALDO emitido por la AFP al : 0871072014 con fecha de cierre al : 061072014"""
    
    print("=== ANALYZING REAL OCR SAMPLE ===")
    print()
    
    # Extract identifiable corruption patterns
    corruptions = [
        ("mportante" ,"Importante"),  # Missing 'I'
        ("estal", "está"),  # Character substitution
        ("Superintendenciall", "Superintendencia"),  # Extra 'l'
        ("comesponde", "corresponde"),  # 'rr' -> 'm'
        ("dlo", "solo"),  # 's' -> 'd'
        ("Techa", "fecha"),  # 'f' -> 'T'
        ("trasparo", "traspaso"),  # 's' -> 'r'
        ("a10", "algo"),  # Character corruption
        ("Citerida", "diferida"),  # Major corruption
        ("erte", "este"),  # Character substitution
        ("16", "se"),  # Number/letter confusion
        ("trancurrido", "transcurrido"),  # Missing 's'
        ("ptazo", "plazo"),  # 'l' -> 'pt'
        ("1de", "desde"),  # Severe corruption
        ("traiparo", "traspaso"),  # Multiple errors
        ("COMPAKIA", "COMPAÑÍA"),  # Ñ -> K + I
        ("SWAGER", "SEGURO"),  # Severe corruption
        ("Ercedente", "Excedente"),  # 'x' -> 'r'
        ("Cisposición", "Disposición"),  # 'D' -> 'C'
        ("solleitado", "solicitado"),  # 'c' -> 'e'
        ("Máimo", "Máximo"),  # 'x' -> 'i'
        ("omecida", "ofrecida"),  # 'fr' -> 'm'
        ("alos", "a los"),  # Missing space
        ("beneTiclarios", "beneficiarios"),  # Multiple errors
        ("aso", "caso"),  # 'c' -> 'a'
        ("penston", "pensión"),  # Multiple errors
        ("maulocimiento", "fallecimiento"),  # Severe corruption
        ("amilado", "afiliado"),  # 'f' -> 'm'
    ]
    
    print("IDENTIFIED CORRUPTION PATTERNS:")
    for corrupted, original in corruptions:
        print(f"  '{original}' → '{corrupted}'")
    
    print(f"\nTOTAL CORRUPTIONS FOUND: {len(corruptions)}")
    
def analyze_real_ocr_sample():
    """Analyze the real OCR sample to understand corruption patterns."""
    
    real_ocr = """Preview X BORRADOR Cotización de Seguro de Renta Vitalicia mportante: No acepte ofrecimientos de dinero u otros incentivos para contratar su pensión.Esto estal N de Cotización externa: 6572492 prohibido por ley y perjudica a los pensionados.Denuncie ofrecimientos de este tipo a la Superintendenciall No Solicitud Oferta SCOMP: 54807101 de Valores y Seguros.Folio Oferta SCOMP: 69241814 ónN - 28/ . ; VALOR UF A FECHA COTIZACION 24.308.17 1.FECHA DE COTIZACION : 281072014 VALIDA HASTA: 12/11/2014 VALOR ........ AFECHA COTIZACIÓN oo 2.TIPO DE PENSIÓN 3.MODALIDAD DE PENSION [ Cláusula Alternativa 4.PERIODO GARANTIZADO Seguro de Vejez Anticipada Diferida a 24 Meses Art.6, póliza 180 Meses (Marcar dlo si comesponde) 6 Tasa de descuento 5.FECHA DE PAGO DE RENTA VITALICIA.B primer pago de Renta Vitalicia depende de la Techa de trasparo de la prima a la Cía.de seguros.En a10 de Renta Vitalicia Citerida erte pago 16 efectuara una vez trancurrido el ptazo convenido, contado de 1de el traiparo de la prima, _ 6.DATOS DEL AFILIADO 7.RENTA MENSUAL OFRECIDA Nombre : JUAN SEGUNDO GARCIA CARRASCO RUT - 7,082,003-K POR COMPAKIA DE SEGUROS AL Dirección : ELIAS 42 VILLA HERMOSA SWAGER AFILIADO. Comuna : CORONEL Ciudad : CORONEL Fija U.F. Fecha Nacimiento: 28/09/1952 Sexo : Masculino Estado Civil : Casado Ercedente Libre Cisposición AFP : CUPRUM Sistema Salud : FONASA F solleitado UF Máimo UF 8.DATOS DE LOS BENEFICIARIOS Y RENTA MENSUAL omecida porla Compañía de seguros alos beneTiclarios en aso de penston de sobrevivencia o de maulocimiento del amilado. - Nombre RUT Relación Sexo Invalidez _ Fecha Nacimiento Renta I 2 ROSA ESTER PENA GONZALEZ 66724859 CONYUGE SIN HIJOS FEMENINO NO 27-11-1953 5.44 9.CERTIFICADO DE SALDO emitido por la AFP al : 0871072014 con fecha de cierre al : 061072014"""
    
    print("=== ANALYZING REAL OCR SAMPLE ===")
    print()
    
    # Extract identifiable corruption patterns
    corruptions = [
        ("mportante" ,"Importante"),  # Missing 'I'
        ("estal", "está"),  # Character substitution
        ("Superintendenciall", "Superintendencia"),  # Extra 'l'
        ("comesponde", "corresponde"),  # 'rr' -> 'm'
        ("dlo", "solo"),  # 's' -> 'd'
        ("Techa", "fecha"),  # 'f' -> 'T'
        ("trasparo", "traspaso"),  # 's' -> 'r'
        ("a10", "algo"),  # Character corruption
        ("Citerida", "diferida"),  # Major corruption
        ("erte", "este"),  # Character substitution
        ("16", "se"),  # Number/letter confusion
        ("trancurrido", "transcurrido"),  # Missing 's'
        ("ptazo", "plazo"),  # 'l' -> 'pt'
        ("1de", "desde"),  # Severe corruption
        ("traiparo", "traspaso"),  # Multiple errors
        ("COMPAKIA", "COMPAÑÍA"),  # Ñ -> K + I
        ("SWAGER", "SEGURO"),  # Severe corruption
        ("Ercedente", "Excedente"),  # 'x' -> 'r'
        ("Cisposición", "Disposición"),  # 'D' -> 'C'
        ("solleitado", "solicitado"),  # 'c' -> 'e'
        ("Máimo", "Máximo"),  # 'x' -> 'i'
        ("omecida", "ofrecida"),  # 'fr' -> 'm'
        ("alos", "a los"),  # Missing space
        ("beneTiclarios", "beneficiarios"),  # Multiple errors
        ("aso", "caso"),  # 'c' -> 'a'
        ("penston", "pensión"),  # Multiple errors
        ("maulocimiento", "fallecimiento"),  # Severe corruption
        ("amilado", "afiliado"),  # 'f' -> 'm'
    ]
    
    print("IDENTIFIED CORRUPTION PATTERNS:")
    for corrupted, original in corruptions:
        print(f"  '{original}' → '{corrupted}'")
    
    print(f"\nTOTAL CORRUPTIONS FOUND: {len(corruptions)}")
    
    # Identify and analyze entities in the OCR text
    print("\n=== ENTITY ANALYSIS IN REAL OCR ===")
    entities = identify_pii_entities(real_ocr)
    
    if entities:
        print("ENTITIES DETECTED IN OCR TEXT:")
        for i, entity in enumerate(entities, 1):
            source_info = f" ({entity['source']})" if 'source' in entity else ""
            print(f"  {i}. {entity['label']}: '{entity['text']}'{source_info}")
    else:
        print("No entities detected in OCR text")
    
    return real_ocr, corruptions, entities

def test_noise_generation_similarity():
    """Test our noise generation against real OCR patterns."""
    
    print("\n" + "="*60)
    print("TESTING OUR NOISE GENERATION WITH ENTITY DETECTION")
    print("="*60)
    
    # Clean example text similar to the pension document
    clean_text = """Cotización de Seguro de Renta Vitalicia. Importante: No acepte ofrecimientos de dinero para contratar su pensión. Número de Cotización externa: 6572492. Folio Oferta SCOMP: 54807101. FECHA DE COTIZACIÓN: 28/10/2014. DATOS DEL AFILIADO. Nombre: JUAN SEGUNDO GARCIA CARRASCO. RUT: 7,082,003-K. Dirección: ELIAS 42 VILLA HERMOSA. Comuna: CORONEL. Fecha Nacimiento: 28/09/1952. DATOS DE LOS BENEFICIARIOS. Nombre: ROSA ESTER PENA GONZALEZ. RUT: 6,672,485-9. Fecha Nacimiento: 27-11-1953."""
    
    print("ORIGINAL CLEAN TEXT:")
    print(clean_text)
    print()
    
    # Show entities in clean text first
    print("ENTITIES IN CLEAN TEXT:")
    clean_entities = identify_pii_entities(clean_text)
    if clean_entities:
        for i, entity in enumerate(clean_entities, 1):
            source_info = f" ({entity['source']})" if 'source' in entity else ""
            print(f"  {i}. {entity['label']}: '{entity['text']}'{source_info}")
    else:
        print("  No entities detected")
    
    print()
    
    # Apply different noise levels and test entity detection
    noise_levels = [(0.3, 1), (0.5, 1), (0.7, 2), (0.9, 2)]
    
    for prob, iterations in noise_levels:
        print(f"--- NOISE LEVEL {prob} (iterations: {iterations}) ---")
        noisy_text = clean_text
        
        # Apply noise functions with controlled probability
        import random
        for i in range(iterations):
            if random.random() < prob:
                noisy_text = _add_ocr_character_noise(noisy_text)
            if random.random() < prob:
                noisy_text = _add_ocr_scanning_noise(noisy_text)
            if random.random() < prob:
                noisy_text = _add_ocr_line_break_noise(noisy_text)
            if random.random() < prob:
                noisy_text = _add_severe_ocr_corruption(noisy_text)
            if random.random() < prob * 0.7:  # Less symbol corruption
                noisy_text = _add_ocr_symbol_corruption(noisy_text)
        
        print("NOISY TEXT:")
        print(noisy_text)
        print()
        
        # Compare entity detection
        compare_entity_detection(clean_text, noisy_text)

def test_pii_extraction():
    """Test PII extraction from noisy text."""
    
    print("\n" + "="*60)
    print("PII EXTRACTION FROM NOISY TEXT")
    print("="*60)
    
    clean_text = "Cliente: JUAN SEGUNDO GARCIA CARRASCO con RUT 7,082,003-K vive en ELIAS 42 VILLA HERMOSA, CORONEL. Fecha nacimiento: 28/09/1952. Teléfono: +56 9 8765 4321. Email: juan.garcia@email.com. Beneficiaria: ROSA ESTER PENA GONZALEZ, RUT 6.672.485-9."
    
    print("ORIGINAL:")
    print(clean_text)
    print()
    
    # Show entities in original
    print("ENTITIES IN ORIGINAL:")
    original_entities = identify_pii_entities(clean_text)
    for i, entity in enumerate(original_entities, 1):
        source_info = f" ({entity['source']})" if 'source' in entity else ""
        print(f"  {i}. {entity['label']}: '{entity['text']}'{source_info}")
    
    print()
    
    # Import the noise application function
    import random
    
    def apply_test_noise(text, iterations=2):
        result = text
        for i in range(iterations):
            if random.random() < 0.6:
                result = _add_ocr_character_noise(result)
            if random.random() < 0.6:
                result = _add_ocr_scanning_noise(result)
            if random.random() < 0.7:
                result = _add_severe_ocr_corruption(result)
            if random.random() < 0.4:
                result = _add_ocr_symbol_corruption(result)
        return result
    
    # Apply noise 3 times to show variation
    for i in range(3):
        noisy = apply_test_noise(clean_text, 2)
        print(f"NOISY VERSION {i+1}:")
        print(noisy)
        print()
        
        # Show entity detection for this noisy version
        compare_entity_detection(clean_text, noisy)

def create_corruption_analyzer():
    """Create a tool to analyze any text for OCR-like corruptions."""
    
    print("\n" + "="*60)
    print("CORRUPTION PATTERN ANALYZER")
    print("="*60)
    
    def analyze_corruption(original, corrupted):
        """Analyze differences between original and corrupted text."""
        print(f"ORIGINAL: {original}")
        print(f"CORRUPT:  {corrupted}")
        
        # Character-level analysis
        if len(original) != len(corrupted):
            print(f"LENGTH CHANGE: {len(original)} → {len(corrupted)}")
        
        # Find character differences
        diffs = []
        max_len = max(len(original), len(corrupted))
        for i in range(max_len):
            orig_char = original[i] if i < len(original) else '∅'
            corr_char = corrupted[i] if i < len(corrupted) else '∅'
            if orig_char != corr_char:
                diffs.append((i, orig_char, corr_char))
        
        if diffs:
            print("CHARACTER DIFFERENCES:")
            for pos, orig, corr in diffs[:10]:  # Show first 10
                print(f"  Position {pos}: '{orig}' → '{corr}'")
            if len(diffs) > 10:
                print(f"  ... and {len(diffs) - 10} more differences")
        
        print("-" * 40)
    
    return analyze_corruption

def interactive_test_mode():
    """Interactive mode to test custom text with entity detection."""
    
    print("\n" + "="*60)
    print("INTERACTIVE OCR NOISE TESTING WITH ENTITY DETECTION")
    print("="*60)
    print("Enter text to apply OCR noise, or 'quit' to exit")
    print()
    
    import random
    
    def apply_interactive_noise(text, iterations=2):
        result = text
        for i in range(iterations):
            if random.random() < 0.7:
                result = _add_ocr_character_noise(result)
            if random.random() < 0.6:
                result = _add_ocr_scanning_noise(result)
            if random.random() < 0.8:
                result = _add_severe_ocr_corruption(result)
            if random.random() < 0.5:
                result = _add_ocr_symbol_corruption(result)
        return result
    
    while True:
        try:
            user_input = input("Enter text: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            print("\nORIGINAL:")
            print(user_input)
            
            # Show original entities
            print("\nENTITIES IN ORIGINAL:")
            original_entities = identify_pii_entities(user_input)
            if original_entities:
                for i, entity in enumerate(original_entities, 1):
                    source_info = f" ({entity['source']})" if 'source' in entity else ""
                    print(f"  {i}. {entity['label']}: '{entity['text']}'{source_info}")
            else:
                print("  No entities detected")
            
            # Apply noise
            noisy_text = apply_interactive_noise(user_input, 2)
            
            print("\nWITH OCR NOISE:")
            print(noisy_text)
            print()
            
            # Compare entity detection
            compare_entity_detection(user_input, noisy_text)
            print("\n" + "-"*40 + "\n")
            
        except KeyboardInterrupt:
            break
    
    print("Goodbye!")

def main():
    """Main testing function."""
    print("OCR NOISE TESTING AND COMPARISON TOOL WITH ENTITY DETECTION")
    print("="*60)
    
    # Analyze real OCR sample
    real_ocr, corruptions, pii_elements = analyze_real_ocr_sample()
    
    # Test our noise generation
    test_noise_generation_similarity()
    
    # Test PII extraction scenarios
    test_pii_extraction()
    
    # Create analyzer tool
    analyzer = create_corruption_analyzer()
    
    # Example analysis
    print("\nEXAMPLE CORRUPTION ANALYSIS:")
    analyzer("Importante", "mportante")
    analyzer("Superintendencia", "Superintendenciall")
    analyzer("COMPAÑÍA", "COMPAKIA")
    
    # Interactive mode
    print("\nDo you want to test custom text with entity detection? (y/n): ", end="")
    try:
        if input().lower().startswith('y'):
            interactive_test_mode()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
