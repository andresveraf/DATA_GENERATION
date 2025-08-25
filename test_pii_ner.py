#!/usr/bin/env python3
"""
PII NER Tester - Test trained spaCy model for Personal Information detection

This script allows you to test the trained NER model to identify:
- CUSTOMER_NAME: Full customer names (with compound names and surnames)
- ID_NUMBER: Government IDs (RUT, DNI, CPF, CI, CURP, etc.)
- ADDRESS: Street addresses and cities
- PHONE_NUMBER: Phone numbers with country codes
- EMAIL: Email addresses
- AMOUNT: Monetary amounts with currencies
- SEQ_NUMBER: Sequential reference numbers

Usage:
    python3 test_pii_ner.py
    
Then enter your text when prompted, or modify the test_texts list below.
"""

import spacy
from spacy import displacy
import sys
from pathlib import Path


def load_trained_model(model_path: str = "./focused_model/model-best"):
    """
    Load the trained spaCy NER model.
    
    Args:
        model_path (str): Path to the trained model directory
                         Default: "./focused_model/model-best" (focused model with correct labels)
        
    Returns:
        spacy.Language: Loaded spaCy model
    """
    model_path = Path(model_path)
    
    # Try different model paths in order of preference
    model_paths_to_try = [
        model_path,                           # Specified path
        Path("./focused_model/model-best"),   # Focused model (correct labels)
        Path("./focused_model/model-last"),   # Focused model last epoch
        Path("./model/model-best"),          # Original best model from training
        Path("./model/model-last"),          # Original last model from training  
        Path("./model")                       # Legacy path
    ]
    
    for path in model_paths_to_try:
        if path.exists():
            try:
                print(f"🔄 Loading trained model from: {path}")
                nlp = spacy.load(path)
                print(f"✅ Model loaded successfully!")
                print(f"📊 Model info: {nlp.meta['name']} v{nlp.meta['version']}")
                return nlp
            except Exception as e:
                print(f"⚠️  Could not load model from {path}: {e}")
                continue
    
    print(f"❌ No trained model found in any of these locations:")
    for path in model_paths_to_try:
        print(f"   - {path}")
    print(f"💡 Make sure you've trained the model first using:")
    print(f"   python3 -m spacy train config.cfg --output ./model --paths.train large_dataset/train.spacy --paths.dev large_dataset/dev.spacy")
    return None


def test_text_with_model(nlp, text: str, show_details: bool = True):
    """
    Test a text string with the trained NER model.
    
    Args:
        nlp: Loaded spaCy model
        text (str): Text to analyze
        show_details (bool): Whether to show detailed entity information
    """
    print(f"\n📝 Testing text: {text}")
    print("=" * 80)
    
    # Process the text
    doc = nlp(text)
    
    # Show entities found
    if doc.ents:
        print(f"🎯 Found {len(doc.ents)} entities:")
        print("-" * 40)
        
        for i, ent in enumerate(doc.ents, 1):
            confidence = f" ({ent._.get('confidence', 'N/A')})" if hasattr(ent._, 'confidence') else ""
            print(f"{i:2d}. {ent.label_:12} | {ent.text:20} | Position: {ent.start_char}-{ent.end_char}{confidence}")
            
            if show_details:
                # Show context around the entity
                start_context = max(0, ent.start_char - 20)
                end_context = min(len(text), ent.end_char + 20)
                context = text[start_context:end_context].replace('\n', ' ')
                print(f"              Context: ...{context}...")
        
        print("-" * 40)
        
        # Group entities by type
        entity_groups = {}
        for ent in doc.ents:
            if ent.label_ not in entity_groups:
                entity_groups[ent.label_] = []
            entity_groups[ent.label_].append(ent.text)
        
        print("📊 Entities by type:")
        for label, entities in entity_groups.items():
            print(f"   {label}: {entities}")
            
    else:
        print("❌ No entities found in the text")
    
    print()


def interactive_mode(nlp):
    """
    Interactive mode - let user input text to test.
    """
    print("\n🔍 INTERACTIVE PII DETECTION MODE")
    print("=" * 50)
    print("Enter text to analyze (or 'quit' to exit):")
    print("Example: 'El cliente Juan Carlos González con RUT 15.234.567-8 vive en Av. Providencia 123, Santiago'")
    print()
    
    while True:
        try:
            user_text = input("📝 Enter text: ").strip()
            
            if user_text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_text:
                print("⚠️  Please enter some text to analyze")
                continue
                
            test_text_with_model(nlp, user_text)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing text: {e}")


def run_predefined_tests(nlp):
    """
    Run tests with predefined text samples.
    """
    print("\n🧪 RUNNING PREDEFINED TESTS")
    print("=" * 50)
    
    # Test texts covering different scenarios
    test_texts = [
        # Chilean examples
        "El cliente Juan Carlos González Rodríguez con RUT 15.234.567-8 vive en Av. Providencia 1234, Santiago.",
        
        # Argentine examples  
        "María José Silva Martínez DNI 32456789 reside en Av. Corrientes 567, Buenos Aires.",
        
        # Brazilian examples
        "O cliente Pedro dos Santos CPF 123.456.789-01 mora na Rua Augusta 890, São Paulo.",
        
        # Mixed entities
        "Cliente Ana Sofía Pérez López con identificación 20.567.234-5 ubicada en Calle San Diego 456, Valparaíso. Tel: +56 9 8765 4321. Email: ana.perez@gmail.com. Monto: $45,000 CLP. Ref: 7892345.",
        
        # Compound names and second surnames
        "Luis Miguel Fernández Sánchez RUT 18.765.432-1 domiciliado en Pasaje Los Álamos 789, Concepción.",
        
        # Different sequence numbers
        "Reclamo 7009808 de José Antonio Ramírez con RUT 12.345.678-9 en Av. Las Heras 234.",
        "Póliza 57575-A para Carmen Rosa Vásquez Morales ID 25.678.901-2.",
        
        # Address variations
        "Cliente registrado: Roberto Carlos Díaz en Calle Portugal 123, La Serena.",
        
        # Portuguese text
        "Cadastro de Fernando Silva Oliveira CPF 987.654.321-00 endereço Av. Paulista 1500, São Paulo.",
        
        # Multiple people in one text
        "Reunión entre Juan Pérez RUT 11.111.111-1 y María González RUT 22.222.222-2 en Av. Apoquindo 500."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🔸 TEST {i}/{len(test_texts)}")
        test_text_with_model(nlp, text, show_details=False)


def main():
    """
    Main function - loads model and runs tests.
    """
    print("🚀 PII NER TESTER - Latin American Customer Data")
    print("=" * 60)
    
    # Load the trained model
    nlp = load_trained_model("./model")
    
    if nlp is None:
        print("\n💡 Alternative: Test with base spaCy model (limited accuracy)")
        choice = input("Load base Spanish model instead? (y/n): ").lower()
        if choice == 'y':
            try:
                nlp = spacy.load("es_core_news_lg")
                print("✅ Loaded base Spanish model (limited PII detection)")
            except OSError:
                try:
                    nlp = spacy.load("es_core_news_sm")  
                    print("✅ Loaded small Spanish model (very limited)")
                except OSError:
                    print("❌ No Spanish models available")
                    return
        else:
            return
    
    # Show model capabilities
    if hasattr(nlp, 'pipe_labels') and 'ner' in nlp.pipe_labels:
        labels = nlp.get_pipe('ner').labels
        print(f"🏷️  Model recognizes {len(labels)} entity types:")
        for label in sorted(labels):
            description = {
                'CUSTOMER_NAME': 'Full customer names',
                'ID_NUMBER': 'Government IDs (RUT, DNI, CPF, etc.)',
                'ADDRESS': 'Street addresses and cities',
                'PHONE_NUMBER': 'Phone numbers',
                'EMAIL': 'Email addresses',
                'AMOUNT': 'Monetary amounts',
                'SEQ_NUMBER': 'Sequential reference numbers'
            }.get(label, 'Unknown entity type')
            print(f"   • {label}: {description}")
    
    # Choose test mode
    print(f"\n🎯 Choose test mode:")
    print(f"   1. Run predefined tests")
    print(f"   2. Interactive mode (enter your own text)")
    print(f"   3. Both")
    
    try:
        choice = input("\nSelect option (1/2/3): ").strip()
        
        if choice in ['1', '3']:
            run_predefined_tests(nlp)
        
        if choice in ['2', '3']:
            interactive_mode(nlp)
        
        if choice not in ['1', '2', '3']:
            print("Invalid choice, running both modes...")
            run_predefined_tests(nlp)
            interactive_mode(nlp)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
