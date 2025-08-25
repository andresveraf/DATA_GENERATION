#!/usr/bin/env python3
"""
Quick PII NER Test - Simple script to test trained model
"""

import spacy

def quick_test():
    """Quick test function"""
    
    # Try to load trained model from different possible locations
    model_paths = [
        "./precise_model/model-best",  # Precise hand-crafted model (highest priority)
        "./focused_model/model-best",  # Fixed focused model (if available)
        "./model/model-best",          # Best model from training (preferred)
        "./model/model-last",          # Last model from training
        "./model"                      # Legacy path
    ]
    
    nlp = None
    for model_path in model_paths:
        try:
            print(f"🔄 Trying to load trained model from: {model_path}")
            nlp = spacy.load(model_path)
            print(f"✅ Trained model loaded successfully from {model_path}!")
            break
        except:
            print(f"⚠️  Model not found at {model_path}")
            continue
    
    if nlp is None:
        print("⚠️  No trained model found, using base Spanish model...")
        try:
            nlp = spacy.load("es_core_news_lg")
            print("✅ Using es_core_news_lg (large Spanish model)")
        except:
            try:
                nlp = spacy.load("es_core_news_sm")
                print("✅ Using es_core_news_sm (small Spanish model)")
            except:
                print("❌ No Spanish models found")
                return
    
    # Test text
    test_text = "El cliente Juan Carlos González con RUT 15.234.567-8 vive en Av. Providencia 123, Santiago."
    
    print(f"\n📝 Testing: {test_text}")
    print("=" * 60)
    
    doc = nlp(test_text)
    
    if doc.ents:
        print(f"🎯 Found {len(doc.ents)} entities:")
        for ent in doc.ents:
            print(f"   {ent.label_:12} | {ent.text}")
    else:
        print("❌ No entities found")
    
    print("\n💡 To test with your own text, modify the test_text variable above")

if __name__ == "__main__":
    quick_test()
