#!/usr/bin/env python3
"""
Test the focused model vs original model
"""

import spacy

def test_both_models():
    """Test both models side by side"""
    
    test_text = "El cliente Juan Carlos González con RUT 15.234.567-8 vive en Av. Providencia 123, Santiago."
    
    print("🔍 MODEL COMPARISON - BEFORE vs AFTER")
    print("=" * 60)
    print(f"📝 Test: {test_text}")
    print()
    
    # Test original model
    print("🔴 ORIGINAL MODEL (./model/model-best):")
    print("-" * 40)
    try:
        original_nlp = spacy.load("./model/model-best")
        original_doc = original_nlp(test_text)
        
        if original_doc.ents:
            for ent in original_doc.ents:
                color = "✅" if ent.label_ in ["CUSTOMER_NAME", "ID_NUMBER"] and "Juan Carlos" in ent.text or "15.234.567-8" in ent.text else "❌"
                print(f"   {color} {ent.label_:12} | {ent.text}")
        else:
            print("   ❌ No entities found")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test focused model
    print("🟢 FOCUSED MODEL (./focused_model/model-best):")
    print("-" * 40)
    try:
        focused_nlp = spacy.load("./focused_model/model-best")
        focused_doc = focused_nlp(test_text)
        
        if focused_doc.ents:
            for ent in focused_doc.ents:
                # Check if entities are correctly labeled
                correct = False
                if "Juan Carlos González" in ent.text and ent.label_ == "CUSTOMER_NAME":
                    correct = True
                elif "15.234.567-8" in ent.text and ent.label_ == "ID_NUMBER":
                    correct = True
                elif ("Av. Providencia 123" in ent.text or "Santiago" in ent.text) and ent.label_ == "ADDRESS":
                    correct = True
                
                color = "✅" if correct else "❌"
                print(f"   {color} {ent.label_:12} | {ent.text}")
        else:
            print("   ❌ No entities found")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    print("💡 EXPECTED RESULTS:")
    print("-" * 20)
    print("   ✅ CUSTOMER_NAME | Juan Carlos González")
    print("   ✅ ID_NUMBER     | 15.234.567-8")
    print("   ✅ ADDRESS       | Av. Providencia 123")
    print("   ✅ ADDRESS       | Santiago")

if __name__ == "__main__":
    test_both_models()
