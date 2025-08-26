"""
Transformer NER Inference Example
=================================

Example script showing how to use trained transformer models for inference
on new PII data in Spanish and Portuguese.

Usage:
    python inference_example.py --model-path models/transformer_ner_*/final_model
"""

import json
import argparse
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class TransformerNERInference:
    def __init__(self, model_path: str):
        """Initialize the NER inference pipeline"""
        print(f"🤖 Loading model from: {model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        # Create pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",  # Groups B- and I- tags
            device=0 if torch.cuda.is_available() else -1
        )
        
        print(f"✅ Model loaded successfully")
        print(f"🏷️  Labels: {list(self.model.config.id2label.values())}")
    
    def predict(self, text: str) -> List[Dict]:
        """Predict entities in a text"""
        try:
            results = self.ner_pipeline(text)
            
            # Clean and format results
            formatted_results = []
            for entity in results:
                formatted_results.append({
                    "text": entity["word"].replace("##", ""),  # Remove BERT subword markers
                    "label": entity["entity_group"],
                    "confidence": round(entity["score"], 4),
                    "start": entity["start"],
                    "end": entity["end"]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return []
    
    def predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Predict entities for multiple texts"""
        results = []
        for i, text in enumerate(texts):
            print(f"Processing {i+1}/{len(texts)}: {text[:50]}...")
            entities = self.predict(text)
            results.append(entities)
        return results
    
    def demo_predictions(self):
        """Run demo predictions on sample texts"""
        print("\n🎭 Demo Predictions")
        print("==================")
        
        # Sample texts in Spanish and Portuguese
        demo_texts = [
            # Spanish examples
            "Cliente: JOSÉ GONZÁLEZ - RUT 12.345.678-9 - Av. Providencia 123, Santiago - Tel: +56 912345678 - Email: jose.gonzalez@empresa.cl - Monto: $50,000 - Ref: CL12345",
            "Datos: MARÍA RODRÍGUEZ (CURP ABCD123456EFGHIJ12) Dirección: Insurgentes Sur 456, Ciudad de México Contacto: +52 5512345678 maria.rodriguez@correo.mx Valor: $75,000 Código: MX67890",
            
            # Portuguese examples  
            "Cliente: JOÃO SILVA - CPF 123.456.789-01 - Av. Paulista 456, São Paulo - Tel: +55 11 987654321 - Email: joao.silva@empresa.com.br - Valor: R$ 25.000,00 - Ref: BR78901",
            "Usuário MARIA SANTOS ID 987.654.321-09 localizado em Copacabana 789, Rio de Janeiro telefone +55 21 876543210 email maria.santos@uol.com.br saldo R$ 15.750,50 operação BR23456",
            
            # Noisy examples (OCR-like)
            "Sr/a CAR1OS MARTINEZ documento 8.765.432-1 domicilio Las C0ndes 321, Santiago fono +56 987654321 email carI0s.martinez@gmail.com pago $30,00O ref A789012"
        ]
        
        for i, text in enumerate(demo_texts, 1):
            print(f"\n📝 Example {i}:")
            print(f"Text: {text}")
            
            entities = self.predict(text)
            
            if entities:
                print(f"🏷️  Entities found ({len(entities)}):")
                for entity in entities:
                    print(f"   - {entity['label']}: '{entity['text']}' (confidence: {entity['confidence']})")
            else:
                print("❌ No entities found")

def main():
    parser = argparse.ArgumentParser(description="Transformer NER Inference")
    parser.add_argument("--model-path", required=True, help="Path to trained model directory")
    parser.add_argument("--text", help="Text to analyze")
    parser.add_argument("--demo", action="store_true", help="Run demo predictions")
    
    args = parser.parse_args()
    
    print("🤖 Transformer NER Inference")
    print("=============================")
    
    # Initialize inference
    try:
        inference = TransformerNERInference(args.model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Make sure the model path exists and contains the trained model files")
        return
    
    if args.demo:
        # Run demo
        inference.demo_predictions()
    
    elif args.text:
        # Predict single text
        print(f"\n📝 Analyzing text: {args.text}")
        entities = inference.predict(args.text)
        
        if entities:
            print(f"\n🏷️  Entities found ({len(entities)}):")
            for entity in entities:
                print(f"   - {entity['label']}: '{entity['text']}' (confidence: {entity['confidence']})")
        else:
            print("❌ No entities found")
    
    else:
        # Interactive mode
        print("\n💬 Interactive Mode (type 'quit' to exit)")
        print("Enter text to analyze:")
        
        while True:
            try:
                text = input("\n> ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if text:
                    entities = inference.predict(text)
                    
                    if entities:
                        print(f"🏷️  Entities ({len(entities)}):")
                        for entity in entities:
                            print(f"   - {entity['label']}: '{entity['text']}' ({entity['confidence']:.3f})")
                    else:
                        print("❌ No entities found")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
