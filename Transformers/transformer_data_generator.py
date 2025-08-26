"""
Transformer NER Dataset Generator for Latin American PII Data
=============================================================

This module generates realistic customer data for multiple Latin American countries
optimized for Transformer-based NER training (BERT, RoBERTa, etc.).

Features:
- Multi-country support: Chile, Mexico, Brazil, Uruguay
- Spanish and Portuguese language support
- Transformer-optimized sentence structures
- BIO/BILOU compatible entity labeling
- OCR noise simulation for robust training
- JSON output format for Hugging Face integration

Author: Andrés Vera Figueroa
Date: August 2025
Purpose: Transformer-based NER model training
"""

import json
import random
import argparse
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class EntitySpan:
    start: int
    end: int
    label: str
    text: str

class TransformerDataGenerator:
    def __init__(self):
        # Country-specific data optimized for transformers
        self.COUNTRY_DATA = {
            "chile": {
                "first_names": [
                    "JOSÉ", "MARÍA", "CARLOS", "ANA", "LUIS", "ELENA", "FERNANDO", "GABRIELA",
                    "RICARDO", "PATRICIA", "MIGUEL", "CAROLINA", "ANDRÉS", "FRANCISCA", 
                    "ANTONIO", "ALEJANDRA", "MANUEL", "CLAUDIA", "JORGE", "MÓNICA"
                ],
                "surnames": [
                    "GONZÁLEZ", "RODRÍGUEZ", "MUÑOZ", "ROJAS", "SILVA", "MORALES", "LÓPEZ",
                    "MARTÍNEZ", "GARCÍA", "HERNÁNDEZ", "PÉREZ", "SÁNCHEZ", "TORRES", "FLORES",
                    "VARGAS", "CASTILLO", "JIMÉNEZ", "MORENO", "GUTIÉRREZ", "HERRERA"
                ],
                "id_prefix": "RUT",
                "streets": [
                    "Av. Providencia", "Los Leones", "Las Condes", "Av. Vitacura", "Santa María",
                    "Av. Apoquindo", "Pedro de Valdivia", "Av. Kennedy", "Manuel Montt", "Suecia"
                ],
                "cities": ["Santiago", "Valparaíso", "Concepción", "La Serena", "Temuco"],
                "phone_prefix": "+56",
                "email_domains": ["gmail.com", "hotmail.com", "empresa.cl", "correo.cl"],
                "currency": "CLP",
                "language": "es"
            },
            "mexico": {
                "first_names": [
                    "JUAN", "MARÍA", "JOSÉ", "ANA", "LUIS", "CARMEN", "CARLOS", "GUADALUPE",
                    "ANTONIO", "ROSA", "JESÚS", "TERESA", "ALEJANDRO", "PATRICIA", "MANUEL",
                    "ELIZABETH", "FRANCISCO", "LETICIA", "DAVID", "YOLANDA"
                ],
                "surnames": [
                    "GARCÍA", "HERNÁNDEZ", "LÓPEZ", "MARTÍNEZ", "GONZÁLEZ", "PÉREZ", "SÁNCHEZ",
                    "RAMÍREZ", "CRUZ", "FLORES", "GÓMEZ", "MORALES", "VÁZQUEZ", "JIMÉNEZ",
                    "RUIZ", "DÍAZ", "MORENO", "MUÑOZ", "ÁLVAREZ", "ROMERO"
                ],
                "id_prefix": "CURP",
                "streets": [
                    "Av. Reforma", "Insurgentes Sur", "Polanco", "Roma Norte", "Condesa",
                    "Santa Fe", "Del Valle", "Coyoacán", "Xochimilco", "Tlalpan"
                ],
                "cities": ["Ciudad de México", "Guadalajara", "Monterrey", "Puebla", "Tijuana"],
                "phone_prefix": "+52",
                "email_domains": ["gmail.com", "hotmail.com", "empresa.mx", "correo.mx"],
                "currency": "MXN",
                "language": "es"
            },
            "brazil": {
                "first_names": [
                    "JOÃO", "MARIA", "JOSÉ", "ANA", "CARLOS", "ANTÔNIA", "PEDRO", "FRANCISCA",
                    "PAULO", "ADRIANA", "LUIZ", "JULIANA", "MARCOS", "MÁRCIA", "ANTÔNIO",
                    "FERNANDA", "FRANCISCO", "PATRÍCIA", "DANIEL", "ALINE"
                ],
                "surnames": [
                    "SILVA", "SANTOS", "OLIVEIRA", "SOUZA", "RODRIGUES", "FERREIRA", "ALVES",
                    "PEREIRA", "LIMA", "GOMES", "RIBEIRO", "CARVALHO", "ALMEIDA", "LOPES",
                    "SOARES", "FERNANDES", "VIEIRA", "BARBOSA", "ROCHA", "DIAS"
                ],
                "id_prefix": "CPF",
                "streets": [
                    "Rua das Flores", "Av. Paulista", "Copacabana", "Ipanema", "Botafogo",
                    "Vila Madalena", "Moema", "Jardins", "Liberdade", "Centro"
                ],
                "cities": ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador", "Brasília"],
                "phone_prefix": "+55",
                "email_domains": ["gmail.com", "hotmail.com", "empresa.com.br", "uol.com.br"],
                "currency": "BRL",
                "language": "pt"
            },
            "uruguay": {
                "first_names": [
                    "CARLOS", "MARÍA", "JOSÉ", "ANA", "LUIS", "ELENA", "FERNANDO", "PATRICIA",
                    "JORGE", "CLAUDIA", "ALBERTO", "MÓNICA", "EDUARDO", "ADRIANA", "DANIEL",
                    "GABRIELA", "RICARDO", "CECILIA", "ALEJANDRO", "MARIANA"
                ],
                "surnames": [
                    "RODRÍGUEZ", "GONZÁLEZ", "GARCÍA", "LÓPEZ", "MARTÍNEZ", "PÉREZ", "FERNÁNDEZ",
                    "SÁNCHEZ", "DÍAZ", "ÁLVAREZ", "ROMERO", "VARGAS", "CASTRO", "RAMOS",
                    "MORALES", "ORTEGA", "DELGADO", "JIMÉNEZ", "RUIZ", "HERNÁNDEZ"
                ],
                "id_prefix": "CI",
                "streets": [
                    "18 de Julio", "Av. Italia", "Pocitos", "Montevideo", "Punta Carretas",
                    "Cordón", "Centro", "Malvín", "Carrasco", "Buceo"
                ],
                "cities": ["Montevideo", "Punta del Este", "Salto", "Rivera", "Maldonado"],
                "phone_prefix": "+598",
                "email_domains": ["gmail.com", "hotmail.com", "empresa.com.uy", "correo.uy"],
                "currency": "UYU",
                "language": "es"
            }
        }
        
        # Transformer-optimized templates (shorter for better tokenization)
        self.templates = {
            "es": [  # Spanish templates
                "Cliente: {} {} - {} {} - {}, {} - Tel: {} - Email: {} - Monto: {} - Ref: {}",
                "Datos: {} {} ({}) Dirección: {}, {} Contacto: {} {} Valor: {} Código: {}",
                "Usuario {} {} ID {} ubicado en {}, {} teléfono {} correo {} saldo {} operación {}",
                "Factura: {} {} - {} {} - {}, {} - {} - {} - Total: {} - N°: {}",
                "Sr/a {} {} documento {} domicilio {}, {} fono {} email {} pago {} trámite {}",
                "Registro: {} {} - {} {} - {}, {} - {} {} - Importe: {} - Serie: {}",
                "Clte: {} {} - Doc: {} - Dir: {}, {} - Tel: {} - @ {} - $$ {} - #: {}",
                "AVISO: {} {} {} vive {}, {} tel {} mail {} debe {} ref {}",
            ],
            "pt": [  # Portuguese templates (for Brazil)
                "Cliente: {} {} - {} {} - {}, {} - Tel: {} - Email: {} - Valor: {} - Ref: {}",
                "Dados: {} {} ({}) Endereço: {}, {} Contato: {} {} Quantia: {} Código: {}",
                "Usuário {} {} ID {} localizado em {}, {} telefone {} email {} saldo {} operação {}",
                "Fatura: {} {} - {} {} - {}, {} - {} - {} - Total: {} - N°: {}",
                "Sr/a {} {} documento {} domicílio {}, {} fone {} email {} pagamento {} processo {}",
                "Registro: {} {} - {} {} - {}, {} - {} {} - Valor: {} - Série: {}",
                "Cte: {} {} - Doc: {} - End: {}, {} - Tel: {} - @ {} - R$ {} - #: {}",
                "AVISO: {} {} {} mora {}, {} tel {} email {} deve {} ref {}",
            ]
        }
    
    def generate_id_number(self, country: str) -> str:
        """Generate realistic country-specific ID numbers"""
        if country == "chile":
            # Chilean RUT format: XX.XXX.XXX-Y
            base = random.randint(10000000, 25999999)
            check_digit = random.choice(['0','1','2','3','4','5','6','7','8','9','K'])
            return f"{base//1000000}.{(base//1000)%1000:03d}.{base%1000:03d}-{check_digit}"
        
        elif country == "mexico":
            # Mexican CURP format: AAAA######AAAAAA##
            letters1 = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
            date = f"{random.randint(800101, 991231)}"
            letters2 = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
            gender = random.choice('HM')
            state = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            consonant = random.choice('BCDFGHJKLMNPQRSTVWXYZ')
            numbers = f"{random.randint(0, 9)}{random.randint(0, 9)}"
            return f"{letters1}{date}{letters2}{gender}{state}{consonant}{numbers}"
        
        elif country == "brazil":
            # Brazilian CPF format: XXX.XXX.XXX-XX
            cpf = ''.join([str(random.randint(0, 9)) for _ in range(11)])
            return f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"
        
        else:  # uruguay
            # Uruguayan CI format: X.XXX.XXX-X
            base = random.randint(1000000, 9999999)
            check_digit = random.randint(0, 9)
            return f"{base//1000000}.{(base//1000)%1000:03d}.{base%1000:03d}-{check_digit}"
    
    def generate_entity_data(self, country: str) -> Dict:
        """Generate realistic entity data for a country"""
        data = self.COUNTRY_DATA[country]
        
        # Generate realistic names
        first_name = random.choice(data["first_names"])
        surname = random.choice(data["surnames"])
        full_name = f"{first_name} {surname}"
        
        # Generate other entities
        id_number = self.generate_id_number(country)
        street = random.choice(data["streets"])
        street_number = random.randint(10, 999)
        address = f"{street} {street_number}"
        city = random.choice(data["cities"])
        
        # Phone number
        if country == "brazil":
            phone = f"{data['phone_prefix']} {random.randint(11, 99)} {random.randint(90000000, 99999999)}"
        else:
            phone = f"{data['phone_prefix']} {random.randint(900000000, 999999999)}"
        
        # Email
        email_user = first_name.lower().replace('ã', 'a').replace('ç', 'c').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        email = f"{email_user}.{surname.lower()}@{random.choice(data['email_domains'])}"
        
        # Amount with currency
        amount_value = random.randint(1000, 999999)
        if country == "brazil":
            amount = f"R$ {amount_value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        else:
            amount = f"${amount_value:,}"
        
        # Sequence number
        seq_patterns = [
            f"{random.randint(100000, 999999)}-{random.choice(['A', 'B', 'C', 'D'])}",
            f"{random.choice(['CL', 'MX', 'BR', 'UY'])}{random.randint(10000, 99999)}",
            f"{random.randint(1000000, 9999999)}",
            f"{random.choice('ABCDEFGH')}{random.randint(100000, 999999)}"
        ]
        seq_number = random.choice(seq_patterns)
        
        return {
            "customer_name": full_name,
            "id_number": id_number,
            "address": address,
            "city": city,
            "phone": phone,
            "email": email,
            "amount": amount,
            "seq_number": seq_number,
            "country": country,
            "language": data["language"]
        }
    
    def apply_noise(self, text: str, noise_level: float = 0.3) -> str:
        """Apply OCR-like noise optimized for transformer training"""
        if random.random() > noise_level:
            return text
        
        # Conservative OCR substitutions for better token alignment
        substitutions = {
            'o': '0', 'O': '0', '0': 'o',
            'i': '1', 'I': '1', 'l': '1', '1': 'i',
            's': '5', 'S': '5', '5': 's',
            'g': '6', 'G': '6', '6': 'g',
            'b': '8', 'B': '8', '8': 'b',
            'ñ': 'n', 'ç': 'c', 'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'â': 'a', 'ã': 'a', 'ê': 'e', 'ô': 'o', 'ü': 'u'
        }
        
        chars = list(text)
        changes_made = 0
        max_changes = max(1, len(text) // 20)  # Limit changes to 5% of characters
        
        for i, char in enumerate(chars):
            if changes_made >= max_changes:
                break
            if random.random() < 0.15 and char in substitutions:
                chars[i] = substitutions[char]
                changes_made += 1
        
        result = ''.join(chars)
        
        # Occasionally add/remove spaces (very conservative)
        if random.random() < 0.1:
            if ' ' in result and random.random() < 0.5:
                result = result.replace(' ', '  ', 1)  # Add extra space
            elif '  ' in result:
                result = result.replace('  ', ' ', 1)  # Remove extra space
        
        return result
    
    def create_training_example(self, country: str, noise_level: float = 0.3) -> Tuple[str, List[EntitySpan]]:
        """Create a single training example with entities"""
        entity_data = self.generate_entity_data(country)
        language = entity_data["language"]
        template = random.choice(self.templates[language])
        
        # Fill template
        sentence = template.format(
            entity_data["customer_name"].split()[0],  # First name
            entity_data["customer_name"].split()[1],  # Last name
            entity_data["id_number"],
            entity_data["address"],
            entity_data["city"],
            entity_data["phone"],
            entity_data["email"],
            entity_data["amount"],
            entity_data["seq_number"]
        )
        
        # Apply noise BEFORE entity detection
        original_sentence = sentence
        noisy_sentence = self.apply_noise(sentence, noise_level)
        
        # Find entity spans in the original sentence first, then map to noisy
        entities = []
        entity_map = [
            (entity_data["customer_name"].split()[0], "CUSTOMER_NAME"),
            (entity_data["customer_name"].split()[1], "CUSTOMER_NAME"),
            (entity_data["id_number"], "ID_NUMBER"),
            (entity_data["address"], "ADDRESS"),
            (entity_data["city"], "ADDRESS"),
            (entity_data["phone"], "PHONE_NUMBER"),
            (entity_data["email"], "EMAIL"),
            (entity_data["amount"], "AMOUNT"),
            (entity_data["seq_number"], "SEQ_NUMBER")
        ]
        
        # Sort by length (longest first) for better matching
        entity_map.sort(key=lambda x: len(x[0]), reverse=True)
        used_positions = set()
        
        for entity_text, label in entity_map:
            # Try to find in noisy sentence
            start = noisy_sentence.find(entity_text)
            if start == -1:
                # Fallback: try with some flexibility
                import re
                pattern = re.escape(entity_text).replace(r'\ ', r'\s+')
                match = re.search(pattern, noisy_sentence, re.IGNORECASE)
                if match:
                    start = match.start()
                    entity_text = match.group()  # Use the matched text
            
            if start != -1:
                end = start + len(entity_text)
                # Check for overlaps
                position_range = set(range(start, end))
                if not position_range.intersection(used_positions):
                    entities.append(EntitySpan(start, end, label, entity_text))
                    used_positions.update(position_range)
        
        return noisy_sentence, entities
    
    def generate_dataset(self, 
                        countries: List[str],
                        train_size: int,
                        dev_size: int,
                        noise_level: float = 0.3,
                        output_dir: str = "output") -> Dict:
        """Generate complete dataset for transformer training"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🚀 Generating {train_size + dev_size} examples for Transformer training...")
        print(f"🌎 Countries: {', '.join(countries)}")
        print(f"🎭 Noise level: {noise_level}")
        
        # Generate training data
        train_examples = []
        entity_stats = {"train": {}}
        
        for i in range(train_size):
            country = random.choice(countries)
            text, entities = self.create_training_example(country, noise_level)
            
            entity_list = [{"start": e.start, "end": e.end, "label": e.label} for e in entities]
            
            train_examples.append({
                "id": f"train_{i}",
                "text": text,
                "entities": entity_list,
                "country": country
            })
            
            # Track entity statistics
            for entity in entities:
                label = entity.label
                entity_stats["train"][label] = entity_stats["train"].get(label, 0) + 1
            
            if (i + 1) % 10000 == 0:
                print(f"✅ Generated {i + 1:,}/{train_size:,} training examples")
        
        # Generate dev data  
        dev_examples = []
        entity_stats["dev"] = {}
        
        for i in range(dev_size):
            country = random.choice(countries)
            text, entities = self.create_training_example(country, noise_level * 0.8)  # Reduce noise for dev
            
            entity_list = [{"start": e.start, "end": e.end, "label": e.label} for e in entities]
            
            dev_examples.append({
                "id": f"dev_{i}",
                "text": text,
                "entities": entity_list,
                "country": country
            })
            
            # Track entity statistics
            for entity in entities:
                label = entity.label
                entity_stats["dev"][label] = entity_stats["dev"].get(label, 0) + 1
            
            if (i + 1) % 5000 == 0:
                print(f"✅ Generated {i + 1:,}/{dev_size:,} dev examples")
        
        # Save datasets
        train_file = os.path.join(output_dir, f"train_transformer_{train_size}.json")
        dev_file = os.path.join(output_dir, f"dev_transformer_{dev_size}.json")
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_examples, f, ensure_ascii=False, indent=2)
        
        with open(dev_file, 'w', encoding='utf-8') as f:
            json.dump(dev_examples, f, ensure_ascii=False, indent=2)
        
        # Calculate statistics
        total_train_entities = sum(entity_stats["train"].values())
        total_dev_entities = sum(entity_stats["dev"].values())
        expected_train_entities = train_size * 9  # 9 entities per example expected
        expected_dev_entities = dev_size * 9
        
        train_success_rate = (total_train_entities / expected_train_entities) * 100 if expected_train_entities > 0 else 0
        dev_success_rate = (total_dev_entities / expected_dev_entities) * 100 if expected_dev_entities > 0 else 0
        
        # Generate comprehensive statistics
        stats = {
            "creation_date": datetime.now().isoformat(),
            "total_examples": train_size + dev_size,
            "train_size": train_size,
            "dev_size": dev_size,
            "countries": countries,
            "noise_level": noise_level,
            "entity_statistics": {
                "train": {
                    "total_entities": total_train_entities,
                    "expected_entities": expected_train_entities,
                    "success_rate": f"{train_success_rate:.2f}%",
                    "by_type": entity_stats["train"]
                },
                "dev": {
                    "total_entities": total_dev_entities,
                    "expected_entities": expected_dev_entities,
                    "success_rate": f"{dev_success_rate:.2f}%",
                    "by_type": entity_stats["dev"]
                }
            },
            "files": {
                "train": train_file,
                "dev": dev_file
            }
        }
        
        stats_file = os.path.join(output_dir, f"transformer_dataset_stats_{train_size + dev_size}.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Dataset generated successfully!")
        print(f"📁 Files saved in: {output_dir}")
        print(f"📊 Training examples: {train_size:,} (Entity success: {train_success_rate:.1f}%)")
        print(f"📊 Dev examples: {dev_size:,} (Entity success: {dev_success_rate:.1f}%)")
        print(f"📄 Statistics: {stats_file}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Generate NER dataset for Transformer training")
    parser.add_argument("--countries", nargs="+", default=["chile", "mexico", "brazil", "uruguay"],
                       choices=["chile", "mexico", "brazil", "uruguay"], 
                       help="Countries to include")
    parser.add_argument("--train-size", type=int, default=50000, help="Training set size")
    parser.add_argument("--dev-size", type=int, default=10000, help="Development set size")
    parser.add_argument("--noise-level", type=float, default=0.3, help="Noise level (0.0-1.0)")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    print("🤖 Transformer NER Dataset Generator")
    print("====================================")
    
    generator = TransformerDataGenerator()
    stats = generator.generate_dataset(
        countries=args.countries,
        train_size=args.train_size,
        dev_size=args.dev_size,
        noise_level=args.noise_level,
        output_dir=args.output_dir
    )
    
    print(f"\n🎉 Dataset generation completed!")
    print(f"📈 Ready for transformer training with:")
    print(f"   - Train: {stats['train_size']:,} examples")
    print(f"   - Dev: {stats['dev_size']:,} examples")
    print(f"   - Languages: Spanish + Portuguese")
    print(f"   - Countries: {', '.join(stats['countries'])}")

if __name__ == "__main__":
    main()
