"""
Complete PII Training Data Generator for Chilean Documents (FIXED VERSION)
User: andresveraf
Date: 2025-08-25 21:43:49 UTC

This script generates large-scale training datasets for PII (Personally Identifiable Information)
detection in Chilean government and financial documents with realistic noise patterns.

FIXES:
- Fixed entity span validation (no leading/trailing whitespace/punctuation)
- Improved noise correction logic
- Standardized entity types for spaCy compatibility
- Added proper entity validation and cleaning
- Fixed character position tracking after noise application

Features:
- 100K+ training examples generation
- PII classification (PERSON, LOCATION, PHONE, MISC)
- Status tracking (OK/NO)
- Realistic Chilean document noise
- Excel export for validation
- spaCy DocBin format for training

Usage:
    # Generate 100K training + 20K test
    python data_generation_noisy_fixed.py --mode create-dataset --train-size 100000 --dev-size 20000
    
    # Create Excel review
    python data_generation_noisy_fixed.py --mode excel-export --excel-examples 1000
    
    # See examples
    python data_generation_noisy_fixed.py --mode demo
"""

import random
import spacy
from spacy.tokens import DocBin
from typing import Tuple, Dict, List, Any, Optional
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import re
import argparse
import string

# Global sequence counter for generating unique sequential IDs
_sequence_counter = 10000

def get_next_sequence() -> int:
    """Generate the next sequential number for record identification."""
    global _sequence_counter
    _sequence_counter += 1
    return _sequence_counter

# -----------------
# DATA SOURCES
# -----------------

# Chilean first names
first_names = [
    # Masculine names
    "AGUSTÍN", "ALEJANDRO", "ALONSO", "ÁLVARO", "ANDRÉS", "ÁXEL", "BAUTISTA", "BENJAMÍN", "BRUNO", "CALEB",
    "CAMILO", "CARLOS", "CRISTÓBAL", "CRISTIAN", "DAMIÁN", "DANIEL", "DAVID", "DIEGO", "EDUARDO", "ELÍAS",
    "EMILIANO", "EMMANUEL", "ENRIQUE", "ESTEBAN", "ETHAN", "FEDERICO", "FERNANDO", "FRANCISCO", "GABRIEL",
    "GAEL", "GASPAR", "GERMÁN", "GUSTAVO", "HERNÁN", "IAN", "IGNACIO", "ISIDORO", "IVÁN", "JAIR", "JAIRO",
    "JASON", "JEREMY", "JHON", "JOAQUÍN", "JORGE", "JUAN", "JULIÁN", "KEVIN", "KIAN", "LEÓN", "LEONARDO",
    "LIAM", "LORENZO", "LUCCA", "LUIS", "MARCELO", "MARCO", "MARTÍN", "MATÍAS", "MATEO", "MAURICIO",
    "MAXIMILIANO", "MIGUEL", "NICOLÁS", "OLIVER", "OMAR", "ORLANDO", "PATRICIO", "PAULO", "PEDRO", "RAFAEL",
    "RAMIRO", "RICARDO", "ROBERTO", "RODRIGO", "RUBÉN", "SAMUEL", "SANTIAGO", "SEBASTIÁN", "SIMÓN", "THIAGO",
    "TOBÍAS", "TOMÁS", "VALENTINO", "VÍCTOR", "VICENTE", "WALTER", "XANDER", "ZAHIR",
    
    # Feminine names
    "AGUSTINA", "AINHOA", "AITANA", "ALBA", "ALEJANDRA", "ALEXA", "ALEXANDRA", "ALMENDRA", "AMANDA", "AMELIA",
    "ANAÍS", "ANTONELLA", "ANTONIA", "ARANTXA", "ARIADNA", "AROHA", "AZUL", "BELÉN", "BLANCA", "BRISA",
    "CAMILA", "CARLA", "CAROLINA", "CATALINA", "CELIA", "CLARA", "CLAUDIA", "CONSTANZA", "DANIELA", "DÉBORA",
    "DIANA", "DOMINIQUE", "ELISA", "ELIZABETH", "EMILIA", "EMMA", "ESMERALDA", "ESTEFANÍA", "FERNANDA",
    "FLORENCIA", "FRANCISCA", "GABRIELA", "GIOVANNA", "ISABELLA", "IVANNA", "JAVIERA", "JIMENA", "JOSEFINA",
    "JUANITA", "JULIETA", "KARINA", "KARLA", "KATIA", "KIARA", "LARA", "LAURA", "LAYLA", "LILA", "LUCIANA",
    "LUISA", "LUNA", "MACARENA", "MAGDALENA", "MANUELA", "MARÍA", "MARTINA", "MATILDA", "MÍA", "MILA",
    "MIREYA", "NATALIA", "NEREA", "NICOLE", "NOELIA", "OLIVIA", "PALOMA", "PAOLA", "PAULINA", "PAZ",
    "PENÉLOPE", "RENATA", "ROCÍO", "ROSA", "ROMINA", "ROSARIO", "SALOMÉ", "SAMANTHA", "SARA", "SOFÍA", "SOL",
    "TAMARA", "VALENTINA", "VALERIA", "VANIA", "VERÓNICA", "VICTORIA", "VIOLETA", "XIMENA", "YASNA",
    "YOLANDA", "ZOE"
]

# Second names (middle names)
second_names = [
    # Masculine
    "CARLOS", "JOSÉ", "LUIS", "ANTONIO", "MANUEL", "FRANCISCO", "MIGUEL", "RAFAEL", "FERNANDO", "RICARDO",
    "ALBERTO", "EDUARDO", "ALEJANDRO", "ANDRÉS", "ROBERTO", "PEDRO", "DANIEL", "GABRIEL", "DIEGO", "SEBASTIÁN",
    
    # Feminine
    "JOSÉ", "MARÍA", "ISABEL", "CRISTINA", "ELENA", "TERESA", "PATRICIA", "CARMEN", "ROSA", "ANA",
    "LAURA", "BEATRIZ", "ESPERANZA", "GUADALUPE", "DOLORES", "PILAR", "MERCEDES", "SOLEDAD", "AMPARO", "ROCÍO"
]

# Chilean surnames
surnames = [
    "GONZÁLEZ", "MUÑOZ", "ROJAS", "DÍAZ", "PÉREZ", "SOTO", "CONTRERAS", "SILVA", "MARTÍNEZ", "SEPÚLVEDA",
    "MORALES", "RODRÍGUEZ", "LÓPEZ", "ARAYA", "FUENTES", "HERNÁNDEZ", "TORRES", "ESPINOZA", "FLORES",
    "CASTILLO", "REYES", "VALENZUELA", "VARGAS", "RAMÍREZ", "GUTIÉRREZ", "HERRERA", "ÁLVAREZ", "VÁSQUEZ",
    "TAPIA", "SÁNCHEZ", "FERNÁNDEZ", "CARRASCO", "CORTÉS", "GÓMEZ", "JARA", "VERGARA", "RIVERA", "NÚÑEZ",
    "BRAVO", "FIGUEROA", "RIQUELME", "MOLINA", "VERA", "SANDOVAL", "GARCÍA", "VEGA", "MIRANDA", "ROMERO",
    "ORTIZ", "SALAZAR", "CAMPOS", "ORELLANA", "OLIVARES", "GARRIDO", "PARRA", "GALLARDO", "SAAVEDRA",
    "ALARCON", "AGUILERA", "PEÑA", "ZÚÑIGA", "RUIZ", "MEDINA", "GUZMÁN", "ESCOBAR", "NAVARRO", "PIZARRO"
]

# Chilean organizations
organizations = [
    # Insurance Companies
    "METLIFE", "MAPFRE", "LIBERTY", "ALLIANZ", "AXA", "ZURICH", "RSA", "BUPA", "SURA", "COLPATRIA",
    "SEGUROS BOLÍVAR", "SEGUROS SURA", "LIBERTY SEGUROS", "MAPFRE SEGUROS", "CHUBB SEGUROS",
    
    # Banks
    "BANCO DE CHILE", "BANCO SANTANDER", "BANCO BCI", "BANCO ESTADO", "BANCO SECURITY",
    "BANCO FALABELLA", "BANCO RIPLEY", "BANCO CONSORCIO", "BANCO PARIS", "BANCO EDWARDS",
    
    # AFP (Pension Funds)
    "AFP CAPITAL", "AFP CUPRUM", "AFP HABITAT", "AFP PLANVITAL", "AFP PROVIDA", "AFP MODELO",
    
    # Government
    "SUPERINTENDENCIA DE VALORES Y SEGUROS", "SUPERINTENDENCIA DE PENSIONES", 
    "MINISTERIO DE HACIENDA", "BANCO CENTRAL", "FONASA", "ISAPRE", "PREVIRED"
]

# Chilean locations
chilean_locations = [
    # Major cities
    "Santiago", "Valparaíso", "Concepción", "La Serena", "Antofagasta", "Temuco", 
    "Rancagua", "Talca", "Arica", "Chillán", "Los Ángeles", "Calama", "Copiapó",
    "Osorno", "Quillota", "Valdivia", "Punta Arenas", "Puerto Montt", "Iquique",
    
    # Santiago communes
    "Las Condes", "Providencia", "Ñuñoa", "Maipú", "La Florida", "Puente Alto",
    "San Miguel", "Independencia", "Recoleta", "Estación Central", "Pedro Aguirre Cerda",
    "Lo Barnechea", "Vitacura", "La Reina", "Peñalolén", "Macul", "San Joaquín",
    
    # Countries
    "Chile", "Argentina", "Perú", "Bolivia", "Brasil", "Colombia", "Ecuador"
]

# Chilean addresses
chilean_addresses = [
    # Real Santiago streets
    "Agustinas 640", "Providencia 1208", "Apoquindo 3000", "Las Condes 9001",
    "Santa Isabel 0155", "San Diego 1570", "Alameda 950", "Vicuña Mackenna 4860",
    "Avenida Italia 1235", "Pedro de Valdivia 2000", "Manuel Montt 1145",
    "Los Leones 1350", "Nueva Providencia 2214", "Andrés Bello 2777",
    "11 de Septiembre 2155", "Isidora Goyenechea 3000", "El Bosque Norte 500",
    
    # Generic patterns
    "Huérfanos 1234", "Ahumada 312", "Estado 123", "Compañía 1040",
    "Moneda 975", "Teatinos 280", "San Martín 440", "Miraflores 222"
]

# -----------------
# STANDARDIZED ENTITY TYPES FOR SPACY
# -----------------

# Use standard spaCy entity types
SPACY_ENTITY_TYPES = {
    "PERSON",    # Person names
    "LOC",       # Locations  
    "ORG",       # Organizations
    "MISC",      # Miscellaneous (phones, IDs, etc.)
}

# -----------------
# PII CLASSIFICATION - UPDATED
# -----------------

# Document structure elements
DOCUMENT_ELEMENTS = {
    "PERSON": [
        "Traspaso de Cierre Casos cerrados de",
        "Rut Cotizante",
        "Nombres Apellidos",
        "Cliente",
        "Nombre del Solicitante",
        "Datos del Afiliado",
        "Beneficiario",
        "Titular de la Cuenta",
        "Información Personal"
    ],
    "MISC": [
        "Numero Cotización",
        "Número Poliza",
        "Folio Oferta",
        "Cotización externa",
        "Código de Verificación",
        "Número de Solicitud",
        "ID de Transacción",
        "Referencia Operacional"
    ]
}

# -----------------
# GENERATION FUNCTIONS
# -----------------

def generate_name_components(include_second_name: bool = True, probability: float = 0.4, 
                           include_second_surname: bool = True, surname_probability: float = 0.8) -> Tuple[str, str, str]:
    """Generate name components with optional second name and surname."""
    first_name = random.choice(first_names)
    paternal_surname = random.choice(surnames)
    
    # Generate first name part
    if include_second_name and random.random() < probability:
        second_name = random.choice(second_names)
        full_name_part = f"{first_name} {second_name}"
    else:
        full_name_part = first_name
    
    # Generate complete surname
    if include_second_surname and random.random() < surname_probability:
        maternal_surname = random.choice(surnames)
        while maternal_surname == paternal_surname:
            maternal_surname = random.choice(surnames)
        complete_surname = f"{paternal_surname} {maternal_surname}"
    else:
        complete_surname = paternal_surname
    
    return first_name, full_name_part, complete_surname

def random_id(country: str) -> str:
    """Generate Chilean RUT format in various styles."""
    num1 = random.randint(10, 30)
    num2 = random.randint(100, 999)
    num3 = random.randint(100, 999)
    verifier = random.choice([str(random.randint(0, 9)), 'K'])
    formats = [
        f"{num1}.{num2}.{num3}-{verifier}",      # With dots
        f"{num1},{num2},{num3}-{verifier}",      # With commas
        f"{num1}{num2}{num3}-{verifier}",        # Only numbers
    ]
    return random.choice(formats)

def random_phone() -> str:
    """Generate Chilean phone number."""
    formats = [
        "600390300",
        f"+56{random.randint(9,9)}{random.randint(10000000,99999999)}",
        f"{random.randint(2,9)}{random.randint(10000000,99999999)}",
        f"{random.randint(600,999)}{random.randint(100,999)}{random.randint(1000,9999)}"
    ]
    return random.choice(formats)

def random_email(name: str, surname: str) -> str:
    """Generate email address."""
    domains = ["gmail.com", "hotmail.com", "yahoo.com", "outlook.com"]
    first_surname = surname.split()[0] if " " in surname else surname
    return f"{name.lower()}.{first_surname.lower()}@{random.choice(domains)}"

def random_amount() -> str:
    """Generate Chilean peso amount."""
    return f"${random.randint(10000, 900000):,}CLP"

def random_sequence_number() -> str:
    """Generate sequence number."""
    types = [
        str(random.randint(6000000, 9999999)),  # 7-digit
        str(random.randint(400000, 599999)),    # 6-digit
        f"{random.randint(10000, 99999)}-{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"  # With letter
    ]
    return random.choice(types)

def generate_organization() -> str:
    """Generate organization name."""
    return random.choice(organizations)

def generate_document_header_elements():
    """Generate document header elements."""
    headers = [
        "Traspaso de Cierre Casos cerrados de",
        "Cotización de Seguro de Renta Vitalicia",
        "Solicitud de Beneficios Previsionales", 
        "Formulario de Traspaso AFP",
        "Certificado de Saldo de Cuenta",
        "Consulta de Estado de Solicitud"
    ]
    return random.choice(headers)

def generate_form_field_labels():
    """Generate form field labels."""
    labels = [
        "Numero Cotización",
        "Número Poliza",
        "Rut Cotizante", 
        "Nombres Apellidos",
        "Fecha de Nacimiento",
        "Estado Civil",
        "Dirección Particular",
        "Teléfono de Contacto"
    ]
    return random.choice(labels)

def generate_document_footer_elements():
    """Generate document footer elements."""
    footers = [
        "volver subir",
        "Página 1 de 2", 
        "Documento válido por 30 días",
        "Consultas 600390300",
        "www.superintendencia.cl"
    ]
    return random.choice(footers)

# -----------------
# ENTITY VALIDATION AND CLEANING
# -----------------

def clean_entity_text(text: str) -> str:
    """Clean entity text by removing leading/trailing whitespace and punctuation."""
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Remove leading and trailing punctuation (but keep internal punctuation)
    while text and text[0] in string.punctuation:
        text = text[1:]
    while text and text[-1] in string.punctuation:
        text = text[:-1]
    
    # Remove any remaining leading/trailing whitespace
    text = text.strip()
    
    return text

def is_valid_entity_span(text: str, start: int, end: int, entity_text: str) -> bool:
    """Validate that an entity span is valid for spaCy training."""
    # Check bounds
    if start < 0 or end > len(text) or start >= end:
        return False
    
    # Check that the span actually contains the expected text
    actual_text = text[start:end]
    if actual_text != entity_text:
        return False
    
    # Check for leading/trailing whitespace or punctuation
    if actual_text != actual_text.strip():
        return False
    
    # Check for leading/trailing punctuation
    if actual_text and (actual_text[0] in string.punctuation or actual_text[-1] in string.punctuation):
        return False
    
    # Must have some alphanumeric content
    if not any(c.isalnum() for c in actual_text):
        return False
    
    # Must be at least 2 characters
    if len(actual_text) < 2:
        return False
    
    return True

def find_entity_in_text(entity_text: str, full_text: str) -> List[Tuple[int, int]]:
    """Find all positions of an entity in text, returning only valid spans."""
    positions = []
    
    # Clean the entity text first
    clean_entity = clean_entity_text(entity_text)
    if not clean_entity:
        return positions
    
    # Find exact matches
    start = 0
    while True:
        pos = full_text.find(clean_entity, start)
        if pos == -1:
            break
        
        end = pos + len(clean_entity)
        if is_valid_entity_span(full_text, pos, end, clean_entity):
            positions.append((pos, end))
        
        start = pos + 1
    
    # If no exact matches, try case-insensitive
    if not positions:
        start = 0
        while True:
            pos = full_text.lower().find(clean_entity.lower(), start)
            if pos == -1:
                break
            
            end = pos + len(clean_entity)
            actual_text = full_text[pos:end]
            if is_valid_entity_span(full_text, pos, end, actual_text):
                positions.append((pos, end))
            
            start = pos + 1
    
    return positions

# -----------------
# NOISE GENERATION - UPDATED
# -----------------

def add_controlled_noise(text: str, noise_rate: float = 0.1) -> str:
    """Add controlled noise that preserves entity boundaries."""
    if random.random() > noise_rate:
        return text
    
    # Only apply safe transformations
    noise_patterns = [
        # Safe character substitutions
        lambda t: t.replace('ó', 'o').replace('í', 'i').replace('á', 'a'),
        # Safe spacing adjustments (but preserve word boundaries)
        lambda t: re.sub(r':\s*', ': ', t),
        lambda t: re.sub(r'-\s*', '- ', t),
        # Safe punctuation adjustments
        lambda t: t.replace(',', ', ') if ', ' not in t else t,
    ]
    
    # Apply only one transformation
    pattern = random.choice(noise_patterns)
    return pattern(text)

def add_document_structure_noise(text: str) -> str:
    """Add document headers and footers safely."""
    headers = [
        "DOCUMENTO CONFIDENCIAL",
        "SUPERINTENDENCIA DE VALORES Y SEGUROS",
        "FORMULARIO DE SOLICITUD",
        "COMPAÑÍA DE SEGUROS"
    ]
    
    footers = [
        "Página 1 de 2",
        "Este documento es válido por 30 días",
        "Para más información visite www.empresa.cl"
    ]
    
    # Add header with safe separator
    if random.random() < 0.3:
        header = random.choice(headers)
        text = f"{header}\n\n{text}"
    
    # Add footer with safe separator
    if random.random() < 0.3:
        footer = random.choice(footers)
        text = f"{text}\n\n{footer}"
    
    return text

# -----------------
# PII GENERATION - FIXED
# -----------------

def generate_pii_example_with_status(country: str = "CL") -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
    """Generate example with PII classification and proper entity validation."""
    # Generate base data
    first_name, full_name_part, complete_surname = generate_name_components()
    complete_full_name = f"{full_name_part} {complete_surname}"
    
    id_number = random_id(country)
    organization = generate_organization()
    sequence = random_sequence_number()
    
    # Chilean-specific data
    street_address = random.choice(chilean_addresses)
    city = random.choice(chilean_locations)
    phone = random_phone()
    
    # Create simple, clean templates
    templates = [
        "Cliente: {name} RUT: {id} Dirección: {address} Teléfono: {phone} Empresa: {org}",
        "Cotización: {seq} Nombre: {name} ID: {id} Ciudad: {city} Tel: {phone} Org: {org}",
        "Solicitud {seq} Cliente {name} Documento {id} Ubicación {address} Contacto {phone} Entidad {org}",
        "Formulario Cliente: {name} RUT: {id} Dir: {address} Tel: {phone} Empresa: {org} Ref: {seq}",
        "Trámite {seq} Beneficiario: {name} Doc: {id} Ciudad: {city} Teléfono: {phone} Organización: {org}"
    ]
    
    template = random.choice(templates)
    
    # Format template
    sentence = template.format(
        name=complete_full_name,
        id=id_number,
        address=street_address,
        city=city,
        phone=phone,
        org=organization,
        seq=sequence
    )
    
    # Define entities with their expected types
    entity_candidates = [
        (complete_full_name, "PERSON"),
        (id_number, "MISC"),
        (street_address, "LOC"),
        (city, "LOC"),
        (phone, "MISC"),
        (organization, "ORG"),
        (sequence, "MISC")
    ]
    
    # Find valid entities in text
    valid_entities = []
    used_positions = set()
    
    for entity_text, pii_type in entity_candidates:
        if not entity_text or not entity_text.strip():
            continue
        
        positions = find_entity_in_text(entity_text, sentence)
        
        for start, end in positions:
            # Check for overlaps
            position_range = set(range(start, end))
            if not position_range.intersection(used_positions):
                valid_entities.append((start, end, pii_type))
                used_positions.update(position_range)
                break  # Only use first non-overlapping occurrence
    
    # Sort by position
    valid_entities.sort(key=lambda x: x[0])
    
    return sentence, {"entities": valid_entities}

def generate_noisy_pii_example(country: str = "CL", noise_level: str = "medium") -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
    """Generate noisy PII example with corrected annotations."""
    # Generate clean example
    clean_text, clean_annotations = generate_pii_example_with_status(country)
    
    # Apply controlled noise
    noisy_text = clean_text
    
    if noise_level in ["medium", "heavy"]:
        noisy_text = add_controlled_noise(noisy_text, 0.1)
        noisy_text = add_document_structure_noise(noisy_text)
    
    # Re-find entities in noisy text
    corrected_entities = []
    used_positions = set()
    
    for start, end, pii_type in clean_annotations["entities"]:
        original_entity = clean_text[start:end]
        
        # Find entity in noisy text
        positions = find_entity_in_text(original_entity, noisy_text)
        
        for new_start, new_end in positions:
            position_range = set(range(new_start, new_end))
            if not position_range.intersection(used_positions):
                corrected_entities.append((new_start, new_end, pii_type))
                used_positions.update(position_range)
                break
    
    # Sort and validate final entities
    corrected_entities.sort(key=lambda x: x[0])
    final_entities = []
    
    for start, end, pii_type in corrected_entities:
        entity_text = noisy_text[start:end]
        if is_valid_entity_span(noisy_text, start, end, entity_text):
            final_entities.append((start, end, pii_type))
    
    return noisy_text, {"entities": final_entities}

# -----------------
# DATASET CREATION - FIXED
# -----------------

def make_pii_docbin(n_total: int = 100000, noise_distribution: Dict[str, float] = None, output_dir: str = ".") -> Tuple[DocBin, Dict[str, int]]:
    """Create spaCy DocBin with validated PII classification."""
    if noise_distribution is None:
        noise_distribution = {"light": 0.3, "medium": 0.5, "heavy": 0.2}
    
    noise_choices = []
    for level, weight in noise_distribution.items():
        noise_choices.extend([level] * int(weight * 100))
    
    countries = ["CL"]
    
    # Load spaCy model
    try:
        nlp = spacy.load("es_core_news_lg")
        print("✅ Using Spanish Large model (es_core_news_lg)")
    except OSError:
        try:
            nlp = spacy.load("es_core_news_md")
            print("✅ Using Spanish Medium model (es_core_news_md)")
        except OSError:
            try:
                nlp = spacy.load("es_core_news_sm")
                print("✅ Using Spanish Small model (es_core_news_sm)")
            except OSError:
                print("✅ Using blank Spanish model")
                nlp = spacy.blank("es")
    
    # Ensure entity types are added to the model
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add all entity labels
    for entity_type in SPACY_ENTITY_TYPES:
        ner.add_label(entity_type)
    
    db = DocBin()
    
    # Statistics
    country_stats = {c: 0 for c in countries}
    noise_stats = {level: 0 for level in noise_distribution.keys()}
    pii_stats = {et: 0 for et in SPACY_ENTITY_TYPES}
    
    created = 0
    failed_examples = 0
    validation_errors = 0
    
    print(f"🏗️  Generating {n_total:,} PII training examples...")
    print(f"📊 Noise distribution: {noise_distribution}")
    print(f"👤 User: andresveraf")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    while created < n_total:
        country = random.choice(countries)
        noise_level = random.choice(noise_choices)
        
        try:
            text, annotations = generate_noisy_pii_example(country, noise_level)
            
            # Basic validation
            if len(text.strip()) < 10 or len(annotations["entities"]) == 0:
                failed_examples += 1
                continue
            
            # Create spaCy doc
            doc = nlp.make_doc(text)
            spans = []
            entity_validation_passed = True
            
            # Validate and create spans
            for (start, end, pii_type) in annotations["entities"]:
                # Double-check validation
                if not is_valid_entity_span(text, start, end, text[start:end]):
                    validation_errors += 1
                    entity_validation_passed = False
                    break
                
                span = doc.char_span(start, end, label=pii_type, alignment_mode="contract")
                if span is not None:
                    spans.append(span)
                    pii_stats[pii_type] += 1
                else:
                    validation_errors += 1
                    entity_validation_passed = False
                    break
            
            # Only add if all entities are valid
            if entity_validation_passed and spans:
                doc.ents = spans
                db.add(doc)
                
                country_stats[country] += 1
                noise_stats[noise_level] += 1
                created += 1
                
                if created % 5000 == 0:
                    print(f"  ✅ Generated {created:,}/{n_total:,} examples ({created/n_total*100:.1f}%)")
                    print(f"     Time: {datetime.now().strftime('%H:%M:%S')} | Failed: {failed_examples:,} | Validation errors: {validation_errors:,}")
            else:
                failed_examples += 1
                
        except Exception as e:
            failed_examples += 1
            if failed_examples % 1000 == 0:
                print(f"  ⚠️  Failed examples: {failed_examples:,} | Last error: {str(e)[:100]}")
            continue
    
    stats = {
        "total_examples": created,
        "failed_examples": failed_examples,
        "validation_errors": validation_errors,
        "success_rate": f"{created/(created+failed_examples)*100:.1f}%",
        "countries": dict(country_stats),
        "noise_levels": dict(noise_stats),
        "pii_types": dict(pii_stats),
        "noise_distribution": noise_distribution,
        "generation_info": {
            "user": "andresveraf",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S")
        }
    }
    
    print(f"✅ PII dataset generation complete!")
    print(f"📈 Statistics:")
    print(f"   - Total examples: {created:,}")
    print(f"   - Failed examples: {failed_examples:,}")
    print(f"   - Validation errors: {validation_errors:,}")
    print(f"   - Success rate: {created/(created+failed_examples)*100:.1f}%")
    for entity_type, count in pii_stats.items():
        print(f"   - {entity_type} entities: {count:,}")
    
    return db, stats

def create_pii_training_dataset(train_size: int = 100000, dev_size: int = 20000, output_dir: str = ".", save_stats: bool = True) -> None:
    """Create complete PII training and test datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("🚀 Creating LARGE-SCALE PII Training Dataset (FIXED VERSION)")
    print("=" * 70)
    print(f"👤 User: andresveraf")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target: {train_size:,} training + {dev_size:,} test examples")
    print(f"🔧 Improvements:")
    print(f"   - Fixed entity span validation")
    print(f"   - Improved noise handling")
    print(f"   - Standardized entity types")
    print(f"   - Enhanced character position tracking")
    print()
    
    # Optimized distributions for large datasets
    train_noise_dist = {"light": 0.15, "medium": 0.70, "heavy": 0.15}
    test_noise_dist = {"light": 0.30, "medium": 0.50, "heavy": 0.20}
    
    # Generate training set
    print("📚 TRAINING SET GENERATION")
    print("-" * 40)
    start_time = datetime.now()
    
    train_db, train_stats = make_pii_docbin(
        n_total=train_size, 
        noise_distribution=train_noise_dist,
        output_dir=output_dir
    )
    train_file = output_path / "train_pii.spacy"
    train_db.to_disk(train_file)
    
    train_time = datetime.now() - start_time
    print(f"✅ Training set completed in: {train_time}")
    
    # Generate test set  
    print(f"\n🔬 TEST SET GENERATION")
    print("-" * 40)
    start_time = datetime.now()
    
    test_db, test_stats = make_pii_docbin(
        n_total=dev_size,
        noise_distribution=test_noise_dist, 
        output_dir=output_dir
    )
    test_file = output_path / "test_pii.spacy"
    test_db.to_disk(test_file)
    
    test_time = datetime.now() - start_time
    print(f"✅ Test set completed in: {test_time}")
    
    # Save statistics
    if save_stats:
        combined_stats = {
            "dataset_info": {
                "purpose": "Large-scale PII Classification Training (FIXED)",
                "description": "Chilean document PII detection with validated entity spans",
                "user": "andresveraf",
                "creation_date": datetime.now().strftime("%Y-%m-%d"),
                "creation_time": datetime.now().strftime("%H:%M:%S"),
                "train_size": train_size,
                "test_size": dev_size,
                "total_size": train_size + dev_size,
                "countries": ["Chile"],
                "languages": ["Spanish"],
                "pii_types": list(SPACY_ENTITY_TYPES),
                "fixes_applied": [
                    "Entity span validation",
                    "Controlled noise application", 
                    "Standardized entity types",
                    "Character position correction"
                ]
            },
            "train_stats": train_stats,
            "test_stats": test_stats,
            "train_noise_distribution": train_noise_dist,
            "test_noise_distribution": test_noise_dist,
            "file_info": {
                "train_file": str(train_file),
                "test_file": str(test_file),
                "train_file_size_mb": f"{train_file.stat().st_size / (1024*1024):.1f}" if train_file.exists() else "N/A",
                "test_file_size_mb": f"{test_file.stat().st_size / (1024*1024):.1f}" if test_file.exists() else "N/A"
            }
        }
        
        stats_file = output_path / "pii_dataset_stats_fixed.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Dataset Statistics saved to: {stats_file}")
    
    total_time = train_time + test_time
    
    print(f"\n🎯 FIXED LARGE-SCALE DATASET CREATED:")
    print(f"   📁 Training set: {train_file} ({train_size:,} examples)")
    print(f"   📁 Test set: {test_file} ({dev_size:,} examples)")
    print(f"   📁 Total examples: {train_size + dev_size:,}")
    print(f"   ⏱️  Total generation time: {total_time}")
    
    # File size information
    if train_file.exists() and test_file.exists():
        train_size_mb = train_file.stat().st_size / (1024*1024)
        test_size_mb = test_file.stat().st_size / (1024*1024)
        total_size_mb = train_size_mb + test_size_mb
        
        print(f"   💾 File sizes:")
        print(f"      - Training: {train_size_mb:.1f} MB")
        print(f"      - Test: {test_size_mb:.1f} MB") 
        print(f"      - Total: {total_size_mb:.1f} MB")
    
    print(f"\n🔧 TRAINING COMMANDS:")
    print(f"   1. Validate data: python -m spacy debug data {train_file} {test_file} --verbose")
    print(f"   2. Create config: python -m spacy init config config.cfg --lang es --pipeline ner --optimize accuracy")
    print(f"   3. Train model: python -m spacy train config.cfg --output ./pii_model --paths.train {train_file} --paths.dev {test_file}")
    
    print(f"\n✨ Fixed dataset ready for training without E024 errors!")

# -----------------
# EXCEL EXPORT - UPDATED
# -----------------

def export_pii_to_excel(n_examples: int = 500, output_file: str = "pii_data_review_fixed.xlsx", countries: Optional[List[str]] = None) -> None:
    """Export PII training data to Excel for review."""
    if countries is None:
        countries = ["CL"]
    
    print(f"📊 Generating {n_examples} PII examples for Excel review (FIXED VERSION)...")
    
    all_data = []
    pii_stats = {et: 0 for et in SPACY_ENTITY_TYPES}
    
    examples_per_country = n_examples // len(countries)
    noise_levels = ["light", "medium", "heavy"]
    
    for country in countries:
        print(f"   🌍 Generating {examples_per_country} examples for {country}...")
        
        for i in range(examples_per_country):
            clean_text, clean_annotations = generate_pii_example_with_status(country)
            
            noise_level = noise_levels[i % len(noise_levels)]
            noisy_text, noisy_annotations = generate_noisy_pii_example(country, noise_level)
            
            # Extract PII entities
            clean_pii_entities = {et: [] for et in SPACY_ENTITY_TYPES}
            noisy_pii_entities = {et: [] for et in SPACY_ENTITY_TYPES}
            
            for start, end, pii_type in clean_annotations["entities"]:
                entity_text = clean_text[start:end]
                clean_pii_entities[pii_type].append(entity_text)
                pii_stats[pii_type] += 1

            for start, end, pii_type in noisy_annotations["entities"]:
                entity_text = noisy_text[start:end]
                noisy_pii_entities[pii_type].append(entity_text)

            row_data = {
                "Example_ID": f"{country}-{noise_level}-{i+1:03d}",
                "Country": country,
                "Noise_Level": noise_level,
                "Clean_Text": clean_text,
                "Noisy_Text": noisy_text,
                "Clean_Entity_Count": len(clean_annotations["entities"]),
                "Noisy_Entity_Count": len(noisy_annotations["entities"]),
                "Detection_Accuracy": len(noisy_annotations["entities"]) / len(clean_annotations["entities"]) * 100 if clean_annotations["entities"] else 0,
                "Generated_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add entity columns for all standardized types
            for et in SPACY_ENTITY_TYPES:
                row_data[f"Clean_{et}"] = "; ".join(clean_pii_entities[et])
                row_data[f"Noisy_{et}"] = "; ".join(noisy_pii_entities[et])

            all_data.append(row_data)
    
    # Create Excel file
    print(f"   📝 Creating PII analysis Excel file: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Summary Sheet
        summary_metrics = [
            "Total Examples Generated",
            "User",
            "Generation Date/Time",
            "Dataset Version",
            "Average Detection Accuracy"
        ]
        summary_values = [
            len(all_data),
            "andresveraf",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "FIXED - Entity validation applied",
            f"{sum(d['Detection_Accuracy'] for d in all_data) / len(all_data):.1f}%"
        ]
        
        # Add entity counts
        for et in SPACY_ENTITY_TYPES:
            summary_metrics.append(f"{et} Entities")
            summary_values.append(pii_stats[et])
        summary_metrics.append("Total Entities")
        summary_values.append(sum(pii_stats.values()))
        
        summary_data = {"Metric": summary_metrics, "Value": summary_values}
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Complete dataset
        all_df = pd.DataFrame(all_data)
        all_df.to_excel(writer, sheet_name='Complete_Dataset', index=False)

        # Entity Distribution
        entity_analysis = []
        total_entities = sum(pii_stats.values())
        for et in SPACY_ENTITY_TYPES:
            count = pii_stats[et]
            entity_analysis.append({
                "PII_Type": et,
                "Count": count,
                "Percentage": f"{count/total_entities*100:.1f}%" if total_entities > 0 else "0%"
            })
        entity_df = pd.DataFrame(entity_analysis)
        entity_df.to_excel(writer, sheet_name='Entity_Distribution', index=False)
    
    print(f"✅ PII Excel file created: {output_file}")
    print(f"📋 File contains {len(all_data)} examples with validated PII classification")
    print(f"🔍 PII Distribution:")
    for pii_type in SPACY_ENTITY_TYPES:
        count = pii_stats[pii_type]
        total = sum(pii_stats.values())
        percentage = count/total*100 if total > 0 else 0
        print(f"   - {pii_type}: {count} entities ({percentage:.1f}%)")

# -----------------
# MAIN FUNCTION - UPDATED
# -----------------

def main():
    """Main function for PII training data generation (FIXED VERSION)."""
    parser = argparse.ArgumentParser(
        description="Complete PII Training Data Generator for Chilean Documents (FIXED)",
        epilog="""
Examples:
  # Generate 100K training + 20K test (FIXED)
  python %(prog)s --mode create-dataset --train-size 100000 --dev-size 20000
  
  # Create Excel review (FIXED)
  python %(prog)s --mode excel-export --excel-examples 1000
  
  # See examples
  python %(prog)s --mode demo
  
FIXES APPLIED:
- Entity span validation (no leading/trailing whitespace/punctuation)
- Controlled noise application
- Standardized entity types for spaCy compatibility
- Character position correction after noise
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--mode", choices=["demo", "excel-export", "create-dataset"], default="demo",
                       help="Mode: 'demo' shows examples, 'excel-export' creates review file, 'create-dataset' generates training data")
    parser.add_argument("--train-size", type=int, default=100000, help="Training set size (default: 100,000)")
    parser.add_argument("--dev-size", type=int, default=20000, help="Test set size (default: 20,000)")
    parser.add_argument("--excel-examples", type=int, default=500, help="Number of examples for Excel export")
    parser.add_argument("--excel-file", type=str, default="pii_data_review_fixed.xlsx", help="Excel output filename")
    parser.add_argument("--output-dir", type=str, default="pii_training_large_fixed", help="Output directory")
    
    args = parser.parse_args()
    
    # Print header
    print("🚀 COMPLETE PII TRAINING DATA GENERATOR (FIXED VERSION)")
    print("=" * 70)
    print(f"👤 User: andresveraf")
    print(f"📅 Date: 2025-08-25 21:43:49 UTC")
    print(f"🎯 Mode: {args.mode}")
    print(f"🔧 Status: FIXED - No more E024 errors!")
    print()
    
    if args.mode == "create-dataset":
        create_pii_training_dataset(
            train_size=args.train_size,
            dev_size=args.dev_size, 
            output_dir=args.output_dir
        )
        
        # Automatically export detailed Excel review after dataset creation
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        detailed_excel_file = output_path / "detailed_review_fixed.xlsx"
        
        print(f"\n📊 Creating detailed Excel review...")
        print(f"📁 File: {detailed_excel_file}")
        
        export_pii_to_excel(
            n_examples=1000,
            output_file=str(detailed_excel_file)
        )
        
    elif args.mode == "excel-export":
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        excel_file = output_path / args.excel_file
        
        export_pii_to_excel(
            n_examples=args.excel_examples,
            output_file=str(excel_file)
        )
        
    else:
        # Demo mode
        print("🎯 PII CLASSIFICATION DEMONSTRATION (FIXED)")
        print("-" * 50)
        
        for i in range(3):
            print(f"\n📍 Example {i+1}")
            print("-" * 30)
            
            text, annotations = generate_pii_example_with_status("CL")
            print(f"Text: {text}")
            print("\nPII Classification:")
            print("PII_Type\tStart\tEnd\tPII_Value\tValid")
            print("-" * 50)
            
            for start, end, pii_type in annotations["entities"]:
                entity_text = text[start:end]
                is_valid = is_valid_entity_span(text, start, end, entity_text)
                print(f"{pii_type}\t{start}\t{end}\t{entity_text}\t{is_valid}")
            print()

if __name__ == "__main__":
    main()
