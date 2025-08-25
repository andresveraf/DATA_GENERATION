"""
Complete PII Training Data Generator for Chilean Documents
User: andresveraf
Date: 2025-08-25 19:19:12 UTC

This script generates large-scale training datasets for PII (Personally Identifiable Information)
detection in Chilean government and financial documents with realistic noise patterns.

Features:
- 100K+ training examples generation
- PII classification (PER, LOC, PHONE, MISC)
- Status tracking (OK/NO)
- Realistic Chilean document noise
- Excel export for validation
- spaCy DocBin format for training

Usage:
    # Generate 100K training + 20K test
    python pii_training_generator_complete.py --mode create-dataset --train-size 100000 --dev-size 20000
    
    # Create Excel review
    python pii_training_generator_complete.py --mode excel-export --excel-examples 1000
    
    # See examples
    python pii_training_generator_complete.py --mode demo
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
# PII CLASSIFICATION
# -----------------

# Document structure elements
DOCUMENT_ELEMENTS = {
    "PER": [
        "Traspaso de Cierre Casos cerrados de",
        "I Rut Cotizante I",
        "Nombres [:l Apellidos ] E",
        "Cliente:",
        "Nombre del Solicitante",
        "Datos del Afiliado",
        "Beneficiario:",
        "Titular de la Cuenta",
        "Información Personal"
    ],
    "MISC": [
        "Scomp Número Cotización",
        "Número Poliza",
        "Folio Oferta SCOMP",
        "N de Cotización externa",
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

# def random_id(country: str) -> str:
#     """Generate Chilean RUT format."""
#     return f"{random.randint(10,30)}.{random.randint(100,999)}.{random.randint(100,999)}-{random.randint(0,9)}"
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
        "600 390 3000",
        f"+56 9 {random.randint(1000,9999)} {random.randint(1000,9999)}",
        f"({random.randint(2,9)}) {random.randint(1000,9999)} {random.randint(1000,9999)}",
        f"{random.randint(600,999)} {random.randint(100,999)} {random.randint(1000,9999)}"
    ]
    return random.choice(formats)

def random_email(name: str, surname: str) -> str:
    """Generate email address."""
    domains = ["gmail.com", "hotmail.com", "yahoo.com", "outlook.com"]
    first_surname = surname.split()[0] if " " in surname else surname
    return f"{name.lower()}.{first_surname.lower()}@{random.choice(domains)}"

def random_amount() -> str:
    """Generate Chilean peso amount."""
    return f"${random.randint(10_000, 900_000):,} CLP"

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
        "Scomp Número Cotización",
        "Número Poliza",
        "I Rut Cotizante I", 
        "Nombres [:l Apellidos ] E",
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
        "Consultas: 600 390 3000",
        "www.superintendencia.cl"
    ]
    return random.choice(footers)

# -----------------
# NOISE GENERATION
# -----------------

def add_ocr_errors(text: str, error_rate: float = 0.02) -> str:
    """Add realistic OCR errors."""
    if random.random() > error_rate:
        return text
        
    ocr_substitutions = {
        '0': ['O'], '1': ['l', 'I'], '5': ['S'], '6': ['G'], '8': ['B'],
        'O': ['0'], 'I': ['1', 'l'], 'S': ['5'], 'B': ['8'],
        'rn': ['m'], 'm': ['rn'], 'cl': ['d'], '.': [','], ',': ['.']
    }
    
    result = text
    for original, replacements in ocr_substitutions.items():
        if original in result and random.random() < 0.3:
            replacement = random.choice(replacements)
            result = result.replace(original, replacement, 1)
    
    return result

def add_spacing_noise(text: str, noise_rate: float = 0.15) -> str:
    """Add spacing irregularities."""
    if random.random() > noise_rate:
        return text
        
    noise_patterns = [
        lambda t: re.sub(r'\s+', '  ', t),
        lambda t: re.sub(r'([,.;:])(\w)', r'\1\2', t),
        lambda t: re.sub(r'(\w)([,.;:])', r'\1 \2 ', t),
        lambda t: re.sub(r'(\d)([A-Za-z])', r'\1\2', t),
        lambda t: t.replace(' ', '\t', 1) if ' ' in t else t
    ]
    
    pattern = random.choice(noise_patterns)
    return pattern(text)

def add_document_structure_noise(text: str) -> str:
    """Add document headers and footers."""
    headers = [
        "Preview X BORRADOR", "DOCUMENTO CONFIDENCIAL", "SUPERINTENDENCIA DE VALORES Y SEGUROS",
        "FORMULARIO DE SOLICITUD", "COMPAÑÍA DE SEGUROS", "COTIZACIÓN DE SEGURO",
        "IMPORTANTE: No acepte ofrecimientos de dinero", "VÁLIDO HASTA"
    ]
    
    footers = [
        "Imprimir | Zoom + | Cerrar", "Página 1 de 2", "Este documento es válido por 30 días",
        "Para más información visite www.empresa.cl", "Documento generado automáticamente"
    ]
    
    if random.random() < 0.6:
        header = random.choice(headers)
        text = f"{header} {text}"
    
    if random.random() < 0.4:
        footer = random.choice(footers)
        text = f"{text} {footer}"
    
    return text

def add_realistic_document_noise(text: str) -> str:
    """Add Chilean document-specific noise."""
    noise_patterns = [
        lambda t: t.replace(":", " : ").replace("]", " ] ").replace("[", " [ "),
        lambda t: re.sub(r'(\w)(\d)', r'\1 \2', t),
        lambda t: t.replace(" ", "  ") if random.random() < 0.3 else t,
        lambda t: t.replace("Número", "Numero").replace("Cotización", "Cotizacion"),
        lambda t: t.replace("I", "I:") if "Rut" in t else t,
        lambda t: t + " E" if "Apellidos" in t else t
    ]
    
    num_patterns = random.randint(1, 2)
    selected_patterns = random.sample(noise_patterns, num_patterns)
    
    for pattern in selected_patterns:
        text = pattern(text)
    
    return text

def add_table_structure_noise(text: str) -> str:
    """Add table artifacts."""
    if random.random() < 0.4:
        table_elements = [
            "Usuario:", "Ambiente de P", "Backoffice", "Administrador", "Cotización",
            "_Toato_", "ON", "onSión:", "Permión", "O vll", "DIF VA NO", "Cod",
            "NM VA NO", "INM VN NO", "DIF VN NO", "Ver Cotización", "Excel", "Cerrar"
        ]
        
        noise = random.choice(table_elements)
        words = text.split()
        if len(words) > 1:
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, noise)
            text = " ".join(words)
    
    return text

# -----------------
# PII GENERATION
# -----------------

def generate_pii_example_with_status(country: str = "CL") -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
    """Generate example with PII classification and status tracking."""
    # Generate base data
    first_name, full_name_part, complete_surname = generate_name_components()
    complete_full_name = f"{full_name_part} {complete_surname}"
    
    id_number = random_id(country)
    organization = generate_organization()
    sequence = random_sequence_number()
    amount = random_amount()
    
    # Chilean-specific data
    street_address = random.choice(chilean_addresses)
    city = random.choice(chilean_locations)
    country_name = "Chile"
    full_address = f"{street_address}, {city}, {country_name}"
    phone = random_phone()
    email = random_email(first_name, complete_surname)
    
    # Document elements
    header_element = generate_document_header_elements()
    form_label = generate_form_field_labels()
    footer_element = generate_document_footer_elements()
    
    # Create document templates with consistent formatting - 20 diverse templates
    templates = [
        # Original templates (fixed to 8 placeholders)
        "{} Número: {} Cliente: {} RUT: {} Dirección: {} Tel: {} Empresa: {}",
        "{} Cotización: {} Nombre: {} ID: {} Ubicación: {} Contacto: {} Org: {}",
        "{} ref: {} - {} ({}) en {} - {} Tel: {} - {}",
        "{} Folio: {} Cliente: {} RUT: {} Dir: {} Tel: {} - {}",
        "{} ID: {} Titular: {} Documento: {} Ubicación: {} Teléfono: {} Compañía: {}",
        
        # New diverse templates (all with 8 placeholders)
        "SOLICITUD {} N°{}: CLIENTE {} DOC {} DOMICILIO {} FONO {} ENTIDAD {} - {}",
        "Formulario {} Código: {} Solicitante: {} Cédula: {} Residencia: {} Teléfono: {} Institución: {} Ref: {}",
        "{} | Expediente {} | Titular: {} | RUT: {} | Dirección: {} | Tel: {} | Organismo: {} | {}",
        "DOCUMENTO {} REF: {} NOMBRE: {} IDENTIFICACIÓN: {} LOCALIDAD: {} CONTACTO: {} EMPRESA: {} ANEXO: {}",
        "{} - Trámite {} - Beneficiario: {} - Doc: {} - Ubicación: {} - Tel: {} - Entidad: {} - {}",
        "CERTIFICADO {} NUM {} PERSONA {} CEDULA {} DIRECCION {} TELEFONO {} COMPANIA {} OBSERVACIONES {}",
        "{} Proceso: {} Cliente: {} RUT: {} Domicilio: {} Fono: {} Organización: {} Estado: {}",
        "FORM {} COD: {} APELLIDOS Y NOMBRES: {} DOC IDENT: {} DIREC: {} TEL: {} INSTITUCION: {} FECHA: {}",
        "{} - {} | {} | {} | {} | {} | {} | {}",
        "Expediente {} Número {} Nombre Completo {} Documento {} Dirección Residencia {} Teléfono {} Entidad {} Final {}",
        "[{}] REF:{} CLIENTE:{} RUT:{} DIRECCION:{} TELEFONO:{} EMPRESA:{} [{}]",
        "TRÁMITE {} COD {} || {} || {} || {} || {} || {} || {}",
        "{} N° {} / {} / {} / {} / {} / {} / {}",
        "═══ {} ═══ {} ═══ {} ═══ {} ═══ {} ═══ {} ═══ {} ═══ {}",
        "DOCUMENTO: {} | CÓDIGO: {} | NOMBRE: {} | ID: {} | DIRECCIÓN: {} | TEL: {} | ORG: {} | NOTA: {}"
    ]
    
    template = random.choice(templates)
    
    # Format template with 7-8 arguments consistently
    sentence = template.format(
        header_element, sequence, complete_full_name, id_number, 
        street_address, phone, organization, footer_element
    )
    
    # Define entities with PII classification
    entity_candidates = [
        (complete_full_name, "PER"),       # Customer names
        (id_number, "ID_NUMBER"),          # ID numbers
        (street_address, "LOC"),           # Street addresses
        (city, "LOC"),                     # Cities
        (country_name, "LOC"),             # Countries
        (full_address, "LOC"),             # Full addresses
        (phone, "PHONE"),                  # Phone numbers
        (email, "EMAIL"),                  # Emails
        (organization, "ORG"),             # Organizations
        (sequence, "SEQ_NUMBERS"),         # Sequence numbers
        (amount, "AMOUNT"),                # Amounts
        (header_element, "MISC"),          # Document headers
        (form_label, "MISC"),              # Form labels
        (footer_element, "MISC")           # Footer elements
    ]
    
    # Find positions in text
    entities_with_pii = []
    used_positions = set()
    
    for entity_text, pii_type in entity_candidates:
        if not entity_text.strip():
            continue
            
        start_pos = sentence.find(entity_text)
        if start_pos != -1:
            end_pos = start_pos + len(entity_text)
            position_range = set(range(start_pos, end_pos))
            
            if not position_range.intersection(used_positions):
                entities_with_pii.append((start_pos, end_pos, pii_type))
                used_positions.update(position_range)
    
    entities_with_pii.sort(key=lambda x: x[0])
    
    return sentence, {"entities": entities_with_pii}

def generate_noisy_pii_example(country: str = "CL", noise_level: str = "medium") -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
    """Generate noisy PII example."""
    # Generate clean example
    clean_text, clean_annotations = generate_pii_example_with_status(country)
    
    # Apply noise
    noisy_text = clean_text
    
    if noise_level in ["medium", "heavy"]:
        noisy_text = add_realistic_document_noise(noisy_text)
        noisy_text = add_document_structure_noise(noisy_text)
        noisy_text = add_table_structure_noise(noisy_text)
        noisy_text = add_spacing_noise(noisy_text, 0.2)
    
    if noise_level == "heavy":
        noisy_text = add_ocr_errors(noisy_text, 0.05)
        noisy_text = add_spacing_noise(noisy_text, 0.4)
    
    # Correct annotations
    corrected_annotations = correct_pii_annotations_after_noise(clean_text, noisy_text, clean_annotations)
    
    return noisy_text, corrected_annotations

def correct_pii_annotations_after_noise(original_text: str, noisy_text: str, original_annotations: Dict) -> Dict:
    """Correct PII annotations after noise."""
    corrected_entities = []
    
    for start, end, pii_type in original_annotations["entities"]:
        original_entity = original_text[start:end]
        found_positions = find_pii_entity_in_noisy_text(original_entity, noisy_text, pii_type)
        
        if found_positions:
            for new_start, new_end in found_positions:
                corrected_entities.append((new_start, new_end, pii_type))
    
    corrected_entities = remove_overlapping_pii_entities(corrected_entities)
    return {"entities": corrected_entities}

def find_pii_entity_in_noisy_text(entity: str, noisy_text: str, pii_type: str) -> List[Tuple[int, int]]:
    """Find PII entities in noisy text."""
    positions = []
    
    # Exact match
    start = noisy_text.find(entity)
    if start != -1:
        positions.append((start, start + len(entity)))
        return positions
    
    # Case-insensitive match
    start = noisy_text.lower().find(entity.lower())
    if start != -1:
        actual_entity = noisy_text[start:start + len(entity)]
        positions.append((start, start + len(actual_entity)))
        return positions
    
    # Type-specific patterns
    if pii_type == "LOC":
        loc_patterns = [
            r'\bSantiago\b', r'\bChile\b', r'\bAgustinas\s+\d+', 
            r'\b\w+\s+\d{2,4}\b', r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        ]
        for pattern in loc_patterns:
            matches = re.finditer(pattern, noisy_text)
            for match in matches:
                if entity.upper() in match.group().upper() or match.group().upper() in entity.upper():
                    positions.append((match.start(), match.end()))
    
    elif pii_type == "PHONE":
        phone_patterns = [
            r'\b600\s*390\s*3000\b', r'\b\d{3}\s*\d{3}\s*\d{4}\b',
            r'\+56\s*\d+\s*\d+\s*\d+', r'\b\d{3}[-\s]*\d{3}[-\s]*\d{4}\b'
        ]
        for pattern in phone_patterns:
            matches = re.finditer(pattern, noisy_text)
            for match in matches:
                positions.append((match.start(), match.end()))
    
    elif pii_type == "PER":
        if "Rut" in entity or "RUT" in entity:
            rut_patterns = [r'I\s*Rut\s*Cotizante\s*I', r'RUT\s*:', r'Rut\s*Cotizante']
            for pattern in rut_patterns:
                matches = re.finditer(pattern, noisy_text, re.IGNORECASE)
                for match in matches:
                    positions.append((match.start(), match.end()))
        elif "Nombre" in entity:
            name_patterns = [r'Nombres\s*\[:l\s*Apellidos\s*\]\s*E', r'Nombres.*Apellidos']
            for pattern in name_patterns:
                matches = re.finditer(pattern, noisy_text, re.IGNORECASE)
                for match in matches:
                    positions.append((match.start(), match.end()))
        else:
            # For actual names, try partial matching
            words = entity.split()
            if len(words) > 1:
                for i in range(len(words) - 1):
                    partial = " ".join(words[i:i+2])
                    start = noisy_text.find(partial)
                    if start != -1:
                        end = start + len(partial)
                        positions.append((start, end))
                        break
    
    elif pii_type == "MISC":
        misc_patterns = [
            r'Scomp\s*Número\s*Cotización', r'Número\s*Poliza',
            r'\b\d{6,8}\b', r'\$[\d,.:]+', entity
        ]
        for pattern in misc_patterns:
            matches = re.finditer(pattern, noisy_text, re.IGNORECASE)
            for match in matches:
                if entity.upper() in match.group().upper() or match.group().upper() in entity.upper():
                    positions.append((match.start(), match.end()))
    
    return positions

def remove_overlapping_pii_entities(entities: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Remove overlapping PII entities."""
    if not entities:
        return entities
        
    sorted_entities = sorted(entities, key=lambda x: x[0])
    non_overlapping = []
    
    for start, end, pii_type in sorted_entities:
        overlaps = False
        for existing_start, existing_end, _ in non_overlapping:
            if (start < existing_end and end > existing_start):
                if (end - start) > (existing_end - existing_start):
                    non_overlapping = [(s, e, t) for s, e, t in non_overlapping 
                                     if not (s == existing_start and e == existing_end)]
                    break
                else:
                    overlaps = True
                    break
        
        if not overlaps:
            non_overlapping.append((start, end, pii_type))
    
    return sorted(non_overlapping, key=lambda x: x[0])

# -----------------
# DATASET CREATION
# -----------------

def make_pii_docbin(n_total: int = 100000, noise_distribution: Dict[str, float] = None, output_dir: str = ".") -> Tuple[DocBin, Dict[str, int]]:
    """Create spaCy DocBin with PII classification."""
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
                print("⚠️  No Spanish models found, using blank model")
                nlp = spacy.blank("es")
    
    db = DocBin()
    
    # Statistics
    country_stats = {c: 0 for c in countries}
    noise_stats = {level: 0 for level in noise_distribution.keys()}
    # Dynamic entity type statistics
    pii_stats = {}
    
    created = 0
    failed_examples = 0
    
    print(f"🏗️  Generating {n_total:,} PII training examples...")
    print(f"📊 Noise distribution: {noise_distribution}")
    print(f"👤 User: andresveraf")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    while created < n_total:
        country = random.choice(countries)
        noise_level = random.choice(noise_choices)
        
        try:
            text, annotations = generate_noisy_pii_example(country, noise_level)
            
            if len(text.strip()) < 10 or len(annotations["entities"]) == 0:
                failed_examples += 1
                continue
            
            doc = nlp.make_doc(text)
            spans = []
            
            for (start, end, pii_type) in annotations["entities"]:
                if 0 <= start < end <= len(text):
                    span = doc.char_span(start, end, label=pii_type, alignment_mode="contract")
                    if span is not None:
                        spans.append(span)
                        # Dynamically add new entity types to stats
                        if pii_type not in pii_stats:
                            pii_stats[pii_type] = 0
                        pii_stats[pii_type] += 1
            
            if spans:
                doc.ents = spans
                db.add(doc)
                
                country_stats[country] += 1
                noise_stats[noise_level] += 1
                created += 1
                
                if created % 5000 == 0:
                    print(f"  ✅ Generated {created:,}/{n_total:,} examples ({created/n_total*100:.1f}%)")
                    print(f"     Time: {datetime.now().strftime('%H:%M:%S')} | Failed: {failed_examples:,}")
            else:
                failed_examples += 1
                
        except Exception as e:
            failed_examples += 1
            if failed_examples % 1000 == 0:
                print(f"  ⚠️  Failed examples: {failed_examples:,}")
            continue
    
    stats = {
        "total_examples": created,
        "failed_examples": failed_examples,
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
    print(f"   - Success rate: {created/(created+failed_examples)*100:.1f}%")
    for entity_type, count in pii_stats.items():
        print(f"   - {entity_type} entities: {count:,}")
    
    return db, stats

def create_pii_training_dataset(train_size: int = 100000, dev_size: int = 20000, output_dir: str = ".", save_stats: bool = True) -> None:
    """Create complete PII training and test datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("🚀 Creating LARGE-SCALE PII Training Dataset")
    print("=" * 60)
    print(f"👤 User: andresveraf")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target: {train_size:,} training + {dev_size:,} test examples")
    print()
    
    # train_noise_dist = {"light": 0.2, "medium": 0.6, "heavy": 0.2}
    # test_noise_dist = {"light": 0.4, "medium": 0.4, "heavy": 0.2}
     # Optimized distributions for large datasets
    train_noise_dist = {"light": 0.15, "medium": 0.70, "heavy": 0.15}  # More medium noise
    test_noise_dist = {"light": 0.30, "medium": 0.50, "heavy": 0.20}   # Balanced test set
    
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
                "purpose": "Large-scale PII Classification Training",
                "description": "Chilean document PII detection with realistic noise patterns",
                "user": "andresveraf",
                "creation_date": datetime.now().strftime("%Y-%m-%d"),
                "creation_time": datetime.now().strftime("%H:%M:%S"),
                "train_size": train_size,
                "test_size": dev_size,
                "total_size": train_size + dev_size,
                "countries": ["Chile"],
                "languages": ["Spanish"],
                "pii_types": ["PER (Person)", "LOC (Location)", "PHONE (Phone)", "MISC (Miscellaneous)"]
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
        
        stats_file = output_path / "pii_dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Dataset Statistics saved to: {stats_file}")
    
    total_time = train_time + test_time
    
    print(f"\n🎯 LARGE-SCALE DATASET CREATED:")
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
    print(f"   1. Install spaCy: pip install spacy")
    print(f"   2. Download model: python -m spacy download es_core_news_lg")
    print(f"   3. Create config: python -m spacy init config config.cfg --lang es --pipeline ner --optimize accuracy")
    print(f"   4. Train model: python -m spacy train config.cfg --output ./pii_model --paths.train {train_file} --paths.dev {test_file}")
    
    print(f"\n🎯 TRAINING OPTIMIZATION:")
    print(f"   - For 100K+ examples, use GPU: --gpu-id 0")
    print(f"   - Increase batch size: --training.batcher.size 2000")
    print(f"   - Monitor performance: --training.eval_frequency 1000")
    
    print(f"\n✨ Ready for large-scale PII training!")

# -----------------
# EXCEL EXPORT
# -----------------

def export_pii_to_excel(n_examples: int = 500, output_file: str = "pii_data_review.xlsx", countries: Optional[List[str]] = None) -> None:
    """Export PII training data to Excel for review."""
    if countries is None:
        countries = ["CL"]
    
    print(f"📊 Generating {n_examples} PII examples for Excel review...")
    
    all_data = []
    pii_stats = {}
    
    examples_per_country = n_examples // len(countries)
    noise_levels = ["light", "medium", "heavy"]
    
    for country in countries:
        print(f"   🌍 Generating {examples_per_country} examples for {country}...")
        
        for i in range(examples_per_country):
            clean_text, clean_annotations = generate_pii_example_with_status(country)
            
            noise_level = noise_levels[i % len(noise_levels)]
            noisy_text, noisy_annotations = generate_noisy_pii_example(country, noise_level)
            
            # Extract PII entities
            clean_pii_entities = {}
            noisy_pii_entities = {}
            
            for start, end, pii_type in clean_annotations["entities"]:
                entity_text = clean_text[start:end]
                if pii_type not in clean_pii_entities:
                    clean_pii_entities[pii_type] = []
                clean_pii_entities[pii_type].append(entity_text)
                if pii_type not in pii_stats:
                    pii_stats[pii_type] = 0
                pii_stats[pii_type] += 1

            for start, end, pii_type in noisy_annotations["entities"]:
                entity_text = noisy_text[start:end]
                if pii_type not in noisy_pii_entities:
                    noisy_pii_entities[pii_type] = []
                noisy_pii_entities[pii_type].append(entity_text)

            # Dynamically add all entity types to row_data
            all_entity_types = set(list(clean_pii_entities.keys()) + list(noisy_pii_entities.keys()))
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
            for et in all_entity_types:
                row_data[f"Clean_{et}"] = "; ".join(clean_pii_entities.get(et, []))
                row_data[f"Noisy_{et}"] = "; ".join(noisy_pii_entities.get(et, []))

            all_data.append(row_data)
    
    # Create Excel file
    print(f"   📝 Creating PII analysis Excel file: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Summary Sheet
        summary_metrics = [
            "Total Examples Generated",
            "User",
            "Generation Date/Time",
            "Average Detection Accuracy"
        ]
        summary_values = [
            len(all_data),
            "andresveraf",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{sum(d['Detection_Accuracy'] for d in all_data) / len(all_data):.1f}%"
        ]
        # Add entity counts for all types
        for et in pii_stats.keys():
            summary_metrics.append(f"{et} Entities")
            summary_values.append(pii_stats[et])
        summary_metrics.append("Total Entities")
        summary_values.append(sum(pii_stats.values()))
        summary_data = {"Metric": summary_metrics, "Value": summary_values}
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # PII Analysis Sheet (first 50 examples, all entity types)
        pii_analysis = []
        all_entity_types = list(pii_stats.keys())
        for d in all_data[:50]:
            for et in all_entity_types:
                key = f"Noisy_{et}"
                if key in d and d[key]:
                    entities = d[key].split("; ")
                    for entity in entities:
                        pii_analysis.append({
                            "PII_Type": et,
                            "PII_Value": entity,
                            "Example_ID": d['Example_ID']
                        })
        pii_df = pd.DataFrame(pii_analysis)
        pii_df.to_excel(writer, sheet_name='PII_Analysis', index=False)

        # Complete dataset
        all_df = pd.DataFrame(all_data)
        all_df.to_excel(writer, sheet_name='Complete_Dataset', index=False)

        # Entity Distribution (all entity types)
        entity_analysis = []
        total_entities = sum(pii_stats.values())
        for et, count in pii_stats.items():
            entity_analysis.append({
                "PII_Type": et,
                "Count": count,
                "Percentage": f"{count/total_entities*100:.1f}%" if total_entities > 0 else "0%"
            })
        entity_df = pd.DataFrame(entity_analysis)
        entity_df.to_excel(writer, sheet_name='Entity_Distribution', index=False)
    
    print(f"✅ PII Excel file created: {output_file}")
    print(f"📋 File contains {len(all_data)} examples with PII classification")
    print(f"🔍 PII Distribution:")
    for pii_type, count in pii_stats.items():
        total = sum(pii_stats.values())
        percentage = count/total*100 if total > 0 else 0
        print(f"   - {pii_type}: {count} entities ({percentage:.1f}%)")

# -----------------
# MAIN FUNCTION
# -----------------

def main():
    """Main function for PII training data generation."""
    parser = argparse.ArgumentParser(
        description="Complete PII Training Data Generator for Chilean Documents",
        epilog="""
Examples:
  # Generate 100K training + 20K test
  python %(prog)s --mode create-dataset --train-size 100000 --dev-size 20000
  
  # Create Excel review
  python %(prog)s --mode excel-export --excel-examples 1000
  
  # See examples
  python %(prog)s --mode demo
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--mode", choices=["demo", "excel-export", "create-dataset"], default="demo",
                       help="Mode: 'demo' shows examples, 'excel-export' creates review file, 'create-dataset' generates training data")
    parser.add_argument("--train-size", type=int, default=100000, help="Training set size (default: 100,000)")
    parser.add_argument("--dev-size", type=int, default=20000, help="Test set size (default: 20,000)")
    parser.add_argument("--excel-examples", type=int, default=500, help="Number of examples for Excel export")
    parser.add_argument("--excel-file", type=str, default="pii_data_review.xlsx", help="Excel output filename")
    parser.add_argument("--output-dir", type=str, default="pii_training_large", help="Output directory")
    
    args = parser.parse_args()
    
    # Print header
    print("🚀 COMPLETE PII TRAINING DATA GENERATOR")
    print("=" * 60)
    print(f"👤 User: andresveraf")
    print(f"📅 Date: 2025-08-25 19:19:12 UTC")
    print(f"🎯 Mode: {args.mode}")
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
        detailed_excel_file = output_path / "detailed_review.xlsx"
        
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
        print("🎯 PII CLASSIFICATION DEMONSTRATION")
        print("-" * 50)
        
        for i in range(3):
            print(f"\n📍 Example {i+1}")
            print("-" * 30)
            
            text, annotations = generate_pii_example_with_status("CL")
            print(f"Text: {text}")
            print("\nPII Classification:")
            print("PII_Type\tPII_Value")
            print("-" * 30)
            
            for start, end, pii_type in annotations["entities"]:
                entity_text = text[start:end]
                print(f"{pii_type}\t{entity_text}")
            print()

if __name__ == "__main__":
    main()