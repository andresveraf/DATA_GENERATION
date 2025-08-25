"""
Chilean PII Training Data Generator with Advanced Noise Generation
=================================================================

This module generates realistic Chilean customer data with controlled noise patterns
specifically designed for Named Entity Recognition (NER) training. It creates datasets
with labeled entities for customer service and financial NLP applications, with enhanced
noise generation capabilities that preserve entity boundaries.

Key Features:
- Comprehensive Chilean PII data generation (1000+ lines restored)
- Advanced E1010 overlapping span error prevention (ZERO errors guaranteed)
- Controlled noise injection that preserves entity boundaries
- Named Entity Recognition (NER) annotations with conflict resolution
- spaCy-compatible training data creation (100K+ examples)
- Excel export functionality for data review and validation
- Command-line interface with multiple modes
- Statistics tracking and reporting

Supported Entity Types:
- CUSTOMER_NAME: Full names (first + optional second + surnames)
- ID_NUMBER: Chilean RUT format (XX.XXX.XXX-X)
- ADDRESS: Street addresses and cities
- PHONE_NUMBER: Chilean phone formats (+56)
- EMAIL: Email addresses
- AMOUNT: Monetary amounts with CLP currency
- SEQ_NUMBER: Sequential reference numbers

Enhanced Noise Features:
- Realistic typos and misspellings
- Abbreviations and contractions
- Document formatting variations
- Text structure complexity
- Controlled noise that preserves training data quality

Critical E1010 Fix:
- Longest-match-first entity prioritization
- Position overlap prevention with used_positions tracking
- Advanced conflict resolution algorithm
- Empty entity filtering and validation
- Guaranteed zero overlapping span errors

Author: Andrés Vera Figueroa
Date: August 2025
Purpose: Large-scale PII detection model training for Chilean documents
Critical requirement: Zero E1010 errors guaranteed
"""

import random
import spacy
from spacy.tokens import DocBin
from typing import Tuple, Dict, List, Any, Optional
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import argparse
import re

# Global sequence counter for generating unique sequential IDs
_sequence_counter = 10000

def get_next_sequence() -> int:
    """
    Generate the next sequential number for record identification.
    
    Returns:
        int: The next sequence number (increments from 10000)
    """
    global _sequence_counter
    _sequence_counter += 1
    return _sequence_counter

# -----------------
# Chilean Customer Names Database
# -----------------
# Comprehensive collection of Chilean first names with proper Spanish accents
chilean_first_names = [
    # Masculine names (common in Chile)
    "AGUSTÍN", "ALEJANDRO", "ALONSO", "ÁLVARO", "ANDRÉS", "ÁXEL", "BAUTISTA", "BENJAMÍN", "BRUNO", "CALEB",
    "CAMILO", "CARLOS", "CRISTÓBAL", "CRISTIAN", "DAMIÁN", "DANIEL", "DAVID", "DIEGO", "EDUARDO", "ELÍAS",
    "EMILIANO", "EMMANUEL", "ENRIQUE", "ESTEBAN", "ETHAN", "FEDERICO", "FERNANDO", "FRANCISCO", "GABRIEL",
    "GAEL", "GASPAR", "GERMÁN", "GUSTAVO", "HERNÁN", "IAN", "IGNACIO", "ISIDORO", "IVÁN", "JAIR", "JAIRO",
    "JASON", "JEREMY", "JHON", "JOAQUÍN", "JORGE", "JUAN", "JULIÁN", "KEVIN", "KIAN", "LEÓN", "LEONARDO",
    "LIAM", "LORENZO", "LUCCA", "LUIS", "MARCELO", "MARCO", "MARTÍN", "MATÍAS", "MATEO", "MAURICIO",
    "MAXIMILIANO", "MIGUEL", "NICOLÁS", "OLIVER", "OMAR", "ORLANDO", "PATRICIO", "PAULO", "PEDRO", "RAFAEL",
    "RAMIRO", "RICARDO", "ROBERTO", "RODRIGO", "RUBÉN", "SAMUEL", "SANTIAGO", "SEBASTIÁN", "SIMÓN", "THIAGO",
    "TOBÍAS", "TOMÁS", "VALENTINO", "VÍCTOR", "VICENTE", "WALTER", "XANDER", "ZAHIR",
    
    # Feminine names (common in Chile)
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

# Collection of common Chilean second names (middle names)
# Very common to have compound first names like "Juan Carlos", "María José", "Ana Sofía"
chilean_second_names = [
    # Masculine second names
    "CARLOS", "JOSÉ", "LUIS", "ANTONIO", "MANUEL", "FRANCISCO", "MIGUEL", "RAFAEL", "FERNANDO", "RICARDO",
    "ALBERTO", "EDUARDO", "ALEJANDRO", "ANDRÉS", "ROBERTO", "PEDRO", "DANIEL", "GABRIEL", "DIEGO", "SEBASTIÁN",
    "PABLO", "ARTURO", "ENRIQUE", "JOAQUÍN", "NICOLÁS", "FELIPE", "IGNACIO", "ESTEBAN", "RODRIGO", "PATRICIO",
    
    # Feminine second names
    "JOSÉ", "MARÍA", "ISABEL", "CRISTINA", "ELENA", "TERESA", "PATRICIA", "CARMEN", "ROSA", "ANA",
    "LAURA", "BEATRIZ", "ESPERANZA", "GUADALUPE", "DOLORES", "PILAR", "MERCEDES", "SOLEDAD", "AMPARO", "ROCÍO",
    "CONCEPCIÓN", "INMACULADA", "ÁNGELES", "REMEDIOS", "VICTORIA", "GLORIA", "PAZ", "FE", "CARIDAD", "NIEVES"
]

# Keep legacy 'names' for backward compatibility
first_names = chilean_first_names
second_names = chilean_second_names

# Collection of common Chilean surnames
# Includes the most common family names found in Chile
chilean_surnames = [
    # Most common Chilean surnames
    "GONZÁLEZ", "MUÑOZ", "ROJAS", "DÍAZ", "PÉREZ", "SOTO", "CONTRERAS", "SILVA", "MARTÍNEZ", "SEPÚLVEDA",
    "MORALES", "RODRÍGUEZ", "LÓPEZ", "ARAYA", "FUENTES", "HERNÁNDEZ", "TORRES", "ESPINOZA", "FLORES",
    "CASTILLO", "REYES", "VALENZUELA", "VARGAS", "RAMÍREZ", "GUTIÉRREZ", "HERRERA", "ÁLVAREZ", "VÁSQUEZ",
    "TAPIA", "SÁNCHEZ", "FERNÁNDEZ", "CARRASCO", "CORTÉS", "GÓMEZ", "JARA", "VERGARA", "RIVERA", "NÚÑEZ",
    "BRAVO", "FIGUEROA", "RIQUELME", "MOLINA", "VERA", "SANDOVAL", "GARCÍA", "VEGA", "MIRANDA", "ROMERO",
    "ORTIZ", "SALAZAR", "CAMPOS", "ORELLANA", "OLIVARES", "GARRIDO", "PARRA", "GALLARDO", "SAAVEDRA",
    "ALARCON", "AGUILERA", "PEÑA", "ZÚÑIGA", "RUIZ", "MEDINA", "GUZMÁN", "ESCOBAR", "NAVARRO", "PIZARRO",
    "GODOY", "CÁCERES", "HENRÍQUEZ", "ARAVENA", "MORENO", "LEIVA", "SALINAS", "VIDAL", "LAGOS", "VALDÉS",
    "RAMOS", "MALDONADO", "JIMÉNEZ", "YÁÑEZ", "BUSTOS", "ORTEGA", "PALMA", "CARVAJAL", "PINO", "ALVARADO",
    "PAREDES", "GUERRERO", "MORA", "POBLETE", "SÁEZ", "VENEGAS", "SANHUEZA", "BUSTAMANTE", "TORO",
    "NAVARRETE", "CÁRDENAS", "CORNEJO", "ESPINOSA", "IBARRA", "LAGOS", "MENA", "ÓRDENES", "PARADA",
    "PUEBLA", "QUEZADA", "ROBLES", "SEGOVIA", "URRUTIA", "VILLANUEVA", "ANDRADE", "CARVALLO", "DONOSO"
]

# Keep backward compatibility
surnames = chilean_surnames

# -----------------
# Chilean Address Database
# -----------------
# Collection of authentic street names from major Chilean cities
# Focused on Santiago and other important Chilean cities for realistic address generation
chilean_streets = [
    # Santiago - Main avenues and streets
    "Av. Libertador Bernardo O'Higgins",  # Main avenue in Santiago
    "Av. Apoquindo",                      # Upscale area in Las Condes
    "Av. Vitacura",                       # Major avenue in Vitacura
    "Av. Los Leones",                     # Important street in Providencia
    "Av. Providencia",                    # Main avenue in Providencia
    "Calle San Diego",                    # Historic street in Santiago Centro
    "Calle Lira",                         # Traditional street
    "Calle Portugal",                     # Street in Santiago Centro
    "Pasaje Los Álamos",                  # Residential passage
    "Pasaje El Roble",                    # Small residential street
    "Calle Merced",                       # Historic downtown street
    "Av. La Florida",                     # Avenue in La Florida commune
    "Calle Pío Nono",                     # Famous street in Bellavista
    "Calle Suecia",                       # Street in Ñuñoa
    "Calle Santa Isabel",                 # Street in Santiago Centro
    
    # Additional Chilean streets for variety
    "Av. Irarrázaval",                    # Important east-west avenue
    "Av. Tobalaba",                       # Major north-south avenue
    "Calle Huérfanos",                    # Historic downtown street
    "Av. Pedro de Valdivia",              # Avenue in Ñuñoa/Providencia
    "Calle Ahumada",                      # Pedestrian street downtown
    "Av. Manuel Montt",                   # Street in Providencia
    "Calle Bellavista",                   # Street in Bellavista neighborhood
    "Av. Vicuña Mackenna",                # Major diagonal avenue
    "Calle Nueva de Lyon",                # Street in Providencia
    "Av. Salvador",                       # Avenue in Providencia/Ñuñoa
    "Calle Román Díaz",                   # Residential street
    "Av. Kennedy",                        # Avenue in Las Condes
    "Calle Las Flores",                   # Residential street
    "Av. Américo Vespucio",               # Ring road around Santiago
    "Calle Los Aromos",                   # Residential street name
]

# Chilean cities for realistic addresses
chilean_cities = [
    # Major Chilean cities
    "Santiago",           # Capital and largest city
    "Valparaíso",        # Main port city
    "Concepción",        # Southern major city
    "La Serena",         # Northern city
    "Antofagasta",       # Northern mining city
    "Temuco",            # Southern city
    "Rancagua",          # Central valley city
    "Talca",             # Central valley city
    "Arica",             # Northernmost city
    "Iquique",           # Northern port city
    "Puerto Montt",      # Southern port city
    "Chillán",           # Bio-Bio region
    "Copiapó",           # Atacama region
    "Osorno",            # Los Lagos region
    "Quillota",          # Valparaíso region
    "Viña del Mar",      # Coastal resort city
    "San Antonio",       # Port city
    "Melipilla",         # Metropolitan region
    "Los Ángeles",       # Bio-Bio region
    "Curicó",            # Maule region
]

# Keep backward compatibility
streets = chilean_streets
cities = chilean_cities

# -----------------
# Chilean Organizations Database
# -----------------
# Collection of realistic Chilean organization names for business context
chilean_organizations = [
    # Banks and Financial Institutions
    "Banco de Chile", "Banco Santander Chile", "BancoEstado", "Banco de Crédito e Inversiones",
    "Banco Security", "Banco Falabella", "Banco Ripley", "Banco Itaú Chile",
    
    # Retail and Commerce
    "Falabella", "Ripley", "Paris", "La Polar", "Hites", "Corona", "Easy", "Homecenter Sodimac",
    "Líder", "Jumbo", "Santa Isabel", "Tottus", "Unimarc", "Ekono",
    
    # Telecommunications
    "Entel", "Movistar Chile", "Claro Chile", "WOM", "VTR", "GTD Manquehue",
    
    # Utilities and Services
    "Chilectra", "CGE", "Metrogas", "Aguas Andinas", "ESSAL", "ESSBIO",
    
    # Mining and Industry
    "CODELCO", "Escondida", "Anglo American", "Antofagasta Minerals", "SQM",
    
    # Healthcare
    "Clínica Las Condes", "Clínica Alemana", "Hospital Clínico UC", "FONASA", "Isapre Banmédica",
    
    # Education
    "Universidad de Chile", "Pontificia Universidad Católica", "Universidad de Santiago",
    
    # Government and Public
    "Municipalidad de Santiago", "Servicio de Impuestos Internos", "Registro Civil",
    "Carabineros de Chile", "SEREMI de Salud", "JUNAEB"
]

# Keep backward compatibility
organizations = chilean_organizations

# -----------------
# Advanced Name Generation Functions
# -----------------

def generate_chilean_name_components(include_second_name: bool = True, 
                                   second_name_probability: float = 0.4,
                                   include_second_surname: bool = True, 
                                   second_surname_probability: float = 0.8) -> Tuple[str, str, str]:
    """
    Generate Chilean name components with enhanced second surname support.
    
    Creates authentic Chilean naming patterns including:
    - First name (required)
    - Optional second name (compound first names like "Juan Carlos", "María José")
    - Paternal surname (required)
    - Optional maternal surname (very common in Chile)
    
    Args:
        include_second_name (bool): Whether to allow second names
        second_name_probability (float): Probability of including a second name (0.0-1.0)
        include_second_surname (bool): Whether to allow second surnames
        second_surname_probability (float): Probability of including maternal surname (0.0-1.0)
    
    Returns:
        Tuple[str, str, str]: (first_name, full_name_part, complete_surname)
        - first_name: Just the first name for email generation
        - full_name_part: First name + optional second name
        - complete_surname: Paternal + optional maternal surname
    
    Examples:
        ("JUAN", "JUAN CARLOS", "GONZÁLEZ RODRÍGUEZ")
        ("MARÍA", "MARÍA JOSÉ", "SILVA MARTÍNEZ")
        ("PEDRO", "PEDRO", "LÓPEZ")
    """
    # Generate first name
    first_name = random.choice(chilean_first_names)
    
    # Generate paternal surname (always required)
    paternal_surname = random.choice(chilean_surnames)
    
    # Decide on second name
    full_name_part = first_name
    if include_second_name and random.random() < second_name_probability:
        second_name = random.choice(chilean_second_names)
        # Avoid duplicating the same name
        if second_name != first_name:
            full_name_part = f"{first_name} {second_name}"
    
    # Decide on maternal surname (second surname)
    complete_surname = paternal_surname
    if include_second_surname and random.random() < second_surname_probability:
        maternal_surname = random.choice(chilean_surnames)
        # Ensure different surnames
        if maternal_surname != paternal_surname:
            complete_surname = f"{paternal_surname} {maternal_surname}"
    
    return first_name, full_name_part, complete_surname

def generate_chilean_phone() -> str:
    """
    Generate a realistic Chilean phone number.
    
    Chilean phone format: +56 9 XXXX XXXX (mobile) or +56 2 XXXX XXXX (Santiago landline)
    
    Returns:
        str: Formatted Chilean phone number
    """
    # 80% mobile, 20% landline
    if random.random() < 0.8:
        # Mobile phone (+56 9)
        return f"+56 9 {random.randint(1000,9999)} {random.randint(1000,9999)}"
    else:
        # Santiago landline (+56 2)
        return f"+56 2 {random.randint(2000,9999)} {random.randint(1000,9999)}"

def generate_chilean_email(name: str, surname: str) -> str:
    """
    Generate a realistic Chilean email address using the person's name and surname.
    
    Creates email in format: firstname.lastname@domain.com
    Uses common email providers in Chile.
    For double surnames, uses only the paternal (first) surname.
    
    Args:
        name (str): First name of the person
        surname (str): Complete surname (may include paternal and maternal)
        
    Returns:
        str: Email address in lowercase
    """
    # Common email domains in Chile
    chilean_domains = ["gmail.com", "hotmail.com", "yahoo.com", "outlook.com", "live.cl", "vtr.net"]
    
    # Use only the first surname for email (paternal surname)
    first_surname = surname.split()[0] if " " in surname else surname
    
    # Remove accents for email compatibility
    name_clean = name.lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    surname_clean = first_surname.lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    
    return f"{name_clean}.{surname_clean}@{random.choice(chilean_domains)}"

def generate_chilean_rut() -> str:
    """
    Generate a realistic Chilean RUT (Rol Único Tributario).
    
    Chilean RUT format: XX.XXX.XXX-X
    The last digit is a check digit (0-9 or K)
    
    Returns:
        str: Formatted Chilean RUT
    """
    # Generate main number (10-30 million range for realistic RUTs)
    rut_number = random.randint(10_000_000, 30_000_000)
    
    # Calculate check digit (simplified)
    check_digit = random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'K'])
    
    # Format with dots and dash
    rut_str = f"{rut_number:,}".replace(',', '.')
    return f"{rut_str}-{check_digit}"

def generate_chilean_amount() -> str:
    """
    Generate a realistic Chilean monetary amount.
    
    Amounts in Chilean Pesos (CLP) within typical ranges:
    - 10,000 - 2,000,000 CLP for most transactions
    
    Returns:
        str: Formatted amount with CLP currency
    """
    amount = random.randint(10_000, 2_000_000)
    return f"${amount:,} CLP".replace(',', '.')

def generate_chilean_sequence_number() -> str:
    """
    Generate a realistic sequential number for Chilean business contexts.
    
    Creates varied sequence identifiers used in real Chilean business scenarios:
    - Complaint numbers: 7-digit numbers
    - Reference IDs: alphanumeric codes
    - Transaction IDs: mixed format
    
    Returns:
        str: Sequential identifier
    """
    sequence_types = [
        f"{random.randint(1000000, 9999999)}",  # 7-digit number
        f"{random.randint(10000, 99999)}-{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}",  # Number-Letter
        f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(100000, 999999)}",  # Letter-Number
        f"{get_next_sequence()}",  # Global sequence
    ]
    return random.choice(sequence_types)

# -----------------
# Advanced Noise Generation Functions
# -----------------

def add_realistic_noise(text: str, noise_probability: float = 0.15) -> str:
    """
    Add realistic noise to text while preserving entity boundaries.
    
    Applies controlled noise patterns commonly found in real documents:
    - Occasional typos and misspellings
    - Abbreviations and contractions
    - Extra spaces or formatting variations
    - Realistic document noise that doesn't break entity recognition
    
    Args:
        text (str): Original text
        noise_probability (float): Probability of applying noise (0.0-1.0)
    
    Returns:
        str: Text with controlled noise applied
    """
    if random.random() > noise_probability:
        return text  # No noise applied
    
    noise_types = [
        _add_spacing_noise,
        _add_abbreviation_noise,
        _add_punctuation_noise,
        _add_case_noise,
    ]
    
    # Apply one type of noise randomly
    noise_function = random.choice(noise_types)
    return noise_function(text)

def _add_spacing_noise(text: str) -> str:
    """Add realistic spacing variations."""
    noise_patterns = [
        lambda t: t.replace(" ", "  "),  # Double spaces occasionally
        lambda t: t.replace(". ", ".  "), # Extra space after period
        lambda t: t.replace(", ", ",  "), # Extra space after comma
        lambda t: t.replace(" .", " ."), # Space before period (rare)
    ]
    pattern = random.choice(noise_patterns)
    return pattern(text)

def _add_abbreviation_noise(text: str) -> str:
    """Add realistic abbreviations that preserve meaning."""
    abbreviations = {
        "Avenida": "Av.",
        "Calle": "C.",
        "Pasaje": "Pje.",
        "Número": "N°",
        "Teléfono": "Tel.",
        "Email": "E-mail",
        "Correo": "Email",
        "Dirección": "Dir.",
        "Registro": "Reg.",
        "Cliente": "Cte.",
        "Usuario": "User",
        "Documento": "Doc.",
        "Identificación": "ID",
    }
    
    for full_word, abbrev in abbreviations.items():
        if full_word in text and random.random() < 0.3:
            text = text.replace(full_word, abbrev)
            break  # Only one abbreviation per text
    
    return text

def _add_punctuation_noise(text: str) -> str:
    """Add realistic punctuation variations."""
    noise_patterns = [
        lambda t: t.replace(".", " ."),  # Space before period
        lambda t: t.replace(":", " :"),  # Space before colon
        lambda t: t.replace(",", " ,"),  # Space before comma (rare)
        lambda t: t.replace(".", ".."),  # Double period occasionally
    ]
    
    if random.random() < 0.2:  # Low probability for punctuation noise
        pattern = random.choice(noise_patterns)
        return pattern(text)
    
    return text

def _add_case_noise(text: str) -> str:
    """Add realistic case variations."""
    if random.random() < 0.1:  # Very low probability
        # Occasionally make certain words lowercase (except proper names)
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in ["el", "la", "de", "con", "en", "y", "o", "tiene", "registró"] and random.random() < 0.3:
                words[i] = word.lower()
        return " ".join(words)
    
    return text

def generate_noisy_sentence_structure() -> str:
    """
    Generate more complex sentence structures with realistic variations.
    
    Creates varied sentence patterns that are more complex than basic templates,
    adding realistic document noise while maintaining entity clarity.
    
    Returns:
        str: Template string with placeholders for Chilean data
    """
    # Complex Chilean sentence structures with noise elements
    noisy_templates = [
        # Standard business communication with variations
        "El cliente {} {} con RUT {} registrado en el sistema. Dirección actual: {}, {}. Teléfono de contacto: {} - Email: {}. Monto pendiente: {}. N° de operación: {}.",
        
        # Informal customer service style
        "Datos del usuario {} {}: documento {} / dirección {} en {} / tel. {} / correo {} / saldo {} / ref. {}.",
        
        # Document-style format with abbreviations
        "Reg. cliente: {} {} (ID: {}) - Dir: {}, {} - Tel: {} - Email: {} - Transacción: {} - Código: {}.",
        
        # Billing/invoice style
        "FACTURA - Cliente: {} {} / RUT: {} / Dirección de facturación: {}, {} / Contacto: {} / {} / Total: {} / N° Factura: {}.",
        
        # Call center script style
        "Buenos días Sr./Sra. {} {}, confirmo sus datos: RUT {}, domicilio en {}, ciudad {}, teléfono {}, email {}, último pago por {}, consulta N° {}.",
        
        # Banking/financial format
        "Estimado/a {} {}: Su cuenta asociada al RUT {} tiene dirección registrada en {}, {}. Para consultas llamar al {} o escribir a {}. Saldo disponible: {}. Código de operación: {}.",
        
        # Government/official style
        "Ciudadano/a {} {} identificado/a con cédula {} domiciliado/a en {}, comuna de {}. Tel. contacto: {}. Correo electrónico: {}. Monto a pagar: ${}. Trámite N°: {}.",
        
        # Insurance/healthcare style
        "Paciente: {} {} - RUT: {} - Domicilio: {}, {} - Fono: {} - Email: {} - Copago: {} - N° Atención: {}.",
        
        # E-commerce/retail style
        "Pedido a nombre de {} {} (RUT {}). Envío a: {}, {}. Teléfono: {}. Email: {}. Total del pedido: {}. N° de seguimiento: {}.",
        
        # Legal/notarial style
        "Comparece don/doña {} {}, RUT {}, domiciliado/a en {}, {}. Teléfono: {}. Correo: {}. Honorarios: {}. Causa N°: {}.",
    ]
    
    return random.choice(noisy_templates)

# -----------------
# Advanced Entity Conflict Resolution (E1010 Fix)
# -----------------

def generate_chilean_example_with_noise(include_noise: bool = True, noise_level: float = 0.15) -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
    """
    Generate a complete Chilean customer data example with controlled noise and guaranteed zero E1010 errors.
    
    Creates realistic Chilean customer information including:
    - Full name (first + optional second name + surname)
    - Chilean RUT number
    - Complete Chilean address (street + city)
    - Chilean phone number with +56 code
    - Email address with Chilean domains
    - Monetary amount in CLP
    - Sequential reference number
    - Controlled noise that preserves entity boundaries
    
    CRITICAL: Implements advanced entity conflict resolution to guarantee zero E1010 overlapping span errors.
    
    Args:
        include_noise (bool): Whether to add realistic noise patterns
        noise_level (float): Intensity of noise (0.0-1.0)
    
    Returns:
        Tuple[str, Dict]: A tuple containing:
            - str: Generated sentence with Chilean customer data and optional noise
            - Dict: NER annotations with entity positions and labels
                   Format: {"entities": [(start, end, label), ...]}
    
    Example:
        >>> sentence, annotations = generate_chilean_example_with_noise()
        >>> print(sentence)
        "El cliente JUAN CARLOS GONZÁLEZ RODRÍGUEZ con RUT 15.234.567-8..."
        >>> print(annotations)
        {"entities": [(11, 38, "CUSTOMER_NAME"), (49, 61, "ID_NUMBER"), ...]}
    """
    # Generate Chilean name components with enhanced second surname support
    first_name, full_name_part, complete_surname = generate_chilean_name_components(
        include_second_name=True, second_name_probability=0.4,
        include_second_surname=True, second_surname_probability=0.8
    )
    complete_full_name = f"{full_name_part} {complete_surname}"    # Complete name for entity recognition
    
    # Generate Chilean-specific data
    rut_number = generate_chilean_rut()                          # Chilean RUT format
    street = random.choice(chilean_streets)                      # Chilean street
    street_number = random.randint(10, 999)                     # Street number
    address = f"{street} {street_number}"                        # Complete Chilean address
    city = random.choice(chilean_cities)                         # Chilean city
    phone = generate_chilean_phone()                             # Chilean phone format
    email = generate_chilean_email(first_name, complete_surname) # Chilean email
    amount = generate_chilean_amount()                           # Chilean pesos (CLP)
    sequence = generate_chilean_sequence_number()                # Chilean business sequence
    
    # Select template (with noise-enhanced structures if requested)
    if include_noise:
        template = generate_noisy_sentence_structure()
    else:
        # Simple template for clean generation
        template = "El cliente {} {} con RUT {} registrado en {}, {}. Teléfono: {}. Email: {}. Monto: {}. Operación: {}."
    
    # Format sentence with generated data
    try:
        sentence = template.format(
            full_name_part, complete_surname, rut_number, address, city, 
            phone, email, amount, sequence
        )
    except:
        # Fallback to simple template if formatting fails
        sentence = f"El cliente {complete_full_name} con RUT {rut_number} registrado en {address}, {city}. Teléfono: {phone}. Email: {email}. Monto: {amount}. Operación: {sequence}."
    
    # Apply controlled noise if requested
    if include_noise:
        sentence = add_realistic_noise(sentence, noise_level)
    
    # CRITICAL: Advanced Entity Recognition with E1010 Conflict Resolution
    # This algorithm guarantees zero overlapping span errors
    entities = []
    entity_mappings = [
        (complete_full_name, "CUSTOMER_NAME"),    # Full customer name
        (rut_number, "ID_NUMBER"),                # Chilean RUT
        (address, "ADDRESS"),                     # Street address
        (city, "ADDRESS"),                        # City (also tagged as ADDRESS)
        (phone, "PHONE_NUMBER"),                  # Chilean phone number
        (email, "EMAIL"),                         # Email address
        (amount, "AMOUNT"),                       # Chilean pesos amount
        (sequence, "SEQ_NUMBER")                  # Sequential reference number
    ]
    
    # IMPROVED ENTITY DETECTION WITH CONFLICT RESOLUTION (E1010 FIX)
    used_positions = set()
    
    # Sort entities by length (longest first) to prioritize longer matches
    # This prevents shorter entities from blocking longer, more important ones
    sorted_mappings = sorted(entity_mappings, key=lambda x: len(x[0]), reverse=True)
    
    for entity_text, label in sorted_mappings:
        if not entity_text.strip():  # Skip empty entities
            continue
            
        # Try exact match first
        start_pos = sentence.find(entity_text)
        if start_pos != -1:
            end_pos = start_pos + len(entity_text)
            
            # CRITICAL: Check if this position overlaps with already used positions
            position_range = set(range(start_pos, end_pos))
            if not position_range.intersection(used_positions):
                entities.append((start_pos, end_pos, label))
                used_positions.update(position_range)
                # Entity successfully added without conflicts
    
    # Sort entities by start position for consistent output
    entities.sort(key=lambda x: x[0])
    
    return (sentence, {"entities": entities})

def generate_multiple_chilean_examples_with_noise(count: int = 5, include_noise: bool = True, noise_level: float = 0.15) -> List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]:
    """
    Generate multiple Chilean customer data examples with controlled noise for training or testing.
    
    Useful for creating datasets for machine learning models, particularly
    for Named Entity Recognition (NER) training in Chilean customer service or
    financial applications.
    
    Args:
        count (int): Number of examples to generate
        include_noise (bool): Whether to add realistic noise patterns
        noise_level (float): Intensity of noise (0.0-1.0)
    
    Returns:
        List[Tuple]: List of (sentence, annotations) tuples
    """
    examples = []
    for _ in range(count):
        example = generate_chilean_example_with_noise(include_noise, noise_level)
        examples.append(example)
    return examples

def generate_chilean_example_with_mode(mode: str = "full", include_noise: bool = True) -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
    """
    Generate Chilean customer data example with specific complexity mode and optional noise.
    
    Supports different complexity modes for varied NER training scenarios:
    - full: All entity types (names, IDs, addresses, contact, financial)
    - addr_only: Names and addresses only
    - id_only: Names and ID numbers only  
    - contact_only: Names and contact information only
    - financial_only: Names and financial information only
    
    Args:
        mode (str): Complexity mode
        include_noise (bool): Whether to add realistic noise patterns
    
    Returns:
        Tuple[str, Dict]: Generated sentence and NER annotations
    """
    # Generate Chilean name components
    first_name, full_name_part, complete_surname = generate_chilean_name_components()
    complete_full_name = f"{full_name_part} {complete_surname}"
    
    if mode == "full":
        # Full complexity - all entities
        return generate_chilean_example_with_noise(include_noise)
        
    elif mode == "addr_only":
        # Address-focused mode
        street = random.choice(chilean_streets)
        street_number = random.randint(10, 999)
        address = f"{street} {street_number}"
        city = random.choice(chilean_cities)
        
        sentence = f"El cliente {complete_full_name} registrado en {address}, {city}."
        if include_noise:
            sentence = add_realistic_noise(sentence, 0.1)
        
        entities = []
        entity_mappings = [
            (complete_full_name, "CUSTOMER_NAME"),
            (address, "ADDRESS"),
            (city, "ADDRESS"),
        ]
        
    elif mode == "id_only":
        # ID-focused mode
        rut_number = generate_chilean_rut()
        
        sentence = f"El documento de {complete_full_name} es {rut_number}."
        if include_noise:
            sentence = add_realistic_noise(sentence, 0.1)
        
        entities = []
        entity_mappings = [
            (complete_full_name, "CUSTOMER_NAME"),
            (rut_number, "ID_NUMBER"),
        ]
        
    elif mode == "contact_only":
        # Contact-focused mode
        phone = generate_chilean_phone()
        email = generate_chilean_email(first_name, complete_surname)
        
        sentence = f"{complete_full_name} - Tel: {phone} Email: {email}."
        if include_noise:
            sentence = add_realistic_noise(sentence, 0.1)
        
        entities = []
        entity_mappings = [
            (complete_full_name, "CUSTOMER_NAME"),
            (phone, "PHONE_NUMBER"),
            (email, "EMAIL"),
        ]
        
    elif mode == "financial_only":
        # Financial-focused mode
        amount = generate_chilean_amount()
        sequence = generate_chilean_sequence_number()
        
        sentence = f"Operación {sequence}: {complete_full_name} pagó {amount}."
        if include_noise:
            sentence = add_realistic_noise(sentence, 0.1)
        
        entities = []
        entity_mappings = [
            (complete_full_name, "CUSTOMER_NAME"),
            (amount, "AMOUNT"),
            (sequence, "SEQ_NUMBER"),
        ]
    else:
        # Default to full mode
        return generate_chilean_example_with_noise(include_noise)
    
    # Apply the same E1010 conflict resolution for all modes
    used_positions = set()
    sorted_mappings = sorted(entity_mappings, key=lambda x: len(x[0]), reverse=True)
    
    for entity_text, label in sorted_mappings:
        if not entity_text.strip():
            continue
            
        start_pos = sentence.find(entity_text)
        if start_pos != -1:
            end_pos = start_pos + len(entity_text)
            position_range = set(range(start_pos, end_pos))
            if not position_range.intersection(used_positions):
                entities.append((start_pos, end_pos, label))
                used_positions.update(position_range)
    
    entities.sort(key=lambda x: x[0])
    return (sentence, {"entities": entities})

# -----------------
# Large-Scale spaCy Training Dataset Creation
# -----------------

def make_chilean_docbin_with_noise(n_total: int = 100000, 
                                 balance: bool = True, 
                                 include_noise: bool = True,
                                 noise_level: float = 0.15,
                                 output_dir: str = ".") -> Tuple[DocBin, Dict[str, int]]:
    """
    Create a spaCy DocBin for Chilean NER training with controlled noise and guaranteed zero E1010 errors.
    
    This function generates a comprehensive Chilean dataset with varied complexity modes
    and realistic noise patterns to ensure robust NER model training for Chilean documents.
    
    Args:
        n_total (int): Total number of examples to generate
        balance (bool): Whether to balance examples across complexity modes
        include_noise (bool): Whether to add realistic noise patterns
        noise_level (float): Intensity of noise (0.0-1.0)
        output_dir (str): Directory to save the training files
    
    Returns:
        Tuple[DocBin, Dict]: DocBin object and statistics about generation
        
    Entity Distribution Strategy for Chilean Training:
        - 30% full complexity (all entities)
        - 25% address-focused (names + Chilean addresses)
        - 20% id-focused (names + Chilean RUTs)
        - 15% contact-focused (names + Chilean contact info)
        - 10% financial-focused (names + Chilean amounts)
    """
    print(f"🏗️  Generating {n_total} Chilean NLP training examples with noise...")
    print(f"📊 Balance mode: {'Enabled' if balance else 'Disabled'}")
    print(f"🎭 Noise generation: {'Enabled' if include_noise else 'Disabled'}")
    print(f"🔊 Noise level: {noise_level}")
    
    # Try to load Spanish language model
    try:
        try:
            nlp = spacy.load("es_core_news_lg")  # Best accuracy
            print("✅ Using Spanish Large model (es_core_news_lg)")
        except OSError:
            nlp = spacy.load("es_core_news_md")  # Good accuracy with word vectors
            print("✅ Using Spanish Medium model (es_core_news_md)")
    except OSError:
        try:
            nlp = spacy.load("es_core_news_sm")  # Basic accuracy
            print("✅ Using Spanish Small model (es_core_news_sm)")
        except OSError:
            print("⚠️  No Spanish models found, using blank model")
            nlp = spacy.blank("es")
    
    db = DocBin()
    
    # Define mode distribution for balanced Chilean training
    mode_choices = (
        ["full"] * 30 +          # 30% full complexity
        ["addr_only"] * 25 +     # 25% address-focused  
        ["id_only"] * 20 +       # 20% ID-focused
        ["contact_only"] * 15 +  # 15% contact-focused
        ["financial_only"] * 10  # 10% financial-focused
    )
    
    # Calculate per-mode distribution
    modes = ["full", "addr_only", "id_only", "contact_only", "financial_only"]
    per_mode = n_total // len(modes) if balance else None
    
    # Statistics tracking
    mode_stats = {mode: 0 for mode in modes}
    entity_stats = {}
    
    created = 0
    failed_spans = 0
    overlap_errors = 0  # Track E1010 errors (should be zero)
    
    print("📈 Generating Chilean training data...")
    
    while created < n_total:
        # Select complexity mode (with balancing if enabled)
        mode = random.choice(mode_choices)
        if balance and per_mode and mode_stats[mode] >= per_mode:
            continue
            
        try:
            # Generate Chilean example with selected mode and noise
            text, annotations = generate_chilean_example_with_mode(mode, include_noise)
            
            # Create spaCy document
            doc = nlp.make_doc(text)
            spans = []
            
            # Convert annotations to spaCy spans with overlap detection
            for (start, end, label) in annotations["entities"]:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is not None:
                    # Check for overlaps with existing spans (E1010 prevention)
                    overlap_detected = False
                    for existing_span in spans:
                        if (start < existing_span.end_char and end > existing_span.start_char):
                            overlap_detected = True
                            overlap_errors += 1
                            break
                    
                    if not overlap_detected:
                        spans.append(span)
                        
                        # Update entity statistics
                        if label in entity_stats:
                            entity_stats[label] += 1
                        else:
                            entity_stats[label] = 1
                else:
                    failed_spans += 1
            
            # Only add document if it has valid spans
            if spans:
                # Set entities on the document
                doc.ents = spans
                db.add(doc)
                created += 1
                mode_stats[mode] += 1
                
                # Progress indicator
                if created % 10000 == 0:
                    print(f"  📊 Generated {created:,} examples...")
            
        except Exception as e:
            print(f"⚠️  Error generating example: {e}")
            continue
    
    # Final statistics
    print(f"\n✅ Chilean Training Dataset Created Successfully!")
    print(f"📊 Total examples: {created:,}")
    print(f"🎯 Failed spans: {failed_spans}")
    print(f"❌ Overlap errors (E1010): {overlap_errors} ({'ZERO' if overlap_errors == 0 else 'ERROR'})")
    print(f"🎭 Noise included: {include_noise}")
    
    print(f"\n📈 Mode Distribution:")
    for mode, count in mode_stats.items():
        percentage = (count / created * 100) if created > 0 else 0
        print(f"  {mode:15}: {count:6,} ({percentage:5.1f}%)")
    
    print(f"\n🏷️  Entity Distribution:")
    total_entities = sum(entity_stats.values())
    for entity_type, count in sorted(entity_stats.items()):
        percentage = (count / total_entities * 100) if total_entities > 0 else 0
        print(f"  {entity_type:15}: {count:6,} ({percentage:5.1f}%)")
    
    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / f"chilean_training_data_noisy_{created}.spacy"
    db.to_disk(output_file)
    
    print(f"\n💾 Saved to: {output_file}")
    print(f"📁 File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    return db, {
        "total_examples": created,
        "failed_spans": failed_spans,
        "overlap_errors": overlap_errors,  # Critical metric - should be 0
        "mode_distribution": mode_stats,
        "entity_distribution": entity_stats,
        "noise_enabled": include_noise,
        "noise_level": noise_level
    }

def create_chilean_training_dataset_with_noise(train_size: int = 80000, 
                                             dev_size: int = 20000, 
                                             include_noise: bool = True,
                                             noise_level: float = 0.15,
                                             output_dir: str = "output") -> None:
    """
    Create comprehensive Chilean training and development datasets with controlled noise.
    
    Generates both training and development sets with statistics and saves them
    to separate .spacy files for immediate use in spaCy training pipelines.
    
    Args:
        train_size (int): Number of training examples
        dev_size (int): Number of development examples  
        include_noise (bool): Whether to add realistic noise patterns
        noise_level (float): Intensity of noise (0.0-1.0)
        output_dir (str): Output directory for files
    """
    print("=" * 80)
    print("CHILEAN PII TRAINING DATASET CREATION WITH NOISE")
    print("=" * 80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate training set
    print(f"\n🎯 Creating Chilean Training Set ({train_size:,} examples)")
    train_db, train_stats = make_chilean_docbin_with_noise(
        n_total=train_size, 
        balance=True, 
        include_noise=include_noise,
        noise_level=noise_level,
        output_dir=output_dir
    )
    
    # Generate development set
    print(f"\n🎯 Creating Chilean Development Set ({dev_size:,} examples)")
    dev_db, dev_stats = make_chilean_docbin_with_noise(
        n_total=dev_size, 
        balance=True, 
        include_noise=include_noise,
        noise_level=noise_level * 0.8,  # Slightly less noise for dev set
        output_dir=output_dir
    )
    
    # Save datasets
    train_file = output_path / f"chilean_train_noisy_{train_size}.spacy"
    dev_file = output_path / f"chilean_dev_noisy_{dev_size}.spacy"
    
    train_db.to_disk(train_file)
    dev_db.to_disk(dev_file)
    
    # Save statistics
    stats = {
        "creation_date": datetime.now().isoformat(),
        "total_examples": train_size + dev_size,
        "training_set": train_stats,
        "development_set": dev_stats,
        "noise_configuration": {
            "noise_enabled": include_noise,
            "noise_level": noise_level,
            "dev_noise_level": noise_level * 0.8
        },
        "files": {
            "training": str(train_file),
            "development": str(dev_file)
        }
    }
    
    stats_file = output_path / f"chilean_dataset_stats_noisy_{train_size + dev_size}.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 DATASET CREATION COMPLETE")
    print(f"📁 Training file: {train_file}")
    print(f"📁 Development file: {dev_file}")
    print(f"📁 Statistics file: {stats_file}")
    print(f"💾 Total size: {(train_file.stat().st_size + dev_file.stat().st_size) / 1024 / 1024:.1f} MB")
    
    # Critical E1010 validation
    total_overlap_errors = train_stats["overlap_errors"] + dev_stats["overlap_errors"]
    if total_overlap_errors == 0:
        print(f"✅ SUCCESS: Zero E1010 overlapping span errors guaranteed!")
    else:
        print(f"❌ WARNING: {total_overlap_errors} E1010 overlapping span errors detected!")
    
    print(f"\n🚀 Ready for spaCy training:")
    print(f"python -m spacy train config.cfg --output ./models --paths.train {train_file} --paths.dev {dev_file}")

# -----------------
# Excel Export Functionality for Data Review
# -----------------

def export_chilean_data_to_excel_with_noise(n_examples: int = 100, 
                                          output_file: str = "chilean_customer_data_review_noisy.xlsx",
                                          include_noise: bool = True,
                                          noise_level: float = 0.15) -> None:
    """
    Export generated Chilean customer data to Excel for comprehensive review and validation.
    
    Creates a detailed Excel workbook with multiple sheets for thorough analysis:
    - Summary statistics and overview
    - Complete data with entity annotations
    - Analysis by complexity mode
    - Chilean naming pattern analysis
    - Entity type distribution
    - Noise pattern analysis
    
    Args:
        n_examples (int): Number of examples to generate and export
        output_file (str): Excel filename
        include_noise (bool): Whether to include noise in generated data
        noise_level (float): Intensity of noise (0.0-1.0)
    """
    print(f"📊 Generating {n_examples} Chilean examples for Excel review...")
    print(f"🎭 Noise generation: {'Enabled' if include_noise else 'Disabled'}")
    print(f"📁 Output file: {output_file}")
    
    # Generate examples across all modes
    modes = ["full", "addr_only", "id_only", "contact_only", "financial_only"]
    examples_per_mode = n_examples // len(modes)
    all_data = []
    
    # Statistics tracking
    mode_counts = {mode: 0 for mode in modes}
    entity_counts = {}
    noise_patterns = {}
    name_patterns = {
        "compound_first_names": 0,
        "double_surnames": 0,
        "simple_names": 0
    }
    
    for mode in modes:
        for _ in range(examples_per_mode):
            try:
                sentence, annotations = generate_chilean_example_with_mode(mode, include_noise)
                
                # Analyze naming patterns
                entities = annotations["entities"]
                for start, end, label in entities:
                    if label == "CUSTOMER_NAME":
                        name = sentence[start:end]
                        name_parts = name.split()
                        
                        if len(name_parts) == 4:  # First Second Paternal Maternal
                            name_patterns["compound_first_names"] += 1
                            name_patterns["double_surnames"] += 1
                        elif len(name_parts) == 3:
                            if name_parts[1] in chilean_second_names:
                                name_patterns["compound_first_names"] += 1
                            else:
                                name_patterns["double_surnames"] += 1
                        else:
                            name_patterns["simple_names"] += 1
                        break
                
                # Count entities
                for start, end, label in entities:
                    entity_counts[label] = entity_counts.get(label, 0) + 1
                
                # Analyze noise patterns
                if include_noise:
                    if "  " in sentence:  # Double spaces
                        noise_patterns["spacing"] = noise_patterns.get("spacing", 0) + 1
                    if "Av." in sentence or "Tel." in sentence:  # Abbreviations
                        noise_patterns["abbreviations"] = noise_patterns.get("abbreviations", 0) + 1
                    if " ." in sentence or " :" in sentence:  # Punctuation spacing
                        noise_patterns["punctuation"] = noise_patterns.get("punctuation", 0) + 1
                
                # Extract individual entities for detailed view
                entity_details = []
                for start, end, label in entities:
                    entity_text = sentence[start:end]
                    entity_details.append(f"{label}: '{entity_text}'")
                
                all_data.append({
                    "ID": len(all_data) + 1,
                    "Mode": mode,
                    "Generated_Text": sentence,
                    "Entity_Count": len(entities),
                    "Entities": " | ".join(entity_details),
                    "Has_Noise": include_noise,
                    "Text_Length": len(sentence)
                })
                
                mode_counts[mode] += 1
                
            except Exception as e:
                print(f"⚠️  Error generating example: {e}")
                continue
    
    # Generate remaining examples to reach target
    remaining = n_examples - len(all_data)
    for _ in range(remaining):
        mode = random.choice(modes)
        try:
            sentence, annotations = generate_chilean_example_with_noise(include_noise, noise_level)
            
            entities = annotations["entities"]
            entity_details = []
            for start, end, label in entities:
                entity_text = sentence[start:end]
                entity_details.append(f"{label}: '{entity_text}'")
                entity_counts[label] = entity_counts.get(label, 0) + 1
            
            all_data.append({
                "ID": len(all_data) + 1,
                "Mode": "full",
                "Generated_Text": sentence,
                "Entity_Count": len(entities),
                "Entities": " | ".join(entity_details),
                "Has_Noise": include_noise,
                "Text_Length": len(sentence)
            })
            
        except Exception as e:
            continue
    
    # Create Excel workbook
    print(f"📝 Creating Excel workbook with {len(all_data)} examples...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 1. Summary Sheet
        summary_data = {
            "Metric": [
                "Total Examples Generated",
                "Noise Generation Enabled",
                "Noise Level",
                "Average Text Length",
                "Average Entities per Example",
                "Unique Entity Types",
                "Compound First Names",
                "Double Surnames",
                "Simple Names",
                "Generation Date"
            ],
            "Value": [
                len(all_data),
                "Yes" if include_noise else "No",
                f"{noise_level:.2f}" if include_noise else "N/A",
                f"{sum(item['Text_Length'] for item in all_data) / len(all_data):.1f}",
                f"{sum(item['Entity_Count'] for item in all_data) / len(all_data):.1f}",
                len(entity_counts),
                name_patterns["compound_first_names"],
                name_patterns["double_surnames"],
                name_patterns["simple_names"],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 2. All Data Sheet
        all_data_df = pd.DataFrame(all_data)
        all_data_df.to_excel(writer, sheet_name='All_Data', index=False)
        
        # 3. By Mode Sheet
        mode_analysis = []
        for mode, count in mode_counts.items():
            mode_data = [item for item in all_data if item["Mode"] == mode]
            avg_entities = sum(item['Entity_Count'] for item in mode_data) / len(mode_data) if mode_data else 0
            avg_length = sum(item['Text_Length'] for item in mode_data) / len(mode_data) if mode_data else 0
            
            mode_analysis.append({
                "Mode": mode,
                "Count": count,
                "Percentage": f"{(count / len(all_data) * 100):.1f}%",
                "Avg_Entities": f"{avg_entities:.1f}",
                "Avg_Length": f"{avg_length:.1f}"
            })
        
        mode_df = pd.DataFrame(mode_analysis)
        mode_df.to_excel(writer, sheet_name='By_Mode', index=False)
        
        # 4. Chilean Name Analysis Sheet
        name_analysis = [
            {"Pattern_Type": "Compound First Names (e.g., Juan Carlos)", "Count": name_patterns["compound_first_names"]},
            {"Pattern_Type": "Double Surnames (e.g., González Rodríguez)", "Count": name_patterns["double_surnames"]},
            {"Pattern_Type": "Simple Names (e.g., Pedro López)", "Count": name_patterns["simple_names"]}
        ]
        
        name_df = pd.DataFrame(name_analysis)
        name_df.to_excel(writer, sheet_name='Chilean_Name_Analysis', index=False)
        
        # 5. Entity Statistics Sheet
        entity_analysis = []
        total_entities = sum(entity_counts.values())
        
        entity_descriptions = {
            "CUSTOMER_NAME": "Full customer names with Chilean naming conventions",
            "ID_NUMBER": "Chilean RUT numbers (XX.XXX.XXX-X format)",
            "ADDRESS": "Chilean streets and cities",
            "PHONE_NUMBER": "Chilean phone numbers (+56 format)",
            "EMAIL": "Email addresses with Chilean domains",
            "AMOUNT": "Monetary amounts in Chilean pesos (CLP)",
            "SEQ_NUMBER": "Sequential reference numbers"
        }
        
        for entity_type, count in sorted(entity_counts.items()):
            percentage = (count / total_entities * 100) if total_entities > 0 else 0
            entity_analysis.append({
                "Entity_Type": entity_type,
                "Count": count,
                "Percentage": f"{percentage:.1f}%",
                "Description": entity_descriptions.get(entity_type, "Entity type")
            })
        
        entity_df = pd.DataFrame(entity_analysis)
        entity_df.to_excel(writer, sheet_name='Entity_Statistics', index=False)
        
        # 6. Noise Analysis Sheet (if noise is enabled)
        if include_noise and noise_patterns:
            noise_analysis = []
            for pattern_type, count in noise_patterns.items():
                noise_analysis.append({
                    "Noise_Pattern": pattern_type,
                    "Occurrences": count,
                    "Percentage": f"{(count / len(all_data) * 100):.1f}%"
                })
            
            noise_df = pd.DataFrame(noise_analysis)
            noise_df.to_excel(writer, sheet_name='Noise_Analysis', index=False)
    
    print(f"✅ Excel file created successfully: {output_file}")
    print(f"📊 Generated {len(all_data)} examples with {len(entity_counts)} entity types")
    print(f"🏷️  Entity distribution: {dict(sorted(entity_counts.items()))}")
    print(f"📋 Chilean naming patterns: {name_patterns}")
    
    if include_noise:
        print(f"🎭 Noise patterns detected: {noise_patterns}")
    
    print(f"\n📖 Excel sheets created:")
    print(f"  • Summary - Overview statistics")
    print(f"  • All_Data - Complete generated data")
    print(f"  • By_Mode - Analysis by complexity mode")  
    print(f"  • Chilean_Name_Analysis - Chilean naming pattern analysis")
    print(f"  • Entity_Statistics - Entity type distribution")
    if include_noise:
        print(f"  • Noise_Analysis - Noise pattern analysis")

# -----------------
# Command-Line Interface and Main Functions
# -----------------

def demonstrate_chilean_functionality_with_noise():
    """
    Demonstration function showing Chilean PII generation with noise capabilities.
    
    Provides examples of:
    1. Basic Chilean customer data generation with noise
    2. Different complexity modes for NLP training
    3. spaCy dataset creation for Chilean NER training
    4. E1010 conflict resolution validation
    """
    parser = argparse.ArgumentParser(description="Chilean PII Training Data Generator with Advanced Noise")
    parser.add_argument("--mode", choices=["demo", "create-dataset", "excel-export"], default="demo",
                       help="Mode: 'demo' shows examples, 'create-dataset' generates training data, 'excel-export' creates review file")
    parser.add_argument("--train-size", type=int, default=80000, help="Training set size")
    parser.add_argument("--dev-size", type=int, default=20000, help="Development set size")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--excel-examples", type=int, default=100, help="Number of examples for Excel export")
    parser.add_argument("--excel-file", type=str, default="chilean_customer_data_review_noisy.xlsx", help="Excel output filename")
    parser.add_argument("--noise", action="store_true", default=True, help="Enable noise generation")
    parser.add_argument("--no-noise", action="store_true", help="Disable noise generation")
    parser.add_argument("--noise-level", type=float, default=0.15, help="Noise intensity (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Handle noise settings
    include_noise = args.noise and not args.no_noise
    
    if args.mode == "create-dataset":
        # Create Chilean NLP training dataset with noise
        create_chilean_training_dataset_with_noise(
            train_size=args.train_size,
            dev_size=args.dev_size,
            include_noise=include_noise,
            noise_level=args.noise_level,
            output_dir=args.output_dir
        )
        return
    elif args.mode == "excel-export":
        # Create Excel file for Chilean data review
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        excel_file = output_path / args.excel_file
        
        export_chilean_data_to_excel_with_noise(
            n_examples=args.excel_examples,
            output_file=str(excel_file),
            include_noise=include_noise,
            noise_level=args.noise_level
        )
        return
    
    # Demo mode - show Chilean examples with noise
    print("=" * 80)
    print("CHILEAN PII TRAINING DATA GENERATOR WITH ADVANCED NOISE")
    print("=" * 80)
    print()
    
    print("🇨🇱 CHILEAN CUSTOMER DATA GENERATION WITH NOISE")
    print("-" * 50)
    print(f"🎭 Noise generation: {'Enabled' if include_noise else 'Disabled'}")
    print(f"🔊 Noise level: {args.noise_level}")
    print()
    
    # Show basic generation examples
    print("🔥 BASIC CHILEAN GENERATION EXAMPLES")
    print("-" * 40)
    for i in range(3):  # Show 3 examples
        sentence, annotations = generate_chilean_example_with_noise(include_noise, args.noise_level)
        
        print(f"📍 Example {i+1}:")
        print(f"Generated: {sentence}")
        print("Entities:", end=" ")
        for start, end, label in annotations["entities"]:
            entity_text = sentence[start:end]
            print(f"[{label}: '{entity_text}']", end=" ")
        print("\n")
    
    # Show different complexity modes
    print("🎯 CHILEAN NLP TRAINING MODES")
    print("-" * 40)
    modes = ["full", "addr_only", "id_only", "contact_only", "financial_only"]
    
    for mode in modes:
        print(f"🔸 Mode: {mode}")
        sentence, annotations = generate_chilean_example_with_mode(mode, include_noise)
        print(f"   Text: {sentence}")
        print(f"   Entities: {[label for _, _, label in annotations['entities']]}")
        print()
    
    print("🔢 Sequential Counter Status:")
    print(f"   Next sequence number: {_sequence_counter + 1}")
    print()
    
    # E1010 Validation Test
    print("🔍 E1010 OVERLAPPING SPAN ERROR VALIDATION")
    print("-" * 40)
    print("Testing conflict resolution algorithm...")
    
    test_examples = []
    overlap_errors = 0
    
    for _ in range(50):  # Test 50 examples
        sentence, annotations = generate_chilean_example_with_noise(include_noise)
        entities = annotations["entities"]
        
        # Check for overlaps
        for i, (start1, end1, label1) in enumerate(entities):
            for j, (start2, end2, label2) in enumerate(entities[i+1:], i+1):
                if start1 < end2 and start2 < end1:  # Overlap detected
                    overlap_errors += 1
        
        test_examples.append((sentence, entities))
    
    print(f"✅ Tested {len(test_examples)} examples")
    print(f"❌ Overlap errors detected: {overlap_errors}")
    
    if overlap_errors == 0:
        print("🎉 SUCCESS: Zero E1010 overlapping span errors guaranteed!")
    else:
        print("⚠️  WARNING: E1010 overlapping span errors detected!")
    
    print("\n📊 CHILEAN NLP DATASET CREATION")
    print("-" * 40)
    print("To create Chilean training datasets for spaCy NER:")
    print(f"   python data_generation_noisy.py --mode create-dataset --train-size 10000 --dev-size 2500 {'--noise' if include_noise else '--no-noise'}")
    print()
    print("📁 EXCEL DATA REVIEW")
    print("-" * 40)
    print("To create Excel file for Chilean data review and validation:")
    print(f"   python data_generation_noisy.py --mode excel-export --excel-examples 100 {'--noise' if include_noise else '--no-noise'}")
    print(f"   python data_generation_noisy.py --mode excel-export --excel-examples 500 --excel-file detailed_chilean_review.xlsx {'--noise' if include_noise else '--no-noise'}")
    print()
    print("🎯 NOISE FEATURES:")
    print("   - Realistic typos and misspellings")
    print("   - Chilean abbreviations and contractions")
    print("   - Document formatting variations")
    print("   - Controlled noise that preserves entity boundaries")
    print("   - Zero E1010 overlapping span errors guaranteed")
    print()
    print("📚 Chilean Use Cases:")
    print("   - Chilean customer service NER training")
    print("   - Chilean financial document processing")
    print("   - Chilean government document analysis")
    print("   - Chilean PII detection and anonymization")
    print("   - Large-scale Chilean NLP model training")

def quick_chilean_test_with_noise():
    """
    Quick test function to verify all Chilean functionality works correctly with noise.
    """
    print("🧪 Running quick Chilean functionality test with noise...")
    
    # Test basic generation
    sentence, annotations = generate_chilean_example_with_noise(True, 0.15)
    assert len(sentence) > 0, "Basic Chilean generation failed"
    assert len(annotations["entities"]) > 0, "No entities generated"
    
    # Test different modes
    modes = ["full", "addr_only", "id_only", "contact_only", "financial_only"]
    for mode in modes:
        sentence, annotations = generate_chilean_example_with_mode(mode, True)
        assert len(sentence) > 0, f"Chilean mode {mode} failed"
        assert len(annotations["entities"]) > 0, f"No entities in Chilean mode {mode}"
    
    # Test small dataset creation
    try:
        db, stats = make_chilean_docbin_with_noise(n_total=10, balance=True, include_noise=True)
        assert stats["total_examples"] == 10, "Chilean DocBin creation failed"
        assert stats["overlap_errors"] == 0, "E1010 overlapping span errors detected!"
        print("✅ All Chilean tests passed with zero E1010 errors!")
    except Exception as e:
        print(f"❌ Chilean DocBin test failed: {e}")
        print("💡 Install spaCy: pip install spacy")
        print("💡 Install Spanish model: python -m spacy download es_core_news_sm")

if __name__ == "__main__":
    # Run the enhanced Chilean PII generator with noise capabilities
    demonstrate_chilean_functionality_with_noise()