"""
Latin American Customer Data Generator for NLP Training

This module generates realistic customer data for Latin American countries specifically
designed for Named Entity Recognition (NER) training. It creates datasets with labeled
entities for customer service and financial NLP applications.

Key Features:
- Realistic customer data generation for 5 Latin American countries
- Named Entity Recognition (NER) annotations
- spaCy-compatible training data creation
- Balanced dataset generation with multiple complexity modes
- Export to .spacy format for training

Supported Countries:
- Chile (CL) - Spanish
- Argentina (AR) - Spanish  
- Brazil (BR) - Portuguese
- Uruguay (UY) - Spanish
- Mexico (MX) - Spanish

Entity Types:
- CUSTOMER_NAME: Full names (first + last)
- ID_NUMBER: Government identification numbers
- ADDRESS: Street addresses and cities
- PHONE_NUMBER: Country-specific phone formats
- EMAIL: Email addresses
- AMOUNT: Monetary amounts with currency
- SEQ_NUMBER: Sequential reference numbers

Author: Data Generation Team
Date: August 2025
Purpose: NLP/NER Model Training Dataset Creation
"""

import random
import spacy
from spacy.tokens import DocBin
from typing import Tuple, Dict, List, Any, Optional
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

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
# Customer Names Database
# -----------------
# Collection of common first names used across Latin American countries
# Includes both masculine and feminine names with proper Spanish/Portuguese accents
first_names = [
    # Masculine names (Spanish/Portuguese origin)
    "AGUSTÍN", "ALEJANDRO", "ALONSO", "ÁLVARO", "ANDRÉS", "ÁXEL", "BAUTISTA", "BENJAMÍN", "BRUNO", "CALEB",
            "CAMILO", "CARLOS", "CRISTÓBAL", "CRISTIAN", "DAMIÁN", "DANIEL", "DAVID", "DIEGO", "EDUARDO", "ELÍAS",
            "EMILIANO", "EMMANUEL", "ENRIQUE", "ESTEBAN", "ETHAN", "FEDERICO", "FERNANDO", "FRANCISCO", "GABRIEL",
            "GAEL", "GASPAR", "GERMÁN", "GUSTAVO", "HERNÁN", "IAN", "IGNACIO", "ISIDORO", "IVÁN", "JAIR", "JAIRO",
            "JASON", "JEREMY", "JHON", "JOAQUÍN", "JORGE", "JUAN", "JULIÁN", "KEVIN", "KIAN", "LEÓN", "LEONARDO",
            "LIAM", "LORENZO", "LUCCA", "LUIS", "MARCELO", "MARCO", "MARTÍN", "MATÍAS", "MATEO", "MAURICIO",
            "MAXIMILIANO", "MIGUEL", "NICOLÁS", "OLIVER", "OMAR", "ORLANDO", "PATRICIO", "PAULO", "PEDRO", "RAFAEL",
            "RAMIRO", "RICARDO", "ROBERTO", "RODRIGO", "RUBÉN", "SAMUEL", "SANTIAGO", "SEBASTIÁN", "SIMÓN", "THIAGO",
            "TOBÍAS", "TOMÁS", "VALENTINO", "VÍCTOR", "VICENTE", "WALTER", "XANDER", "ZAHIR",
            
    # Feminine names (Spanish/Portuguese origin)
    "AGUSTINA", "AINHOA", "AITANA", "ALBA", "ALEJANDRA", "ALEXA", "ALEXANDRA", "ALMENDRA", "AMANDA", "AMELIA",
            "ANAÍS", "ANTONELLA", "ANTONIA", "ARANTXA", "ARIADNA", "AROHA", "AZUL", "BELÉN", "BLANCA", "BRISA",
            "CAMILA", "CARLA", "CAROLINA", "CATALINA", "CELIA", "CLARA", "CLAUDIA", "CONSTANZA", "DANIELA", "DÉBORA",
            "DIANA", "DOMINIQUE", "ELISA", "ELIZABETH", "EMILIA", "EMMA", "ESMERALDA", "ESTEFANÍA", "FERNANDA",
            "FLORENCIA", "FRANCISCA", "GABRIELA", "GIOVANNA", "ISABELLA", "IVANNA", "JAVIERA", "JIMENA", "JOSEFINA",
            "JUANITA", "JULIETA", "KARINA", "KARLA", "KATIA", "KIARA", "LARA", "LAURA", "LAYLA", "LILA", "LUCIANA",
            "LUISA", "LUNA", "MACARENA", "MAGDALENA", "MANUELA", "MARÍA", "MARTINA", "MATILDA", "MÍA", "MILA",
            "MIREYA", "NATALIA", "NEREA", "NICOLE", "NOELIA", "OLIVIA", "PALOMA", "PAOLA", "PAULINA", "PAZ",
            "PENÉLOPE", "RENATA", "ROCÍO","ROSA", "ROMINA", "ROSARIO", "SALOMÉ", "SAMANTHA", "SARA", "SOFÍA", "SOL",
            "TAMARA", "VALENTINA", "VALERIA", "VANIA", "VERÓNICA", "VICTORIA", "VIOLETA", "XIMENA", "YASNA",
            "YOLANDA", "ZOE"
]

# Collection of common second names (middle names) in Latin America
# Very common to have compound first names like "Juan Carlos", "María José", "Ana Sofía"
second_names = [
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
names = first_names

# Collection of common surnames across Latin American countries
# Includes Spanish and Portuguese family names commonly found in the region
surnames = [
    # Spanish surnames (most common in Chile, Argentina, Uruguay, Mexico)
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
            "NAVARRETE", "CÁRDENAS", 
            
    # Portuguese/Brazilian surnames
    "DA SILVA", "DOS SANTOS", "DE SOUZA", "RODRIGUES", "COSTA", "SANTOS", "SOUSA",
            "OLIVEIRA", "LIMA", "PEREIRA"
]

# -----------------
# Address Database
# -----------------
# Collection of authentic street names from major Latin American cities
# Organized by country with real street names for realistic address generation
streets = [
    # --- Chile (15 streets) ---
    # Major avenues and streets from Santiago and other Chilean cities
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

    # --- Argentina (15 streets) ---
    # Famous streets and avenues from Buenos Aires and other Argentine cities
    "Av. 9 de Julio",                     # Widest avenue in the world (Buenos Aires)
    "Av. Corrientes",                     # Famous cultural avenue
    "Av. Santa Fe",                       # Major shopping avenue
    "Av. Rivadavia",                      # One of the longest streets in the world
    "Av. Callao",                         # Important north-south avenue
    "Calle San Martín",                   # Historic street
    "Calle Lavalle",                      # Pedestrian shopping street
    "Calle Perú",                         # Street in San Telmo
    "Pasaje Güemes",                      # Historic passage
    "Calle Defensa",                      # Famous street in San Telmo
    "Av. Belgrano",                       # Major east-west avenue
    "Av. Las Heras",                      # Avenue in Recoleta
    "Calle Florida",                      # Famous pedestrian shopping street
    "Calle Esmeralda",                    # Downtown street
    "Calle Tucumán",                      # Street in the microcentro

    # --- Brazil (15 streets) ---
    # Iconic streets from São Paulo, Rio de Janeiro, and other Brazilian cities
    "Av. Paulista",                       # Most famous avenue in São Paulo
    "Rua Augusta",                        # Important street in São Paulo
    "Av. Rio Branco",                     # Main avenue in Rio de Janeiro downtown
    "Av. Atlântica",                      # Beachfront avenue in Copacabana
    "Av. Beira-Mar",                      # Coastal avenue
    "Rua das Palmeiras",                  # Street with palm trees
    "Rua São João",                       # Historic street
    "Rua da Consolação",                  # Important street in São Paulo
    "Travessa das Laranjeiras",           # Small street (travessa)
    "Rua XV de Novembro",                 # Historic independence date street
    "Av. Brasil",                         # Common avenue name
    "Rua Frei Caneca",                    # Street in São Paulo
    "Rua da Carioca",                     # Street in Rio de Janeiro
    "Av. Presidente Vargas",              # Avenue named after president
    "Rua Visconde de Pirajá",             # Street in Ipanema

    # --- Mexico (15 streets) ---
    # Representative streets from Mexico City and other Mexican cities
    "Paseo de la Reforma",                # Most important avenue in Mexico City
    "Av. Insurgentes",                    # Longest avenue in Mexico City
    "Calzada de Tlalpan",                 # Major south avenue
    "Av. Juárez",                         # Historic avenue
    "Av. Universidad",                    # Avenue near UNAM
    "Calle Madero",                       # Historic downtown pedestrian street
    "Calle López",                        # Common street name
    "Calle Tacuba",                       # Historic street
    "Calle República de Cuba",            # Street in downtown Mexico City
    "Privada Las Rosas",                  # Private residential street
    "Av. Revolución",                     # Avenue commemorating the Revolution
    "Eje Central Lázaro Cárdenas",        # Major north-south axis
    "Av. Álvaro Obregón",                 # Avenue named after president
    "Calle Durango",                      # Street in Colonia Roma
    "Calle Londres",                      # Street in Zona Rosa

    # --- Uruguay (15 streets) ---
    # Representative streets from Montevideo and other Uruguayan cities
    "Bvar. Artigas",                      # Boulevard named after national hero
    "Av. 18 de Julio",                    # Main avenue in Montevideo
    "Av. Italia",                         # Important avenue
    "Av. Rivera",                         # Major avenue
    "Av. Gral. Flores",                   # Avenue named after general
    "Calle Colonia",                      # Historic street
    "Calle Soriano",                      # Street in downtown Montevideo
    "Calle Yi",                           # Short downtown street
    "Calle Durazno",                      # Street named after fruit
    "Pasaje Pérez Castellanos",           # Small passage
    "Calle Canelones",                    # Street named after department
    "Calle San José",                     # Street named after saint
    "Calle Río Branco",                   # Street named after river
    "Calle Cerro Largo",                  # Street named after department
    "Calle Andes"                         # Street named after mountain range
]

# Collection of major cities from Latin American countries
cities = [
    # --- Chile ---
    "Santiago",       # Capital and largest city
    "Valparaíso",     # Historic port city
    "Concepción",     # Major southern city
    "La Serena",      # Northern coastal city
    "Antofagasta",    # Mining city in the north

    # --- Argentina ---
    "Buenos Aires",   # Capital and largest city
    "Córdoba",        # Second largest city
    "Rosario",        # Important industrial city
    "Mendoza",        # Wine region capital
    "La Plata",       # Capital of Buenos Aires Province

    # --- Brazil ---
    "São Paulo",      # Largest city in South America
    "Rio de Janeiro", # Former capital, major tourist destination
    "Belo Horizonte", # Major southeastern city
    "Brasília",       # Current capital
    "Salvador",       # Historic northeastern city

    # --- Mexico ---
    "Ciudad de México", # Capital and largest city
    "Guadalajara",      # Second largest city
    "Monterrey",        # Major industrial city
    "Puebla",           # Historic colonial city
    "Tijuana",          # Border city with USA

    # --- Uruguay ---
    "Montevideo",     # Capital and largest city
    "Salto",          # Second largest city
    "Paysandú",       # Important river port
    "Las Piedras",    # Suburban city near Montevideo
    "Rivera"          # Border city with Brazil
]

# -----------------
# Data Generation Functions
# -----------------

def generate_full_name(include_second_name: bool = True, probability: float = 0.4, include_second_surname: bool = True, surname_probability: float = 0.8) -> str:
    """
    Generate a realistic full name with optional second name (middle name) and second surname.
    
    In Latin American countries, it's very common to have:
    - Compound first names: "Juan Carlos", "María José", "Ana Sofía", "Luis Miguel"
    - Two surnames (paternal + maternal): "González Rodríguez", "Silva Martínez"
    
    Args:
        include_second_name (bool): Whether to potentially include a second name
        probability (float): Probability of including a second name (0.0 to 1.0)
        include_second_surname (bool): Whether to potentially include a second surname
        surname_probability (float): Probability of including a second surname (0.0 to 1.0)
    
    Returns:
        str: Full name with optional second name and optional second surname
        
    Examples:
        "JUAN CARLOS GONZÁLEZ RODRÍGUEZ"
        "MARÍA JOSÉ SILVA MARTÍNEZ" 
        "PEDRO SANTOS LÓPEZ" (no second name, but second surname)
        "ANA GARCÍA" (no second name or surname)
    """
    first_name = random.choice(first_names)
    paternal_surname = random.choice(surnames)
    
    # Build the name progressively
    name_parts = [first_name]
    
    # Add second name if requested
    if include_second_name and random.random() < probability:
        second_name = random.choice(second_names)
        name_parts.append(second_name)
    
    # Add paternal surname
    name_parts.append(paternal_surname)
    
    # Add maternal surname if requested
    if include_second_surname and random.random() < surname_probability:
        maternal_surname = random.choice(surnames)
        # Ensure maternal surname is different from paternal
        while maternal_surname == paternal_surname:
            maternal_surname = random.choice(surnames)
        name_parts.append(maternal_surname)
    
    return " ".join(name_parts)


def generate_name_components(include_second_name: bool = True, probability: float = 0.4, include_second_surname: bool = True, surname_probability: float = 0.8) -> Tuple[str, str, str]:
    """
    Generate name components separately for more flexible use.
    
    Args:
        include_second_name (bool): Whether to potentially include a second name
        probability (float): Probability of including a second name (0.0 to 1.0)
        include_second_surname (bool): Whether to potentially include a second surname
        surname_probability (float): Probability of including a second surname (0.0 to 1.0)
    
    Returns:
        Tuple[str, str, str]: (first_name, full_name_part, complete_surname) where:
            - first_name: Just the first name for email generation
            - full_name_part: First name + optional second name
            - complete_surname: Paternal surname + optional maternal surname
    """
    first_name = random.choice(first_names)
    paternal_surname = random.choice(surnames)
    
    # Generate first name part (with optional second name)
    if include_second_name and random.random() < probability:
        second_name = random.choice(second_names)
        full_name_part = f"{first_name} {second_name}"
    else:
        full_name_part = first_name
    
    # Generate complete surname (with optional maternal surname)
    if include_second_surname and random.random() < surname_probability:
        maternal_surname = random.choice(surnames)
        # Ensure maternal surname is different from paternal
        while maternal_surname == paternal_surname:
            maternal_surname = random.choice(surnames)
        complete_surname = f"{paternal_surname} {maternal_surname}"
    else:
        complete_surname = paternal_surname
    
    return first_name, full_name_part, complete_surname


def random_phone(country: str) -> str:
    """
    Generate a realistic phone number for the specified country.
    
    Phone number formats follow each country's standard:
    - Chile (CL): +56 9 XXXX XXXX (mobile format)
    - Argentina (AR): +54 11 XXXX XXXX (Buenos Aires area code)
    - Brazil (BR): +55 21 9XXXX-XXXX (Rio area code with mobile prefix)
    - Uruguay (UY): +598 9 XXX XXX (mobile format)
    - Mexico (MX): +52 55 XXXX XXXX (Mexico City area code)
    - Other: +1 XXX XXX XXXX (US format as fallback)
    
    Args:
        country (str): Two-letter country code (CL, AR, BR, UY, MX)
        
    Returns:
        str: Formatted phone number with country code
    """
    if country == "CL":  # Chile
        return f"+56 9 {random.randint(1000,9999)} {random.randint(1000,9999)}"
    elif country == "AR":  # Argentina
        return f"+54 11 {random.randint(1000,9999)} {random.randint(1000,9999)}"
    elif country == "BR":  # Brazil
        return f"+55 21 9{random.randint(1000,9999)}-{random.randint(1000,9999)}"
    elif country == "UY":  # Uruguay
        return f"+598 9 {random.randint(100,999)} {random.randint(100,999)}"
    elif country == "MX":  # Mexico
        return f"+52 55 {random.randint(1000,9999)} {random.randint(1000,9999)}"
    else:  # Default to US format
        return f"+1 {random.randint(200,999)} {random.randint(100,999)} {random.randint(1000,9999)}"


def random_email(name: str, surname: str) -> str:
    """
    Generate a realistic email address using the person's name and surname.
    
    Creates email in format: firstname.lastname@domain.com
    Uses common email providers in Latin America.
    For double surnames, uses only the paternal (first) surname.
    
    Args:
        name (str): First name of the person
        surname (str): Complete surname (may include paternal and maternal)
        
    Returns:
        str: Email address in lowercase
    """
    # Common email domains in Latin America
    domains = ["gmail.com", "hotmail.com", "yahoo.com", "outlook.com"]
    
    # Use only the first surname for email (paternal surname)
    first_surname = surname.split()[0] if " " in surname else surname
    
    return f"{name.lower()}.{first_surname.lower()}@{random.choice(domains)}"


def random_id(country: str) -> str:
    """
    Generate a realistic identification number for the specified country.
    
    Each country has its own ID format:
    - Chile (CL): RUT format XX.XXX.XXX-X
    - Argentina (AR): DNI format XXXXXXXX
    - Brazil (BR): CPF format XXX.XXX.XXX-XX
    - Uruguay (UY): CI format X.XXX.XXX-X
    - Mexico (MX): CURP-like format (simplified)
    - Other: 8-digit number
    
    Args:
        country (str): Two-letter country code
        
    Returns:
        str: Formatted identification number
    """
    if country == "CL":  # Chilean RUT (Rol Único Tributario)
        return f"{random.randint(10,30)}.{random.randint(100,999)}.{random.randint(100,999)}-{random.randint(0,9)}"
    elif country == "AR":  # Argentina DNI (Documento Nacional de Identidad)
        return f"{random.randint(10_000_000, 45_000_000)}"
    elif country == "BR":  # Brazilian CPF (Cadastro de Pessoas Físicas)
        return f"{random.randint(100,999)}.{random.randint(100,999)}.{random.randint(100,999)}-{random.randint(10,99)}"
    elif country == "UY":  # Uruguay CI (Cédula de Identidad)
        return f"{random.randint(1,9)}.{random.randint(100,999)}.{random.randint(100,999)}-{random.randint(0,9)}"
    elif country == "MX":  # Mexico CURP-like (simplified version)
        return f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('AEIOU')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(10,99)}{random.randint(1,12):02}{random.randint(1,28):02}HDF{random.randint(1000,9999)}"
    else:  # Generic 8-digit ID
        return str(random.randint(10000000, 99999999))


def random_amount(country: str) -> str:
    """
    Generate a realistic monetary amount for the specified country.
    
    Amounts are generated within typical ranges for each country's currency:
    - Chile (CL): 10,000 - 900,000 CLP
    - Argentina (AR): 5,000 - 700,000 ARS
    - Brazil (BR): 50 - 5,000 BRL
    - Uruguay (UY): 1,000 - 200,000 UYU
    - Mexico (MX): 500 - 100,000 MXN
    - Other: 100 - 5,000 USD
    
    Args:
        country (str): Two-letter country code
        
    Returns:
        str: Formatted amount with currency symbol and code
    """
    if country == "CL":  # Chile - Chilean Peso
        return f"${random.randint(10_000, 900_000):,} CLP"
    elif country == "AR":  # Argentina - Argentine Peso
        return f"${random.randint(5_000, 700_000):,} ARS"
    elif country == "BR":  # Brazil - Brazilian Real
        return f"R$ {random.randint(50, 5000):,} BRL"
    elif country == "UY":  # Uruguay - Uruguayan Peso
        return f"${random.randint(1000, 200_000):,} UYU"
    elif country == "MX":  # Mexico - Mexican Peso
        return f"${random.randint(500, 100_000):,} MXN"
    else:  # Default to US Dollar
        return f"${random.randint(100, 5000):,} USD"


def random_sequence_number(country: str) -> str:
    """
    Generate a realistic sequential number for different business contexts.
    
    Creates varied sequence identifiers used in real business scenarios:
    - Complaint numbers: 7-digit numbers (7009808)
    - Response IDs: 7-digit numbers (6039383) 
    - Purchase IDs: 7-digit numbers (5656575)
    - Policy numbers: 5-digit + letter (57575-A)
    - Transaction IDs: Country prefix format (CL-10001)
    - Case numbers: 6-digit numbers (483729)
    - Order numbers: 8-digit numbers (20240815)
    
    Args:
        country (str): Two-letter country code
        
    Returns:
        str: Formatted sequence number in various realistic formats
    """
    import random
    
    # Define different sequence number formats with their probabilities
    sequence_types = [
        # Pure numeric sequences (most common in business)
        ("complaint", 0.20),      # Reclamo: 7009808
        ("response", 0.15),       # Respuesta: 6039383  
        ("purchase", 0.20),       # ID Compra: 5656575
        ("case", 0.15),           # Caso: 483729
        ("order", 0.15),          # Pedido: 20240815
        # Alphanumeric sequences
        ("policy", 0.10),         # Póliza: 57575-A
        # Traditional country prefix (backward compatibility)
        ("country_prefix", 0.05)  # CL-10001
    ]
    
    # Select sequence type based on probability weights
    rand_val = random.random()
    cumulative = 0
    selected_type = "complaint"  # default
    
    for seq_type, probability in sequence_types:
        cumulative += probability
        if rand_val <= cumulative:
            selected_type = seq_type
            break
    
    # Generate sequence based on selected type
    if selected_type == "complaint":
        # 7-digit complaint numbers (7000000-7999999)
        return str(random.randint(7000000, 7999999))
        
    elif selected_type == "response":
        # 7-digit response numbers (6000000-6999999) 
        return str(random.randint(6000000, 6999999))
        
    elif selected_type == "purchase":
        # 7-digit purchase IDs (5000000-5999999)
        return str(random.randint(5000000, 5999999))
        
    elif selected_type == "case":
        # 6-digit case numbers (400000-599999)
        return str(random.randint(400000, 599999))
        
    elif selected_type == "order":
        # 8-digit order numbers (date-like format)
        return str(random.randint(20240001, 20241231))
        
    elif selected_type == "policy":
        # 5-digit + letter policy numbers (10000-99999 + A-Z)
        base_number = random.randint(10000, 99999)
        letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return f"{base_number}-{letter}"
        
    else:  # country_prefix (original format for backward compatibility)
        sequence = get_next_sequence()
        return f"{country}-{sequence:05d}"


# -----------------
# Text Templates for Data Generation
# -----------------

# Collection of text templates in Spanish and Portuguese for generating realistic sentences
# Templates use placeholders for: name, surname, id, address, city, phone, email, amount, sequence
templates = [
    # Spanish templates
    "El cliente {} {} con identificación {} y dirección {} en {} tiene el teléfono {} y el correo {}. Monto: {}. Consulta N°: {}.",
    "{} {} registró su documento {} en el sistema. Dirección: {}, {}. Tel: {}. Email: {}. Pago pendiente de {}. Ref: {}.",
    "El usuario {} {} (ID: {}) vive en {}, {}. Teléfono: {}. Email: {}. Saldo actual: {}. Operación: {}.",
    
    # Portuguese template
    "O CPF de {} {} é {}. Endereço: {}, {}. Telefone: {}. Email: {}. Valor da compra: {}. Número: {}.",
    
    # Additional templates
    "Registro de {} {} con cédula {}. Ubicado en {}, {}. Contacto: {} / {}. Transacción: {} - ID: {}."
]

# -----------------
# Main Data Generator Function
# -----------------
def generate_example(country: str = "CL") -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
    """
    Generate a complete customer data example for the specified country.
    
    Creates realistic customer information including:
    - Full name (first + optional second name + surname)
    - Government ID number
    - Complete address (street + city)
    - Phone number with country code
    - Email address
    - Monetary amount in local currency
    - Sequential reference number
    
    The function also performs Named Entity Recognition (NER) annotation,
    identifying the positions of each data type within the generated text.
    
    Args:
        country (str): Two-letter country code (CL, AR, BR, UY, MX)
                      Defaults to "CL" (Chile)
    
    Returns:
        Tuple[str, Dict]: A tuple containing:
            - str: Generated sentence with customer data
            - Dict: NER annotations with entity positions and labels
                   Format: {"entities": [(start, end, label), ...]}
    
    Example:
        >>> sentence, annotations = generate_example("CL")
        >>> print(sentence)
        "El cliente JUAN CARLOS GONZÁLEZ con identificación 15.234.567-8..."
        >>> print(annotations)
        {"entities": [(11, 29, "CUSTOMER_NAME"), (45, 56, "ID_NUMBER"), ...]}
    """
    # Generate individual customer data components using new name system with second surname
    first_name, full_name_part, complete_surname = generate_name_components(
        include_second_name=True, probability=0.4,
        include_second_surname=True, surname_probability=0.8
    )
    complete_full_name = f"{full_name_part} {complete_surname}"    # Complete name for entity recognition
    
    # Generate country-specific identification
    id_number = random_id(country)
    
    # Create realistic address
    street = random.choice(streets)               # Select random street
    street_number = random.randint(10, 999)      # Generate street number
    address = f"{street} {street_number}"         # Complete street address
    city = random.choice(cities)                  # Select random city
    
    # Generate contact information
    phone = random_phone(country)                 # Country-specific phone format
    email = random_email(first_name, complete_surname)  # Email based on first name and paternal surname
    
    # Generate financial information
    amount = random_amount(country)               # Country-specific currency/amount
    sequence = random_sequence_number(country)    # Unique sequential ID
    
    # Select random template and format with generated data
    template = random.choice(templates)
    
    # Extract individual names for template formatting
    name_parts = full_name_part.split()
    if len(name_parts) == 2:  # Has second name
        first, second = name_parts
        sentence = template.format(first, f"{second} {complete_surname}", id_number, address, city, 
                                  phone, email, amount, sequence)
    else:  # No second name
        sentence = template.format(full_name_part, complete_surname, id_number, address, city, 
                                  phone, email, amount, sequence)
    
    # Perform Named Entity Recognition (NER) annotation
    # Find the position of each entity within the generated sentence
    entities = []
    entity_mappings = [
        (complete_full_name, "CUSTOMER_NAME"),    # Full customer name with potential second name
        (id_number, "ID_NUMBER"),                 # Government identification
        (address, "ADDRESS"),                     # Street address
        (city, "ADDRESS"),                        # City (also tagged as ADDRESS)
        (phone, "PHONE_NUMBER"),                  # Phone number with country code
        (email, "EMAIL"),                         # Email address
        (amount, "AMOUNT"),                       # Monetary amount with currency
        (sequence, "SEQ_NUMBER")                  # Sequential reference number
    ]
    
    # Improved entity detection with conflict resolution
    used_positions = set()
    
    # Sort entities by length (longest first) to prioritize longer matches
    sorted_mappings = sorted(entity_mappings, key=lambda x: len(x[0]), reverse=True)
    
    for entity_text, label in sorted_mappings:
        if not entity_text.strip():  # Skip empty entities
            continue
            
        # Try exact match first
        start_pos = sentence.find(entity_text)
        if start_pos != -1:
            end_pos = start_pos + len(entity_text)
            
            # Check if this position overlaps with already used positions
            position_range = set(range(start_pos, end_pos))
            if not position_range.intersection(used_positions):
                entities.append((start_pos, end_pos, label))
                used_positions.update(position_range)
    
    # Sort entities by start position for consistent output
    entities.sort(key=lambda x: x[0])
    
    return (sentence, {"entities": entities})
    
    return (sentence, {"entities": entities})


def generate_multiple_examples(country: str = "CL", count: int = 5) -> List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]]:
    """
    Generate multiple customer data examples for training or testing purposes.
    
    Useful for creating datasets for machine learning models, particularly
    for Named Entity Recognition (NER) training in customer service or
    financial applications.
    
    Args:
        country (str): Two-letter country code (CL, AR, BR, UY, MX)
        count (int): Number of examples to generate
    
    Returns:
        List[Tuple]: List of (sentence, annotations) tuples
    """
    examples = []
    for _ in range(count):
        example = generate_example(country)
        examples.append(example)
    return examples


# -----------------
# Enhanced NLP Dataset Generation
# -----------------

def generate_example_with_mode(country: str = "CL", mode: str = "full") -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
    """
    Generate customer data with specific entity complexity modes for varied NLP training.
    
    This function creates different complexity levels of entity annotations to improve
    model robustness and handle various real-world scenarios.
    
    Args:
        country (str): Two-letter country code (CL, AR, BR, UY, MX)
        mode (str): Complexity mode for entity generation
            - "full": All entity types (name, id, address, phone, email, amount, sequence)
            - "essential_only": Focus on essential entities: names, RUT/ID, and addresses
            - "addr_only": Focus on names and addresses only
            - "id_only": Focus on names and ID numbers only  
            - "contact_only": Focus on names, phone, and email only
            - "financial_only": Focus on names, amounts, and sequences only
            - "minimal": Only names and one random entity type
    
    Returns:
        Tuple[str, Dict]: Generated sentence with targeted entity annotations
    
    Example:
        >>> sentence, annotations = generate_example_with_mode("CL", "addr_only")
        # Will focus on CUSTOMER_NAME and ADDRESS entities only
    """
    # Generate all data components using new name system with second surname
    first_name, full_name_part, complete_surname = generate_name_components(
        include_second_name=True, probability=0.4,
        include_second_surname=True, surname_probability=0.8
    )
    complete_full_name = f"{full_name_part} {complete_surname}"
    
    id_number = random_id(country)
    street = random.choice(streets)
    street_number = random.randint(10, 999)
    address = f"{street} {street_number}"
    city = random.choice(cities)
    phone = random_phone(country)
    email = random_email(first_name, complete_surname)  # Use first name and complete surname
    amount = random_amount(country)
    sequence = random_sequence_number(country)
    
    # Helper function to format names in templates
    def format_template_with_names(template, *args):
        name_parts = full_name_part.split()
        if len(name_parts) == 2:  # Has second name
            first, second = name_parts
            return template.format(first, f"{second} {complete_surname}", *args)
        else:  # No second name
            return template.format(full_name_part, complete_surname, *args)
    
    # Select templates and entities based on mode
    if mode == "essential_only":
        # Templates focusing on ESSENTIAL entities: names, RUT/ID, and addresses
        mode_templates = [
            "El cliente {} {} con RUT {} vive en {}, {}.",
            "Registro de {} {} RUT {} domiciliado en {}, {}.",
            "Cliente {} {} (RUT: {}) reside en {}, {}.",
            "{} {} con identificación {} ubicado en {}, {}.",
            "Datos de {} {}: RUT {}, dirección {}, {}."
        ]
        entity_mappings = [
            (complete_full_name, "CUSTOMER_NAME"),
            (id_number, "ID_NUMBER"),
            (address, "ADDRESS"),
            (city, "ADDRESS")
        ]
        template = random.choice(mode_templates)
        sentence = format_template_with_names(template, id_number, address, city)
        
    elif mode == "addr_only":
        # Templates focusing on names and addresses
        mode_templates = [
            "El cliente {} {} vive en {}, {}.",
            "Registro de {} {} con dirección en {}, {}.",
            "Cliente {} {} domiciliado en {}, {}.",
            "{} {} reside en {}, {}."
        ]
        entity_mappings = [
            (complete_full_name, "CUSTOMER_NAME"),
            (address, "ADDRESS"),
            (city, "ADDRESS")
        ]
        template = random.choice(mode_templates)
        sentence = format_template_with_names(template, address, city)
        
    elif mode == "id_only":
        # Templates focusing on names and IDs
        mode_templates = [
            "El documento de {} {} es {}.",
            "Cliente {} {} con identificación {}.",
            "{} {} registrado con ID {}.",
            "Identificación de {} {}: {}."
        ]
        entity_mappings = [
            (complete_full_name, "CUSTOMER_NAME"),
            (id_number, "ID_NUMBER")
        ]
        template = random.choice(mode_templates)
        sentence = format_template_with_names(template, id_number)
        
    elif mode == "contact_only":
        # Templates focusing on contact information
        mode_templates = [
            "Contactar a {} {} al {} o {}.",
            "{} {} - Tel: {} Email: {}.",
            "Cliente {} {} disponible en {} / {}.",
            "Datos de contacto: {} {} - {} - {}."
        ]
        entity_mappings = [
            (complete_full_name, "CUSTOMER_NAME"),
            (phone, "PHONE_NUMBER"),
            (email, "EMAIL")
        ]
        template = random.choice(mode_templates)
        sentence = format_template_with_names(template, phone, email)
        
    elif mode == "financial_only":
        # Templates focusing on financial information
        mode_templates = [
            "Transacción {} para {} {} por {}.",
            "{} {} - Monto: {} - Ref: {}.",
            "Cliente {} {} procesó {} con número {}.",
            "Operación {}: {} {} pagó {}."
        ]
        entity_mappings = [
            (complete_full_name, "CUSTOMER_NAME"),
            (amount, "AMOUNT"),
            (sequence, "SEQ_NUMBER")
        ]
        template = random.choice(mode_templates)
        if "Transacción" in template:
            sentence = template.format(sequence, full_name_part, complete_surname, amount)
        elif "Operación" in template:
            sentence = template.format(sequence, full_name_part, complete_surname, amount)
        else:
            sentence = format_template_with_names(template, amount, sequence)
        
    elif mode == "minimal":
        # Templates with minimal information
        minimal_options = [
            ("id", [
                "Cliente {} {} ID: {}.",
                "{} {} - {}."
            ], [(complete_full_name, "CUSTOMER_NAME"), (id_number, "ID_NUMBER")]),
            ("phone", [
                "{} {} teléfono {}.",
                "Llamar a {} {} al {}."
            ], [(complete_full_name, "CUSTOMER_NAME"), (phone, "PHONE_NUMBER")]),
            ("email", [
                "{} {} correo: {}.",
                "Email de {} {}: {}."
            ], [(complete_full_name, "CUSTOMER_NAME"), (email, "EMAIL")])
        ]
        
        choice = random.choice(minimal_options)
        _, mode_templates, entity_mappings = choice
        template = random.choice(mode_templates)
        
        if "ID" in template:
            sentence = format_template_with_names(template, id_number)
        elif "teléfono" in template or "Llamar" in template:
            sentence = format_template_with_names(template, phone)
        else:  # email templates
            sentence = format_template_with_names(template, email)
            
    else:  # mode == "full" or any other value
        # Use the original full-featured generation
        return generate_example(country)
    
    # Find entity positions in the generated sentence
    entities = []
    for entity_text, label in entity_mappings:
        start_pos = sentence.find(entity_text)
        if start_pos != -1:
            end_pos = start_pos + len(entity_text)
            entities.append((start_pos, end_pos, label))
    
    return (sentence, {"entities": entities})


def make_docbin(n_total: int = 100000, balance: bool = True, output_dir: str = ".") -> Tuple[DocBin, Dict[str, int]]:
    """
    Create a spaCy DocBin for NER training with balanced entity distribution.
    
    This function generates a comprehensive dataset with varied complexity modes
    to ensure robust NER model training across different customer service scenarios.
    
    Args:
        n_total (int): Total number of examples to generate
        balance (bool): Whether to balance examples across countries
        output_dir (str): Directory to save the training files
    
    Returns:
        Tuple[DocBin, Dict]: DocBin object and statistics about generation
        
    Entity Distribution Strategy:
        - 30% full complexity (all entities)
        - 25% address-focused (names + addresses)
        - 20% ID-focused (names + IDs)
        - 15% contact-focused (names + contact info)
        - 10% financial-focused (names + amounts)
    """
    # Define mode distribution for balanced training
    mode_choices = (
        ["full"] * 3 +           # 30% - Complete entity sets
        ["addr_only"] * 25 +     # 25% - Address focus 
        ["id_only"] * 2 +        # 20% - ID focus
        ["contact_only"] * 15 +  # 15% - Contact focus
        ["financial_only"] * 1   # 10% - Financial focus
    )
    
    countries = ["CL", "AR", "BR", "UY", "MX"]
    
    # Use the best available Spanish model for Latin American text processing
    # Priority: lg (large) > md (medium) > sm (small) > blank
    try:
        nlp = spacy.load("es_core_news_lg")  # Best accuracy with word vectors
        print("✅ Using Spanish Large model (es_core_news_lg)")
    except OSError:
        try:
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
    
    # Calculate per-country distribution
    per_country = n_total // len(countries) if balance else None
    
    # Statistics tracking
    count = {c: 0 for c in countries}
    mode_stats = {mode: 0 for mode in set(mode_choices)}
    entity_stats = {}
    
    created = 0
    failed_spans = 0
    
    print(f"🏗️  Generating {n_total} NLP training examples...")
    print(f"📊 Balance mode: {'Enabled' if balance else 'Disabled'}")
    
    while created < n_total:
        # Select country (with balancing if enabled)
        country = random.choice(countries)
        if balance and count[country] >= per_country:
            continue
            
        # Select complexity mode
        mode = random.choice(mode_choices)
        
        try:
            # Generate example with selected mode
            text, annotations = generate_example_with_mode(country, mode)
            
            # Create spaCy document
            doc = nlp.make_doc(text)
            spans = []
            
            # Convert annotations to spaCy spans
            for (start, end, label) in annotations["entities"]:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is not None:
                    spans.append(span)
                    # Track entity statistics
                    entity_stats[label] = entity_stats.get(label, 0) + 1
                else:
                    failed_spans += 1
            
            # Set entities and add to DocBin
            doc.ents = spans
            db.add(doc)
            
            # Update counters
            count[country] += 1
            mode_stats[mode] += 1
            created += 1
            
            # Progress indicator
            if created % 1000 == 0:
                print(f"  ✅ Generated {created}/{n_total} examples")
                
        except Exception as e:
            print(f"  ⚠️  Error generating example: {e}")
            continue
    
    # Compile statistics
    stats = {
        "total_examples": created,
        "countries": dict(count),
        "modes": dict(mode_stats),
        "entities": dict(entity_stats),
        "failed_spans": failed_spans
    }
    
    print(f"✅ Dataset generation complete!")
    print(f"📈 Statistics:")
    print(f"   - Total examples: {created}")
    print(f"   - Failed spans: {failed_spans}")
    print(f"   - Countries: {dict(count)}")
    print(f"   - Entity types: {len(entity_stats)}")
    
    return db, stats


def create_training_dataset(train_size: int = 80000, dev_size: int = 20000, output_dir: str = ".", save_stats: bool = True) -> None:
    """
    Create complete training and development datasets for spaCy NER training.
    
    This function generates balanced training and development sets with detailed
    statistics and saves them in spaCy's native format for efficient training.
    
    Args:
        train_size (int): Number of training examples
        dev_size (int): Number of development/validation examples  
        output_dir (str): Output directory for files
        save_stats (bool): Whether to save detailed statistics
    
    Creates:
        - train.spacy: Training dataset
        - dev.spacy: Development/validation dataset
        - dataset_stats.json: Detailed generation statistics (if save_stats=True)
    
    Example:
        >>> create_training_dataset(train_size=10000, dev_size=2500)
        # Creates train.spacy (10K examples) and dev.spacy (2.5K examples)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("🚀 Creating NLP Training Dataset for Latin American Customer NER")
    print("=" * 70)
    
    # Generate training set
    print("\n📚 TRAINING SET")
    print("-" * 30)
    train_db, train_stats = make_docbin(n_total=train_size, balance=True, output_dir=output_dir)
    train_file = output_path / "train.spacy"
    train_db.to_disk(train_file)
    
    # Generate development set  
    print(f"\n🔬 DEVELOPMENT SET")
    print("-" * 30)
    dev_db, dev_stats = make_docbin(n_total=dev_size, balance=True, output_dir=output_dir)
    dev_file = output_path / "dev.spacy"
    dev_db.to_disk(dev_file)
    
    # Save comprehensive statistics
    if save_stats:
        combined_stats = {
            "dataset_info": {
                "purpose": "Latin American Customer NER Training",
                "train_size": train_size,
                "dev_size": dev_size,
                "total_size": train_size + dev_size,
                "countries": ["Chile", "Argentina", "Brazil", "Uruguay", "Mexico"],
                "languages": ["Spanish", "Portuguese"],
                "entity_types": ["CUSTOMER_NAME", "ID_NUMBER", "ADDRESS", "PHONE_NUMBER", "EMAIL", "AMOUNT", "SEQ_NUMBER"]
            },
            "train_stats": train_stats,
            "dev_stats": dev_stats
        }
        
        stats_file = output_path / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Dataset Statistics saved to: {stats_file}")
    
    print(f"\n🎯 TRAINING FILES CREATED:")
    print(f"   📁 Training set: {train_file} ({train_size:,} examples)")
    print(f"   📁 Dev set: {dev_file} ({dev_size:,} examples)")
    print(f"   📁 Total examples: {train_size + dev_size:,}")
    
    print(f"\n🔧 NEXT STEPS:")
    print(f"   1. Install spaCy: pip install spacy")
    print(f"   2. Create config: python -m spacy init config config.cfg --lang es --pipeline ner")
    print(f"   3. Train model: python -m spacy train config.cfg --output ./model --paths.train {train_file} --paths.dev {dev_file}")
    
    print(f"\n✨ Ready for NER training!")


def export_to_excel(n_examples: int = 100, output_file: str = "customer_data_review.xlsx", countries: Optional[List[str]] = None) -> None:
    """
    Export generated customer data to Excel for easy review and validation.
    
    Creates a comprehensive Excel file with separate sheets for different aspects
    of the generated data, making it easy to review the quality and accuracy.
    
    Args:
        n_examples (int): Number of examples to generate for review
        output_file (str): Name of the Excel file to create
        countries (List[str], optional): List of country codes to include. 
                                       If None, uses all supported countries.
    
    Creates Excel sheets:
        - Summary: Overview and statistics
        - All_Data: Complete dataset with all fields
        - By_Country: Data grouped by country
        - Name_Analysis: Analysis of naming patterns
        - Entity_Statistics: Count of each entity type
    """
    if countries is None:
        countries = ["CL", "AR", "BR", "UY", "MX"]
    
    print(f"📊 Generating {n_examples} examples for Excel review...")
    
    # Generate data for all countries
    all_data = []
    country_stats = {}
    entity_stats = {}
    name_patterns = {"has_second_name": 0, "has_second_surname": 0, "compound_names": 0}
    
    examples_per_country = n_examples // len(countries)
    
    for country in countries:
        country_data = []
        print(f"   🌍 Generating {examples_per_country} examples for {country}...")
        
        for i in range(examples_per_country):
            # Generate example data
            sentence, annotations = generate_example(country)
            
            # Extract entities from annotations
            entities_dict = {}
            for start, end, label in annotations["entities"]:
                entity_text = sentence[start:end]
                if label not in entities_dict:
                    entities_dict[label] = []
                entities_dict[label].append(entity_text)
                entity_stats[label] = entity_stats.get(label, 0) + 1
            
            # Get individual components for detailed analysis
            first_name, full_name_part, complete_surname = generate_name_components(
                include_second_name=True, probability=0.4,
                include_second_surname=True, surname_probability=0.8
            )
            
            # Analyze naming patterns
            if " " in full_name_part:
                name_patterns["has_second_name"] += 1
            if " " in complete_surname:
                name_patterns["has_second_surname"] += 1
            if " " in full_name_part and " " in complete_surname:
                name_patterns["compound_names"] += 1
            
            # Create row data
            row_data = {
                "Example_ID": f"{country}-{i+1:03d}",
                "Country": country,
                "Complete_Sentence": sentence,
                "Customer_Name": entities_dict.get("CUSTOMER_NAME", [""])[0] if "CUSTOMER_NAME" in entities_dict else "",
                "First_Name_Only": first_name,
                "Full_Name_Part": full_name_part,
                "Complete_Surname": complete_surname,
                "Has_Second_Name": "Yes" if " " in full_name_part else "No",
                "Has_Second_Surname": "Yes" if " " in complete_surname else "No",
                "ID_Number": entities_dict.get("ID_NUMBER", [""])[0] if "ID_NUMBER" in entities_dict else "",
                "Address": entities_dict.get("ADDRESS", [""])[0] if "ADDRESS" in entities_dict else "",
                "Phone": entities_dict.get("PHONE_NUMBER", [""])[0] if "PHONE_NUMBER" in entities_dict else "",
                "Email": entities_dict.get("EMAIL", [""])[0] if "EMAIL" in entities_dict else "",
                "Amount": entities_dict.get("AMOUNT", [""])[0] if "AMOUNT" in entities_dict else "",
                "Sequence": entities_dict.get("SEQ_NUMBER", [""])[0] if "SEQ_NUMBER" in entities_dict else "",
                "Entity_Count": len(annotations["entities"]),
                "Generated_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            all_data.append(row_data)
            country_data.append(row_data)
        
        country_stats[country] = len(country_data)
    
    # Create Excel file with multiple sheets
    print(f"   📝 Creating Excel file: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: Summary information
        summary_data = {
            "Metric": [
                "Total Examples Generated",
                "Countries Included", 
                "Examples with Second Names",
                "Examples with Second Surnames",
                "Examples with Both (Compound)",
                "Generation Date",
                "Dataset Purpose",
                "Supported Languages"
            ],
            "Value": [
                len(all_data),
                ", ".join(countries),
                f"{name_patterns['has_second_name']} ({name_patterns['has_second_name']/len(all_data)*100:.1f}%)",
                f"{name_patterns['has_second_surname']} ({name_patterns['has_second_surname']/len(all_data)*100:.1f}%)",
                f"{name_patterns['compound_names']} ({name_patterns['compound_names']/len(all_data)*100:.1f}%)",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Latin American Customer NER Training",
                "Spanish, Portuguese"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Complete dataset
        all_df = pd.DataFrame(all_data)
        all_df.to_excel(writer, sheet_name='All_Data', index=False)
        
        # Sheet 3: Data by country
        country_summary = []
        for country in countries:
            country_examples = [d for d in all_data if d["Country"] == country]
            has_second_names = sum(1 for d in country_examples if d["Has_Second_Name"] == "Yes")
            has_second_surnames = sum(1 for d in country_examples if d["Has_Second_Surname"] == "Yes")
            
            country_summary.append({
                "Country": country,
                "Total_Examples": len(country_examples),
                "Second_Names_Count": has_second_names,
                "Second_Names_Percentage": f"{has_second_names/len(country_examples)*100:.1f}%" if country_examples else "0%",
                "Second_Surnames_Count": has_second_surnames,
                "Second_Surnames_Percentage": f"{has_second_surnames/len(country_examples)*100:.1f}%" if country_examples else "0%",
                "Avg_Entities_Per_Example": f"{sum(d['Entity_Count'] for d in country_examples)/len(country_examples):.1f}" if country_examples else "0"
            })
        
        country_df = pd.DataFrame(country_summary)
        country_df.to_excel(writer, sheet_name='By_Country', index=False)
        
        # Sheet 4: Name pattern analysis
        name_analysis = []
        unique_first_names = set(d["First_Name_Only"] for d in all_data)
        unique_surnames = set()
        for d in all_data:
            if d["Complete_Surname"]:
                surnames = d["Complete_Surname"].split()
                unique_surnames.update(surnames)
        
        name_analysis.extend([
            {"Analysis_Type": "Unique First Names", "Count": len(unique_first_names), "Examples": ", ".join(list(unique_first_names)[:10]) + "..."},
            {"Analysis_Type": "Unique Surnames", "Count": len(unique_surnames), "Examples": ", ".join(list(unique_surnames)[:10]) + "..."},
            {"Analysis_Type": "Names with Second Name", "Count": name_patterns["has_second_name"], "Examples": "Juan Carlos, María José, Ana Sofía"},
            {"Analysis_Type": "Names with Second Surname", "Count": name_patterns["has_second_surname"], "Examples": "González Rodríguez, Silva Martínez"},
            {"Analysis_Type": "Full Compound Names", "Count": name_patterns["compound_names"], "Examples": "Juan Carlos González Rodríguez"}
        ])
        
        name_df = pd.DataFrame(name_analysis)
        name_df.to_excel(writer, sheet_name='Name_Analysis', index=False)
        
        # Sheet 5: Entity statistics
        entity_list = []
        for entity_type, count in entity_stats.items():
            percentage = count / len(all_data) * 100
            entity_list.append({
                "Entity_Type": entity_type,
                "Total_Count": count,
                "Percentage": f"{percentage:.1f}%",
                "Description": {
                    "CUSTOMER_NAME": "Full customer names with optional second names and surnames",
                    "ID_NUMBER": "Government identification numbers",
                    "ADDRESS": "Street addresses and cities", 
                    "PHONE_NUMBER": "Country-specific phone numbers",
                    "EMAIL": "Email addresses based on names",
                    "AMOUNT": "Monetary amounts with local currencies",
                    "SEQ_NUMBER": "Sequential reference numbers"
                }.get(entity_type, "Unknown entity type")
            })
        
        entity_df = pd.DataFrame(entity_list)
        entity_df.to_excel(writer, sheet_name='Entity_Statistics', index=False)
    
    print(f"✅ Excel file created successfully: {output_file}")
    print(f"📋 File contains {len(all_data)} examples across {len(countries)} countries")
    print(f"📊 Sheets created: Summary, All_Data, By_Country, Name_Analysis, Entity_Statistics")
    print(f"🔍 Review the file to validate:")
    print(f"   - Naming patterns (second names and surnames)")
    print(f"   - Country-specific formats (IDs, phones, currencies)")
    print(f"   - Entity recognition accuracy")
    print(f"   - Data diversity and realism")


# -----------------
# Demo and Testing
# -----------------
def main():
    """
    Demonstration function showing both basic generation and NLP dataset creation.
    
    Provides examples of:
    1. Basic customer data generation
    2. Different complexity modes for NLP training
    3. spaCy dataset creation for NER training
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Latin American Customer Data Generator for NLP")
    parser.add_argument("--mode", choices=["demo", "create-dataset", "excel-export"], default="demo",
                       help="Mode: 'demo' shows examples, 'create-dataset' generates training data, 'excel-export' creates review file")
    parser.add_argument("--train-size", type=int, default=80000, help="Training set size")
    parser.add_argument("--dev-size", type=int, default=20000, help="Development set size")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--excel-examples", type=int, default=100, help="Number of examples for Excel export")
    parser.add_argument("--excel-file", type=str, default="customer_data_review.xlsx", help="Excel output filename")
    
    args = parser.parse_args()
    
    if args.mode == "create-dataset":
        # Create NLP training dataset
        create_training_dataset(
            train_size=args.train_size,
            dev_size=args.dev_size, 
            output_dir=args.output_dir
        )
        return
    elif args.mode == "excel-export":
        # Create Excel file for data review
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        excel_file = output_path / args.excel_file
        
        export_to_excel(
            n_examples=args.excel_examples,
            output_file=str(excel_file)
        )
        return
    
    # Demo mode - show examples
    print("=" * 80)
    print("LATIN AMERICAN CUSTOMER DATA GENERATOR FOR NLP")
    print("=" * 80)
    print()
    
    # Supported countries with their full names
    countries = {
        "CL": "Chile",
        "AR": "Argentina", 
        "BR": "Brazil",
        "UY": "Uruguay",
        "MX": "Mexico"
    }
    
    # Show basic generation examples
    print("🔥 BASIC GENERATION EXAMPLES")
    print("-" * 40)
    for country_code, country_name in list(countries.items())[:2]:  # Show 2 countries
        print(f"📍 {country_name} ({country_code})")
        sentence, annotations = generate_example(country_code)
        
        print(f"Generated: {sentence}")
        print("Entities:", end=" ")
        for start, end, label in annotations["entities"]:
            entity_text = sentence[start:end]
            print(f"[{label}: '{entity_text}']", end=" ")
        print("\n")
    
    # Show different complexity modes
    print("🎯 NLP TRAINING MODES")
    print("-" * 40)
    modes = ["full", "addr_only", "id_only", "contact_only", "financial_only"]
    
    for mode in modes:
        print(f"🔸 Mode: {mode}")
        sentence, annotations = generate_example_with_mode("CL", mode)
        print(f"   Text: {sentence}")
        print(f"   Entities: {[label for _, _, label in annotations['entities']]}")
        print()
    
    print("🔢 Sequential Counter Status:")
    print(f"   Next sequence number: {_sequence_counter + 1}")
    print()
    
    print("📊 NLP DATASET CREATION")
    print("-" * 40)
    print("To create training datasets for spaCy NER:")
    print("   python data_generation.py --mode create-dataset --train-size 10000 --dev-size 2500")
    print()
    print("📁 EXCEL DATA REVIEW")
    print("-" * 40)
    print("To create Excel file for data review and validation:")
    print("   python data_generation.py --mode excel-export --excel-examples 100")
    print("   python data_generation.py --mode excel-export --excel-examples 500 --excel-file detailed_review.xlsx")
    print()
    print("📚 Usage Examples:")
    print("   - Customer service NER training")
    print("   - Financial document processing")
    print("   - Multi-language Latin American NLP")
    print("   - Personal information detection")
    print("   - Data privacy and anonymization")


def quick_test():
    """
    Quick test function to verify all functionality works correctly.
    """
    print("🧪 Running quick functionality test...")
    
    # Test basic generation
    sentence, annotations = generate_example("CL")
    assert len(sentence) > 0, "Basic generation failed"
    assert len(annotations["entities"]) > 0, "No entities generated"
    
    # Test different modes
    modes = ["full", "addr_only", "id_only", "contact_only", "financial_only"]
    for mode in modes:
        sentence, annotations = generate_example_with_mode("BR", mode)
        assert len(sentence) > 0, f"Mode {mode} failed"
        assert len(annotations["entities"]) > 0, f"No entities in mode {mode}"
    
    # Test small dataset creation
    try:
        db, stats = make_docbin(n_total=10, balance=True)
        assert stats["total_examples"] == 10, "DocBin creation failed"
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ DocBin test failed: {e}")
        print("💡 Install spaCy: pip install spacy")


if __name__ == "__main__":
    # Only run main when explicitly called as a script
    main()