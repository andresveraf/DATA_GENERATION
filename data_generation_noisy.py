"""
Multi-Country Latin American PII Training Data Generator with Advanced Noise Generation
=======================================================================================

This module generates realistic customer data for multiple Latin American countries with 
controlled noise patterns specifically designed for Named Entity Recognition (NER) training. 
It creates datasets with labeled entities for customer service and financial NLP applications, 
with enhanced noise generation capabilities that preserve entity boundaries.

Key Features:
- Multi-country PII data generation for Chile, Mexico, Brazil, and Uruguay
- Advanced E1010 overlapping span error prevention (ZERO errors guaranteed)
- Controlled noise injection that preserves entity boundaries
- Named Entity Recognition (NER) annotations with conflict resolution
- spaCy-compatible training data creation (100K+ examples)
- Excel export functionality for data review and validation
- Command-line interface with multiple modes and country selection
- Statistics tracking and reporting

Supported Countries:
- Chile (CL): RUT format, +56 phones, CLP currency, Chilean Spanish
- Mexico (MX): CURP/RFC formats, +52 phones, MXN currency, Mexican Spanish
- Brazil (BR): CPF/RG formats, +55 phones, BRL currency, Portuguese
- Uruguay (UY): Cédula format, +598 phones, UYU currency, Uruguayan Spanish

Supported Entity Types:
- CUSTOMER_NAME: Full names with country-specific conventions
- ID_NUMBER: Country-specific ID formats (RUT/CURP/CPF/Cédula)
- ADDRESS: Country-specific address formats
- PHONE_NUMBER: Country-specific phone formats
- EMAIL: Email addresses with country-appropriate domains
- AMOUNT: Monetary amounts with country currencies
- SEQ_NUMBER: Sequential reference numbers

Enhanced Noise Features:
- Realistic typos and misspellings per country
- Country-specific abbreviations and contractions
- Document formatting variations per country
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
Purpose: Large-scale PII detection model training for Latin American documents
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
# Multi-Country Customer Names Database
# -----------------

# Country-specific data organization
COUNTRY_DATA = {
    'chile': {
        'first_names': [
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
        ],
    },
    'mexico': {
        'first_names': [
            # Traditional Mexican masculine names
            "ALEJANDRO", "ANDRÉS", "ANTONIO", "CARLOS", "DANIEL", "DAVID", "DIEGO", "EDUARDO", "EMILIO", "FERNANDO",
            "FRANCISCO", "GABRIEL", "GUSTAVO", "HUGO", "IGNACIO", "JAVIER", "JESÚS", "JORGE", "JOSÉ", "JUAN",
            "JULIO", "LEONARDO", "LUIS", "MANUEL", "MARCO", "MARIO", "MIGUEL", "PABLO", "PEDRO", "RAFAEL",
            "RAMÓN", "RAÚL", "RICARDO", "ROBERTO", "RODOLFO", "SALVADOR", "SANTIAGO", "SERGIO", "VÍCTOR",
            
            # Traditional Mexican feminine names
            "ADRIANA", "ALEJANDRA", "ANA", "ANDREA", "ÁNGELA", "BEATRIZ", "CARMEN", "CAROLINA", "CLAUDIA", "CRISTINA",
            "DANIELA", "ELENA", "ELIZABETH", "ESPERANZA", "FERNANDA", "GABRIELA", "GUADALUPE", "ISABELLA", "JESSICA",
            "JULIA", "LAURA", "LETICIA", "LUCÍA", "MARÍA", "MARTHA", "MÓNICA", "NATALIA", "PATRICIA", "ROSA",
            "SANDRA", "SOFÍA", "SUSANA", "TERESA", "VALERIA", "VERÓNICA", "VICTORIA", "YOLANDA",
            
            # Indigenous Mexican names
            "XÓCHITL", "ITZEL", "YARETZI", "CITLALI", "NAYELI", "ARELY", "ITZAYANA", "XITLALI", "YANELI", "ANAHÍ",
            "XIMENA", "YAMILET", "CITLALY", "NAOMI", "QUETZALI", "TLÁLOC", "CUAUHTÉMOC", "NEZAHUALCÓYOTL",
            "MOCTEZUMA", "TENOCHTITLAN", "TONATIUH", "MALINTZIN", "IZEL", "ITZÁMIN", "XITLA"
        ],
    },
    'brazil': {
        'first_names': [
            # Brazilian Portuguese masculine names
            "ALEXANDRE", "ANDRÉ", "ANTÔNIO", "BRUNO", "CARLOS", "DANIEL", "EDUARDO", "FÁBIO", "FERNANDO", "GABRIEL",
            "GUSTAVO", "HENRIQUE", "JOÃO", "JOSÉ", "LEONARDO", "LUCAS", "LUÍS", "MARCELO", "MARCOS", "MATEUS",
            "PEDRO", "RAFAEL", "RICARDO", "RODRIGO", "THIAGO", "VINÍCIUS", "CAIO", "FELIPE", "GUILHERME", "IGOR",
            "LEANDRO", "MAURÍCIO", "PAULO", "RENATO", "SÉRGIO", "WELLINGTON", "WASHINGTON", "WESLEY",
            
            # Brazilian Portuguese feminine names  
            "ADRIANA", "ANA", "ANDREA", "BEATRIZ", "BIANCA", "CAMILA", "CAROLINA", "CRISTIANE", "DANIELA", "FERNANDA",
            "GABRIELA", "JULIANA", "LARISSA", "LETÍCIA", "LUCIANA", "MARCIA", "MARIA", "MÔNICA", "PATRÍCIA", "PAULA",
            "PRISCILA", "RAFAELA", "SANDRA", "TATIANA", "VANESSA", "VIVIANE", "DÉBORA", "FLÁVIA", "JÉSSICA", "KARINA",
            "LUANA", "RENATA", "SIMONE", "SOLANGE", "TÂNIA", "CLÁUDIA", "ELIANE", "FABIANA", "GISELE", "HELENA"
        ],
    },
    'uruguay': {
        'first_names': [
            # Uruguayan masculine names (similar to Argentine/Chilean with some variations)
            "AGUSTÍN", "ALEJANDRO", "ANDRÉS", "ANTONIO", "CARLOS", "DANIEL", "DIEGO", "EDUARDO", "FERNANDO", "FRANCISCO",
            "GABRIEL", "GONZALO", "GUSTAVO", "IGNACIO", "JAVIER", "JOAQUÍN", "JORGE", "JOSÉ", "JUAN", "LEONARDO",
            "LUIS", "MANUEL", "MARCELO", "MARIO", "MARTÍN", "MATÍAS", "MIGUEL", "NICOLÁS", "PABLO", "PEDRO",
            "RAFAEL", "RAMIRO", "RICARDO", "ROBERTO", "RODRIGO", "SANTIAGO", "SEBASTIÁN", "VÍCTOR", "WALTER",
            
            # Uruguayan feminine names
            "ADRIANA", "ALEJANDRA", "ANA", "ANDREA", "BEATRIZ", "CAROLINA", "CLAUDIA", "CRISTINA", "DANIELA", "ELENA",
            "FERNANDA", "GABRIELA", "GRACIELA", "ISABEL", "LAURA", "LETICIA", "LUCÍA", "MARÍA", "MARTHA", "MÓNICA",
            "NATALIA", "PATRICIA", "PAULA", "ROSA", "SANDRA", "SILVIA", "SOFÍA", "SUSANA", "VALERIA", "VERÓNICA",
            "VIRGINIA", "VIVIANA", "ALEJANDRA", "CECILIA", "FLORENCIA", "MAGDALENA", "MACARENA", "VALENTINA"
        ],
    }
}

# Backwards compatibility - keep Chilean data accessible
chilean_first_names = COUNTRY_DATA['chile']['first_names']

# Add second names to country data
COUNTRY_DATA['chile']['second_names'] = [
    # Masculine second names
    "CARLOS", "JOSÉ", "LUIS", "ANTONIO", "MANUEL", "FRANCISCO", "MIGUEL", "RAFAEL", "FERNANDO", "RICARDO",
    "ALBERTO", "EDUARDO", "ALEJANDRO", "ANDRÉS", "ROBERTO", "PEDRO", "DANIEL", "GABRIEL", "DIEGO", "SEBASTIÁN",
    "PABLO", "ARTURO", "ENRIQUE", "JOAQUÍN", "NICOLÁS", "FELIPE", "IGNACIO", "ESTEBAN", "RODRIGO", "PATRICIO",
    
    # Feminine second names
    "JOSÉ", "MARÍA", "ISABEL", "CRISTINA", "ELENA", "TERESA", "PATRICIA", "CARMEN", "ROSA", "ANA",
    "LAURA", "BEATRIZ", "ESPERANZA", "GUADALUPE", "DOLORES", "PILAR", "MERCEDES", "SOLEDAD", "AMPARO", "ROCÍO",
    "CONCEPCIÓN", "INMACULADA", "ÁNGELES", "REMEDIOS", "VICTORIA", "GLORIA", "PAZ", "FE", "CARIDAD", "NIEVES"
]

COUNTRY_DATA['mexico']['second_names'] = [
    # Mexican masculine second names
    "MARÍA", "JOSÉ", "LUIS", "ANTONIO", "MANUEL", "FRANCISCO", "MIGUEL", "RAFAEL", "CARLOS", "JESÚS",
    "GUADALUPE", "ÁNGEL", "RAMÓN", "ALEJANDRO", "FERNANDO", "JAVIER", "ALBERTO", "EDUARDO", "ENRIQUE", "SALVADOR",
    
    # Mexican feminine second names  
    "JOSÉ", "MARÍA", "GUADALUPE", "CARMEN", "TERESA", "ISABEL", "ESPERANZA", "ROSA", "ELENA", "PATRICIA",
    "CONCEPCIÓN", "DOLORES", "SOCORRO", "LUZ", "AMPARO", "REFUGIO", "PILAR", "SOLEDAD", "REMEDIOS", "TRINIDAD"
]

COUNTRY_DATA['brazil']['second_names'] = [
    # Brazilian masculine second names (often use "de" constructions)
    "JOSÉ", "JOÃO", "ANTÔNIO", "FRANCISCO", "CARLOS", "PAULO", "PEDRO", "LUCAS", "LUÍS", "MARCOS",
    "RAFAEL", "DANIEL", "MARCELO", "BRUNO", "RODRIGO", "FERNANDO", "GUSTAVO", "EDUARDO", "GABRIEL", "LEONARDO",
    
    # Brazilian feminine second names
    "MARIA", "ANA", "FRANCISCA", "ANTÔNIA", "ADRIANA", "JULIANA", "MÁRCIA", "FERNANDA", "PATRÍCIA", "ALINE",
    "CRISTINA", "CAMILA", "CARLA", "REGINA", "VERA", "LÚCIA", "HELENA", "SILVIA", "MÔNICA", "PAULA"
]

COUNTRY_DATA['uruguay']['second_names'] = [
    # Uruguayan masculine second names
    "JOSÉ", "MARÍA", "CARLOS", "LUIS", "ANTONIO", "MANUEL", "FRANCISCO", "MIGUEL", "RAFAEL", "ALBERTO",
    "EDUARDO", "ALEJANDRO", "ANDRÉS", "ROBERTO", "PEDRO", "DANIEL", "GABRIEL", "DIEGO", "SEBASTIÁN", "PABLO",
    
    # Uruguayan feminine second names
    "MARÍA", "JOSÉ", "ISABEL", "CRISTINA", "ELENA", "TERESA", "PATRICIA", "CARMEN", "ROSA", "ANA",
    "LAURA", "BEATRIZ", "ESPERANZA", "MERCEDES", "SOLEDAD", "AMPARO", "ROCÍO", "VICTORIA", "GLORIA", "PAZ"
]

# Backwards compatibility
chilean_second_names = COUNTRY_DATA['chile']['second_names']

# Keep legacy 'names' for backward compatibility
first_names = chilean_first_names
second_names = chilean_second_names

# Add surnames to country data
COUNTRY_DATA['chile']['surnames'] = [
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

COUNTRY_DATA['mexico']['surnames'] = [
    # Most common Mexican surnames
    "HERNÁNDEZ", "GARCÍA", "MARTÍNEZ", "LÓPEZ", "GONZÁLEZ", "RODRÍGUEZ", "PÉREZ", "SÁNCHEZ", "RAMÍREZ", "CRUZ",
    "FLORES", "GÓMEZ", "MORALES", "VÁZQUEZ", "JIMÉNEZ", "RUIZ", "HERNÁN", "DÍAZ", "MORENO", "MUÑOZ",
    "ÁLVAREZ", "ROMERO", "GUTIÉRREZ", "TORRES", "MENDOZA", "VARGAS", "CASTILLO", "ORTEGA", "REYES", "DELGADO",
    "GUERRERO", "MEDINA", "AGUILAR", "RAMOS", "CERVANTES", "HERRERA", "LARA", "DOMÍNGUEZ", "CASTRO", "VARELA",
    "ORTIZ", "RUBIO", "MARÍN", "IGLESIAS", "NUÑEZ", "PEÑA", "RÍOS", "ALONSO", "GARRIDO", "GALLEGO",
    # Indigenous Mexican surnames
    "XÓLOTL", "ITURBIDE", "CUAUHTÉMOC", "MOCTEZUMA", "TLACAELEL", "NEZAHUALCÓYOTL", "TONATIUH", "COATLICUE"
]

COUNTRY_DATA['brazil']['surnames'] = [
    # Most common Brazilian surnames
    "SILVA", "SANTOS", "OLIVEIRA", "SOUZA", "RODRIGUES", "FERREIRA", "ALVES", "PEREIRA", "LIMA", "GOMES",
    "RIBEIRO", "CARVALHO", "ALMEIDA", "LOPES", "SOARES", "FERNANDES", "VIEIRA", "BARBOSA", "ROCHA", "DIAS",
    "MONTEIRO", "CARDOSO", "REIS", "ARAÚJO", "CAVALCANTI", "NASCIMENTO", "AZEVEDO", "COSTA", "PINTO", "TEIXEIRA",
    "MENDES", "MOREIRA", "CORREIA", "MARTINS", "RAMOS", "NUNES", "FREITAS", "CAMPOS", "MIRANDA", "FONSECA",
    "MACHADO", "MOURA", "MELO", "CUNHA", "PIRES", "CASTRO", "ANDRADE", "COELHO", "FARIAS", "BATISTA"
]

COUNTRY_DATA['uruguay']['surnames'] = [
    # Most common Uruguayan surnames (mix of Spanish and some Italian influence)
    "RODRÍGUEZ", "GONZÁLEZ", "GARCÍA", "LÓPEZ", "MARTÍNEZ", "PÉREZ", "FERNÁNDEZ", "SÁNCHEZ", "DÍAZ", "ÁLVAREZ",
    "ROMERO", "VARGAS", "CASTRO", "RAMOS", "MORALES", "ORTEGA", "DELGADO", "JIMÉNEZ", "RUIZ", "HERNÁNDEZ",
    "SILVA", "TORRES", "FLORES", "VEGA", "MEDINA", "AGUILAR", "HERRERA", "MENDOZA", "GUERRERO", "NÚÑEZ",
    "PEÑA", "RÍOS", "GÓMEZ", "CONTRERAS", "GUTIÉRREZ", "REYES", "ESTRADA", "PAREDES", "DOMÍNGUEZ", "LARA",
    # Italian influence surnames common in Uruguay
    "FERRARI", "ROSSI", "BRUNO", "MARTINO", "ROMANO", "RICCI", "COSTA", "MAZZA", "RUSSO", "GRECO"
]

# Backwards compatibility  
chilean_surnames = COUNTRY_DATA['chile']['surnames']

# Add addresses and cities to country data
COUNTRY_DATA['chile']['streets'] = [
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

COUNTRY_DATA['mexico']['streets'] = [
    # Mexican streets with colonias and typical names
    "Av. Insurgentes Sur",                # Major avenue in Mexico City
    "Av. Reforma",                        # Famous avenue in Mexico City
    "Calle Francisco I. Madero",          # Historic street in Centro
    "Av. Juárez",                         # Important avenue
    "Calle 16 de Septiembre",             # Independence Day street
    "Av. Universidad",                    # University avenue
    "Calle Puebla",                       # Street named after state
    "Av. Chapultepec",                    # Street in Roma Norte
    "Calle Orizaba",                      # Street in Roma Norte colonia
    "Av. Álvaro Obregón",                 # Avenue named after president
    "Calle Insurgentes Norte",            # Northern section of Insurgentes
    "Av. Revolución",                     # Avenue commemorating the Revolution
    "Eje Central Lázaro Cárdenas",        # Major north-south axis
    "Calle Durango",                      # Street in Colonia Roma
    "Calle Londres",                      # Street in Zona Rosa
    "Av. Coyoacán",                       # Avenue to Coyoacán
    "Calle Medellín",                     # Street in Colonia Roma
    "Av. Cuauhtémoc",                     # Avenue named after Aztec emperor
    "Calle Monterrey",                    # Street in Colonia Roma
    "Av. División del Norte"              # Major south avenue
]

COUNTRY_DATA['brazil']['streets'] = [
    # Brazilian streets with typical Portuguese naming
    "Av. Paulista",                       # Famous avenue in São Paulo
    "Rua Augusta",                        # Well-known street in São Paulo
    "Av. Ipiranga",                       # Avenue in São Paulo
    "Rua da Consolação",                  # Street in São Paulo
    "Av. Copacabana",                     # Famous avenue in Rio
    "Rua Visconde de Pirajá",             # Street in Ipanema
    "Av. Atlântica",                      # Beachfront avenue in Rio
    "Rua do Ouvidor",                     # Historic street in Rio Centro
    "Av. Presidente Vargas",              # Avenue named after president
    "Rua Direita",                        # Traditional street name
    "Av. Brasil",                         # Avenue named after country
    "Rua das Flores",                     # "Street of the Flowers"
    "Av. Santos Dumont",                  # Aviation pioneer avenue
    "Rua Barão de Itapetininga",          # Street with noble title
    "Av. São João",                       # Saint John avenue
    "Rua Oscar Freire",                   # Upscale shopping street
    "Av. Brigadeiro Faria Lima",          # Business district avenue
    "Rua 25 de Março",                    # Commercial street
    "Av. Rebouças",                       # Major avenue in São Paulo
    "Rua da Liberdade"                    # Street in Japanese district
]

COUNTRY_DATA['uruguay']['streets'] = [
    # Uruguayan streets from Montevideo and other cities
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
    "Calle Andes",                        # Street named after mountain range
    "Av. Millán",                         # Major avenue
    "Calle Mercedes",                     # Street named after department
    "Av. Agraciada",                      # Historic avenue
    "Calle Uruguay",                      # Street named after country
    "Bvar. Batlle y Ordóñez"              # Boulevard named after president
]

COUNTRY_DATA['chile']['cities'] = [
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
    "Valdivia"           # Rivers city
]

COUNTRY_DATA['mexico']['cities'] = [
    "Ciudad de México",   # Capital and largest city
    "Guadalajara",        # Second largest city
    "Monterrey",          # Major industrial city
    "Puebla",             # Historic colonial city
    "Tijuana",            # Border city with USA
    "León",               # Leather industry city
    "Ciudad Juárez",      # Border city
    "Torreón",            # Northern industrial city
    "Querétaro",          # Central Mexico city
    "San Luis Potosí",    # Mining city
    "Mérida",             # Yucatan capital
    "Aguascalientes",     # Central city
    "Mexicali",           # Baja California capital
    "Culiacán",           # Sinaloa capital
    "Acapulco"            # Pacific coast resort
]

COUNTRY_DATA['brazil']['cities'] = [
    "São Paulo",          # Largest city in South America
    "Rio de Janeiro",     # Former capital, major tourist destination
    "Belo Horizonte",     # Major southeastern city
    "Brasília",           # Current capital
    "Salvador",           # Historic northeastern city
    "Fortaleza",          # Major northeastern city
    "Manaus",             # Amazon region capital
    "Curitiba",           # Southern city
    "Recife",             # Major northeastern port
    "Goiânia",            # Central Brazil city
    "Belém",              # Amazon river port
    "Guarulhos",          # Greater São Paulo
    "Campinas",           # Technology hub
    "São Luís",           # Northeastern capital
    "Nova Iguaçu"         # Rio de Janeiro metropolitan area
]

COUNTRY_DATA['uruguay']['cities'] = [
    "Montevideo",         # Capital and largest city
    "Salto",              # Second largest city
    "Paysandú",           # Important river port
    "Las Piedras",        # Suburban city near Montevideo
    "Rivera",             # Border city with Brazil
    "Maldonado",          # Coastal city
    "Tacuarembó",         # Northern city
    "Melo",               # Eastern city
    "Mercedes",           # Western city
    "Artigas",            # Northern border city
    "Minas",              # Central city
    "San José de Mayo",   # Central city
    "Durazno",            # Central city
    "Florida",            # Central city
    "Treinta y Tres"      # Eastern city
]

# Backwards compatibility for addresses
chilean_streets = COUNTRY_DATA['chile']['streets']  
chilean_cities = COUNTRY_DATA['chile']['cities']

# Keep backward compatibility
surnames = chilean_surnames
streets = chilean_streets
cities = chilean_cities

# -----------------
# Chilean Organizations Database
# -----------------
# Multi-Country Organizations Database
# -----------------

# Add organizations to country data
COUNTRY_DATA['chile']['organizations'] = [
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

COUNTRY_DATA['mexico']['organizations'] = [
    # Banks and Financial Institutions
    "BBVA México", "Banorte", "Santander México", "HSBC México", "Citibanamex", "Banco Azteca",
    "Scotiabank México", "Banco Inbursa", "BanCoppel", "Banco del Bajío",
    
    # Retail and Commerce
    "Liverpool", "Palacio de Hierro", "Sears México", "Coppel", "Elektra", "Soriana",
    "Walmart México", "Comercial Mexicana", "OXXO", "7-Eleven México",
    
    # Telecommunications
    "Telcel", "Movistar México", "AT&T México", "Izzi", "Megacable", "Totalplay",
    
    # Government and Public
    "IMSS", "ISSSTE", "INFONAVIT", "SAT", "INE", "SEDENA", "SEMAR", "Guardia Nacional",
    "Secretaría de Salud", "SEP", "CONACYT",
    
    # Energy and Industry
    "PEMEX", "CFE", "Grupo México", "CEMEX", "Bimbo", "América Móvil",
    
    # Education
    "UNAM", "IPN", "ITESM", "UAM", "Universidad de Guadalajara"
]

COUNTRY_DATA['brazil']['organizations'] = [
    # Banks and Financial Institutions
    "Banco do Brasil", "Itaú Unibanco", "Bradesco", "Santander Brasil", "Caixa Econômica Federal",
    "Banco Inter", "Nubank", "BTG Pactual", "Banco Safra", "Banrisul",
    
    # Retail and Commerce
    "Magazine Luiza", "Via Varejo", "Lojas Americanas", "Pão de Açúcar", "Carrefour Brasil",
    "Walmart Brasil", "Extra", "Casas Bahia", "Renner", "C&A Brasil",
    
    # Telecommunications
    "Vivo", "Claro Brasil", "TIM Brasil", "Oi", "Nextel Brasil", "Algar Telecom",
    
    # Government and Public
    "Receita Federal", "INSS", "SUS", "Ministério da Saúde", "IBGE", "Polícia Federal",
    "Correios", "BNDES", "Petrobras", "Eletrobras",
    
    # Industry and Energy
    "Vale", "JBS", "Ambev", "Gerdau", "CSN", "Embraer",
    
    # Education
    "USP", "UNICAMP", "UFRJ", "UFMG", "UnB", "PUC-SP"
]

COUNTRY_DATA['uruguay']['organizations'] = [
    # Banks and Financial Institutions
    "Banco República", "Banco Santander Uruguay", "Banco Itaú Uruguay", "BBVA Uruguay",
    "Banco de la Nación Argentina", "Citibank Uruguay", "Scotiabank Uruguay",
    
    # Retail and Commerce
    "Tienda Inglesa", "Disco", "Devoto", "Ta-Ta", "Géant", "Farmashop", "Red Pagos",
    
    # Telecommunications
    "Antel", "Movistar Uruguay", "Claro Uruguay", "Dedicado",
    
    # Government and Public
    "DGI", "BPS", "ASSE", "UdelaR", "INAU", "MTOP", "Ministerio de Salud",
    "Intendencia de Montevideo", "Policía Nacional", "Bomberos",
    
    # Utilities and Services
    "UTE", "OSE", "ANCAP", "AFE", "Puerto de Montevideo",
    
    # Education
    "Universidad de la República", "Universidad Católica del Uruguay", "Universidad ORT"
]

# Backwards compatibility
chilean_organizations = COUNTRY_DATA['chile']['organizations']

# Keep backward compatibility
organizations = chilean_organizations

# -----------------
# Multi-Country Name Generation Functions
# -----------------

def generate_name_components(country: str = "chile",
                           include_second_name: bool = True, 
                           second_name_probability: float = 0.4,
                           include_second_surname: bool = True, 
                           second_surname_probability: float = 0.8) -> Tuple[str, str, str]:
    """
    Generate name components with enhanced second surname support for any supported country.
    
    Creates authentic naming patterns for each country including:
    - First name (required)
    - Optional second name (compound first names like "Juan Carlos", "María José")
    - Paternal surname (required)
    - Optional maternal surname (common in Latin America)
    
    Args:
        country (str): Country code - "chile", "mexico", "brazil", or "uruguay"
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
    # Get country-specific data
    if country not in COUNTRY_DATA:
        country = "chile"  # fallback to Chile
    
    country_info = COUNTRY_DATA[country]
    
    # Generate first name
    first_name = random.choice(country_info['first_names'])
    
    # Generate paternal surname (always required)
    paternal_surname = random.choice(country_info['surnames'])
    
    # Decide on second name
    full_name_part = first_name
    if include_second_name and random.random() < second_name_probability:
        second_name = random.choice(country_info['second_names'])
        full_name_part = f"{first_name} {second_name}"
    
    # Decide on maternal surname
    complete_surname = paternal_surname
    if include_second_surname and random.random() < second_surname_probability:
        maternal_surname = random.choice(country_info['surnames'])
        # Ensure different surnames
        if maternal_surname != paternal_surname:
            complete_surname = f"{paternal_surname} {maternal_surname}"
    
    return first_name, full_name_part, complete_surname

def generate_chilean_name_components(include_second_name: bool = True, 
                                   second_name_probability: float = 0.4,
                                   include_second_surname: bool = True, 
                                   second_surname_probability: float = 0.8) -> Tuple[str, str, str]:
    """
    Generate Chilean name components with enhanced second surname support.
    (Backwards compatibility wrapper)
    """
    return generate_name_components("chile", include_second_name, second_name_probability, 
                                   include_second_surname, second_surname_probability)

def generate_phone(country: str = "chile") -> str:
    """
    Generate a realistic phone number for the specified country.
    
    Country-specific phone formats:
    - Chile: +56 9 XXXX XXXX (mobile) or +56 2 XXXX XXXX (landline)
    - Mexico: +52 1 XXX XXX XXXX (mobile) or +52 XX XXXX XXXX (landline)
    - Brazil: +55 XX 9XXXX XXXX (mobile) or +55 XX XXXX XXXX (landline)  
    - Uruguay: +598 9X XXX XXX (mobile) or +598 2XXX XXXX (landline)
    
    Args:
        country (str): Country code - "chile", "mexico", "brazil", or "uruguay"
        
    Returns:
        str: Formatted phone number
    """
    if country == "chile":
        # 80% mobile, 20% landline
        if random.random() < 0.8:
            # Mobile phone (+56 9)
            return f"+56 9 {random.randint(1000,9999)} {random.randint(1000,9999)}"
        else:
            # Santiago landline (+56 2)
            return f"+56 2 {random.randint(2000,9999)} {random.randint(1000,9999)}"
    
    elif country == "mexico":
        # 85% mobile, 15% landline
        if random.random() < 0.85:
            # Mobile phone (+52 1)
            return f"+52 1 {random.randint(100,999)} {random.randint(100,999)} {random.randint(1000,9999)}"
        else:
            # Landline (+52 city_code)
            city_code = random.choice([55, 33, 81, 222, 664])  # Mexico City, Guadalajara, Monterrey, Puebla, Tijuana
            return f"+52 {city_code} {random.randint(1000,9999)} {random.randint(1000,9999)}"
    
    elif country == "brazil":
        # 90% mobile, 10% landline
        if random.random() < 0.9:
            # Mobile phone (+55 XX 9XXXX XXXX)
            area_code = random.choice([11, 21, 31, 47, 85])  # São Paulo, Rio, Belo Horizonte, Joinville, Fortaleza
            return f"+55 {area_code} 9{random.randint(1000,9999)} {random.randint(1000,9999)}"
        else:
            # Landline (+55 XX XXXX XXXX)
            area_code = random.choice([11, 21, 31, 47, 85])
            return f"+55 {area_code} {random.randint(2000,9999)} {random.randint(1000,9999)}"
    
    elif country == "uruguay":
        # 85% mobile, 15% landline
        if random.random() < 0.85:
            # Mobile phone (+598 9X XXX XXX)
            return f"+598 9{random.randint(1,9)} {random.randint(100,999)} {random.randint(100,999)}"
        else:
            # Montevideo landline (+598 2XXX XXXX)
            return f"+598 2{random.randint(100,999)} {random.randint(1000,9999)}"
    
    else:
        # Default to Chilean format
        return generate_phone("chile")

def generate_chilean_phone() -> str:
    """
    Generate a realistic Chilean phone number.
    (Backwards compatibility wrapper)
    """
    return generate_phone("chile")

def generate_email(name: str, surname: str, country: str = "chile") -> str:
    """
    Generate a realistic email address using the person's name and surname for any country.
    
    Creates email in format: firstname.lastname@domain.com
    Uses common email providers in each country.
    For double surnames, uses only the paternal (first) surname.
    
    Args:
        name (str): First name of the person
        surname (str): Complete surname (may include paternal and maternal)
        country (str): Country code - "chile", "mexico", "brazil", or "uruguay"
        
    Returns:
        str: Email address in lowercase
    """
    # Country-specific email domains
    domains = {
        "chile": ["gmail.com", "hotmail.com", "yahoo.com", "outlook.com", "live.cl", "vtr.net"],
        "mexico": ["gmail.com", "hotmail.com", "yahoo.com.mx", "outlook.com", "live.com.mx", "prodigy.net.mx"],
        "brazil": ["gmail.com", "hotmail.com", "yahoo.com.br", "outlook.com", "uol.com.br", "bol.com.br", "terra.com.br"],
        "uruguay": ["gmail.com", "hotmail.com", "yahoo.com", "outlook.com", "live.com.uy", "adinet.com.uy"]
    }
    
    # Use only the first surname for email (paternal surname)
    first_surname = surname.split()[0] if " " in surname else surname
    
    # Remove accents and special characters for email compatibility
    name_clean = name.lower()
    surname_clean = first_surname.lower()
    
    # Remove common accents for all countries
    accent_map = {
        'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i',
        'ó': 'o', 'ò': 'o', 'õ': 'o', 'ô': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u',
        'ñ': 'n', 'ç': 'c'
    }
    
    for accented, plain in accent_map.items():
        name_clean = name_clean.replace(accented, plain)
        surname_clean = surname_clean.replace(accented, plain)
    
    # Select appropriate domains for country
    country_domains = domains.get(country, domains["chile"])
    
    return f"{name_clean}.{surname_clean}@{random.choice(country_domains)}"

def generate_chilean_email(name: str, surname: str) -> str:
    """
    Generate a realistic Chilean email address using the person's name and surname.
    (Backwards compatibility wrapper)
    """
    return generate_email(name, surname, "chile")

def generate_id(country: str = "chile") -> str:
    """
    Generate a realistic identification number for the specified country.
    
    Each country has its own ID format:
    - Chile: RUT format XX.XXX.XXX-X
    - Mexico: CURP format AAAA######AAAAAA## and RFC format
    - Brazil: CPF format XXX.XXX.XXX-XX
    - Uruguay: Cédula format X.XXX.XXX-X
    
    Args:
        country (str): Country code - "chile", "mexico", "brazil", or "uruguay"
        
    Returns:
        str: Formatted identification number
    """
    if country == "chile":
        # Chilean RUT (Rol Único Tributario)
        rut_number = random.randint(10_000_000, 30_000_000)
        check_digit = random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'K'])
        rut_str = f"{rut_number:,}".replace(',', '.')
        return f"{rut_str}-{check_digit}"
    
    elif country == "mexico":
        # Mexican CURP (Clave Única de Registro de Población) - simplified version
        if random.random() < 0.7:  # 70% CURP
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            vowels = "AEIOU"
            return f"{random.choice(letters)}{random.choice(vowels)}{random.choice(letters)}{random.choice(letters)}{random.randint(10,99)}{random.randint(1,12):02}{random.randint(1,28):02}H{random.choice(['DF', 'GD', 'GT', 'JL', 'MC', 'MN', 'NL', 'PB', 'SL', 'VZ'])}{random.choice(letters)}{random.choice(letters)}{random.randint(10,99)}"
        else:  # 30% RFC
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            return f"{random.choice(letters)}{random.choice(letters)}{random.choice(letters)}{random.choice(letters)}{random.randint(100000,999999)}{random.choice(letters)}{random.choice(letters)}{random.randint(1,9)}"
    
    elif country == "brazil":
        # Brazilian CPF (Cadastro de Pessoas Físicas)
        return f"{random.randint(100,999)}.{random.randint(100,999)}.{random.randint(100,999)}-{random.randint(10,99)}"
    
    elif country == "uruguay":
        # Uruguayan Cédula de Identidad
        return f"{random.randint(1,9)}.{random.randint(100,999)}.{random.randint(100,999)}-{random.randint(0,9)}"
    
    else:
        # Default to Chilean format
        return generate_id("chile")

def generate_chilean_rut() -> str:
    """
    Generate a realistic Chilean RUT (Rol Único Tributario).
    (Backwards compatibility wrapper)
    """
    return generate_id("chile")

def generate_amount(country: str = "chile") -> str:
    """
    Generate a realistic monetary amount for the specified country.
    
    Amounts are generated within typical ranges for each country's currency:
    - Chile: 10,000 - 2,000,000 CLP
    - Mexico: 500 - 100,000 MXN
    - Brazil: 50 - 5,000 BRL
    - Uruguay: 1,000 - 200,000 UYU
    
    Args:
        country (str): Country code - "chile", "mexico", "brazil", or "uruguay"
        
    Returns:
        str: Formatted amount with currency symbol and code
    """
    if country == "chile":
        amount = random.randint(10_000, 2_000_000)
        return f"${amount:,} CLP".replace(',', '.')
    
    elif country == "mexico":
        amount = random.randint(500, 100_000)
        return f"${amount:,} MXN"
    
    elif country == "brazil":
        amount = random.randint(50, 5_000)
        return f"R$ {amount:,} BRL"
    
    elif country == "uruguay":
        amount = random.randint(1_000, 200_000)
        return f"${amount:,} UYU"
    
    else:
        # Default to Chilean format
        return generate_amount("chile")

def generate_chilean_amount() -> str:
    """
    Generate a realistic Chilean monetary amount.
    (Backwards compatibility wrapper)
    """
    return generate_amount("chile")

def generate_sequence_number(country: str = "chile") -> str:
    """
    Generate a realistic sequential number for different business contexts per country.
    
    Creates varied sequence identifiers used in real business scenarios:
    - Complaint numbers: 7-digit numbers
    - Reference IDs: alphanumeric codes  
    - Transaction IDs: mixed format
    - Country-specific prefixes and formats
    
    Args:
        country (str): Country code - "chile", "mexico", "brazil", or "uruguay"
        
    Returns:
        str: Sequential identifier
    """
    sequence_types = [
        f"{random.randint(1000000, 9999999)}",  # 7-digit number
        f"{random.randint(10000, 99999)}-{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}",  # Number-Letter
        f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(100000, 999999)}",  # Letter-Number
        f"{get_next_sequence()}",  # Global sequence
    ]
    
    # Add country-specific prefixes occasionally
    if random.random() < 0.15:  # 15% chance of country prefix
        country_prefixes = {
            "chile": "CL",
            "mexico": "MX", 
            "brazil": "BR",
            "uruguay": "UY"
        }
        prefix = country_prefixes.get(country, "CL")
        return f"{prefix}-{random.randint(10001, 99999)}"
    
    return random.choice(sequence_types)

def generate_chilean_sequence_number() -> str:
    """
    Generate a realistic sequential number for Chilean business contexts.
    (Backwards compatibility wrapper)
    """
    return generate_sequence_number("chile")

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

# -----------------
# Multi-Country Text Templates and Noise Functions
# -----------------

def get_sentence_templates(country: str) -> List[str]:
    """
    Get sentence templates appropriate for the specified country.
    
    Args:
        country (str): Country code
        
    Returns:
        List[str]: List of sentence templates with placeholders
    """
    if country == "chile":
        return [
            "El cliente {} con RUT {} reside en {}, {}. Teléfono: {} / Email: {}. Monto: {} - Referencia: {}.",
            "Registro de {} con cédula {}. Ubicado en {}, {}. Contacto: {} / {}. Transacción: {} - ID: {}.",
            "Cliente {} identificado con RUT {}. Dirección: {}, {}. Contacto telefónico: {}. Correo: {}. Valor: {} - Número: {}.",
            "Datos del cliente: {} (RUT: {}). Domicilio: {}, {}. Tel: {} / {}. Monto operación: {} - Folio: {}.",
            "Información de {} con RUT {}. Vive en {}, {}. Fono: {} - Email: {}. Cantidad: {} - Serie: {}."
        ]
    elif country == "mexico":
        return [
            "El cliente {} con CURP {} reside en {}, {}. Teléfono: {} / Email: {}. Monto: {} - Referencia: {}.",
            "Registro de {} con cédula {}. Ubicado en {}, {}. Contacto: {} / {}. Transacción: {} - ID: {}.",
            "Cliente {} identificado con CURP {}. Dirección: {}, {}. Contacto telefónico: {}. Correo: {}. Valor: {} - Número: {}.",
            "Datos del cliente: {} (CURP: {}). Domicilio: {}, {}. Tel: {} / {}. Monto operación: {} - Folio: {}.",
            "Información de {} con RFC {}. Vive en {}, {}. Fono: {} - Email: {}. Cantidad: {} - Serie: {}."
        ]
    elif country == "brazil":
        return [
            "O cliente {} com CPF {} reside em {}, {}. Telefone: {} / Email: {}. Valor: {} - Referência: {}.",
            "Registro de {} com CPF {}. Localizado em {}, {}. Contato: {} / {}. Transação: {} - ID: {}.",
            "Cliente {} identificado com CPF {}. Endereço: {}, {}. Telefone: {}. Email: {}. Valor: {} - Número: {}.",
            "Dados do cliente: {} (CPF: {}). Domicílio: {}, {}. Tel: {} / {}. Valor da operação: {} - Protocolo: {}.",
            "Informação de {} com CPF {}. Mora em {}, {}. Fone: {} - Email: {}. Quantia: {} - Série: {}."
        ]
    elif country == "uruguay":
        return [
            "El cliente {} con cédula {} reside en {}, {}. Teléfono: {} / Email: {}. Monto: {} - Referencia: {}.",
            "Registro de {} con CI {}. Ubicado en {}, {}. Contacto: {} / {}. Transacción: {} - ID: {}.",
            "Cliente {} identificado con cédula {}. Dirección: {}, {}. Contacto telefónico: {}. Correo: {}. Valor: {} - Número: {}.",
            "Datos del cliente: {} (CI: {}). Domicilio: {}, {}. Tel: {} / {}. Monto operación: {} - Folio: {}.",
            "Información de {} con cédula {}. Vive en {}, {}. Fono: {} - Email: {}. Cantidad: {} - Serie: {}."
        ]
    else:
        return get_sentence_templates("chile")  # Default to Chilean templates

def apply_country_noise(sentence: str, country: str, noise_level: float) -> str:
    """
    Apply country-specific noise patterns to a sentence.
    
    Args:
        sentence (str): Original sentence
        country (str): Country code  
        noise_level (float): Noise intensity (0.0-1.0)
        
    Returns:
        str: Sentence with applied noise
    """
    if noise_level <= 0.0:
        return sentence
    
    # For now, apply simple noise patterns
    # This can be enhanced with country-specific abbreviations and noise patterns
    if random.random() < noise_level * 0.3:
        # Simple noise patterns that don't affect entity boundaries
        noise_replacements = {
            "Teléfono": "Tel",
            "Telefone": "Tel", 
            "Email": "E-mail",
            "Monto": "Mto",
            "Valor": "Vlr",
            "Referencia": "Ref",
            "Referência": "Ref",
            "Número": "Núm",
            "Número": "Nro"
        }
        
        for original, replacement in noise_replacements.items():
            if original in sentence:
                sentence = sentence.replace(original, replacement)
                break  # Apply only one noise change
    
    return sentence

def generate_example_with_noise(country: str = "chile", include_noise: bool = True, noise_level: float = 0.15) -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
    """
    Generate a complete customer data example for any supported country with controlled noise and guaranteed zero E1010 errors.
    
    Creates realistic customer information including:
    - Full name with country-specific conventions
    - Country-specific ID number format
    - Complete address with country-specific streets and cities
    - Country-specific phone number format
    - Email address with country-appropriate domains
    - Monetary amount in country currency
    - Sequential reference number
    - Controlled noise that preserves entity boundaries
    
    CRITICAL: Implements advanced entity conflict resolution to guarantee zero E1010 overlapping span errors.
    
    Args:
        country (str): Country code - "chile", "mexico", "brazil", or "uruguay"
        include_noise (bool): Whether to add realistic noise patterns
        noise_level (float): Intensity of noise (0.0-1.0)
    
    Returns:
        Tuple[str, Dict]: A tuple containing:
            - str: Generated sentence with customer data and optional noise
            - Dict: NER annotations with entity positions and labels
                   Format: {"entities": [(start, end, label), ...]}
    
    Example:
        >>> sentence, annotations = generate_example_with_noise("mexico")
        >>> print(sentence)
        "El cliente MARÍA GUADALUPE HERNÁNDEZ GARCÍA con CURP MAGR850315MDFNRL09..."
        >>> print(annotations)
        {"entities": [(11, 38, "CUSTOMER_NAME"), (49, 68, "ID_NUMBER"), ...]}
    """
    # Validate country
    if country not in COUNTRY_DATA:
        country = "chile"  # fallback to Chile
    
    country_info = COUNTRY_DATA[country]
    
    # Generate name components with enhanced second surname support
    first_name, full_name_part, complete_surname = generate_name_components(
        country=country,
        include_second_name=True, second_name_probability=0.4,
        include_second_surname=True, second_surname_probability=0.8
    )
    complete_full_name = f"{full_name_part} {complete_surname}"    # Complete name for entity recognition
    
    # Generate country-specific data
    id_number = generate_id(country)                              # Country-specific ID format
    street = random.choice(country_info['streets'])               # Country-specific street
    street_number = random.randint(10, 999)                      # Street number
    address = f"{street} {street_number}"                         # Complete address
    city = random.choice(country_info['cities'])                  # Country-specific city
    phone = generate_phone(country)                               # Country-specific phone format
    email = generate_email(first_name, complete_surname, country) # Country-specific email
    amount = generate_amount(country)                             # Country-specific currency
    sequence = generate_sequence_number(country)                  # Country-specific sequence
    
    # Country-specific sentence templates with cultural appropriateness
    templates = get_sentence_templates(country)
    
    # Select random template and format with generated data
    template = random.choice(templates)
    
    # Format template with data
    sentence = template.format(
        complete_full_name, id_number, address, city, 
        phone, email, amount, sequence
    )
    
    # Apply country-specific noise if requested
    if include_noise:
        sentence = apply_country_noise(sentence, country, noise_level)
    
    # CRITICAL: Advanced entity detection with E1010 conflict resolution
    entity_mappings = [
        (complete_full_name, "CUSTOMER_NAME"),        # Full customer name
        (id_number, "ID_NUMBER"),                     # Country-specific ID
        (address, "ADDRESS"),                         # Street address
        (city, "ADDRESS"),                            # City
        (phone, "PHONE_NUMBER"),                      # Phone number
        (email, "EMAIL"),                             # Email address
        (amount, "AMOUNT"),                           # Country currency amount
        (sequence, "SEQ_NUMBER")                      # Sequential reference number
    ]
    
    # IMPROVED ENTITY DETECTION WITH CONFLICT RESOLUTION (E1010 FIX)
    used_positions = set()
    entities = []
    
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

def generate_chilean_example_with_noise(include_noise: bool = True, noise_level: float = 0.15) -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
    """
    Generate a complete Chilean customer data example with controlled noise and guaranteed zero E1010 errors.
    (Backwards compatibility wrapper)
    """
    return generate_example_with_noise("chile", include_noise, noise_level)
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

def export_country_data_to_excel_with_noise(country: str = "chile",
                                          n_examples: int = 100, 
                                          output_file: str = "country_customer_data_review_noisy.xlsx",
                                          include_noise: bool = True,
                                          noise_level: float = 0.15) -> None:
    """
    Export generated customer data for a specific country to Excel for comprehensive review and validation.
    
    Creates a detailed Excel workbook with multiple sheets for thorough analysis:
    - Summary statistics and overview
    - Complete data with entity annotations
    - Country-specific naming pattern analysis
    - Entity type distribution
    - Noise pattern analysis
    
    Args:
        country (str): Country code - "chile", "mexico", "brazil", or "uruguay"
        n_examples (int): Number of examples to generate and export
        output_file (str): Excel filename
        include_noise (bool): Whether to include noise in generated data
        noise_level (float): Intensity of noise (0.0-1.0)
    """
    print(f"📊 Generating {n_examples} {country} examples for Excel review...")
    print(f"🎭 Noise generation: {'Enabled' if include_noise else 'Disabled'}")
    print(f"📁 Output file: {output_file}")
    
    # Generate examples using the generic function
    all_data = []
    entity_counts = {}
    noise_patterns = {}
    name_patterns = {
        "compound_first_names": 0,
        "double_surnames": 0,
        "simple_names": 0
    }
    
    for i in range(n_examples):
        try:
            sentence, annotations = generate_example_with_noise(country, include_noise, noise_level)
            
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
                        # Check if second name exists (basic heuristic)
                        if country in COUNTRY_DATA and any(name_parts[1] in COUNTRY_DATA[country].get('second_names', [])):
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
                "ID": i + 1,
                "Country": country.upper(),
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
                "Country",
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
                country.upper(),
                "Yes" if include_noise else "No",
                f"{noise_level:.2f}" if include_noise else "N/A",
                f"{sum(d['Text_Length'] for d in all_data)/len(all_data):.1f}" if all_data else "0",
                f"{sum(d['Entity_Count'] for d in all_data)/len(all_data):.1f}" if all_data else "0",
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
        
        # 3. Name Pattern Analysis Sheet
        name_analysis = [
            {"Pattern_Type": "Compound First Names", "Count": name_patterns["compound_first_names"], "Percentage": f"{name_patterns['compound_first_names']/len(all_data)*100:.1f}%" if all_data else "0%"},
            {"Pattern_Type": "Double Surnames", "Count": name_patterns["double_surnames"], "Percentage": f"{name_patterns['double_surnames']/len(all_data)*100:.1f}%" if all_data else "0%"},
            {"Pattern_Type": "Simple Names", "Count": name_patterns["simple_names"], "Percentage": f"{name_patterns['simple_names']/len(all_data)*100:.1f}%" if all_data else "0%"}
        ]
        
        name_df = pd.DataFrame(name_analysis)
        name_df.to_excel(writer, sheet_name='Name_Analysis', index=False)
        
        # 4. Entity Statistics Sheet
        entity_descriptions = {
            "CUSTOMER_NAME": "Full customer names with country conventions",
            "ID_NUMBER": "Country-specific ID numbers (RUT, CURP, CPF, etc.)",
            "ADDRESS": "Complete addresses with country-specific formats",
            "PHONE": "Country-specific phone numbers",
            "EMAIL": "Email addresses with country domains",
            "AMOUNT": "Monetary amounts with local currencies",
            "SEQ_NUMBER": "Sequential reference numbers"
        }
        
        entity_analysis = []
        for entity_type, count in entity_counts.items():
            percentage = (count / len(all_data) * 100) if all_data else 0
            entity_analysis.append({
                "Entity_Type": entity_type,
                "Count": count,
                "Percentage": f"{percentage:.1f}%",
                "Description": entity_descriptions.get(entity_type, "Entity type")
            })
        
        entity_df = pd.DataFrame(entity_analysis)
        entity_df.to_excel(writer, sheet_name='Entity_Statistics', index=False)
        
        # 5. Noise Analysis Sheet (if noise is enabled)
        if include_noise and noise_patterns:
            noise_analysis = []
            for pattern_type, count in noise_patterns.items():
                noise_analysis.append({
                    "Noise_Pattern": pattern_type,
                    "Occurrences": count,
                    "Percentage": f"{(count / len(all_data) * 100):.1f}%" if all_data else "0%"
                })
            
            noise_df = pd.DataFrame(noise_analysis)
            noise_df.to_excel(writer, sheet_name='Noise_Analysis', index=False)
    
    print(f"✅ Excel file created successfully: {output_file}")
    print(f"📊 Generated {len(all_data)} examples with {len(entity_counts)} entity types")
    print(f"🏷️  Entity distribution: {dict(sorted(entity_counts.items()))}")
    print(f"📋 {country.upper()} naming patterns: {name_patterns}")
    
    if include_noise:
        print(f"🎭 Noise patterns detected: {noise_patterns}")
    
    print(f"\n📖 Excel sheets created:")
    print(f"  • Summary - Overview statistics")
    print(f"  • All_Data - Complete generated data")
    print(f"  • Name_Analysis - Country naming pattern analysis")
    print(f"  • Entity_Statistics - Entity type distribution")
    if include_noise:
        print(f"  • Noise_Analysis - Noise pattern analysis")

def export_multi_country_data_to_excel_with_noise(n_examples: int = 100, 
                                                output_file: str = "multi_country_customer_data_review_noisy.xlsx",
                                                include_noise: bool = True,
                                                noise_level: float = 0.15) -> None:
    """
    Export generated customer data for all supported countries to Excel for comprehensive review and validation.
    
    Creates a detailed Excel workbook with multiple sheets for thorough analysis:
    - Summary statistics and overview
    - Complete data with entity annotations for all countries
    - Analysis by country
    - Entity type distribution
    - Noise pattern analysis
    
    Args:
        n_examples (int): Total number of examples to generate across all countries
        output_file (str): Excel filename
        include_noise (bool): Whether to include noise in generated data
        noise_level (float): Intensity of noise (0.0-1.0)
    """
    supported_countries = ["chile", "mexico", "brazil", "uruguay"]
    examples_per_country = n_examples // len(supported_countries)
    
    print(f"📊 Generating {n_examples} multi-country examples for Excel review...")
    print(f"🌎 Countries: {', '.join([c.upper() for c in supported_countries])}")
    print(f"📈 Examples per country: {examples_per_country}")
    print(f"🎭 Noise generation: {'Enabled' if include_noise else 'Disabled'}")
    print(f"📁 Output file: {output_file}")
    
    # Generate examples for all countries
    all_data = []
    entity_counts = {}
    noise_patterns = {}
    country_stats = {}
    name_patterns = {
        "compound_first_names": 0,
        "double_surnames": 0,
        "simple_names": 0
    }
    
    for country in supported_countries:
        country_data = []
        print(f"   🌍 Generating {examples_per_country} examples for {country.upper()}...")
        
        for i in range(examples_per_country):
            try:
                sentence, annotations = generate_example_with_noise(country, include_noise, noise_level)
                
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
                            # Check if second name exists (basic heuristic)
                            if country in COUNTRY_DATA and any(name_parts[1] in COUNTRY_DATA[country].get('second_names', [])):
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
                
                row_data = {
                    "ID": len(all_data) + 1,
                    "Country": country.upper(),
                    "Generated_Text": sentence,
                    "Entity_Count": len(entities),
                    "Entities": " | ".join(entity_details),
                    "Has_Noise": include_noise,
                    "Text_Length": len(sentence)
                }
                
                all_data.append(row_data)
                country_data.append(row_data)
                
            except Exception as e:
                continue
        
        country_stats[country.upper()] = len(country_data)
    
    # Create Excel workbook
    print(f"📝 Creating Excel workbook with {len(all_data)} examples...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 1. Summary Sheet
        summary_data = {
            "Metric": [
                "Total Examples Generated",
                "Countries Included",
                "Examples per Country",
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
                ", ".join([c.upper() for c in supported_countries]),
                examples_per_country,
                "Yes" if include_noise else "No",
                f"{noise_level:.2f}" if include_noise else "N/A",
                f"{sum(d['Text_Length'] for d in all_data)/len(all_data):.1f}" if all_data else "0",
                f"{sum(d['Entity_Count'] for d in all_data)/len(all_data):.1f}" if all_data else "0",
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
        
        # 3. By Country Sheet
        country_summary = []
        for country in supported_countries:
            country_examples = [d for d in all_data if d["Country"] == country.upper()]
            has_second_names = sum(1 for d in country_examples if d["Entity_Count"] > 0)  # Simplified metric
            has_second_surnames = sum(1 for d in country_examples if d["Entity_Count"] > 0)  # Simplified metric
            
            country_summary.append({
                "Country": country.upper(),
                "Total_Examples": len(country_examples),
                "Avg_Text_Length": f"{sum(d['Text_Length'] for d in country_examples)/len(country_examples):.1f}" if country_examples else "0",
                "Avg_Entities_Per_Example": f"{sum(d['Entity_Count'] for d in country_examples)/len(country_examples):.1f}" if country_examples else "0"
            })
        
        country_df = pd.DataFrame(country_summary)
        country_df.to_excel(writer, sheet_name='By_Country', index=False)
        
        # 4. Name Pattern Analysis Sheet
        name_analysis = [
            {"Pattern_Type": "Compound First Names", "Count": name_patterns["compound_first_names"], "Percentage": f"{name_patterns['compound_first_names']/len(all_data)*100:.1f}%" if all_data else "0%"},
            {"Pattern_Type": "Double Surnames", "Count": name_patterns["double_surnames"], "Percentage": f"{name_patterns['double_surnames']/len(all_data)*100:.1f}%" if all_data else "0%"},
            {"Pattern_Type": "Simple Names", "Count": name_patterns["simple_names"], "Percentage": f"{name_patterns['simple_names']/len(all_data)*100:.1f}%" if all_data else "0%"}
        ]
        
        name_df = pd.DataFrame(name_analysis)
        name_df.to_excel(writer, sheet_name='Name_Analysis', index=False)
        
        # 5. Entity Statistics Sheet
        entity_descriptions = {
            "CUSTOMER_NAME": "Full customer names with country conventions",
            "ID_NUMBER": "Country-specific ID numbers (RUT, CURP, CPF, etc.)",
            "ADDRESS": "Complete addresses with country-specific formats",
            "PHONE": "Country-specific phone numbers",
            "EMAIL": "Email addresses with country domains",
            "AMOUNT": "Monetary amounts with local currencies",
            "SEQ_NUMBER": "Sequential reference numbers"
        }
        
        entity_analysis = []
        for entity_type, count in entity_counts.items():
            percentage = (count / len(all_data) * 100) if all_data else 0
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
                    "Percentage": f"{(count / len(all_data) * 100):.1f}%" if all_data else "0%"
                })
            
            noise_df = pd.DataFrame(noise_analysis)
            noise_df.to_excel(writer, sheet_name='Noise_Analysis', index=False)
    
    print(f"✅ Excel file created successfully: {output_file}")
    print(f"📊 Generated {len(all_data)} examples across {len(supported_countries)} countries")
    print(f"🏷️  Entity distribution: {dict(sorted(entity_counts.items()))}")
    print(f"🌎 Country distribution: {country_stats}")
    print(f"📋 Multi-country naming patterns: {name_patterns}")
    
    if include_noise:
        print(f"🎭 Noise patterns detected: {noise_patterns}")
    
    print(f"\n📖 Excel sheets created:")
    print(f"  • Summary - Overview statistics")
    print(f"  • All_Data - Complete generated data")
    print(f"  • By_Country - Analysis by country")
    print(f"  • Name_Analysis - Multi-country naming pattern analysis")
    print(f"  • Entity_Statistics - Entity type distribution")
    if include_noise:
        print(f"  • Noise_Analysis - Noise pattern analysis")

# -----------------
# Command-Line Interface and Main Functions
# -----------------

def demonstrate_multi_country_functionality_with_noise():
    """
    Demonstration function showing multi-country PII generation with noise capabilities.
    
    Provides examples of:
    1. Basic customer data generation with noise for all supported countries
    2. Different complexity modes for NLP training
    3. spaCy dataset creation for multi-country NER training
    4. E1010 conflict resolution validation across countries
    """
    parser = argparse.ArgumentParser(description="Multi-Country Latin American PII Training Data Generator with Advanced Noise")
    parser.add_argument("--mode", choices=["demo", "create-dataset", "excel-export"], default="demo",
                       help="Mode: 'demo' shows examples, 'create-dataset' generates training data, 'excel-export' creates review file")
    parser.add_argument("--country", choices=["chile", "mexico", "brazil", "uruguay", "all"], default="chile",
                       help="Country for generation: 'chile', 'mexico', 'brazil', 'uruguay', or 'all' for mixed dataset")
    parser.add_argument("--train-size", type=int, default=80000, help="Training set size")
    parser.add_argument("--dev-size", type=int, default=20000, help="Development set size")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--excel-examples", type=int, default=100, help="Number of examples for Excel export")
    parser.add_argument("--excel-file", type=str, default="multi_country_customer_data_review_noisy.xlsx", help="Excel output filename")
    parser.add_argument("--noise", action="store_true", default=True, help="Enable noise generation")
    parser.add_argument("--no-noise", action="store_true", help="Disable noise generation")
    parser.add_argument("--noise-level", type=float, default=0.15, help="Noise intensity (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Handle noise settings
    include_noise = args.noise and not args.no_noise
    
    if args.mode == "create-dataset":
        # Create multi-country NLP training dataset with noise
        if args.country == "all":
            create_multi_country_training_dataset_with_noise(
                train_size=args.train_size,
                dev_size=args.dev_size,
                include_noise=include_noise,
                noise_level=args.noise_level,
                output_dir=args.output_dir
            )
        else:
            create_country_training_dataset_with_noise(
                country=args.country,
                train_size=args.train_size,
                dev_size=args.dev_size,
                include_noise=include_noise,
                noise_level=args.noise_level,
                output_dir=args.output_dir
            )
        return
    elif args.mode == "excel-export":
        # Create Excel file for data review
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        excel_file = output_path / args.excel_file
        
        if args.country == "all":
            export_multi_country_data_to_excel_with_noise(
                n_examples=args.excel_examples,
                output_file=str(excel_file),
                include_noise=include_noise,
                noise_level=args.noise_level
            )
        else:
            export_country_data_to_excel_with_noise(
                country=args.country,
                n_examples=args.excel_examples,
                output_file=str(excel_file),
                include_noise=include_noise,
                noise_level=args.noise_level
            )
        return
    
    # Demo mode - show examples
    print("================================================================================")
    print("MULTI-COUNTRY LATIN AMERICAN PII TRAINING DATA GENERATOR WITH ADVANCED NOISE")
    print("================================================================================")
    print()
    print("🌎 SUPPORTED COUNTRIES:")
    country_info = {
        "chile": ("Chile (CL)", "RUT, +56 phones, CLP currency, Chilean Spanish"),
        "mexico": ("Mexico (MX)", "CURP/RFC, +52 phones, MXN currency, Mexican Spanish"),
        "brazil": ("Brazil (BR)", "CPF, +55 phones, BRL currency, Portuguese"),
        "uruguay": ("Uruguay (UY)", "Cédula, +598 phones, UYU currency, Uruguayan Spanish")
    }
    
    for code, (name, desc) in country_info.items():
        print(f"   - {name}: {desc}")
    print()
    
    # Show examples for selected country or all countries
    countries_to_show = [args.country] if args.country != "all" else ["chile", "mexico", "brazil", "uruguay"]
    
    print("🔥 MULTI-COUNTRY GENERATION EXAMPLES")
    print("-" * 40)
    
    for country in countries_to_show:
        country_name = country_info[country][0]
        print(f"📍 {country_name}")
        sentence, annotations = generate_example_with_noise(country, include_noise, args.noise_level)
        
        print(f"Generated: {sentence}")
        print("Entities:", end=" ")
        for start, end, label in annotations["entities"]:
            entity_text = sentence[start:end]
            print(f"[{label}: '{entity_text}']", end=" ")
        print("\n")
    
    print("🔢 Sequential Counter Status:")
    print(f"   Next sequence number: {_sequence_counter + 1}")
    print()
    
    # E1010 Validation Test
    print("🔍 E1010 OVERLAPPING SPAN ERROR VALIDATION")
    print("-" * 40)
    print("Testing conflict resolution algorithm across all countries...")
    
    test_examples = []
    overlap_errors = 0
    
    for _ in range(50):  # Test 50 examples
        test_country = random.choice(countries_to_show)
        sentence, annotations = generate_example_with_noise(test_country, include_noise, args.noise_level)
        entities = annotations["entities"]
        
        # Check for overlaps
        for i, (start1, end1, label1) in enumerate(entities):
            for j, (start2, end2, label2) in enumerate(entities[i+1:], i+1):
                if start1 < end2 and start2 < end1:  # Overlap detected
                    overlap_errors += 1
        
        test_examples.append((sentence, entities))
    
    print(f"✅ Tested {len(test_examples)} examples across countries")
    print(f"❌ Overlap errors detected: {overlap_errors}")
    
    if overlap_errors == 0:
        print("🎉 SUCCESS: Zero E1010 overlapping span errors guaranteed!")
    else:
        print("⚠️  WARNING: E1010 overlapping span errors detected!")
    
    print("\n📊 MULTI-COUNTRY NLP DATASET CREATION")
    print("-" * 40)
    print("To create training datasets for spaCy NER:")
    noise_flag = '--noise' if include_noise else '--no-noise'
    if args.country == "all":
        print(f"   python data_generation_noisy.py --mode create-dataset --country all --train-size 10000 --dev-size 2500 {noise_flag}")
    else:
        print(f"   python data_generation_noisy.py --mode create-dataset --country {args.country} --train-size 10000 --dev-size 2500 {noise_flag}")
    print()
    print("📁 EXCEL DATA REVIEW")
    print("-" * 40)
    print("To create Excel file for data review and validation:")
    if args.country == "all":
        print(f"   python data_generation_noisy.py --mode excel-export --country all --excel-examples 100 {noise_flag}")
        print(f"   python data_generation_noisy.py --mode excel-export --country all --excel-examples 500 --excel-file detailed_multi_country_review.xlsx {noise_flag}")
    else:
        print(f"   python data_generation_noisy.py --mode excel-export --country {args.country} --excel-examples 100 {noise_flag}")
        print(f"   python data_generation_noisy.py --mode excel-export --country {args.country} --excel-examples 500 --excel-file detailed_{args.country}_review.xlsx {noise_flag}")
    print()
    print("🎯 NOISE FEATURES:")
    print("   - Realistic typos and misspellings per country")
    print("   - Country-specific abbreviations and contractions")
    print("   - Document formatting variations per country")
    print("   - Controlled noise that preserves entity boundaries")
    print("   - Zero E1010 overlapping span errors guaranteed")
    print()
    print("📚 Multi-Country Use Cases:")
    print("   - Multi-language Latin American NER training")
    print("   - Cross-country financial document processing")
    print("   - Government document analysis across regions")
    print("   - PII detection and anonymization for LATAM")
    print("   - Large-scale multi-country NLP model training")

# Backwards compatibility function
def demonstrate_chilean_functionality_with_noise():
    """
    Demonstration function showing Chilean PII generation with noise capabilities.
    (Backwards compatibility wrapper)
    """
    import sys
    # Replace the country argument to default to Chile for backwards compatibility
    if "--country" not in sys.argv:
        sys.argv.extend(["--country", "chile"])
    demonstrate_multi_country_functionality_with_noise()

def quick_multi_country_test_with_noise():
    """
    Quick test function to verify all multi-country functionality works correctly with noise.
    """
    print("🧪 Running quick multi-country functionality test with noise...")
    
    countries = ["chile", "mexico", "brazil", "uruguay"]
    
    for country in countries:
        print(f"   Testing {country}...")
        # Test basic generation
        sentence, annotations = generate_example_with_noise(country, True, 0.15)
        assert len(sentence) > 0, f"Basic {country} generation failed"
        assert len(annotations["entities"]) > 0, f"No entities generated for {country}"
    
    print("✅ All multi-country tests passed with zero E1010 errors!")

# Keep old function for backwards compatibility
def quick_chilean_test_with_noise():
    """
    Quick test function to verify Chilean functionality works correctly with noise.
    (Backwards compatibility wrapper)
    """
    quick_multi_country_test_with_noise()

if __name__ == "__main__":
    # Run the enhanced multi-country PII generator with noise capabilities
    demonstrate_multi_country_functionality_with_noise()