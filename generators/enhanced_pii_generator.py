"""
Enhanced PII Generator with Additional Types and Variety
=======================================================

This module extends the existing PII generation capabilities with:
- Additional PII types (DIRECTION, LOCATION, POSTAL_CODE, etc.)
- Enhanced variety in existing types
- Better country-specific localization
- Integration with NLP augmentation

Author: Andrés Vera Figueroa (Enhanced by Codegen)
Date: October 2024
Purpose: Comprehensive PII generation for robust NER training
"""

import random
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class PIIEntity:
    """Represents a PII entity with metadata"""
    text: str
    label: str
    start: int
    end: int
    country: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

class EnhancedPIIGenerator:
    """Enhanced PII generator with additional types and variety"""
    
    def __init__(self):
        self.sequence_counter = 10000
        self.setup_country_data()
    
    def setup_country_data(self):
        """Setup enhanced country-specific data"""
        self.ENHANCED_COUNTRY_DATA = {
            'chile': {
                'first_names': [
                    # Enhanced Chilean names with more variety
                    "AGUSTÍN", "ALEJANDRO", "ALONSO", "ÁLVARO", "ANDRÉS", "ÁXEL", "BAUTISTA", 
                    "BENJAMÍN", "BRUNO", "CALEB", "CAMILO", "CARLOS", "CRISTÓBAL", "CRISTIAN",
                    "DAMIÁN", "DANIEL", "DAVID", "DIEGO", "EDUARDO", "ELÍAS", "EMILIANO",
                    "EMMANUEL", "ENRIQUE", "ESTEBAN", "ETHAN", "FEDERICO", "FERNANDO",
                    "FRANCISCO", "GABRIEL", "GAEL", "GASPAR", "GERMÁN", "GUSTAVO", "HERNÁN",
                    # Female names
                    "MARÍA", "FRANCISCA", "SOFÍA", "JAVIERA", "VALENTINA", "ISIDORA", "AMANDA",
                    "ANTONIA", "EMILIA", "CATALINA", "TRINIDAD", "ESPERANZA", "CONSTANZA",
                    "MAGDALENA", "CAROLINA", "PATRICIA", "ALEJANDRA", "CLAUDIA", "MÓNICA"
                ],
                'surnames': [
                    "GONZÁLEZ", "RODRÍGUEZ", "MUÑOZ", "ROJAS", "SILVA", "MORALES", "LÓPEZ",
                    "MARTÍNEZ", "GARCÍA", "HERNÁNDEZ", "PÉREZ", "SÁNCHEZ", "TORRES", "FLORES",
                    "VARGAS", "CASTILLO", "JIMÉNEZ", "MORENO", "GUTIÉRREZ", "HERRERA",
                    "CONTRERAS", "SEPÚLVEDA", "ESPINOZA", "ARAYA", "TAPIA", "PIZARRO"
                ],
                'directions': [
                    # Cardinal directions
                    "Norte", "Sur", "Este", "Oeste", "Noreste", "Noroeste", "Sureste", "Suroeste",
                    # Relative directions
                    "hacia el centro", "rumbo al puerto", "dirección cordillera", "hacia la costa",
                    "en dirección norte", "con rumbo sur", "hacia el oriente", "rumbo poniente",
                    # Specific Chilean directions
                    "hacia Las Condes", "rumbo a Providencia", "dirección Maipú", "hacia Ñuñoa",
                    "rumbo a Valparaíso", "hacia la Quinta Región", "dirección aeropuerto"
                ],
                'locations': [
                    # Specific locations beyond cities
                    "Mall Plaza Norte", "Costanera Center", "Plaza de Armas", "Cerro San Cristóbal",
                    "Parque Forestal", "Barrio Bellavista", "Las Condes", "Providencia",
                    "Centro Histórico", "Puerto de Valparaíso", "Viña del Mar", "Aeropuerto SCL",
                    "Terminal de Buses", "Metro Universidad de Chile", "Estación Central"
                ],
                'postal_codes': [
                    # Chilean postal codes format
                    "7500000", "8320000", "4030000", "1230000", "5110000", "8340000",
                    "7630000", "3460000", "2820000", "5480000", "6200000", "1070000"
                ],
                'regions': [
                    "Región de Arica y Parinacota", "Región de Tarapacá", "Región de Antofagasta",
                    "Región de Atacama", "Región de Coquimbo", "Región de Valparaíso",
                    "Región Metropolitana", "Región del Libertador", "Región del Maule",
                    "Región del Biobío", "Región de La Araucanía", "Región de Los Ríos",
                    "Región de Los Lagos", "Región de Aysén", "Región de Magallanes"
                ]
            },
            'mexico': {
                'first_names': [
                    "JUAN", "MARÍA", "JOSÉ", "ANA", "LUIS", "CARMEN", "CARLOS", "GUADALUPE",
                    "ANTONIO", "ROSA", "JESÚS", "TERESA", "ALEJANDRO", "PATRICIA", "MANUEL",
                    "ELIZABETH", "FRANCISCO", "LETICIA", "DAVID", "YOLANDA", "MIGUEL",
                    "MARTHA", "PEDRO", "LAURA", "JORGE", "SILVIA", "RICARDO", "ELENA"
                ],
                'surnames': [
                    "GARCÍA", "HERNÁNDEZ", "LÓPEZ", "MARTÍNEZ", "GONZÁLEZ", "PÉREZ", "SÁNCHEZ",
                    "RAMÍREZ", "CRUZ", "FLORES", "GÓMEZ", "MORALES", "VÁZQUEZ", "JIMÉNEZ",
                    "RUIZ", "DÍAZ", "MORENO", "MUÑOZ", "ÁLVAREZ", "ROMERO", "GUTIÉRREZ"
                ],
                'directions': [
                    "Norte", "Sur", "Este", "Oeste", "Noreste", "Noroeste", "Sureste", "Suroeste",
                    "hacia el centro", "rumbo al zócalo", "dirección Polanco", "hacia Coyoacán",
                    "en dirección norte", "con rumbo sur", "hacia el oriente", "rumbo poniente",
                    "hacia Santa Fe", "rumbo a Insurgentes", "dirección Roma Norte"
                ],
                'locations': [
                    "Zócalo", "Chapultepec", "Polanco", "Roma Norte", "Condesa", "Santa Fe",
                    "Coyoacán", "Xochimilco", "Centro Histórico", "Aeropuerto CDMX",
                    "Terminal Norte", "Metro Insurgentes", "Plaza Garibaldi"
                ],
                'postal_codes': [
                    "01000", "03100", "06700", "11000", "03810", "06140", "04100", "06600"
                ],
                'regions': [
                    "Ciudad de México", "Estado de México", "Jalisco", "Nuevo León", "Puebla",
                    "Guanajuato", "Veracruz", "Michoacán", "Oaxaca", "Chiapas"
                ]
            },
            'brazil': {
                'first_names': [
                    "JOÃO", "MARIA", "JOSÉ", "ANA", "CARLOS", "ANTÔNIA", "PEDRO", "FRANCISCA",
                    "PAULO", "ADRIANA", "LUIZ", "JULIANA", "MARCOS", "MÁRCIA", "ANTÔNIO",
                    "FERNANDA", "FRANCISCO", "PATRÍCIA", "DANIEL", "ALINE", "RAFAEL", "CAMILA"
                ],
                'surnames': [
                    "SILVA", "SANTOS", "OLIVEIRA", "SOUZA", "RODRIGUES", "FERREIRA", "ALVES",
                    "PEREIRA", "LIMA", "GOMES", "RIBEIRO", "CARVALHO", "ALMEIDA", "LOPES",
                    "SOARES", "FERNANDES", "VIEIRA", "BARBOSA", "ROCHA", "DIAS", "NASCIMENTO"
                ],
                'directions': [
                    "Norte", "Sul", "Leste", "Oeste", "Nordeste", "Noroeste", "Sudeste", "Sudoeste",
                    "em direção ao centro", "rumo à zona sul", "direção Copacabana", "para Ipanema",
                    "em direção norte", "com rumo sul", "para o leste", "rumo oeste",
                    "para a Barra", "rumo ao Leblon", "direção Botafogo"
                ],
                'locations': [
                    "Copacabana", "Ipanema", "Leblon", "Barra da Tijuca", "Botafogo", "Centro",
                    "Zona Sul", "Zona Norte", "Aeroporto GIG", "Rodoviária Novo Rio",
                    "Metro Carioca", "Cristo Redentor", "Pão de Açúcar", "Maracanã"
                ],
                'postal_codes': [
                    "01000-000", "20000-000", "30000-000", "40000-000", "50000-000", "60000-000"
                ],
                'regions': [
                    "São Paulo", "Rio de Janeiro", "Minas Gerais", "Bahia", "Paraná", "Rio Grande do Sul",
                    "Pernambuco", "Ceará", "Pará", "Santa Catarina", "Goiás", "Maranhão"
                ]
            },
            'uruguay': {
                'first_names': [
                    "JUAN", "MARÍA", "CARLOS", "ANA", "LUIS", "ELENA", "JOSÉ", "PATRICIA",
                    "MIGUEL", "CAROLINA", "FERNANDO", "GABRIELA", "RICARDO", "MÓNICA",
                    "ANTONIO", "ALEJANDRA", "MANUEL", "CLAUDIA", "JORGE", "FRANCISCA"
                ],
                'surnames': [
                    "RODRÍGUEZ", "GONZÁLEZ", "PÉREZ", "LÓPEZ", "MARTÍNEZ", "GARCÍA", "FERNÁNDEZ",
                    "SÁNCHEZ", "ROMERO", "ÁLVAREZ", "TORRES", "FLORES", "VARGAS", "CASTRO"
                ],
                'directions': [
                    "Norte", "Sur", "Este", "Oeste", "Noreste", "Noroeste", "Sureste", "Suroeste",
                    "hacia el centro", "rumbo a la rambla", "dirección Ciudad Vieja", "hacia Pocitos",
                    "en dirección norte", "con rumbo sur", "hacia el este", "rumbo oeste",
                    "hacia Carrasco", "rumbo a Punta del Este", "dirección Colonia"
                ],
                'locations': [
                    "Ciudad Vieja", "Pocitos", "Carrasco", "Punta Carretas", "Centro", "Cordón",
                    "Rambla", "Puerto de Montevideo", "Aeropuerto de Carrasco", "Terminal Tres Cruces",
                    "Plaza Independencia", "Mercado del Puerto", "Parque Rodó"
                ],
                'postal_codes': [
                    "11000", "11100", "11200", "11300", "11400", "11500", "20000", "30000"
                ],
                'regions': [
                    "Montevideo", "Canelones", "Maldonado", "Colonia", "San José", "Florida",
                    "Soriano", "Paysandú", "Salto", "Artigas", "Rivera", "Tacuarembó"
                ]
            }
        }
    
    def generate_direction(self, country: str = "chile") -> str:
        """Generate direction/orientation information"""
        country_data = self.ENHANCED_COUNTRY_DATA.get(country, self.ENHANCED_COUNTRY_DATA['chile'])
        return random.choice(country_data['directions'])
    
    def generate_location(self, country: str = "chile") -> str:
        """Generate specific location information"""
        country_data = self.ENHANCED_COUNTRY_DATA.get(country, self.ENHANCED_COUNTRY_DATA['chile'])
        return random.choice(country_data['locations'])
    
    def generate_postal_code(self, country: str = "chile") -> str:
        """Generate postal/zip codes"""
        country_data = self.ENHANCED_COUNTRY_DATA.get(country, self.ENHANCED_COUNTRY_DATA['chile'])
        return random.choice(country_data['postal_codes'])
    
    def generate_region(self, country: str = "chile") -> str:
        """Generate region/state information"""
        country_data = self.ENHANCED_COUNTRY_DATA.get(country, self.ENHANCED_COUNTRY_DATA['chile'])
        return random.choice(country_data['regions'])
    
    def generate_enhanced_phone(self, country: str = "chile") -> str:
        """Generate enhanced phone numbers with more variety"""
        if country == "chile":
            formats = [
                "+56 9 {} {} {}",  # Standard mobile format
                "+56-9-{}-{}-{}",  # Dash format
                "(+56) 9 {} {} {}",  # Parentheses format
                "56 9 {} {} {}",    # No plus
                "09 {} {} {}",      # Local format
                "+56 2 {} {} {}",   # Santiago landline
                "+56 32 {} {} {}"   # Valparaíso landline
            ]
            
            if "9" in formats[random.randint(0, 4)]:  # Mobile
                num1 = random.randint(1000, 9999)
                num2 = random.randint(1000, 9999)
                num3 = random.randint(100, 999)
                return random.choice(formats[:5]).format(num1, num2, num3)
            else:  # Landline
                num1 = random.randint(100, 999)
                num2 = random.randint(1000, 9999)
                num3 = random.randint(100, 999)
                return random.choice(formats[5:]).format(num1, num2, num3)
        
        elif country == "mexico":
            formats = [
                "+52 {} {} {}",     # Standard format
                "+52-{}-{}-{}",     # Dash format
                "(+52) {} {} {}",   # Parentheses format
                "52 {} {} {}",      # No plus
                "01 {} {} {}"       # Local format
            ]
            area_codes = ["55", "33", "81", "222", "664", "477", "656", "871", "442", "444"]
            area = random.choice(area_codes)
            num1 = random.randint(100, 999)
            num2 = random.randint(1000, 9999)
            return random.choice(formats).format(area, num1, num2)
        
        elif country == "brazil":
            formats = [
                "+55 {} {} {}",     # Standard format
                "+55-{}-{}-{}",     # Dash format
                "(+55) {} {} {}",   # Parentheses format
                "55 {} {} {}",      # No plus
                "0{} {} {}"         # Local format
            ]
            area_codes = ["11", "21", "31", "41", "51", "61", "71", "81", "85", "62"]
            area = random.choice(area_codes)
            if random.choice([True, False]):  # Mobile
                num1 = random.randint(90000, 99999)
                num2 = random.randint(1000, 9999)
            else:  # Landline
                num1 = random.randint(2000, 5999)
                num2 = random.randint(1000, 9999)
            return random.choice(formats).format(area, num1, num2)
        
        elif country == "uruguay":
            formats = [
                "+598 {} {} {}",    # Standard format
                "+598-{}-{}-{}",    # Dash format
                "(+598) {} {} {}",  # Parentheses format
                "598 {} {} {}",     # No plus
                "0{} {} {}"         # Local format
            ]
            if random.choice([True, False]):  # Mobile
                area = "9"
                num1 = random.randint(100, 999)
                num2 = random.randint(1000, 9999)
            else:  # Landline
                area = "2"  # Montevideo
                num1 = random.randint(100, 999)
                num2 = random.randint(1000, 9999)
            return random.choice(formats).format(area, num1, num2)
        
        return self.generate_enhanced_phone("chile")  # Default fallback
    
    def generate_enhanced_sequence(self, country: str = "chile") -> str:
        """Generate enhanced sequence numbers with more variety"""
        self.sequence_counter += 1
        
        formats = [
            # Standard formats
            f"{self.sequence_counter}",
            f"#{self.sequence_counter}",
            f"N°{self.sequence_counter}",
            f"REF-{self.sequence_counter}",
            f"DOC-{self.sequence_counter}",
            f"TRX-{self.sequence_counter}",
            
            # Country-specific formats
            f"OP-{self.sequence_counter}",      # Operación
            f"FOLIO-{self.sequence_counter}",   # Folio
            f"BOLETA-{self.sequence_counter}",  # Boleta
            f"FACTURA-{self.sequence_counter}", # Factura
            
            # Alphanumeric formats
            f"A{self.sequence_counter}",
            f"B{self.sequence_counter}",
            f"C{self.sequence_counter}",
            f"X{self.sequence_counter}",
            f"Z{self.sequence_counter}",
            
            # Complex formats
            f"{random.choice(['A', 'B', 'C'])}{self.sequence_counter:06d}",
            f"{datetime.now().year}{self.sequence_counter:04d}",
            f"{random.randint(10, 99)}-{self.sequence_counter}",
            f"{self.sequence_counter}-{random.randint(10, 99)}"
        ]
        
        return random.choice(formats)
    
    def generate_enhanced_date(self, country: str = "chile") -> str:
        """Generate enhanced dates with more variety and contexts"""
        # Generate a realistic recent date (within last 3 years)
        start_date = datetime.now() - timedelta(days=1095)  # 3 years ago
        end_date = datetime.now() + timedelta(days=365)     # 1 year future
        
        time_between = end_date - start_date
        random_days = random.randint(0, time_between.days)
        random_date = start_date + timedelta(days=random_days)
        
        if country == "chile":
            formats = [
                random_date.strftime("%d/%m/%Y"),        # 15/08/2024
                random_date.strftime("%d-%m-%Y"),        # 15-08-2024
                random_date.strftime("%d.%m.%Y"),        # 15.08.2024
                random_date.strftime("%d/%m/%y"),        # 15/08/24
                random_date.strftime("%d de %B de %Y"),  # 15 de agosto de 2024
                random_date.strftime("%d-%b-%Y"),        # 15-ago-2024
                random_date.strftime("%d%m%Y"),          # 15082024
                random_date.strftime("%Y%m%d"),          # 20240815
                random_date.strftime("%m%d%Y"),          # 08152024
                random_date.strftime("%Y-%m-%d"),        # 2024-08-15 (ISO)
                f"Fecha: {random_date.strftime('%d/%m/%Y')}",
                f"Vencimiento: {random_date.strftime('%d/%m/%Y')}",
                f"Emisión: {random_date.strftime('%d-%m-%Y')}",
                f"Vigencia: {random_date.strftime('%d.%m.%Y')}"
            ]
        elif country == "mexico":
            formats = [
                random_date.strftime("%d/%m/%Y"),        # 15/08/2024
                random_date.strftime("%d-%m-%Y"),        # 15-08-2024
                random_date.strftime("%m/%d/%Y"),        # 08/15/2024 (US influence)
                random_date.strftime("%d de %B de %Y"),  # 15 de agosto de 2024
                random_date.strftime("%Y-%m-%d"),        # 2024-08-15
                f"Fecha: {random_date.strftime('%d/%m/%Y')}",
                f"Vencimiento: {random_date.strftime('%d/%m/%Y')}",
                f"Emisión: {random_date.strftime('%d-%m-%Y')}"
            ]
        elif country == "brazil":
            formats = [
                random_date.strftime("%d/%m/%Y"),        # 15/08/2024
                random_date.strftime("%d-%m-%Y"),        # 15-08-2024
                random_date.strftime("%d.%m.%Y"),        # 15.08.2024
                random_date.strftime("%d de %B de %Y"),  # 15 de agosto de 2024
                random_date.strftime("%Y-%m-%d"),        # 2024-08-15
                f"Data: {random_date.strftime('%d/%m/%Y')}",
                f"Vencimento: {random_date.strftime('%d/%m/%Y')}",
                f"Emissão: {random_date.strftime('%d-%m-%Y')}"
            ]
        elif country == "uruguay":
            formats = [
                random_date.strftime("%d/%m/%Y"),        # 15/08/2024
                random_date.strftime("%d-%m-%Y"),        # 15-08-2024
                random_date.strftime("%d.%m.%Y"),        # 15.08.2024
                random_date.strftime("%d de %B de %Y"),  # 15 de agosto de 2024
                random_date.strftime("%Y-%m-%d"),        # 2024-08-15
                f"Fecha: {random_date.strftime('%d/%m/%Y')}",
                f"Vencimiento: {random_date.strftime('%d/%m/%Y')}"
            ]
        else:
            formats = [
                random_date.strftime("%d/%m/%Y"),
                random_date.strftime("%d-%m-%Y"),
                random_date.strftime("%Y-%m-%d")
            ]
        
        return random.choice(formats)
    
    def generate_all_pii_types(self, country: str = "chile") -> Dict[str, str]:
        """Generate all PII types for comprehensive testing"""
        # Import existing functions (would need to be imported from main module)
        from Spacy.data_generation_noisy import (
            generate_name_components, generate_id, generate_email, generate_amount
        )
        
        # Generate basic PII
        first_name, last_name, complete_surname = generate_name_components(country)
        complete_name = f"{first_name} {complete_surname}"
        
        return {
            'CUSTOMER_NAME': complete_name,
            'ID_NUMBER': generate_id(country),
            'ADDRESS': f"{random.choice(self.ENHANCED_COUNTRY_DATA[country]['streets'])} {random.randint(10, 999)}",
            'PHONE_NUMBER': self.generate_enhanced_phone(country),
            'EMAIL': generate_email(first_name, last_name, country),
            'AMOUNT': generate_amount(country),
            'SEQ_NUMBER': self.generate_enhanced_sequence(country),
            'DATE': self.generate_enhanced_date(country),
            'DIRECTION': self.generate_direction(country),
            'LOCATION': self.generate_location(country),
            'POSTAL_CODE': self.generate_postal_code(country),
            'REGION': self.generate_region(country)
        }
    
    def validate_pii_variety(self, country: str = "chile", samples: int = 100) -> Dict[str, Any]:
        """Validate that PII generation has sufficient variety"""
        results = {pii_type: set() for pii_type in [
            'CUSTOMER_NAME', 'ID_NUMBER', 'ADDRESS', 'PHONE_NUMBER', 'EMAIL', 
            'AMOUNT', 'SEQ_NUMBER', 'DATE', 'DIRECTION', 'LOCATION', 'POSTAL_CODE', 'REGION'
        ]}
        
        for _ in range(samples):
            pii_data = self.generate_all_pii_types(country)
            for pii_type, value in pii_data.items():
                results[pii_type].add(value)
        
        # Calculate variety metrics
        variety_report = {}
        for pii_type, values in results.items():
            unique_count = len(values)
            variety_percentage = (unique_count / samples) * 100
            variety_report[pii_type] = {
                'unique_values': unique_count,
                'total_samples': samples,
                'variety_percentage': variety_percentage,
                'sufficient_variety': variety_percentage > 70  # 70% threshold
            }
        
        return variety_report

# Global instance for easy access
enhanced_pii_generator = EnhancedPIIGenerator()

# Convenience functions for backward compatibility
def generate_enhanced_direction(country: str = "chile") -> str:
    return enhanced_pii_generator.generate_direction(country)

def generate_enhanced_location(country: str = "chile") -> str:
    return enhanced_pii_generator.generate_location(country)

def generate_enhanced_postal_code(country: str = "chile") -> str:
    return enhanced_pii_generator.generate_postal_code(country)

def generate_enhanced_region(country: str = "chile") -> str:
    return enhanced_pii_generator.generate_region(country)

def generate_enhanced_phone(country: str = "chile") -> str:
    return enhanced_pii_generator.generate_enhanced_phone(country)

def generate_enhanced_sequence(country: str = "chile") -> str:
    return enhanced_pii_generator.generate_enhanced_sequence(country)

def generate_enhanced_date(country: str = "chile") -> str:
    return enhanced_pii_generator.generate_enhanced_date(country)

def validate_pii_variety(country: str = "chile", samples: int = 100) -> Dict[str, Any]:
    return enhanced_pii_generator.validate_pii_variety(country, samples)
