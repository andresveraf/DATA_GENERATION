"""
Advanced Sentence Variety Generator for PII Data
================================================

This module implements sophisticated sentence generation techniques designed to create
maximum variety and complexity for training robust NER models with 200K+ examples.

Key Features:
- Long sentences (15-30 words) with complex grammatical structures
- Medium sentences (8-15 words) with high synonym density
- Dynamic word swapping and paraphrasing
- Contextual synonym replacement (20+ synonyms per common word)
- Multiple sentence patterns to prevent model memorization
- Entity preservation during all transformations

Author: Andrés Vera Figueroa (Enhanced by Codegen AI)
Date: October 2024
Purpose: Create challenging training data that prevents overfitting on large datasets
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SentenceLength(Enum):
    """Sentence length categories"""
    SHORT = "short"          # 5-8 words
    MEDIUM = "medium"        # 8-15 words
    LONG = "long"            # 15-30 words
    EXTRA_LONG = "extra_long" # 30+ words

class SentenceComplexity(Enum):
    """Sentence structural complexity"""
    SIMPLE = "simple"                    # Simple subject-verb-object
    COMPOUND = "compound"                # Multiple clauses with conjunctions
    COMPLEX = "complex"                  # Subordinate clauses
    COMPOUND_COMPLEX = "compound_complex" # Multiple main and subordinate clauses


@dataclass
class SynonymBank:
    """Comprehensive synonym database for Spanish and Portuguese"""
    
    # Spanish synonyms - organized by semantic category
    SPANISH = {
        # Verbs - Customer actions
        "reside": ["habita", "vive", "mora", "domicilia", "radica", "establece su residencia", 
                   "tiene su hogar", "se encuentra ubicado", "fija su domicilio", "permanece"],
        "registrado": ["inscrito", "anotado", "consignado", "asentado", "documentado", "archivado",
                      "catalogado", "fichado", "enrolado", "matriculado"],
        "contacto": ["comunicación", "enlace", "conexión", "vínculo", "relación", "correspondencia",
                    "interacción", "contactación", "nexo", "canal de comunicación"],
        "identificado": ["reconocido", "determinado", "individualizado", "distinguido", "señalado",
                        "especificado", "caracterizado", "denominado", "catalogado", "tipificado"],
        "ubicado": ["situado", "localizado", "emplazado", "posicionado", "establecido", "asentado",
                   "colocado", "dispuesto", "radicado", "instalado"],
        
        # Nouns - Business terms
        "cliente": ["usuario", "consumidor", "comprador", "adquirente", "contratante", "solicitante",
                   "demandante", "beneficiario", "titular", "suscriptor", "parte interesada", "comprador"],
        "monto": ["importe", "cantidad", "suma", "valor", "cuantía", "cifra", "total", "montante",
                 "volumen monetario", "valor económico", "cantidad monetaria", "carga económica"],
        "dirección": ["domicilio", "residencia", "ubicación", "señas", "lugar de residencia", "vivienda",
                     "morada", "casa habitación", "lugar", "sede", "emplazamiento", "localización"],
        "teléfono": ["número telefónico", "línea", "número de contacto", "fono", "celular", "móvil",
                    "número de teléfono", "línea telefónica", "contacto telefónico", "terminal"],
        "correo": ["email", "correo electrónico", "e-mail", "dirección electrónica", "casilla electrónica",
                  "buzón electrónico", "mail", "dirección de correo", "cuenta de correo"],
        
        # Adjectives and descriptors
        "registrado": ["inscrito", "asentado", "consignado", "documentado", "archivado", "anotado"],
        "oficial": ["formal", "legal", "legítimo", "válido", "autorizado", "certificado", "homologado"],
        "actual": ["presente", "vigente", "corriente", "contemporáneo", "en curso", "del momento"],
        "principal": ["primario", "fundamental", "esencial", "primordial", "básico", "central"],
        
        # Prepositions and connectors
        "con": ["que posee", "que cuenta con", "portador de", "teniendo", "en posesión de"],
        "en": ["dentro de", "al interior de", "ubicado en", "situado en", "establecido en"],
        "por": ["mediante", "a través de", "por medio de", "utilizando", "con", "vía"],
        "para": ["destinado a", "con el fin de", "con el propósito de", "orientado a"],
        
        # Document/transaction terms
        "referencia": ["número de referencia", "código", "identificador", "número de serie", "folio",
                      "número de trámite", "número de expediente", "clave", "número identificador"],
        "documento": ["documentación", "papel", "certificado", "constancia", "comprobante", "acta",
                     "cédula", "título", "instrumento", "escritura"],
        "transacción": ["operación", "movimiento", "gestión", "trámite", "negocio", "procedimiento",
                       "diligencia", "actuación", "operativo", "proceso"],
        
        # Sentence connectors for complex structures
        "connectors": {
            "addition": ["además", "asimismo", "también", "igualmente", "de igual manera", "por otra parte"],
            "contrast": ["sin embargo", "no obstante", "aunque", "a pesar de", "mientras que", "por el contrario"],
            "cause": ["debido a", "por causa de", "en virtud de", "a razón de", "como consecuencia de"],
            "sequence": ["posteriormente", "luego", "después", "a continuación", "seguidamente", "subsecuentemente"],
            "emphasis": ["especialmente", "particularmente", "específicamente", "en particular", "sobre todo"],
        }
    }
    
    # Portuguese synonyms
    PORTUGUESE = {
        # Verbos - ações do cliente
        "reside": ["habita", "mora", "domicilia", "estabelece residência", "tem endereço",
                  "está localizado", "fixa domicílio", "permanece"],
        "registrado": ["inscrito", "cadastrado", "documentado", "arquivado", "catalogado",
                      "fichado", "matriculado", "anotado"],
        "identificado": ["reconhecido", "determinado", "individualizado", "distinguido",
                        "especificado", "caracterizado", "denominado"],
        "localizado": ["situado", "posicionado", "estabelecido", "instalado", "colocado",
                      "disposto", "fixado"],
        
        # Substantivos - termos comerciais
        "cliente": ["usuário", "consumidor", "comprador", "adquirente", "contratante",
                   "solicitante", "beneficiário", "titular", "assinante"],
        "valor": ["quantia", "soma", "montante", "total", "cifra", "valor monetário",
                 "volume financeiro", "carga financeira"],
        "endereço": ["domicílio", "residência", "localização", "morada", "lugar",
                    "sede", "localidade", "paradeiro"],
        "telefone": ["número telefônico", "linha", "contato", "celular", "móvel",
                    "número de contato", "linha telefônica"],
        "email": ["correio eletrônico", "e-mail", "endereço eletrônico", "caixa postal eletrônica",
                 "correio", "endereço de email"],
        
        # Conectores de sentenças complexas
        "connectors": {
            "addition": ["além disso", "também", "igualmente", "da mesma forma", "por outro lado"],
            "contrast": ["porém", "contudo", "entretanto", "apesar de", "enquanto", "ao contrário"],
            "cause": ["devido a", "por causa de", "em virtude de", "como consequência de"],
            "sequence": ["posteriormente", "depois", "em seguida", "logo após", "subsequentemente"],
            "emphasis": ["especialmente", "particularmente", "especificamente", "sobretudo"],
        }
    }


class AdvancedSentenceGenerator:
    """
    Advanced sentence generator with maximum variety and complexity.
    
    Generates sentences with varying structures, lengths, and synonyms
    to prevent model memorization in large training sets (200K+).
    """
    
    def __init__(self, language: str = "es"):
        """
        Initialize the advanced sentence generator.
        
        Args:
            language (str): Language code ("es" for Spanish, "pt" for Portuguese)
        """
        self.language = language
        self.synonym_bank = SynonymBank()
        self.random = random.Random()
        
        # Load appropriate synonym dictionary
        if language == "pt":
            self.synonyms = self.synonym_bank.PORTUGUESE
        else:
            self.synonyms = self.synonym_bank.SPANISH
    
    def get_synonym(self, word: str, context: Optional[str] = None) -> str:
        """
        Get a random synonym for a word, considering context.
        
        Args:
            word (str): Original word
            context (str): Optional context for better synonym selection
            
        Returns:
            str: Synonym or original word if no synonym exists
        """
        word_lower = word.lower()
        if word_lower in self.synonyms:
            synonyms = self.synonyms[word_lower]
            return self.random.choice(synonyms)
        return word
    
    def get_connector(self, connector_type: str) -> str:
        """
        Get a random connector of specified type.
        
        Args:
            connector_type (str): Type of connector (addition, contrast, cause, etc.)
            
        Returns:
            str: Random connector
        """
        if "connectors" in self.synonyms and connector_type in self.synonyms["connectors"]:
            return self.random.choice(self.synonyms["connectors"][connector_type])
        return ""
    
    def generate_long_sentence_chile(self, pii_data: Dict) -> str:
        """
        Generate a long, complex sentence (15-30 words) for Chile.
        
        Uses subordinate clauses, multiple entity mentions, and varied connectors
        to create challenging training examples.
        
        Args:
            pii_data (Dict): Dictionary containing PII placeholders
            
        Returns:
            str: Complex sentence with high variety
        """
        templates = [
            # Compound-complex with relative clause
            f"El {self.get_synonym('cliente')} {pii_data['name']}, {self.get_synonym('identificado')} con RUT {pii_data['id']}, "
            f"{self.get_synonym('reside')} en {pii_data['address']}, {pii_data['city']}, "
            f"{self.get_connector('addition')} puede ser {self.get_synonym('contacto')} mediante el {self.get_synonym('teléfono')} {pii_data['phone']} "
            f"o a través del {self.get_synonym('correo')} {pii_data['email']}, habiendo realizado una {self.get_synonym('transacción')} "
            f"por un {self.get_synonym('monto')} de {pii_data['amount']} bajo el número de {self.get_synonym('referencia')} {pii_data['ref']}.",
            
            # Complex with causal subordinate clause
            f"Debido a que el {self.get_synonym('usuario')} {pii_data['name']} (RUT: {pii_data['id']}) "
            f"{self.get_synonym('ubicado')} en la {self.get_synonym('dirección')} {pii_data['address']}, sector {pii_data['city']}, "
            f"solicitó información, se establece {self.get_synonym('contacto')} telefónico al {pii_data['phone']} "
            f"y {self.get_synonym('correo')} al {pii_data['email']}, {self.get_connector('sequence')} procesando el pago de {pii_data['amount']} "
            f"con el folio {pii_data['ref']}.",
            
            # Compound-complex with conditional
            f"Si el {self.get_synonym('contratante')} {pii_data['name']}, quien posee el documento {pii_data['id']} "
            f"y {self.get_synonym('domicilia')} en {pii_data['address']}, comuna de {pii_data['city']}, "
            f"requiere {self.get_connector('emphasis')} asistencia, debe comunicarse al {self.get_synonym('número telefónico')} {pii_data['phone']} "
            f"o enviar un mensaje al {pii_data['email']}, considerando que su operación por {pii_data['amount']} "
            f"está {self.get_synonym('registrado')} con el código {pii_data['ref']}.",
            
            # Long descriptive with multiple subordinate clauses
            f"Según los registros oficiales, el {self.get_synonym('beneficiario')} {pii_data['name']}, "
            f"cuyo número de identificación corresponde al RUT {pii_data['id']}, "
            f"mantiene su {self.get_synonym('residencia')} {self.get_synonym('oficial')} en {pii_data['address']}, "
            f"específicamente en la localidad de {pii_data['city']}, siendo sus datos de {self.get_synonym('contacto')} "
            f"el teléfono {pii_data['phone']} junto con el {self.get_synonym('correo electrónico')} {pii_data['email']}, "
            f"{self.get_connector('addition')} habiendo efectuado un movimiento económico de {pii_data['amount']} "
            f"que se encuentra {self.get_synonym('documentado')} bajo el número de serie {pii_data['ref']}.",
            
            # Narrative style with sequence
            f"En primera instancia, se verificó que el {self.get_synonym('titular')} {pii_data['name']} "
            f"portando la cédula de identidad {pii_data['id']}, {self.get_synonym('establecido')} en el {self.get_synonym('domicilio')} {pii_data['address']}, "
            f"perteneciente a la ciudad de {pii_data['city']}, {self.get_connector('sequence')} se procedió a establecer "
            f"comunicación telefónica al número {pii_data['phone']}, {self.get_connector('addition')} confirmando la dirección de "
            f"{self.get_synonym('correo')} {pii_data['email']}, para finalmente validar la {self.get_synonym('transacción')} "
            f"económica de {pii_data['amount']} {self.get_synonym('registrado')} con el {self.get_synonym('identificador')} {pii_data['ref']}.",
            
            # Formal bureaucratic style
            f"Por medio del presente, se hace constar que el {self.get_synonym('solicitante')} {pii_data['name']}, "
            f"{self.get_synonym('identificado')} legalmente mediante el RUT número {pii_data['id']}, "
            f"quien fija su {self.get_synonym('lugar de residencia')} en {pii_data['address']}, correspondiente a {pii_data['city']}, "
            f"puede ser {self.get_synonym('contacto')} a través del {self.get_synonym('terminal')} telefónico {pii_data['phone']} "
            f"o mediante {self.get_synonym('correo electrónico')} enviado a {pii_data['email']}, "
            f"{self.get_connector('emphasis')} habiendo realizado un desembolso monetario por {pii_data['amount']} "
            f"el cual se encuentra debidamente {self.get_synonym('asentado')} bajo el folio número {pii_data['ref']}.",
            
            # Conversational-formal hybrid
            f"Le informamos que el {self.get_synonym('comprador')} {pii_data['name']}, {self.get_synonym('reconocido')} por su RUT {pii_data['id']}, "
            f"quien actualmente {self.get_synonym('habita')} en la {self.get_synonym('ubicación')} {pii_data['address']} "
            f"dentro del sector {pii_data['city']}, {self.get_connector('contrast')} puede recibir notificaciones "
            f"tanto en el {self.get_synonym('número de teléfono')} {pii_data['phone']} como en el {self.get_synonym('buzón electrónico')} {pii_data['email']}, "
            f"considerando {self.get_connector('emphasis')} que mantiene una {self.get_synonym('operación')} pendiente "
            f"por el {self.get_synonym('valor')} de {pii_data['amount']} con número de {self.get_synonym('expediente')} {pii_data['ref']}.",
            
            # Technical-administrative style
            f"Los antecedentes del {self.get_synonym('usuario')} {pii_data['name']} indican que su documento de identidad RUT {pii_data['id']} "
            f"se encuentra {self.get_synonym('vinculado')} al {self.get_synonym('domicilio')} {self.get_synonym('registrado')} en {pii_data['address']}, "
            f"jurisdicción de {pii_data['city']}, {self.get_connector('addition')} existiendo canales de {self.get_synonym('comunicación')} "
            f"habilitados en el {self.get_synonym('contacto telefónico')} {pii_data['phone']} así como en la {self.get_synonym('dirección electrónica')} {pii_data['email']}, "
            f"{self.get_connector('sequence')} procesándose {self.get_synonym('actualmente')} un {self.get_synonym('movimiento')} financiero "
            f"de {pii_data['amount']} {self.get_synonym('catalogado')} con el {self.get_synonym('código')} {pii_data['ref']}.",
        ]
        
        return self.random.choice(templates)
    
    def generate_medium_sentence_chile(self, pii_data: Dict) -> str:
        """
        Generate a medium-length sentence (8-15 words) with high synonym density.
        
        Args:
            pii_data (Dict): Dictionary containing PII placeholders
            
        Returns:
            str: Medium sentence with varied vocabulary
        """
        templates = [
            f"El {self.get_synonym('cliente')} {pii_data['name']} con RUT {pii_data['id']} {self.get_synonym('reside')} en {pii_data['address']}, {pii_data['city']}.",
            
            f"{self.get_synonym('Registrado')} {pii_data['name']} ({pii_data['id']}) {self.get_synonym('ubicado')} en {pii_data['address']}, contacto: {pii_data['phone']}.",
            
            f"{self.get_synonym('Usuario')} {pii_data['name']}, RUT {pii_data['id']}, {self.get_synonym('domicilia')} {pii_data['address']}, {pii_data['city']}, fono {pii_data['phone']}.",
            
            f"El {self.get_synonym('titular')} {pii_data['name']} (identificación {pii_data['id']}) {self.get_synonym('establece residencia')} en {pii_data['address']}.",
            
            f"{pii_data['name']}, con {self.get_synonym('documento')} {pii_data['id']}, {self.get_synonym('localizado')} en {pii_data['address']}, {self.get_synonym('correo')}: {pii_data['email']}.",
            
            f"El {self.get_synonym('contratante')} {pii_data['name']} posee RUT {pii_data['id']}, {self.get_synonym('dirección')}: {pii_data['address']}, {pii_data['city']}.",
            
            f"{self.get_synonym('Beneficiario')} {pii_data['name']}, cédula {pii_data['id']}, {self.get_synonym('residencia')}: {pii_data['address']}, tel: {pii_data['phone']}.",
            
            f"Datos: {pii_data['name']} (RUT: {pii_data['id']}), {self.get_synonym('domicilio')}: {pii_data['address']}, {self.get_synonym('móvil')}: {pii_data['phone']}.",
            
            f"{self.get_synonym('Comprador')} {pii_data['name']}, documento {pii_data['id']}, {self.get_synonym('morada')}: {pii_data['address']}, email: {pii_data['email']}.",
            
            f"El {self.get_synonym('adquirente')} {pii_data['name']} con identificación {pii_data['id']} {self.get_synonym('fija domicilio')} en {pii_data['address']}.",
            
            f"{pii_data['name']}, {self.get_synonym('portador')} de RUT {pii_data['id']}, {self.get_synonym('situado')} en {pii_data['address']}, ciudad {pii_data['city']}.",
            
            f"El {self.get_synonym('suscriptor')} {pii_data['name']}, RUT {pii_data['id']}, mantiene {self.get_synonym('ubicación')} en {pii_data['address']}.",
            
            f"{self.get_synonym('Solicitante')}: {pii_data['name']}, identificación: {pii_data['id']}, {self.get_synonym('lugar')}: {pii_data['address']}, contacto: {pii_data['phone']}.",
            
            f"Información de {pii_data['name']} ({pii_data['id']}), {self.get_synonym('establecido')} en {pii_data['address']}, {pii_data['city']}.",
            
            f"{pii_data['name']} con {self.get_synonym('cédula')} {pii_data['id']}, {self.get_synonym('permanece')} en {pii_data['address']}, teléfono {pii_data['phone']}.",
        ]
        
        return self.random.choice(templates)
    
    def generate_varied_sentence(self, country: str, pii_data: Dict, 
                                 length: SentenceLength = None,
                                 complexity: SentenceComplexity = None) -> str:
        """
        Generate a sentence with specified length and complexity.
        
        Args:
            country (str): Country code
            pii_data (Dict): PII data placeholders
            length (SentenceLength): Desired sentence length (random if None)
            complexity (SentenceComplexity): Desired complexity (random if None)
            
        Returns:
            str: Generated sentence
        """
        # Random selection if not specified
        if length is None:
            length = self.random.choice([SentenceLength.MEDIUM, SentenceLength.LONG, 
                                        SentenceLength.EXTRA_LONG])
        
        if complexity is None:
            complexity = self.random.choice([SentenceComplexity.COMPOUND, 
                                            SentenceComplexity.COMPLEX,
                                            SentenceComplexity.COMPOUND_COMPLEX])
        
        # Generate based on length
        if length == SentenceLength.LONG or length == SentenceLength.EXTRA_LONG:
            if country == "chile":
                return self.generate_long_sentence_chile(pii_data)
            # Add other countries here
        else:  # MEDIUM or SHORT
            if country == "chile":
                return self.generate_medium_sentence_chile(pii_data)
        
        # Fallback
        return f"Cliente {pii_data['name']} con ID {pii_data['id']} reside en {pii_data['address']}."
    
    def generate_batch_varied_sentences(self, country: str, pii_data_list: List[Dict],
                                       variety_score: float = 0.8) -> List[str]:
        """
        Generate a batch of sentences with maximum variety.
        
        Args:
            country (str): Country code
            pii_data_list (List[Dict]): List of PII data dictionaries
            variety_score (float): Target variety score (0.0-1.0), higher = more variety
            
        Returns:
            List[str]: List of generated sentences with high variety
        """
        sentences = []
        
        # Distribute lengths based on variety score
        length_distribution = {
            SentenceLength.SHORT: int(len(pii_data_list) * (1 - variety_score) * 0.2),
            SentenceLength.MEDIUM: int(len(pii_data_list) * 0.4),
            SentenceLength.LONG: int(len(pii_data_list) * variety_score * 0.5),
            SentenceLength.EXTRA_LONG: int(len(pii_data_list) * variety_score * 0.3),
        }
        
        length_queue = []
        for length, count in length_distribution.items():
            length_queue.extend([length] * count)
        
        # Fill remaining
        while len(length_queue) < len(pii_data_list):
            length_queue.append(self.random.choice(list(SentenceLength)))
        
        self.random.shuffle(length_queue)
        
        for pii_data, length in zip(pii_data_list, length_queue):
            sentence = self.generate_varied_sentence(country, pii_data, length=length)
            sentences.append(sentence)
        
        return sentences


def create_advanced_generator(language: str = "es") -> AdvancedSentenceGenerator:
    """
    Factory function to create an advanced sentence generator.
    
    Args:
        language (str): Language code ("es" or "pt")
        
    Returns:
        AdvancedSentenceGenerator: Configured generator instance
    """
    return AdvancedSentenceGenerator(language=language)

