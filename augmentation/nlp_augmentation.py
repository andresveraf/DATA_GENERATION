"""
NLP Data Augmentation Module
============================

This module provides advanced NLP-based data augmentation techniques while preserving PII integrity:
- NLTK-based synonym replacement for non-PII words
- Contextual noise injection (typos, OCR errors)
- Text structure variations
- Entity boundary preservation
- Multi-language support (Spanish/Portuguese)

Author: Andrés Vera Figueroa (Enhanced by Codegen)
Date: October 2024
Purpose: Robust NER training data augmentation
"""

import random
import re
import string
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

@dataclass
class EntitySpan:
    """Represents an entity span with boundaries"""
    start: int
    end: int
    label: str
    text: str

@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters"""
    synonym_replacement_rate: float = 0.3  # 30% of non-PII words
    noise_injection_rate: float = 0.1      # 10% character-level noise
    preserve_entities: bool = True          # Always preserve PII entities
    language: str = "es"                    # Spanish by default
    max_synonyms_per_word: int = 3          # Max synonyms to consider
    ocr_error_rate: float = 0.05           # 5% OCR-like errors

class NLPAugmentator:
    """Advanced NLP augmentation with entity preservation"""
    
    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        self.setup_language_resources()
        self.setup_augmentation_patterns()
    
    def setup_language_resources(self):
        """Setup language-specific resources"""
        # Spanish/Portuguese common words that can be safely replaced
        self.REPLACEABLE_WORDS = {
            'es': {
                # Common verbs
                'tiene': ['posee', 'cuenta con', 'dispone de'],
                'está': ['se encuentra', 'se ubica', 'radica'],
                'vive': ['reside', 'habita', 'domicilia'],
                'trabaja': ['labora', 'se desempeña', 'ejerce'],
                'compra': ['adquiere', 'obtiene', 'consigue'],
                'paga': ['abona', 'cancela', 'liquida'],
                'solicita': ['requiere', 'demanda', 'pide'],
                
                # Common nouns
                'cliente': ['usuario', 'consumidor', 'comprador'],
                'persona': ['individuo', 'sujeto', 'ciudadano'],
                'documento': ['papel', 'certificado', 'constancia'],
                'información': ['datos', 'detalles', 'antecedentes'],
                'dirección': ['domicilio', 'residencia', 'ubicación'],
                'teléfono': ['fono', 'contacto', 'número'],
                'empresa': ['compañía', 'firma', 'organización'],
                'servicio': ['atención', 'prestación', 'asistencia'],
                
                # Common adjectives
                'nuevo': ['reciente', 'actual', 'moderno'],
                'viejo': ['antiguo', 'previo', 'anterior'],
                'grande': ['amplio', 'extenso', 'considerable'],
                'pequeño': ['reducido', 'mínimo', 'limitado'],
                'importante': ['relevante', 'significativo', 'crucial'],
                'necesario': ['requerido', 'indispensable', 'preciso'],
                
                # Common prepositions and connectors
                'con': ['junto a', 'mediante', 'a través de'],
                'para': ['hacia', 'destinado a', 'con el fin de'],
                'desde': ['a partir de', 'comenzando en', 'iniciando en'],
                'hasta': ['llegando a', 'culminando en', 'finalizando en']
            },
            'pt': {
                # Portuguese equivalents
                'tem': ['possui', 'conta com', 'dispõe de'],
                'está': ['encontra-se', 'localiza-se', 'situa-se'],
                'mora': ['reside', 'habita', 'domicilia'],
                'trabalha': ['labora', 'atua', 'exerce'],
                'compra': ['adquire', 'obtém', 'consegue'],
                'paga': ['quita', 'liquida', 'salda'],
                'solicita': ['requer', 'demanda', 'pede'],
                
                'cliente': ['usuário', 'consumidor', 'comprador'],
                'pessoa': ['indivíduo', 'sujeito', 'cidadão'],
                'documento': ['papel', 'certificado', 'comprovante'],
                'informação': ['dados', 'detalhes', 'informações'],
                'endereço': ['domicílio', 'residência', 'localização'],
                'telefone': ['fone', 'contato', 'número'],
                'empresa': ['companhia', 'firma', 'organização'],
                'serviço': ['atendimento', 'prestação', 'assistência']
            }
        }
        
        # Load spaCy model for better tokenization
        try:
            if self.config.language == 'es':
                self.nlp = spacy.load('es_core_news_sm')
            elif self.config.language == 'pt':
                self.nlp = spacy.load('pt_core_news_sm')
            else:
                self.nlp = None
        except OSError:
            print(f"Warning: spaCy model for {self.config.language} not found. Using basic tokenization.")
            self.nlp = None
    
    def setup_augmentation_patterns(self):
        """Setup patterns for different types of augmentation"""
        # OCR-like character substitutions
        self.OCR_SUBSTITUTIONS = {
            'o': ['0', 'ó', 'ò'],
            '0': ['o', 'O'],
            'i': ['1', 'l', 'í', 'ì'],
            '1': ['i', 'l', 'I'],
            'l': ['1', 'i', 'I'],
            'e': ['é', 'è', 'ê'],
            'a': ['á', 'à', 'â'],
            'u': ['ú', 'ù', 'û'],
            'n': ['ñ', 'm'],
            'c': ['ç'],
            'rn': ['m'],
            'cl': ['d'],
            'nn': ['m'],
            'vv': ['w'],
            'ii': ['u']
        }
        
        # Typing errors (adjacent keys)
        self.TYPING_ERRORS = {
            'a': ['s', 'q', 'w'],
            's': ['a', 'd', 'w', 'e'],
            'd': ['s', 'f', 'e', 'r'],
            'f': ['d', 'g', 'r', 't'],
            'g': ['f', 'h', 't', 'y'],
            'h': ['g', 'j', 'y', 'u'],
            'j': ['h', 'k', 'u', 'i'],
            'k': ['j', 'l', 'i', 'o'],
            'l': ['k', 'o', 'p'],
            'q': ['w', 'a'],
            'w': ['q', 'e', 'a', 's'],
            'e': ['w', 'r', 's', 'd'],
            'r': ['e', 't', 'd', 'f'],
            't': ['r', 'y', 'f', 'g'],
            'y': ['t', 'u', 'g', 'h'],
            'u': ['y', 'i', 'h', 'j'],
            'i': ['u', 'o', 'j', 'k'],
            'o': ['i', 'p', 'k', 'l'],
            'p': ['o', 'l']
        }
    
    def get_entity_spans(self, text: str, entities: List[Tuple[int, int, str]]) -> List[EntitySpan]:
        """Convert entity tuples to EntitySpan objects"""
        return [EntitySpan(start, end, label, text[start:end]) for start, end, label in entities]
    
    def is_word_in_entity(self, word_start: int, word_end: int, entity_spans: List[EntitySpan]) -> bool:
        """Check if a word overlaps with any entity"""
        for entity in entity_spans:
            if not (word_end <= entity.start or word_start >= entity.end):
                return True
        return False
    
    def get_synonyms_nltk(self, word: str, pos: str = None) -> List[str]:
        """Get synonyms using NLTK WordNet"""
        synonyms = set()
        
        # Map POS tags to WordNet POS
        pos_map = {
            'NN': wordnet.NOUN, 'NNS': wordnet.NOUN, 'NNP': wordnet.NOUN, 'NNPS': wordnet.NOUN,
            'VB': wordnet.VERB, 'VBD': wordnet.VERB, 'VBG': wordnet.VERB, 'VBN': wordnet.VERB,
            'VBP': wordnet.VERB, 'VBZ': wordnet.VERB,
            'JJ': wordnet.ADJ, 'JJR': wordnet.ADJ, 'JJS': wordnet.ADJ,
            'RB': wordnet.ADV, 'RBR': wordnet.ADV, 'RBS': wordnet.ADV
        }
        
        wn_pos = pos_map.get(pos, wordnet.NOUN) if pos else wordnet.NOUN
        
        for syn in wordnet.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower() and len(synonym) > 2:
                    synonyms.add(synonym)
        
        return list(synonyms)[:self.config.max_synonyms_per_word]
    
    def get_synonyms_custom(self, word: str) -> List[str]:
        """Get synonyms from custom dictionary"""
        lang_dict = self.REPLACEABLE_WORDS.get(self.config.language, {})
        return lang_dict.get(word.lower(), [])
    
    def apply_synonym_replacement(self, text: str, entities: List[Tuple[int, int, str]]) -> str:
        """Apply synonym replacement while preserving entities"""
        if not self.config.preserve_entities:
            return text
        
        entity_spans = self.get_entity_spans(text, entities)
        
        # Tokenize text
        if self.nlp:
            doc = self.nlp(text)
            tokens = [(token.text, token.idx, token.idx + len(token.text), token.pos_) for token in doc]
        else:
            # Fallback to NLTK tokenization
            words = word_tokenize(text)
            pos_tags = pos_tag(words)
            tokens = []
            current_pos = 0
            for (word, pos) in pos_tags:
                start = text.find(word, current_pos)
                if start != -1:
                    end = start + len(word)
                    tokens.append((word, start, end, pos))
                    current_pos = end
        
        # Apply replacements
        replacements = []
        for word, start, end, pos in tokens:
            # Skip if word is part of an entity
            if self.is_word_in_entity(start, end, entity_spans):
                continue
            
            # Skip if word is too short or is punctuation
            if len(word) < 3 or word in string.punctuation:
                continue
            
            # Apply replacement with probability
            if random.random() < self.config.synonym_replacement_rate:
                # Try custom synonyms first
                synonyms = self.get_synonyms_custom(word)
                
                # If no custom synonyms, try NLTK
                if not synonyms:
                    synonyms = self.get_synonyms_nltk(word, pos)
                
                if synonyms:
                    replacement = random.choice(synonyms)
                    replacements.append((start, end, replacement))
        
        # Apply replacements from right to left to maintain positions
        replacements.sort(key=lambda x: x[0], reverse=True)
        result_text = text
        for start, end, replacement in replacements:
            result_text = result_text[:start] + replacement + result_text[end:]
        
        return result_text
    
    def apply_character_noise(self, text: str, entities: List[Tuple[int, int, str]]) -> str:
        """Apply character-level noise while preserving entities"""
        if not self.config.preserve_entities:
            return text
        
        entity_spans = self.get_entity_spans(text, entities)
        result_chars = list(text)
        
        for i, char in enumerate(result_chars):
            # Skip if character is part of an entity
            if any(entity.start <= i < entity.end for entity in entity_spans):
                continue
            
            # Skip if character is whitespace or punctuation
            if char.isspace() or char in string.punctuation:
                continue
            
            # Apply noise with probability
            if random.random() < self.config.noise_injection_rate:
                noise_type = random.choice(['ocr', 'typing', 'deletion', 'insertion'])
                
                if noise_type == 'ocr' and char.lower() in self.OCR_SUBSTITUTIONS:
                    result_chars[i] = random.choice(self.OCR_SUBSTITUTIONS[char.lower()])
                
                elif noise_type == 'typing' and char.lower() in self.TYPING_ERRORS:
                    result_chars[i] = random.choice(self.TYPING_ERRORS[char.lower()])
                
                elif noise_type == 'deletion':
                    result_chars[i] = ''  # Delete character
                
                elif noise_type == 'insertion':
                    # Insert random character
                    random_char = random.choice(string.ascii_lowercase)
                    result_chars[i] = char + random_char
        
        return ''.join(result_chars)
    
    def apply_ocr_errors(self, text: str, entities: List[Tuple[int, int, str]]) -> str:
        """Apply OCR-specific errors while preserving entities"""
        entity_spans = self.get_entity_spans(text, entities)
        result_text = text
        
        # Common OCR patterns
        ocr_patterns = [
            (r'\brn\b', 'm'),      # 'rn' -> 'm'
            (r'\bcl\b', 'd'),      # 'cl' -> 'd'
            (r'\bnn\b', 'm'),      # 'nn' -> 'm'
            (r'\bvv\b', 'w'),      # 'vv' -> 'w'
            (r'\bii\b', 'u'),      # 'ii' -> 'u'
        ]
        
        for pattern, replacement in ocr_patterns:
            if random.random() < self.config.ocr_error_rate:
                # Find matches that don't overlap with entities
                for match in re.finditer(pattern, result_text, re.IGNORECASE):
                    start, end = match.span()
                    if not any(entity.start <= start < entity.end or entity.start < end <= entity.end 
                              for entity in entity_spans):
                        result_text = result_text[:start] + replacement + result_text[end:]
        
        return result_text
    
    def augment_text(self, text: str, entities: List[Tuple[int, int, str]], 
                    augmentation_types: List[str] = None) -> Tuple[str, List[Tuple[int, int, str]]]:
        """
        Apply comprehensive text augmentation while preserving entity boundaries
        
        Args:
            text: Original text
            entities: List of (start, end, label) tuples
            augmentation_types: Types of augmentation to apply
            
        Returns:
            Tuple of (augmented_text, updated_entities)
        """
        if augmentation_types is None:
            augmentation_types = ['synonyms', 'noise', 'ocr']
        
        augmented_text = text
        
        # Apply augmentations in sequence
        if 'synonyms' in augmentation_types:
            augmented_text = self.apply_synonym_replacement(augmented_text, entities)
        
        if 'noise' in augmentation_types:
            augmented_text = self.apply_character_noise(augmented_text, entities)
        
        if 'ocr' in augmentation_types:
            augmented_text = self.apply_ocr_errors(augmented_text, entities)
        
        # Update entity positions if text length changed
        updated_entities = self.update_entity_positions(text, augmented_text, entities)
        
        return augmented_text, updated_entities
    
    def update_entity_positions(self, original_text: str, augmented_text: str, 
                               entities: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """Update entity positions after text augmentation"""
        # For now, return original entities (assumes entity text is preserved)
        # In a more sophisticated implementation, we would track character-level changes
        updated_entities = []
        
        for start, end, label in entities:
            entity_text = original_text[start:end]
            
            # Find the entity in the augmented text
            new_start = augmented_text.find(entity_text)
            if new_start != -1:
                new_end = new_start + len(entity_text)
                updated_entities.append((new_start, new_end, label))
            else:
                # If exact match not found, keep original positions as fallback
                updated_entities.append((start, end, label))
        
        return updated_entities
    
    def generate_augmented_variants(self, text: str, entities: List[Tuple[int, int, str]], 
                                   num_variants: int = 3) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
        """Generate multiple augmented variants of the same text"""
        variants = []
        
        for i in range(num_variants):
            # Vary augmentation intensity for each variant
            original_synonym_rate = self.config.synonym_replacement_rate
            original_noise_rate = self.config.noise_injection_rate
            
            # Create different intensity levels
            if i == 0:  # Light augmentation
                self.config.synonym_replacement_rate *= 0.5
                self.config.noise_injection_rate *= 0.5
            elif i == 1:  # Medium augmentation (original)
                pass  # Keep original rates
            else:  # Heavy augmentation
                self.config.synonym_replacement_rate *= 1.5
                self.config.noise_injection_rate *= 1.5
            
            # Generate variant
            augmented_text, updated_entities = self.augment_text(text, entities)
            variants.append((augmented_text, updated_entities))
            
            # Restore original rates
            self.config.synonym_replacement_rate = original_synonym_rate
            self.config.noise_injection_rate = original_noise_rate
        
        return variants

# Convenience functions
def create_augmentator(language: str = "es", synonym_rate: float = 0.3, 
                      noise_rate: float = 0.1) -> NLPAugmentator:
    """Create an NLP augmentator with specified parameters"""
    config = AugmentationConfig(
        synonym_replacement_rate=synonym_rate,
        noise_injection_rate=noise_rate,
        language=language
    )
    return NLPAugmentator(config)

def augment_training_example(text: str, entities: List[Tuple[int, int, str]], 
                           language: str = "es") -> Tuple[str, List[Tuple[int, int, str]]]:
    """Quick function to augment a single training example"""
    augmentator = create_augmentator(language)
    return augmentator.augment_text(text, entities)
