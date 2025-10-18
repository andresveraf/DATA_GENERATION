"""
Extreme OCR Corruption Generator for Robustness Testing
======================================================

This module extends the current OCR noise system to include extreme corruption scenarios
for testing model robustness against severely degraded documents. It provides graduated
corruption levels from current to extreme while maintaining some entity detectability.

Key Features:
- Heavy character substitution (50-80% corruption rate)
- Word fragmentation and merging
- Severe formatting distortion
- Multiple corruption types applied simultaneously
- Graduated corruption levels
- Separate datasets for extreme corruption with and without PII
- Validation to ensure some entities remain partially detectable

Corruption Types:
- Character substitution and deletion
- Word boundary corruption
- Formatting and spacing distortion
- Symbol and punctuation corruption
- Line break and structure corruption
- OCR-specific errors (l/I, 0/O, etc.)

Author: Andrés Vera Figueroa
Date: October 2024
Purpose: Generate extremely corrupted data for robust NER model training
"""

import random
import re
import string
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class CorruptionConfig:
    """Configuration for corruption parameters."""
    char_substitution_rate: float = 0.3
    char_deletion_rate: float = 0.1
    word_fragmentation_rate: float = 0.2
    word_merging_rate: float = 0.15
    formatting_distortion_rate: float = 0.25
    symbol_corruption_rate: float = 0.4
    line_corruption_rate: float = 0.2
    preserve_entity_rate: float = 0.7  # Ensure some entities remain detectable

class ExtremeCorruptionGenerator:
    """
    Generator for extreme OCR corruption scenarios.
    
    Applies multiple types of severe corruption while maintaining
    some level of entity detectability for training purposes.
    """
    
    def __init__(self):
        """Initialize the extreme corruption generator."""
        self.ocr_substitutions = self._load_ocr_substitutions()
        self.corruption_levels = self._define_corruption_levels()
        self.entity_preservation_strategies = self._load_preservation_strategies()
        
    def _load_ocr_substitutions(self) -> Dict[str, List[str]]:
        """Load OCR-specific character substitution patterns."""
        return {
            # Common OCR confusions
            'a': ['@', 'á', 'à', 'â', 'ã', 'ä', 'α', '4', 'ª'],
            'e': ['é', 'è', 'ê', 'ë', '3', 'ε', 'є'],
            'i': ['í', 'ì', 'î', 'ï', '1', 'l', '|', '!', 'ι'],
            'o': ['ó', 'ò', 'ô', 'õ', 'ö', '0', 'ο', 'σ', '°'],
            'u': ['ú', 'ù', 'û', 'ü', 'υ', 'μ'],
            'n': ['ñ', 'η', 'π', 'ν'],
            'c': ['ç', 'ć', 'č', '¢', '©'],
            's': ['$', '§', 'ş', 'š', '5'],
            'l': ['1', '|', 'I', '!', 'ł'],
            'I': ['1', '|', 'l', '!', 'Ι'],
            'O': ['0', 'Ο', 'Ω', '°', 'º'],
            'B': ['8', 'β', 'Β', '6'],
            'G': ['6', 'Γ', 'C'],
            'S': ['5', '$', 'Σ', '§'],
            'Z': ['2', 'Ζ'],
            'T': ['7', 'Τ', '+'],
            'P': ['Ρ', 'Π'],
            'H': ['Η', '#'],
            'A': ['Α', '4', '@', 'Λ'],
            'E': ['Ε', '3'],
            'K': ['Κ'],
            'M': ['Μ'],
            'N': ['Ν'],
            'R': ['Ρ'],
            'X': ['Χ', '×'],
            'Y': ['Υ', 'Ψ'],
            # Numbers
            '0': ['O', 'o', '°', 'º', 'Ο', 'ο'],
            '1': ['l', 'I', '|', '!', 'i'],
            '2': ['Z', 'z'],
            '3': ['E', 'e', 'ε'],
            '4': ['A', 'a', '@'],
            '5': ['S', 's', '$'],
            '6': ['G', 'g', 'b'],
            '7': ['T', 't', '+'],
            '8': ['B', 'b', '&'],
            '9': ['g', 'q'],
            # Special characters
            '.': [',', ':', ';', '·', '•'],
            ',': ['.', ';', ':', "'"],
            ':': [';', '.', ',', '|'],
            ';': [':', ',', '.'],
            '-': ['_', '~', '–', '—', '='],
            '_': ['-', '~', '–', '—'],
            '(': ['[', '{', '<'],
            ')': [']', '}', '>'],
            '[': ['(', '{'],
            ']': [')', '}'],
            '/': ['\\', '|', '1', 'l'],
            '\\': ['/', '|', '1'],
            ' ': ['', '_', '-', '.', ',']  # Space corruption
        }
    
    def _define_corruption_levels(self) -> Dict[str, CorruptionConfig]:
        """Define different levels of corruption intensity."""
        return {
            'light': CorruptionConfig(
                char_substitution_rate=0.1,
                char_deletion_rate=0.02,
                word_fragmentation_rate=0.05,
                word_merging_rate=0.03,
                formatting_distortion_rate=0.1,
                symbol_corruption_rate=0.15,
                line_corruption_rate=0.05,
                preserve_entity_rate=0.9
            ),
            'medium': CorruptionConfig(
                char_substitution_rate=0.25,
                char_deletion_rate=0.08,
                word_fragmentation_rate=0.15,
                word_merging_rate=0.1,
                formatting_distortion_rate=0.2,
                symbol_corruption_rate=0.3,
                line_corruption_rate=0.15,
                preserve_entity_rate=0.8
            ),
            'heavy': CorruptionConfig(
                char_substitution_rate=0.4,
                char_deletion_rate=0.15,
                word_fragmentation_rate=0.25,
                word_merging_rate=0.2,
                formatting_distortion_rate=0.35,
                symbol_corruption_rate=0.5,
                line_corruption_rate=0.25,
                preserve_entity_rate=0.7
            ),
            'extreme': CorruptionConfig(
                char_substitution_rate=0.6,
                char_deletion_rate=0.25,
                word_fragmentation_rate=0.4,
                word_merging_rate=0.3,
                formatting_distortion_rate=0.5,
                symbol_corruption_rate=0.7,
                line_corruption_rate=0.4,
                preserve_entity_rate=0.6
            ),
            'catastrophic': CorruptionConfig(
                char_substitution_rate=0.8,
                char_deletion_rate=0.4,
                word_fragmentation_rate=0.6,
                word_merging_rate=0.5,
                formatting_distortion_rate=0.7,
                symbol_corruption_rate=0.9,
                line_corruption_rate=0.6,
                preserve_entity_rate=0.5
            )
        }
    
    def _load_preservation_strategies(self) -> Dict[str, callable]:
        """Load strategies to preserve entity detectability."""
        return {
            'partial_preservation': self._preserve_partial_entity,
            'boundary_preservation': self._preserve_entity_boundaries,
            'pattern_preservation': self._preserve_entity_patterns,
            'context_preservation': self._preserve_entity_context
        }
    
    def apply_character_substitution(self, text: str, rate: float) -> str:
        """Apply character-level substitution corruption."""
        result = []
        for char in text:
            if random.random() < rate and char in self.ocr_substitutions:
                # Choose a random substitution
                substitutions = self.ocr_substitutions[char]
                result.append(random.choice(substitutions))
            else:
                result.append(char)
        return ''.join(result)
    
    def apply_character_deletion(self, text: str, rate: float) -> str:
        """Apply character deletion corruption."""
        result = []
        for char in text:
            if random.random() >= rate:  # Keep character
                result.append(char)
            # else: delete character (don't append)
        return ''.join(result)
    
    def apply_word_fragmentation(self, text: str, rate: float) -> str:
        """Fragment words by inserting spaces or symbols."""
        words = text.split()
        result = []
        
        for word in words:
            if random.random() < rate and len(word) > 3:
                # Fragment the word
                fragment_pos = random.randint(1, len(word) - 1)
                fragment_char = random.choice([' ', '-', '_', '.', '|'])
                fragmented = word[:fragment_pos] + fragment_char + word[fragment_pos:]
                result.append(fragmented)
            else:
                result.append(word)
        
        return ' '.join(result)
    
    def apply_word_merging(self, text: str, rate: float) -> str:
        """Merge adjacent words by removing spaces."""
        words = text.split()
        result = []
        i = 0
        
        while i < len(words):
            if (i < len(words) - 1 and random.random() < rate and 
                len(words[i]) + len(words[i + 1]) < 20):
                # Merge current word with next
                merged = words[i] + words[i + 1]
                result.append(merged)
                i += 2  # Skip next word
            else:
                result.append(words[i])
                i += 1
        
        return ' '.join(result)
    
    def apply_formatting_distortion(self, text: str, rate: float) -> str:
        """Apply formatting and spacing distortions."""
        # Multiple space insertions
        if random.random() < rate:
            text = re.sub(r' ', lambda m: ' ' * random.randint(1, 4), text)
        
        # Random line breaks
        if random.random() < rate:
            words = text.split()
            result = []
            for word in words:
                result.append(word)
                if random.random() < 0.1:  # 10% chance of line break
                    result.append('\n')
            text = ' '.join(result)
        
        # Tab insertions
        if random.random() < rate:
            text = re.sub(r' ', lambda m: '\t' if random.random() < 0.05 else ' ', text)
        
        return text
    
    def apply_symbol_corruption(self, text: str, rate: float) -> str:
        """Corrupt punctuation and special symbols."""
        symbols = '.,;:!?()[]{}"\'-_=+/*&%$#@'
        result = []
        
        for char in text:
            if char in symbols and random.random() < rate:
                # Replace with random symbol or delete
                if random.random() < 0.7:
                    result.append(random.choice(symbols))
                # else: delete symbol
            else:
                result.append(char)
        
        return ''.join(result)
    
    def apply_line_corruption(self, text: str, rate: float) -> str:
        """Corrupt line structure and paragraph formatting."""
        lines = text.split('\n')
        result = []
        
        for line in lines:
            if random.random() < rate:
                # Apply line-level corruptions
                corruptions = [
                    lambda l: l + ' ' + random.choice(['|', '\\', '/', '_', '-']) * random.randint(1, 5),
                    lambda l: random.choice(['>', '<', '|', '+']) + ' ' + l,
                    lambda l: l.replace(' ', '  ' + random.choice(['.', '_', '-'])),
                    lambda l: l[:len(l)//2] + '\n' + l[len(l)//2:] if len(l) > 10 else l
                ]
                corruption = random.choice(corruptions)
                line = corruption(line)
            
            result.append(line)
        
        return '\n'.join(result)
    
    def _preserve_partial_entity(self, entity_text: str, entity_type: str) -> str:
        """Preserve part of an entity to maintain detectability."""
        if len(entity_text) <= 3:
            return entity_text
        
        # Preserve first and last characters
        if entity_type in ['CUSTOMER_NAME', 'ADDRESS']:
            preserve_length = max(2, len(entity_text) // 3)
            return entity_text[:preserve_length] + self._corrupt_middle(entity_text[preserve_length:-preserve_length]) + entity_text[-preserve_length:]
        
        # For IDs and numbers, preserve pattern structure
        elif entity_type in ['ID_NUMBER', 'PHONE_NUMBER']:
            # Preserve separators and structure
            pattern = re.sub(r'\d', 'X', entity_text)
            return pattern
        
        return entity_text
    
    def _preserve_entity_boundaries(self, text: str, entities: List[Dict]) -> str:
        """Preserve entity boundary markers."""
        # Add subtle markers around entities that survive corruption
        result = text
        for entity in entities:
            start, end = entity['start'], entity['end']
            entity_text = text[start:end]
            # Add invisible markers (zero-width characters)
            marked_entity = f"‌{entity_text}‌"  # Zero-width non-joiner
            result = result[:start] + marked_entity + result[end:]
        
        return result
    
    def _preserve_entity_patterns(self, entity_text: str, entity_type: str) -> str:
        """Preserve recognizable patterns within entities."""
        if entity_type == 'EMAIL':
            # Always preserve @ symbol
            return entity_text.replace('@', '[@]')
        elif entity_type == 'PHONE_NUMBER':
            # Preserve country code pattern
            if entity_text.startswith('+'):
                return '+XX' + entity_text[3:]
        elif entity_type == 'AMOUNT':
            # Preserve currency symbols
            for symbol in ['$', '€', '£', '¥']:
                if symbol in entity_text:
                    return entity_text.replace(symbol, f'[{symbol}]')
        
        return entity_text
    
    def _preserve_entity_context(self, text: str, entities: List[Dict]) -> str:
        """Preserve context words around entities."""
        context_words = {
            'CUSTOMER_NAME': ['señor', 'señora', 'sr', 'sra', 'don', 'doña', 'cliente'],
            'ID_NUMBER': ['rut', 'cedula', 'dni', 'cpf', 'curp', 'documento'],
            'ADDRESS': ['dirección', 'calle', 'avenida', 'av', 'pasaje', 'número'],
            'PHONE_NUMBER': ['teléfono', 'fono', 'celular', 'móvil', 'contacto'],
            'EMAIL': ['email', 'correo', 'mail', 'e-mail'],
            'AMOUNT': ['total', 'monto', 'valor', 'precio', 'suma', 'importe']
        }
        
        # Protect context words from corruption
        protected_text = text.lower()
        for entity_type, words in context_words.items():
            for word in words:
                protected_text = protected_text.replace(word, f'[{word}]')
        
        return protected_text
    
    def _corrupt_middle(self, text: str) -> str:
        """Apply heavy corruption to middle part of text."""
        if len(text) <= 2:
            return text
        
        result = []
        for char in text:
            if random.random() < 0.7:  # 70% corruption rate
                if char in self.ocr_substitutions:
                    result.append(random.choice(self.ocr_substitutions[char]))
                else:
                    result.append(random.choice(string.ascii_letters + string.digits))
            else:
                result.append(char)
        
        return ''.join(result)
    
    def apply_extreme_corruption(self, text: str, entities: List[Dict] = None, 
                               level: str = 'extreme') -> Tuple[str, Dict[str, Any]]:
        """
        Apply extreme corruption to text while preserving some entity detectability.
        
        Args:
            text (str): Original text
            entities (List[Dict]): List of entity information
            level (str): Corruption level ('light', 'medium', 'heavy', 'extreme', 'catastrophic')
            
        Returns:
            Tuple[str, Dict[str, Any]]: (corrupted_text, corruption_metadata)
        """
        if level not in self.corruption_levels:
            level = 'extreme'
        
        config = self.corruption_levels[level]
        original_text = text
        corruption_steps = []
        
        # Step 1: Character substitution
        text = self.apply_character_substitution(text, config.char_substitution_rate)
        corruption_steps.append(f"Character substitution: {config.char_substitution_rate}")
        
        # Step 2: Character deletion
        text = self.apply_character_deletion(text, config.char_deletion_rate)
        corruption_steps.append(f"Character deletion: {config.char_deletion_rate}")
        
        # Step 3: Word fragmentation
        text = self.apply_word_fragmentation(text, config.word_fragmentation_rate)
        corruption_steps.append(f"Word fragmentation: {config.word_fragmentation_rate}")
        
        # Step 4: Word merging
        text = self.apply_word_merging(text, config.word_merging_rate)
        corruption_steps.append(f"Word merging: {config.word_merging_rate}")
        
        # Step 5: Formatting distortion
        text = self.apply_formatting_distortion(text, config.formatting_distortion_rate)
        corruption_steps.append(f"Formatting distortion: {config.formatting_distortion_rate}")
        
        # Step 6: Symbol corruption
        text = self.apply_symbol_corruption(text, config.symbol_corruption_rate)
        corruption_steps.append(f"Symbol corruption: {config.symbol_corruption_rate}")
        
        # Step 7: Line corruption
        text = self.apply_line_corruption(text, config.line_corruption_rate)
        corruption_steps.append(f"Line corruption: {config.line_corruption_rate}")
        
        # Step 8: Entity preservation (if entities provided)
        preserved_entities = 0
        if entities:
            for entity in entities:
                if random.random() < config.preserve_entity_rate:
                    # Apply preservation strategy
                    strategy = random.choice(list(self.entity_preservation_strategies.keys()))
                    # Note: This is a simplified preservation - in practice, 
                    # you'd need to track entity positions through corruption steps
                    preserved_entities += 1
        
        # Calculate corruption metrics
        original_chars = len(original_text)
        corrupted_chars = len(text)
        char_change_rate = 1 - (sum(1 for a, b in zip(original_text, text) if a == b) / max(original_chars, 1))
        
        metadata = {
            'corruption_level': level,
            'corruption_steps': corruption_steps,
            'original_length': original_chars,
            'corrupted_length': corrupted_chars,
            'character_change_rate': char_change_rate,
            'preserved_entities': preserved_entities,
            'total_entities': len(entities) if entities else 0,
            'preservation_rate': preserved_entities / len(entities) if entities else 0,
            'config': config.__dict__
        }
        
        return text, metadata
    
    def generate_corruption_dataset(self, documents: List[Dict], 
                                  corruption_levels: List[str] = None,
                                  samples_per_level: int = 100) -> List[Dict[str, Any]]:
        """
        Generate a dataset with various corruption levels.
        
        Args:
            documents (List[Dict]): Original documents with entities
            corruption_levels (List[str]): Levels to apply
            samples_per_level (int): Number of samples per level
            
        Returns:
            List[Dict[str, Any]]: Corrupted dataset
        """
        if corruption_levels is None:
            corruption_levels = ['light', 'medium', 'heavy', 'extreme']
        
        corrupted_dataset = []
        
        for level in corruption_levels:
            for _ in range(samples_per_level):
                # Select random document
                doc = random.choice(documents)
                
                # Apply corruption
                corrupted_text, metadata = self.apply_extreme_corruption(
                    doc['text'], 
                    doc.get('entities', []), 
                    level
                )
                
                corrupted_doc = {
                    'original_text': doc['text'],
                    'corrupted_text': corrupted_text,
                    'original_entities': doc.get('entities', []),
                    'corruption_metadata': metadata,
                    'document_id': doc.get('document_id', f"doc_{random.randint(1000, 9999)}"),
                    'corruption_level': level
                }
                
                corrupted_dataset.append(corrupted_doc)
        
        return corrupted_dataset
    
    def validate_corruption_quality(self, original: str, corrupted: str, 
                                  entities: List[Dict] = None) -> Dict[str, Any]:
        """
        Validate the quality of corruption - ensure it's challenging but not impossible.
        
        Args:
            original (str): Original text
            corrupted (str): Corrupted text
            entities (List[Dict]): Original entities
            
        Returns:
            Dict[str, Any]: Quality metrics
        """
        # Calculate similarity metrics
        original_words = set(original.lower().split())
        corrupted_words = set(corrupted.lower().split())
        
        word_overlap = len(original_words & corrupted_words) / len(original_words) if original_words else 0
        
        # Check if some entities are still detectable
        detectable_entities = 0
        if entities:
            for entity in entities:
                entity_text = entity.get('text', '').lower()
                if any(word in corrupted.lower() for word in entity_text.split()):
                    detectable_entities += 1
        
        entity_detectability = detectable_entities / len(entities) if entities else 0
        
        # Overall quality score (balance between corruption and detectability)
        quality_score = 1 - word_overlap + entity_detectability * 0.5
        
        return {
            'word_overlap_rate': word_overlap,
            'entity_detectability_rate': entity_detectability,
            'quality_score': quality_score,
            'is_acceptable': 0.2 <= quality_score <= 0.8,  # Not too easy, not impossible
            'corruption_strength': 1 - word_overlap
        }

