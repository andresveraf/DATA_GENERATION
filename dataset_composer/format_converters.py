"""
Format Converters for NER Data Export
====================================

This module provides converter classes to transform internal JSON format data
into specialized formats required by different NER training frameworks:

1. SpacyFormatConverter: Converts to spaCy DocBin (.spacy) binary format
2. TransformersFormatConverter: Converts to CONLL/BIO tagging format

Both converters handle:
- Entity span alignment and validation
- Tokenization differences between formats
- Edge cases like overlapping entities
- Train/dev split preservation
- Error handling and logging

Author: Andrés Vera Figueroa
Date: October 2024
Purpose: Enable dual-format export for comprehensive NER training
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class SpacyFormatConverter:
    """
    Converts internal JSON format to spaCy DocBin (.spacy) binary format.
    
    This converter creates spaCy Doc objects with entity spans and serializes
    them using DocBin for efficient storage and loading during training.
    """
    
    def __init__(self, language: str = 'es'):
        """
        Initialize the spaCy converter.
        
        Args:
            language (str): Language code for spaCy model (default: 'es')
        """
        self.language = language
        self.nlp = None
        self._initialize_spacy()
    
    def _initialize_spacy(self):
        """Initialize spaCy language model."""
        try:
            import spacy
            from spacy.tokens import DocBin
            
            # Create blank language model
            self.nlp = spacy.blank(self.language)
            self.DocBin = DocBin
            logger.info(f"Initialized spaCy converter for language: {self.language}")
            
        except ImportError:
            logger.error("spaCy not installed. Please install with: pip install spacy")
            raise ImportError("spaCy is required for spaCy format conversion")
        except Exception as e:
            logger.error(f"Failed to initialize spaCy: {e}")
            raise
    
    def convert_documents_to_docbin(self, documents: List[Dict[str, Any]]) -> 'DocBin':
        """
        Convert list of documents to spaCy DocBin.
        
        Args:
            documents (List[Dict[str, Any]]): List of document dictionaries
            
        Returns:
            DocBin: spaCy DocBin object containing all documents
        """
        if not self.nlp:
            raise RuntimeError("spaCy not properly initialized")
        
        doc_bin = self.DocBin(attrs=["ORTH", "TAG", "HEAD", "DEP", "ENT_IOB", "ENT_TYPE"])
        
        for doc_data in documents:
            try:
                doc = self._create_spacy_doc(doc_data)
                if doc:
                    doc_bin.add(doc)
            except Exception as e:
                logger.warning(f"Failed to convert document {doc_data.get('document_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Converted {len(doc_bin)} documents to DocBin")
        return doc_bin
    
    def _create_spacy_doc(self, doc_data: Dict[str, Any]) -> Optional['Doc']:
        """
        Create a spaCy Doc object from document data.
        
        Args:
            doc_data (Dict[str, Any]): Document data with text and entities
            
        Returns:
            Optional[Doc]: spaCy Doc object or None if conversion fails
        """
        text = doc_data.get('text', '')
        entities = doc_data.get('entities', [])
        
        if not text:
            logger.warning(f"Empty text in document {doc_data.get('document_id', 'unknown')}")
            return None
        
        # Create Doc object
        doc = self.nlp(text)
        
        # Prepare entity spans
        entity_spans = []
        for entity in entities:
            try:
                start = entity.get('start', 0)
                end = entity.get('end', 0)
                label = entity.get('label', 'UNKNOWN')
                
                # Validate span boundaries
                if start < 0 or end > len(text) or start >= end:
                    logger.warning(f"Invalid entity span: {start}-{end} in text of length {len(text)}")
                    continue
                
                # Validate entity text matches
                entity_text = text[start:end]
                expected_text = entity.get('text', '')
                if expected_text and entity_text != expected_text:
                    logger.warning(f"Entity text mismatch: expected '{expected_text}', got '{entity_text}'")
                
                # Find character span in Doc
                char_span = doc.char_span(start, end, label=label, alignment_mode="expand")
                if char_span:
                    entity_spans.append(char_span)
                else:
                    logger.warning(f"Could not create span for entity: {start}-{end} '{entity_text}'")
                    
            except Exception as e:
                logger.warning(f"Error processing entity {entity}: {e}")
                continue
        
        # Set entities on Doc
        try:
            doc.ents = entity_spans
        except ValueError as e:
            logger.warning(f"Could not set entities on doc: {e}")
            # Try to resolve overlapping entities
            doc.ents = self._resolve_overlapping_entities(entity_spans)
        
        return doc
    
    def _resolve_overlapping_entities(self, spans: List['Span']) -> List['Span']:
        """
        Resolve overlapping entity spans by keeping the longest ones.
        
        Args:
            spans (List[Span]): List of potentially overlapping spans
            
        Returns:
            List[Span]: Non-overlapping spans
        """
        if not spans:
            return []
        
        # Sort by start position, then by length (descending)
        sorted_spans = sorted(spans, key=lambda x: (x.start, -(x.end - x.start)))
        
        non_overlapping = []
        for span in sorted_spans:
            # Check if this span overlaps with any already accepted span
            overlaps = False
            for accepted_span in non_overlapping:
                if (span.start < accepted_span.end and span.end > accepted_span.start):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(span)
        
        logger.info(f"Resolved {len(spans)} spans to {len(non_overlapping)} non-overlapping spans")
        return non_overlapping
    
    def save_docbin(self, doc_bin: 'DocBin', output_path: str) -> bool:
        """
        Save DocBin to file.
        
        Args:
            doc_bin (DocBin): DocBin object to save
            output_path (str): Path to save the .spacy file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            doc_bin.to_disk(output_path)
            
            file_size = output_path.stat().st_size
            logger.info(f"Saved spaCy DocBin to {output_path} ({file_size:,} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save DocBin to {output_path}: {e}")
            return False
    
    def convert_and_save(self, documents: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Convert documents and save directly to file.
        
        Args:
            documents (List[Dict[str, Any]]): Documents to convert
            output_path (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            doc_bin = self.convert_documents_to_docbin(documents)
            return self.save_docbin(doc_bin, output_path)
        except Exception as e:
            logger.error(f"Failed to convert and save documents: {e}")
            return False


class TransformersFormatConverter:
    """
    Converts internal JSON format to CONLL/BIO tagging format.
    
    This converter tokenizes text and applies BIO tagging scheme for
    compatibility with transformer-based NER models.
    """
    
    def __init__(self, tokenization_method: str = 'whitespace'):
        """
        Initialize the transformers converter.
        
        Args:
            tokenization_method (str): Tokenization method ('whitespace', 'simple')
        """
        self.tokenization_method = tokenization_method
        logger.info(f"Initialized Transformers converter with {tokenization_method} tokenization")
    
    def convert_documents_to_conll(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Convert documents to CONLL format lines.
        
        Args:
            documents (List[Dict[str, Any]]): Documents to convert
            
        Returns:
            List[str]: CONLL format lines
        """
        conll_lines = []
        
        for doc_data in documents:
            try:
                doc_lines = self._convert_single_document(doc_data)
                conll_lines.extend(doc_lines)
                conll_lines.append("")  # Blank line between documents
            except Exception as e:
                logger.warning(f"Failed to convert document {doc_data.get('document_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Converted {len(documents)} documents to CONLL format")
        return conll_lines
    
    def _convert_single_document(self, doc_data: Dict[str, Any]) -> List[str]:
        """
        Convert a single document to CONLL format.
        
        Args:
            doc_data (Dict[str, Any]): Document data
            
        Returns:
            List[str]: CONLL format lines for this document
        """
        text = doc_data.get('text', '')
        entities = doc_data.get('entities', [])
        
        if not text:
            return []
        
        # Tokenize text
        tokens = self._tokenize_text(text)
        
        # Create BIO tags
        bio_tags = self._create_bio_tags(tokens, entities, text)
        
        # Create CONLL lines
        conll_lines = []
        for token, tag in zip(tokens, bio_tags):
            # Clean token for CONLL format
            clean_token = self._clean_token(token['text'])
            conll_lines.append(f"{clean_token}\t{tag}")
        
        return conll_lines
    
    def _tokenize_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Tokenize text and return tokens with positions.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[Dict[str, Any]]: Tokens with text, start, and end positions
        """
        if self.tokenization_method == 'whitespace':
            return self._whitespace_tokenize(text)
        elif self.tokenization_method == 'simple':
            return self._simple_tokenize(text)
        else:
            raise ValueError(f"Unknown tokenization method: {self.tokenization_method}")
    
    def _whitespace_tokenize(self, text: str) -> List[Dict[str, Any]]:
        """Simple whitespace tokenization."""
        tokens = []
        current_pos = 0
        
        for match in re.finditer(r'\S+', text):
            token_text = match.group()
            start_pos = match.start()
            end_pos = match.end()
            
            tokens.append({
                'text': token_text,
                'start': start_pos,
                'end': end_pos
            })
        
        return tokens
    
    def _simple_tokenize(self, text: str) -> List[Dict[str, Any]]:
        """Simple tokenization that splits on whitespace and punctuation."""
        tokens = []
        current_pos = 0
        
        # Split on whitespace and common punctuation
        pattern = r'(\w+|[^\w\s])'
        
        for match in re.finditer(pattern, text):
            token_text = match.group()
            start_pos = match.start()
            end_pos = match.end()
            
            tokens.append({
                'text': token_text,
                'start': start_pos,
                'end': end_pos
            })
        
        return tokens
    
    def _create_bio_tags(self, tokens: List[Dict[str, Any]], entities: List[Dict[str, Any]], 
                        original_text: str) -> List[str]:
        """
        Create BIO tags for tokens based on entities.
        
        Args:
            tokens (List[Dict[str, Any]]): Tokenized text
            entities (List[Dict[str, Any]]): Entity annotations
            original_text (str): Original text for validation
            
        Returns:
            List[str]: BIO tags for each token
        """
        # Initialize all tags as 'O' (Outside)
        bio_tags = ['O'] * len(tokens)
        
        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda x: x.get('start', 0))
        
        for entity in sorted_entities:
            entity_start = entity.get('start', 0)
            entity_end = entity.get('end', 0)
            entity_label = entity.get('label', 'UNKNOWN')
            
            if entity_start >= entity_end:
                continue
            
            # Find tokens that overlap with this entity
            overlapping_tokens = []
            for i, token in enumerate(tokens):
                token_start = token['start']
                token_end = token['end']
                
                # Check if token overlaps with entity
                if (token_start < entity_end and token_end > entity_start):
                    overlapping_tokens.append(i)
            
            # Apply BIO tagging
            if overlapping_tokens:
                # First token gets B- (Beginning)
                bio_tags[overlapping_tokens[0]] = f"B-{entity_label}"
                
                # Subsequent tokens get I- (Inside)
                for token_idx in overlapping_tokens[1:]:
                    bio_tags[token_idx] = f"I-{entity_label}"
        
        return bio_tags
    
    def _clean_token(self, token: str) -> str:
        """
        Clean token for CONLL format.
        
        Args:
            token (str): Raw token text
            
        Returns:
            str: Cleaned token
        """
        # Replace tabs and newlines to avoid CONLL format issues
        cleaned = token.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # If token becomes empty, use placeholder
        if not cleaned:
            cleaned = '[EMPTY]'
        
        return cleaned
    
    def save_conll(self, conll_lines: List[str], output_path: str) -> bool:
        """
        Save CONLL format lines to file.
        
        Args:
            conll_lines (List[str]): CONLL format lines
            output_path (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in conll_lines:
                    f.write(line + '\n')
            
            file_size = output_path.stat().st_size
            logger.info(f"Saved CONLL format to {output_path} ({file_size:,} bytes, {len(conll_lines)} lines)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save CONLL format to {output_path}: {e}")
            return False
    
    def convert_and_save(self, documents: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Convert documents and save directly to file.
        
        Args:
            documents (List[Dict[str, Any]]): Documents to convert
            output_path (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conll_lines = self.convert_documents_to_conll(documents)
            return self.save_conll(conll_lines, output_path)
        except Exception as e:
            logger.error(f"Failed to convert and save documents: {e}")
            return False
    
    def validate_conll_format(self, conll_lines: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate CONLL format for consistency.
        
        Args:
            conll_lines (List[str]): CONLL format lines to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        current_entity = None
        
        for i, line in enumerate(conll_lines):
            line = line.strip()
            
            # Skip empty lines (document separators)
            if not line:
                current_entity = None
                continue
            
            # Parse line
            parts = line.split('\t')
            if len(parts) != 2:
                errors.append(f"Line {i+1}: Invalid format, expected 2 columns, got {len(parts)}")
                continue
            
            token, tag = parts
            
            # Validate BIO tagging consistency
            if tag.startswith('B-'):
                current_entity = tag[2:]  # Remove 'B-' prefix
            elif tag.startswith('I-'):
                entity_type = tag[2:]  # Remove 'I-' prefix
                if current_entity != entity_type:
                    errors.append(f"Line {i+1}: I-{entity_type} without preceding B-{entity_type}")
            elif tag == 'O':
                current_entity = None
            else:
                errors.append(f"Line {i+1}: Invalid tag format: {tag}")
        
        is_valid = len(errors) == 0
        return is_valid, errors


def get_converter(format_type: str, **kwargs) -> Optional[object]:
    """
    Factory function to get appropriate converter.
    
    Args:
        format_type (str): Format type ('spacy' or 'transformers')
        **kwargs: Additional arguments for converter initialization
        
    Returns:
        Optional[object]: Converter instance or None if format not supported
    """
    if format_type.lower() == 'spacy':
        return SpacyFormatConverter(**kwargs)
    elif format_type.lower() in ['transformers', 'conll', 'bio']:
        return TransformersFormatConverter(**kwargs)
    else:
        logger.error(f"Unsupported format type: {format_type}")
        return None

