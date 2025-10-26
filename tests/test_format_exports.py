"""
Test Format Export Functionality
===============================

This module tests the format converters for spaCy and Transformers exports
to ensure data integrity and format compliance.

Tests include:
- spaCy DocBin format validation
- CONLL/BIO format validation  
- Entity preservation and alignment
- Round-trip conversion testing
- Error handling and edge cases

Author: Andrés Vera Figueroa
Date: October 2024
Purpose: Validate dual-format export functionality
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any

# Import the converters
import sys
sys.path.append(str(Path(__file__).parent.parent))

from dataset_composer.format_converters import SpacyFormatConverter, TransformersFormatConverter


class TestSpacyFormatConverter:
    """Test cases for spaCy format converter."""
    
    @pytest.fixture
    def sample_documents(self) -> List[Dict[str, Any]]:
        """Sample documents for testing."""
        return [
            {
                'document_id': 'test_001',
                'text': 'El cliente Juan Pérez con RUT 12.345.678-9 reside en Santiago.',
                'entities': [
                    {'start': 11, 'end': 21, 'label': 'CUSTOMER_NAME', 'text': 'Juan Pérez'},
                    {'start': 26, 'end': 39, 'label': 'ID_NUMBER', 'text': '12.345.678-9'},
                    {'start': 50, 'end': 58, 'label': 'ADDRESS', 'text': 'Santiago'}
                ]
            },
            {
                'document_id': 'test_002',
                'text': 'Factura número 12345 por $50.000 pesos.',
                'entities': [
                    {'start': 15, 'end': 20, 'label': 'SEQ_NUMBER', 'text': '12345'},
                    {'start': 25, 'end': 32, 'label': 'AMOUNT', 'text': '$50.000'}
                ]
            },
            {
                'document_id': 'test_003',
                'text': 'Documento sin entidades PII.',
                'entities': []
            }
        ]
    
    @pytest.fixture
    def spacy_converter(self):
        """Create spaCy converter instance."""
        return SpacyFormatConverter(language='es')
    
    def test_converter_initialization(self, spacy_converter):
        """Test converter initializes correctly."""
        assert spacy_converter.language == 'es'
        assert spacy_converter.nlp is not None
        assert spacy_converter.DocBin is not None
    
    def test_create_spacy_doc(self, spacy_converter, sample_documents):
        """Test creation of spaCy Doc objects."""
        doc_data = sample_documents[0]
        doc = spacy_converter._create_spacy_doc(doc_data)
        
        assert doc is not None
        assert doc.text == doc_data['text']
        assert len(doc.ents) == len(doc_data['entities'])
        
        # Check entity details
        for i, ent in enumerate(doc.ents):
            expected_entity = doc_data['entities'][i]
            assert ent.start_char == expected_entity['start']
            assert ent.end_char == expected_entity['end']
            assert ent.label_ == expected_entity['label']
            assert ent.text == expected_entity['text']
    
    def test_convert_documents_to_docbin(self, spacy_converter, sample_documents):
        """Test conversion of documents to DocBin."""
        doc_bin = spacy_converter.convert_documents_to_docbin(sample_documents)
        
        assert doc_bin is not None
        assert len(doc_bin) == len(sample_documents)
        
        # Test loading docs from DocBin
        docs = list(doc_bin.get_docs(spacy_converter.nlp.vocab))
        assert len(docs) == len(sample_documents)
        
        # Verify first document
        doc = docs[0]
        assert doc.text == sample_documents[0]['text']
        assert len(doc.ents) == len(sample_documents[0]['entities'])
    
    def test_save_and_load_docbin(self, spacy_converter, sample_documents):
        """Test saving and loading DocBin files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test.spacy'
            
            # Convert and save
            success = spacy_converter.convert_and_save(sample_documents, str(output_path))
            assert success
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            
            # Load and verify
            try:
                from spacy.tokens import DocBin
                loaded_doc_bin = DocBin().from_disk(output_path)
                loaded_docs = list(loaded_doc_bin.get_docs(spacy_converter.nlp.vocab))
                
                assert len(loaded_docs) == len(sample_documents)
                assert loaded_docs[0].text == sample_documents[0]['text']
                
            except ImportError:
                pytest.skip("spaCy not available for loading test")
    
    def test_overlapping_entities_resolution(self, spacy_converter):
        """Test handling of overlapping entities."""
        doc_data = {
            'document_id': 'overlap_test',
            'text': 'Juan Pérez Martínez',
            'entities': [
                {'start': 0, 'end': 10, 'label': 'CUSTOMER_NAME', 'text': 'Juan Pérez'},
                {'start': 5, 'end': 19, 'label': 'CUSTOMER_NAME', 'text': 'Pérez Martínez'}
            ]
        }
        
        doc = spacy_converter._create_spacy_doc(doc_data)
        assert doc is not None
        # Should resolve overlapping entities (keep one)
        assert len(doc.ents) <= len(doc_data['entities'])
    
    def test_empty_document_handling(self, spacy_converter):
        """Test handling of empty documents."""
        empty_doc = {
            'document_id': 'empty_test',
            'text': '',
            'entities': []
        }
        
        doc = spacy_converter._create_spacy_doc(empty_doc)
        assert doc is None  # Should return None for empty text
    
    def test_invalid_entity_spans(self, spacy_converter):
        """Test handling of invalid entity spans."""
        invalid_doc = {
            'document_id': 'invalid_test',
            'text': 'Short text',
            'entities': [
                {'start': -1, 'end': 5, 'label': 'TEST', 'text': 'invalid'},  # Negative start
                {'start': 5, 'end': 100, 'label': 'TEST', 'text': 'invalid'},  # End beyond text
                {'start': 8, 'end': 5, 'label': 'TEST', 'text': 'invalid'}   # Start > end
            ]
        }
        
        doc = spacy_converter._create_spacy_doc(invalid_doc)
        assert doc is not None
        # Should skip invalid entities
        assert len(doc.ents) == 0


class TestTransformersFormatConverter:
    """Test cases for Transformers format converter."""
    
    @pytest.fixture
    def sample_documents(self) -> List[Dict[str, Any]]:
        """Sample documents for testing."""
        return [
            {
                'document_id': 'test_001',
                'text': 'Juan Pérez vive en Santiago.',
                'entities': [
                    {'start': 0, 'end': 10, 'label': 'CUSTOMER_NAME', 'text': 'Juan Pérez'},
                    {'start': 19, 'end': 27, 'label': 'ADDRESS', 'text': 'Santiago'}
                ]
            },
            {
                'document_id': 'test_002',
                'text': 'Factura 12345.',
                'entities': [
                    {'start': 8, 'end': 13, 'label': 'SEQ_NUMBER', 'text': '12345'}
                ]
            }
        ]
    
    @pytest.fixture
    def transformers_converter(self):
        """Create Transformers converter instance."""
        return TransformersFormatConverter(tokenization_method='whitespace')
    
    def test_converter_initialization(self, transformers_converter):
        """Test converter initializes correctly."""
        assert transformers_converter.tokenization_method == 'whitespace'
    
    def test_whitespace_tokenization(self, transformers_converter):
        """Test whitespace tokenization."""
        text = "Juan Pérez vive en Santiago."
        tokens = transformers_converter._whitespace_tokenize(text)
        
        expected_tokens = ["Juan", "Pérez", "vive", "en", "Santiago."]
        assert len(tokens) == len(expected_tokens)
        
        for i, token in enumerate(tokens):
            assert token['text'] == expected_tokens[i]
            assert 'start' in token
            assert 'end' in token
            assert token['start'] < token['end']
    
    def test_bio_tag_creation(self, transformers_converter):
        """Test BIO tag creation."""
        text = "Juan Pérez vive en Santiago."
        tokens = transformers_converter._whitespace_tokenize(text)
        entities = [
            {'start': 0, 'end': 10, 'label': 'CUSTOMER_NAME', 'text': 'Juan Pérez'},
            {'start': 19, 'end': 27, 'label': 'ADDRESS', 'text': 'Santiago'}
        ]
        
        bio_tags = transformers_converter._create_bio_tags(tokens, entities, text)
        
        # Expected: ["B-CUSTOMER_NAME", "I-CUSTOMER_NAME", "O", "O", "B-ADDRESS"]
        assert len(bio_tags) == len(tokens)
        assert bio_tags[0] == "B-CUSTOMER_NAME"
        assert bio_tags[1] == "I-CUSTOMER_NAME"
        assert bio_tags[2] == "O"
        assert bio_tags[3] == "O"
        assert bio_tags[4] == "B-ADDRESS"
    
    def test_convert_single_document(self, transformers_converter, sample_documents):
        """Test conversion of single document."""
        doc_data = sample_documents[0]
        conll_lines = transformers_converter._convert_single_document(doc_data)
        
        assert len(conll_lines) > 0
        
        # Check format: each line should be "token\ttag"
        for line in conll_lines:
            parts = line.split('\t')
            assert len(parts) == 2
            token, tag = parts
            assert len(token) > 0
            assert tag in ['O'] or tag.startswith('B-') or tag.startswith('I-')
    
    def test_convert_documents_to_conll(self, transformers_converter, sample_documents):
        """Test conversion of multiple documents."""
        conll_lines = transformers_converter.convert_documents_to_conll(sample_documents)
        
        assert len(conll_lines) > 0
        
        # Should have blank lines between documents
        blank_lines = [i for i, line in enumerate(conll_lines) if line == ""]
        assert len(blank_lines) >= len(sample_documents) - 1
    
    def test_save_and_load_conll(self, transformers_converter, sample_documents):
        """Test saving and loading CONLL files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test.conll'
            
            # Convert and save
            success = transformers_converter.convert_and_save(sample_documents, str(output_path))
            assert success
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            
            # Load and verify
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            assert len(lines) > 0
            
            # Check format
            for line in lines:
                line = line.strip()
                if line:  # Skip blank lines
                    parts = line.split('\t')
                    assert len(parts) == 2
    
    def test_validate_conll_format(self, transformers_converter):
        """Test CONLL format validation."""
        # Valid CONLL lines
        valid_lines = [
            "Juan\tB-CUSTOMER_NAME",
            "Pérez\tI-CUSTOMER_NAME",
            "vive\tO",
            "",  # Document separator
            "Santiago\tB-ADDRESS"
        ]
        
        is_valid, errors = transformers_converter.validate_conll_format(valid_lines)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid CONLL lines
        invalid_lines = [
            "Juan\tB-CUSTOMER_NAME",
            "Pérez\tI-ADDRESS",  # Wrong entity type after B-CUSTOMER_NAME
            "invalid_format",    # Missing tab
            "word\tINVALID_TAG"  # Invalid tag format
        ]
        
        is_valid, errors = transformers_converter.validate_conll_format(invalid_lines)
        assert not is_valid
        assert len(errors) > 0
    
    def test_token_cleaning(self, transformers_converter):
        """Test token cleaning for CONLL format."""
        # Test various problematic tokens
        test_cases = [
            ("normal", "normal"),
            ("with\ttab", "with tab"),
            ("with\nnewline", "with newline"),
            ("", "[EMPTY]"),
            ("  spaces  ", "spaces")
        ]
        
        for input_token, expected_output in test_cases:
            cleaned = transformers_converter._clean_token(input_token)
            assert cleaned == expected_output
    
    def test_empty_document_handling(self, transformers_converter):
        """Test handling of empty documents."""
        empty_doc = {
            'document_id': 'empty_test',
            'text': '',
            'entities': []
        }
        
        conll_lines = transformers_converter._convert_single_document(empty_doc)
        assert len(conll_lines) == 0
    
    def test_no_entities_document(self, transformers_converter):
        """Test document with no entities."""
        no_entities_doc = {
            'document_id': 'no_entities_test',
            'text': 'This document has no entities.',
            'entities': []
        }
        
        conll_lines = transformers_converter._convert_single_document(no_entities_doc)
        assert len(conll_lines) > 0
        
        # All tags should be 'O'
        for line in conll_lines:
            parts = line.split('\t')
            assert len(parts) == 2
            token, tag = parts
            assert tag == 'O'


class TestFormatConverterIntegration:
    """Integration tests for both converters."""
    
    @pytest.fixture
    def sample_dataset(self) -> Dict[str, Any]:
        """Sample dataset structure."""
        return {
            'train_documents': [
                {
                    'document_id': 'train_001',
                    'text': 'El cliente Juan Pérez con RUT 12.345.678-9.',
                    'entities': [
                        {'start': 11, 'end': 21, 'label': 'CUSTOMER_NAME', 'text': 'Juan Pérez'},
                        {'start': 26, 'end': 39, 'label': 'ID_NUMBER', 'text': '12.345.678-9'}
                    ]
                }
            ],
            'dev_documents': [
                {
                    'document_id': 'dev_001',
                    'text': 'Factura número 12345.',
                    'entities': [
                        {'start': 15, 'end': 20, 'label': 'SEQ_NUMBER', 'text': '12345'}
                    ]
                }
            ]
        }
    
    def test_entity_count_preservation(self, sample_dataset):
        """Test that entity counts are preserved across formats."""
        # Count original entities
        original_train_entities = sum(len(doc['entities']) for doc in sample_dataset['train_documents'])
        original_dev_entities = sum(len(doc['entities']) for doc in sample_dataset['dev_documents'])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test spaCy format
            try:
                spacy_converter = SpacyFormatConverter(language='es')
                train_path = Path(temp_dir) / 'train.spacy'
                dev_path = Path(temp_dir) / 'dev.spacy'
                
                spacy_converter.convert_and_save(sample_dataset['train_documents'], str(train_path))
                spacy_converter.convert_and_save(sample_dataset['dev_documents'], str(dev_path))
                
                # Load and count entities
                from spacy.tokens import DocBin
                train_doc_bin = DocBin().from_disk(train_path)
                dev_doc_bin = DocBin().from_disk(dev_path)
                
                train_docs = list(train_doc_bin.get_docs(spacy_converter.nlp.vocab))
                dev_docs = list(dev_doc_bin.get_docs(spacy_converter.nlp.vocab))
                
                spacy_train_entities = sum(len(doc.ents) for doc in train_docs)
                spacy_dev_entities = sum(len(doc.ents) for doc in dev_docs)
                
                assert spacy_train_entities == original_train_entities
                assert spacy_dev_entities == original_dev_entities
                
            except ImportError:
                pytest.skip("spaCy not available for integration test")
            
            # Test Transformers format
            transformers_converter = TransformersFormatConverter()
            train_conll_path = Path(temp_dir) / 'train.conll'
            dev_conll_path = Path(temp_dir) / 'dev.conll'
            
            transformers_converter.convert_and_save(sample_dataset['train_documents'], str(train_conll_path))
            transformers_converter.convert_and_save(sample_dataset['dev_documents'], str(dev_conll_path))
            
            # Count entities in CONLL format
            def count_entities_in_conll(file_path):
                entity_count = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and '\t' in line:
                            token, tag = line.split('\t')
                            if tag.startswith('B-'):
                                entity_count += 1
                return entity_count
            
            conll_train_entities = count_entities_in_conll(train_conll_path)
            conll_dev_entities = count_entities_in_conll(dev_conll_path)
            
            assert conll_train_entities == original_train_entities
            assert conll_dev_entities == original_dev_entities
    
    def test_format_compatibility(self, sample_dataset):
        """Test that both formats can be generated from the same data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate both formats
            try:
                spacy_converter = SpacyFormatConverter(language='es')
                spacy_path = Path(temp_dir) / 'data.spacy'
                spacy_success = spacy_converter.convert_and_save(
                    sample_dataset['train_documents'], str(spacy_path)
                )
                assert spacy_success
                assert spacy_path.exists()
                
            except ImportError:
                pytest.skip("spaCy not available for compatibility test")
            
            transformers_converter = TransformersFormatConverter()
            conll_path = Path(temp_dir) / 'data.conll'
            conll_success = transformers_converter.convert_and_save(
                sample_dataset['train_documents'], str(conll_path)
            )
            assert conll_success
            assert conll_path.exists()


def test_get_converter_factory():
    """Test the converter factory function."""
    from dataset_composer.format_converters import get_converter
    
    # Test spaCy converter
    spacy_converter = get_converter('spacy', language='es')
    assert isinstance(spacy_converter, SpacyFormatConverter)
    assert spacy_converter.language == 'es'
    
    # Test Transformers converter
    transformers_converter = get_converter('transformers', tokenization_method='whitespace')
    assert isinstance(transformers_converter, TransformersFormatConverter)
    assert transformers_converter.tokenization_method == 'whitespace'
    
    # Test alternative names
    conll_converter = get_converter('conll')
    assert isinstance(conll_converter, TransformersFormatConverter)
    
    bio_converter = get_converter('bio')
    assert isinstance(bio_converter, TransformersFormatConverter)
    
    # Test unsupported format
    unsupported_converter = get_converter('unsupported')
    assert unsupported_converter is None


if __name__ == '__main__':
    # Run tests if script is executed directly
    pytest.main([__file__, '-v'])

