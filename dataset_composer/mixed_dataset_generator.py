"""
Mixed Dataset Generator for Balanced NER Training
================================================

This module creates balanced training sets that mix PII documents, negative examples,
and various corruption levels to ensure comprehensive NER model training that reflects
real-world document distributions.

Key Features:
- Configurable ratios of PII vs non-PII documents
- Distribution control across corruption levels
- Country and entity type balancing
- Stratified sampling for consistent distributions
- Dataset splitting with maintained balance
- Statistics tracking for dataset composition
- Export formats for different training frameworks

Supported Compositions:
- Pure PII datasets (traditional approach)
- Negative-heavy datasets (robustness focus)
- Balanced mixed datasets (recommended)
- Corruption-graduated datasets (progressive difficulty)
- Country-specific balanced datasets

Author: Andrés Vera Figueroa
Date: October 2024
Purpose: Generate balanced datasets for robust NER model training
"""

import random
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from pathlib import Path
import logging

# Import format converters
from .format_converters import SpacyFormatConverter, TransformersFormatConverter

logger = logging.getLogger(__name__)

@dataclass
class DatasetComposition:
    """Configuration for dataset composition."""
    pii_ratio: float = 0.7  # Percentage of documents with PII
    negative_ratio: float = 0.3  # Percentage of documents without PII
    
    # Corruption level distribution (must sum to 1.0)
    corruption_distribution: Dict[str, float] = None
    
    # Country distribution (must sum to 1.0)
    country_distribution: Dict[str, float] = None
    
    # Entity type balance (relative weights)
    entity_type_weights: Dict[str, float] = None
    
    # Document type distribution for negative examples
    negative_doc_types: Dict[str, float] = None
    
    def __post_init__(self):
        if self.corruption_distribution is None:
            self.corruption_distribution = {
                'none': 0.2,
                'light': 0.3,
                'medium': 0.25,
                'heavy': 0.15,
                'extreme': 0.1
            }
        
        if self.country_distribution is None:
            self.country_distribution = {
                'chile': 0.4,
                'mexico': 0.3,
                'brazil': 0.2,
                'uruguay': 0.1
            }
        
        if self.entity_type_weights is None:
            self.entity_type_weights = {
                'CUSTOMER_NAME': 1.2,
                'ID_NUMBER': 1.5,
                'ADDRESS': 1.1,
                'PHONE_NUMBER': 1.0,
                'EMAIL': 1.0,
                'AMOUNT': 1.3,
                'SEQ_NUMBER': 0.8
            }
        
        if self.negative_doc_types is None:
            self.negative_doc_types = {
                'invoice': 0.3,
                'report': 0.25,
                'form': 0.25,
                'legal': 0.2
            }

class MixedDatasetGenerator:
    """
    Generator for balanced mixed datasets combining PII and non-PII documents.
    
    Creates comprehensive training sets that reflect real-world document
    distributions and corruption patterns.
    """
    
    def __init__(self, pii_generator=None, negative_generator=None, 
                 corruption_generator=None, database_manager=None):
        """
        Initialize the mixed dataset generator.
        
        Args:
            pii_generator: PII data generator instance
            negative_generator: Negative examples generator instance
            corruption_generator: Extreme corruption generator instance
            database_manager: Database manager for storage
        """
        self.pii_generator = pii_generator
        self.negative_generator = negative_generator
        self.corruption_generator = corruption_generator
        self.database_manager = database_manager
        
        # Predefined composition templates
        self.composition_templates = self._load_composition_templates()
    
    def _load_composition_templates(self) -> Dict[str, DatasetComposition]:
        """Load predefined dataset composition templates."""
        return {
            'balanced': DatasetComposition(
                pii_ratio=0.7,
                negative_ratio=0.3,
                corruption_distribution={
                    'none': 0.15,
                    'light': 0.35,
                    'medium': 0.25,
                    'heavy': 0.15,
                    'extreme': 0.1
                }
            ),
            'robustness_focused': DatasetComposition(
                pii_ratio=0.6,
                negative_ratio=0.4,
                corruption_distribution={
                    'none': 0.1,
                    'light': 0.2,
                    'medium': 0.3,
                    'heavy': 0.25,
                    'extreme': 0.15
                }
            ),
            'high_precision': DatasetComposition(
                pii_ratio=0.8,
                negative_ratio=0.2,
                corruption_distribution={
                    'none': 0.3,
                    'light': 0.4,
                    'medium': 0.2,
                    'heavy': 0.1,
                    'extreme': 0.0
                }
            ),
            'extreme_robustness': DatasetComposition(
                pii_ratio=0.5,
                negative_ratio=0.5,
                corruption_distribution={
                    'none': 0.05,
                    'light': 0.15,
                    'medium': 0.25,
                    'heavy': 0.3,
                    'extreme': 0.25
                }
            ),
            'production_ready': DatasetComposition(
                pii_ratio=0.75,
                negative_ratio=0.25,
                corruption_distribution={
                    'none': 0.2,
                    'light': 0.4,
                    'medium': 0.25,
                    'heavy': 0.1,
                    'extreme': 0.05
                }
            )
        }
    
    def generate_mixed_dataset(self, total_size: int, composition: DatasetComposition,
                             train_ratio: float = 0.8, session_id: str = None) -> Dict[str, Any]:
        """
        Generate a mixed dataset with specified composition.
        
        Args:
            total_size (int): Total number of documents to generate
            composition (DatasetComposition): Dataset composition configuration
            train_ratio (float): Ratio for train/dev split
            session_id (str): Session ID for tracking
            
        Returns:
            Dict[str, Any]: Generated dataset with metadata
        """
        logger.info(f"Generating mixed dataset of {total_size} documents")
        
        # Calculate document counts
        pii_count = int(total_size * composition.pii_ratio)
        negative_count = total_size - pii_count
        
        # Generate PII documents
        pii_documents = self._generate_pii_documents(pii_count, composition)
        
        # Generate negative examples
        negative_documents = self._generate_negative_documents(negative_count, composition)
        
        # Combine and shuffle
        all_documents = pii_documents + negative_documents
        random.shuffle(all_documents)
        
        # Split into train/dev
        train_size = int(len(all_documents) * train_ratio)
        train_documents = all_documents[:train_size]
        dev_documents = all_documents[train_size:]
        
        # Generate statistics
        stats = self._calculate_dataset_statistics(all_documents, composition)
        
        # Store in database if available
        if self.database_manager and session_id:
            self._store_dataset_in_database(all_documents, session_id, stats)
        
        dataset = {
            'train_documents': train_documents,
            'dev_documents': dev_documents,
            'composition': asdict(composition),
            'statistics': stats,
            'metadata': {
                'total_size': total_size,
                'train_size': len(train_documents),
                'dev_size': len(dev_documents),
                'pii_count': pii_count,
                'negative_count': negative_count,
                'generation_timestamp': np.datetime64('now').isoformat()
            }
        }
        
        logger.info(f"Generated mixed dataset: {len(train_documents)} train, {len(dev_documents)} dev")
        return dataset
    
    def _generate_pii_documents(self, count: int, composition: DatasetComposition) -> List[Dict[str, Any]]:
        """Generate PII documents according to composition."""
        documents = []
        
        # Distribute across countries
        country_counts = self._distribute_counts(count, composition.country_distribution)
        
        # Distribute across corruption levels
        corruption_counts = self._distribute_counts(count, composition.corruption_distribution)
        
        for country, country_count in country_counts.items():
            for corruption_level, corruption_count in corruption_counts.items():
                # Calculate documents for this country-corruption combination
                docs_to_generate = int((country_count / count) * (corruption_count / count) * count)
                
                if docs_to_generate == 0:
                    continue
                
                # Generate documents using PII generator
                if self.pii_generator:
                    country_docs = self._generate_pii_batch(
                        docs_to_generate, country, corruption_level, composition
                    )
                    documents.extend(country_docs)
        
        # Ensure we have the exact count
        while len(documents) < count:
            # Generate additional documents
            country = random.choices(
                list(composition.country_distribution.keys()),
                weights=list(composition.country_distribution.values())
            )[0]
            corruption = random.choices(
                list(composition.corruption_distribution.keys()),
                weights=list(composition.corruption_distribution.values())
            )[0]
            
            doc = self._generate_single_pii_document(country, corruption, composition)
            documents.append(doc)
        
        return documents[:count]
    
    def _generate_negative_documents(self, count: int, composition: DatasetComposition) -> List[Dict[str, Any]]:
        """Generate negative example documents."""
        documents = []
        
        # Distribute across document types
        type_counts = self._distribute_counts(count, composition.negative_doc_types)
        
        # Distribute across corruption levels
        corruption_counts = self._distribute_counts(count, composition.corruption_distribution)
        
        for doc_type, type_count in type_counts.items():
            for corruption_level, corruption_count in corruption_counts.items():
                docs_to_generate = int((type_count / count) * (corruption_count / count) * count)
                
                if docs_to_generate == 0:
                    continue
                
                # Generate negative documents
                if self.negative_generator:
                    negative_docs = self._generate_negative_batch(
                        docs_to_generate, doc_type, corruption_level
                    )
                    documents.extend(negative_docs)
        
        # Ensure we have the exact count
        while len(documents) < count:
            doc_type = random.choices(
                list(composition.negative_doc_types.keys()),
                weights=list(composition.negative_doc_types.values())
            )[0]
            corruption = random.choices(
                list(composition.corruption_distribution.keys()),
                weights=list(composition.corruption_distribution.values())
            )[0]
            
            doc = self._generate_single_negative_document(doc_type, corruption)
            documents.append(doc)
        
        return documents[:count]
    
    def _generate_pii_batch(self, count: int, country: str, corruption_level: str,
                           composition: DatasetComposition) -> List[Dict[str, Any]]:
        """Generate a batch of PII documents for specific country and corruption level."""
        documents = []
        
        for _ in range(count):
            doc = self._generate_single_pii_document(country, corruption_level, composition)
            documents.append(doc)
        
        return documents
    
    def _generate_single_pii_document(self, country: str, corruption_level: str,
                                    composition: DatasetComposition) -> Dict[str, Any]:
        """Generate a single PII document."""
        # This would integrate with your existing PII generator
        # For now, creating a placeholder structure
        
        document = {
            'text': f"Sample PII document for {country} with {corruption_level} corruption",
            'entities': [],  # Would be populated by PII generator
            'country': country,
            'corruption_level': corruption_level,
            'document_type': 'pii_document',
            'has_pii': True,
            'metadata': {
                'generation_method': 'pii_generator',
                'entity_types': list(composition.entity_type_weights.keys())
            }
        }
        
        return document
    
    def _generate_negative_batch(self, count: int, doc_type: str, 
                               corruption_level: str) -> List[Dict[str, Any]]:
        """Generate a batch of negative example documents."""
        documents = []
        
        for _ in range(count):
            doc = self._generate_single_negative_document(doc_type, corruption_level)
            documents.append(doc)
        
        return documents
    
    def _generate_single_negative_document(self, doc_type: str, 
                                         corruption_level: str) -> Dict[str, Any]:
        """Generate a single negative example document."""
        if self.negative_generator:
            negative_doc = self.negative_generator.generate_negative_example(doc_type)
            
            # Apply corruption if needed
            if corruption_level != 'none' and self.corruption_generator:
                corrupted_text, corruption_metadata = self.corruption_generator.apply_extreme_corruption(
                    negative_doc['text'], [], corruption_level
                )
                negative_doc['text'] = corrupted_text
                negative_doc['corruption_metadata'] = corruption_metadata
            
            negative_doc.update({
                'corruption_level': corruption_level,
                'document_type': 'negative_example',
                'has_pii': False,
                'entities': []  # No entities in negative examples
            })
            
            return negative_doc
        
        # Fallback if no negative generator available
        return {
            'text': f"Sample negative document of type {doc_type} with {corruption_level} corruption",
            'entities': [],
            'document_type': 'negative_example',
            'corruption_level': corruption_level,
            'has_pii': False,
            'metadata': {
                'generation_method': 'fallback',
                'doc_type': doc_type
            }
        }
    
    def _distribute_counts(self, total: int, distribution: Dict[str, float]) -> Dict[str, int]:
        """Distribute total count according to distribution weights."""
        counts = {}
        remaining = total
        
        # Sort by value to handle rounding better
        sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        
        for key, ratio in sorted_items[:-1]:
            count = int(total * ratio)
            counts[key] = count
            remaining -= count
        
        # Assign remaining to last item
        if sorted_items:
            counts[sorted_items[-1][0]] = remaining
        
        return counts
    
    def _calculate_dataset_statistics(self, documents: List[Dict[str, Any]], 
                                    composition: DatasetComposition) -> Dict[str, Any]:
        """Calculate comprehensive dataset statistics."""
        stats = {
            'total_documents': len(documents),
            'pii_documents': sum(1 for doc in documents if doc.get('has_pii', False)),
            'negative_documents': sum(1 for doc in documents if not doc.get('has_pii', False)),
            'country_distribution': Counter(doc.get('country', 'unknown') for doc in documents),
            'corruption_distribution': Counter(doc.get('corruption_level', 'unknown') for doc in documents),
            'document_type_distribution': Counter(doc.get('document_type', 'unknown') for doc in documents),
            'entity_statistics': self._calculate_entity_statistics(documents),
            'text_statistics': self._calculate_text_statistics(documents)
        }
        
        # Convert counters to regular dicts for JSON serialization
        stats['country_distribution'] = dict(stats['country_distribution'])
        stats['corruption_distribution'] = dict(stats['corruption_distribution'])
        stats['document_type_distribution'] = dict(stats['document_type_distribution'])
        
        return stats
    
    def _calculate_entity_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate entity-related statistics."""
        total_entities = 0
        entity_type_counts = Counter()
        
        for doc in documents:
            entities = doc.get('entities', [])
            total_entities += len(entities)
            
            for entity in entities:
                entity_type = entity.get('type', entity.get('label', 'unknown'))
                entity_type_counts[entity_type] += 1
        
        return {
            'total_entities': total_entities,
            'entity_type_distribution': dict(entity_type_counts),
            'average_entities_per_document': total_entities / len(documents) if documents else 0
        }
    
    def _calculate_text_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate text-related statistics."""
        word_counts = [len(doc.get('text', '').split()) for doc in documents]
        char_counts = [len(doc.get('text', '')) for doc in documents]
        
        return {
            'average_words_per_document': np.mean(word_counts) if word_counts else 0,
            'average_chars_per_document': np.mean(char_counts) if char_counts else 0,
            'min_words': min(word_counts) if word_counts else 0,
            'max_words': max(word_counts) if word_counts else 0,
            'total_words': sum(word_counts),
            'total_characters': sum(char_counts)
        }
    
    def _store_dataset_in_database(self, documents: List[Dict[str, Any]], 
                                 session_id: str, stats: Dict[str, Any]):
        """Store generated dataset in database."""
        if not self.database_manager:
            return
        
        try:
            for doc in documents:
                # Store document
                doc_db_id = self.database_manager.store_document(
                    document_id=doc.get('document_id', f"mixed_{random.randint(1000, 9999)}"),
                    country_code=doc.get('country', 'unknown'),
                    document_type=doc.get('document_type', 'mixed_document'),
                    corruption_level=doc.get('corruption_level', 'none'),
                    original_text=doc.get('text', ''),
                    corrupted_text=doc.get('corrupted_text'),
                    template_used=doc.get('template_used'),
                    generation_mode='mixed_dataset',
                    session_id=session_id
                )
                
                # Store entities
                entities = doc.get('entities', [])
                for entity in entities:
                    self.database_manager.store_entity(
                        document_db_id=doc_db_id,
                        entity_type=entity.get('type', entity.get('label', 'unknown')),
                        original_text=entity.get('text', ''),
                        corrupted_text=entity.get('corrupted_text', entity.get('text', '')),
                        start_pos=entity.get('start', 0),
                        end_pos=entity.get('end', 0),
                        is_preserved=entity.get('is_preserved', True)
                    )
                
                # Update document stats
                self.database_manager.update_document_stats(
                    doc_db_id, len(entities), 
                    sum(1 for e in entities if e.get('is_preserved', True)),
                    sum(1 for e in entities if not e.get('is_preserved', True))
                )
            
            logger.info(f"Stored {len(documents)} documents in database for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store dataset in database: {e}")
    
    def export_dataset(self, dataset: Dict[str, Any], output_dir: str, 
                      formats: List[str] = None) -> Dict[str, str]:
        """
        Export dataset in various formats.
        
        Args:
            dataset (Dict[str, Any]): Generated dataset
            output_dir (str): Output directory
            formats (List[str]): Export formats ('spacy', 'json', 'csv')
            
        Returns:
            Dict[str, str]: Mapping of format to file path
        """
        if formats is None:
            formats = ['json', 'spacy']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for format_type in formats:
            if format_type == 'json':
                file_path = output_path / 'mixed_dataset.json'
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False, default=str)
                exported_files['json'] = str(file_path)
            
            elif format_type == 'spacy':
                # Export spaCy format using format converter
                try:
                    spacy_converter = SpacyFormatConverter(language='es')
                    
                    # Convert and save train data
                    train_path = output_path / 'train.spacy'
                    if dataset.get('train_documents'):
                        success_train = spacy_converter.convert_and_save(
                            dataset['train_documents'], str(train_path)
                        )
                        if success_train:
                            exported_files['spacy_train'] = str(train_path)
                    
                    # Convert and save dev data
                    dev_path = output_path / 'dev.spacy'
                    if dataset.get('dev_documents'):
                        success_dev = spacy_converter.convert_and_save(
                            dataset['dev_documents'], str(dev_path)
                        )
                        if success_dev:
                            exported_files['spacy_dev'] = str(dev_path)
                    
                    logger.info(f"spaCy export completed: train={train_path.exists()}, dev={dev_path.exists()}")
                    
                except Exception as e:
                    logger.error(f"Failed to export spaCy format: {e}")
                    # Continue with other formats even if spaCy export fails
            
            elif format_type == 'csv':
                # Export CSV format for analysis
                import pandas as pd
                
                # Flatten documents for CSV
                rows = []
                for doc in dataset['train_documents'] + dataset['dev_documents']:
                    rows.append({
                        'text': doc.get('text', ''),
                        'country': doc.get('country', ''),
                        'corruption_level': doc.get('corruption_level', ''),
                        'document_type': doc.get('document_type', ''),
                        'has_pii': doc.get('has_pii', False),
                        'entity_count': len(doc.get('entities', []))
                    })
                
                df = pd.DataFrame(rows)
                csv_path = output_path / 'mixed_dataset.csv'
                df.to_csv(csv_path, index=False)
                exported_files['csv'] = str(csv_path)
            
            elif format_type in ['transformers', 'conll', 'bio']:
                # Export Transformers/CONLL format using format converter
                try:
                    transformers_converter = TransformersFormatConverter(tokenization_method='whitespace')
                    
                    # Convert and save train data
                    train_path = output_path / 'train.conll'
                    if dataset.get('train_documents'):
                        success_train = transformers_converter.convert_and_save(
                            dataset['train_documents'], str(train_path)
                        )
                        if success_train:
                            exported_files['transformers_train'] = str(train_path)
                    
                    # Convert and save dev data
                    dev_path = output_path / 'dev.conll'
                    if dataset.get('dev_documents'):
                        success_dev = transformers_converter.convert_and_save(
                            dataset['dev_documents'], str(dev_path)
                        )
                        if success_dev:
                            exported_files['transformers_dev'] = str(dev_path)
                    
                    logger.info(f"Transformers export completed: train={train_path.exists()}, dev={dev_path.exists()}")
                    
                except Exception as e:
                    logger.error(f"Failed to export Transformers format: {e}")
                    # Continue with other formats even if Transformers export fails
        
        return exported_files
    
    def validate_composition(self, composition: DatasetComposition) -> Tuple[bool, List[str]]:
        """
        Validate dataset composition configuration.
        
        Args:
            composition (DatasetComposition): Composition to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check ratios sum to 1.0
        if abs(composition.pii_ratio + composition.negative_ratio - 1.0) > 0.001:
            errors.append("PII ratio and negative ratio must sum to 1.0")
        
        # Check distribution sums
        if abs(sum(composition.corruption_distribution.values()) - 1.0) > 0.001:
            errors.append("Corruption distribution must sum to 1.0")
        
        if abs(sum(composition.country_distribution.values()) - 1.0) > 0.001:
            errors.append("Country distribution must sum to 1.0")
        
        if abs(sum(composition.negative_doc_types.values()) - 1.0) > 0.001:
            errors.append("Negative document types distribution must sum to 1.0")
        
        # Check for negative values
        if any(v < 0 for v in [composition.pii_ratio, composition.negative_ratio]):
            errors.append("Ratios cannot be negative")
        
        return len(errors) == 0, errors
