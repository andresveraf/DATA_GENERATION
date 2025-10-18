"""
Enhanced Multi-Country PII Data Generation Pipeline
==================================================

This is the main integration pipeline that combines all enhanced features:
- Database integration for systematic PII data storage
- Negative examples generation (documents without PII)
- Extreme corruption scenarios for robustness testing
- Mixed dataset composition with configurable ratios
- Optimized spaCy configuration generation
- Comprehensive statistics and export functionality

Usage Examples:
    # Generate balanced mixed dataset
    python main_pipeline.py --mode mixed-dataset --size 10000 --composition balanced
    
    # Generate extreme corruption dataset
    python main_pipeline.py --mode extreme-corruption --size 5000 --corruption-level extreme
    
    # Generate negative examples only
    python main_pipeline.py --mode negative-only --size 2000 --doc-types invoice,report
    
    # Generate with database storage
    python main_pipeline.py --mode full-pipeline --size 20000 --store-db --export-formats json,spacy,csv

Author: Andrés Vera Figueroa
Date: October 2024
Purpose: Comprehensive NER training data generation with all enhanced features
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import all the new components
from database.database_manager import DatabaseManager
from generators.negative_examples_generator import NegativeExamplesGenerator
from corruption.extreme_corruption import ExtremeCorruptionGenerator
from dataset_composer.mixed_dataset_generator import MixedDatasetGenerator, DatasetComposition

# Import existing components (would need to be adapted)
# from Spacy.data_generation_noisy import PII_Generator  # Placeholder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedPIIDataPipeline:
    """
    Main pipeline class that orchestrates all data generation components.
    
    Integrates database storage, negative examples, extreme corruption,
    and mixed dataset generation into a unified system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the enhanced pipeline.
        
        Args:
            config (Dict[str, Any]): Pipeline configuration
        """
        self.config = config or {}
        
        # Initialize components
        self.db_manager = DatabaseManager(
            self.config.get('database_path', 'database/pii_generation.db')
        )
        
        self.negative_generator = NegativeExamplesGenerator(
            language=self.config.get('language', 'es')
        )
        
        self.corruption_generator = ExtremeCorruptionGenerator()
        
        self.mixed_dataset_generator = MixedDatasetGenerator(
            pii_generator=None,  # Would be initialized with existing PII generator
            negative_generator=self.negative_generator,
            corruption_generator=self.corruption_generator,
            database_manager=self.db_manager
        )
        
        logger.info("Enhanced PII Data Pipeline initialized")
    
    def generate_mixed_dataset(self, size: int, composition_name: str = 'balanced',
                             custom_composition: Dict = None, output_dir: str = 'output',
                             export_formats: List[str] = None) -> Dict[str, Any]:
        """
        Generate a mixed dataset with PII and negative examples.
        
        Args:
            size (int): Total number of documents to generate
            composition_name (str): Predefined composition template name
            custom_composition (Dict): Custom composition configuration
            output_dir (str): Output directory for exports
            export_formats (List[str]): Export formats
            
        Returns:
            Dict[str, Any]: Generated dataset with metadata
        """
        logger.info(f"Generating mixed dataset: {size} documents, composition: {composition_name}")
        
        # Create session
        session_id = self.db_manager.create_session(
            country_filter='all',
            train_size=int(size * 0.8),
            dev_size=int(size * 0.2),
            noise_enabled=True,
            output_directory=output_dir
        )
        
        # Get composition
        if custom_composition:
            composition = DatasetComposition(**custom_composition)
        else:
            composition = self.mixed_dataset_generator.composition_templates.get(
                composition_name, 
                self.mixed_dataset_generator.composition_templates['balanced']
            )
        
        # Validate composition
        is_valid, errors = self.mixed_dataset_generator.validate_composition(composition)
        if not is_valid:
            raise ValueError(f"Invalid composition: {', '.join(errors)}")
        
        # Generate dataset
        dataset = self.mixed_dataset_generator.generate_mixed_dataset(
            total_size=size,
            composition=composition,
            session_id=session_id
        )
        
        # Export dataset
        if export_formats:
            exported_files = self.mixed_dataset_generator.export_dataset(
                dataset, output_dir, export_formats
            )
            dataset['exported_files'] = exported_files
            
            # Record exports in database
            for format_type, file_path in exported_files.items():
                file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
                self.db_manager.record_export(
                    session_id, format_type, file_path, file_size, 
                    len(dataset['train_documents']) + len(dataset['dev_documents'])
                )
        
        # Update session statistics
        total_docs = len(dataset['train_documents']) + len(dataset['dev_documents'])
        total_entities = sum(len(doc.get('entities', [])) for doc in dataset['train_documents'] + dataset['dev_documents'])
        successful_entities = sum(
            sum(1 for e in doc.get('entities', []) if e.get('is_preserved', True))
            for doc in dataset['train_documents'] + dataset['dev_documents']
        )
        
        self.db_manager.update_session_stats(
            session_id, total_docs, total_docs, total_entities, successful_entities
        )
        
        logger.info(f"Mixed dataset generated successfully: {total_docs} documents")
        return dataset
    
    def generate_negative_examples(self, size: int, doc_types: List[str] = None,
                                 corruption_levels: List[str] = None,
                                 output_dir: str = 'output') -> List[Dict[str, Any]]:
        """
        Generate negative examples (documents without PII).
        
        Args:
            size (int): Number of negative examples to generate
            doc_types (List[str]): Document types to generate
            corruption_levels (List[str]): Corruption levels to apply
            output_dir (str): Output directory
            
        Returns:
            List[Dict[str, Any]]: Generated negative examples
        """
        logger.info(f"Generating {size} negative examples")
        
        if doc_types is None:
            doc_types = ['invoice', 'report', 'form', 'legal']
        
        if corruption_levels is None:
            corruption_levels = ['none', 'light', 'medium']
        
        # Create session
        session_id = self.db_manager.create_session(
            country_filter='all',
            train_size=size,
            dev_size=0,
            noise_enabled=len([c for c in corruption_levels if c != 'none']) > 0,
            output_directory=output_dir
        )
        
        negative_examples = []
        
        for _ in range(size):
            # Select random document type and corruption level
            doc_type = self.negative_generator.random.choice(doc_types)
            corruption_level = self.negative_generator.random.choice(corruption_levels)
            
            # Generate negative example
            negative_doc = self.negative_generator.generate_negative_example(doc_type)
            
            # Apply corruption if needed
            if corruption_level != 'none':
                corrupted_text, corruption_metadata = self.corruption_generator.apply_extreme_corruption(
                    negative_doc['text'], [], corruption_level
                )
                negative_doc['text'] = corrupted_text
                negative_doc['corruption_metadata'] = corruption_metadata
            
            negative_doc['corruption_level'] = corruption_level
            negative_doc['document_id'] = f"neg_{len(negative_examples):06d}"
            
            # Store in database
            doc_db_id = self.db_manager.store_document(
                document_id=negative_doc['document_id'],
                country_code='all',
                document_type='negative_example',
                corruption_level=corruption_level,
                original_text=negative_doc['text'],
                session_id=session_id
            )
            
            # Update document stats (no entities for negative examples)
            self.db_manager.update_document_stats(doc_db_id, 0, 0, 0)
            
            negative_examples.append(negative_doc)
        
        # Export negative examples
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'negative_examples.json', 'w', encoding='utf-8') as f:
            json.dump(negative_examples, f, indent=2, ensure_ascii=False, default=str)
        
        # Record export
        file_path = str(output_path / 'negative_examples.json')
        file_size = Path(file_path).stat().st_size
        self.db_manager.record_export(session_id, 'json', file_path, file_size, len(negative_examples))
        
        # Update session stats
        self.db_manager.update_session_stats(session_id, len(negative_examples), len(negative_examples), 0, 0)
        
        logger.info(f"Generated {len(negative_examples)} negative examples")
        return negative_examples
    
    def generate_extreme_corruption_dataset(self, base_documents: List[Dict], 
                                          corruption_levels: List[str] = None,
                                          samples_per_level: int = 100,
                                          output_dir: str = 'output') -> List[Dict[str, Any]]:
        """
        Generate dataset with extreme corruption scenarios.
        
        Args:
            base_documents (List[Dict]): Base documents to corrupt
            corruption_levels (List[str]): Corruption levels to apply
            samples_per_level (int): Number of samples per corruption level
            output_dir (str): Output directory
            
        Returns:
            List[Dict[str, Any]]: Corrupted dataset
        """
        logger.info(f"Generating extreme corruption dataset")
        
        if corruption_levels is None:
            corruption_levels = ['heavy', 'extreme', 'catastrophic']
        
        # Create session
        session_id = self.db_manager.create_session(
            country_filter='all',
            train_size=len(corruption_levels) * samples_per_level,
            dev_size=0,
            noise_enabled=True,
            noise_level=0.8,  # High noise level
            output_directory=output_dir
        )
        
        # Generate corrupted dataset
        corrupted_dataset = self.corruption_generator.generate_corruption_dataset(
            base_documents, corruption_levels, samples_per_level
        )
        
        # Store in database
        for doc in corrupted_dataset:
            doc_db_id = self.db_manager.store_document(
                document_id=doc['document_id'],
                country_code='all',
                document_type='pii_document',
                corruption_level=doc['corruption_level'],
                original_text=doc['original_text'],
                corrupted_text=doc['corrupted_text'],
                session_id=session_id
            )
            
            # Store entities
            entities = doc.get('original_entities', [])
            preserved_count = 0
            
            for entity in entities:
                is_preserved = entity.get('is_preserved', True)
                if is_preserved:
                    preserved_count += 1
                
                self.db_manager.store_entity(
                    document_db_id=doc_db_id,
                    entity_type=entity.get('type', 'unknown'),
                    original_text=entity.get('text', ''),
                    corrupted_text=entity.get('corrupted_text', entity.get('text', '')),
                    start_pos=entity.get('start', 0),
                    end_pos=entity.get('end', 0),
                    is_preserved=is_preserved
                )
            
            # Update document stats
            self.db_manager.update_document_stats(
                doc_db_id, len(entities), preserved_count, len(entities) - preserved_count
            )
        
        # Export dataset
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'extreme_corruption_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(corrupted_dataset, f, indent=2, ensure_ascii=False, default=str)
        
        # Record export
        file_path = str(output_path / 'extreme_corruption_dataset.json')
        file_size = Path(file_path).stat().st_size
        self.db_manager.record_export(session_id, 'json', file_path, file_size, len(corrupted_dataset))
        
        # Update session stats
        total_entities = sum(len(doc.get('original_entities', [])) for doc in corrupted_dataset)
        preserved_entities = sum(
            sum(1 for e in doc.get('original_entities', []) if e.get('is_preserved', True))
            for doc in corrupted_dataset
        )
        
        self.db_manager.update_session_stats(
            session_id, len(corrupted_dataset), len(corrupted_dataset), 
            total_entities, preserved_entities
        )
        
        logger.info(f"Generated extreme corruption dataset: {len(corrupted_dataset)} documents")
        return corrupted_dataset
    
    def generate_spacy_config(self, output_path: str = 'Spacy/config.cfg',
                            optimization_level: str = 'balanced') -> str:
        """
        Generate optimized spaCy configuration file.
        
        Args:
            output_path (str): Path to save configuration file
            optimization_level (str): Optimization level ('fast', 'balanced', 'accurate')
            
        Returns:
            str: Path to generated configuration file
        """
        logger.info(f"Generating spaCy configuration: {optimization_level}")
        
        # Configuration templates
        configs = {
            'fast': 'configs/fast_config.cfg',
            'balanced': 'configs/optimized_config.cfg',
            'accurate': 'configs/accurate_config.cfg'
        }
        
        # Use existing optimized config as base
        source_config = configs.get(optimization_level, configs['balanced'])
        
        # Copy configuration file
        import shutil
        if Path(source_config).exists():
            shutil.copy(source_config, output_path)
            logger.info(f"spaCy configuration generated: {output_path}")
        else:
            logger.warning(f"Source config not found: {source_config}, using default")
            # Create basic config if source doesn't exist
            self._create_basic_config(output_path)
        
        return output_path
    
    def _create_basic_config(self, output_path: str):
        """Create a basic spaCy configuration file."""
        basic_config = """
[paths]
train = null
dev = null

[system]
gpu_allocator = null

[nlp]
lang = "es"
pipeline = ["tok2vec","ner"]
batch_size = 1000

[components]

[components.ner]
factory = "ner"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
hidden_width = 64
maxout_pieces = 2

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = 96

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
max_steps = 20000

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
        """.strip()
        
        with open(output_path, 'w') as f:
            f.write(basic_config)
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return self.db_manager.get_database_stats()
    
    def cleanup_old_data(self, days_old: int = 30) -> int:
        """Clean up old data from database."""
        return self.db_manager.cleanup_old_data(days_old)

def main():
    """Main CLI interface for the enhanced pipeline."""
    parser = argparse.ArgumentParser(description='Enhanced PII Data Generation Pipeline')
    
    parser.add_argument('--mode', choices=[
        'mixed-dataset', 'negative-only', 'extreme-corruption', 'full-pipeline', 'spacy-config'
    ], required=True, help='Generation mode')
    
    parser.add_argument('--size', type=int, default=1000, help='Number of documents to generate')
    parser.add_argument('--composition', default='balanced', help='Dataset composition template')
    parser.add_argument('--corruption-level', default='medium', help='Corruption level for extreme corruption')
    parser.add_argument('--doc-types', default='invoice,report,form,legal', help='Document types for negative examples')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--export-formats', default='json', help='Export formats (comma-separated)')
    parser.add_argument('--language', default='es', help='Language (es/pt)')
    parser.add_argument('--config-path', default='Spacy/config.cfg', help='spaCy config output path')
    parser.add_argument('--optimization-level', default='balanced', help='spaCy optimization level')
    parser.add_argument('--store-db', action='store_true', help='Store results in database')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    config = {
        'language': args.language,
        'database_path': 'database/pii_generation.db' if args.store_db else None
    }
    
    pipeline = EnhancedPIIDataPipeline(config)
    
    try:
        if args.mode == 'mixed-dataset':
            dataset = pipeline.generate_mixed_dataset(
                size=args.size,
                composition_name=args.composition,
                output_dir=args.output_dir,
                export_formats=args.export_formats.split(',')
            )
            print(f"Generated mixed dataset: {len(dataset['train_documents'])} train, {len(dataset['dev_documents'])} dev")
        
        elif args.mode == 'negative-only':
            doc_types = args.doc_types.split(',')
            negative_examples = pipeline.generate_negative_examples(
                size=args.size,
                doc_types=doc_types,
                output_dir=args.output_dir
            )
            print(f"Generated {len(negative_examples)} negative examples")
        
        elif args.mode == 'extreme-corruption':
            # For demo purposes, create some base documents
            base_docs = [
                {'text': 'Sample document with PII entities', 'entities': [], 'document_id': 'base_001'}
            ]
            corrupted_dataset = pipeline.generate_extreme_corruption_dataset(
                base_documents=base_docs,
                corruption_levels=[args.corruption_level],
                samples_per_level=args.size,
                output_dir=args.output_dir
            )
            print(f"Generated extreme corruption dataset: {len(corrupted_dataset)} documents")
        
        elif args.mode == 'spacy-config':
            config_path = pipeline.generate_spacy_config(
                output_path=args.config_path,
                optimization_level=args.optimization_level
            )
            print(f"Generated spaCy configuration: {config_path}")
        
        elif args.mode == 'full-pipeline':
            # Run complete pipeline
            print("Running full pipeline...")
            
            # Generate mixed dataset
            dataset = pipeline.generate_mixed_dataset(
                size=args.size,
                composition_name=args.composition,
                output_dir=args.output_dir,
                export_formats=args.export_formats.split(',')
            )
            
            # Generate spaCy config
            config_path = pipeline.generate_spacy_config(
                optimization_level=args.optimization_level
            )
            
            # Show statistics
            stats = pipeline.get_pipeline_statistics()
            print(f"Pipeline completed successfully!")
            print(f"Database statistics: {stats}")
        
        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

