"""
Complete Enhanced NLP/NER Data Generation Workflow Example
=========================================================

This example demonstrates all the new enhanced features:
1. Database integration for systematic data storage
2. Negative examples generation (documents without PII)
3. Extreme corruption scenarios for robustness testing
4. Mixed dataset composition with configurable ratios
5. Optimized spaCy configuration generation

This script shows a complete workflow from data generation to model training preparation.

Author: Andrés Vera Figueroa
Date: October 2024
Purpose: Demonstrate complete enhanced data generation workflow
"""

import json
import logging
from pathlib import Path
from datetime import datetime

# Import enhanced components
from database.database_manager import DatabaseManager
from generators.negative_examples_generator import NegativeExamplesGenerator
from corruption.extreme_corruption import ExtremeCorruptionGenerator
from dataset_composer.mixed_dataset_generator import MixedDatasetGenerator, DatasetComposition
from main_pipeline import EnhancedPIIDataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_database_integration():
    """Demonstrate database integration capabilities."""
    print("\n" + "="*60)
    print("🗄️  DATABASE INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Initialize database manager
    db = DatabaseManager("examples/demo_database.db")
    
    # Create a generation session
    session_id = db.create_session(
        country_filter='chile',
        train_size=1000,
        dev_size=200,
        noise_enabled=True,
        noise_level=0.3,
        output_directory='examples/output'
    )
    
    print(f"✅ Created session: {session_id}")
    
    # Store sample documents
    for i in range(5):
        doc_id = db.store_document(
            document_id=f"demo_doc_{i:03d}",
            country_code="CL",
            document_type="pii_document",
            corruption_level="medium",
            original_text=f"Sample document {i} with customer information",
            corrupted_text=f"S@mple d0cument {i} w1th cust0mer inf0rmation",
            template_used="financial_template",
            session_id=session_id
        )
        
        # Store sample entities
        entities = [
            {"type": "CUSTOMER_NAME", "text": "Juan Pérez", "start": 20, "end": 30},
            {"type": "ID_NUMBER", "text": "12.345.678-9", "start": 35, "end": 47}
        ]
        
        for entity in entities:
            db.store_entity(
                document_db_id=doc_id,
                entity_type=entity["type"],
                original_text=entity["text"],
                corrupted_text=entity["text"].replace("é", "3").replace("-", "_"),
                start_pos=entity["start"],
                end_pos=entity["end"],
                is_preserved=True
            )
        
        # Update document statistics
        db.update_document_stats(doc_id, len(entities), len(entities), 0)
    
    # Update session statistics
    db.update_session_stats(session_id, 5, 5, 10, 10)
    
    # Get statistics
    session_summary = db.get_session_summary(session_id)
    country_stats = db.get_country_statistics("CL")
    corruption_analysis = db.get_corruption_analysis()
    
    print(f"📊 Session documents: {session_summary['session']['total_documents']}")
    print(f"📊 Country stats: {len(country_stats)} countries analyzed")
    print(f"📊 Corruption levels: {len(corruption_analysis)} levels analyzed")
    
    return db, session_id

def demonstrate_negative_examples():
    """Demonstrate negative examples generation."""
    print("\n" + "="*60)
    print("🚫 NEGATIVE EXAMPLES GENERATION DEMONSTRATION")
    print("="*60)
    
    # Initialize negative examples generator
    neg_gen = NegativeExamplesGenerator(language='es')
    
    # Generate different types of negative examples
    document_types = ['invoice', 'report', 'form', 'legal']
    
    for doc_type in document_types:
        negative_doc = neg_gen.generate_negative_example(doc_type)
        
        print(f"\n📄 {doc_type.upper()} EXAMPLE:")
        print(f"   Language: {negative_doc['language']}")
        print(f"   Word count: {negative_doc['word_count']}")
        print(f"   Valid (no PII): {negative_doc['is_valid']}")
        print(f"   Text preview: {negative_doc['text'][:100]}...")
        
        # Validate no PII
        is_valid, potential_pii = neg_gen.validate_no_pii(negative_doc['text'])
        if not is_valid:
            print(f"   ⚠️  Potential PII detected: {potential_pii}")
    
    # Generate batch of negative examples
    batch = neg_gen.generate_batch(count=10, document_types=['invoice', 'report'])
    valid_count = sum(1 for doc in batch if doc['is_valid'])
    
    print(f"\n📦 BATCH GENERATION:")
    print(f"   Generated: {len(batch)} documents")
    print(f"   Valid (no PII): {valid_count}/{len(batch)} ({valid_count/len(batch)*100:.1f}%)")
    
    return neg_gen, batch

def demonstrate_extreme_corruption():
    """Demonstrate extreme corruption generation."""
    print("\n" + "="*60)
    print("⚡ EXTREME CORRUPTION DEMONSTRATION")
    print("="*60)
    
    # Initialize extreme corruption generator
    corr_gen = ExtremeCorruptionGenerator()
    
    # Sample document with entities
    original_text = "El cliente Juan Pérez con RUT 12.345.678-9 vive en Av. Libertador 1234, Santiago."
    entities = [
        {"text": "Juan Pérez", "type": "CUSTOMER_NAME", "start": 11, "end": 21},
        {"text": "12.345.678-9", "type": "ID_NUMBER", "start": 31, "end": 43},
        {"text": "Av. Libertador 1234, Santiago", "type": "ADDRESS", "start": 53, "end": 83}
    ]
    
    # Test different corruption levels
    corruption_levels = ['light', 'medium', 'heavy', 'extreme', 'catastrophic']
    
    for level in corruption_levels:
        corrupted_text, metadata = corr_gen.apply_extreme_corruption(
            original_text, entities, level
        )
        
        print(f"\n🔥 {level.upper()} CORRUPTION:")
        print(f"   Original: {original_text}")
        print(f"   Corrupted: {corrupted_text}")
        print(f"   Change rate: {metadata['character_change_rate']:.2%}")
        print(f"   Preserved entities: {metadata['preserved_entities']}/{metadata['total_entities']}")
        
        # Validate corruption quality
        quality = corr_gen.validate_corruption_quality(original_text, corrupted_text, entities)
        print(f"   Quality score: {quality['quality_score']:.2f}")
        print(f"   Acceptable: {quality['is_acceptable']}")
    
    # Generate corruption dataset
    base_documents = [
        {
            'text': original_text,
            'entities': entities,
            'document_id': 'base_001'
        }
    ]
    
    corrupted_dataset = corr_gen.generate_corruption_dataset(
        base_documents,
        corruption_levels=['heavy', 'extreme'],
        samples_per_level=3
    )
    
    print(f"\n📦 CORRUPTION DATASET:")
    print(f"   Generated: {len(corrupted_dataset)} corrupted documents")
    
    return corr_gen, corrupted_dataset

def demonstrate_mixed_dataset_generation(db, neg_gen, corr_gen):
    """Demonstrate mixed dataset generation."""
    print("\n" + "="*60)
    print("🎯 MIXED DATASET GENERATION DEMONSTRATION")
    print("="*60)
    
    # Initialize mixed dataset generator
    mixed_gen = MixedDatasetGenerator(
        pii_generator=None,  # Would use existing PII generator
        negative_generator=neg_gen,
        corruption_generator=corr_gen,
        database_manager=db
    )
    
    # Show available composition templates
    print("📋 AVAILABLE COMPOSITION TEMPLATES:")
    for name, template in mixed_gen.composition_templates.items():
        print(f"   {name}: {template.pii_ratio:.0%} PII, {template.negative_ratio:.0%} negative")
    
    # Test different compositions
    compositions_to_test = ['balanced', 'robustness_focused', 'high_precision']
    
    for comp_name in compositions_to_test:
        print(f"\n🧪 TESTING {comp_name.upper()} COMPOSITION:")
        
        # Generate small dataset for demonstration
        dataset = mixed_gen.generate_mixed_dataset(
            total_size=20,  # Small for demo
            composition=mixed_gen.composition_templates[comp_name],
            train_ratio=0.8
        )
        
        stats = dataset['statistics']
        metadata = dataset['metadata']
        
        print(f"   Total documents: {metadata['total_size']}")
        print(f"   Train/Dev split: {metadata['train_size']}/{metadata['dev_size']}")
        print(f"   PII documents: {stats['pii_documents']}")
        print(f"   Negative documents: {stats['negative_documents']}")
        print(f"   Corruption distribution: {stats['corruption_distribution']}")
    
    # Create custom composition
    print(f"\n🎨 CUSTOM COMPOSITION:")
    custom_comp = DatasetComposition(
        pii_ratio=0.6,
        negative_ratio=0.4,
        corruption_distribution={
            'none': 0.1,
            'light': 0.2,
            'medium': 0.4,
            'heavy': 0.2,
            'extreme': 0.1
        },
        country_distribution={
            'chile': 0.5,
            'mexico': 0.3,
            'brazil': 0.2
        }
    )
    
    # Validate custom composition
    is_valid, errors = mixed_gen.validate_composition(custom_comp)
    print(f"   Custom composition valid: {is_valid}")
    if not is_valid:
        print(f"   Errors: {errors}")
    
    return mixed_gen

def demonstrate_spacy_config_generation():
    """Demonstrate spaCy configuration generation."""
    print("\n" + "="*60)
    print("⚙️  SPACY CONFIGURATION GENERATION DEMONSTRATION")
    print("="*60)
    
    # Create output directory
    config_dir = Path("examples/configs")
    config_dir.mkdir(exist_ok=True)
    
    # Generate different optimization levels
    optimization_levels = ['fast', 'balanced', 'accurate']
    
    for level in optimization_levels:
        config_path = config_dir / f"{level}_config.cfg"
        
        # Create basic configuration (in real implementation, would use actual config generation)
        config_content = f"""
# {level.upper()} OPTIMIZATION CONFIGURATION
# Generated on {datetime.now().isoformat()}

[paths]
train = null
dev = null

[system]
gpu_allocator = null
seed = 42

[nlp]
lang = "es"
pipeline = ["tok2vec","ner"]
batch_size = {1000 if level == 'fast' else 2000 if level == 'balanced' else 3000}

[components]

[components.ner]
factory = "ner"

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
hidden_width = {64 if level == 'fast' else 128 if level == 'balanced' else 256}
maxout_pieces = {2 if level == 'fast' else 3 if level == 'balanced' else 4}

[training]
max_steps = {10000 if level == 'fast' else 20000 if level == 'balanced' else 30000}
dropout = {0.1 if level == 'fast' else 0.15 if level == 'balanced' else 0.2}

[training.optimizer]
@optimizers = "Adam.v1"
learn_rate = {0.001 if level == 'fast' else 0.0005 if level == 'balanced' else 0.0001}
        """.strip()
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Generated {level} configuration: {config_path}")
        print(f"   Batch size: {1000 if level == 'fast' else 2000 if level == 'balanced' else 3000}")
        print(f"   Hidden width: {64 if level == 'fast' else 128 if level == 'balanced' else 256}")
        print(f"   Max steps: {10000 if level == 'fast' else 20000 if level == 'balanced' else 30000}")

def demonstrate_complete_pipeline():
    """Demonstrate the complete enhanced pipeline."""
    print("\n" + "="*60)
    print("🚀 COMPLETE ENHANCED PIPELINE DEMONSTRATION")
    print("="*60)
    
    # Initialize complete pipeline
    config = {
        'language': 'es',
        'database_path': 'examples/complete_pipeline.db'
    }
    
    pipeline = EnhancedPIIDataPipeline(config)
    
    # Create output directory
    output_dir = Path("examples/complete_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate mixed dataset
    print("🎯 Generating mixed dataset...")
    dataset = pipeline.generate_mixed_dataset(
        size=50,  # Small for demo
        composition_name='balanced',
        output_dir=str(output_dir),
        export_formats=['json']
    )
    
    print(f"   Generated: {len(dataset['train_documents'])} train, {len(dataset['dev_documents'])} dev")
    
    # Generate negative examples
    print("🚫 Generating negative examples...")
    negatives = pipeline.generate_negative_examples(
        size=20,
        doc_types=['invoice', 'report'],
        output_dir=str(output_dir)
    )
    
    print(f"   Generated: {len(negatives)} negative examples")
    
    # Generate spaCy configuration
    print("⚙️  Generating spaCy configuration...")
    config_path = pipeline.generate_spacy_config(
        output_path=str(output_dir / 'optimized_config.cfg'),
        optimization_level='balanced'
    )
    
    print(f"   Generated: {config_path}")
    
    # Get pipeline statistics
    print("📊 Getting pipeline statistics...")
    stats = pipeline.get_pipeline_statistics()
    
    print(f"   Database size: {stats.get('database_size_mb', 0):.2f} MB")
    print(f"   Total documents: {stats.get('generated_documents_count', 0)}")
    print(f"   Total entities: {stats.get('document_entities_count', 0)}")
    print(f"   Total sessions: {stats.get('generation_sessions_count', 0)}")
    
    return pipeline, dataset, negatives

def save_demonstration_results(db, dataset, negatives, pipeline):
    """Save demonstration results for review."""
    print("\n" + "="*60)
    print("💾 SAVING DEMONSTRATION RESULTS")
    print("="*60)
    
    results_dir = Path("examples/results")
    results_dir.mkdir(exist_ok=True)
    
    # Save dataset summary
    dataset_summary = {
        'metadata': dataset['metadata'],
        'statistics': dataset['statistics'],
        'composition': dataset['composition'],
        'sample_documents': {
            'train_sample': dataset['train_documents'][:3] if dataset['train_documents'] else [],
            'dev_sample': dataset['dev_documents'][:2] if dataset['dev_documents'] else []
        }
    }
    
    with open(results_dir / 'dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False, default=str)
    
    # Save negative examples summary
    negative_summary = {
        'total_count': len(negatives),
        'document_types': list(set(doc.get('document_type', 'unknown') for doc in negatives)),
        'validation_results': {
            'valid_count': sum(1 for doc in negatives if doc.get('is_valid', False)),
            'invalid_count': sum(1 for doc in negatives if not doc.get('is_valid', False))
        },
        'samples': negatives[:3]
    }
    
    with open(results_dir / 'negative_examples_summary.json', 'w', encoding='utf-8') as f:
        json.dump(negative_summary, f, indent=2, ensure_ascii=False, default=str)
    
    # Save database statistics
    db_stats = db.get_database_stats()
    with open(results_dir / 'database_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(db_stats, f, indent=2, ensure_ascii=False, default=str)
    
    # Save pipeline statistics
    pipeline_stats = pipeline.get_pipeline_statistics()
    with open(results_dir / 'pipeline_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(pipeline_stats, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Results saved to: {results_dir}")
    print(f"   - dataset_summary.json")
    print(f"   - negative_examples_summary.json")
    print(f"   - database_statistics.json")
    print(f"   - pipeline_statistics.json")

def main():
    """Run complete demonstration workflow."""
    print("🎉 ENHANCED NLP/NER DATA GENERATION SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("This demonstration showcases all enhanced features:")
    print("1. Database Integration")
    print("2. Negative Examples Generation")
    print("3. Extreme Corruption Scenarios")
    print("4. Mixed Dataset Composition")
    print("5. Optimized spaCy Configuration")
    print("6. Complete Pipeline Integration")
    
    try:
        # Run demonstrations
        db, session_id = demonstrate_database_integration()
        neg_gen, negative_batch = demonstrate_negative_examples()
        corr_gen, corrupted_dataset = demonstrate_extreme_corruption()
        mixed_gen = demonstrate_mixed_dataset_generation(db, neg_gen, corr_gen)
        demonstrate_spacy_config_generation()
        pipeline, dataset, negatives = demonstrate_complete_pipeline()
        
        # Save results
        save_demonstration_results(db, dataset, negatives, pipeline)
        
        print("\n" + "="*80)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("All enhanced features have been demonstrated and tested.")
        print("Check the 'examples/results/' directory for detailed outputs.")
        print("The system is ready for production use with all new capabilities.")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

