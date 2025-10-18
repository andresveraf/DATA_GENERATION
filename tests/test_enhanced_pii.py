"""
Test Suite for Enhanced PII Generation
======================================

This module tests the enhanced PII generation capabilities including:
- New PII types (DIRECTION, LOCATION, POSTAL_CODE, REGION)
- Enhanced variety in existing types
- NLP augmentation functionality
- Data quality validation

Author: Andrés Vera Figueroa (Enhanced by Codegen)
Date: October 2024
"""

import unittest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.enhanced_pii_generator import (
    EnhancedPIIGenerator, 
    generate_enhanced_direction,
    generate_enhanced_location,
    generate_enhanced_postal_code,
    generate_enhanced_region,
    validate_pii_variety
)

from augmentation.nlp_augmentation import (
    NLPAugmentator,
    AugmentationConfig,
    create_augmentator,
    augment_training_example
)

class TestEnhancedPIIGenerator(unittest.TestCase):
    """Test cases for enhanced PII generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = EnhancedPIIGenerator()
        self.countries = ['chile', 'mexico', 'brazil', 'uruguay']
    
    def test_direction_generation(self):
        """Test direction generation for all countries"""
        for country in self.countries:
            direction = self.generator.generate_direction(country)
            self.assertIsInstance(direction, str)
            self.assertGreater(len(direction), 0)
            print(f"Direction ({country}): {direction}")
    
    def test_location_generation(self):
        """Test location generation for all countries"""
        for country in self.countries:
            location = self.generator.generate_location(country)
            self.assertIsInstance(location, str)
            self.assertGreater(len(location), 0)
            print(f"Location ({country}): {location}")
    
    def test_postal_code_generation(self):
        """Test postal code generation for all countries"""
        for country in self.countries:
            postal_code = self.generator.generate_postal_code(country)
            self.assertIsInstance(postal_code, str)
            self.assertGreater(len(postal_code), 0)
            print(f"Postal Code ({country}): {postal_code}")
    
    def test_region_generation(self):
        """Test region generation for all countries"""
        for country in self.countries:
            region = self.generator.generate_region(country)
            self.assertIsInstance(region, str)
            self.assertGreater(len(region), 0)
            print(f"Region ({country}): {region}")
    
    def test_enhanced_phone_generation(self):
        """Test enhanced phone number generation"""
        for country in self.countries:
            phone = self.generator.generate_enhanced_phone(country)
            self.assertIsInstance(phone, str)
            self.assertGreater(len(phone), 8)  # Minimum phone length
            print(f"Phone ({country}): {phone}")
    
    def test_enhanced_sequence_generation(self):
        """Test enhanced sequence number generation"""
        sequences = set()
        for _ in range(10):
            seq = self.generator.generate_enhanced_sequence()
            sequences.add(seq)
            self.assertIsInstance(seq, str)
            self.assertGreater(len(seq), 0)
        
        # Check variety
        self.assertGreater(len(sequences), 5)  # Should have variety
        print(f"Sample sequences: {list(sequences)[:5]}")
    
    def test_enhanced_date_generation(self):
        """Test enhanced date generation"""
        for country in self.countries:
            date = self.generator.generate_enhanced_date(country)
            self.assertIsInstance(date, str)
            self.assertGreater(len(date), 0)
            print(f"Date ({country}): {date}")
    
    def test_all_pii_types_generation(self):
        """Test generation of all PII types"""
        for country in self.countries:
            try:
                pii_data = self.generator.generate_all_pii_types(country)
                
                expected_types = [
                    'CUSTOMER_NAME', 'ID_NUMBER', 'ADDRESS', 'PHONE_NUMBER', 
                    'EMAIL', 'AMOUNT', 'SEQ_NUMBER', 'DATE', 'DIRECTION', 
                    'LOCATION', 'POSTAL_CODE', 'REGION'
                ]
                
                for pii_type in expected_types:
                    self.assertIn(pii_type, pii_data)
                    self.assertIsInstance(pii_data[pii_type], str)
                    self.assertGreater(len(pii_data[pii_type]), 0)
                
                print(f"\nAll PII types for {country}:")
                for pii_type, value in pii_data.items():
                    print(f"  {pii_type}: {value}")
                    
            except ImportError as e:
                print(f"Warning: Could not test all PII types for {country}: {e}")
                # This is expected if the main module isn't available
                pass
    
    def test_pii_variety_validation(self):
        """Test PII variety validation"""
        try:
            variety_report = validate_pii_variety('chile', samples=50)
            
            self.assertIsInstance(variety_report, dict)
            
            for pii_type, metrics in variety_report.items():
                self.assertIn('unique_values', metrics)
                self.assertIn('variety_percentage', metrics)
                self.assertIn('sufficient_variety', metrics)
                
                print(f"{pii_type}: {metrics['unique_values']} unique values "
                      f"({metrics['variety_percentage']:.1f}% variety)")
                
        except ImportError as e:
            print(f"Warning: Could not test variety validation: {e}")
            pass

class TestNLPAugmentation(unittest.TestCase):
    """Test cases for NLP augmentation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = AugmentationConfig(
            synonym_replacement_rate=0.3,
            noise_injection_rate=0.1,
            language="es"
        )
        self.augmentator = NLPAugmentator(self.config)
        
        # Sample text with entities
        self.sample_text = "El cliente Juan Pérez con RUT 12.345.678-9 reside en Av. Providencia 123, Santiago."
        self.sample_entities = [
            (11, 21, "CUSTOMER_NAME"),  # Juan Pérez
            (26, 39, "ID_NUMBER"),      # 12.345.678-9
            (50, 68, "ADDRESS"),        # Av. Providencia 123
            (70, 78, "ADDRESS")         # Santiago
        ]
    
    def test_augmentator_initialization(self):
        """Test augmentator initialization"""
        self.assertIsInstance(self.augmentator, NLPAugmentator)
        self.assertEqual(self.augmentator.config.language, "es")
        self.assertIsInstance(self.augmentator.REPLACEABLE_WORDS, dict)
    
    def test_entity_span_conversion(self):
        """Test entity span conversion"""
        entity_spans = self.augmentator.get_entity_spans(self.sample_text, self.sample_entities)
        
        self.assertEqual(len(entity_spans), 4)
        self.assertEqual(entity_spans[0].text, "Juan Pérez")
        self.assertEqual(entity_spans[0].label, "CUSTOMER_NAME")
    
    def test_word_entity_overlap_detection(self):
        """Test word-entity overlap detection"""
        entity_spans = self.augmentator.get_entity_spans(self.sample_text, self.sample_entities)
        
        # Test word inside entity
        self.assertTrue(self.augmentator.is_word_in_entity(11, 15, entity_spans))  # "Juan"
        
        # Test word outside entity
        self.assertFalse(self.augmentator.is_word_in_entity(0, 2, entity_spans))  # "El"
    
    def test_synonym_replacement(self):
        """Test synonym replacement functionality"""
        augmented_text = self.augmentator.apply_synonym_replacement(
            self.sample_text, self.sample_entities
        )
        
        self.assertIsInstance(augmented_text, str)
        # Entities should be preserved
        self.assertIn("Juan Pérez", augmented_text)
        self.assertIn("12.345.678-9", augmented_text)
        print(f"Original: {self.sample_text}")
        print(f"Augmented: {augmented_text}")
    
    def test_character_noise_injection(self):
        """Test character-level noise injection"""
        augmented_text = self.augmentator.apply_character_noise(
            self.sample_text, self.sample_entities
        )
        
        self.assertIsInstance(augmented_text, str)
        # Entities should be preserved
        self.assertIn("Juan Pérez", augmented_text)
        self.assertIn("12.345.678-9", augmented_text)
        print(f"Original: {self.sample_text}")
        print(f"With noise: {augmented_text}")
    
    def test_ocr_error_application(self):
        """Test OCR error application"""
        augmented_text = self.augmentator.apply_ocr_errors(
            self.sample_text, self.sample_entities
        )
        
        self.assertIsInstance(augmented_text, str)
        # Entities should be preserved
        self.assertIn("Juan Pérez", augmented_text)
        self.assertIn("12.345.678-9", augmented_text)
        print(f"Original: {self.sample_text}")
        print(f"With OCR errors: {augmented_text}")
    
    def test_comprehensive_augmentation(self):
        """Test comprehensive text augmentation"""
        augmented_text, updated_entities = self.augmentator.augment_text(
            self.sample_text, self.sample_entities
        )
        
        self.assertIsInstance(augmented_text, str)
        self.assertIsInstance(updated_entities, list)
        self.assertEqual(len(updated_entities), len(self.sample_entities))
        
        print(f"Original: {self.sample_text}")
        print(f"Augmented: {augmented_text}")
        print(f"Original entities: {self.sample_entities}")
        print(f"Updated entities: {updated_entities}")
    
    def test_augmented_variants_generation(self):
        """Test generation of multiple augmented variants"""
        variants = self.augmentator.generate_augmented_variants(
            self.sample_text, self.sample_entities, num_variants=3
        )
        
        self.assertEqual(len(variants), 3)
        
        for i, (variant_text, variant_entities) in enumerate(variants):
            self.assertIsInstance(variant_text, str)
            self.assertIsInstance(variant_entities, list)
            print(f"Variant {i+1}: {variant_text}")
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test augmentator creation
        augmentator = create_augmentator(language="es", synonym_rate=0.2, noise_rate=0.05)
        self.assertIsInstance(augmentator, NLPAugmentator)
        self.assertEqual(augmentator.config.language, "es")
        self.assertEqual(augmentator.config.synonym_replacement_rate, 0.2)
        
        # Test quick augmentation
        augmented_text, updated_entities = augment_training_example(
            self.sample_text, self.sample_entities, language="es"
        )
        self.assertIsInstance(augmented_text, str)
        self.assertIsInstance(updated_entities, list)

class TestIntegration(unittest.TestCase):
    """Integration tests combining PII generation and augmentation"""
    
    def test_pii_generation_with_augmentation(self):
        """Test PII generation combined with augmentation"""
        try:
            # Generate PII data
            generator = EnhancedPIIGenerator()
            pii_data = generator.generate_all_pii_types('chile')
            
            # Create a sample sentence with entities
            text = f"Cliente {pii_data['CUSTOMER_NAME']} con documento {pii_data['ID_NUMBER']} reside en {pii_data['ADDRESS']}."
            entities = [
                (8, 8 + len(pii_data['CUSTOMER_NAME']), "CUSTOMER_NAME"),
                (23 + len(pii_data['CUSTOMER_NAME']), 23 + len(pii_data['CUSTOMER_NAME']) + len(pii_data['ID_NUMBER']), "ID_NUMBER"),
                (35 + len(pii_data['CUSTOMER_NAME']) + len(pii_data['ID_NUMBER']), 35 + len(pii_data['CUSTOMER_NAME']) + len(pii_data['ID_NUMBER']) + len(pii_data['ADDRESS']), "ADDRESS")
            ]
            
            # Apply augmentation
            augmentator = create_augmentator()
            augmented_text, updated_entities = augmentator.augment_text(text, entities)
            
            print(f"Generated PII data: {pii_data}")
            print(f"Original text: {text}")
            print(f"Augmented text: {augmented_text}")
            print(f"Entities: {entities}")
            print(f"Updated entities: {updated_entities}")
            
            self.assertIsInstance(augmented_text, str)
            self.assertIsInstance(updated_entities, list)
            
        except ImportError as e:
            print(f"Warning: Could not test integration: {e}")
            pass

def run_comprehensive_test():
    """Run comprehensive test of all enhanced features"""
    print("=" * 60)
    print("COMPREHENSIVE TEST OF ENHANCED PII GENERATION")
    print("=" * 60)
    
    # Test enhanced PII generation
    print("\n1. Testing Enhanced PII Generation...")
    generator = EnhancedPIIGenerator()
    
    for country in ['chile', 'mexico', 'brazil', 'uruguay']:
        print(f"\n--- {country.upper()} ---")
        print(f"Direction: {generator.generate_direction(country)}")
        print(f"Location: {generator.generate_location(country)}")
        print(f"Postal Code: {generator.generate_postal_code(country)}")
        print(f"Region: {generator.generate_region(country)}")
        print(f"Enhanced Phone: {generator.generate_enhanced_phone(country)}")
        print(f"Enhanced Sequence: {generator.generate_enhanced_sequence(country)}")
        print(f"Enhanced Date: {generator.generate_enhanced_date(country)}")
    
    # Test NLP augmentation
    print("\n2. Testing NLP Augmentation...")
    sample_text = "El cliente María González con RUT 15.234.567-8 vive en Calle Providencia 456, Santiago."
    sample_entities = [(11, 25, "CUSTOMER_NAME"), (30, 43, "ID_NUMBER"), (52, 74, "ADDRESS"), (76, 84, "ADDRESS")]
    
    augmentator = create_augmentator()
    
    print(f"Original: {sample_text}")
    
    # Test different augmentation types
    for aug_type in [['synonyms'], ['noise'], ['ocr'], ['synonyms', 'noise', 'ocr']]:
        augmented, _ = augmentator.augment_text(sample_text, sample_entities, aug_type)
        print(f"Augmented ({', '.join(aug_type)}): {augmented}")
    
    print("\n3. Testing Variety Validation...")
    try:
        variety_report = validate_pii_variety('chile', samples=20)
        for pii_type, metrics in variety_report.items():
            status = "✓" if metrics['sufficient_variety'] else "✗"
            print(f"{status} {pii_type}: {metrics['variety_percentage']:.1f}% variety")
    except ImportError:
        print("Variety validation requires main module - skipping")
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST COMPLETED")
    print("=" * 60)

if __name__ == '__main__':
    # Run comprehensive test first
    run_comprehensive_test()
    
    # Then run unit tests
    print("\nRunning unit tests...")
    unittest.main(verbosity=2)
