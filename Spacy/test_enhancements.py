#!/usr/bin/env python3
"""
Test script for enhanced templates and entity shuffling functionality.
"""

from data_generation_noisy import generate_example_with_custom_mode, get_sentence_templates
import random

def test_enhanced_templates():
    """Test the enhanced template system."""
    print('🔄 Testing Enhanced Templates and Entity Shuffling')
    print('=' * 60)
    
    countries = ['chile', 'mexico', 'brazil', 'uruguay']
    
    # Test template counts
    for country in countries:
        templates = get_sentence_templates(country)
        print(f'\n📍 {country.upper()} Templates: {len(templates)} total')
        
        # Show sample templates from different categories
        if len(templates) > 3:
            print(f'   Sample templates:')
            for i, template in enumerate(templates[:3]):
                print(f'   {i+1}. {template[:80]}...')
    
    print('\n🎯 Testing Entity Shuffling in Different Modes:')
    print('=' * 60)
    
    modes = ['full', 'personal_id', 'address_focused', 'contact_only', 'financial_heavy', 'table_heavy', 'minimal_entities']
    
    for mode in modes:
        print(f'\n🔀 Mode: {mode}')
        try:
            # Generate 2 examples to see shuffling variations
            for i in range(2):
                sentence, annotations = generate_example_with_custom_mode(
                    country='chile', 
                    mode=mode, 
                    include_noise=True,
                    noise_level=0.15
                )
                
                # Count entities
                entity_count = len(annotations['entities'])
                entity_types = [label for _, _, label in annotations['entities']]
                
                print(f'   Example {i+1}: {entity_count} entities')
                print(f'   Types: {list(set(entity_types))}')
                print(f'   Text: {sentence[:100]}...')
            print()
        except Exception as e:
            print(f'   Error in mode {mode}: {e}')
    
    print('✅ Enhanced templates and entity shuffling test completed!')

def test_specific_country_templates():
    """Test specific country template diversity."""
    print('\n🌎 Testing Country-Specific Template Diversity:')
    print('=' * 60)
    
    countries = ['chile', 'mexico', 'brazil', 'uruguay']
    
    for country in countries:
        print(f'\n🏛️ {country.upper()} Examples:')
        for i in range(3):
            sentence, annotations = generate_example_with_custom_mode(
                country=country, 
                mode='full', 
                include_noise=True
            )
            print(f'   {i+1}. {sentence[:120]}...')

if __name__ == "__main__":
    test_enhanced_templates()
    test_specific_country_templates()
