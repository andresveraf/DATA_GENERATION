#!/usr/bin/env python3
"""
Test script for improved entity detection logic
"""

from data_generation_noisy import generate_example_with_noise
import random

def test_entity_detection():
    print('🧪 Testing improved entity detection...')
    print()

    # Test with different countries
    countries = ['chile', 'mexico', 'brazil', 'uruguay']
    total_examples = 0
    total_failed = 0

    for country in countries:
        print(f'🌎 Testing {country.upper()}:')
        failed_count = 0
        example_count = 10
        entity_counts = []
        
        for i in range(example_count):
            try:
                sentence, annotations = generate_example_with_noise(country)
                
                # Check entity detection
                entities = annotations['entities']
                entity_counts.append(len(entities))
                found_types = set([entity[2] for entity in entities])
                
                # Show first example for each country
                if i == 0:
                    print(f'  📝 Sample: {sentence[:100]}...')
                    print(f'  🏷️  Entities: {len(entities)} found')
                    print(f'  📊 Types: {sorted(found_types)}')
                
                # Consider it failed if too few entities found
                if len(entities) < 4:  # At least 4 entities should be found typically
                    failed_count += 1
                    
            except Exception as e:
                print(f'  ❌ Error in example {i}: {str(e)}')
                failed_count += 1
        
        avg_entities = sum(entity_counts) / len(entity_counts) if entity_counts else 0
        success_rate = ((example_count - failed_count) / example_count) * 100
        print(f'  📈 Generated: {example_count} examples')
        print(f'  📉 Failed: {failed_count} examples') 
        print(f'  📊 Avg entities per example: {avg_entities:.1f}')
        print(f'  ✅ Success rate: {success_rate:.1f}%')
        print()
        
        total_examples += example_count
        total_failed += failed_count

    overall_success = ((total_examples - total_failed) / total_examples) * 100
    print(f'📊 OVERALL RESULTS:')
    print(f'Total examples: {total_examples}')
    print(f'Total failed: {total_failed}')
    print(f'Overall success rate: {overall_success:.1f}%')

if __name__ == "__main__":
    test_entity_detection()
