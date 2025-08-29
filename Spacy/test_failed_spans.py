#!/usr/bin/env python3
"""
Test the improved entity detection with a larger dataset to measure failed spans reduction
"""

from data_generation_noisy import generate_example_with_noise, generate_example_with_custom_mode
import random

def test_failed_spans_improvement():
    print('🧪 Testing failed spans improvement with larger dataset...')
    print()
    
    # Generate a moderate-sized test dataset
    test_size = 500  # Manageable size for testing
    countries = ['chile', 'mexico', 'brazil', 'uruguay']
    
    print(f'📊 Generating {test_size} examples to test failed spans...')
    
    total_examples = 0
    total_failed_spans = 0
    total_entities = 0
    examples_with_failed_spans = 0
    
    try:
        for country in countries:
            country_examples = test_size // 4  # Distribute evenly across countries
            country_failed = 0
            country_entities = 0
            
            print(f'🌎 Testing {country.upper()}: {country_examples} examples')
            
            for i in range(country_examples):
                # Use different generation modes for variety
                modes = ['full', 'personal_id', 'address_focused', 'contact_only', 'minimal_entities']
                mode = random.choice(modes)
                
                sentence, annotations = generate_example_with_custom_mode(
                    country=country,
                    mode=mode,
                    include_noise=False,
                    noise_level=0.0
                )
                
                entities = annotations['entities']
                entity_count = len(entities)
                total_entities += entity_count
                
                # Count as failed if too few entities detected
                # Expected minimum varies by mode
                expected_min = {
                    'full': 6,
                    'personal_id': 2,
                    'address_focused': 3,
                    'contact_only': 3,
                    'minimal_entities': 3,
                    'financial_heavy': 3,
                    'table_heavy': 4
                }
                
                min_expected = expected_min.get(mode, 3)
                if entity_count < min_expected:
                    country_failed += 1
                    examples_with_failed_spans += 1
                
                total_examples += 1
                
            print(f'  ✅ Generated {country_examples} examples')
            print(f'  📊 Failed: {country_failed} examples')
        
        # Calculate overall statistics
        if total_examples > 0:
            failed_span_rate = (examples_with_failed_spans / total_examples) * 100
            avg_entities = total_entities / total_examples
            
            print(f'\n📈 RESULTS:')
            print(f'Total examples generated: {total_examples:,}')
            print(f'Examples with insufficient entities: {examples_with_failed_spans:,}')
            print(f'Failed span rate: {failed_span_rate:.1f}%')
            print(f'Average entities per example: {avg_entities:.1f}')
            print()
            
            if failed_span_rate < 20.0:
                print('✅ EXCELLENT: Failed span rate under 20%!')
            elif failed_span_rate < 30.0:
                print('✅ VERY GOOD: Failed span rate under 30%!')
            elif failed_span_rate < 35.0:
                print('✅ GOOD: Failed span rate under 35%!')
            elif failed_span_rate < 40.0:
                print('🟡 ACCEPTABLE: Failed span rate under 40%')
            else:
                print('🔴 NEEDS IMPROVEMENT: Failed span rate over 40%')
                
            print(f'🎯 Improvement target: Previous rate was ~36%, aiming for <30%')
            
            # Additional analysis
            print(f'\n📊 ANALYSIS:')
            if failed_span_rate < 30:
                improvement = 36 - failed_span_rate
                print(f'🎉 SUCCESS: Improved by {improvement:.1f} percentage points!')
                print(f'🔥 Entity detection improvements are working effectively!')
            else:
                print(f'🔧 Room for further improvement in entity detection logic')
                
        else:
            print('❌ No examples generated')
            
    except Exception as e:
        print(f'❌ Error during generation: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_failed_spans_improvement()
