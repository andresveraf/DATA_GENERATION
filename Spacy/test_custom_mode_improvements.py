#!/usr/bin/env python3
"""
Test script to verify the enhanced entity detection in generate_example_with_custom_mode
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_generation_noisy import generate_example_with_custom_mode

def test_custom_modes_for_failed_spans():
    """Test all custom modes to check for failed spans"""
    
    modes = ["personal_id", "address_focused", "contact_only", "financial_heavy", "table_heavy", "minimal_entities"]
    countries = ["chile", "mexico", "brazil", "uruguay"]
    
    print("Testing custom modes for failed spans...")
    print("=" * 60)
    
    total_failed = 0
    total_tested = 0
    
    for mode in modes:
        print(f"\nTesting mode: {mode}")
        print("-" * 40)
        
        mode_failed = 0
        mode_tested = 0
        
        for country in countries:
            for i in range(10):  # Test 10 examples per country per mode
                try:
                    sentence, annotations = generate_example_with_custom_mode(
                        country=country, 
                        mode=mode, 
                        include_noise=False
                    )
                    
                    entities = annotations.get("entities", [])
                    mode_tested += 1
                    total_tested += 1
                    
                    # Check for failed spans
                    failed_spans = []
                    for start, end, label in entities:
                        if start >= end or start < 0 or end > len(sentence):
                            failed_spans.append((start, end, label))
                            continue
                        
                        entity_text = sentence[start:end]
                        if not entity_text.strip():
                            failed_spans.append((start, end, label, "empty_text"))
                    
                    if failed_spans:
                        mode_failed += 1
                        total_failed += 1
                        print(f"  FAILED {country} #{i+1}: {failed_spans}")
                        print(f"    Sentence: {sentence}")
                        print(f"    Entities: {entities}")
                    
                except Exception as e:
                    print(f"  ERROR {country} #{i+1}: {e}")
                    mode_failed += 1
                    total_failed += 1
        
        success_rate = ((mode_tested - mode_failed) / mode_tested * 100) if mode_tested > 0 else 0
        print(f"  Mode {mode}: {mode_tested - mode_failed}/{mode_tested} successful ({success_rate:.1f}%)")
    
    overall_success_rate = ((total_tested - total_failed) / total_tested * 100) if total_tested > 0 else 0
    print(f"\n" + "=" * 60)
    print(f"OVERALL RESULTS:")
    print(f"Total tested: {total_tested}")
    print(f"Total failed: {total_failed}")
    print(f"Success rate: {overall_success_rate:.1f}%")
    print(f"Failed spans rate: {(total_failed / total_tested * 100):.1f}%")
    
    return total_failed == 0

def test_specific_mode_examples():
    """Test specific examples from different modes"""
    
    print("\n" + "=" * 60)
    print("TESTING SPECIFIC EXAMPLES")
    print("=" * 60)
    
    test_cases = [
        ("personal_id", "chile"),
        ("address_focused", "mexico"), 
        ("contact_only", "brazil"),
        ("financial_heavy", "uruguay"),
        ("table_heavy", "chile"),
        ("minimal_entities", "mexico")
    ]
    
    for mode, country in test_cases:
        print(f"\nTesting {mode} with {country}:")
        print("-" * 40)
        
        for i in range(3):
            sentence, annotations = generate_example_with_custom_mode(
                country=country, 
                mode=mode, 
                include_noise=False
            )
            
            entities = annotations.get("entities", [])
            
            print(f"Example {i+1}:")
            print(f"  Sentence: {sentence}")
            print(f"  Entities: {len(entities)} found")
            
            for start, end, label in entities:
                entity_text = sentence[start:end]
                print(f"    {label}: '{entity_text}' [{start}:{end}]")
            
            # Check for any issues
            issues = []
            for start, end, label in entities:
                if start >= end:
                    issues.append(f"Invalid span: {start} >= {end}")
                elif start < 0 or end > len(sentence):
                    issues.append(f"Out of bounds: {start}:{end} for text length {len(sentence)}")
                else:
                    entity_text = sentence[start:end]
                    if not entity_text.strip():
                        issues.append(f"Empty entity text at {start}:{end}")
            
            if issues:
                print(f"    ISSUES: {issues}")
            else:
                print(f"    STATUS: OK")

if __name__ == "__main__":
    print("Testing Enhanced Custom Mode Entity Detection")
    print("=" * 60)
    
    # Test for failed spans
    success = test_custom_modes_for_failed_spans()
    
    # Test specific examples  
    test_specific_mode_examples()
    
    print(f"\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED - No failed spans detected!")
    else:
        print("❌ SOME TESTS FAILED - Failed spans still present")
    print("=" * 60)
