#!/usr/bin/env python3
"""
Comprehensive test of the improved custom mode function
"""

from data_generation_noisy import generate_example_with_custom_mode

def test_all_modes():
    """Test all custom modes for failed spans"""
    
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
            for i in range(5):  # Test 5 examples per country per mode
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
                    else:
                        print(f"  OK {country} #{i+1}: {len(entities)} entities")
                    
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

if __name__ == "__main__":
    print("Testing Enhanced Custom Mode Entity Detection")
    print("=" * 60)
    
    # Test for failed spans
    success = test_all_modes()
    
    print(f"\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED - No failed spans detected!")
    else:
        print("❌ SOME TESTS FAILED - Failed spans still present")
    print("=" * 60)