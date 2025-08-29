#!/usr/bin/env python3

# Quick test of the enhanced templates
try:
    from data_generation_noisy import get_sentence_templates
    
    countries = ['chile', 'mexico', 'brazil', 'uruguay']
    
    print("Enhanced Template Counts:")
    print("-" * 30)
    
    for country in countries:
        templates = get_sentence_templates(country)
        print(f"{country.capitalize()}: {len(templates)} templates")
        
        # Show first template as sample
        if templates:
            print(f"  Sample: {templates[0][:60]}...")
        print()
    
    print("✅ Template enhancement successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
