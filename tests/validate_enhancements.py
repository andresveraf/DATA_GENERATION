#!/usr/bin/env python3
"""
Quick validation script for enhanced PII generation
==================================================

This script validates the enhanced PII generation capabilities without requiring
external dependencies like NLTK or spaCy models.

Author: Andrés Vera Figueroa (Enhanced by Codegen)
Date: October 2024
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_enhanced_pii_generator():
    """Test enhanced PII generator functionality"""
    print("🧪 Testing Enhanced PII Generator...")
    
    try:
        from generators.enhanced_pii_generator import EnhancedPIIGenerator
        
        generator = EnhancedPIIGenerator()
        countries = ['chile', 'mexico', 'brazil', 'uruguay']
        
        print("✅ Enhanced PII Generator imported successfully")
        
        # Test new PII types
        for country in countries:
            print(f"\n--- Testing {country.upper()} ---")
            
            direction = generator.generate_direction(country)
            location = generator.generate_location(country)
            postal_code = generator.generate_postal_code(country)
            region = generator.generate_region(country)
            
            print(f"  Direction: {direction}")
            print(f"  Location: {location}")
            print(f"  Postal Code: {postal_code}")
            print(f"  Region: {region}")
            
            # Test enhanced existing types
            phone = generator.generate_enhanced_phone(country)
            sequence = generator.generate_enhanced_sequence()
            date = generator.generate_enhanced_date(country)
            
            print(f"  Enhanced Phone: {phone}")
            print(f"  Enhanced Sequence: {sequence}")
            print(f"  Enhanced Date: {date}")
        
        print("\n✅ All enhanced PII types generated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced PII generator: {e}")
        return False

def test_nlp_augmentation():
    """Test NLP augmentation functionality (basic test without NLTK)"""
    print("\n🧪 Testing NLP Augmentation (Basic)...")
    
    try:
        from augmentation.nlp_augmentation import AugmentationConfig, EntitySpan
        
        # Test configuration
        config = AugmentationConfig(
            synonym_replacement_rate=0.3,
            noise_injection_rate=0.1,
            language="es"
        )
        
        print("✅ Augmentation configuration created successfully")
        print(f"  Language: {config.language}")
        print(f"  Synonym rate: {config.synonym_replacement_rate}")
        print(f"  Noise rate: {config.noise_injection_rate}")
        
        # Test entity span
        entity = EntitySpan(
            start=10,
            end=20,
            label="CUSTOMER_NAME",
            text="Juan Pérez"
        )
        
        print(f"✅ Entity span created: {entity.text} ({entity.label})")
        
        print("\n✅ Basic NLP augmentation components work!")
        print("ℹ️  Full augmentation requires NLTK installation")
        return True
        
    except Exception as e:
        print(f"❌ Error testing NLP augmentation: {e}")
        return False

def test_project_structure():
    """Test project structure and file organization"""
    print("\n🧪 Testing Project Structure...")
    
    required_dirs = [
        'generators',
        'augmentation', 
        'data_spacy',
        'data_transformers',
        'tests',
        'Spacy',
        'Transformers',
        'database',
        'corruption',
        'dataset_composer'
    ]
    
    required_files = [
        'main_pipeline.py',
        'requirements.txt',
        'README.md',
        'generators/enhanced_pii_generator.py',
        'augmentation/nlp_augmentation.py',
        'data_spacy/README.md',
        'data_transformers/README.md',
        'tests/test_enhanced_pii.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    # Check directories
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
        else:
            print(f"✅ Directory exists: {dir_name}/")
    
    # Check files
    for file_name in required_files:
        if not Path(file_name).exists():
            missing_files.append(file_name)
        else:
            print(f"✅ File exists: {file_name}")
    
    if missing_dirs:
        print(f"❌ Missing directories: {', '.join(missing_dirs)}")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
    
    if not missing_dirs and not missing_files:
        print("\n✅ All required project structure components are present!")
        return True
    else:
        return False

def test_data_variety():
    """Test data variety in enhanced PII generation"""
    print("\n🧪 Testing Data Variety...")
    
    try:
        from generators.enhanced_pii_generator import EnhancedPIIGenerator
        
        generator = EnhancedPIIGenerator()
        
        # Test variety in sequences
        sequences = set()
        for _ in range(20):
            seq = generator.generate_enhanced_sequence()
            sequences.add(seq)
        
        variety_percentage = (len(sequences) / 20) * 100
        print(f"✅ Sequence variety: {len(sequences)}/20 unique ({variety_percentage:.1f}%)")
        
        # Test variety in phone numbers
        phones = set()
        for _ in range(10):
            phone = generator.generate_enhanced_phone('chile')
            phones.add(phone)
        
        phone_variety = (len(phones) / 10) * 100
        print(f"✅ Phone variety: {len(phones)}/10 unique ({phone_variety:.1f}%)")
        
        # Test variety in dates
        dates = set()
        for _ in range(15):
            date = generator.generate_enhanced_date('chile')
            dates.add(date)
        
        date_variety = (len(dates) / 15) * 100
        print(f"✅ Date variety: {len(dates)}/15 unique ({date_variety:.1f}%)")
        
        if variety_percentage > 70 and phone_variety > 50 and date_variety > 60:
            print("\n✅ Sufficient variety in generated data!")
            return True
        else:
            print("\n⚠️  Some variety metrics below expected thresholds")
            return False
            
    except Exception as e:
        print(f"❌ Error testing data variety: {e}")
        return False

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("🚀 ENHANCED PII GENERATION VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Enhanced PII Generator", test_enhanced_pii_generator),
        ("NLP Augmentation", test_nlp_augmentation),
        ("Data Variety", test_data_variety)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All validation tests passed! The enhanced PII generation system is ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the issues above.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
