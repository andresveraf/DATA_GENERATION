#!/usr/bin/env python3
"""
Test script for Excel export functionality
"""
import data_generation_noisy as dg
import pandas as pd
import os

def test_excel_exports():
    """Test all Excel export functions"""
    print("🧪 Testing Excel Export Functions")
    print("=" * 50)
    
    # Test 1: Chilean Excel export
    print("\n📊 Test 1: Chilean Excel Export")
    try:
        dg.export_chilean_data_to_excel_with_noise(
            n_examples=5, 
            output_file='test_chilean_export.xlsx',
            include_noise=True,
            noise_level=0.2
        )
        print("✅ Chilean Excel export successful!")
        
        # Check if file was created
        if os.path.exists('test_chilean_export.xlsx'):
            print(f"📁 File created: test_chilean_export.xlsx ({os.path.getsize('test_chilean_export.xlsx')} bytes)")
        else:
            print("❌ File was not created")
            
    except Exception as e:
        print(f"❌ Chilean Excel export failed: {e}")
    
    # Test 2: Multi-country Excel export
    print("\n🌎 Test 2: Multi-Country Excel Export")
    try:
        dg.export_multi_country_data_to_excel_with_noise(
            n_examples=20,  # 5 per country
            output_file='test_multi_country_export.xlsx',
            include_noise=True,
            noise_level=0.15
        )
        print("✅ Multi-country Excel export successful!")
        
        # Check if file was created
        if os.path.exists('test_multi_country_export.xlsx'):
            print(f"📁 File created: test_multi_country_export.xlsx ({os.path.getsize('test_multi_country_export.xlsx')} bytes)")
        else:
            print("❌ File was not created")
            
    except Exception as e:
        print(f"❌ Multi-country Excel export failed: {e}")
    
    # Test 3: Single country Excel export (Mexico)
    print("\n🇲🇽 Test 3: Mexico Excel Export")
    try:
        dg.export_country_data_to_excel_with_noise(
            country='mexico',
            n_examples=5,
            output_file='test_mexico_export.xlsx',
            include_noise=True,
            noise_level=0.1
        )
        print("✅ Mexico Excel export successful!")
        
        # Check if file was created
        if os.path.exists('test_mexico_export.xlsx'):
            print(f"📁 File created: test_mexico_export.xlsx ({os.path.getsize('test_mexico_export.xlsx')} bytes)")
        else:
            print("❌ File was not created")
            
    except Exception as e:
        print(f"❌ Mexico Excel export failed: {e}")
    
    # Test 4: Verify Excel content
    print("\n🔍 Test 4: Verify Excel Content")
    try:
        if os.path.exists('test_chilean_export.xlsx'):
            # Read the Excel file to verify structure
            sheets = pd.read_excel('test_chilean_export.xlsx', sheet_name=None)
            print(f"📋 Sheets found: {list(sheets.keys())}")
            
            # Check All_Data sheet
            if 'All_Data' in sheets:
                all_data = sheets['All_Data']
                print(f"📊 All_Data sheet: {len(all_data)} rows, {len(all_data.columns)} columns")
                print(f"📝 Columns: {list(all_data.columns)}")
                
                # Show sample data
                if len(all_data) > 0:
                    print(f"🔍 Sample text: {all_data.iloc[0]['Generated_Text'][:100]}...")
                    print(f"🏷️  Sample entities: {all_data.iloc[0]['Entities'][:150]}...")
            
    except Exception as e:
        print(f"❌ Excel verification failed: {e}")
    
    print("\n✅ Excel Export Tests Completed!")
    print("\n📋 Summary of Excel Export Features:")
    print("   • export_chilean_data_to_excel_with_noise() - Chilean-specific data with modes")
    print("   • export_multi_country_data_to_excel_with_noise() - All countries balanced")
    print("   • export_country_data_to_excel_with_noise() - Single country specific")
    print("\n📖 Each Excel file contains comprehensive sheets:")
    print("   • Summary - Overview statistics")
    print("   • All_Data - Complete generated data with entities")
    print("   • By_Country/By_Mode - Breakdown analysis")
    print("   • Name_Analysis - Naming pattern analysis")
    print("   • Entity_Statistics - Entity type distribution")
    print("   • Noise_Analysis - Noise pattern detection (if enabled)")

if __name__ == "__main__":
    test_excel_exports()
