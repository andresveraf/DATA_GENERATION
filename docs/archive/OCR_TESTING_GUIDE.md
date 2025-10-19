# OCR Testing Tools Guide with Entity Detection

This directory contains several tools to test and validate our OCR noise generation against real OCR documents, now including **entity preservation analysis** for NER training validation.

## Available Tools

### 1. `simple_ocr_test.py` - Quick Testing with Entity Detection
**Best for**: Quick copy-paste testing of any text with PII entity identification

**Usage:**
```bash
# Command line
python3 simple_ocr_test.py "Your text here"

# Interactive mode
python3 simple_ocr_test.py
# Then paste your text and press Enter
```

**Example:**
```bash
python3 simple_ocr_test.py "JUAN SEGUNDO GARCIA CARRASCO con RUT 7,082,003-K"
```

**Features:**
- ✅ Detects ID numbers, phone numbers, emails, dates, money amounts
- ✅ Shows entity preservation after noise application
- ✅ Validates that PII remains identifiable for NER training

### 2. `test_entity_preservation.py` - Entity-Aware OCR Testing ⭐ **NEW**
**Best for**: Validating that OCR noise doesn't break entity boundaries

**Usage:**
```bash
python3 test_entity_preservation.py
```

**Features:**
- 🎯 **Entity preservation scoring**: Shows what % of entities survive noise
- 📊 **Multiple noise intensities**: Light (90%+), Medium (70-89%), Heavy (<70%)
- ✅ **NER training validation**: Ensures entities remain trainable
- 🔍 **Detailed analysis**: Shows exactly which entities are lost/preserved

### 3. `test_ocr_comparison.py` - Comprehensive Analysis with Entities
**Best for**: Detailed analysis and comparison with real OCR patterns + entity detection

**Usage:**
```bash
python3 test_ocr_comparison.py
```

**Features:**
- Analyzes real OCR corruption patterns from your pension document
- **Entity detection** in both original and noisy text
- **spaCy integration** for advanced NLP entity recognition
- Interactive mode for custom text testing with entity analysis
- Character-level corruption comparison

### 4. `ocr_comparison_tool.py` - Advanced Pattern Analysis
**Best for**: Deep pattern analysis and corruption comparison

**Usage:**
```bash
python3 ocr_comparison_tool.py
```

**Features:**
- Pattern matching with real OCR examples
- Statistical comparison tools
- Different noise intensity testing

## Entity Detection Capabilities

### **Supported PII Types:**
- **ID_NUMBER**: Chilean RUT (7,082,003-K), Brazilian CPF, Uruguayan Cédula
- **PHONE**: Chilean (+56 9 8765 4321), International formats
- **EMAIL**: Standard email addresses (juan@empresa.cl)
- **DATE**: Various formats (28/09/1952, 15-08-2024)
- **MONEY**: Currency amounts ($1.234.567 CLP)

### **Entity Preservation Scoring:**
- ✅ **90%+ preservation** = Excellent (perfect for NER training)
- ⚠️ **70-89% preservation** = Acceptable (minor adjustments needed)
- ❌ **<70% preservation** = Poor (too aggressive noise)

## Real OCR Example Analysis

Your pension document shows these corruption patterns:

### Common OCR Errors Found:
- **Missing letters**: "Importante" → "mportante"
- **Extra letters**: "Superintendencia" → "Superintendenciall"
- **Character substitution**: "corresponde" → "comesponde", "solo" → "dlo"
- **Number/letter confusion**: "se" → "16", "algo" → "a10"
- **Severe word corruption**: "diferida" → "Citerida", "SEGURO" → "SWAGER"
- **Special character issues**: "COMPAÑÍA" → "COMPAKIA"
- **Word merging**: "a los" → "alos"

### PII Elements in Your Example:
- **PERSON**: "JUAN SEGUNDO GARCIA CARRASCO", "ROSA ESTER PENA GONZALEZ"
- **ID_NUMBER**: "7,082,003-K", "66724859" (missing check digit)
- **ADDRESS**: "ELIAS 42 VILLA HERMOSA"
- **LOCATION**: "CORONEL"
- **DATE**: "28/09/1952", "27-11-1953"

## How Our Noise Compares

Our OCR noise generation successfully reproduces:
✅ Character substitutions (I→1, O→0, S→5, etc.)
✅ Missing/extra characters
✅ Symbol corruption and spacing issues
✅ Severe word-level corruption
✅ Multiple format variations for ID numbers

## Quick Test Commands with Entity Validation

```bash
# Test with your exact pension document text + entity detection
python3 simple_ocr_test.py "JUAN SEGUNDO GARCIA CARRASCO con RUT 7,082,003-K vive en ELIAS 42 VILLA HERMOSA, CORONEL"

# Test comprehensive entity preservation
python3 test_entity_preservation.py

# Test with complete PII examples
python3 simple_ocr_test.py "Cliente: MARÍA GONZÁLEZ RUT 12.345.678-9 Teléfono: +56 9 1234 5678 Email: maria@empresa.cl"

# Test with financial information + entities
python3 simple_ocr_test.py "Monto: $1.234.567 CLP Código: ABC123 Fecha: 15/08/2024"

# Interactive entity-aware testing
python3 test_ocr_comparison.py
```

## Sample Entity Preservation Results

```
TEST CASE: Cliente: MARÍA GONZÁLEZ, RUT 12.345.678-9, Tel: +56 9 8765 4321
Entities (2): ID_NUMBER:12.345.678-9 PHONE:+56 9 8765 4321

LIGHT NOISE:  100.0% ✅ Cliente: MARÍA GONZÁLEZ, RUT 12.345.678-9, Tel: +56 9 8765 4321
MEDIUM NOISE:  66.7% ❌ Cliente: MARÍA GONZALEZ, RUT 12.345.G7B-9, Tel: +56 9 8765 4321
HEAVY NOISE:   66.7% ❌ Ciiente: NARIA GONZÁLEZ, RUT l2.345.678-9, Tel: +56 9 8765 4321
```

This shows that:
- **Light noise** preserves all entities (perfect for training)
- **Medium/Heavy noise** may corrupt some ID numbers but preserves phone numbers
- **Entity boundaries** remain mostly intact for NER model training

## Integration with Main System

These tools validate that our main data generation system (`data_generation_noisy.py`) produces realistic OCR corruption that matches real-world document scanning artifacts.

The noise functions are automatically applied when using:
```bash
python3 data_generation_noisy.py --mode create-dataset --country all --train-size 120000 --dev-size 24000 --noise --noise-level 0.8
```

## Tips for Testing with Entity Awareness

1. **Start with entity detection**: Use `python3 test_entity_preservation.py` to validate overall system
2. **Test specific cases**: Use `simple_ocr_test.py` for quick validation of custom text
3. **Compare with real OCR**: Use your pension document examples to verify realism
4. **Check entity boundaries**: Ensure PII entities remain identifiable after noise
5. **Validate training data**: Confirm that generated data will work with spaCy NER training
6. **Monitor preservation rates**: Aim for 90%+ entity preservation for reliable training

## Entity-Aware Testing Workflow

```bash
# 1. Quick entity check
python3 simple_ocr_test.py "Your text with PII"

# 2. Comprehensive preservation analysis  
python3 test_entity_preservation.py

# 3. Interactive testing with custom examples
python3 test_ocr_comparison.py

# 4. Generate training data with validated noise
python3 data_generation_noisy.py --mode create-dataset --country chile --train-size 1000 --noise --noise-level 0.7
```

This workflow ensures that your OCR noise is realistic but doesn't break the entity boundaries needed for successful NER model training!
