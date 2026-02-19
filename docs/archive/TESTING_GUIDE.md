# Testing Guide: OCR Noise & Entity Preservation

## 🎯 Purpose

This guide provides comprehensive testing procedures for validating OCR noise generation and entity preservation in the PII data generation system.

---

## 📋 Table of Contents

1. [Testing Tools Overview](#testing-tools-overview)
2. [Entity Preservation Testing](#entity-preservation-testing)
3. [OCR Noise Validation](#ocr-noise-validation)
4. [Quick Test Commands](#quick-test-commands)
5. [Expected Results](#expected-results)
6. [Troubleshooting](#troubleshooting)

---

## 🛠️ Testing Tools Overview

### Available Testing Scripts

| Tool | Purpose | Best For |
|------|---------|----------|
| `simple_ocr_test.py` | Quick copy-paste testing | Fast validation of custom text |
| `test_entity_preservation.py` | Entity-aware validation | Comprehensive preservation analysis |
| `test_ocr_comparison.py` | Detailed OCR analysis | Pattern comparison with real OCR |
| `ocr_comparison_tool.py` | Advanced pattern analysis | Deep statistical analysis |

### Supported PII Entity Types

- **ID_NUMBER**: Chilean RUT, Brazilian CPF, Uruguayan Cédula
- **PHONE**: Country-specific formats (+56, +52, +55, +598)
- **EMAIL**: Standard email addresses
- **DATE**: Various date formats
- **MONEY**: Currency amounts (CLP, MXN, BRL, UYU)
- **ADDRESS**: Street addresses
- **CUSTOMER_NAME**: Person names

---

## 🎯 Entity Preservation Testing

### Understanding Preservation Scores

**Preservation Levels**:
- ✅ **90%+**: Excellent (perfect for NER training)
- ⚠️ **70-89%**: Acceptable (minor adjustments needed)
- ❌ **<70%**: Poor (too aggressive noise)

### Test Entity Preservation

```bash
# Run comprehensive entity preservation test
python Spacy/test_entity_preservation.py
```

**Expected Output**:
```
TEST CASE: Cliente: MARÍA GONZÁLEZ, RUT 12.345.678-9, Tel: +56 9 8765 4321
Entities (3): ID_NUMBER, PHONE, CUSTOMER_NAME

LIGHT NOISE:  100.0% ✅ All entities preserved
MEDIUM NOISE:  66.7% ⚠️ 2/3 entities preserved
HEAVY NOISE:   66.7% ⚠️ 2/3 entities preserved
```

### Entity-Specific Testing

**Test Email Preservation** (Critical - @ symbols):
```bash
python3 Spacy/simple_ocr_test.py "Contact: juan.perez@empresa.cl"
```

**Test Sequence Numbers** (Critical - number corruption):
```bash
python3 Spacy/simple_ocr_test.py "Referencia: REF-12345, FOLIO-67890"
```

**Test ID Numbers** (Critical - format preservation):
```bash
python3 Spacy/simple_ocr_test.py "RUT: 12.345.678-9"
```

---

## 🔍 OCR Noise Validation

### Real OCR Pattern Examples

Your pension document shows these corruption patterns:

#### Common OCR Errors:
- **Missing letters**: "Importante" → "mportante"
- **Extra letters**: "Superintendencia" → "Superintendenciall"
- **Character substitution**: "corresponde" → "comesponde"
- **Number/letter confusion**: "se" → "16", "algo" → "a10"
- **Severe word corruption**: "diferida" → "Citerida"
- **Special character issues**: "COMPAÑÍA" → "COMPAKIA"

### Validate Against Real OCR

```bash
# Test with your exact pension document text
python3 Spacy/simple_ocr_test.py "JUAN SEGUNDO GARCIA CARRASCO con RUT 7,082,003-K vive en ELIAS 42 VILLA HERMOSA, CORONEL"
```

**Expected Behavior**:
- ✅ Character substitutions (I→1, O→0, S→5)
- ✅ Missing/extra characters
- ✅ Symbol corruption (@, +, -)
- ✅ Format variations maintained

---

## 🚀 Quick Test Commands

### Step 1: Quick Validation (1,000 examples)

```bash
python Spacy/data_generation_noisy.py \
  --mode create-dataset \
  --country all \
  --train-size 800 \
  --dev-size 200 \
  --noise \
  --noise-level 0.6
```

### Step 2: Check Results

```bash
# View failed spans statistics
cat Spacy/output/multi_country_dataset_stats_noisy_1000.json | grep "failed_spans"
```

**Expected Results** (after optimization):
- Failed Spans: 80-120 (8-12% rate)
- Entity Preservation: >90%
- Zero E1010 errors

### Step 3: Generate Production Dataset

```bash
# If test shows improvement, generate full dataset
python Spacy/data_generation_noisy.py \
  --mode create-dataset \
  --country all \
  --train-size 200000 \
  --dev-size 50000 \
  --noise \
  --noise-level 0.6
```

---

## 📊 Expected Results

### Noise Level Comparison

| Noise Type | Before Optimization | After Optimization | Improvement |
|------------|-------------------|-------------------|-------------|
| OCR Character Noise | 70% | 25% | **-45%** |
| Multiple Noise Types | 50% | 30% | **-20%** |
| Severe Corruption | 15% | 5% | **-10%** |
| Symbol Corruption | 30% | 15% | **-15%** |
| **Failed Spans Rate** | **31.7%** | **3.2%** | **89% reduction** |

### Entity-Specific Performance

| Entity Type | Preservation Rate | Status |
|-------------|-------------------|--------|
| CUSTOMER_NAME | 99.5% | ✅ Excellent |
| ADDRESS | 99.2% | ✅ Excellent |
| PHONE_NUMBER | 98.8% | ✅ Excellent |
| AMOUNT | 98.5% | ✅ Excellent |
| EMAIL | 96.0% | ✅ Good |
| ID_NUMBER | 96.5% | ✅ Good |
| SEQ_NUMBER | 95.0% | ✅ Good |

---

## 🧪 Complete Testing Workflow

### Phase 1: Entity Validation

```bash
# 1. Quick entity check
python3 Spacy/simple_ocr_test.py "Cliente: MARÍA GONZÁLEZ, RUT 12.345.678-9"

# 2. Comprehensive preservation analysis
python3 Spacy/test_entity_preservation.py

# 3. Interactive testing with custom examples
python3 Spacy/test_ocr_comparison.py
```

### Phase 2: Dataset Generation Testing

```bash
# Generate small test dataset
python Spacy/data_generation_noisy.py \
  --mode create-dataset \
  --country chile \
  --train-size 1000 \
  --noise \
  --noise-level 0.6

# Check statistics
python -c "
import json
with open('Spacy/output/multi_country_dataset_stats_noisy_1000.json') as f:
    stats = json.load(f)
    print(f\"Failed Spans: {stats['failed_spans']}\")
    print(f\"Failed Rate: {stats['failed_spans']/stats['total_examples']*100:.1f}%\")
"
```

### Phase 3: Production Validation

```bash
# Generate production dataset
python Spacy/data_generation_noisy.py \
  --mode create-dataset \
  --country all \
  --train-size 200000 \
  --dev-size 50000 \
  --noise \
  --noise-level 0.6

# Validate entity distribution
python -c "
import json
with open('Spacy/output/multi_country_dataset_stats_noisy_250000.json') as f:
    stats = json.load(f)
    for entity, count in stats['entity_distribution'].items():
        print(f'{entity}: {count}')
"
```

---

## 🔧 Troubleshooting

### Issue 1: High Failed Spans Rate (>10%)

**Symptoms**: Failed spans >10% in generated data

**Solutions**:
```bash
# Reduce noise level
--noise-level 0.4  # From default 0.6

# Or disable specific noise types
--disable-ocr-noise
--disable-severe-corruption
```

### Issue 2: Low Entity Preservation (<80%)

**Symptoms**: Entities not preserved after noise

**Diagnosis**:
```bash
# Test entity preservation specifically
python3 Spacy/test_entity_preservation.py

# Check which entities are failing
# Look for preservation <70% for any entity type
```

**Solutions**:
```python
# Adjust entity-specific noise levels in code:
ENTITY_SAFE_NOISE_LEVELS = {
    'EMAIL': 0.03,        # Reduce further
    'SEQ_NUMBER': 0.02,   # Reduce further
    # ... other entities
}
```

### Issue 3: Unrealistic OCR Patterns

**Symptoms**: Generated noise doesn't match real OCR

**Validation**:
```bash
# Compare with real OCR examples
python3 Spacy/simple_ocr_test.py "Your real OCR text here"

# Check if patterns match
# - Character substitutions (I→1, O→0)
# - Missing/extra letters
# - Symbol corruption
```

**Solutions**:
- Adjust substitution probabilities
- Add new corruption patterns
- Test against more real OCR samples

### Issue 4: Memory Issues with Large Datasets

**Symptoms**: Out of memory errors during generation

**Solutions**:
```bash
# Generate in batches
python Spacy/data_generation_noisy.py \
  --mode create-dataset \
  --train-size 50000  # Multiple runs

# Enable garbage collection
--gc-enabled

# Reduce batch size
--batch-size 1000
```

---

## 📈 Performance Metrics

### Target Metrics (After Optimization)

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| Failed Spans Rate | <5% | 5-10% | >10% |
| Entity Preservation | >95% | 90-95% | <90% |
| E1010 Errors | 0 | 0 | >0 |
| Pattern Variety | >85% | 70-85% | <70% |

### Quality Checks

```python
# Run quality validation
from generators.enhanced_pii_generator import validate_pii_variety

variety_report = validate_pii_variety('chile', samples=1000)

for pii_type, metrics in variety_report.items():
    status = "✅" if metrics['sufficient_variety'] else "❌"
    print(f"{status} {pii_type}: {metrics['variety_percentage']:.1f}% variety")
```

---

## 🎓 Best Practices

### For Consistent Testing

1. **Always test with small datasets first**
   - 1,000 examples for quick validation
   - Check failed spans rate
   - Verify entity preservation

2. **Monitor entity distribution**
   - Ensure all entity types represented
   - Check for imbalances (>5% variance)
   - Validate country-specific formats

3. **Validate against real OCR**
   - Test with actual scanned documents
   - Compare corruption patterns
   - Adjust noise parameters to match

4. **Document test results**
   - Keep statistics for each run
   - Track improvements over time
   - Note what works/doesn't work

### For Production Deployment

1. **Final validation before scaling**
   - Test with 10K examples
   - Validate <5% failed spans
   - Check all entity types >90% preservation

2. **Generate in chunks**
   - 50K per generation run
   - Validate each chunk
   - Merge validated chunks

3. **Continuous monitoring**
   - Track failed spans rate
   - Monitor entity distribution
   - Alert on degradation

---

## 📚 Related Documentation

- **Optimization Journey**: See `OPTIMIZATION_JOURNEY_2025.md` for complete optimization history
- **System Overview**: See root `README.md` for complete system documentation
- **Code Implementation**: See `Spacy/data_generation_noisy.py` for noise generation code

---

**Last Updated**: October 2025
**Maintained By**: Andrés Vera Figueroa
**Status**: ✅ Production-Ready Testing Procedures