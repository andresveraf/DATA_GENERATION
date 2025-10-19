# Failed Spans Analysis and Reduction Strategy

## 🚨 Current Problem Analysis

**Your Statistics:**
- Total Examples: 120,000
- Failed Spans: 26,807
- **Failure Rate: 22.3%**
- Overlap Errors: 0 (Good!)

## 🔍 Step-by-Step Root Cause Analysis

### **1. Primary Causes of Failed Spans**

#### **A. Aggressive Noise Generation (70% probability)**
```python
# Current noise function applies OCR corruption at 70% rate
if random.random() < 0.7:  # TOO AGGRESSIVE
    # Character substitutions that break entity boundaries
    'ó': 'o', 'JOSÉ': 'JOSE'  # Name corruption
    'RUT: 12.345.678-9' -> 'BUT: 12.345.678-9'  # ID corruption
```

#### **B. Character-Level Span Alignment Issues**
```python
# When noise corrupts text, spaCy char_span fails:
span = doc.char_span(start, end, label=label, alignment_mode="contract")
if span is not None:  # <-- This fails when noise corrupts boundaries
```

#### **C. Entity Boundary Corruption**
- **Accented characters removed**: "José" → "Jose" (span mismatch)
- **OCR substitutions**: "RUT" → "BUT", "EMAIL" → "EMALL"
- **Case changes**: "CUSTOMER" → "customer" (span offset)

### **2. Specific Problem Areas**

#### **High-Risk Entity Types for Failure:**
1. **CUSTOMER_NAME** (105,953 → likely ~23K failed)
   - Accented names: José, María, etc.
   - Compound names: Juan Carlos, Ana María
   - Noise corrupts name boundaries

2. **EMAIL** (105,179 → likely ~22K failed)  
   - @ symbol corruption: @ → a, @ → o
   - Domain corruption: .com → .corn, .cl → .d

3. **SEQ_NUMBER** (94,711 → likely ~20K failed)
   - Number corruption: 10001 → 1OOO1 (O/0 confusion)
   - Format changes: SEQ-001 → 5EQ-OO1

## 🛠️ **Solution Strategy: Reduce Failed Spans to <10%**

### **Solution 1: Reduce Noise Aggressiveness (Quick Fix)**

```python
# CHANGE 1: Reduce OCR corruption probability
def _add_ocr_character_noise(text: str) -> str:
    if random.random() < 0.3:  # Reduce from 0.7 to 0.3 (30% instead of 70%)
        # Apply character substitutions
```

### **Solution 2: Entity-Aware Noise Generation (Recommended)**

```python
# CHANGE 2: Preserve entity boundaries during noise application
def apply_noise_preserving_entities(text: str, entities: List[Tuple], noise_level: float) -> str:
    """Apply noise while preserving entity character positions."""
    
    # Create protected ranges for entities
    protected_ranges = set()
    for start, end, label in entities:
        for i in range(start, end):
            protected_ranges.add(i)
    
    # Apply noise only to non-entity characters
    result = list(text)
    for i, char in enumerate(result):
        if i not in protected_ranges and random.random() < noise_level * 0.5:
            # Apply safe noise that doesn't affect entity spans
            if char.isspace():
                if random.random() < 0.3:
                    result[i] = '  '  # Double space
            elif char == '.':
                if random.random() < 0.2:
                    result[i] = ','  # Punctuation change
    
    return ''.join(result)
```

### **Solution 3: Pre-Noise Entity Position Tracking**

```python
# CHANGE 3: Track entity positions before and after noise
def generate_example_with_noise_tracking(country: str, mode: str, noise_level: float):
    # 1. Generate clean text and entities
    clean_text, entities = generate_base_example(country, mode)
    
    # 2. Apply controlled noise
    noisy_text = apply_safe_noise(clean_text, entities, noise_level)
    
    # 3. Recalculate entity positions in noisy text
    updated_entities = []
    for start, end, label in entities:
        entity_value = clean_text[start:end]
        
        # Find entity in noisy text (fuzzy matching)
        new_start = noisy_text.find(entity_value)
        if new_start != -1:
            new_end = new_start + len(entity_value)
            updated_entities.append((new_start, new_end, label))
        # Skip if entity not found (acceptable loss)
    
    return noisy_text, updated_entities
```

## 🎯 **Immediate Action Plan**

### **Step 1: Quick Fix (Reduce to ~15% failure rate)**

```bash
# Edit the noise probability in data_generation_noisy.py
# Line ~1005: Change from 0.7 to 0.3
if random.random() < 0.3:  # Reduced from 0.7
```

### **Step 2: Better Fix (Reduce to ~8% failure rate)**

```bash
# Add entity-boundary-aware noise generation
# Modify the apply_noise function to avoid entity regions
```

### **Step 3: Advanced Fix (Reduce to ~3% failure rate)**

```bash
# Implement fuzzy entity matching after noise application
# Use edit distance to relocate entities in noisy text
```

## 📊 **Expected Results After Fixes**

| Fix Level | Current | After Fix | Improvement |
|-----------|---------|-----------|-------------|
| **Quick** | 22.3% failed | ~15% failed | **7.3% reduction** |
| **Better** | 22.3% failed | ~8% failed | **14.3% reduction** |
| **Advanced** | 22.3% failed | ~3% failed | **19.3% reduction** |

### **Impact on Your Dataset:**
- **Current**: 26,807 failed spans
- **After Quick Fix**: ~18,000 failed spans (**-8,807 spans**)
- **After Better Fix**: ~9,600 failed spans (**-17,207 spans**)
- **After Advanced Fix**: ~3,600 failed spans (**-23,207 spans**)

## 🚀 **Implementation Priority**

### **Immediate (Today):**
1. Reduce noise probability from 0.7 to 0.3
2. Test with small dataset (1,000 examples)
3. Measure improvement

### **This Week:**
1. Implement entity-boundary-aware noise
2. Add fuzzy entity matching
3. Generate new 250K dataset with <10% failure rate

### **Recommended Command:**
```bash
# Test with reduced noise first
python data_generation_noisy.py --mode create-dataset --country all \
  --train-size 10000 --dev-size 2500 --noise --noise-level 0.5
```

The key insight is that **aggressive noise generation (70% probability) is breaking entity boundaries**, causing spaCy's `char_span` to fail when trying to create entity annotations.
