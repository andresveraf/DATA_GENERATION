# Failed Spans Fix Plan: Reduce from 22.4% to <8%

## 🎯 **Current Status**
- **Test Result**: 179 failed spans out of 800 examples = **22.4% failure rate**
- **Problem**: Still too high for production NER training
- **Target**: Reduce to <8% (64 failed spans per 800 examples)

## 📋 **Step-by-Step Fix Plan**

### **Phase 1: Advanced Noise Reduction (Immediate)**

#### **Fix 1.1: Further Reduce OCR Character Corruption**
```python
# Current: 25% probability → Target: 15% probability
if random.random() < 0.15:  # Reduce from 0.25 to 0.15
```

#### **Fix 1.2: Implement Entity-Aware Noise**
```python
def apply_noise_preserving_entities(text: str, entities: List, noise_level: float) -> str:
    """Apply noise while protecting entity boundaries."""
    # Create character-level protection map
    protected_positions = set()
    for start, end, label in entities:
        for pos in range(start, end):
            protected_positions.add(pos)
    
    # Apply noise only to non-entity characters
    result = list(text)
    for i, char in enumerate(result):
        if i not in protected_positions and random.random() < noise_level * 0.3:
            # Apply safe character-level noise
            if char.isalpha():
                result[i] = apply_safe_char_noise(char)
    
    return ''.join(result)
```

#### **Fix 1.3: Smarter Character Substitutions**
```python
# Avoid entity-breaking substitutions
SAFE_SUBSTITUTIONS = {
    # Keep critical entity chars intact
    '@': '@',  # Preserve email symbols
    '.': '.',  # Preserve ID number dots
    '-': '-',  # Preserve ID number dashes
    '+': '+',  # Preserve phone prefixes
}
```

### **Phase 2: Enhanced Entity Position Tracking**

#### **Fix 2.1: Pre-Post Noise Entity Matching**
```python
def generate_example_with_entity_tracking(country: str, mode: str, noise_level: float):
    # 1. Generate clean example
    clean_text, clean_entities = generate_base_example(country, mode)
    
    # 2. Apply controlled noise
    noisy_text = apply_entity_aware_noise(clean_text, clean_entities, noise_level)
    
    # 3. Re-find entities in noisy text using fuzzy matching
    validated_entities = []
    for start, end, label in clean_entities:
        original_entity = clean_text[start:end]
        new_position = find_entity_in_noisy_text(original_entity, noisy_text)
        
        if new_position:
            validated_entities.append((new_position[0], new_position[1], label))
    
    return noisy_text, validated_entities
```

#### **Fix 2.2: Fuzzy Entity Matching**
```python
from difflib import SequenceMatcher

def find_entity_in_noisy_text(entity: str, noisy_text: str, threshold: float = 0.8):
    """Find entity in noisy text using fuzzy matching."""
    entity_len = len(entity)
    best_match = None
    best_score = 0
    
    # Sliding window search
    for i in range(len(noisy_text) - entity_len + 1):
        candidate = noisy_text[i:i + entity_len]
        score = SequenceMatcher(None, entity.lower(), candidate.lower()).ratio()
        
        if score > threshold and score > best_score:
            best_score = score
            best_match = (i, i + entity_len)
    
    return best_match
```

### **Phase 3: Smart Noise Application Strategy**

#### **Fix 3.1: Entity-Type-Specific Noise Levels**
```python
ENTITY_NOISE_LEVELS = {
    'CUSTOMER_NAME': 0.3,    # Low noise - names are critical
    'EMAIL': 0.2,            # Very low noise - @ symbols critical
    'ID_NUMBER': 0.25,       # Low noise - format critical
    'PHONE_NUMBER': 0.3,     # Low noise - + symbols critical
    'ADDRESS': 0.5,          # Higher noise - more tolerant
    'AMOUNT': 0.4,           # Medium noise - currency symbols
    'SEQ_NUMBER': 0.2,       # Very low noise - numbers critical
}
```

#### **Fix 3.2: Progressive Noise Application**
```python
def apply_progressive_noise(text: str, entities: List, base_noise: float):
    """Apply noise in stages to monitor entity preservation."""
    
    # Stage 1: Safe noise (spacing, punctuation)
    stage1_text = apply_safe_noise(text, noise_level=base_noise * 0.5)
    preserved_entities = validate_entities(stage1_text, entities, threshold=0.9)
    
    # Stage 2: Character noise (only if entities still valid)
    if len(preserved_entities) >= len(entities) * 0.8:  # 80% preservation
        stage2_text = apply_char_noise(stage1_text, noise_level=base_noise * 0.3)
        preserved_entities = validate_entities(stage2_text, entities, threshold=0.8)
    else:
        stage2_text = stage1_text
    
    # Stage 3: OCR noise (only if entities still mostly valid)
    if len(preserved_entities) >= len(entities) * 0.7:  # 70% preservation
        final_text = apply_ocr_noise(stage2_text, noise_level=base_noise * 0.2)
    else:
        final_text = stage2_text
    
    return final_text
```

### **Phase 4: Implementation Steps**

#### **Step 4.1: Create Enhanced Noise Functions**
```bash
# Create new noise module with entity awareness
touch enhanced_noise_generation.py
```

#### **Step 4.2: Modify Main Generation Function**
```python
# Update create_dataset_training_examples function
def create_dataset_training_examples(
    size: int, 
    countries: List[str], 
    include_noise: bool = True, 
    noise_level: float = 0.6,
    entity_preservation_mode: bool = True  # NEW parameter
):
```

#### **Step 4.3: Add Validation Layer**
```python
def validate_example_quality(text: str, entities: List) -> float:
    """Calculate example quality score based on entity preservation."""
    total_entities = len(entities)
    valid_entities = 0
    
    nlp = spacy.load("es_core_news_sm")
    doc = nlp.make_doc(text)
    
    for start, end, label in entities:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is not None:
            valid_entities += 1
    
    return valid_entities / total_entities if total_entities > 0 else 0
```

## 🚀 **Implementation Timeline**

### **Day 1: Quick Fixes (Target: 15% failure rate)**
1. ✅ Reduce OCR noise probability: 25% → 15%
2. ✅ Reduce severe corruption: 5% → 2%
3. ✅ Test with 1,000 examples

### **Day 2: Entity-Aware Noise (Target: 10% failure rate)**
1. Implement entity position protection
2. Add fuzzy entity matching
3. Test with 5,000 examples

### **Day 3: Advanced Validation (Target: <8% failure rate)**
1. Add progressive noise application
2. Implement entity-type-specific noise levels
3. Test with 10,000 examples

### **Day 4: Production Dataset**
1. Generate full 250K dataset with <8% failure rate
2. Validate entity distribution balance
3. Performance testing

## 📊 **Expected Results Timeline**

| Phase | Target Failed Spans | Improvement | Action |
|-------|-------------------|-------------|--------|
| **Current** | 179/800 (22.4%) | - | Baseline |
| **Phase 1** | 120/800 (15%) | -7.4% | Quick noise reduction |
| **Phase 2** | 80/800 (10%) | -12.4% | Entity-aware noise |
| **Phase 3** | 64/800 (8%) | -14.4% | Advanced validation |
| **Phase 4** | 48/800 (6%) | -16.4% | Production ready |

## 🎯 **Success Metrics**

### **Primary Goals:**
- ✅ **Failed spans < 8%** (64 per 800 examples)
- ✅ **Entity distribution balance** (±5% variance)
- ✅ **Zero overlap errors** (maintained)
- ✅ **Realistic noise preservation** for robust training

### **Secondary Goals:**
- 📈 **Model F1 score improvement**: 85% → 92%+
- ⚡ **Training efficiency**: Better convergence
- 🎯 **Production accuracy**: <5% entity detection loss

## 🔧 **Monitoring & Testing**

### **Test Commands:**
```bash
# Phase 1 test
python data_generation_noisy.py --mode create-dataset --country all \
  --train-size 1000 --dev-size 250 --noise --noise-level 0.5

# Phase 2 test  
python data_generation_noisy.py --mode create-dataset --country all \
  --train-size 5000 --dev-size 1250 --noise --noise-level 0.6

# Phase 3 test
python data_generation_noisy.py --mode create-dataset --country all \
  --train-size 10000 --dev-size 2500 --noise --noise-level 0.7

# Production dataset
python data_generation_noisy.py --mode create-dataset --country all \
  --train-size 200000 --dev-size 50000 --noise --noise-level 0.6
```

### **Quality Checks:**
```bash
# Monitor failed spans
grep "failed_spans" output/multi_country_dataset_stats_*.json

# Check entity balance
grep -A 10 "entity_distribution" output/multi_country_dataset_stats_*.json
```

This plan will systematically reduce your failed spans from 22.4% to <8% while maintaining realistic noise for robust NER training.
