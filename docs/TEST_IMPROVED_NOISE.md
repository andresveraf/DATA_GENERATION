# Quick Test Commands to Validate Failed Spans Reduction

## 🧪 Test the Improved Code

### **Step 1: Test Small Dataset (Quick Validation)**
```bash
# Test with 1,000 examples to see immediate improvement
python data_generation_noisy.py --mode create-dataset --country all \
  --train-size 800 --dev-size 200 --noise --noise-level 0.6
```

### **Step 2: Compare Results**
```bash
# Check the stats file for improvement
cat output/multi_country_dataset_stats_noisy_1000.json | grep "failed_spans"

# Expected results:
# Before: ~220 failed spans (22% rate)
# After:  ~80-120 failed spans (8-12% rate)
```

### **Step 3: Generate Production Dataset**
```bash
# If test shows improvement, generate full dataset
python data_generation_noisy.py --mode create-dataset --country all \
  --train-size 200000 --dev-size 50000 --noise --noise-level 0.6
```

## 📊 Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| OCR Character Noise | 70% | 25% | **-45%** |
| Multiple Noise Types | 50% | 30% | **-20%** |
| Severe Corruption | 15% | 5% | **-10%** |
| Symbol Corruption | 30% | 15% | **-15%** |
| **Expected Failed Spans** | **22.3%** | **~8-12%** | **~10-14% reduction** |

## 🎯 Key Changes Made

### **1. Reduced OCR Character Noise**
- **Before**: 70% probability → 25% probability  
- **Impact**: Fewer corrupted entity boundaries

### **2. Conservative Multiple Noise**
- **Before**: 50% chance of 2-4 noise types → 30% chance of 1-2 noise types
- **Impact**: Less aggressive noise stacking

### **3. Reduced Severe Corruption**
- **Before**: 15% severe corruption → 5% severe corruption
- **Impact**: Fewer completely garbled entities

### **4. Reduced Symbol Corruption**
- **Before**: 30% symbol corruption → 15% symbol corruption  
- **Impact**: Better preservation of email @ symbols, phone + symbols

## 🚀 Next Steps

1. **Run the test command above**
2. **Check the failed_spans count in the stats JSON**
3. **If satisfied, generate your production 250K dataset**
4. **Monitor entity distribution balance**

The goal is to achieve **<10% failed spans** while maintaining realistic noise for robust NER training.
