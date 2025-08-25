# Excel Export Guide - Latin American Customer Data Generator

## 🎉 New Features Added

### ✅ Second Surname Implementation
- **Realistic Latin American naming**: Now supports **double surnames** (paternal + maternal)
- **Compound first names**: Includes second names like "Juan Carlos", "María José"
- **Cultural authenticity**: Follows actual Latin American naming conventions

### ✅ Excel Export for Data Review
- **Comprehensive review files**: Export generated data to Excel for validation
- **Multiple analysis sheets**: Summary, detailed data, country breakdown, name analysis
- **Quality assurance**: Easy way to review and validate the generated data

---

## 📊 How to Use Excel Export

### Basic Usage
```bash
# Create Excel file with 100 examples for review
python3 data_generation.py --mode excel-export

# Custom number of examples
python3 data_generation.py --mode excel-export --excel-examples 50

# Custom filename and output directory
python3 data_generation.py --mode excel-export --excel-examples 200 --excel-file detailed_review.xlsx --output-dir output
```

### Excel File Contents

The generated Excel file contains **5 sheets**:

#### 1. **Summary** Sheet
- Total examples generated
- Countries included
- Percentage of names with second names
- Percentage of names with second surnames
- Generation statistics

#### 2. **All_Data** Sheet
- Complete dataset with all generated examples
- Columns include:
  - `Customer_Name`: Full name with second names and surnames
  - `First_Name_Only`: Just the first name
  - `Full_Name_Part`: First + optional second name
  - `Complete_Surname`: Paternal + optional maternal surname
  - `Has_Second_Name`: Yes/No indicator
  - `Has_Second_Surname`: Yes/No indicator
  - All entity data (ID, address, phone, email, amount, etc.)

#### 3. **By_Country** Sheet
- Statistics broken down by country
- Second name and surname percentages per country
- Average entities per example

#### 4. **Name_Analysis** Sheet
- Analysis of naming patterns
- Count of unique first names and surnames
- Examples of compound names

#### 5. **Entity_Statistics** Sheet
- Count and percentage of each entity type
- Descriptions of entity types

---

## 🔍 What to Review in the Excel File

### ✅ Naming Quality
- **Second Names**: Look for realistic combinations like "Juan Carlos", "María José"
- **Second Surnames**: Verify authentic double surnames like "González Rodríguez"
- **Cultural Authenticity**: Names should reflect Latin American conventions

### ✅ Country-Specific Formats
- **Chile (CL)**: RUT format (XX.XXX.XXX-X), +56 phones, CLP currency
- **Argentina (AR)**: DNI format (XXXXXXXX), +54 phones, ARS currency
- **Brazil (BR)**: CPF format (XXX.XXX.XXX-XX), +55 phones, BRL currency
- **Uruguay (UY)**: CI format (X.XXX.XXX-X), +598 phones, UYU currency
- **Mexico (MX)**: CURP-like format, +52 phones, MXN currency

### ✅ Entity Recognition
- All entities should be properly identified and extracted
- Customer names should include the full compound names
- Email addresses should use only the paternal surname

### ✅ Data Diversity
- Good mix of simple and compound names
- Variety in all generated fields
- Balanced distribution across countries

---

## 📈 Example Names Generated

### With Second Names:
- `JUAN CARLOS GONZÁLEZ RODRÍGUEZ`
- `MARÍA JOSÉ SILVA MARTÍNEZ`
- `ANA SOFÍA TORRES PAREDES`
- `LUIS MIGUEL HERNÁNDEZ VEGA`

### With Second Surnames Only:
- `PEDRO GONZÁLEZ RODRÍGUEZ`
- `CAMILA SILVA MARTÍNEZ`
- `DIEGO TORRES PAREDES`

### Simple Names:
- `FERNANDO LÓPEZ`
- `VALENTINA GARCÍA`
- `SEBASTIÁN DÍAZ`

---

## 🚀 Quick Start Commands

```bash
# 1. Generate demo examples to see the new features
python3 data_generation.py --mode demo

# 2. Create Excel review file (25 examples - quick review)
python3 data_generation.py --mode excel-export --excel-examples 25

# 3. Create comprehensive Excel file (200 examples - thorough review)
python3 data_generation.py --mode excel-export --excel-examples 200 --excel-file full_review.xlsx

# 4. Create training dataset with new naming features
python3 data_generation.py --mode create-dataset --train-size 80000 --dev-size 20000
```

---

## 🎯 Benefits for NLP Training

### Enhanced Realism
- **Authentic Latin American names** improve model performance
- **Cultural accuracy** for customer service applications
- **Realistic email patterns** based on naming conventions

### Better Entity Recognition
- **Compound names** train models to handle complex naming patterns
- **Multiple surnames** improve recognition of full customer identities
- **Diverse formats** increase model robustness

### Quality Assurance
- **Excel export** allows manual validation of generated data
- **Statistical analysis** ensures proper distribution
- **Pattern verification** confirms cultural authenticity

---

## 📋 Requirements

- **Python 3.7+**
- **pandas** (for Excel export)
- **openpyxl** (for Excel file creation)
- **spacy** (for NLP processing)

Install requirements:
```bash
pip3 install pandas openpyxl spacy
```

---

## 🎉 Summary

The enhanced data generator now provides:

1. **✅ Realistic second surnames** following Latin American conventions
2. **✅ Compound first names** like "Juan Carlos", "María José"
3. **✅ Excel export functionality** for easy data review and validation
4. **✅ Comprehensive analysis** with multiple Excel sheets
5. **✅ Cultural authenticity** for better NLP model training

Use the Excel export feature to review and validate your generated data before using it for NLP training!