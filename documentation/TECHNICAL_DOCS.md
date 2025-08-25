# Technical Documentation - Latin American NLP Data Generator

## 🏗️ System Architecture

### Component Interaction Diagram

```mermaid
flowchart TB
    subgraph "User Interface Layer"
        CLI[Command Line Interface]
        API[Python API Functions]
    end
    
    subgraph "Configuration Layer"
        CONFIG[Configuration Manager]
        ARGS[Argument Parser]
        DEFAULTS[Default Settings]
    end
    
    subgraph "Data Generation Core"
        NAMEGEN[Name Generator]
        IDGEN[ID Generator] 
        ADDRGEN[Address Generator]
        PHONEGEN[Phone Generator]
        EMAILGEN[Email Generator]
        AMOUNTGEN[Amount Generator]
        SEQGEN[Sequence Generator]
    end
    
    subgraph "Country-Specific Data"
        CLDATA[Chile Data]
        ARDATA[Argentina Data]
        BRDATA[Brazil Data]
        UYDATA[Uruguay Data]
        MXDATA[Mexico Data]
    end
    
    subgraph "NLP Processing Engine"
        SPACY[spaCy NLP Pipeline]
        TOKENIZER[Tokenizer]
        ENTITYEXT[Entity Extractor]
        ANNOTATOR[Annotation Creator]
    end
    
    subgraph "Output Management"
        DOCBIN[DocBin Creator]
        FILEWRITER[File Writer]
        STATS[Statistics Generator]
    end
    
    subgraph "Storage Layer"
        TRAINFILE[train.spacy]
        DEVFILE[dev.spacy]
        STATSFILE[statistics.json]
    end
    
    CLI --> CONFIG
    API --> CONFIG
    CONFIG --> ARGS
    CONFIG --> DEFAULTS
    
    CONFIG --> NAMEGEN
    CONFIG --> IDGEN
    CONFIG --> ADDRGEN
    CONFIG --> PHONEGEN
    CONFIG --> EMAILGEN
    CONFIG --> AMOUNTGEN
    CONFIG --> SEQGEN
    
    NAMEGEN --> CLDATA
    IDGEN --> ARDATA
    ADDRGEN --> BRDATA
    PHONEGEN --> UYDATA
    EMAILGEN --> MXDATA
    
    NAMEGEN --> SPACY
    IDGEN --> SPACY
    ADDRGEN --> SPACY
    PHONEGEN --> SPACY
    EMAILGEN --> SPACY
    AMOUNTGEN --> SPACY
    SEQGEN --> SPACY
    
    SPACY --> TOKENIZER
    TOKENIZER --> ENTITYEXT
    ENTITYEXT --> ANNOTATOR
    
    ANNOTATOR --> DOCBIN
    DOCBIN --> FILEWRITER
    FILEWRITER --> STATS
    
    FILEWRITER --> TRAINFILE
    FILEWRITER --> DEVFILE
    STATS --> STATSFILE
    
    style CLI fill:#e3f2fd
    style SPACY fill:#f3e5f5
    style DOCBIN fill:#e8f5e8
    style TRAINFILE fill:#fff3e0
    style DEVFILE fill:#fff3e0
```

## 🔄 Data Flow Architecture

### End-to-End Data Processing

```mermaid
flowchart LR
    subgraph "Input Processing"
        A[User Requirements] --> B[Parse Arguments]
        B --> C[Validate Configuration]
        C --> D[Initialize Generators]
    end
    
    subgraph "Generation Pipeline"
        D --> E[Select Country]
        E --> F[Choose Generation Mode]
        F --> G[Generate Customer Data]
        G --> H[Create Text Template]
        H --> I[Format Output String]
    end
    
    subgraph "NLP Processing"
        I --> J[Load spaCy Model]
        J --> K[Process Text]
        K --> L[Extract Entities]
        L --> M[Create Annotations]
        M --> N[Validate Entities]
    end
    
    subgraph "Dataset Creation"
        N --> O[Add to DocBin]
        O --> P{Batch Complete?}
        P -->|No| E
        P -->|Yes| Q[Split Train/Dev]
        Q --> R[Save Files]
        R --> S[Generate Statistics]
    end
    
    subgraph "Output Management"
        S --> T[train.spacy File]
        S --> U[dev.spacy File]
        S --> V[Statistics Report]
        S --> W[Console Output]
    end
    
    style A fill:#e1f5fe
    style J fill:#f3e5f5
    style T fill:#e8f5e8
    style U fill:#e8f5e8
    style V fill:#fff3e0
```

## 🎯 Entity Recognition Pipeline

### NER Processing Workflow

```mermaid
flowchart TB
    subgraph "Text Input"
        INPUT["Cliente María González, RUT 12.345.678-9, vive en Av. Libertador 1234, Santiago"]
    end
    
    subgraph "spaCy Processing"
        TOKENIZE[Tokenization]
        POS[POS Tagging]
        LEMMA[Lemmatization]
        PARSE[Dependency Parsing]
    end
    
    subgraph "Entity Extraction"
        PATTERNS[Pattern Matching]
        RULES[Rule-based NER]
        POSITIONS[Position Calculation]
    end
    
    subgraph "Entity Classification"
        CUSTOMER["CUSTOMER_NAME: 'María González'<br/>Start: 8, End: 21"]
        ID["ID_NUMBER: '12.345.678-9'<br/>Start: 27, End: 39"]
        ADDRESS["ADDRESS: 'Av. Libertador 1234, Santiago'<br/>Start: 49, End: 78"]
    end
    
    subgraph "Annotation Output"
        ENTITIES["entities: [(8, 21, 'CUSTOMER_NAME'),<br/>(27, 39, 'ID_NUMBER'),<br/>(49, 78, 'ADDRESS')]"]
    end
    
    INPUT --> TOKENIZE
    TOKENIZE --> POS
    POS --> LEMMA
    LEMMA --> PARSE
    PARSE --> PATTERNS
    PATTERNS --> RULES
    RULES --> POSITIONS
    POSITIONS --> CUSTOMER
    POSITIONS --> ID
    POSITIONS --> ADDRESS
    CUSTOMER --> ENTITIES
    ID --> ENTITIES
    ADDRESS --> ENTITIES
    
    style INPUT fill:#e3f2fd
    style ENTITIES fill:#e8f5e8
    style CUSTOMER fill:#fff3e0
    style ID fill:#fff3e0
    style ADDRESS fill:#fff3e0
```

## 🌍 Country-Specific Data Models

### Multi-Country Data Generation

```mermaid
flowchart TB
    subgraph "Country Selection Logic"
        START[Start Generation] --> COUNTRY{Select Country}
        COUNTRY -->|CL| CHILE[Chile Pipeline]
        COUNTRY -->|AR| ARGENTINA[Argentina Pipeline]
        COUNTRY -->|BR| BRAZIL[Brazil Pipeline]
        COUNTRY -->|UY| URUGUAY[Uruguay Pipeline]
        COUNTRY -->|MX| MEXICO[Mexico Pipeline]
    end
    
    subgraph "Chile (CL) - Spanish"
        CHILE --> CL_NAME[Names: María, José, Carmen]
        CHILE --> CL_ID[ID: 12.345.678-9]
        CHILE --> CL_PHONE[Phone: +56 9 8765 4321]
        CHILE --> CL_ADDR[Address: Santiago, Valparaíso]
        CHILE --> CL_CURR[Currency: CLP]
    end
    
    subgraph "Argentina (AR) - Spanish"
        ARGENTINA --> AR_NAME[Names: Juan, Ana, Carlos]
        ARGENTINA --> AR_ID[ID: 12.345.678]
        ARGENTINA --> AR_PHONE[Phone: +54 11 1234 5678]
        ARGENTINA --> AR_ADDR[Address: Buenos Aires, Córdoba]
        ARGENTINA --> AR_CURR[Currency: ARS]
    end
    
    subgraph "Brazil (BR) - Portuguese"
        BRAZIL --> BR_NAME[Names: João, Maria, Pedro]
        BRAZIL --> BR_ID[ID: 123.456.789-01]
        BRAZIL --> BR_PHONE[Phone: +55 11 98765 4321]
        BRAZIL --> BR_ADDR[Address: São Paulo, Rio de Janeiro]
        BRAZIL --> BR_CURR[Currency: BRL]
    end
    
    subgraph "Data Consolidation"
        CL_CURR --> CONSOLIDATE[Consolidate Country Data]
        AR_CURR --> CONSOLIDATE
        BR_CURR --> CONSOLIDATE
        CONSOLIDATE --> TEMPLATE[Generate Text Template]
        TEMPLATE --> OUTPUT[Final Customer Record]
    end
    
    style START fill:#4caf50
    style COUNTRY fill:#ff9800
    style CONSOLIDATE fill:#2196f3
    style OUTPUT fill:#8bc34a
```

## 📊 Dataset Size Configuration Matrix

### Size Impact Analysis

```mermaid
flowchart TB
    subgraph "Configuration Options"
        CLI_ARGS[Command Line Arguments]
        FUNC_PARAMS[Function Parameters]
        DEFAULT_VALS[Default Values]
    end
    
    subgraph "Size Categories"
        SMALL[Small: 1K examples]
        MEDIUM[Medium: 10K examples]
        LARGE[Large: 100K examples]
        XLARGE[X-Large: 1M+ examples]
    end
    
    subgraph "Resource Requirements"
        MEM_SMALL[Memory: ~10MB]
        MEM_MEDIUM[Memory: ~100MB]
        MEM_LARGE[Memory: ~1GB]
        MEM_XLARGE[Memory: ~10GB+]
        
        TIME_SMALL[Time: ~30 seconds]
        TIME_MEDIUM[Time: ~5 minutes]
        TIME_LARGE[Time: ~45 minutes]
        TIME_XLARGE[Time: ~8+ hours]
    end
    
    subgraph "Use Cases"
        TEST[Testing & Development]
        DEMO[Demonstration]
        RESEARCH[Research Projects]
        PRODUCTION[Production Systems]
    end
    
    CLI_ARGS --> SMALL
    CLI_ARGS --> MEDIUM
    CLI_ARGS --> LARGE
    CLI_ARGS --> XLARGE
    
    SMALL --> MEM_SMALL
    SMALL --> TIME_SMALL
    SMALL --> TEST
    
    MEDIUM --> MEM_MEDIUM
    MEDIUM --> TIME_MEDIUM
    MEDIUM --> DEMO
    
    LARGE --> MEM_LARGE
    LARGE --> TIME_LARGE
    LARGE --> RESEARCH
    
    XLARGE --> MEM_XLARGE
    XLARGE --> TIME_XLARGE
    XLARGE --> PRODUCTION
    
    style CLI_ARGS fill:#e3f2fd
    style SMALL fill:#4caf50
    style MEDIUM fill:#ff9800
    style LARGE fill:#f44336
    style XLARGE fill:#9c27b0
```

## 🔧 Generation Mode Comparison

### Mode-Specific Entity Distribution

```mermaid
flowchart LR
    subgraph "Generation Modes"
        FULL[full mode]
        ADDR[addr_only mode]
        ID[id_only mode]
        CONTACT[contact_only mode]
        FINANCIAL[financial_only mode]
    end
    
    subgraph "Entity Types Generated"
        subgraph "Full Mode Entities"
            F_NAME[CUSTOMER_NAME]
            F_ID[ID_NUMBER]
            F_ADDR[ADDRESS]
            F_PHONE[PHONE_NUMBER]
            F_EMAIL[EMAIL]
            F_AMOUNT[AMOUNT]
            F_SEQ[SEQ_NUMBER]
        end
        
        subgraph "Address Only"
            A_ADDR[ADDRESS]
        end
        
        subgraph "ID Only"
            I_ID[ID_NUMBER]
        end
        
        subgraph "Contact Only"
            C_PHONE[PHONE_NUMBER]
            C_EMAIL[EMAIL]
        end
        
        subgraph "Financial Only"
            FI_AMOUNT[AMOUNT]
            FI_SEQ[SEQ_NUMBER]
        end
    end
    
    FULL --> F_NAME
    FULL --> F_ID
    FULL --> F_ADDR
    FULL --> F_PHONE
    FULL --> F_EMAIL
    FULL --> F_AMOUNT
    FULL --> F_SEQ
    
    ADDR --> A_ADDR
    ID --> I_ID
    CONTACT --> C_PHONE
    CONTACT --> C_EMAIL
    FINANCIAL --> FI_AMOUNT
    FINANCIAL --> FI_SEQ
    
    style FULL fill:#4caf50
    style ADDR fill:#2196f3
    style ID fill:#ff9800
    style CONTACT fill:#9c27b0
    style FINANCIAL fill:#f44336
```

## 🚀 Performance Optimization Strategies

### Scalability Architecture

```mermaid
flowchart TB
    subgraph "Input Optimization"
        BATCH[Batch Processing]
        PARALLEL[Parallel Generation]
        MEMORY[Memory Management]
    end
    
    subgraph "Processing Optimization"
        CACHE[Data Caching]
        POOL[Process Pooling]
        STREAM[Streaming Output]
    end
    
    subgraph "Output Optimization"
        COMPRESS[Data Compression]
        CHUNK[Chunked Writing]
        BUFFER[Output Buffering]
    end
    
    subgraph "Monitoring"
        METRICS[Performance Metrics]
        PROGRESS[Progress Tracking]
        ERRORS[Error Handling]
    end
    
    BATCH --> CACHE
    PARALLEL --> POOL
    MEMORY --> STREAM
    
    CACHE --> COMPRESS
    POOL --> CHUNK
    STREAM --> BUFFER
    
    COMPRESS --> METRICS
    CHUNK --> PROGRESS
    BUFFER --> ERRORS
    
    style BATCH fill:#4caf50
    style PARALLEL fill:#2196f3
    style CACHE fill:#ff9800
    style STREAM fill:#9c27b0
```

## 🔍 Quality Assurance Pipeline

### Testing and Validation Workflow

```mermaid
flowchart LR
    subgraph "Input Validation"
        VAL_PARAMS[Parameter Validation]
        VAL_CONFIG[Configuration Check]
        VAL_DEPS[Dependency Verification]
    end
    
    subgraph "Generation Testing"
        TEST_SINGLE[Single Example Test]
        TEST_BATCH[Batch Generation Test]
        TEST_MODES[Mode-specific Tests]
        TEST_COUNTRIES[Country-specific Tests]
    end
    
    subgraph "Quality Checks"
        CHECK_ENTITIES[Entity Validation]
        CHECK_FORMAT[Format Verification]
        CHECK_BALANCE[Distribution Balance]
        CHECK_UNIQUENESS[Uniqueness Check]
    end
    
    subgraph "Output Validation"
        VAL_FILES[File Integrity Check]
        VAL_SIZE[Size Verification]
        VAL_STATS[Statistics Validation]
        VAL_COMPAT[spaCy Compatibility]
    end
    
    VAL_PARAMS --> TEST_SINGLE
    VAL_CONFIG --> TEST_BATCH
    VAL_DEPS --> TEST_MODES
    
    TEST_SINGLE --> CHECK_ENTITIES
    TEST_BATCH --> CHECK_FORMAT
    TEST_MODES --> CHECK_BALANCE
    TEST_COUNTRIES --> CHECK_UNIQUENESS
    
    CHECK_ENTITIES --> VAL_FILES
    CHECK_FORMAT --> VAL_SIZE
    CHECK_BALANCE --> VAL_STATS
    CHECK_UNIQUENESS --> VAL_COMPAT
    
    style VAL_PARAMS fill:#e3f2fd
    style TEST_SINGLE fill:#f3e5f5
    style CHECK_ENTITIES fill:#e8f5e8
    style VAL_FILES fill:#fff3e0
```

## 📈 Monitoring and Analytics

### Runtime Performance Dashboard

```mermaid
flowchart TB
    subgraph "Performance Metrics"
        SPEED[Generation Speed<br/>examples/second]
        MEMORY[Memory Usage<br/>MB/1K examples]
        CPU[CPU Utilization<br/>% usage]
        IO[I/O Operations<br/>write speed]
    end
    
    subgraph "Quality Metrics"
        ENTITY_ACC[Entity Accuracy<br/>% correct positions]
        BALANCE[Country Balance<br/>distribution ratio]
        UNIQUE[Uniqueness Rate<br/>% unique examples]
        FORMAT[Format Compliance<br/>% valid format]
    end
    
    subgraph "Error Tracking"
        GEN_ERRORS[Generation Errors]
        PARSING_ERRORS[Parsing Errors]
        FILE_ERRORS[File I/O Errors]
        MEMORY_ERRORS[Memory Errors]
    end
    
    subgraph "Alerts & Notifications"
        PERF_ALERT[Performance Degradation]
        ERROR_ALERT[Error Rate Threshold]
        RESOURCE_ALERT[Resource Exhaustion]
        QUALITY_ALERT[Quality Issues]
    end
    
    SPEED --> PERF_ALERT
    MEMORY --> RESOURCE_ALERT
    ENTITY_ACC --> QUALITY_ALERT
    GEN_ERRORS --> ERROR_ALERT
    
    style SPEED fill:#4caf50
    style ENTITY_ACC fill:#2196f3
    style GEN_ERRORS fill:#f44336
    style PERF_ALERT fill:#ff9800
```

## 🧪 PII Detection Testing Framework

### Testing Script Architecture

```mermaid
flowchart TB
    subgraph "Testing Scripts"
        QUICK[quick_test.py<br/>Simple Testing]
        FULL[test_pii_ner.py<br/>Comprehensive Testing]
    end
    
    subgraph "Model Loading"
        TRAINED[Trained Model<br/>./model/]
        FALLBACK[Base Spanish Model<br/>es_core_news_lg]
        MINIMAL[Small Model<br/>es_core_news_sm]
    end
    
    subgraph "Test Modes"
        INTERACTIVE[Interactive Mode<br/>User Input]
        PREDEFINED[Predefined Tests<br/>10+ Examples]
        CUSTOM[Custom Text<br/>Modifiable]
    end
    
    subgraph "Analysis Output"
        ENTITIES[Entity Detection<br/>Type + Position]
        CONTEXT[Context Display<br/>Surrounding Text]
        STATS[Statistics<br/>Counts + Groups]
        CONFIDENCE[Confidence Scores<br/>Model Certainty]
    end
    
    QUICK --> TRAINED
    FULL --> TRAINED
    TRAINED --> FALLBACK
    FALLBACK --> MINIMAL
    
    FULL --> INTERACTIVE
    FULL --> PREDEFINED
    QUICK --> CUSTOM
    
    INTERACTIVE --> ENTITIES
    PREDEFINED --> CONTEXT
    CUSTOM --> STATS
    ENTITIES --> CONFIDENCE
    
    style QUICK fill:#e3f2fd
    style FULL fill:#f3e5f5
    style TRAINED fill:#e8f5e8
    style INTERACTIVE fill:#fff3e0
```

### Testing Script Implementation Details

#### quick_test.py Architecture

```python
def quick_test():
    """
    Lightweight testing function for basic model validation
    
    Flow:
    1. Attempt to load trained model (./model/)
    2. Fallback to base Spanish models if needed
    3. Process predefined test text
    4. Display detected entities with labels
    5. Provide modification guidance
    """
    
    # Model Loading Priority Chain
    # 1. ./model/ (trained model)
    # 2. es_core_news_lg (large base model)  
    # 3. es_core_news_sm (small base model)
    # 4. Error if none available
```

#### test_pii_ner.py Architecture

```python
class PIITester:
    """
    Comprehensive PII detection testing framework
    
    Features:
    - Model validation and fallback handling
    - Interactive text input mode
    - Predefined test suite execution
    - Detailed entity analysis and reporting
    - Context visualization around entities
    - Performance metrics collection
    """
    
    def load_trained_model(self, model_path: str) -> spacy.Language:
        """Load and validate trained spaCy model"""
        
    def test_text_with_model(self, nlp, text: str, show_details: bool) -> None:
        """Process text and analyze detected entities"""
        
    def interactive_mode(self, nlp) -> None:
        """Interactive testing with user input"""
        
    def run_predefined_tests(self, nlp) -> None:
        """Execute comprehensive test suite"""
```

### Test Case Coverage Matrix

| Test Category | Entity Types | Languages | Countries | Purpose |
|---------------|-------------|-----------|-----------|---------|
| **Basic Validation** | NAME, ID, ADDRESS | ES | CL | Core functionality |
| **Compound Names** | CUSTOMER_NAME | ES | CL, AR, UY, MX | Second names/surnames |
| **Multi-Country IDs** | ID_NUMBER | ES, PT | All 5 | Format variations |
| **Address Variations** | ADDRESS | ES, PT | All 5 | Street + city patterns |
| **Contact Info** | PHONE, EMAIL | ES, PT | All 5 | Communication data |
| **Financial Data** | AMOUNT, SEQ_NUMBER | ES, PT | All 5 | Transaction info |
| **Mixed Entities** | All types | ES, PT | All 5 | Real-world complexity |
| **Edge Cases** | Partial matches | ES | CL | Error handling |

### Performance Testing Metrics

```python
class TestingMetrics:
    """
    Metrics collection for PII detection testing
    """
    
    precision: float        # True positives / (True positives + False positives)
    recall: float          # True positives / (True positives + False negatives)  
    f1_score: float        # 2 * (precision * recall) / (precision + recall)
    
    entity_accuracy: Dict[str, float]  # Accuracy per entity type
    processing_speed: float            # Examples processed per second
    memory_usage: float               # Peak memory during testing
    
    def calculate_metrics(self, predictions: List, ground_truth: List) -> None:
        """Calculate comprehensive testing metrics"""
        
    def generate_report(self) -> str:
        """Generate formatted performance report"""
```

### Error Handling and Diagnostics

```python
class TestingDiagnostics:
    """
    Diagnostic tools for PII testing troubleshooting
    """
    
    @staticmethod
    def diagnose_model_load_failure(model_path: str) -> List[str]:
        """Analyze and report model loading issues"""
        
    @staticmethod  
    def analyze_entity_mismatches(predicted: List, expected: List) -> Dict:
        """Compare predicted vs expected entities"""
        
    @staticmethod
    def validate_text_encoding(text: str) -> bool:
        """Check text encoding compatibility"""
        
    @staticmethod
    def check_spacy_installation() -> Dict[str, str]:
        """Verify spaCy installation and models"""
```

## 🔗 Integration Points

### External System Integration

```mermaid
flowchart LR
    subgraph "Data Generator"
        CORE[Core Generator]
    end
    
    subgraph "NLP Frameworks"
        SPACY[spaCy Integration]
        HUGGINGFACE[Hugging Face Models]
        TRANSFORMERS[Transformer Pipelines]
    end
    
    subgraph "Database Systems"
        POSTGRES[PostgreSQL]
        MONGO[MongoDB]
        ELASTIC[Elasticsearch]
    end
    
    subgraph "Cloud Services"
        AWS[AWS S3]
        GCP[Google Cloud Storage]
        AZURE[Azure Blob Storage]
    end
    
    subgraph "ML Platforms"
        MLflow[MLflow Tracking]
        WANDB[Weights & Biases]
        TENSORBOARD[TensorBoard]
    end
    
    CORE --> SPACY
    CORE --> HUGGINGFACE
    CORE --> TRANSFORMERS
    
    CORE --> POSTGRES
    CORE --> MONGO
    CORE --> ELASTIC
    
    CORE --> AWS
    CORE --> GCP
    CORE --> AZURE
    
    CORE --> MLflow
    CORE --> WANDB
    CORE --> TENSORBOARD
    
    style CORE fill:#4caf50
    style SPACY fill:#2196f3
    style AWS fill:#ff9800
    style MLflow fill:#9c27b0
```

---

This technical documentation provides detailed insights into the system architecture, data flows, and optimization strategies for the Latin American NLP Data Generator.
