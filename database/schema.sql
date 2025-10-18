-- Multi-Country PII Data Generation Database Schema
-- Comprehensive schema for storing generated PII data, entities, and metadata
-- Supports multi-country, multi-entity type data with corruption tracking

-- Countries table for supported regions
CREATE TABLE IF NOT EXISTS countries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code VARCHAR(2) UNIQUE NOT NULL,  -- 'CL', 'MX', 'BR', 'UY'
    name VARCHAR(50) NOT NULL,        -- 'Chile', 'Mexico', 'Brazil', 'Uruguay'
    language VARCHAR(10) NOT NULL,    -- 'es', 'pt'
    currency VARCHAR(3) NOT NULL,     -- 'CLP', 'MXN', 'BRL', 'UYU'
    phone_prefix VARCHAR(5) NOT NULL, -- '+56', '+52', '+55', '+598'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Entity types supported by the system
CREATE TABLE IF NOT EXISTS entity_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL, -- 'CUSTOMER_NAME', 'ID_NUMBER', etc.
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Corruption levels for OCR noise simulation
CREATE TABLE IF NOT EXISTS corruption_levels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL, -- 'none', 'light', 'medium', 'heavy', 'extreme'
    level FLOAT NOT NULL,             -- 0.0 to 1.0
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document types for negative examples and PII documents
CREATE TABLE IF NOT EXISTS document_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL, -- 'pii_document', 'negative_example', 'mixed'
    description TEXT,
    has_pii BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Main table for generated documents
CREATE TABLE IF NOT EXISTS generated_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id VARCHAR(100) UNIQUE NOT NULL, -- Unique identifier for each document
    country_id INTEGER NOT NULL,
    document_type_id INTEGER NOT NULL,
    corruption_level_id INTEGER NOT NULL,
    
    -- Document content and metadata
    original_text TEXT NOT NULL,
    corrupted_text TEXT,
    template_used VARCHAR(100),
    generation_mode VARCHAR(50), -- 'create-dataset', 'test', 'validation'
    
    -- Statistics
    total_entities INTEGER DEFAULT 0,
    successful_entities INTEGER DEFAULT 0,
    failed_entities INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraints
    FOREIGN KEY (country_id) REFERENCES countries(id),
    FOREIGN KEY (document_type_id) REFERENCES document_types(id),
    FOREIGN KEY (corruption_level_id) REFERENCES corruption_levels(id)
);

-- Individual entities found in documents
CREATE TABLE IF NOT EXISTS document_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    entity_type_id INTEGER NOT NULL,
    
    -- Entity details
    original_text VARCHAR(500) NOT NULL,
    corrupted_text VARCHAR(500),
    start_pos INTEGER NOT NULL,
    end_pos INTEGER NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    
    -- Status tracking
    is_preserved BOOLEAN DEFAULT TRUE,
    is_overlapping BOOLEAN DEFAULT FALSE,
    validation_status VARCHAR(20) DEFAULT 'valid', -- 'valid', 'failed', 'corrupted'
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraints
    FOREIGN KEY (document_id) REFERENCES generated_documents(id) ON DELETE CASCADE,
    FOREIGN KEY (entity_type_id) REFERENCES entity_types(id)
);

-- Generation sessions for batch tracking
CREATE TABLE IF NOT EXISTS generation_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Session parameters
    country_filter VARCHAR(20), -- 'all', 'chile', 'mexico', etc.
    train_size INTEGER,
    dev_size INTEGER,
    noise_enabled BOOLEAN DEFAULT FALSE,
    noise_level FLOAT DEFAULT 0.0,
    
    -- Session statistics
    total_documents INTEGER DEFAULT 0,
    successful_documents INTEGER DEFAULT 0,
    total_entities INTEGER DEFAULT 0,
    successful_entities INTEGER DEFAULT 0,
    overall_success_rate FLOAT DEFAULT 0.0,
    
    -- Timing
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds INTEGER,
    
    -- Configuration
    custom_weights TEXT, -- JSON string of custom weights
    output_directory VARCHAR(500),
    export_formats TEXT  -- JSON array of export formats
);

-- Dataset exports tracking
CREATE TABLE IF NOT EXISTS dataset_exports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    export_type VARCHAR(50) NOT NULL, -- 'spacy', 'json', 'excel', 'csv'
    file_path VARCHAR(500) NOT NULL,
    file_size INTEGER,
    record_count INTEGER,
    export_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (session_id) REFERENCES generation_sessions(id)
);

-- Statistics aggregation table for quick queries
CREATE TABLE IF NOT EXISTS statistics_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code VARCHAR(2),
    entity_type VARCHAR(50),
    corruption_level VARCHAR(50),
    document_type VARCHAR(50),
    
    -- Aggregated counts
    total_documents INTEGER DEFAULT 0,
    total_entities INTEGER DEFAULT 0,
    successful_entities INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0,
    
    -- Date aggregation
    date_created DATE NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint for aggregation
    UNIQUE(country_code, entity_type, corruption_level, document_type, date_created)
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_documents_country ON generated_documents(country_id);
CREATE INDEX IF NOT EXISTS idx_documents_type ON generated_documents(document_type_id);
CREATE INDEX IF NOT EXISTS idx_documents_corruption ON generated_documents(corruption_level_id);
CREATE INDEX IF NOT EXISTS idx_documents_created ON generated_documents(created_at);
CREATE INDEX IF NOT EXISTS idx_entities_document ON document_entities(document_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON document_entities(entity_type_id);
CREATE INDEX IF NOT EXISTS idx_entities_preserved ON document_entities(is_preserved);
CREATE INDEX IF NOT EXISTS idx_sessions_country ON generation_sessions(country_filter);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON generation_sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_statistics_date ON statistics_summary(date_created);
CREATE INDEX IF NOT EXISTS idx_statistics_country ON statistics_summary(country_code);

-- Views for common queries
CREATE VIEW IF NOT EXISTS v_document_summary AS
SELECT 
    gd.id,
    gd.document_id,
    c.name as country_name,
    c.code as country_code,
    dt.name as document_type,
    cl.name as corruption_level,
    gd.total_entities,
    gd.successful_entities,
    gd.success_rate,
    gd.created_at
FROM generated_documents gd
JOIN countries c ON gd.country_id = c.id
JOIN document_types dt ON gd.document_type_id = dt.id
JOIN corruption_levels cl ON gd.corruption_level_id = cl.id;

CREATE VIEW IF NOT EXISTS v_entity_summary AS
SELECT 
    de.id,
    gd.document_id,
    et.name as entity_type,
    de.original_text,
    de.corrupted_text,
    de.is_preserved,
    de.validation_status,
    c.code as country_code
FROM document_entities de
JOIN generated_documents gd ON de.document_id = gd.id
JOIN entity_types et ON de.entity_type_id = et.id
JOIN countries c ON gd.country_id = c.id;

CREATE VIEW IF NOT EXISTS v_session_performance AS
SELECT 
    gs.session_id,
    gs.country_filter,
    gs.train_size,
    gs.dev_size,
    gs.noise_level,
    gs.total_documents,
    gs.overall_success_rate,
    gs.duration_seconds,
    COUNT(de.id) as export_count
FROM generation_sessions gs
LEFT JOIN dataset_exports de ON gs.id = de.session_id
GROUP BY gs.id;

