"""
Database Manager for Multi-Country PII Data Generation System
============================================================

This module provides comprehensive database management functionality for the PII data generation system.
It handles SQLite database operations, data storage, retrieval, and analysis for generated documents,
entities, and metadata.

Key Features:
- Complete CRUD operations for all database tables
- Transaction management and error handling
- Data validation and integrity checks
- Performance optimization with prepared statements
- Statistics aggregation and reporting
- Export functionality integration
- Session tracking and management

Author: Andrés Vera Figueroa
Date: October 2024
Purpose: Systematic storage and management of generated PII training data
"""

import sqlite3
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from contextlib import contextmanager
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Comprehensive database manager for PII data generation system.
    
    Handles all database operations including document storage, entity tracking,
    session management, and statistics aggregation.
    """
    
    def __init__(self, db_path: str = "database/pii_generation.db"):
        """
        Initialize database manager with SQLite database.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize database with schema and default data."""
        try:
            with self.get_connection() as conn:
                # Read and execute schema
                schema_path = Path(__file__).parent / "schema.sql"
                if schema_path.exists():
                    with open(schema_path, 'r', encoding='utf-8') as f:
                        schema_sql = f.read()
                    conn.executescript(schema_sql)
                    
                # Insert default data
                self._insert_default_data(conn)
                conn.commit()
                logger.info(f"Database initialized successfully at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections with automatic cleanup.
        
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable column access by name
            conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _insert_default_data(self, conn: sqlite3.Connection):
        """Insert default reference data into the database."""
        
        # Default countries
        countries = [
            ('CL', 'Chile', 'es', 'CLP', '+56'),
            ('MX', 'Mexico', 'es', 'MXN', '+52'),
            ('BR', 'Brazil', 'pt', 'BRL', '+55'),
            ('UY', 'Uruguay', 'es', 'UYU', '+598')
        ]
        
        conn.executemany("""
            INSERT OR IGNORE INTO countries (code, name, language, currency, phone_prefix)
            VALUES (?, ?, ?, ?, ?)
        """, countries)
        
        # Default entity types
        entity_types = [
            ('CUSTOMER_NAME', 'Full customer names with country-specific conventions'),
            ('ID_NUMBER', 'Country-specific ID formats (RUT/CURP/CPF/Cédula)'),
            ('ADDRESS', 'Country-specific address formats'),
            ('PHONE_NUMBER', 'Country-specific phone number formats'),
            ('EMAIL', 'Email addresses with country-appropriate domains'),
            ('AMOUNT', 'Monetary amounts with country currencies'),
            ('SEQ_NUMBER', 'Sequential reference numbers')
        ]
        
        conn.executemany("""
            INSERT OR IGNORE INTO entity_types (name, description)
            VALUES (?, ?)
        """, entity_types)
        
        # Default corruption levels
        corruption_levels = [
            ('none', 0.0, 'No corruption applied'),
            ('light', 0.1, 'Light OCR noise (10% corruption)'),
            ('medium', 0.3, 'Medium OCR noise (30% corruption)'),
            ('heavy', 0.5, 'Heavy OCR noise (50% corruption)'),
            ('extreme', 0.8, 'Extreme OCR corruption (80% corruption)')
        ]
        
        conn.executemany("""
            INSERT OR IGNORE INTO corruption_levels (name, level, description)
            VALUES (?, ?, ?)
        """, corruption_levels)
        
        # Default document types
        document_types = [
            ('pii_document', 'Document containing PII entities', True),
            ('negative_example', 'Document without PII entities', False),
            ('mixed_document', 'Document with mixed PII and non-PII content', True)
        ]
        
        conn.executemany("""
            INSERT OR IGNORE INTO document_types (name, description, has_pii)
            VALUES (?, ?, ?)
        """, document_types)
    
    # ==========================================
    # Session Management
    # ==========================================
    
    def create_session(self, country_filter: str = 'all', train_size: int = 0, 
                      dev_size: int = 0, noise_enabled: bool = False, 
                      noise_level: float = 0.0, custom_weights: Dict = None,
                      output_directory: str = None) -> str:
        """
        Create a new generation session.
        
        Args:
            country_filter: Country filter ('all', 'chile', etc.)
            train_size: Training set size
            dev_size: Development set size
            noise_enabled: Whether noise is enabled
            noise_level: Noise level (0.0 to 1.0)
            custom_weights: Custom template weights
            output_directory: Output directory path
            
        Returns:
            str: Session ID
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO generation_sessions 
                (session_id, country_filter, train_size, dev_size, noise_enabled, 
                 noise_level, custom_weights, output_directory)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, country_filter, train_size, dev_size, noise_enabled,
                  noise_level, json.dumps(custom_weights) if custom_weights else None,
                  output_directory))
            conn.commit()
            
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def update_session_stats(self, session_id: str, total_documents: int = 0,
                           successful_documents: int = 0, total_entities: int = 0,
                           successful_entities: int = 0, duration_seconds: int = None):
        """Update session statistics."""
        with self.get_connection() as conn:
            success_rate = (successful_entities / total_entities * 100) if total_entities > 0 else 0.0
            
            conn.execute("""
                UPDATE generation_sessions 
                SET total_documents = ?, successful_documents = ?, 
                    total_entities = ?, successful_entities = ?,
                    overall_success_rate = ?, end_time = CURRENT_TIMESTAMP,
                    duration_seconds = ?
                WHERE session_id = ?
            """, (total_documents, successful_documents, total_entities, 
                  successful_entities, success_rate, duration_seconds, session_id))
            conn.commit()
    
    # ==========================================
    # Document Management
    # ==========================================
    
    def store_document(self, document_id: str, country_code: str, document_type: str,
                      corruption_level: str, original_text: str, corrupted_text: str = None,
                      template_used: str = None, generation_mode: str = 'create-dataset',
                      session_id: str = None) -> int:
        """
        Store a generated document in the database.
        
        Args:
            document_id: Unique document identifier
            country_code: Country code (CL, MX, BR, UY)
            document_type: Type of document (pii_document, negative_example, etc.)
            corruption_level: Corruption level name
            original_text: Original document text
            corrupted_text: Corrupted document text (optional)
            template_used: Template identifier used
            generation_mode: Generation mode
            session_id: Associated session ID
            
        Returns:
            int: Database ID of stored document
        """
        with self.get_connection() as conn:
            # Get foreign key IDs
            country_id = conn.execute("SELECT id FROM countries WHERE code = ?", (country_code,)).fetchone()[0]
            doc_type_id = conn.execute("SELECT id FROM document_types WHERE name = ?", (document_type,)).fetchone()[0]
            corruption_id = conn.execute("SELECT id FROM corruption_levels WHERE name = ?", (corruption_level,)).fetchone()[0]
            
            cursor = conn.execute("""
                INSERT INTO generated_documents 
                (document_id, country_id, document_type_id, corruption_level_id,
                 original_text, corrupted_text, template_used, generation_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (document_id, country_id, doc_type_id, corruption_id,
                  original_text, corrupted_text, template_used, generation_mode))
            
            doc_db_id = cursor.lastrowid
            conn.commit()
            return doc_db_id
    
    def store_entity(self, document_db_id: int, entity_type: str, original_text: str,
                    corrupted_text: str, start_pos: int, end_pos: int,
                    is_preserved: bool = True, confidence: float = 1.0) -> int:
        """
        Store an entity associated with a document.
        
        Args:
            document_db_id: Database ID of the document
            entity_type: Type of entity (CUSTOMER_NAME, ID_NUMBER, etc.)
            original_text: Original entity text
            corrupted_text: Corrupted entity text
            start_pos: Start position in document
            end_pos: End position in document
            is_preserved: Whether entity was preserved after corruption
            confidence: Confidence score
            
        Returns:
            int: Database ID of stored entity
        """
        with self.get_connection() as conn:
            entity_type_id = conn.execute("SELECT id FROM entity_types WHERE name = ?", (entity_type,)).fetchone()[0]
            
            cursor = conn.execute("""
                INSERT INTO document_entities 
                (document_id, entity_type_id, original_text, corrupted_text,
                 start_pos, end_pos, is_preserved, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (document_db_id, entity_type_id, original_text, corrupted_text,
                  start_pos, end_pos, is_preserved, confidence))
            
            entity_id = cursor.lastrowid
            conn.commit()
            return entity_id
    
    def update_document_stats(self, document_db_id: int, total_entities: int,
                            successful_entities: int, failed_entities: int):
        """Update document entity statistics."""
        success_rate = (successful_entities / total_entities * 100) if total_entities > 0 else 0.0
        
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE generated_documents 
                SET total_entities = ?, successful_entities = ?, 
                    failed_entities = ?, success_rate = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (total_entities, successful_entities, failed_entities, success_rate, document_db_id))
            conn.commit()
    
    # ==========================================
    # Query and Analysis Methods
    # ==========================================
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        with self.get_connection() as conn:
            session = conn.execute("""
                SELECT * FROM generation_sessions WHERE session_id = ?
            """, (session_id,)).fetchone()
            
            if not session:
                return None
                
            # Get document counts by type
            doc_counts = conn.execute("""
                SELECT dt.name, COUNT(*) as count
                FROM generated_documents gd
                JOIN document_types dt ON gd.document_type_id = dt.id
                JOIN generation_sessions gs ON gs.session_id = ?
                WHERE gd.created_at >= gs.start_time
                GROUP BY dt.name
            """, (session_id,)).fetchall()
            
            # Get entity statistics
            entity_stats = conn.execute("""
                SELECT et.name, 
                       COUNT(*) as total,
                       SUM(CASE WHEN de.is_preserved THEN 1 ELSE 0 END) as preserved
                FROM document_entities de
                JOIN entity_types et ON de.entity_type_id = et.id
                JOIN generated_documents gd ON de.document_id = gd.id
                JOIN generation_sessions gs ON gs.session_id = ?
                WHERE gd.created_at >= gs.start_time
                GROUP BY et.name
            """, (session_id,)).fetchall()
            
            return {
                'session': dict(session),
                'document_counts': [dict(row) for row in doc_counts],
                'entity_statistics': [dict(row) for row in entity_stats]
            }
    
    def get_country_statistics(self, country_code: str = None, 
                             start_date: date = None, end_date: date = None) -> List[Dict]:
        """Get statistics by country with optional date filtering."""
        with self.get_connection() as conn:
            query = """
                SELECT c.name as country, c.code,
                       COUNT(DISTINCT gd.id) as total_documents,
                       COUNT(de.id) as total_entities,
                       SUM(CASE WHEN de.is_preserved THEN 1 ELSE 0 END) as preserved_entities,
                       ROUND(AVG(gd.success_rate), 2) as avg_success_rate
                FROM countries c
                LEFT JOIN generated_documents gd ON c.id = gd.country_id
                LEFT JOIN document_entities de ON gd.id = de.document_id
                WHERE 1=1
            """
            params = []
            
            if country_code:
                query += " AND c.code = ?"
                params.append(country_code)
                
            if start_date:
                query += " AND DATE(gd.created_at) >= ?"
                params.append(start_date.isoformat())
                
            if end_date:
                query += " AND DATE(gd.created_at) <= ?"
                params.append(end_date.isoformat())
                
            query += " GROUP BY c.id ORDER BY c.name"
            
            return [dict(row) for row in conn.execute(query, params).fetchall()]
    
    def get_corruption_analysis(self) -> List[Dict]:
        """Analyze performance across different corruption levels."""
        with self.get_connection() as conn:
            return [dict(row) for row in conn.execute("""
                SELECT cl.name as corruption_level, cl.level,
                       COUNT(DISTINCT gd.id) as total_documents,
                       COUNT(de.id) as total_entities,
                       SUM(CASE WHEN de.is_preserved THEN 1 ELSE 0 END) as preserved_entities,
                       ROUND(
                           CASE WHEN COUNT(de.id) > 0 
                           THEN (SUM(CASE WHEN de.is_preserved THEN 1 ELSE 0 END) * 100.0 / COUNT(de.id))
                           ELSE 0 END, 2
                       ) as preservation_rate
                FROM corruption_levels cl
                LEFT JOIN generated_documents gd ON cl.id = gd.corruption_level_id
                LEFT JOIN document_entities de ON gd.id = de.document_id
                GROUP BY cl.id
                ORDER BY cl.level
            """).fetchall()]
    
    def search_documents(self, country_code: str = None, document_type: str = None,
                        corruption_level: str = None, has_entities: bool = None,
                        limit: int = 100, offset: int = 0) -> List[Dict]:
        """Search documents with flexible filtering."""
        with self.get_connection() as conn:
            query = """
                SELECT gd.document_id, c.name as country, dt.name as doc_type,
                       cl.name as corruption_level, gd.total_entities, gd.success_rate,
                       gd.created_at
                FROM generated_documents gd
                JOIN countries c ON gd.country_id = c.id
                JOIN document_types dt ON gd.document_type_id = dt.id
                JOIN corruption_levels cl ON gd.corruption_level_id = cl.id
                WHERE 1=1
            """
            params = []
            
            if country_code:
                query += " AND c.code = ?"
                params.append(country_code)
                
            if document_type:
                query += " AND dt.name = ?"
                params.append(document_type)
                
            if corruption_level:
                query += " AND cl.name = ?"
                params.append(corruption_level)
                
            if has_entities is not None:
                if has_entities:
                    query += " AND gd.total_entities > 0"
                else:
                    query += " AND gd.total_entities = 0"
            
            query += " ORDER BY gd.created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            return [dict(row) for row in conn.execute(query, params).fetchall()]
    
    # ==========================================
    # Export and Integration Methods
    # ==========================================
    
    def record_export(self, session_id: str, export_type: str, file_path: str,
                     file_size: int = None, record_count: int = None):
        """Record an export operation."""
        with self.get_connection() as conn:
            # Get session database ID
            session_db_id = conn.execute("""
                SELECT id FROM generation_sessions WHERE session_id = ?
            """, (session_id,)).fetchone()[0]
            
            conn.execute("""
                INSERT INTO dataset_exports 
                (session_id, export_type, file_path, file_size, record_count)
                VALUES (?, ?, ?, ?, ?)
            """, (session_db_id, export_type, file_path, file_size, record_count))
            conn.commit()
    
    def get_export_history(self, session_id: str = None) -> List[Dict]:
        """Get export history with optional session filtering."""
        with self.get_connection() as conn:
            query = """
                SELECT de.export_type, de.file_path, de.file_size, de.record_count,
                       de.export_time, gs.session_id
                FROM dataset_exports de
                JOIN generation_sessions gs ON de.session_id = gs.id
            """
            params = []
            
            if session_id:
                query += " WHERE gs.session_id = ?"
                params.append(session_id)
                
            query += " ORDER BY de.export_time DESC"
            
            return [dict(row) for row in conn.execute(query, params).fetchall()]
    
    def cleanup_old_data(self, days_old: int = 30) -> int:
        """Clean up old data beyond specified days."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM generated_documents 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days_old))
            
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Cleaned up {deleted_count} old documents")
            return deleted_count
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        with self.get_connection() as conn:
            stats = {}
            
            # Table counts
            tables = ['generated_documents', 'document_entities', 'generation_sessions', 'dataset_exports']
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[f'{table}_count'] = count
            
            # Database size
            stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            
            # Latest activity
            latest = conn.execute("""
                SELECT MAX(created_at) FROM generated_documents
            """).fetchone()[0]
            stats['latest_activity'] = latest
            
            return stats

