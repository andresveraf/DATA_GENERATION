"""
Negative Examples Generator for NER Training
===========================================

This module generates realistic business documents and text samples that contain NO PII entities.
These negative examples are crucial for training robust NER models that can distinguish between
PII and non-PII content in real-world documents.

Key Features:
- Business document templates (invoices, reports, forms)
- Industry-specific text generation (financial, legal, administrative)
- OCR noise application to negative examples
- Validation system to ensure zero PII entities
- Multi-language support (Spanish, Portuguese)
- Integration with main data generation pipeline

Supported Document Types:
- Business invoices and receipts
- Financial reports and statements
- Legal documents and contracts
- Administrative forms and notices
- Technical documentation
- Marketing materials

Author: Andrés Vera Figueroa
Date: October 2024
Purpose: Generate negative training examples for robust NER model training
"""

import random
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

class NegativeExamplesGenerator:
    """
    Generator for realistic business documents without PII entities.
    
    Creates diverse text samples that resemble real business documents
    but contain no personally identifiable information.
    """
    
    def __init__(self, language: str = 'es'):
        """
        Initialize the negative examples generator.
        
        Args:
            language (str): Language code ('es' for Spanish, 'pt' for Portuguese)
        """
        self.language = language
        self.templates = self._load_templates()
        self.business_terms = self._load_business_terms()
        self.pii_patterns = self._load_pii_patterns()
        
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load document templates by category."""
        if self.language == 'es':
            return {
                'invoice_headers': [
                    "FACTURA COMERCIAL",
                    "COMPROBANTE DE VENTA",
                    "DOCUMENTO TRIBUTARIO",
                    "BOLETA DE VENTA",
                    "NOTA DE VENTA"
                ],
                'business_documents': [
                    "INFORME MENSUAL DE ACTIVIDADES",
                    "REPORTE DE GESTIÓN COMERCIAL",
                    "ANÁLISIS DE MERCADO",
                    "ESTUDIO DE FACTIBILIDAD",
                    "PROPUESTA COMERCIAL"
                ],
                'administrative_forms': [
                    "FORMULARIO DE SOLICITUD",
                    "DECLARACIÓN JURADA",
                    "CERTIFICADO DE CUMPLIMIENTO",
                    "ACTA DE REUNIÓN",
                    "PROTOCOLO DE PROCEDIMIENTOS"
                ],
                'legal_documents': [
                    "CONTRATO DE SERVICIOS",
                    "TÉRMINOS Y CONDICIONES",
                    "POLÍTICA DE PRIVACIDAD",
                    "REGLAMENTO INTERNO",
                    "CÓDIGO DE CONDUCTA"
                ]
            }
        else:  # Portuguese
            return {
                'invoice_headers': [
                    "NOTA FISCAL",
                    "COMPROVANTE DE VENDA",
                    "DOCUMENTO FISCAL",
                    "RECIBO DE PAGAMENTO",
                    "CUPOM FISCAL"
                ],
                'business_documents': [
                    "RELATÓRIO MENSAL DE ATIVIDADES",
                    "RELATÓRIO DE GESTÃO COMERCIAL",
                    "ANÁLISE DE MERCADO",
                    "ESTUDO DE VIABILIDADE",
                    "PROPOSTA COMERCIAL"
                ],
                'administrative_forms': [
                    "FORMULÁRIO DE SOLICITAÇÃO",
                    "DECLARAÇÃO",
                    "CERTIFICADO DE CONFORMIDADE",
                    "ATA DE REUNIÃO",
                    "PROTOCOLO DE PROCEDIMENTOS"
                ],
                'legal_documents': [
                    "CONTRATO DE SERVIÇOS",
                    "TERMOS E CONDIÇÕES",
                    "POLÍTICA DE PRIVACIDADE",
                    "REGULAMENTO INTERNO",
                    "CÓDIGO DE CONDUTA"
                ]
            }
    
    def _load_business_terms(self) -> Dict[str, List[str]]:
        """Load business terminology by category."""
        if self.language == 'es':
            return {
                'departments': [
                    "Departamento de Ventas", "Área Comercial", "División Marketing",
                    "Gerencia General", "Recursos Humanos", "Contabilidad",
                    "Administración", "Logística", "Producción", "Calidad"
                ],
                'products': [
                    "Producto A", "Artículo B", "Servicio Premium", "Plan Básico",
                    "Paquete Estándar", "Solución Integral", "Módulo Principal",
                    "Componente Adicional", "Licencia Software", "Mantenimiento"
                ],
                'processes': [
                    "Proceso de Calidad", "Procedimiento Estándar", "Metodología Ágil",
                    "Sistema de Gestión", "Control de Inventario", "Auditoría Interna",
                    "Evaluación de Desempeño", "Análisis de Riesgos", "Plan Estratégico"
                ],
                'metrics': [
                    "Indicador de Gestión", "KPI Principal", "Métrica de Calidad",
                    "Índice de Satisfacción", "Ratio de Conversión", "Tasa de Retención",
                    "Margen de Contribución", "ROI del Proyecto", "Eficiencia Operacional"
                ]
            }
        else:  # Portuguese
            return {
                'departments': [
                    "Departamento de Vendas", "Área Comercial", "Divisão Marketing",
                    "Gerência Geral", "Recursos Humanos", "Contabilidade",
                    "Administração", "Logística", "Produção", "Qualidade"
                ],
                'products': [
                    "Produto A", "Artigo B", "Serviço Premium", "Plano Básico",
                    "Pacote Padrão", "Solução Integral", "Módulo Principal",
                    "Componente Adicional", "Licença Software", "Manutenção"
                ],
                'processes': [
                    "Processo de Qualidade", "Procedimento Padrão", "Metodologia Ágil",
                    "Sistema de Gestão", "Controle de Estoque", "Auditoria Interna",
                    "Avaliação de Desempenho", "Análise de Riscos", "Plano Estratégico"
                ],
                'metrics': [
                    "Indicador de Gestão", "KPI Principal", "Métrica de Qualidade",
                    "Índice de Satisfação", "Taxa de Conversão", "Taxa de Retenção",
                    "Margem de Contribuição", "ROI do Projeto", "Eficiência Operacional"
                ]
            }
    
    def _load_pii_patterns(self) -> List[str]:
        """Load regex patterns to detect and avoid PII entities."""
        return [
            # Names patterns (to avoid)
            r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b',
            # Email patterns
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            # Phone patterns
            r'\+?[\d\s\-\(\)]{8,15}',
            # ID number patterns (RUT, CPF, etc.)
            r'\b\d{1,2}\.?\d{3}\.?\d{3}[-\.]?[\dkK]\b',
            r'\b\d{3}\.?\d{3}\.?\d{3}[-\.]?\d{2}\b',
            # Address patterns (street numbers + names)
            r'\b\d+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ\s]+\d*\b',
            # Amount patterns with currency
            r'\$\s*[\d,\.]+|\b\d+[,\.]\d{2}\s*(CLP|MXN|BRL|UYU|USD)\b'
        ]
    
    def generate_business_invoice(self) -> str:
        """Generate a business invoice without PII."""
        header = random.choice(self.templates['invoice_headers'])
        
        if self.language == 'es':
            template = f"""
{header}

Número de Documento: DOC-{random.randint(100000, 999999)}
Fecha de Emisión: {self._generate_business_date()}
Tipo de Documento: Factura Comercial

DETALLE DE PRODUCTOS Y SERVICIOS:

Código    Descripción                    Cantidad    Precio Unit.    Total
------------------------------------------------------------------------
{self._generate_product_line()}
{self._generate_product_line()}
{self._generate_product_line()}

                                        SUBTOTAL:    ${random.randint(50000, 500000):,}
                                        IVA (19%):   ${random.randint(9500, 95000):,}
                                        TOTAL:       ${random.randint(59500, 595000):,}

CONDICIONES COMERCIALES:
- Forma de Pago: Transferencia Bancaria
- Plazo de Pago: 30 días desde la fecha de emisión
- Validez de la Oferta: 15 días calendario

OBSERVACIONES:
Los precios incluyen todos los impuestos aplicables según la legislación vigente.
Este documento cumple con las normativas tributarias establecidas.
            """.strip()
        else:  # Portuguese
            template = f"""
{header}

Número do Documento: DOC-{random.randint(100000, 999999)}
Data de Emissão: {self._generate_business_date()}
Tipo de Documento: Nota Fiscal Comercial

DETALHAMENTO DE PRODUTOS E SERVIÇOS:

Código    Descrição                      Quantidade  Preço Unit.     Total
------------------------------------------------------------------------
{self._generate_product_line()}
{self._generate_product_line()}
{self._generate_product_line()}

                                        SUBTOTAL:    R$ {random.randint(5000, 50000):,}
                                        ICMS (18%):  R$ {random.randint(900, 9000):,}
                                        TOTAL:       R$ {random.randint(5900, 59000):,}

CONDIÇÕES COMERCIAIS:
- Forma de Pagamento: Transferência Bancária
- Prazo de Pagamento: 30 dias da data de emissão
- Validade da Proposta: 15 dias corridos

OBSERVAÇÕES:
Os preços incluem todos os impostos aplicáveis conforme legislação vigente.
Este documento atende às normas tributárias estabelecidas.
            """.strip()
        
        return template
    
    def generate_business_report(self) -> str:
        """Generate a business report without PII."""
        title = random.choice(self.templates['business_documents'])
        
        if self.language == 'es':
            template = f"""
{title}

Período de Análisis: {self._generate_period()}
Fecha de Elaboración: {self._generate_business_date()}
Versión del Documento: v{random.randint(1, 5)}.{random.randint(0, 9)}

RESUMEN EJECUTIVO:

Durante el período analizado se observó un crecimiento sostenido en los principales 
indicadores de gestión. {random.choice(self.business_terms['departments'])} reportó 
un incremento del {random.randint(5, 25)}% en las métricas establecidas.

INDICADORES PRINCIPALES:

• {random.choice(self.business_terms['metrics'])}: {random.randint(85, 98)}%
• {random.choice(self.business_terms['metrics'])}: {random.randint(75, 95)}%
• {random.choice(self.business_terms['metrics'])}: {random.randint(80, 100)}%

ANÁLISIS POR ÁREA:

{random.choice(self.business_terms['departments'])}:
- Cumplimiento de objetivos: {random.randint(90, 100)}%
- Eficiencia operacional: {random.randint(85, 95)}%
- Satisfacción interna: {random.randint(80, 90)}%

RECOMENDACIONES:

1. Implementar {random.choice(self.business_terms['processes'])}
2. Optimizar {random.choice(self.business_terms['products'])}
3. Fortalecer {random.choice(self.business_terms['departments'])}

CONCLUSIONES:

Los resultados obtenidos demuestran la efectividad de las estrategias implementadas
y confirman el cumplimiento de los objetivos establecidos para este período.
            """.strip()
        else:  # Portuguese
            template = f"""
{title}

Período de Análise: {self._generate_period()}
Data de Elaboração: {self._generate_business_date()}
Versão do Documento: v{random.randint(1, 5)}.{random.randint(0, 9)}

RESUMO EXECUTIVO:

Durante o período analisado observou-se um crescimento sustentado nos principais 
indicadores de gestão. {random.choice(self.business_terms['departments'])} reportou 
um incremento de {random.randint(5, 25)}% nas métricas estabelecidas.

INDICADORES PRINCIPAIS:

• {random.choice(self.business_terms['metrics'])}: {random.randint(85, 98)}%
• {random.choice(self.business_terms['metrics'])}: {random.randint(75, 95)}%
• {random.choice(self.business_terms['metrics'])}: {random.randint(80, 100)}%

ANÁLISE POR ÁREA:

{random.choice(self.business_terms['departments'])}:
- Cumprimento de objetivos: {random.randint(90, 100)}%
- Eficiência operacional: {random.randint(85, 95)}%
- Satisfação interna: {random.randint(80, 90)}%

RECOMENDAÇÕES:

1. Implementar {random.choice(self.business_terms['processes'])}
2. Otimizar {random.choice(self.business_terms['products'])}
3. Fortalecer {random.choice(self.business_terms['departments'])}

CONCLUSÕES:

Os resultados obtidos demonstram a efetividade das estratégias implementadas
e confirmam o cumprimento dos objetivos estabelecidos para este período.
            """.strip()
        
        return template
    
    def generate_administrative_form(self) -> str:
        """Generate an administrative form without PII."""
        form_type = random.choice(self.templates['administrative_forms'])
        
        if self.language == 'es':
            template = f"""
{form_type}

Número de Formulario: FORM-{random.randint(1000, 9999)}
Fecha de Creación: {self._generate_business_date()}
Estado del Documento: Activo

INFORMACIÓN GENERAL:

Tipo de Solicitud: {random.choice(self.business_terms['processes'])}
Área Responsable: {random.choice(self.business_terms['departments'])}
Prioridad: {random.choice(['Alta', 'Media', 'Baja'])}
Categoría: {random.choice(['Operacional', 'Administrativa', 'Técnica'])}

DESCRIPCIÓN DEL REQUERIMIENTO:

Se solicita la implementación de mejoras en {random.choice(self.business_terms['products'])}
con el objetivo de optimizar {random.choice(self.business_terms['metrics'])}.

ESPECIFICACIONES TÉCNICAS:

• Alcance del proyecto: {random.choice(['Nacional', 'Regional', 'Local'])}
• Duración estimada: {random.randint(1, 12)} meses
• Recursos requeridos: {random.choice(['Básicos', 'Intermedios', 'Avanzados'])}
• Presupuesto aproximado: ${random.randint(100000, 1000000):,}

CRITERIOS DE ACEPTACIÓN:

1. Cumplimiento de normativas vigentes
2. Integración con sistemas existentes
3. Capacitación del personal involucrado
4. Documentación técnica completa

APROBACIONES REQUERIDAS:

□ Gerencia Técnica
□ Administración
□ Control de Calidad
□ Finanzas

Observaciones: Este formulario debe ser completado siguiendo los procedimientos
establecidos en el manual de gestión organizacional.
            """.strip()
        else:  # Portuguese
            template = f"""
{form_type}

Número do Formulário: FORM-{random.randint(1000, 9999)}
Data de Criação: {self._generate_business_date()}
Status do Documento: Ativo

INFORMAÇÕES GERAIS:

Tipo de Solicitação: {random.choice(self.business_terms['processes'])}
Área Responsável: {random.choice(self.business_terms['departments'])}
Prioridade: {random.choice(['Alta', 'Média', 'Baixa'])}
Categoria: {random.choice(['Operacional', 'Administrativa', 'Técnica'])}

DESCRIÇÃO DO REQUERIMENTO:

Solicita-se a implementação de melhorias em {random.choice(self.business_terms['products'])}
com o objetivo de otimizar {random.choice(self.business_terms['metrics'])}.

ESPECIFICAÇÕES TÉCNICAS:

• Escopo do projeto: {random.choice(['Nacional', 'Regional', 'Local'])}
• Duração estimada: {random.randint(1, 12)} meses
• Recursos necessários: {random.choice(['Básicos', 'Intermediários', 'Avançados'])}
• Orçamento aproximado: R$ {random.randint(10000, 100000):,}

CRITÉRIOS DE ACEITAÇÃO:

1. Cumprimento de normativas vigentes
2. Integração com sistemas existentes
3. Capacitação do pessoal envolvido
4. Documentação técnica completa

APROVAÇÕES NECESSÁRIAS:

□ Gerência Técnica
□ Administração
□ Controle de Qualidade
□ Finanças

Observações: Este formulário deve ser preenchido seguindo os procedimentos
estabelecidos no manual de gestão organizacional.
            """.strip()
        
        return template
    
    def generate_legal_document(self) -> str:
        """Generate a legal document template without PII."""
        doc_type = random.choice(self.templates['legal_documents'])
        
        if self.language == 'es':
            template = f"""
{doc_type}

Documento Legal N°: LEG-{random.randint(10000, 99999)}
Fecha de Vigencia: {self._generate_business_date()}
Versión: {random.randint(1, 10)}.0

CLÁUSULAS GENERALES:

PRIMERA: OBJETO DEL DOCUMENTO
El presente documento establece las condiciones generales aplicables a 
{random.choice(self.business_terms['products'])} ofrecidos por la organización.

SEGUNDA: ALCANCE Y APLICACIÓN
Las disposiciones contenidas en este documento son de aplicación general
para todas las operaciones relacionadas con {random.choice(self.business_terms['processes'])}.

TERCERA: RESPONSABILIDADES
{random.choice(self.business_terms['departments'])} será responsable de
garantizar el cumplimiento de las normativas establecidas.

CUARTA: VIGENCIA
Este documento entrará en vigencia a partir de su publicación y permanecerá
activo hasta nueva disposición de la autoridad competente.

QUINTA: MODIFICACIONES
Cualquier modificación a este documento deberá ser aprobada por
{random.choice(self.business_terms['departments'])} y comunicada oportunamente.

DISPOSICIONES FINALES:

Para efectos de interpretación de este documento, se aplicarán las normas
legales vigentes en la jurisdicción correspondiente.

Las controversias que pudieran surgir serán resueltas mediante los mecanismos
establecidos en la legislación aplicable.

Este documento ha sido elaborado conforme a los estándares legales requeridos
y cuenta con las validaciones técnicas correspondientes.
            """.strip()
        else:  # Portuguese
            template = f"""
{doc_type}

Documento Legal N°: LEG-{random.randint(10000, 99999)}
Data de Vigência: {self._generate_business_date()}
Versão: {random.randint(1, 10)}.0

CLÁUSULAS GERAIS:

PRIMEIRA: OBJETO DO DOCUMENTO
O presente documento estabelece as condições gerais aplicáveis a 
{random.choice(self.business_terms['products'])} oferecidos pela organização.

SEGUNDA: ESCOPO E APLICAÇÃO
As disposições contidas neste documento são de aplicação geral
para todas as operações relacionadas com {random.choice(self.business_terms['processes'])}.

TERCEIRA: RESPONSABILIDADES
{random.choice(self.business_terms['departments'])} será responsável por
garantir o cumprimento das normativas estabelecidas.

QUARTA: VIGÊNCIA
Este documento entrará em vigência a partir de sua publicação e permanecerá
ativo até nova disposição da autoridade competente.

QUINTA: MODIFICAÇÕES
Qualquer modificação a este documento deverá ser aprovada por
{random.choice(self.business_terms['departments'])} e comunicada oportunamente.

DISPOSIÇÕES FINAIS:

Para efeitos de interpretação deste documento, aplicar-se-ão as normas
legais vigentes na jurisdição correspondente.

As controvérsias que possam surgir serão resolvidas mediante os mecanismos
estabelecidos na legislação aplicável.

Este documento foi elaborado conforme os padrões legais requeridos
e conta com as validações técnicas correspondentes.
            """.strip()
        
        return template
    
    def _generate_business_date(self) -> str:
        """Generate a realistic business date."""
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        random_date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        return random_date.strftime("%d/%m/%Y")
    
    def _generate_period(self) -> str:
        """Generate a business period description."""
        if self.language == 'es':
            periods = [
                "Enero - Marzo 2024", "Abril - Junio 2024", "Julio - Septiembre 2024",
                "Octubre - Diciembre 2024", "Primer Semestre 2024", "Segundo Semestre 2024",
                "Trimestre I 2024", "Trimestre II 2024", "Año Fiscal 2024"
            ]
        else:
            periods = [
                "Janeiro - Março 2024", "Abril - Junho 2024", "Julho - Setembro 2024",
                "Outubro - Dezembro 2024", "Primeiro Semestre 2024", "Segundo Semestre 2024",
                "Trimestre I 2024", "Trimestre II 2024", "Ano Fiscal 2024"
            ]
        return random.choice(periods)
    
    def _generate_product_line(self) -> str:
        """Generate a product line for invoices."""
        code = f"PROD{random.randint(100, 999)}"
        product = random.choice(self.business_terms['products'])
        quantity = random.randint(1, 100)
        unit_price = random.randint(1000, 50000)
        total = quantity * unit_price
        
        return f"{code:<8} {product:<30} {quantity:<11} ${unit_price:<11,} ${total:,}"
    
    def validate_no_pii(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate that text contains no PII entities.
        
        Args:
            text (str): Text to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_potential_pii_found)
        """
        potential_pii = []
        
        for pattern in self.pii_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                potential_pii.extend(matches)
        
        # Additional checks for common PII indicators
        words = text.split()
        for i, word in enumerate(words):
            # Check for potential names (capitalized words that might be names)
            if (word.istitle() and len(word) > 2 and 
                i < len(words) - 1 and words[i + 1].istitle() and
                word not in ['Departamento', 'Área', 'División', 'Gerencia', 'Producto', 'Servicio']):
                potential_pii.append(f"{word} {words[i + 1]}")
        
        return len(potential_pii) == 0, potential_pii
    
    def generate_negative_example(self, document_type: str = None) -> Dict[str, Any]:
        """
        Generate a complete negative example document.
        
        Args:
            document_type (str): Type of document to generate
            
        Returns:
            Dict[str, Any]: Generated document with metadata
        """
        if document_type is None:
            document_type = random.choice(['invoice', 'report', 'form', 'legal'])
        
        generators = {
            'invoice': self.generate_business_invoice,
            'report': self.generate_business_report,
            'form': self.generate_administrative_form,
            'legal': self.generate_legal_document
        }
        
        if document_type not in generators:
            document_type = 'report'
        
        # Generate document
        text = generators[document_type]()
        
        # Validate no PII
        is_valid, potential_pii = self.validate_no_pii(text)
        
        # If PII detected, regenerate with more generic terms
        if not is_valid:
            text = self._sanitize_text(text, potential_pii)
            is_valid, potential_pii = self.validate_no_pii(text)
        
        return {
            'text': text,
            'document_type': document_type,
            'language': self.language,
            'is_valid': is_valid,
            'potential_pii': potential_pii,
            'word_count': len(text.split()),
            'generated_at': datetime.now().isoformat()
        }
    
    def _sanitize_text(self, text: str, potential_pii: List[str]) -> str:
        """Remove or replace potential PII from text."""
        sanitized = text
        
        for pii in potential_pii:
            # Replace with generic terms
            if '@' in pii:  # Email
                sanitized = sanitized.replace(pii, 'correo@empresa.com')
            elif any(char.isdigit() for char in pii):  # Numbers/IDs
                sanitized = sanitized.replace(pii, 'DOCUMENTO-123456')
            else:  # Potential names
                sanitized = sanitized.replace(pii, 'EMPRESA COMERCIAL')
        
        return sanitized
    
    def generate_batch(self, count: int = 100, document_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generate a batch of negative examples.
        
        Args:
            count (int): Number of documents to generate
            document_types (List[str]): Types of documents to generate
            
        Returns:
            List[Dict[str, Any]]: List of generated documents
        """
        if document_types is None:
            document_types = ['invoice', 'report', 'form', 'legal']
        
        documents = []
        for _ in range(count):
            doc_type = random.choice(document_types)
            doc = self.generate_negative_example(doc_type)
            documents.append(doc)
        
        return documents

