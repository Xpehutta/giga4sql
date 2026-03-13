import os
import re
import asyncio
from typing import List, Dict, Any, Optional

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_gigachat.chat_models import GigaChat


class LineageOutput(BaseModel):
    """Simplified Pydantic model – only target and sources."""
    target: str = Field(description="Fully qualified target table name")
    sources: List[str] = Field(description="List of fully qualified source table names")


class SQLLineageExtractor:
    """
    SQL lineage extractor using GigaChat LLM and LangChain Expression Language (LCEL).
    Returns only {"target": ..., "sources": [...]} .
    """

    def __init__(
        self,
        credentials: Optional[str] = None,
        model: str = "GigaChat",
        verify_ssl_certs: bool = False,
        scope: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.credentials = credentials or os.getenv("GIGACHAT_API_KEY")
        if not self.credentials:
            raise ValueError(
                "GigaChat API key must be provided via 'credentials' parameter "
                "or set in GIGACHAT_API_KEY environment variable."
            )

        self.model = model
        self.verify_ssl_certs = verify_ssl_certs
        self.scope = scope
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Output parser for simplified model
        self.output_parser = PydanticOutputParser(pydantic_object=LineageOutput)

        # Prompt template (no extra fields asked)
        self.prompt = PromptTemplate(
            template="""You are a SQL lineage extraction expert. Extract source-to-target lineage from the SQL statement below and return **ONLY** a JSON object with two keys: "target" and "sources".

**SQL Statement:**
{sql_text}

**Extraction Rules:**
1. **Target**: The main object being created/modified (fully qualified name).
2. **Sources**: All base tables/views referenced in the query (fully qualified names, no aliases).
3. Strip all quotes (e.g., "schema"."table" → schema.table).
4. Exclude derived tables, CTEs, system tables.
5. Remove duplicates from sources.
6. Use lowercase for all names.

**Output Format:**
{format_instructions}

**Example:**
SQL: INSERT INTO analytics.sales_summary SELECT p.category, SUM(s.amount) FROM products.raw_data p JOIN sales.transactions s ON p.id = s.product_id;
Output: {{"target": "analytics.sales_summary", "sources": ["products.raw_data", "sales.transactions"]}}

**Your Response (JSON only):**""",
            input_variables=["sql_text"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )

        self.llm = GigaChat(
            credentials=self.credentials,
            model=self.model,
            verify_ssl_certs=self.verify_ssl_certs,
            scope=self.scope,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=120
        )

        self.chain = self.prompt | self.llm | self.output_parser

    async def extract_lineage(self, sql_text: str) -> Dict[str, Any]:
        """
        Extract lineage and return a simplified dictionary with 'target' and 'sources'.
        """
        try:
            cleaned_sql = self._clean_sql(sql_text)
            lineage_obj: LineageOutput = await self.chain.ainvoke({"sql_text": cleaned_sql})
            # Return exactly the fields we need
            return {
                "target": lineage_obj.target.lower(),
                "sources": [s.lower() for s in lineage_obj.sources]
            }
        except Exception as e:
            return {
                "error": str(e),
                "target": "",
                "sources": []
            }

    def extract(self, sql_text: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for extract_lineage.
        """
        return asyncio.run(self.extract_lineage(sql_text))

    def _clean_sql(self, sql_text: str) -> str:
        # Remove single-line comments
        sql_text = re.sub(r'--.*$', '', sql_text, flags=re.MULTILINE)
        # Remove multi-line comments
        sql_text = re.sub(r'/\*.*?\*/', '', sql_text, flags=re.DOTALL)
        # Collapse whitespace
        sql_text = re.sub(r'\s+', ' ', sql_text)
        # Remove trailing semicolon
        sql_text = sql_text.strip().rstrip(';')
        return sql_text.strip()

    def batch_extract(self, sql_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple SQL statements concurrently.
        """
        async def batch_process():
            tasks = [self.extract_lineage(sql) for sql in sql_texts]
            return await asyncio.gather(*tasks)

        return asyncio.run(batch_process())