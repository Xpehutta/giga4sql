import re
from typing import Dict

class RegexSQLExtractor:
    def __init__(self):
        # Pattern to match "s_grnplm" prefixed objects with schema and table
        self.pattern = r'\b(s_grnplm[\w\.]+?\.(?:"[^"]+"|[\w\$]+))'
        self.exclude_patterns = [
            r'\(',  # Exclude function calls
            r'\)',  # Exclude function calls
            r'\bAS\b',  # Exclude aliases
            r'\d+s_grnplm',  # Exclude numeric prefixes
            r'\.\*'  # Exclude wildcard selections
        ]

    def extract(self, sql_query) -> Dict:
        # Clean SQL by removing problematic sections
        cleaned_sql = self._clean_sql(sql_query)

        # Identify target table from relevant clauses
        target = self._extract_target(cleaned_sql)

        # Find all valid source tables
        source_matches = re.findall(self.pattern, cleaned_sql, re.IGNORECASE)
        sources = set()

        for match in source_matches:
            if self._is_valid_object(match):
                normalized = match.replace('"', '').lower()
                # Exclude target from sources
                if normalized != target:
                    sources.add(normalized)


        return {
            "target": target,
            "sources": sorted(sources)
        }

    @staticmethod
    def _clean_sql(sql):
        """Remove problematic SQL sections that cause false positives"""
        # Remove string literals
        sql = re.sub(r"'.*?'", "", sql, flags=re.DOTALL)
        # Remove comments
        sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        # Remove numeric suffixes
        sql = re.sub(r"::\w+", "", sql)
        return sql

    def _is_valid_object(self, candidate):
        """Validate if candidate is a proper schema.object"""
        # Check against exclusion patterns
        return not any(re.search(pattern, candidate) for pattern in self.exclude_patterns)

    def _extract_target(self, sql):
        """Extract target table from relevant SQL clauses"""
        # Patterns for different SQL commands (order matters for overlapping matches)
        target_patterns = [
            r'INSERT\s+INTO\s+([^\s\(\)]+)',  # INSERT INTO
            r'CREATE\s+TABLE\b(?:\s+IF\s+NOT\s+EXISTS)?\s+([^\s\(\)]+)',  # CREATE TABLE
            r'MERGE\s+INTO\s+([^\s\(\)]+)',  # MERGE INTO
            r'UPDATE\s+([^\s\(\)]+)'  # UPDATE
        ]

        for pattern in target_patterns:
            match = re.search(pattern, sql, re.IGNORECASE)
            if match:
                target_candidate = match.group(1)
                # Use main pattern to extract valid s_grnplm object
                target_match = re.search(self.pattern, target_candidate, re.IGNORECASE)
                if target_match and self._is_valid_object(target_match.group(0)):
                    return target_match.group(0).replace('"', '').lower()
        return None