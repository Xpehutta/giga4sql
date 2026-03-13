import re

class SQLLineageValidator:

    @staticmethod
    def validate_output_format(result):
        """Validate that output follows expected JSON structure"""
        if not isinstance(result, dict):
            return False, "Result should be a dictionary"

        if "target" not in result:
            return False, "Missing 'target' field"

        if "sources" not in result:
            return False, "Missing 'sources' field"

        if not isinstance(result["target"], str):
            return False, "'target' should be a string"

        if not isinstance(result["sources"], list):
            return False, "'sources' should be a list"

        return True, "Valid format"

    @staticmethod
    def validate_target_name(target):
        """Validate target name format"""
        if not target:
            return False, "Target cannot be empty"

        # Check if it's fully qualified (schema.table)
        parts = target.split('.')
        if len(parts) != 2:
            return False, f"Target '{target}' should be fully qualified (schema.table)"

        schema, table = parts
        if not schema or not table:
            return False, f"Target '{target}' has missing schema or table name"

        return True, "Valid target name"

    @staticmethod
    def validate_source_names(sources):
        """Validate all source names"""
        if not sources:
            return True, "No sources (valid case)"

        errors = []
        for i, source in enumerate(sources):
            if not isinstance(source, str):
                errors.append(f"Source {i} is not a string: {source}")
                continue

            if not source:
                errors.append(f"Source {i} is empty")
                continue

            # Check if it's fully qualified (schema.table)
            parts = source.split('.')
            if len(parts) != 2:
                errors.append(f"Source '{source}' should be fully qualified (schema.table)")
                continue

            schema, table = parts
            if not schema or not table:
                errors.append(f"Source '{source}' has missing schema or table name")
                continue

        if errors:
            return False, "; ".join(errors)

        return True, "All source names valid"

    @staticmethod
    def validate_no_derived_tables(sources, target=None):
        """Validate no derived tables (aliases) in sources"""
        derived_patterns = [
            r'^t_\d+$',  # Simple alias pattern
            r'^subquery_\d+$',  # Subquery aliases
            r'^cte_\d+$',  # CTE aliases
            r'^.*_alias$',  # Generic alias pattern
        ]

        errors = []
        for source in sources:
            # Check if source looks like a derived table
            for pattern in derived_patterns:
                if re.match(pattern, source.lower()):
                    errors.append(f"Derived table detected: {source}")
                    break

            # Check for common derived table patterns
            if source.lower() in ['t1', 't2', 'subquery', 'derived']:
                errors.append(f"Derived table detected: {source}")

        if target and target in sources:
            errors.append(f"Target '{target}' should not appear in sources")

        if errors:
            return False, "; ".join(errors)

        return True, "No derived tables detected"

    @staticmethod
    def validate_unique_sources(sources):
        """Validate sources are unique"""
        seen = set()
        duplicates = []

        for source in sources:
            if source.lower() in seen:
                duplicates.append(source)
            else:
                seen.add(source.lower())

        if duplicates:
            return False, f"Duplicate sources found: {duplicates}"

        return True, "All sources are unique"

    @staticmethod
    def validate_fully_qualified_names(names):
        """Validate all names are fully qualified"""
        errors = []
        for name in names:
            if not name or '.' not in name:
                errors.append(f"Name '{name}' is not fully qualified")

        if errors:
            return False, "; ".join(errors)

        return True, "All names are fully qualified"

    @staticmethod
    def calculate_precision_recall_f1(expected, actual):
        """Calculate precision, recall, and F1 score for lineage extraction"""
        expected_sources = set(expected.get('sources', []))
        actual_sources = set(actual.get('sources', []))

        # Handle case where both are empty
        if not expected_sources and not actual_sources:
            return 1.0, 1.0, 1.0

        # Calculate true positives, false positives, false negatives
        tp = len(expected_sources.intersection(actual_sources))
        fp = len(actual_sources - expected_sources)
        fn = len(expected_sources - actual_sources)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    @staticmethod
    def run_comprehensive_validation(extractor, sql_query, expected_result=None):
        """Run comprehensive validation on extracted results"""
        result = extractor.extract(sql_query)

        # Format validation
        is_valid, message = SQLLineageValidator.validate_output_format(result)
        if not is_valid:
            return {
                "status": "FAILED",
                "validation_type": "format",
                "message": message,
                "result": result
            }

        # Target validation
        is_valid, message = SQLLineageValidator.validate_target_name(result["target"])
        if not is_valid:
            return {
                "status": "FAILED",
                "validation_type": "target",
                "message": message,
                "result": result
            }

        # Source validation
        is_valid, message = SQLLineageValidator.validate_source_names(result["sources"])
        if not is_valid:
            return {
                "status": "FAILED",
                "validation_type": "sources",
                "message": message,
                "result": result
            }

        # Derived tables validation
        is_valid, message = SQLLineageValidator.validate_no_derived_tables(
            result["sources"],
            result["target"]
        )
        if not is_valid:
            return {
                "status": "FAILED",
                "validation_type": "derived_tables",
                "message": message,
                "result": result
            }

        # Uniqueness validation
        is_valid, message = SQLLineageValidator.validate_unique_sources(result["sources"])
        if not is_valid:
            return {
                "status": "FAILED",
                "validation_type": "uniqueness",
                "message": message,
                "result": result
            }

        # Fully qualified validation
        all_names = [result["target"]] + result["sources"]
        is_valid, message = SQLLineageValidator.validate_fully_qualified_names(all_names)
        if not is_valid:
            return {
                "status": "FAILED",
                "validation_type": "qualification",
                "message": message,
                "result": result
            }

        # If expected result provided, calculate metrics
        if expected_result:
            precision, recall, f1 = SQLLineageValidator.calculate_precision_recall_f1(
                expected_result, result
            )
            return {
                "status": "SUCCESS",
                "validation_type": "comprehensive",
                "message": "All validations passed",
                "result": result,
                "metrics": {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                }
            }

        return {
            "status": "SUCCESS",
            "validation_type": "comprehensive",
            "message": "All validations passed",
            "result": result
        }