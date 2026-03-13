import re
from enum import Enum


class Flavor(Enum):
    ANSI = "ansi"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"


class SQLLineageValidator:
    # Updated patterns: allow letters, digits, underscore, $, #, and hyphen in each name part
    FQN_PATTERNS = {
        Flavor.ANSI: r'^[a-zA-Z0-9_$#-]+\.[a-zA-Z0-9_$#-]+$',  # schema.table
        Flavor.BIGQUERY: r'^([a-zA-Z0-9_$#-]+\.)?[a-zA-Z0-9_$#-]+\.[a-zA-Z0-9_$#-]+$',  # [project.]dataset.table
        Flavor.SNOWFLAKE: r'^[a-zA-Z0-9_$#-]+\.[a-zA-Z0-9_$#-]+\.[a-zA-Z0-9_$#-]+$',  # db.schema.table
    }

    DERIVED_PATTERNS = {
        Flavor.ANSI: [
            r'^t_\d+$', r'^subquery_\d+$', r'^cte_\d+$', r'^.*_alias$',
            r'^t1$', r'^t2$', r'^subquery$', r'^derived$'
        ],
        Flavor.BIGQUERY: [
            r'^t_\d+$', r'^subquery_\d+$', r'^cte_\d+$', r'^.*_alias$',
            r'^t1$', r'^t2$', r'^subquery$', r'^derived$'
        ],
        Flavor.SNOWFLAKE: [
            r'^t_\d+$', r'^subquery_\d+$', r'^cte_\d+$', r'^.*_alias$',
            r'^t1$', r'^t2$', r'^subquery$', r'^derived$'
        ],
    }

    @staticmethod
    def get_flavor(result_dict, default=Flavor.ANSI):
        """Extract flavor from result dict or use default."""
        flavor_str = result_dict.get('flavor', default.value)
        try:
            return Flavor(flavor_str)
        except ValueError:
            return default

    @staticmethod
    def validate_output_format(result):
        """Validate that output follows expected JSON structure (flavor-agnostic)."""
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

    @classmethod
    def validate_target_name(cls, target, flavor=Flavor.ANSI):
        """Validate target name format according to flavor."""
        if not target:
            return False, "Target cannot be empty"

        pattern = cls.FQN_PATTERNS.get(flavor)
        if not pattern:
            return False, f"Unsupported flavor: {flavor}"

        if not re.match(pattern, target):
            expected_format = {
                Flavor.ANSI: "schema.table",
                Flavor.BIGQUERY: "[project.]dataset.table",
                Flavor.SNOWFLAKE: "database.schema.table"
            }.get(flavor, "fully qualified name")
            return False, f"Target '{target}' does not match expected format for {flavor.value}: {expected_format}"

        return True, "Valid target name"

    @classmethod
    def validate_source_names(cls, sources, flavor=Flavor.ANSI):
        """Validate all source names according to flavor."""
        if not sources:
            return True, "No sources (valid case)"

        errors = []
        pattern = cls.FQN_PATTERNS.get(flavor)
        if not pattern:
            return False, f"Unsupported flavor: {flavor}"

        for i, source in enumerate(sources):
            if not isinstance(source, str):
                errors.append(f"Source {i} is not a string: {source}")
                continue
            if not source:
                errors.append(f"Source {i} is empty")
                continue
            if not re.match(pattern, source):
                expected_format = {
                    Flavor.ANSI: "schema.table",
                    Flavor.BIGQUERY: "[project.]dataset.table",
                    Flavor.SNOWFLAKE: "database.schema.table"
                }.get(flavor, "fully qualified name")
                errors.append(f"Source '{source}' does not match expected format for {flavor.value}: {expected_format}")

        if errors:
            return False, "; ".join(errors)
        return True, "All source names valid"

    @classmethod
    def validate_no_derived_tables(cls, sources, target=None, flavor=Flavor.ANSI):
        """Validate no derived tables (aliases) in sources based on flavor patterns."""
        patterns = cls.DERIVED_PATTERNS.get(flavor, cls.DERIVED_PATTERNS[Flavor.ANSI])

        errors = []
        for source in sources:
            for pattern in patterns:
                if re.match(pattern, source.lower()):
                    errors.append(f"Derived table detected: {source}")
                    break

        if target and target in sources:
            errors.append(f"Target '{target}' should not appear in sources")

        if errors:
            return False, "; ".join(errors)
        return True, "No derived tables detected"

    @staticmethod
    def validate_unique_sources(sources):
        """Validate sources are unique (flavor-agnostic)."""
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

    @classmethod
    def validate_fully_qualified_names(cls, names, flavor=Flavor.ANSI):
        """Validate all names are fully qualified per flavor."""
        errors = []
        pattern = cls.FQN_PATTERNS.get(flavor)
        if not pattern:
            return False, f"Unsupported flavor: {flavor}"

        for name in names:
            if not name or not re.match(pattern, name):
                errors.append(f"Name '{name}' is not fully qualified for {flavor.value}")

        if errors:
            return False, "; ".join(errors)
        return True, "All names are fully qualified"

    @classmethod
    def normalize_name(cls, name, flavor):
        """Convert a flavor-specific fully qualified name to a canonical form for comparison."""
        # Extend this method if needed for case-insensitivity or project stripping
        return name

    @classmethod
    def calculate_precision_recall_f1(cls, expected, actual):
        """Calculate metrics after normalizing names based on flavors."""
        exp_flavor = cls.get_flavor(expected)
        act_flavor = cls.get_flavor(actual)

        exp_target = cls.normalize_name(expected.get('target', ''), exp_flavor)
        act_target = cls.normalize_name(actual.get('target', ''), act_flavor)

        exp_sources = {cls.normalize_name(s, exp_flavor) for s in expected.get('sources', [])}
        act_sources = {cls.normalize_name(s, act_flavor) for s in actual.get('sources', [])}

        if not exp_sources and not act_sources:
            return 1.0, 1.0, 1.0

        tp = len(exp_sources & act_sources)
        fp = len(act_sources - exp_sources)
        fn = len(exp_sources - act_sources)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    @classmethod
    async def run_comprehensive_validation(cls, extractor, sql_query, expected_result=None, flavor=None):
        """
        Run comprehensive validation on extracted results.
        If flavor is None, it will be inferred from expected_result or default to ANSI.
        """
        result = await extractor.extract_lineage(sql_query)

        # Determine flavor: use provided, or from expected_result, or from result, or default
        if flavor is None:
            if expected_result and 'flavor' in expected_result:
                flavor = cls.get_flavor(expected_result)
            elif 'flavor' in result:
                flavor = cls.get_flavor(result)
            else:
                flavor = Flavor.ANSI
        else:
            flavor = Flavor(flavor) if isinstance(flavor, str) else flavor

        # Optionally add flavor to result if missing
        if 'flavor' not in result:
            result['flavor'] = flavor.value

        # Format validation
        is_valid, message = cls.validate_output_format(result)
        if not is_valid:
            return {"status": "FAILED", "validation_type": "format", "message": message, "result": result}

        # Target validation
        is_valid, message = cls.validate_target_name(result["target"], flavor)
        if not is_valid:
            return {"status": "FAILED", "validation_type": "target", "message": message, "result": result}

        # Source validation
        is_valid, message = cls.validate_source_names(result["sources"], flavor)
        if not is_valid:
            return {"status": "FAILED", "validation_type": "sources", "message": message, "result": result}

        # Derived tables validation
        is_valid, message = cls.validate_no_derived_tables(result["sources"], result["target"], flavor)
        if not is_valid:
            return {"status": "FAILED", "validation_type": "derived_tables", "message": message, "result": result}

        # Uniqueness validation
        is_valid, message = cls.validate_unique_sources(result["sources"])
        if not is_valid:
            return {"status": "FAILED", "validation_type": "uniqueness", "message": message, "result": result}

        # Fully qualified validation
        all_names = [result["target"]] + result["sources"]
        is_valid, message = cls.validate_fully_qualified_names(all_names, flavor)
        if not is_valid:
            return {"status": "FAILED", "validation_type": "qualification", "message": message, "result": result}

        # If expected result provided, calculate metrics
        if expected_result:
            precision, recall, f1 = cls.calculate_precision_recall_f1(expected_result, result)
            return {
                "status": "SUCCESS",
                "validation_type": "comprehensive",
                "message": "All validations passed",
                "result": result,
                "metrics": {"precision": precision, "recall": recall, "f1_score": f1}
            }

        return {
            "status": "SUCCESS",
            "validation_type": "comprehensive",
            "message": "All validations passed",
            "result": result
        }