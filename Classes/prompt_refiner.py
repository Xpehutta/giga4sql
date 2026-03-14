import os
import re
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, TypedDict
from numpy import mean

from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_gigachat import GigaChat

# Import your existing core classes – adjust paths as needed
from Classes.model_classes import SQLLineageExtractor, LineageOutput
from Classes.validation_classes import SQLLineageValidator


class AgentState(TypedDict):
    sql: str
    current_prompt: str
    validation_result: Dict[str, Any]
    iteration: int
    max_iterations: int
    refined_prompts: List[str]
    validation_history: List[Dict[str, Any]]
    expected_result: Optional[Dict[str, Any]]
    should_continue: bool


class GigaChatSQLLineageAgent:
    """
    Prompt optimization agent using GigaChat (via langchain_gigachat).
    Fully async – use await agent.optimize_prompt(...) in async code.
    For synchronous environments, use agent.optimize_prompt_sync(...)
    (but beware of nested event loops).
    """

    def __init__(
        self,
        credentials: Optional[str] = None,
        model: str = "GigaChat",
        verify_ssl_certs: bool = False,
        scope: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.01,
        max_tokens: int = 2048,
        max_iterations: int = 5,
        base_extraction_template: Optional[str] = None,
        extractor: Optional[SQLLineageExtractor] = None
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
        self.max_iterations = min(max_iterations, 5)

        # 1. GigaChat LLM for reflection
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

        # 2. Extractor management
        self.extractor = extractor
        if self.extractor is None:
            self.extractor = SQLLineageExtractor(
                credentials=self.credentials,
                model=self.model,
                verify_ssl_certs=self.verify_ssl_certs,
                scope=self.scope,
                base_url=self.base_url,
                temperature=0.1,
                max_tokens=self.max_tokens,
            )
            self.base_extraction_template = self.extractor.prompt.template
        else:
            self.base_extraction_template = extractor.prompt.template

        if base_extraction_template:
            self.base_extraction_template = base_extraction_template
            if self.extractor:
                self.extractor.prompt.template = self._ensure_required_placeholders(base_extraction_template)

        # 3. Reflection prompt template – strengthened to prevent deviation
        #    NOTE: {{format_instructions}} is double‑braced to become a literal,
        #    not a variable to be substituted.
        self.reflection_template = PromptTemplate(
            input_variables=["current_prompt", "sql", "errors", "current_result", "iteration"],
            template="""# Task: Improve SQL Lineage Extraction Prompt

## Context
You are refining a prompt that extracts source-to-target lineage from SQL DDL statements.
The goal is to get perfect JSON output: {{"target": "schema.table", "sources": ["schema1.table1", ...]}}

## Current Prompt
{current_prompt}

## SQL Query Being Analyzed
{sql}

## Validation Issues (Iteration {iteration})
{errors}

## Current Extraction Result
{current_result}

## Improvement Instructions
Analyze the errors above and create an improved prompt that specifically addresses them.
Focus on:
1. Clarifying JSON output format requirements
2. Ensuring fully qualified names (schema.table)
3. Excluding derived tables, CTEs, subqueries, aliases
4. Removing duplicates from sources
5. Clear identification of target table

Make the prompt more precise and explicit about the rules.

**CRITICAL FORMATTING RULES:**
- Output ONLY the new prompt – no explanations, no commentary.
- Do NOT wrap the prompt in markdown code fences (triple backticks) or any other formatting.
- The prompt MUST contain the placeholders `{{sql_text}}` and `{{format_instructions}}` exactly as shown.
- The prompt MUST instruct the model to return **ONLY** a JSON object with exactly two keys: `"target"` (string) and `"sources"` (array of strings).
- The prompt MUST NOT instruct the model to return error messages under any other key (e.g., `"message"`). Error handling is done by the system, not by the LLM.
- The format instructions provided via `{{format_instructions}}` already specify the required JSON structure – the prompt should reinforce this but never override it.

## Improved Prompt (write ONLY the new prompt):
"""
        )

    @staticmethod
    def _ensure_required_placeholders(prompt: str) -> str:
        """Make sure the prompt contains both {sql_text} and {format_instructions}."""
        if "{sql_text}" not in prompt:
            prompt += "\n\nSQL Query:\n{sql_text}"
        if "{format_instructions}" not in prompt:
            prompt += "\n\nOutput format:\n{format_instructions}"
        return prompt

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences (triple backticks) from the text."""
        # Remove opening ``` ... and closing ```
        text = re.sub(r'^```\w*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
        return text.strip()

    def _generate_text(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    async def _extract_with_retry(self, prompt: str, sql: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Async extraction with retry. Handles both sync and async extractor methods.
        """
        extraction_start = datetime.now()
        for attempt in range(max_retries):
            try:
                print(f"\n{'=' * 40}")
                print(f"EXTRACTION ATTEMPT {attempt + 1}/{max_retries}")
                print(f"{'=' * 40}")

                # Ensure the prompt has the required placeholders
                safe_prompt = self._ensure_required_placeholders(prompt)

                # Update extractor's prompt
                if hasattr(self.extractor, 'prompt') and hasattr(self.extractor.prompt, 'template'):
                    self.extractor.prompt.template = safe_prompt
                    # No need to rebuild chain – the prompt object is mutable and chain references it

                # Handle both sync and async extract methods
                extract_method = self.extractor.extract
                if asyncio.iscoroutinefunction(extract_method):
                    result_dict = await extract_method(sql)
                else:
                    # Run sync method in a thread to avoid blocking
                    result_dict = await asyncio.to_thread(extract_method, sql)

                if "error" in result_dict:
                    raise ValueError(f"Extractor returned error: {result_dict['error']}")

                if not isinstance(result_dict, dict):
                    result_dict = {"target": "", "sources": []}
                if "target" not in result_dict:
                    result_dict["target"] = ""
                if "sources" not in result_dict:
                    result_dict["sources"] = []
                if isinstance(result_dict["sources"], str):
                    result_dict["sources"] = [result_dict["sources"]]
                elif not isinstance(result_dict["sources"], list):
                    result_dict["sources"] = []

                result_dict["target"] = str(result_dict.get("target", "")).strip().lower()
                result_dict["sources"] = [
                    s.strip().lower() for s in result_dict["sources"]
                    if s and str(s).strip()
                ]

                print(f"✓ Extraction successful in {(datetime.now() - extraction_start).total_seconds():.2f}s")
                return result_dict

            except Exception as e:
                print(f"✗ Extraction attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait = 2 * (attempt + 1)
                    print(f"Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    return {
                        "error": f"Extraction failed after {max_retries} attempts: {e}",
                        "target": "",
                        "sources": []
                    }
        return {"error": "Extraction failed", "target": "", "sources": []}

    async def run_extraction(self, prompt: str, sql: str) -> Dict[str, Any]:
        return await self._extract_with_retry(prompt, sql, max_retries=3)

    async def validate_extraction(self, prompt: str, sql: str, expected: Optional[Dict] = None) -> Dict[str, Any]:
        print(f"\n{'=' * 60}")
        print(f"VALIDATION")
        print(f"{'=' * 60}")

        extraction_result = await self.run_extraction(prompt, sql)

        # Async extractor for the validator
        class TempExtractor:
            def __init__(self, agent, prompt):
                self.agent = agent
                self.prompt = prompt

            async def extract_lineage(self, sql_query):
                return await self.agent.run_extraction(self.prompt, sql_query)

        extractor_wrapper = TempExtractor(self, prompt)

        try:
            validation_result = await SQLLineageValidator.run_comprehensive_validation(
                extractor=extractor_wrapper,
                sql_query=sql,
                expected_result=expected
            )
        except Exception as e:
            validation_result = {
                "status": "FAILED",
                "validation_type": "system",
                "message": f"Validation error: {e}",
                "result": extraction_result
            }

        if "result" not in validation_result:
            validation_result["result"] = extraction_result

        return validation_result

    def _format_errors_for_reflection(self, validation_result: Dict) -> str:
        if validation_result["status"] == "SUCCESS":
            if "metrics" in validation_result:
                f1 = validation_result["metrics"].get("f1_score", 0)
                if f1 >= 1.0:
                    return (f"🎉 PERFECT F1 SCORE = 1.0!\n"
                            f"Precision: {validation_result['metrics']['precision']:.4f}, "
                            f"Recall: {validation_result['metrics']['recall']:.4f}")
                else:
                    return (f"✅ SUCCESS with F1 score: {f1:.4f}\n"
                            f"Precision: {validation_result['metrics']['precision']:.4f}, "
                            f"Recall: {validation_result['metrics']['recall']:.4f}")
            return "✅ All validations passed"

        errors = []
        if validation_result.get("message"):
            errors.append(f"❌ {validation_result.get('validation_type', 'validation').upper()}: "
                         f"{validation_result['message']}")
        if "result" in validation_result and isinstance(validation_result["result"], dict):
            if "error" in validation_result["result"]:
                errors.append(f"⚠️ Extraction error: {validation_result['result']['error']}")
            else:
                current = validation_result["result"]
                errors.append(f"📊 Current extraction: target='{current.get('target', 'N/A')}', "
                             f"sources={current.get('sources', [])}")
        return "\n".join(errors) if errors else "Unknown validation error"

    async def reflect_and_improve(self, state: Dict) -> Dict:
        print(f"\n{'=' * 60}")
        print(f"REFLECTION - Iteration {state['iteration'] + 1}")
        print(f"{'=' * 60}")

        if state["iteration"] >= state["max_iterations"]:
            print(f"Max iterations ({state['max_iterations']}) reached. Stopping.")
            state["should_continue"] = False
            return state

        if (state["validation_result"]["status"] == "SUCCESS" and
                "metrics" in state["validation_result"] and
                state["validation_result"]["metrics"].get("f1_score", 0) >= 1.0):
            print(f"🎉 Perfect F1 score achieved! Stopping reflection.")
            state["should_continue"] = False
            return state

        errors = self._format_errors_for_reflection(state["validation_result"])
        current_result = json.dumps(state["validation_result"].get("result", {}), indent=2)

        reflection_prompt = self.reflection_template.format(
            current_prompt=state["current_prompt"],
            sql=state["sql"],
            errors=errors,
            current_result=current_result,
            iteration=state["iteration"] + 1
        )

        try:
            improved_prompt = self._generate_text(reflection_prompt).strip()

            # Remove markdown code fences
            improved_prompt = self._strip_code_fences(improved_prompt)

            # Clean common meta‑lines (e.g. "Improved Prompt:" or comments)
            lines = [
                line for line in improved_prompt.split('\n')
                if not line.startswith('#')
                and not line.lower().startswith('improved prompt:')
            ]
            improved_prompt = '\n'.join(lines).strip()

            # Ensure required placeholders are present
            improved_prompt = self._ensure_required_placeholders(improved_prompt)

            if len(improved_prompt) < 50:
                raise ValueError("Generated prompt too short")

            print(f"✓ Improved prompt generated ({len(improved_prompt)} chars)")
        except Exception as e:
            print(f"✗ Reflection failed: {e}. Using fallback prompt.")
            # Fallback: add error‑specific instruction but preserve required format
            improved_prompt = state["current_prompt"] + (
                "\n\nIMPORTANT: Avoid the validation errors shown above. "
                "Return valid JSON with 'target' (string) and 'sources' (list of strings)."
            )
            improved_prompt = self._ensure_required_placeholders(improved_prompt)

        state["current_prompt"] = improved_prompt
        state["refined_prompts"].append(improved_prompt)
        state["iteration"] += 1

        return state

    async def acreate_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        async def validate_node(state: AgentState) -> AgentState:
            print(f"\n{'#' * 80}")
            print(f"VALIDATE NODE - Iteration {state['iteration'] + 1}/{state['max_iterations']}")
            print(f"{'#' * 80}")

            validation_result = await self.validate_extraction(
                prompt=state["current_prompt"],
                sql=state["sql"],
                expected=state.get("expected_result")
            )

            state["validation_result"] = validation_result
            state["validation_history"].append(validation_result)

            if validation_result["status"] == "FAILED":
                state["should_continue"] = True
            elif state.get("expected_result") and "metrics" in validation_result:
                f1 = validation_result["metrics"].get("f1_score", 0)
                state["should_continue"] = f1 < 1.0
            elif state.get("expected_result"):
                state["should_continue"] = True
            else:
                state["should_continue"] = (state["iteration"] < state["max_iterations"] - 1)

            if state["iteration"] >= state["max_iterations"]:
                state["should_continue"] = False

            return state

        async def should_continue(state: AgentState) -> str:
            return "continue" if state.get("should_continue", True) else "end"

        workflow.add_node("validate", validate_node)
        workflow.add_node("reflect", self.reflect_and_improve)

        workflow.set_entry_point("validate")
        workflow.add_conditional_edges(
            "validate",
            should_continue,
            {"continue": "reflect", "end": END}
        )
        workflow.add_edge("reflect", "validate")

        return workflow

    async def optimize_prompt(
        self,
        sql: str,
        initial_prompt: Optional[str] = None,
        expected_result: Optional[Dict] = None,
        output_file: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Async optimization entry point.
        """
        print(f"\n{'=' * 80}")
        print("STARTING PROMPT OPTIMIZATION (GigaChat Agent - Async)")
        print(f"{'=' * 80}")

        effective_initial = initial_prompt or self.base_extraction_template
        effective_initial = self._ensure_required_placeholders(effective_initial)
        if verbose:
            print(f"\n📝 Initial prompt preview:\n{effective_initial[:200]}...\n")

        initial_state = {
            "sql": sql,
            "current_prompt": effective_initial,
            "validation_result": {},
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "refined_prompts": [],
            "validation_history": [],
            "expected_result": expected_result,
            "should_continue": True
        }

        # Initialize extractor with the initial prompt
        if hasattr(self.extractor, 'prompt') and hasattr(self.extractor.prompt, 'template'):
            self.extractor.prompt.template = effective_initial

        workflow = await self.acreate_workflow()
        app = workflow.compile()

        final_state = initial_state
        for i in range(self.max_iterations):
            print(f"\n{'#' * 80}")
            print(f"ITERATION {i + 1} / {self.max_iterations}")
            print(f"{'#' * 80}")

            final_state = await app.ainvoke(final_state)

            if verbose and final_state.get("validation_result"):
                val = final_state["validation_result"]
                f1 = val.get("metrics", {}).get("f1_score", 0) if "metrics" in val else 0
                target = val.get("result", {}).get("target", "N/A")
                src_count = len(val.get("result", {}).get("sources", []))
                print(f"\n📊 Iteration {i + 1} summary:")
                print(f"   F1 score   : {f1:.4f}")
                print(f"   Target      : {target}")
                print(f"   Sources     : {src_count}")
                print(f"   Prompt preview (first 200 chars):\n{final_state['current_prompt'][:200]}...\n")

            if not final_state.get("should_continue", True):
                print(f"\n⏹️ Stopping early at iteration {i + 1} – should_continue = False")
                break

        perfect_prompt = None
        best_f1 = -1.0
        best_prompt = initial_state["current_prompt"]
        best_iter = -1

        for i, val in enumerate(final_state.get("validation_history", [])):
            if val["status"] == "SUCCESS" and "metrics" in val:
                f1 = val["metrics"].get("f1_score", 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_iter = i
                    if i == 0:
                        best_prompt = initial_state["current_prompt"]
                    else:
                        best_prompt = final_state["refined_prompts"][i - 1]
                if f1 >= 1.0:
                    perfect_prompt = best_prompt
                    best_f1 = f1
                    break

        if perfect_prompt is None:
            perfect_prompt = best_prompt

        result = {
            "optimized_prompt": perfect_prompt,
            "initial_prompt": effective_initial,
            "f1_score": best_f1,
            "is_perfect": best_f1 >= 1.0,
            "iterations": final_state["iteration"],
            "iteration_found": best_iter + 1,
            "final_validation": final_state.get("validation_result", {}),
            "all_prompts": final_state["refined_prompts"],
            "validation_history": final_state["validation_history"],
            "extractor_used": True
        }

        if output_file:
            self._save_history_to_file(result, output_file, sql, expected_result)

        return result

    def optimize_prompt_sync(
        self,
        sql: str,
        initial_prompt: Optional[str] = None,
        expected_result: Optional[Dict] = None,
        output_file: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for optimize_prompt.
        Raises RuntimeError if called from inside a running event loop.
        Use await optimize_prompt(...) in async code.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(
                self.optimize_prompt(sql, initial_prompt, expected_result, output_file, verbose)
            )
        else:
            raise RuntimeError(
                "Running event loop detected. Use 'await agent.optimize_prompt(...)' "
                "instead of the synchronous wrapper."
            )

    def _save_history_to_file(self, result: Dict, output_file: str, sql: str, expected: Optional[Dict]):
        import json
        from datetime import datetime

        history = {
            "timestamp": datetime.now().isoformat(),
            "sql": sql,
            "expected_result": expected,
            "optimization_result": result,
            "iteration_details": []
        }

        initial_prompt = result.get("initial_prompt", self.base_extraction_template)

        for i, val in enumerate(result["validation_history"]):
            if i == 0:
                prompt = initial_prompt
            else:
                prompt = result["all_prompts"][i - 1] if i - 1 < len(result["all_prompts"]) else ""

            history["iteration_details"].append({
                "iteration": i + 1,
                "prompt": prompt,
                "validation": val
            })

        with open(output_file, "w") as f:
            json.dump(history, f, indent=2)

        print(f"\n💾 Optimisation history saved to {output_file}")


# ----------------------------------------------------------------------
# AdvancedReflexionAgent – same pattern, extended with few‑shot examples
# ----------------------------------------------------------------------
class AdvancedGigaChatReflexionAgent(GigaChatSQLLineageAgent):
    """Extended version with few‑shot examples and enhanced reflection prompt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.few_shot_examples = [
            {
                "problem": "Extracting CTE as source",
                "bad_prompt": "Extract all tables mentioned in the SQL",
                "good_prompt": "Extract only base tables from FROM and JOIN clauses. Exclude CTEs defined in WITH clauses."
            },
            {
                "problem": "Not fully qualified names",
                "bad_prompt": "List the source tables",
                "good_prompt": "Extract fully qualified table names in format 'schema.table'. If schema is not specified, infer from context or use 'public' as default."
            },
            {
                "problem": "Including derived tables",
                "bad_prompt": "Get all tables used",
                "good_prompt": "Identify only base physical tables. Exclude derived tables, subqueries, aliases like t1, t2, subq, etc."
            }
        ]

        self.enhanced_reflection_template = PromptTemplate(
            input_variables=["current_prompt", "sql", "errors", "current_result", "iteration", "examples"],
            template="""# PROMPT OPTIMIZATION FOR SQL LINEAGE EXTRACTION

## Current Prompt (Iteration {iteration})
{current_prompt}

## SQL Query
{sql}

## Validation Issues
{errors}

## Current Extraction
{current_result}

## Example Improvements
{examples}

## Instructions
Analyze the specific validation errors above. Compare with the examples to understand common issues.
Create a new prompt that:
1. Explicitly addresses the specific errors shown above
2. Is more precise and unambiguous than the current prompt
3. Includes clear formatting instructions for JSON output
4. Specifies rules to avoid the validation failures

**CRITICAL FORMATTING RULES:**
- Output ONLY the new prompt – no explanations, no commentary.
- Do NOT wrap the prompt in markdown code fences (triple backticks) or any other formatting.
- The prompt MUST contain the placeholders `{{sql_text}}` and `{{format_instructions}}` exactly as shown.
- The prompt MUST instruct the model to return **ONLY** a JSON object with exactly two keys: `"target"` (string) and `"sources"` (array of strings).
- The prompt MUST NOT instruct the model to return error messages under any other key (e.g., `"message"`). Error handling is done by the system, not by the LLM.
- The format instructions provided via `{{format_instructions}}` already specify the required JSON structure – the prompt should reinforce this but never override it.

Write ONLY the improved prompt, no explanations:"""
        )

    async def reflect_and_improve(self, state: Dict) -> Dict:
        print(f"\n{'=' * 60}")
        print(f"ADVANCED REFLECTION - Iteration {state['iteration'] + 1}")
        print(f"{'=' * 60}")

        if state["iteration"] >= state["max_iterations"]:
            state["should_continue"] = False
            return state

        if (state["validation_result"]["status"] == "SUCCESS" and
                "metrics" in state["validation_result"] and
                state["validation_result"]["metrics"].get("f1_score", 0) >= 1.0):
            state["should_continue"] = False
            return state

        errors = self._format_errors_for_reflection(state["validation_result"])
        current_result = json.dumps(state["validation_result"].get("result", {}), indent=2)

        examples_text = "\n".join([
            f"Problem: {ex['problem']}\nBad: {ex['bad_prompt']}\nGood: {ex['good_prompt']}\n"
            for ex in self.few_shot_examples
        ])

        reflection_prompt = self.enhanced_reflection_template.format(
            current_prompt=state["current_prompt"],
            sql=state["sql"],
            errors=errors,
            current_result=current_result,
            iteration=state["iteration"] + 1,
            examples=examples_text
        )

        try:
            improved_prompt = self._generate_text(reflection_prompt).strip()

            # Remove markdown code fences
            improved_prompt = self._strip_code_fences(improved_prompt)

            lines = [
                line for line in improved_prompt.split('\n')
                if not line.startswith('#') and not line.lower().startswith('improved prompt:')
            ]
            improved_prompt = '\n'.join(lines).strip()
            improved_prompt = self._ensure_required_placeholders(improved_prompt)
        except Exception as e:
            print(f"✗ Advanced reflection failed: {e}. Using fallback.")
            improved_prompt = state["current_prompt"] + (
                "\n\nADDITIONAL RULE: Avoid the validation errors shown above. "
                "Return valid JSON with 'target' (string) and 'sources' (list of strings)."
            )
            improved_prompt = self._ensure_required_placeholders(improved_prompt)

        state["current_prompt"] = improved_prompt
        state["refined_prompts"].append(improved_prompt)
        state["iteration"] += 1

        return state



class BatchAgentState(TypedDict):
    sqls: List[str]
    expected_results: Optional[List[Dict[str, Any]]]
    current_prompt: str
    validation_results: List[Dict[str, Any]]
    iteration: int
    max_iterations: int
    refined_prompts: List[str]
    validation_history: List[List[Dict[str, Any]]]
    aggregated_metrics: Dict[str, float]
    should_continue: bool


class GigaChatBatchSQLLineageAgent:
    """
    Prompt optimization agent for a batch of SQL statements.
    Finds a single prompt that works well across all queries.
    Limits concurrency to avoid overwhelming the API.
    """

    def __init__(
        self,
        credentials: Optional[str] = None,
        model: str = "GigaChat",
        verify_ssl_certs: bool = False,
        scope: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.01,
        max_tokens: int = 1024,          # reduced from 2048 for speed
        max_iterations: int = 5,
        base_extraction_template: Optional[str] = None,
        extractor: Optional[SQLLineageExtractor] = None,
        max_concurrent: int = 5           # new parameter
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
        self.max_iterations = min(max_iterations, 5)
        self.max_concurrent = max_concurrent

        # 1. GigaChat LLM for reflection
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

        # 2. Extractor management
        self.extractor = extractor
        if self.extractor is None:
            self.extractor = SQLLineageExtractor(
                credentials=self.credentials,
                model=self.model,
                verify_ssl_certs=self.verify_ssl_certs,
                scope=self.scope,
                base_url=self.base_url,
                temperature=0.1,
                max_tokens=self.max_tokens,
            )
            self.base_extraction_template = self.extractor.prompt.template
        else:
            self.base_extraction_template = extractor.prompt.template

        if base_extraction_template:
            self.base_extraction_template = base_extraction_template
            if self.extractor:
                self.extractor.prompt.template = self._ensure_required_placeholders(base_extraction_template)

        # 3. Reflection prompt template – adapted for batch feedback
        self.reflection_template = PromptTemplate(
            input_variables=["current_prompt", "batch_summary", "iteration"],
            template="""# Task: Improve SQL Lineage Extraction Prompt for Multiple Queries

## Context
You are refining a prompt that extracts source-to-target lineage from SQL statements.
The goal is to get perfect JSON output: {{"target": "schema.table", "sources": ["schema1.table1", ...]}}.
The same prompt will be used for **all** SQL queries in a batch.

## Current Prompt
{current_prompt}

## Batch Summary (Iteration {iteration})
{batch_summary}

## Improvement Instructions
Analyze the errors and successes above. Create an improved prompt that:
1. Addresses common failure patterns across multiple queries.
2. Clarifies JSON output format requirements.
3. Ensures fully qualified names (schema.table) and excludes derived tables/CTEs.
4. Removes duplicates and uses lowercase.
5. Is generic enough to work for all SQL statements in the batch.

**CRITICAL FORMATTING RULES:**
- Output ONLY the new prompt – no explanations, no commentary.
- Do NOT wrap the prompt in markdown code fences.
- The prompt MUST contain the placeholders `{{sql_text}}` and `{{format_instructions}}`.
- The prompt MUST instruct the model to return **ONLY** a JSON object with keys `"target"` (string) and `"sources"` (array of strings).
- The prompt MUST NOT instruct the model to return error messages under any other key.
- The format instructions provided via `{{format_instructions}}` already specify the required JSON structure – the prompt should reinforce this but never override it.

## Improved Prompt (write ONLY the new prompt):
"""
        )

    @staticmethod
    def _ensure_required_placeholders(prompt: str) -> str:
        if "{sql_text}" not in prompt:
            prompt += "\n\nSQL Query:\n{sql_text}"
        if "{format_instructions}" not in prompt:
            prompt += "\n\nOutput format:\n{format_instructions}"
        return prompt

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = re.sub(r'^```\w*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
        return text.strip()

    def _generate_text(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    async def _extract_with_retry(self, prompt: str, sql: str, max_retries: int = 3) -> Dict[str, Any]:
        """Async extraction with retry. Minimal logging to avoid clutter."""
        for attempt in range(max_retries):
            try:
                safe_prompt = self._ensure_required_placeholders(prompt)
                if hasattr(self.extractor, 'prompt') and hasattr(self.extractor.prompt, 'template'):
                    self.extractor.prompt.template = safe_prompt

                extract_method = self.extractor.extract
                if asyncio.iscoroutinefunction(extract_method):
                    result_dict = await extract_method(sql)
                else:
                    result_dict = await asyncio.to_thread(extract_method, sql)

                if "error" in result_dict:
                    raise ValueError(f"Extractor returned error: {result_dict['error']}")

                if not isinstance(result_dict, dict):
                    result_dict = {"target": "", "sources": []}
                result_dict.setdefault("target", "")
                result_dict.setdefault("sources", [])
                if isinstance(result_dict["sources"], str):
                    result_dict["sources"] = [result_dict["sources"]]
                elif not isinstance(result_dict["sources"], list):
                    result_dict["sources"] = []

                result_dict["target"] = str(result_dict["target"]).strip().lower()
                result_dict["sources"] = [
                    s.strip().lower() for s in result_dict["sources"]
                    if s and str(s).strip()
                ]
                return result_dict

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 * (attempt + 1)
                    await asyncio.sleep(wait)
                else:
                    # Only print failure on last attempt to reduce noise
                    print(f"⚠️ Extraction failed after {max_retries} attempts: {e}")
                    return {"error": str(e), "target": "", "sources": []}
        return {"error": "Extraction failed", "target": "", "sources": []}

    async def run_extraction(self, prompt: str, sql: str) -> Dict[str, Any]:
        return await self._extract_with_retry(prompt, sql)

    async def validate_single(self, prompt: str, sql: str, expected: Optional[Dict] = None) -> Dict[str, Any]:
        extraction_result = await self.run_extraction(prompt, sql)

        class TempExtractor:
            def __init__(self, agent, prompt):
                self.agent = agent
                self.prompt = prompt
            async def extract_lineage(self, sql_query):
                return await self.agent.run_extraction(self.prompt, sql_query)

        extractor_wrapper = TempExtractor(self, prompt)

        try:
            validation = await SQLLineageValidator.run_comprehensive_validation(
                extractor=extractor_wrapper,
                sql_query=sql,
                expected_result=expected
            )
        except Exception as e:
            validation = {
                "status": "FAILED",
                "validation_type": "system",
                "message": f"Validation error: {e}",
                "result": extraction_result
            }

        if "result" not in validation:
            validation["result"] = extraction_result
        return validation

    async def validate_batch(self, prompt: str, sqls: List[str], expected_results: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Validate the prompt against all SQLs in parallel, but with a concurrency limit.
        """
        # Set the extractor's prompt once (it will be reused)
        if hasattr(self.extractor, 'prompt') and hasattr(self.extractor.prompt, 'template'):
            self.extractor.prompt.template = self._ensure_required_placeholders(prompt)

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_validate(index: int):
            sql = sqls[index]
            expected = expected_results[index] if expected_results and index < len(expected_results) else None
            async with semaphore:
                return await self.validate_single(prompt, sql, expected)

        tasks = [limited_validate(i) for i in range(len(sqls))]
        return await asyncio.gather(*tasks)

    def _aggregate_metrics(self, validation_results: List[Dict]) -> Dict[str, float]:
        """Compute aggregated metrics from a list of validation results."""
        f1_scores = []
        for v in validation_results:
            if v["status"] == "SUCCESS" and "metrics" in v:
                f1_scores.append(v["metrics"].get("f1_score", 0))
            else:
                f1_scores.append(0.0)

        avg_f1 = mean(f1_scores) if f1_scores else 0.0
        min_f1 = min(f1_scores) if f1_scores else 0.0
        success_rate = sum(1 for s in f1_scores if s >= 1.0) / len(f1_scores) if f1_scores else 0.0
        return {
            "avg_f1": avg_f1,
            "min_f1": min_f1,
            "success_rate": success_rate,
            "num_queries": len(validation_results)
        }

    def _format_batch_summary(self, validation_results: List[Dict], sqls: List[str]) -> str:
        """Create a human‑readable summary of batch validation results for the reflection prompt."""
        lines = []
        for i, (sql, val) in enumerate(zip(sqls, validation_results)):
            status = val["status"]
            msg = val.get("message", "")
            target = val.get("result", {}).get("target", "N/A")
            sources = val.get("result", {}).get("sources", [])
            if "metrics" in val:
                f1 = val["metrics"].get("f1_score", 0)
                lines.append(f"SQL {i+1}: {status} | F1={f1:.3f} | target={target} | sources={sources}")
            else:
                lines.append(f"SQL {i+1}: {status} | {msg} | target={target} | sources={sources}")
        return "\n".join(lines)

    async def reflect_and_improve(self, state: BatchAgentState) -> BatchAgentState:
        print(f"\n{'=' * 60}")
        print(f"BATCH REFLECTION - Iteration {state['iteration'] + 1}")
        print(f"{'=' * 60}")

        if state["iteration"] >= state["max_iterations"]:
            state["should_continue"] = False
            return state

        # Check stopping criterion: avg F1 == 1.0 (all perfect)
        if state["aggregated_metrics"].get("avg_f1", 0) >= 1.0:
            print("🎉 All queries perfect! Stopping.")
            state["should_continue"] = False
            return state

        batch_summary = self._format_batch_summary(state["validation_results"], state["sqls"])
        reflection_prompt = self.reflection_template.format(
            current_prompt=state["current_prompt"],
            batch_summary=batch_summary,
            iteration=state["iteration"] + 1
        )

        try:
            improved_prompt = self._generate_text(reflection_prompt).strip()
            improved_prompt = self._strip_code_fences(improved_prompt)
            # Clean common meta‑lines
            lines = [
                line for line in improved_prompt.split('\n')
                if not line.startswith('#') and not line.lower().startswith('improved prompt:')
            ]
            improved_prompt = '\n'.join(lines).strip()
            improved_prompt = self._ensure_required_placeholders(improved_prompt)

            if len(improved_prompt) < 50:
                raise ValueError("Generated prompt too short")
            print(f"✓ Improved prompt generated ({len(improved_prompt)} chars)")
        except Exception as e:
            print(f"✗ Reflection failed: {e}. Using fallback.")
            improved_prompt = state["current_prompt"] + (
                "\n\nIMPORTANT: Avoid the validation errors shown above. "
                "Return valid JSON with 'target' (string) and 'sources' (list of strings)."
            )
            improved_prompt = self._ensure_required_placeholders(improved_prompt)

        state["current_prompt"] = improved_prompt
        state["refined_prompts"].append(improved_prompt)
        state["iteration"] += 1
        return state

    async def acreate_workflow(self) -> StateGraph:
        workflow = StateGraph(BatchAgentState)

        async def validate_node(state: BatchAgentState) -> BatchAgentState:
            print(f"\n{'#' * 80}")
            print(f"VALIDATE BATCH NODE - Iteration {state['iteration'] + 1}/{state['max_iterations']}")
            print(f"{'#' * 80}")

            validation_results = await self.validate_batch(
                prompt=state["current_prompt"],
                sqls=state["sqls"],
                expected_results=state.get("expected_results")
            )

            state["validation_results"] = validation_results
            state["validation_history"].append(validation_results)
            state["aggregated_metrics"] = self._aggregate_metrics(validation_results)

            # Decision logic: continue if avg F1 < 1.0 and iteration limit not reached
            avg_f1 = state["aggregated_metrics"]["avg_f1"]
            state["should_continue"] = (avg_f1 < 1.0) and (state["iteration"] < state["max_iterations"] - 1)

            if state["iteration"] >= state["max_iterations"]:
                state["should_continue"] = False

            return state

        async def should_continue(state: BatchAgentState) -> str:
            return "continue" if state.get("should_continue", True) else "end"

        workflow.add_node("validate", validate_node)
        workflow.add_node("reflect", self.reflect_and_improve)

        workflow.set_entry_point("validate")
        workflow.add_conditional_edges(
            "validate",
            should_continue,
            {"continue": "reflect", "end": END}
        )
        workflow.add_edge("reflect", "validate")

        return workflow

    async def optimize_prompt_batch(
        self,
        sqls: List[str],
        initial_prompt: Optional[str] = None,
        expected_results: Optional[List[Dict]] = None,
        output_file: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run batch prompt optimization.

        Args:
            sqls: List of SQL queries.
            initial_prompt: Starting prompt (if None, uses base_extraction_template).
            expected_results: Optional list of expected lineage dicts (same order as sqls).
            output_file: Optional path to save iteration history.
            verbose: If True, print progress.

        Returns:
            Dictionary with optimization results.
        """
        print(f"\n{'=' * 80}")
        print("STARTING BATCH PROMPT OPTIMIZATION (GigaChat Agent)")
        print(f"{'=' * 80}")

        effective_initial = initial_prompt or self.base_extraction_template
        effective_initial = self._ensure_required_placeholders(effective_initial)
        if verbose:
            print(f"\n📝 Initial prompt preview:\n{effective_initial[:200]}...\n")

        initial_state: BatchAgentState = {
            "sqls": sqls,
            "expected_results": expected_results,
            "current_prompt": effective_initial,
            "validation_results": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "refined_prompts": [],
            "validation_history": [],
            "aggregated_metrics": {},
            "should_continue": True
        }

        # Set initial prompt in extractor
        if hasattr(self.extractor, 'prompt') and hasattr(self.extractor.prompt, 'template'):
            self.extractor.prompt.template = effective_initial

        workflow = await self.acreate_workflow()
        app = workflow.compile()

        final_state = initial_state
        for i in range(self.max_iterations):
            print(f"\n{'#' * 80}")
            print(f"ITERATION {i + 1} / {self.max_iterations}")
            print(f"{'#' * 80}")

            final_state = await app.ainvoke(final_state)

            if verbose and final_state.get("aggregated_metrics"):
                metrics = final_state["aggregated_metrics"]
                print(f"\n📊 Iteration {i + 1} aggregated metrics:")
                print(f"   Avg F1   : {metrics['avg_f1']:.4f}")
                print(f"   Min F1   : {metrics['min_f1']:.4f}")
                print(f"   Success rate: {metrics['success_rate']:.2%}")
                print(f"   Prompt preview (first 200 chars):\n{final_state['current_prompt'][:200]}...\n")

            if not final_state.get("should_continue", True):
                print(f"\n⏹️ Stopping early at iteration {i + 1} – should_continue = False")
                break

        # Find best prompt (highest avg F1)
        best_avg_f1 = -1.0
        best_prompt = initial_state["current_prompt"]
        best_iter = -1
        for i, val_hist in enumerate(final_state.get("validation_history", [])):
            metrics = self._aggregate_metrics(val_hist)
            if metrics["avg_f1"] > best_avg_f1:
                best_avg_f1 = metrics["avg_f1"]
                best_iter = i
                if i == 0:
                    best_prompt = initial_state["current_prompt"]
                else:
                    best_prompt = final_state["refined_prompts"][i - 1]

        result = {
            "optimized_prompt": best_prompt,
            "initial_prompt": effective_initial,
            "best_avg_f1": best_avg_f1,
            "iterations": final_state["iteration"],
            "iteration_found": best_iter + 1,
            "final_validation": final_state.get("validation_results", []),
            "all_prompts": final_state["refined_prompts"],
            "validation_history": final_state["validation_history"],
            "aggregated_metrics": final_state.get("aggregated_metrics", {})
        }

        if output_file:
            self._save_history_to_file(result, output_file, sqls, expected_results)

        return result

    def optimize_prompt_batch_sync(
        self,
        sqls: List[str],
        initial_prompt: Optional[str] = None,
        expected_results: Optional[List[Dict]] = None,
        output_file: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for batch optimization.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.optimize_prompt_batch(sqls, initial_prompt, expected_results, output_file, verbose)
            )
        else:
            raise RuntimeError(
                "Running event loop detected. Use 'await agent.optimize_prompt_batch(...)' "
                "instead of the synchronous wrapper."
            )

    def _save_history_to_file(self, result: Dict, output_file: str, sqls: List[str], expected: Optional[List[Dict]]):
        import json
        from datetime import datetime

        history = {
            "timestamp": datetime.now().isoformat(),
            "sqls": sqls,
            "expected_results": expected,
            "optimization_result": result,
            "iteration_details": []
        }

        initial_prompt = result.get("initial_prompt", self.base_extraction_template)

        for i, val_hist in enumerate(result["validation_history"]):
            if i == 0:
                prompt = initial_prompt
            else:
                prompt = result["all_prompts"][i - 1] if i - 1 < len(result["all_prompts"]) else ""

            history["iteration_details"].append({
                "iteration": i + 1,
                "prompt": prompt,
                "validation_results": val_hist,
                "aggregated_metrics": self._aggregate_metrics(val_hist)
            })

        with open(output_file, "w") as f:
            json.dump(history, f, indent=2)

        print(f"\n💾 Optimisation history saved to {output_file}")