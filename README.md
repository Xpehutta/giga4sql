# SQL Lineage Tool – GigaChat Edition

A powerful tool that uses **LangChain + GigaChat** to extract source‑to‑target lineage from SQL statements and automatically **optimise the extraction prompt** via reflexion agents (single‑query and batch).  
The project includes a **Streamlit web interface** for interactive exploration and batch processing.

---

## ✨ Features

- **Single‑query lineage extraction** – get target table and source tables as JSON + graph.
- **Batch processing** – upload multiple `.sql` / `.txt` files (each may contain several statements, split by `;`).
- **Table‑centric view** – enter a table name to see all queries where it is target or source, plus a dependency graph.
- **Prompt optimisation agents**  
  - **Single‑query agent** (`GigaChatSQLLineageAgent`): iteratively improves a prompt for one SQL using a reflexion loop (F1‑score guided).  
  - **Batch agent** (`GigaChatBatchSQLLineageAgent`): optimises a single prompt to work well across **multiple** SQL statements simultaneously, aggregating metrics and using a batch‑aware reflection prompt.
- **LangChain GigaChat integration** – uses `langchain_gigachat` for both extraction and reflection.
- **All outputs saved** – optimisation history (prompts, validation results) can be written to JSON.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd sql-lineage-tool
```

### 2. Install dependencies

We recommend using **uv** (fast Python package installer) or plain `pip`:

```bash
# Using uv
uv venv
source .venv/bin/activate      # macOS/Linux
# or .venv\Scripts\activate    # Windows
uv pip install -e .

# Using pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Set your GigaChat credentials

Create a `.env` file in the project root:

```env
GIGACHAT_CREDENTIALS=your_gigachat_api_key_here
```

Or export it as an environment variable:

```bash
export GIGACHAT_CREDENTIALS=your_gigachat_api_key_here
```

Optionally, you can also set `GIGACHAT_SCOPE` and `GIGACHAT_BASE_URL` if needed.

---

## 🚀 Running the Streamlit App

The main entry point for interactive use is `Web/app.py`.

```bash
streamlit run Web/app.py
```

The app will open in your browser at `http://localhost:8501`.

### App layout

- **Left sidebar**: configure GigaChat model (`GigaChat`, `GigaChat-Pro`, `GigaChat-Max`), credentials, temperature, max tokens, etc.
- **Two main tabs**:
  - **Single Query Lineage** – paste one SQL, get JSON + graph.
  - **Table Lineage (Batch)** – upload files, see an overview, click a target to explore.

---

## 📚 How the Classes Are Connected

The project is structured into several classes, each with a clear responsibility. Below is a simplified diagram for the GigaChat version:

```text
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Frontend                         │
│                         (Web/app.py)                            │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SQLLineageExtractor                         │
│                  (langchain_gigachat.GigaChat)                  │
│  - prompt template (contains {sql_text} & {format_instructions})│
│  - extract(sql)               → {"target": ..., "sources": ...} │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SQLLineageValidator                         │
│  (from validation_classes)                                      │
│  - run_comprehensive_validation(extractor, sql, expected)       │
│    → returns {"status": "SUCCESS"/"FAILED", "metrics": {...}}   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  GigaChatSQLLineageAgent                        │
│           (single‑query prompt optimisation)                    │
│  - owns an extractor (for lineage extraction)                   │
│  - owns a separate GigaChat LLM (for reflection)                │
│  - create_workflow() → LangGraph with validate + reflect nodes  │
│  - optimize_prompt() → best prompt & F1 history                 │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│               GigaChatBatchSQLLineageAgent                      │
│           (batch prompt optimisation)                           │
│  - validates a prompt against multiple SQLs in parallel         │
│  - aggregates F1 scores, min F1, success rate                   │
│  - batch‑aware reflection prompt                                │
│  - stops when average F1 == 1.0                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed connections

#### 1. `SQLLineageExtractor`
- Wraps `GigaChat` from `langchain_gigachat`.  
- Its prompt template is the **prompt being optimised**.  
- Provides the `extract()` method used by both the app and the agents.

#### 2. `SQLLineageValidator`
- Compares the extractor’s output against a **ground‑truth result**.  
- Returns **F1 score, precision, recall** – metrics that the agent uses to decide if the prompt is good enough.

#### 3. `GigaChatSQLLineageAgent`
- **Reuses the same `SQLLineageExtractor`** (or creates one) to perform extraction during optimisation.  
- Maintains **its own** `GigaChat` instance (with identical parameters) to generate improved prompts.  
- Implements a **reflexion loop** as a LangGraph workflow:
  - `validate_node`: runs extraction + validation, records F1.
  - `reflect_node`: feeds errors and current prompt to the chat model → produces a refined prompt.
- Stops when F1 = 1.0 or max iterations reached.

#### 4. `GigaChatBatchSQLLineageAgent`
- Inherits from the single‑query agent but overrides `validate_batch` and the reflection prompt.  
- Processes all SQLs in parallel with a concurrency limit (`max_concurrent`).  
- Aggregates metrics (average F1, min F1, success rate) to decide when to stop.  
- The reflection prompt receives a summary of all validation results and must propose a prompt that works for the whole batch.

#### 5. Streamlit app (`Web/app.py`)
- Instantiates `SQLLineageExtractor` (cached) with GigaChat parameters.  
- For the batch tab, stores every extracted statement together with its **full SQL** and lineage.  
- When a user clicks a target table, the lookup input is populated and the downstream/upstream graph is drawn.  
- The app **does not** directly use the agents – they are meant for **offline prompt optimisation**.

This modular design keeps the interactive web interface separate from the prompt‑optimisation logic, making both parts easier to maintain and extend.

---

## 🧪 Example Workflows

### Workflow 1: Optimise an extraction prompt for a single query

```python
from Classes.model_classes import SQLLineageExtractor
from Classes.prompt_refiner import GigaChatSQLLineageAgent
import os

# Create extractor
extractor = SQLLineageExtractor(
    credentials=os.environ["GIGACHAT_CREDENTIALS"],
    model="GigaChat-Pro",
    temperature=0.1,
    max_tokens=1024
)

# Create agent, passing the extractor (so they share the same model)
agent = GigaChatSQLLineageAgent(
    credentials=os.environ["GIGACHAT_CREDENTIALS"],
    model="GigaChat-Pro",
    extractor=extractor,
    max_iterations=5
)

# Ground truth (what the correct lineage should be)
expected = {
    "target": "analytics.sales_summary",
    "sources": ["products.raw_data", "sales.transactions"]
}

# Run optimisation
result = agent.optimize_prompt_sync(
    sql="INSERT INTO analytics.sales_summary SELECT p.category, SUM(s.amount) FROM products.raw_data p JOIN sales.transactions s ON p.id = s.product_id",
    expected_result=expected,
    output_file="optimisation_log.json",
    verbose=True
)

print("Best prompt:\n", result["optimized_prompt"])
print("F1 score:", result["f1_score"])
```

### Workflow 2: Optimise a prompt for a batch of SQLs

```python
from Classes.model_classes import SQLLineageExtractor
from Classes.prompt_refiner import GigaChatBatchSQLLineageAgent

extractor = SQLLineageExtractor(credentials="...", model="GigaChat")
agent = GigaChatBatchSQLLineageAgent(
    credentials="...",
    model="GigaChat",
    extractor=extractor,
    max_concurrent=3,          # limit simultaneous calls
    max_iterations=5
)

sqls = [
    "INSERT INTO schema1.table1 SELECT ...",
    "UPDATE schema2.table2 SET ..."
]
expected = [
    {"target": "schema1.table1", "sources": ["source1"]},
    {"target": "schema2.table2", "sources": ["source2"]}
]

result = await agent.optimize_prompt_batch(
    sqls=sqls,
    expected_results=expected,
    output_file="batch_optimisation.json",
    verbose=True
)
print("Best batch prompt:", result["optimized_prompt"])
print("Average F1:", result["best_avg_f1"])
```

### Workflow 3: Interactive Exploration in the Streamlit App

This workflow guides you through using the **Table Lineage (Batch)** tab of the web interface to explore lineage across multiple SQL scripts.

#### Prerequisites
- You have started the app with `streamlit run Web/app.py`
- Your GigaChat credentials are entered in the sidebar (or set in `.env`)

---

#### Step‑by‑Step

1. **Open the “Table Lineage (Batch)” tab**  
   Click the second tab at the top of the page.

2. **Upload one or more SQL files**  
   - Drag & drop `.sql` or `.txt` files into the file uploader, or click “Browse files”.  
   - Files may contain **multiple SQL statements** separated by semicolons (`;`).  
   - Example file content:
     ```sql
     INSERT INTO target1 SELECT * FROM source1;
     INSERT INTO target2 SELECT * FROM source2;
     ```
   ![Loading DDLs](https://github.com/Xpehutta/giga4sql/blob/main/data/Add_DDLs.png)
   
4. **Processing**  
   The app will:
   - Split each file into individual statements.
   - Run lineage extraction on each statement using the `SQLLineageExtractor`.
   - Display a progress bar and show any errors per statement.
   - Store results in the session.

5. **View the extracted lineage overview**  
   After processing, you’ll see a table with:
   - **File** name
   - **Statement** number
   - **Target** table (clickable button)
   - **Sources** count

   ![Overview table](https://github.com/Xpehutta/giga4sql/blob/main/data/Results.png)

6. **Click on a target table**  
   - Clicking any target button automatically fills the “Look up a table” input field.  
   - The page scrolls to the lookup section and displays:
     - How many times the table appears as **Target** and as **Source**.
     - Expandable sections listing every occurrence.

7. **Explore occurrences**  
   - **As Target**: For each occurrence you see the source tables and the **full SQL** in a code block (with a copy button).  
   - **As Source**: For each occurrence you see the target table and the full SQL.

8. **Visualise the lineage graph**  
   Below the occurrence lists, an interactive **Graphviz graph** shows:
   - **Upstream sources** (tables that feed into the selected table) on the left.
   - **Downstream targets** (tables that use the selected table as a source) on the right.
   - The central node is your selected table.

 ![Lineage graph](https://github.com/Xpehutta/giga4sql/blob/main/data/Graph.png)

9. **Copy any SQL**  
   - Every displayed SQL code block has a **copy icon** in the top‑right corner – click it to copy the entire statement to your clipboard.

10. **Clear the session**  
   - Use the “Clear all stored lineage results” button at the bottom to reset and start fresh.

### This diagram shows the steps a user takes when using the Streamlit web interface, from loading the app to exploring lineage results.

![User Interaction Diagram](https://github.com/Xpehutta/giga4sql/blob/main/data/WorkFlow_Giga.png)

---

#### Tips
- The **Single Query Lineage** tab works the same way, but only for one SQL at a time – you’ll get immediate JSON and a graph.
- If you have many statements, the graph limits nodes to 15 for readability, with a `…` indicator if more exist.
- The app caches the extractor, so repeated lookups are fast.

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` (e.g., `langchain_gigachat`) | Install the missing package: `pip install langchain-gigachat`. |
| Graphviz not rendering | Install system Graphviz: `sudo apt install graphviz` (Ubuntu), `brew install graphviz` (macOS), or download from [graphviz.org](https://graphviz.org/download/) (Windows). |
| `GIGACHAT_CREDENTIALS` errors | Ensure credentials are set in `.env` or as environment variable. |
| `streamlit: command not found` | Install Streamlit: `pip install streamlit` or add it to your dependencies. |
| Model fails to load | Verify model name (`GigaChat`, `GigaChat-Pro`, `GigaChat-Max`) and that your credentials have access to it. |
| App is slow | Reduce `max_tokens` or `max_concurrent` (in the batch agent). First extraction may be slower due to model loading. |
| Agent returns empty results after reflection | Check that the refined prompt contains `{sql_text}` and `{format_instructions}` – the agent automatically adds them if missing. |
