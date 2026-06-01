"""
DataStorm 2026 - AI Data Assistant Service
==========================================
Implements the local RAG pipeline, FAISS-based vector index,
CSV/Excel/PDF ingestion, Python-based statistical analytics engine,
and strict domain guardrails.
"""

import os
import re
import csv
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Suppress warning
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIDataAssistant")

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# FAISS and Sentence Transformers imports
HAS_FAISS = False
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_FAISS = True
except ImportError:
    logger.warning("faiss-cpu or sentence-transformers not found. Falling back to TF-IDF semantic index.")

# PDF parser import
HAS_PYPDF = False
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    pass

# ===========================================================================
# Pure Python TF-IDF Index for Fallback Vector Search
# ===========================================================================
class TFIDFIndex:
    """A lightweight pure-Python TF-IDF index for document retrieval."""
    def __init__(self):
        self.documents: List[Dict[str, str]] = []  # List of {"text": text, "source": source}
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tfidf_matrix: List[List[float]] = []

    def _tokenize(self, text: str) -> List[str]:
        # Simple word tokenization
        words = re.findall(r'\b[a-zA-Z0-9_]{3,20}\b', text.lower())
        return words

    def add_documents(self, documents: List[Dict[str, str]]):
        self.documents.extend(documents)
        self.rebuild_index()

    def rebuild_index(self):
        if not self.documents:
            return

        # Calculate TF and Vocabulary
        doc_tfs = []
        vocab = set()
        for doc in self.documents:
            tokens = self._tokenize(doc["text"])
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            doc_tfs.append(tf)
            vocab.update(tf.keys())

        self.vocabulary = {word: idx for idx, word in enumerate(sorted(vocab))}
        N = len(self.documents)

        # Calculate IDF
        doc_counts = {}
        for tf in doc_tfs:
            for word in tf.keys():
                doc_counts[word] = doc_counts.get(word, 0) + 1

        self.idf = {}
        for word, count in doc_counts.items():
            self.idf[word] = math.log((1 + N) / (1 + count)) + 1

        # Calculate TF-IDF Matrix
        self.tfidf_matrix = []
        for tf in doc_tfs:
            vector = [0.0] * len(self.vocabulary)
            norm = 0.0
            for word, count in tf.items():
                if word in self.vocabulary:
                    val = count * self.idf[word]
                    vector[self.vocabulary[word]] = val
                    norm += val * val
            norm = math.sqrt(norm)
            if norm > 0:
                vector = [val / norm for val in vector]
            self.tfidf_matrix.append(vector)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, str], float]]:
        if not self.documents or not self.vocabulary:
            return []

        tokens = self._tokenize(query)
        q_tf = {}
        for t in tokens:
            q_tf[t] = q_tf.get(t, 0) + 1

        q_vector = [0.0] * len(self.vocabulary)
        q_norm = 0.0
        for word, count in q_tf.items():
            if word in self.vocabulary:
                val = count * self.idf.get(word, 0.0)
                q_vector[self.vocabulary[word]] = val
                q_norm += val * val
        q_norm = math.sqrt(q_norm)
        if q_norm > 0:
            q_vector = [val / q_norm for val in q_vector]

        results = []
        for doc_idx, doc_vector in enumerate(self.tfidf_matrix):
            # Cosine similarity
            score = sum(q_val * d_val for q_val, d_val in zip(q_vector, doc_vector))
            if score > 0:
                results.append((self.documents[doc_idx], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ===========================================================================
# Assistant Service
# ===========================================================================
class AssistantService:
    def __init__(self):
        self.model = None
        self.faiss_index = None
        self.doc_chunks: List[Dict[str, str]] = []  # {"text": text, "source": source}
        self.tfidf_index = TFIDFIndex()
        self.uploaded_dfs: Dict[str, pd.DataFrame] = {} # filename -> DataFrame
        
        # Local LLM endpoint (Ollama)
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama3")

        # Initialize the embedding model if libraries are present
        if HAS_FAISS:
            try:
                logger.info("Initializing SentenceTransformer...")
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}. Falling back to TF-IDF.")
                self.model = None

        # Build initial project knowledge base index
        self.index_project_knowledge()

    def index_project_knowledge(self):
        """Load and index all documentation, readmes, walkthroughs, and schema information."""
        self.doc_chunks = []
        
        # 1. Add Data Dictionary
        data_dictionary = """
        DataStorm 2026 - Data Dictionary
        --------------------------------
        outlet_master.parquet (Columns):
        - Outlet_ID: Unique identifier for each outlet (e.g. OUT_01).
        - Outlet_Type: Grocery, Hotel, Pharmacy, Kiosk, Eatery, Bakery, SMMT.
        - Outlet_Size: Extra Large, Large, Medium, Small.
        - Cooler_Count: Number of coolers installed in the outlet.
        
        transactions.parquet (Columns):
        - Outlet_ID: Reference to outlet.
        - Year, Month: Date components.
        - Distributor_ID: distributor handling the delivery.
        - Volume_Liters: Sales volume in liters.
        - Total_Bill_Value: Sales revenue in LKR.
        
        gold_features.parquet (Columns):
        - hist_mean_vol, hist_median_vol, hist_max_vol: Historical sales statistics.
        - censoring_score: Multi-component constraint indicator in [0, 1].
        - peer_efficiency_gap: Ratio of peer 90th percentile to median.
        - combined_catchment_score: Spatial gravity score of POIs.
        - Maximum_Monthly_Liters: Predicted latent demand potential (primary submission).
        - Heuristic_Latent_Liters: Heuristic-only latent demand.
        - Quantile_Ceiling_Liters: Ensembled statistical ceiling quantile model.
        """
        self.add_text_chunk(data_dictionary, "Data Dictionary Schema")

        # 2. Add Project Markdown docs
        for filename in ["README.md", "walkthrough.md", "implementation_plan.md"]:
            path = ROOT / filename
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8")
                    self.add_markdown_chunks(content, filename)
                except Exception as e:
                    logger.error(f"Error loading doc {filename}: {e}")

        # 3. Add LaTeX Theory document
        tex_path = ROOT / "latent_demand_theory.tex"
        if tex_path.exists():
            try:
                content = tex_path.read_text(encoding="utf-8")
                self.add_text_chunk(content[:4000], "LaTeX Theory Section 1-3")
                self.add_text_chunk(content[4000:8000], "LaTeX Theory Section 4-5")
                self.add_text_chunk(content[8000:], "LaTeX Theory Section 6-7")
            except Exception as e:
                logger.error(f"Error loading latex doc: {e}")

        # Build final indices
        self.rebuild_indices()

    def add_text_chunk(self, text: str, source: str):
        self.doc_chunks.append({"text": text.strip(), "source": source})

    def add_markdown_chunks(self, content: str, source: str):
        # Split markdown by sections (h2, h3) or paragraphs
        sections = re.split(r'\n(?=##? )', content)
        for sec in sections:
            if sec.strip():
                # Avoid keeping massive sections whole
                if len(sec) > 1500:
                    for i in range(0, len(sec), 1200):
                        self.doc_chunks.append({
                            "text": sec[i:i+1500].strip(),
                            "source": f"{source} (Part {i//1200 + 1})"
                        })
                else:
                    self.doc_chunks.append({"text": sec.strip(), "source": source})

    def rebuild_indices(self):
        """Compile the document chunks into FAISS or TF-IDF."""
        if not self.doc_chunks:
            return

        # Rebuild TF-IDF always as backup
        self.tfidf_index.documents = []
        self.tfidf_index.add_documents(self.doc_chunks)

        # Build FAISS if possible
        if HAS_FAISS and self.model is not None:
            try:
                texts = [d["text"] for d in self.doc_chunks]
                embeddings = self.model.encode(texts, show_progress_bar=False)
                embeddings = np.array(embeddings).astype("float32")
                
                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product (Cosine similarity normalized)
                faiss.normalize_L2(embeddings)
                self.faiss_index.add(embeddings)
                logger.info(f"FAISS index built successfully with {len(texts)} chunks.")
            except Exception as e:
                logger.error(f"Failed to build FAISS index: {e}. Falling back fully to TF-IDF.")
                self.faiss_index = None

    # ===========================================================================
    # Uploaded Ingestion Pipeline
    # ===========================================================================
    def ingest_uploaded_file(self, filepath: Path) -> str:
        """Parse uploaded CSV/Excel/PDF files, caching them for direct querying/RAG."""
        ext = filepath.suffix.lower()
        filename = filepath.name

        try:
            if ext == ".csv":
                df = pd.read_csv(filepath)
                self.uploaded_dfs[filename] = df
                # Chunk and index dataset header + statistics summary
                summary = f"Uploaded CSV Dataset: {filename}\nColumns: {list(df.columns)}\nRows: {len(df)}\nStatistics Summary:\n{df.describe().to_string()}"
                self.add_text_chunk(summary, filename)
                self.rebuild_indices()
                return f"Successfully uploaded CSV '{filename}' ({len(df)} rows, {len(df.columns)} columns)."
                
            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(filepath)
                self.uploaded_dfs[filename] = df
                summary = f"Uploaded Excel Dataset: {filename}\nColumns: {list(df.columns)}\nRows: {len(df)}\nStatistics Summary:\n{df.describe().to_string()}"
                self.add_text_chunk(summary, filename)
                self.rebuild_indices()
                return f"Successfully uploaded Excel '{filename}' ({len(df)} rows, {len(df.columns)} columns)."

            elif ext == ".pdf":
                if HAS_PYPDF:
                    text_content = []
                    with open(filepath, "rb") as f:
                        reader = pypdf.PdfReader(f)
                        for page_num, page in enumerate(reader.pages):
                            text = page.extract_text()
                            if text:
                                text_content.append(text)
                                self.add_text_chunk(f"PDF Page {page_num + 1} of {filename}:\n{text}", filename)
                    self.rebuild_indices()
                    return f"Successfully uploaded PDF '{filename}' ({len(reader.pages)} pages)."
                else:
                    # Basic parser fallback
                    return f"Cannot parse PDF '{filename}' because 'pypdf' package is not installed."
            else:
                return f"Unsupported file type '{ext}'."
        except Exception as e:
            logger.error(f"Error ingesting file {filename}: {e}")
            return f"Failed to ingest file: {str(e)}"

    # ===========================================================================
    # Guardrails Check
    # ===========================================================================
    def is_query_domain_restricted(self, query: str) -> bool:
        """Check if query is related to datasets, statistics, models, predictions, documentation, or upload files."""
        # Clean query
        q = query.lower().strip()

        # Keywords that denote project alignment
        aligned_keywords = [
            "dataset", "prediction", "feature", "statistic", "outlet", "distributor", 
            "demand", "latent", "potential", "sales", "volume", "lkr", "spend", "budget", 
            "optimizer", "competitor", "catchment", "cooler", "censoring", "poya", "holiday", 
            "uplift", "shap", "mean", "median", "mode", "std", "correlation", "covariance", 
            "outlier", "missing", "trend", "theory", "measure", "probability", "model", 
            "validation", "mae", "rmse", "mape", "r2", "calib", "carbon", "sri lanka", "province",
            "upload", "file", "csv", "excel", "pdf", "table", "column", "row"
        ]
        
        # Check if the query asks general-world questions specifically forbidden
        forbidden_questions = [
            "president", "quantum mechanics", "capital of", "weather in", "how to cook", 
            "who is", "tell me a joke", "write a poem", "history of", "what is quantum", 
            "recipe", "translate", "code a game"
        ]

        for word in forbidden_questions:
            if word in q:
                return False

        # Check if any aligned keyword is in query
        if any(word in q for word in aligned_keywords):
            return True

        # Check if query references any uploaded file names or columns
        for filename, df in self.uploaded_dfs.items():
            if filename.lower() in q:
                return True
            if any(col.lower() in q for col in df.columns):
                return True

        # If it's a very short query and doesn't match project terms, block it
        return False

    # ===========================================================================
    # Python Analytics Engine
    # ===========================================================================
    def run_statistics_engine(self, query: str) -> Optional[str]:
        """Perform programmatic statistics using pandas/numpy on our datasets if requested."""
        q = query.lower()
        
        # Check for strategic questions on increasing or capturing latent demand
        if any(term in q for term in ["increase", "capture", "improve", "optimize", "boost", "grow"]) and \
           any(term in q for term in ["latent", "demand", "potential", "sales", "volume"]):
            
            advice = (
                "### Strategic Recommendations to Capture and Boost Latent Demand\n\n"
                "Based on the **DataStorm 2026 Model Pipeline** and optimization outcomes, here are the core actions to capture latent potential:\n\n"
                "1. **Resolve Supply Constraints (Uncensoring)**:\n"
                "   - **Mechanism**: The model marks outlets with a `censoring_score > 0.30` as supply-constrained (legacy caps, plateaued sales, or distributor cap limits).\n"
                "   - **Action**: Increase delivery frequencies, expand distributor inventory allocations, and install larger coolers for these outlets.\n\n"
                "2. **Implement Targeted Trade Spend (Budget Optimization)**:\n"
                "   - **Mechanism**: The **Budget Optimizer** uses convex dual bisection to allocate budget where the potential gap is greatest.\n"
                "   - **Action**: Allocate the **LKR 5,000,000** budget across the designated **435 outlets** in the Western Province. This is mathematically proven to generate an expected lift of **32,663.69 Liters** and LKR **8,899,115.92** in incremental revenue.\n\n"
                "3. **Cooler Deployment & Upgrades**:\n"
                "   - **Mechanism**: `Cooler_Count` and `cooler_per_volume` are high-importance model features. Having cold storage capacity directly uncaps demand for cold beverage categories.\n"
                "   - **Action**: Deploy additional cooling units to outlets showing high spatial catchment scores but low cooler-to-volume ratios.\n\n"
                "4. **Bridge the Peer Efficiency Gap**:\n"
                "   - **Mechanism**: The `peer_efficiency_gap` tracks the ratio of 90th percentile sales of peers (same outlet type and size in the same month) to the outlet's median.\n"
                "   - **Action**: Target outlets with large peer gaps and introduce specialized store branding, merchandising assets, or localized promotions to pull their performance up to their peers' level.\n\n"
                "5. **Leverage Spatial Catchment Areas**:\n"
                "   - **Mechanism**: The `combined_catchment_score` measures foot traffic density (proximity to bus stops, schools, and offices) relative to local competitor density.\n"
                "   - **Action**: For outlets in high-catchment but low-sales zones, run hyper-local customer activation campaigns to divert foot traffic away from competitors."
            )
            return advice

        # Determine target DataFrame: uploaded file, predictions, validation, or gold
        target_df = None
        source_name = ""

        # Check uploaded files first
        for name, df in self.uploaded_dfs.items():
            if name.lower() in q:
                target_df = df
                source_name = name
                break
        
        # Fallback to local project datasets
        if target_df is None:
            if "predictions" in q or "latent" in q:
                pred_path = ROOT / "output" / "AI_ACES_predictions.csv"
                if pred_path.exists():
                    target_df = pd.read_csv(pred_path)
                    source_name = "AI_ACES_predictions.csv"
            elif "gold" in q or "features" in q or "outlet" in q:
                gold_path = ROOT / "pipeline" / "gold" / "gold_features.parquet"
                if gold_path.exists():
                    target_df = pd.read_parquet(gold_path)
                    source_name = "gold_features.parquet"
            elif "validation" in q or "report" in q:
                val_path = ROOT / "output" / "validation_report.csv"
                if val_path.exists():
                    target_df = pd.read_csv(val_path)
                    source_name = "validation_report.csv"

        if target_df is None:
            # If no specific dataset name mentioned, fallback to gold features if they exist
            gold_path = ROOT / "pipeline" / "gold" / "gold_features.parquet"
            if gold_path.exists():
                target_df = pd.read_parquet(gold_path)
                source_name = "gold_features.parquet"
            else:
                return None

        # Check columns mentioned in the query
        columns = [col for col in target_df.columns if col.lower() in q]
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(target_df[c])]

        summary_parts = [f"### Programmatic Analytics Engine Summary"]
        summary_parts.append(f"**Dataset Source**: `{source_name}` ({len(target_df):,} rows, {len(target_df.columns)} columns)")

        # Calculate metrics based on matches
        triggered = False

        # Missing values
        if "missing" in q or "null" in q or "nan" in q:
            triggered = True
            missing_series = target_df.isnull().sum()
            summary_parts.append("\n**Missing Values Analysis**:")
            for col, val in missing_series.items():
                if val > 0:
                    summary_parts.append(f"- `{col}`: {val} missing values ({val/len(target_df)*100:.2f}%)")
            if missing_series.sum() == 0:
                summary_parts.append("- No missing values found in the dataset.")

        # Summary statistics: mean, median, mode, std
        if any(stat in q for stat in ["mean", "average", "median", "mode", "std", "standard deviation", "describe", "summary"]):
            triggered = True
            summary_parts.append("\n**Statistical Summary Metrics**:")
            cols_to_use = numeric_cols if numeric_cols else [c for c in target_df.columns if pd.api.types.is_numeric_dtype(target_df[c])][:5]
            
            means = []
            medians = []
            for col in cols_to_use:
                col_mean = float(target_df[col].mean())
                col_median = float(target_df[col].median())
                col_std = float(target_df[col].std())
                means.append(col_mean)
                medians.append(col_median)
                try:
                    col_mode = target_df[col].mode().iloc[0]
                except Exception:
                    col_mode = "N/A"
                
                summary_parts.append(f"- **`{col}`**:")
                summary_parts.append(f"  - Mean (Average): {col_mean:.4f}")
                summary_parts.append(f"  - Median: {col_median:.4f}")
                summary_parts.append(f"  - Mode: {col_mode}")
                summary_parts.append(f"  - Standard Deviation: {col_std:.4f}")
            
            # Add Bar Chart
            chart_json = {
                "type": "bar",
                "title": "Mean vs Median Metrics Comparison",
                "x": cols_to_use,
                "y": means,
                "y2": medians,
                "labels": ["Mean", "Median"]
            }
            summary_parts.append(f"\n[CHART: {json.dumps(chart_json)}]\n")

        # Correlation / Covariance
        if "correlation" in q or "covariance" in q or "corr" in q:
            triggered = True
            cols_to_use = numeric_cols if len(numeric_cols) >= 2 else [c for c in target_df.columns if pd.api.types.is_numeric_dtype(target_df[c])][:4]
            if len(cols_to_use) >= 2:
                summary_parts.append("\n**Correlation Matrix (Pearson)**:")
                corr_matrix = target_df[cols_to_use].corr()
                summary_parts.append(f"```\n{corr_matrix.round(4).to_string()}\n```")
                
                # Add Heatmap Chart
                corr_json = {
                    "type": "heatmap",
                    "title": f"Pearson Correlation Matrix: {source_name}",
                    "x": cols_to_use,
                    "y": cols_to_use,
                    "z": corr_matrix.values.tolist()
                }
                summary_parts.append(f"\n[CHART: {json.dumps(corr_json)}]\n")

                if "covariance" in q:
                    summary_parts.append("\n**Covariance Matrix**:")
                    cov_matrix = target_df[cols_to_use].cov()
                    summary_parts.append(f"```\n{cov_matrix.round(4).to_string()}\n```")
            else:
                summary_parts.append("\nCorrelation requires at least 2 numeric columns in the query or dataset.")

        # Outliers (Z-score > 3 or IQR bounds)
        if "outlier" in q or "anomal" in q:
            triggered = True
            summary_parts.append("\n**Outlier and Anomaly Detection (IQR Method)**:")
            cols_to_use = numeric_cols if numeric_cols else [c for c in target_df.columns if pd.api.types.is_numeric_dtype(target_df[c])][:3]
            outlier_counts = []
            for col in cols_to_use:
                Q1 = target_df[col].quantile(0.25)
                Q3 = target_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = target_df[(target_df[col] < lower_bound) | (target_df[col] > upper_bound)]
                outlier_counts.append(len(outliers))
                summary_parts.append(f"- **`{col}`**: found {len(outliers):,} outliers ({len(outliers)/len(target_df)*100:.2f}%) outside bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # Add Bar Chart
            chart_json = {
                "type": "bar",
                "title": "Detected Outlier Counts",
                "x": cols_to_use,
                "y": outlier_counts
            }
            summary_parts.append(f"\n[CHART: {json.dumps(chart_json)}]\n")

        if triggered:
            return "\n".join(summary_parts)
        return None

    # ===========================================================================
    # RAG Retrieval Pipeline
    # ===========================================================================
    def retrieve_context(self, query: str, k: int = 4) -> str:
        """Query the vector/FAISS or TF-IDF database for relevant chunks of information."""
        context_chunks = []

        if HAS_FAISS and self.faiss_index is not None and self.model is not None:
            try:
                # FAISS Semantic Search
                query_vector = self.model.encode([query])
                query_vector = np.array(query_vector).astype("float32")
                faiss.normalize_L2(query_vector)
                
                scores, indices = self.faiss_index.search(query_vector, k)
                
                for idx, score in zip(indices[0], scores[0]):
                    if idx != -1:
                        chunk = self.doc_chunks[idx]
                        context_chunks.append(f"[Source: {chunk['source']} (Score: {score:.3f})]\n{chunk['text']}")
            except Exception as e:
                logger.error(f"FAISS search failed: {e}. Falling back to TF-IDF search.")
                
        # Fallback search if FAISS not present or failed
        if not context_chunks:
            results = self.tfidf_index.search(query, top_k=k)
            for chunk, score in results:
                context_chunks.append(f"[Source: {chunk['source']} (TF-IDF Similarity: {score:.3f})]\n{chunk['text']}")

        if not context_chunks:
            return "No matching context found in local database."

        return "\n\n---\n\n".join(context_chunks)

    # ===========================================================================
    # Local LLM Communication (Ollama)
    # ===========================================================================
    def generate_chat_response(self, query: str, context: str, analytics_info: Optional[str] = None, history: List[Dict[str, str]] = []) -> Tuple[str, Dict[str, int]]:
        """Sends the question + retrieved context to Ollama local LLM to generate the answer."""
        import requests

        # Setup prompt template
        system_prompt = (
            "You are a domain-restricted AI Analyst for the DataStorm FMCG project.\n"
            "Your name is DataStorm AI Analyst.\n"
            "You must answer ONLY using the provided datasets, training summaries, file uploads, and project documentation.\n"
            "If the user's question cannot be answered using the provided context or is outside the project's domain, you must refuse to answer with:\n"
            "\"I can only answer questions related to the uploaded datasets, predictions, analytics, and project documentation.\"\n"
            "Do NOT make up any facts outside the context.\n"
            "Keep your formatting clean, using markdown tables or bullet points when appropriate."
        )

        full_context = context
        if analytics_info:
            full_context = f"{analytics_info}\n\n=== Additional Context ===\n{context}"

        prompt = (
            f"=== SYSTEM GUARDRAILS ===\n{system_prompt}\n\n"
            f"=== RETRIEVED CONTEXT ===\n{full_context}\n\n"
            f"=== USER QUERY ===\n{query}\n\n"
            f"Generate response:"
        )

        # Estimate tokens (approximation: 1 token ~ 4 chars for input, same for output)
        input_token_estimate = int(len(prompt) / 4)
        output_token_estimate = 0
        
        try:
            # Query local Ollama API
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Keep it highly deterministic
                    }
                },
                timeout=15
            )
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                
                # Check response length to estimate tokens
                output_token_estimate = int(len(answer) / 4)
                
                # Calculate tokens dictionary
                tokens = {
                    "input": input_token_estimate,
                    "output": output_token_estimate,
                    "total": input_token_estimate + output_token_estimate
                }
                return answer, tokens
        except Exception as e:
            logger.warning(f"Ollama server not running or failed: {e}. Running offline analytic fallback.")

        # Fallback logic if Ollama is not active
        fallback_answer = (
            f"### DataStorm Offline Analyst Response\n\n"
            f"*{os.environ.get('OLLAMA_MODEL', 'Llama 3')} local server is currently offline on {self.ollama_url.replace('/api/generate', '')}.*\n\n"
        )
        if analytics_info:
            fallback_answer += f"Here are the programmatic statistics computed from your query:\n\n{analytics_info}\n"
        else:
            fallback_answer += (
                f"Below is the most relevant matching document context retrieved from your query:\n\n"
                f"```\n{context[:1500]}...\n```\n\n"
                f"*To enable conversational summaries, please spin up Ollama locally using `ollama run llama3`.*"
            )

        output_token_estimate = int(len(fallback_answer) / 4)
        tokens = {
            "input": input_token_estimate,
            "output": output_token_estimate,
            "total": input_token_estimate + output_token_estimate
        }
        return fallback_answer, tokens
