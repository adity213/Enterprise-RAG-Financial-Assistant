# 🚀 Hybrid GraphRAG — Complete Execution Guide

> **Goal:** Build a production-grade Hybrid Knowledge Graph + Vector RAG API that answers both simple and multi-hop questions about financial documents, with source citations.

> [!IMPORTANT]
> **What's already done:** We have already built Phase 0 (project setup) and Phase 1 (vector ingestion + query pipeline with FAISS). The project files are at:
> `C:\Users\darsh\OneDrive\Desktop\Placement\RAG project\rag-api\`
> 
> This guide picks up from there and takes you all the way to a resume-ready, deployed project.

---

## 📋 Pre-Flight Checklist

Before starting, confirm you have:
- [x] Python 3.12 installed
- [x] GitHub account
- [x] Groq API Key (already in `.env`)
- [ ] Docker Desktop installed and running
- [ ] Virtual environment set up in the new project folder

---

## STAGE 1: Environment Setup (One-Time)

### Step 1.1 — Install Docker Desktop

1. Go to [https://docker.com/products/docker-desktop](https://docker.com/products/docker-desktop)
2. Download **Docker Desktop for Windows**
3. Run the installer — accept all defaults
4. If prompted about **WSL 2**, say **Yes** to enable it
5. **Restart your PC** when prompted

### Step 1.2 — Verify Docker Works

After restart, open **PowerShell** and run:
```powershell
docker --version
```

**Expected output:**
```
Docker version 27.x.x, build xxxxxxx
```

If you see this, Docker is ready. If you get an error about virtualization, you need to enable **VT-x** in your BIOS (see troubleshooting section at the bottom).

### Step 1.3 — Start Neo4j (Graph Database)

This single command downloads and starts a Neo4j graph database in a Docker container:

```powershell
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/ragproject123 neo4j:community
```

**What each flag means:**
| Flag | Meaning |
|------|---------|
| `-d` | Run in background (detached mode) |
| `--name neo4j` | Give the container a friendly name |
| `-p 7474:7474` | Expose the Neo4j Browser UI |
| `-p 7687:7687` | Expose the Bolt protocol (how Python talks to Neo4j) |
| `-e NEO4J_AUTH=neo4j/ragproject123` | Set username=`neo4j`, password=`ragproject123` |
| `neo4j:community` | Use the free community edition image |

### Step 1.4 — Verify Neo4j Works

1. Open your browser and go to **`http://localhost:7474`**
2. You should see the Neo4j Browser interface
3. Log in with:
   - **Username:** `neo4j`
   - **Password:** `ragproject123`
4. You should see an empty database — that's perfect!

> [!TIP]
> **Useful Docker commands you'll need later:**
> ```powershell
> docker start neo4j       # Start the container (after PC restart)
> docker stop neo4j        # Stop the container
> docker ps                # See running containers
> docker logs neo4j        # See Neo4j logs if something breaks
> ```

### Step 1.5 — Set Up Python Virtual Environment

Navigate to the project folder and create a fresh virtual environment:

```powershell
cd "C:\Users\darsh\OneDrive\Desktop\Placement\RAG project\rag-api"

python -m venv venv

.\venv\Scripts\pip install -r requirements.txt
```

> [!NOTE]
> This will take 5-10 minutes because it downloads PyTorch (~120MB), FAISS, sentence-transformers, and all their dependencies. Wait for it to finish completely.

### Step 1.6 — Add Neo4j Driver to Dependencies

After the base install finishes, install the Neo4j Python driver:

```powershell
.\venv\Scripts\pip install neo4j==5.20.0
```

Then add it to `requirements.txt` so it's tracked:
```
neo4j==5.20.0
```

### Step 1.7 — Update `.env` with Neo4j Credentials

Open the `.env` file and add these lines at the bottom:
```env
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="ragproject123"
```

### Step 1.8 — Verify Everything Works Together

Start the FastAPI server:
```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Started server process [XXXXX]
INFO:     Application startup complete.
```

Open **`http://localhost:8000/docs`** in your browser — you should see the Swagger UI.

> [!IMPORTANT]
> **Checkpoint:** At this point you should have 3 things running:
> 1. ✅ Neo4j on `localhost:7474` (Docker)
> 2. ✅ FastAPI on `localhost:8000` (Python)
> 3. ✅ FAISS vector database (file-based, automatic)

---

## STAGE 2: Knowledge Graph Construction (Phase 2)

This is the **core differentiator** of your project. We are building a pipeline that reads PDF text, extracts real-world entities and relationships using AI, and stores them as a graph.

### Step 2.1 — Define the Ontology

Before writing any code, we define **what types of entities and relationships** we care about. This prevents the graph from becoming a mess of random connections.

**Entity Types (nodes in the graph):**
| Entity Type | Example |
|-------------|---------|
| `Company` | Apple Inc., Foxconn |
| `Person` | Tim Cook, Luca Maestri |
| `Product` | iPhone, MacBook |
| `Location` | Cupertino, California |
| `Financial_Metric` | Revenue, Net Income |
| `Regulation` | SOX, GDPR |

**Relationship Types (edges in the graph):**
| Relationship | Example |
|-------------|---------|
| `SUBSIDIARY_OF` | Beats Electronics → Apple |
| `CEO_OF` / `EMPLOYS` | Tim Cook → Apple |
| `SUPPLIES_TO` | Foxconn → Apple |
| `HEADQUARTERED_IN` | Apple → Cupertino |
| `REPORTED_METRIC` | Apple → Revenue: $394B |
| `COMPETES_WITH` | Apple → Samsung |
| `REGULATED_BY` | Apple → SEC |

### Step 2.2 — Build `graph_store.py`

**File:** `app/core/graph_store.py`

This module handles all communication with Neo4j. Key design decisions:
- Uses `MERGE` instead of `CREATE` — so re-uploading the same document doesn't create duplicate nodes
- Every relationship carries a `source_chunk_id` — so any graph fact can be traced back to the exact sentence it came from
- Parameterized queries only — **never** raw LLM-generated Cypher (security best practice)

### Step 2.3 — Build `entity_extraction.py`

**File:** `app/services/entity_extraction.py`

This is where the magic happens. For each text chunk:
1. Send the chunk to the Groq LLM with a strict JSON schema prompt
2. The LLM returns structured entities and relationships
3. We validate the JSON response against our ontology
4. If validation fails, we retry (up to 2 times)
5. Valid entities get sent to `graph_store.py` for writing to Neo4j

**The prompt engineering is critical here.** We use few-shot examples to teach the LLM exactly what format we expect.

### Step 2.4 — Build `entity_resolution.py`

**File:** `app/services/entity_resolution.py`

This solves the "same entity, different names" problem:
- "Apple Inc." and "Apple" and "AAPL" should all be **one node**
- We embed each entity name and compare via cosine similarity
- If similarity > 0.85, we collapse them into a single canonical name
- An alias list is maintained (so we can still search by any variant)

### Step 2.5 — Update `ingest_service.py`

The existing ingestion pipeline only feeds FAISS. Now we upgrade it to:
1. Parse PDF → extract text (already done)
2. Chunk text (already done)
3. Embed chunks → store in FAISS (already done)
4. **NEW:** Extract entities from each chunk → resolve duplicates → store in Neo4j
5. Every Neo4j edge gets tagged with the `chunk_id` that produced it

### Step 2.6 — Test: Idempotent Ingestion

Upload the same PDF twice via `/api/v1/upload`. Then open Neo4j Browser (`localhost:7474`) and run:
```cypher
MATCH (n) RETURN count(n)
```

The node count should be the **same** both times. If it doubled, the `MERGE` logic has a bug.

---

## STAGE 3: Query Router + Graph Query Engine (Phase 3)

### Step 3.1 — Build `query_router.py`

**File:** `app/services/query_router.py`

This is a cheap, fast LLM classifier. When a user asks a question, it decides:

| Question Type | Route | Example |
|--------------|-------|---------|
| Single-fact lookup | `vector` | "What is the company's revenue?" |
| Multi-hop relationship | `graph` | "Which subsidiaries share a board member?" |
| Unclear / could be both | `both` | "How does Company X's supply chain work?" |

We use a **few-shot prompt** (not fine-tuning) — the LLM sees 5-6 example classifications and returns one of three labels: `vector`, `graph`, or `both`.

### Step 3.2 — Build `graph_query_service.py`

**File:** `app/services/graph_query_service.py`

This module:
1. Takes the user's question
2. Extracts entity names from it (using the LLM)
3. Resolves those names to existing Neo4j node IDs (using `entity_resolution.py`)
4. Picks the right **Cypher template** based on the query type
5. Runs the parameterized Cypher query
6. Converts the graph paths into readable sentences

**Cypher Template Library** (pre-written, never LLM-generated):
```cypher
// Find direct relationships between two entities
MATCH (a)-[r]-(b) WHERE a.name = $entity1 AND b.name = $entity2 RETURN *

// Find 2-hop path between entities
MATCH path = (a)-[*1..2]-(b) WHERE a.name = $entity1 AND b.name = $entity2 RETURN path

// List all entities related to X
MATCH (a)-[r]->(b) WHERE a.name = $entity RETURN type(r), b.name
```

### Step 3.3 — Log Every Routing Decision

Every time the router makes a decision, we log:
- The original question
- The chosen path (`vector` / `graph` / `both`)
- The response quality (did the LLM find an answer?)

This data is gold for the benchmark phase later.

---

## STAGE 4: Hybrid Answer Synthesis (Phase 4)

### Step 4.1 — Update `query_service.py`

The current query service only does: question → FAISS → LLM → answer.

Now it becomes:
1. **Router** classifies the question
2. **If `vector`:** Use FAISS (existing logic)
3. **If `graph`:** Use Neo4j graph traversal
4. **If `both`:** Run both paths, merge the results
5. **Deduplicate** overlapping information from both sources
6. **Answer Synthesizer** sends merged context to the LLM
7. LLM generates answer with **per-claim citations**
8. **Citation Validator** checks every cited `chunk_id` was actually retrieved — if not, regenerate

### Step 4.2 — Update `/api/v1/ask` Response Schema

The response now includes routing information:
```json
{
  "status": "success",
  "data": {
    "question": "Which subsidiaries of Apple supply Samsung?",
    "route": "graph",
    "answer": "Based on the 10-K filing, ...",
    "sources": [
      {"type": "graph", "path": "Apple -[SUBSIDIARY_OF]-> Beats -[SUPPLIES_TO]-> Samsung", "chunk_id": "..."},
      {"type": "vector", "text_preview": "...", "similarity_score": 0.82}
    ]
  }
}
```

### Step 4.3 — Test Both Paths

Test with two different questions:
1. **Vector path:** `"What is the total revenue reported?"`  
   → Should route to `vector`, pull from FAISS
2. **Graph path:** `"Which executives also serve on the board of a subsidiary?"`  
   → Should route to `graph`, traverse Neo4j

---

## STAGE 5: Containerization (Phase 5)

### Step 5.1 — Write the Dockerfile

**File:** `Dockerfile` (in project root)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 5.2 — Write docker-compose.yml

**File:** `docker-compose.yml` (in project root)

This spins up **all 3 services** with one command:
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - neo4j

  neo4j:
    image: neo4j:community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/ragproject123
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

### Step 5.3 — Test: One-Command Startup

```powershell
docker-compose up --build
```

**Expected result:** Both the API (`localhost:8000`) and Neo4j (`localhost:7474`) start up together. The entire system runs from a single command.

---

## STAGE 6: CI/CD Pipeline (Phase 6)

### Step 6.1 — Update GitHub Actions Workflow

**File:** `.github/workflows/ci.yml`

The pipeline now:
1. On every `git push` → triggers automatically
2. Spins up a Neo4j service container (for integration tests)
3. Installs Python dependencies
4. Runs all `pytest` tests
5. Builds the Docker image (validates it compiles)

### Step 6.2 — Push to GitHub

```powershell
cd "C:\Users\darsh\OneDrive\Desktop\Placement\RAG project\rag-api"
git init
git add .
git commit -m "feat: Hybrid GraphRAG with Knowledge Graph + Vector retrieval"
git remote add origin https://github.com/YOUR_USERNAME/enterprise-rag-api.git
git push -u origin main
```

### Step 6.3 — Verify Pipeline

Go to your GitHub repo → **Actions** tab → You should see the CI pipeline running. Wait for the green ✅ checkmark.

---

## STAGE 7: Benchmark — Hybrid vs. Vector-Only (Phase 7)

This is the **portfolio artifact** — the proof that your system actually works better.

### Step 7.1 — Create Benchmark Question Set

**File:** `eval/benchmark_questions.json`

Create 50-100 questions, stratified by difficulty:

| Category | Count | Example |
|----------|-------|---------|
| Single-hop (easy) | 20 | "What was total revenue?" |
| Two-hop | 15 | "Who is the CEO of Company X's largest subsidiary?" |
| Three-hop | 10 | "Which suppliers of Company X also supply Company Y?" |
| Aggregation | 5 | "How many subsidiaries does Company X have?" |
| Out-of-scope | 5 | "What's the weather today?" (should refuse) |

### Step 7.2 — Build Benchmark Runner

**File:** `eval/run_benchmark.py`

This script:
1. Runs every question through the **vector-only** pipeline
2. Runs every question through the **hybrid** pipeline
3. Scores each answer (correct / partially correct / wrong / refused)
4. Groups results by hop count

### Step 7.3 — Generate the Results Table

The output should look like this:

```
| Hop Count   | Vector-Only | Hybrid  | Δ Accuracy |
|-------------|-------------|---------|------------|
| Single-hop  | 85%         | 87%     | +2%        |
| Two-hop     | 42%         | 78%     | +36%       |
| Three-hop   | 15%         | 65%     | +50%       |
| Aggregation | 30%         | 72%     | +42%       |
| Latency p95 | 1.2s        | 2.8s    | +1.6s      |
```

> [!IMPORTANT]
> **This table goes at the TOP of your README.** It's the first thing a recruiter sees. The widening gap as hop count increases is the entire value proposition.

---

## STAGE 8: Polish & Resume-Ready (Phase 8)

### Step 8.1 — Add Structured Logging

Every API request logs:
- Timestamp
- Question asked
- Route chosen (vector / graph / both)
- Response time
- Sources used

### Step 8.2 — Write the README

Structure your `README.md` like this:
1. **Benchmark table** (first thing visible!)
2. **Architecture diagram** (the ASCII art from the project plan)
3. **Quick Start** (how to run with `docker-compose up`)
4. **API Documentation** (link to `/docs`)
5. **Tech Stack** table
6. **"What Didn't Work"** section (shows intellectual honesty — interviewers love this)

### Step 8.3 — Record a 90-Second Demo Video

1. Open the Swagger UI (`localhost:8000/docs`)
2. Upload a 10-K PDF
3. Ask a single-fact question → show it routes to vector
4. Ask a multi-hop question → show it routes to graph
5. Show the Neo4j Browser with the knowledge graph visually

### Step 8.4 — Push Final Version

```powershell
git add .
git commit -m "feat: benchmarks, README, and final polish"
git push
```

---

## 🔧 Troubleshooting

### Docker won't start — "virtualization not detected"
1. Restart PC → Enter BIOS (F2/F10/Del during boot)
2. Find **Intel VT-x** or **AMD SVM Mode** → Enable it
3. Save & Exit → Restart
4. Open PowerShell as Admin → Run `wsl --install` → Restart again

### Neo4j container won't start
```powershell
docker logs neo4j          # Check what went wrong
docker rm neo4j             # Remove broken container
# Then re-run the docker run command from Step 1.3
```

### FAISS "directory not found" error
The vector database directory was deleted. Just re-upload your PDF — the directory is recreated automatically now.

### Groq model "decommissioned" or "not found"
Run `list_models.py` to see which models are currently available on your API key, then update `.env` accordingly.

### Port 8000 already in use
```powershell
# Find and kill the process using port 8000
$processId = (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue).OwningProcess
Stop-Process -Id $processId -Force
```

---

## 📌 Resume Line (When Complete)

> *Built a hybrid knowledge-graph + vector RAG system over SEC 10-K filings using FastAPI, FAISS, Neo4j, and Groq LLaMA; raised multi-hop question accuracy from X% to Y% against a vector-only baseline, containerized with Docker Compose and validated via GitHub Actions CI/CD.*
