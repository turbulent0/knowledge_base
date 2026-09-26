# Snowflake — Level Up Hub (agenda notes)

Quick map of the **Level Up Hub** learning path (start → core topics → Snowpark).

## Start here

1. **First concepts** — platform intro, who teaches what; entry point before architecture.

## Core path (numbered modules)

2. **Snowflake architecture** — Level Up hub uses classic **3 tiers** (cloud services → compute → storage). First-concepts course adds **5 layers** (below); same idea: cloud-only, storage/compute split, scale per workload.

### Snowflake's five layers (brief)

Built for the cloud: less ops than legacy platforms; security and query optimization largely automated.

| Layer | What it does |
|--------|----------------|
| **Storage** | Centralized, managed object storage (AWS/Azure/GCP); data persisted once, independent of compute. |
| **Compute** | Virtual warehouses — elastic clusters for queries, loads, transforms; scale up/out without touching storage. |
| **Cortex AI** | In-account AI/ML (LLMs, embeddings, Search, Analyst, etc.) on governed data. |
| **Cloud services** | Control plane: auth, metadata, optimizer, transactions, Time Travel metadata — coordinates storage + compute. |
| **SnowGrid** | Global layer: replication, failover, and secure sharing across regions/clouds (Snowflake-managed connectivity). |

3. **Data loading & staging**

   **Stages** — holding area before data lands in a table.
   - **Internal** — Snowflake-managed storage in your account.
   - **External** — pointer to cloud object storage (S3, GCS, Azure Blob).

   **COPY INTO** — main load from stage → table; CSV, JSON, Parquet, etc.; optional light transforms while loading. **Needs a virtual warehouse.**

   **Snowpipe** — continuous / event-driven loads when new files hit an external stage; near-real-time without manual `COPY`. **Uses Snowflake-managed serverless compute** (not your warehouse).

   Typical flow: external stage on bucket → `COPY INTO` (or Snowpipe for ongoing drops).

### Snippets — stage & load (module 3)

```sql
-- external stage → S3 (credentials/integration depend on your account setup)
CREATE OR REPLACE STAGE my_s3_stage
  URL = 's3://mybucket/path/'
  FILE_FORMAT = (TYPE = CSV);

COPY INTO sales_table
FROM @my_s3_stage/sales_data.csv
FILE_FORMAT = (TYPE = CSV);

-- course-style path (same load, file named explicitly)
COPY INTO sales_table
FROM 's3://mybucket/sales_data.csv'
FILE_FORMAT = (TYPE = 'CSV');
```

4. **Resource monitors** — credit usage tracking and limits on **virtual warehouses** (cost/control).

5. **Snowflake ecosystem** — partners, connectors, and tools around the platform (integration landscape).

6. **Snowflake accounts and assurances** — org/account model, security, compliance, trust boundaries.

7. **Object hierarchy** — naming/containment: account → database → schema → tables/views/stages/etc.

8. **Backup and recovery** — Time Travel, Fail-safe, continuity (point-in-time, retention concepts).

## Cortex AI (minimal map)

Snowflake’s AI/ML layer on top of your data — LLMs and embeddings run in-account; no data leaves for inference when using Cortex services.

| Piece | Role |
|--------|------|
| **Cortex Search** | Semantic / hybrid search over documents and text in Snowflake (RAG-style retrieval). |
| **Cortex Analyst** | Natural-language questions → SQL / charts on governed structured data. |
| **Snowpark** | In-platform code (Python, etc.) for transforms, ML, and calling Cortex from pipelines. |
| **Copilot** | AI assistant in Snowsight / SQL worksheets (explain, generate, refine SQL and related tasks). |
| **Streamlit in Snowflake** | Build and host data apps (UI) inside the account, often paired with Snowpark + Cortex. |

## Next track

- **Explore Level Up: Snowpark** → **Snowpark Essentials — Getting Started** (Python, Streamlit; in-platform data engineering/ML-style workloads).

## Study order (from the hub graphic)

```
First concepts → Architecture → Data loading → Resource monitors
     ↑                              ↓
Ecosystem    Accounts & assurances   Object hierarchy
                    ↓
            Backup and recovery → Snowpark (optional branch)
```

*Fill in course takeaways, labs, and links under each section as you complete modules.*
