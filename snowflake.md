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

   **Bulk data loading with `COPY INTO`** — one-off or scheduled batch loads from a stage into a native table. Supports CSV, JSON, Parquet, etc.; column mapping and light transforms at load time. **Runs on a virtual warehouse** you start/suspend.

   **Continuous data loading with Snowpipe** — micro-batch ingestion: when new files land in external stage storage, Snowpipe ingests them automatically (event/notifications). Good for frequent small drops without rerunning manual `COPY`. **Serverless compute** (Snowflake-managed), not your warehouse.

   **Querying data via external tables** — metadata object over files in external stage/cloud storage; query with SQL **without** copying data into Snowflake tables first. Data stays in the bucket; warehouse reads files at query time (good for exploration, lakehouse patterns; different cost/latency vs loading into tables).

   | Pattern | Data in Snowflake tables? | Typical compute |
   |--------|---------------------------|-----------------|
   | `COPY INTO` | Yes (persisted) | Virtual warehouse |
   | Snowpipe | Yes (persisted) | Serverless (Snowpipe) |
   | External table | No (files stay external) | Virtual warehouse |

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

**FLOW** — local files / cloud storage → stage → pipe (optional) → table

4. **Resource monitors** — track **credit usage** for one or more virtual warehouses; quotas, resets, alerts, and auto-suspend to control cost.

   **Creating & assigning**
   - Attach a monitor to **one warehouse** or **several**; shared monitor = **combined** usage against **one** quota.
   - Set **credit quota** and **reset frequency** (daily, weekly, monthly — monthly is default).
   - Manage with **`CREATE RESOURCE MONITOR`**, **`ALTER RESOURCE MONITOR`**, and UI.
   - *Practice:* per department/warehouse monitors avoid one team’s spike affecting others.

   **Thresholds & actions**
   - Define usage levels (e.g. 50%, 80%, 100% of quota).
   - At each level: **`NOTIFY`**, **`SUSPEND`** (finish running queries), or **`SUSPEND_IMMEDIATE`**.
   - Trigger **below 100%** (e.g. 90%) so admins can react before overrun.

   **Notifications**
   - Enable alerts (email / in-account) for admins or designated users at chosen thresholds.
   - Review usage regularly; resize warehouses or pause non-critical work when warned.

   | Pitfall | Better approach |
   |--------|------------------|
   | Many warehouses, one monitor | Separate monitors per warehouse or team |
   | Actions only at 100% | Notify earlier; suspend with buffer under quota |
   | Notifications off | Enable for admins; watch usage reports |

5. **Snowflake ecosystem** — first-party CLI/SDKs plus partner connectors for ingest, analytics, and ML.

   **Built-in tools**
   - **SnowCLI** — command line: run SQL, manage objects/data, automate jobs (reports, backups, scripts).
   - **Snowpark** — Python, Java, Scala in Snowflake: transforms, pipelines, analytics/ML on live data (e.g. daily sales summary without exporting).

   **Third-party integrations**
   - **Data integration (ELT/ETL)** — Matillion, Fivetran, Informatica: sync from apps/DBs into Snowflake with minimal custom load code.
   - **BI** — Tableau, Looker, Power BI: dashboards and reporting on Snowflake tables.
   - **ML** — DataRobot, SageMaker, etc.: train/score using data in Snowflake for forecasting and advanced analytics.

   *Example chain:* Fivetran → Snowflake (sales, inventory, logistics) → Tableau dashboards; optional ML tool for demand forecasting.

6. **Snowflake accounts and assurances**

   **Account management**
   - Each Snowflake **account** is tied to one **cloud** (AWS, Azure, GCP) and **region**; configuration and data residency follow that choice.
   - Accounts are uniquely identified and administered for access, billing, and cross-team use.

   **Account identifiers**
   - Every account has a **unique identifier** (includes cloud + region context) used in URLs, connectors, and cross-account sharing — keep identifiers straight for global teams and integrations.

   **Access control framework (DAC + RBAC)**
   - **DAC (discretionary):** every securable object has an **owner** who can grant access to others.
   - **RBAC:** **privileges** attach to **roles**; roles assign to **users** (and roles can grant to other roles → **role hierarchy**).
   - Default stance: **no access** unless explicitly **granted**.

   | Term | Meaning |
   |------|---------|
   | **Securable object** | Entity you can grant access to (DB, schema, table, warehouse, …). |
   | **Privilege** | Defined level of access on an object (SELECT, USAGE, …). |
   | **Role** | Bundle of privileges; assigned to users and/or other roles. |
   | **User** | Identity Snowflake recognizes (person or service/program). |

   *Model in one line:* users assume roles → roles hold privileges → privileges unlock securable objects (plus ownership grants via DAC).

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
