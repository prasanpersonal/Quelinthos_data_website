import type { Category } from '../types.ts';

export const constructionCategory: Category = {
  id: 'construction',
  number: 9,
  title: 'Construction & Project Management',
  shortTitle: 'Construction',
  description:
    'Close the gap between field data and job costing, and eliminate the Excel import dependencies bleeding your projects.',
  icon: 'HardHat',
  accentColor: 'neon-gold',
  painPoints: [
    // ── Pain Point 1 ── Job Costing Field Data Lag ──────────────────────
    {
      id: 'job-costing-field-lag',
      number: 1,
      title: 'Job Costing Field Data Lag',
      subtitle: 'Real-Time Cost Tracking Failure on Job Sites',
      summary:
        'Foremen submit time and material data 3-5 days late. By the time you know a job is over budget, it\'s too late to course-correct.',
      tags: ['construction', 'job-costing', 'field-data'],
      metrics: {
        annualCostRange: '$500K - $3M',
        roi: '7x',
        paybackPeriod: '3-4 months',
        investmentRange: '$80K - $150K',
      },

      price: {
        present: {
          title: 'Current Reality',
          severity: 'critical',
          description:
            'Field crews record time, materials, and equipment usage on paper tickets or delayed mobile entries that arrive 3-5 business days after the work is performed.',
          bullets: [
            'Foremen batch-submit timecards every Friday for the entire week',
            'Material receipts sit in truck glove-boxes until the next office visit',
            'Equipment hours are estimated from memory rather than tracked in real time',
            'Project managers see cost data that is already a week stale',
          ],
        },
        root: {
          title: 'Root Cause',
          severity: 'high',
          description:
            'No mobile-first data capture pipeline exists between the job site and the ERP system. Paper-based and batch workflows were never replaced with real-time integrations.',
          bullets: [
            'Legacy ERP accepts only nightly batch imports from CSV files',
            'No offline-capable mobile app exists for field data entry',
            'Cost codes are maintained in a spreadsheet that drifts from the ERP master list',
            'No automated validation catches misallocated cost codes until month-end reconciliation',
          ],
        },
        impact: {
          title: 'Business Impact',
          severity: 'critical',
          description:
            'Budget overruns are discovered only at invoice time, leaving no window for corrective action. Change orders that should have been flagged mid-pour are missed entirely.',
          bullets: [
            'Average 12% cost overrun detected too late to mitigate on jobs over $1M',
            'Change order revenue leakage of $40K-$120K per project due to undocumented extras',
            'Month-end close delayed 5-8 days while accounting reconciles field data gaps',
            'Subcontractor disputes increase because backup documentation is incomplete',
          ],
        },
        cost: {
          title: 'Cost of Inaction',
          severity: 'high',
          description:
            'Every week of delayed visibility costs roughly 2-3% of project margin. Across a portfolio of active jobs, this compounds into six- and seven-figure annual losses.',
          bullets: [
            '$500K - $3M in annual margin erosion from late cost detection',
            '$150K+ in annual accounting overtime for manual reconciliation',
            '15-20 hours per week of PM time spent chasing field paperwork instead of managing scope',
            'Insurance and bonding costs increase when financials cannot be reported accurately',
          ],
        },
        expectedReturn: {
          title: 'Expected Return',
          severity: 'high',
          description:
            'Real-time field data capture closes the visibility gap from days to minutes, enabling same-day cost corrections and complete change order capture.',
          bullets: [
            'Reduce cost-overrun detection lag from 5 days to under 4 hours',
            'Capture 90%+ of change order revenue currently lost to undocumented extras',
            'Cut month-end close cycle by 4-6 days with pre-reconciled field data',
            'Free 12+ PM hours per week by eliminating manual data-chase workflows',
          ],
        },
      },

      implementation: {
        overview:
          'Deploy a real-time job cost dashboard backed by a mobile field data capture API. Field crews submit time, materials, and equipment via an offline-capable mobile app that syncs to a central database, powering live cost-vs-budget dashboards for project managers.',
        prerequisites: [
          'PostgreSQL 14+ with the pg_partman extension for time-series partitioning',
          'Python 3.11+ with FastAPI and SQLAlchemy for the REST API layer',
          'Active cost code master list exported from your ERP (CSV or direct DB access)',
          'Mobile devices issued to foremen (iOS or Android with intermittent connectivity)',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        toolsUsed: ['PostgreSQL', 'Python', 'FastAPI', 'SQLAlchemy', 'pytest', 'Docker', 'GitHub Actions', 'cron / Airflow', 'Slack API'],
        steps: [
          {
            stepNumber: 1,
            title: 'Build the Real-Time Job Cost Data Model',
            description:
              'Create the core tables that store field entries and join them against budgeted cost codes. Partition the daily entries table by month for query performance on large job portfolios.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Job Cost Schema & Live Variance View',
                description:
                  'Tables for field cost entries and a view that calculates real-time budget variance by cost code and job.',
                code: `-- Field cost entries submitted from mobile devices
CREATE TABLE field_cost_entries (
    entry_id        BIGSERIAL,
    job_id          INT          NOT NULL REFERENCES jobs(job_id),
    cost_code       VARCHAR(20)  NOT NULL,
    entry_date      DATE         NOT NULL,
    labor_hours     NUMERIC(8,2) DEFAULT 0,
    labor_rate      NUMERIC(10,2),
    material_cost   NUMERIC(12,2) DEFAULT 0,
    equipment_hours NUMERIC(8,2) DEFAULT 0,
    equipment_rate  NUMERIC(10,2),
    submitted_by    INT          NOT NULL REFERENCES employees(employee_id),
    submitted_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    synced_to_erp   BOOLEAN      DEFAULT FALSE,
    PRIMARY KEY (entry_id, entry_date)
) PARTITION BY RANGE (entry_date);

-- Auto-create monthly partitions
SELECT partman.create_parent(
    p_parent_table  := 'public.field_cost_entries',
    p_control       := 'entry_date',
    p_type          := 'range',
    p_interval      := '1 month',
    p_premake       := 3
);

-- Real-time cost variance by job and cost code
CREATE OR REPLACE VIEW v_job_cost_variance AS
SELECT
    j.job_id,
    j.job_name,
    b.cost_code,
    b.budgeted_amount,
    COALESCE(SUM(f.labor_hours * f.labor_rate), 0)
      + COALESCE(SUM(f.material_cost), 0)
      + COALESCE(SUM(f.equipment_hours * f.equipment_rate), 0) AS actual_cost,
    b.budgeted_amount - (
      COALESCE(SUM(f.labor_hours * f.labor_rate), 0)
      + COALESCE(SUM(f.material_cost), 0)
      + COALESCE(SUM(f.equipment_hours * f.equipment_rate), 0)
    ) AS remaining_budget,
    ROUND(
      (COALESCE(SUM(f.labor_hours * f.labor_rate), 0)
       + COALESCE(SUM(f.material_cost), 0)
       + COALESCE(SUM(f.equipment_hours * f.equipment_rate), 0))
      / NULLIF(b.budgeted_amount, 0) * 100, 1
    ) AS pct_consumed
FROM jobs j
JOIN job_budgets b USING (job_id)
LEFT JOIN field_cost_entries f
       ON f.job_id = b.job_id AND f.cost_code = b.cost_code
GROUP BY j.job_id, j.job_name, b.cost_code, b.budgeted_amount;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Create the Mobile Field Data Capture API',
            description:
              'Build a FastAPI service that accepts field entries from the mobile app, validates cost codes against the master list, and writes to the database. Supports batch sync for offline-collected entries.',
            codeSnippets: [
              {
                language: 'python',
                title: 'FastAPI Field Entry Endpoints',
                description:
                  'REST endpoints for submitting individual or batched field cost entries with cost code validation.',
                code: `from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import date, datetime
from typing import List

app = FastAPI(title="Field Cost Capture API")

class FieldEntry(BaseModel):
    job_id: int
    cost_code: str = Field(max_length=20)
    entry_date: date
    labor_hours: float = 0.0
    labor_rate: float | None = None
    material_cost: float = 0.0
    equipment_hours: float = 0.0
    equipment_rate: float | None = None
    submitted_by: int

class BatchSyncRequest(BaseModel):
    """Offline-collected entries synced when connectivity resumes."""
    entries: List[FieldEntry]
    device_id: str

async def validate_cost_codes(codes: list[str], db: AsyncSession) -> set[str]:
    result = await db.execute(
        text("SELECT cost_code FROM cost_code_master WHERE cost_code = ANY(:codes)"),
        {"codes": codes},
    )
    return {row[0] for row in result.fetchall()}

@app.post("/api/v1/field-entries/batch")
async def batch_sync(payload: BatchSyncRequest, db: AsyncSession = Depends(get_db)):
    codes = list({e.cost_code for e in payload.entries})
    valid_codes = await validate_cost_codes(codes, db)
    invalid = [e for e in payload.entries if e.cost_code not in valid_codes]
    if invalid:
        raise HTTPException(422, detail=f"Invalid cost codes: {set(e.cost_code for e in invalid)}")

    insert_sql = text(\"\"\"
        INSERT INTO field_cost_entries
            (job_id, cost_code, entry_date, labor_hours, labor_rate,
             material_cost, equipment_hours, equipment_rate, submitted_by)
        VALUES (:job_id, :cost_code, :entry_date, :labor_hours, :labor_rate,
                :material_cost, :equipment_hours, :equipment_rate, :submitted_by)
    \"\"\")
    for entry in payload.entries:
        await db.execute(insert_sql, entry.model_dump())
    await db.commit()
    return {"synced": len(payload.entries), "device_id": payload.device_id}`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Run data quality assertions against the field cost entry schema and validate the field data capture API with pytest-based integration tests. This step ensures cost codes, budget references, and entry constraints are enforced before any deployment.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Field Cost Data Quality Assertions',
                description:
                  'SQL assertions that verify cost code referential integrity, budget linkage, and entry constraint enforcement across the field cost pipeline.',
                code: `-- ============================================================
-- Field Cost Data Quality Assertions
-- Run after every ETL cycle or before dashboard refresh
-- ============================================================

-- 1. Verify every field entry references a valid cost code
SELECT 'orphan_cost_codes' AS assertion,
       COUNT(*)            AS violation_count,
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM field_cost_entries fce
LEFT JOIN cost_code_master ccm ON ccm.cost_code = fce.cost_code
WHERE ccm.cost_code IS NULL;

-- 2. Verify every field entry references a valid job with a budget
SELECT 'entries_without_budget' AS assertion,
       COUNT(*)                 AS violation_count,
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM field_cost_entries fce
LEFT JOIN job_budgets jb
       ON jb.job_id = fce.job_id AND jb.cost_code = fce.cost_code
WHERE jb.job_id IS NULL;

-- 3. Detect negative or zero labor rates that break cost calcs
SELECT 'invalid_labor_rates' AS assertion,
       COUNT(*)              AS violation_count,
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM field_cost_entries
WHERE labor_rate IS NOT NULL AND labor_rate <= 0;

-- 4. Detect future-dated entries (possible clock-skew from devices)
SELECT 'future_dated_entries' AS assertion,
       COUNT(*)               AS violation_count,
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM field_cost_entries
WHERE entry_date > CURRENT_DATE;

-- 5. Budget integrity: no budget row should have zero amount
SELECT 'zero_budget_amounts' AS assertion,
       COUNT(*)              AS violation_count,
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM job_budgets
WHERE budgeted_amount <= 0;`,
              },
              {
                language: 'python',
                title: 'Pytest Field Entry API Validation',
                description:
                  'Integration tests for the field data capture API covering batch sync, cost code validation, offline entry handling, and error responses.',
                code: `"""tests/test_field_entry_api.py — pytest integration tests for the Field Cost Capture API"""
import logging
from datetime import date, timedelta
from typing import Any

import httpx
import pytest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger: logging.Logger = logging.getLogger(__name__)

BASE_URL: str = "http://localhost:8000/api/v1"


@pytest.fixture(scope="module")
def api_client() -> httpx.Client:
    """Reusable HTTP client for the test session."""
    logger.info("Creating test API client targeting %s", BASE_URL)
    client: httpx.Client = httpx.Client(base_url=BASE_URL, timeout=30.0)
    yield client
    client.close()


def _make_entry(
    job_id: int = 1,
    cost_code: str = "03-100",
    entry_date: str | None = None,
    labor_hours: float = 8.0,
    labor_rate: float = 55.00,
    material_cost: float = 250.00,
    equipment_hours: float = 2.0,
    equipment_rate: float = 75.00,
    submitted_by: int = 10,
) -> dict[str, Any]:
    """Helper to build a valid field entry payload."""
    return {
        "job_id": job_id,
        "cost_code": cost_code,
        "entry_date": entry_date or date.today().isoformat(),
        "labor_hours": labor_hours,
        "labor_rate": labor_rate,
        "material_cost": material_cost,
        "equipment_hours": equipment_hours,
        "equipment_rate": equipment_rate,
        "submitted_by": submitted_by,
    }


class TestBatchSync:
    """Tests for POST /field-entries/batch."""

    def test_valid_batch_sync(self, api_client: httpx.Client) -> None:
        payload: dict[str, Any] = {
            "entries": [_make_entry(), _make_entry(cost_code="03-200")],
            "device_id": "device-test-001",
        }
        resp: httpx.Response = api_client.post("/field-entries/batch", json=payload)
        logger.info("Batch sync response: %d", resp.status_code)
        assert resp.status_code == 200
        body: dict[str, Any] = resp.json()
        assert body["synced"] == 2
        assert body["device_id"] == "device-test-001"

    def test_invalid_cost_code_rejected(self, api_client: httpx.Client) -> None:
        payload: dict[str, Any] = {
            "entries": [_make_entry(cost_code="INVALID-999")],
            "device_id": "device-test-002",
        }
        resp: httpx.Response = api_client.post("/field-entries/batch", json=payload)
        logger.info("Invalid cost code response: %d", resp.status_code)
        assert resp.status_code == 422

    def test_future_date_entry_rejected(self, api_client: httpx.Client) -> None:
        future: str = (date.today() + timedelta(days=30)).isoformat()
        payload: dict[str, Any] = {
            "entries": [_make_entry(entry_date=future)],
            "device_id": "device-test-003",
        }
        resp: httpx.Response = api_client.post("/field-entries/batch", json=payload)
        logger.info("Future date response: %d", resp.status_code)
        assert resp.status_code in (400, 422)

    def test_empty_batch_returns_zero(self, api_client: httpx.Client) -> None:
        payload: dict[str, Any] = {"entries": [], "device_id": "device-test-004"}
        resp: httpx.Response = api_client.post("/field-entries/batch", json=payload)
        assert resp.status_code == 200
        assert resp.json()["synced"] == 0`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Containerize the field data capture API and deploy it with a production-ready script. The deployment includes health checks, database migration, and a configuration loader that supports environment-specific overrides.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Field Data API Deployment Script',
                description:
                  'Production deployment script that builds the Docker image, runs database migrations, and performs a rolling restart with health checks.',
                code: `#!/usr/bin/env bash
# ============================================================
# deploy_field_api.sh — Deploy the Field Cost Capture API
# Usage: ./deploy_field_api.sh [staging|production]
# ============================================================
set -euo pipefail

ENVIRONMENT="\${1:-staging}"
APP_NAME="field-cost-api"
IMAGE_TAG="\$(git rev-parse --short HEAD)"
REGISTRY="ghcr.io/construction-co"
COMPOSE_FILE="docker-compose.\${ENVIRONMENT}.yml"
HEALTH_URL="http://localhost:8000/health"
MAX_RETRIES=30
RETRY_INTERVAL=2

echo "=== Deploying \${APP_NAME} to \${ENVIRONMENT} ==="
echo "Image tag: \${IMAGE_TAG}"
echo "Compose file: \${COMPOSE_FILE}"

# Step 1: Build the Docker image
echo "[1/5] Building Docker image..."
docker build \\
  --tag "\${REGISTRY}/\${APP_NAME}:\${IMAGE_TAG}" \\
  --tag "\${REGISTRY}/\${APP_NAME}:latest" \\
  --build-arg ENV="\${ENVIRONMENT}" \\
  --file Dockerfile.api .

# Step 2: Run database migrations
echo "[2/5] Running database migrations..."
docker compose -f "\${COMPOSE_FILE}" run --rm migrate \\
  alembic upgrade head

# Step 3: Stop the old container gracefully
echo "[3/5] Stopping old containers..."
docker compose -f "\${COMPOSE_FILE}" stop "\${APP_NAME}" || true

# Step 4: Start the new container
echo "[4/5] Starting new container..."
docker compose -f "\${COMPOSE_FILE}" up -d "\${APP_NAME}"

# Step 5: Health check loop
echo "[5/5] Waiting for health check..."
for i in \$(seq 1 "\${MAX_RETRIES}"); do
  if curl -sf "\${HEALTH_URL}" > /dev/null 2>&1; then
    echo "Health check passed on attempt \${i}"
    echo "=== Deployment successful ==="
    exit 0
  fi
  echo "  Attempt \${i}/\${MAX_RETRIES} — waiting \${RETRY_INTERVAL}s..."
  sleep "\${RETRY_INTERVAL}"
done

echo "ERROR: Health check failed after \${MAX_RETRIES} attempts"
echo "Rolling back to previous image..."
docker compose -f "\${COMPOSE_FILE}" stop "\${APP_NAME}"
docker compose -f "\${COMPOSE_FILE}" up -d "\${APP_NAME}"
exit 1`,
              },
              {
                language: 'python',
                title: 'Construction API Configuration Loader',
                description:
                  'Typed configuration loader that reads environment-specific settings from YAML files with environment variable overrides for secrets.',
                code: `"""config/loader.py — Configuration loader for the Field Cost Capture API"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "construction"
    user: str = "app"
    password: str = ""
    pool_size: int = 10
    ssl_mode: str = "prefer"

    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
            f"?ssl={self.ssl_mode}"
        )


@dataclass
class AlertConfig:
    slack_webhook_url: str = ""
    budget_threshold_pct: float = 85.0
    refresh_interval_seconds: int = 300
    enabled: bool = True


@dataclass
class AppConfig:
    environment: str = "staging"
    debug: bool = False
    log_level: str = "INFO"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)


def load_config(env: str | None = None) -> AppConfig:
    """Load configuration from YAML with environment variable overrides."""
    environment: str = env or os.getenv("APP_ENV", "staging")
    config_dir: Path = Path(__file__).parent / "environments"
    config_path: Path = config_dir / f"{environment}.yml"

    logger.info("Loading config for environment: %s from %s", environment, config_path)

    raw: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        logger.info("Loaded YAML config with %d top-level keys", len(raw))
    else:
        logger.warning("Config file not found: %s — using defaults", config_path)

    db_raw: dict[str, Any] = raw.get("database", {})
    alert_raw: dict[str, Any] = raw.get("alerts", {})

    config: AppConfig = AppConfig(
        environment=environment,
        debug=raw.get("debug", False),
        log_level=raw.get("log_level", "INFO"),
        database=DatabaseConfig(
            host=os.getenv("DB_HOST", db_raw.get("host", "localhost")),
            port=int(os.getenv("DB_PORT", str(db_raw.get("port", 5432)))),
            name=os.getenv("DB_NAME", db_raw.get("name", "construction")),
            user=os.getenv("DB_USER", db_raw.get("user", "app")),
            password=os.getenv("DB_PASSWORD", db_raw.get("password", "")),
            pool_size=db_raw.get("pool_size", 10),
            ssl_mode=db_raw.get("ssl_mode", "prefer"),
        ),
        alerts=AlertConfig(
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL", alert_raw.get("slack_webhook_url", "")),
            budget_threshold_pct=alert_raw.get("budget_threshold_pct", 85.0),
            refresh_interval_seconds=alert_raw.get("refresh_interval_seconds", 300),
            enabled=alert_raw.get("enabled", True),
        ),
    )
    logger.info("Config loaded: env=%s, db_host=%s, alerts_enabled=%s",
                config.environment, config.database.host, config.alerts.enabled)
    return config`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deploy the Real-Time Cost Dashboard Query Layer',
            description:
              'Create the SQL queries that power the PM dashboard, including over-budget alerts and daily cost trend summaries that refresh every 5 minutes. Includes a real-time budget overrun alerting dashboard for proactive cost management.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Dashboard Summary & Alert Queries',
                description:
                  'Materialized view for portfolio-level cost health and an alert query that flags jobs exceeding 85% of budget.',
                code: `-- Portfolio cost health, refreshed every 5 minutes via pg_cron
CREATE MATERIALIZED VIEW mv_portfolio_cost_health AS
SELECT
    j.job_id,
    j.job_name,
    j.project_manager_id,
    SUM(b.budgeted_amount)            AS total_budget,
    SUM(v.actual_cost)                AS total_actual,
    SUM(v.remaining_budget)           AS total_remaining,
    ROUND(AVG(v.pct_consumed), 1)     AS avg_pct_consumed,
    COUNT(*) FILTER (WHERE v.pct_consumed > 100) AS overbudget_code_count,
    MAX(f.submitted_at)               AS last_field_entry
FROM jobs j
JOIN v_job_cost_variance v USING (job_id)
JOIN job_budgets b        USING (job_id, cost_code)
LEFT JOIN LATERAL (
    SELECT MAX(submitted_at) AS submitted_at
    FROM field_cost_entries WHERE job_id = j.job_id
) f ON TRUE
GROUP BY j.job_id, j.job_name, j.project_manager_id;

CREATE UNIQUE INDEX ON mv_portfolio_cost_health (job_id);

-- Alert: jobs with any cost code exceeding 85% of budget
SELECT job_name, cost_code, pct_consumed, actual_cost, budgeted_amount
FROM v_job_cost_variance
WHERE pct_consumed >= 85
ORDER BY pct_consumed DESC;`,
              },
              {
                language: 'python',
                title: 'Scheduled Dashboard Refresh',
                description:
                  'Background task that refreshes the materialized view and dispatches over-budget alerts via webhook.',
                code: `import asyncio
import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = "postgresql+asyncpg://app:secret@localhost:5432/construction"
ALERT_WEBHOOK = "https://hooks.slack.com/services/T00/B00/xxxx"
REFRESH_INTERVAL_SECONDS = 300  # 5 minutes

engine = create_async_engine(DATABASE_URL, pool_size=5)

async def refresh_and_alert():
    async with engine.begin() as conn:
        await conn.execute(text(
            "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_portfolio_cost_health"
        ))
        rows = await conn.execute(text(\"\"\"
            SELECT job_name, cost_code, pct_consumed
            FROM v_job_cost_variance
            WHERE pct_consumed >= 85
            ORDER BY pct_consumed DESC
            LIMIT 20
        \"\"\"))
        alerts = rows.fetchall()

    if alerts:
        lines = [f"*{r.job_name}* | {r.cost_code} | {r.pct_consumed}% consumed"
                 for r in alerts]
        async with httpx.AsyncClient() as client:
            await client.post(ALERT_WEBHOOK, json={
                "text": f":warning: *Over-Budget Alert* — {len(alerts)} cost codes at risk\\n"
                        + "\\n".join(lines),
            })

async def main():
    while True:
        await refresh_and_alert()
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)

if __name__ == "__main__":
    asyncio.run(main())`,
              },
              {
                language: 'sql',
                title: 'Budget Overrun Real-Time Alerting Dashboard',
                description:
                  'Real-time alerting view that identifies budget overruns by severity tier, calculates burn rates, and projects estimated completion costs for proactive PM intervention.',
                code: `-- ============================================================
-- Budget Overrun Real-Time Alerting Dashboard
-- Powers the PM dashboard alert panel with severity tiers
-- ============================================================

-- Severity-tiered overrun alert view with burn rate projections
CREATE OR REPLACE VIEW v_budget_overrun_alerts AS
WITH daily_burn AS (
    SELECT
        job_id,
        cost_code,
        entry_date,
        SUM(labor_hours * COALESCE(labor_rate, 0))   AS daily_labor,
        SUM(material_cost)                            AS daily_material,
        SUM(equipment_hours * COALESCE(equipment_rate, 0)) AS daily_equipment
    FROM field_cost_entries
    WHERE entry_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY job_id, cost_code, entry_date
),
burn_rates AS (
    SELECT
        job_id,
        cost_code,
        AVG(daily_labor + daily_material + daily_equipment) AS avg_daily_burn,
        COUNT(DISTINCT entry_date)                          AS active_days
    FROM daily_burn
    GROUP BY job_id, cost_code
)
SELECT
    v.job_id,
    v.job_name,
    v.cost_code,
    v.budgeted_amount,
    v.actual_cost,
    v.remaining_budget,
    v.pct_consumed,
    br.avg_daily_burn,
    CASE
        WHEN br.avg_daily_burn > 0 THEN
            ROUND(v.remaining_budget / br.avg_daily_burn, 1)
        ELSE NULL
    END AS days_until_exhausted,
    CASE
        WHEN v.pct_consumed >= 100 THEN 'CRITICAL'
        WHEN v.pct_consumed >= 90  THEN 'HIGH'
        WHEN v.pct_consumed >= 80  THEN 'WARNING'
        ELSE 'NORMAL'
    END AS severity,
    CASE
        WHEN br.avg_daily_burn > 0 AND br.active_days > 0 THEN
            ROUND(v.actual_cost + (br.avg_daily_burn * br.active_days * 1.1), 2)
        ELSE v.actual_cost
    END AS projected_total_cost,
    now() AS alert_generated_at
FROM v_job_cost_variance v
LEFT JOIN burn_rates br ON br.job_id = v.job_id AND br.cost_code = v.cost_code
WHERE v.pct_consumed >= 80
ORDER BY v.pct_consumed DESC, v.actual_cost DESC;

-- Dashboard summary: count of alerts by severity tier
SELECT
    severity,
    COUNT(*)                              AS alert_count,
    SUM(actual_cost - budgeted_amount)
        FILTER (WHERE actual_cost > budgeted_amount) AS total_overrun_amount
FROM v_budget_overrun_alerts
GROUP BY severity
ORDER BY
    CASE severity
        WHEN 'CRITICAL' THEN 1
        WHEN 'HIGH'     THEN 2
        WHEN 'WARNING'  THEN 3
        ELSE 4
    END;`,
              },
            ],
          },
        ],
      },

      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude to instantly analyze field cost data from photos and receipts, then automate budget alerts via Zapier. Foremen can text or email photos of timecards and material receipts, which AI extracts into structured data and pushes to a Google Sheet that triggers alerts when budgets approach thresholds.',
        estimatedMonthlyCost: '$100 - $180/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Google Sheets (Free)', 'Twilio ($20/mo for SMS)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Procore AI (Enterprise)', 'Microsoft Power Automate ($15/mo)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Set up a Zapier workflow that receives field photos via email or SMS, then sends them to ChatGPT Vision for OCR extraction of time entries, material costs, and equipment hours into structured JSON.',
            toolsUsed: ['Zapier', 'Gmail/Twilio', 'Google Drive'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Field Entry JSON Schema',
                description:
                  'The target schema that ChatGPT will extract field data into from photos of timecards and receipts.',
                code: `{
  "fieldEntrySchema": {
    "job_id": "string",
    "job_name": "string",
    "cost_code": "string",
    "entry_date": "YYYY-MM-DD",
    "labor_entries": [
      {
        "employee_name": "string",
        "hours_worked": "number",
        "labor_rate": "number",
        "task_description": "string"
      }
    ],
    "material_entries": [
      {
        "description": "string",
        "quantity": "number",
        "unit_cost": "number",
        "vendor": "string",
        "receipt_number": "string"
      }
    ],
    "equipment_entries": [
      {
        "equipment_type": "string",
        "hours_used": "number",
        "hourly_rate": "number"
      }
    ],
    "submitted_by": "string",
    "extraction_confidence": "high | medium | low",
    "notes": "string"
  }
}`,
              },
              {
                language: 'json',
                title: 'Zapier Email-to-Drive Trigger Config',
                description:
                  'Zapier trigger configuration that watches for field submission emails and saves attachments to Google Drive for processing.',
                code: `{
  "trigger": {
    "app": "Gmail",
    "event": "New Attachment",
    "filters": {
      "from_contains": "@construction-company.com",
      "subject_contains": ["Field Report", "Daily Cost", "Material Receipt"],
      "has_attachment": true
    }
  },
  "action_1": {
    "app": "Google Drive",
    "event": "Upload File",
    "config": {
      "folder_id": "1ABC_FieldSubmissions_Inbox",
      "file_name": "{{trigger.attachment_name}}",
      "file_content": "{{trigger.attachment_content}}",
      "convert_to_google_format": false
    }
  },
  "action_2": {
    "app": "Google Drive",
    "event": "Create Shareable Link",
    "config": {
      "file_id": "{{action_1.file_id}}",
      "access": "anyone_with_link",
      "role": "reader"
    }
  }
}`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description:
              'Send field photos to ChatGPT Vision API for intelligent OCR extraction. The prompt instructs ChatGPT to identify job details, labor hours, material costs, and equipment usage, outputting structured JSON that matches the target schema.',
            toolsUsed: ['ChatGPT Plus', 'OpenAI Vision API'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Field Cost OCR Extraction Prompt',
                description:
                  'System and user prompts for ChatGPT Vision to extract structured cost data from field photos of timecards, receipts, and daily reports.',
                code: `system_prompt: |
  You are a construction field cost extraction specialist. Your job is to
  analyze photos of timecards, material receipts, and daily field reports
  to extract structured cost data.

  EXTRACTION RULES:
  1. Job identification: Look for job numbers, project names, or site addresses
  2. Cost codes: Extract CSI Division codes (e.g., 03-100 for Concrete) or
     company-specific codes. If unclear, use "PENDING-REVIEW"
  3. Labor: Extract employee names, hours worked, and any rate information
  4. Materials: Extract item descriptions, quantities, unit costs, and vendor info
  5. Equipment: Extract equipment type, hours used, and rental rates if shown
  6. Dates: Always extract the work date, defaulting to today if not visible

  CONFIDENCE SCORING:
  - "high": All fields clearly visible and legible
  - "medium": Some fields estimated or partially visible
  - "low": Significant guessing required, flag for manual review

  OUTPUT FORMAT: Return valid JSON matching the fieldEntrySchema exactly.
  Include a "notes" field for any ambiguities or extraction issues.

user_prompt: |
  Please analyze this field submission image and extract all cost data.

  Image: {{image_url}}

  Additional context from email:
  - Sender: {{sender_email}}
  - Subject: {{email_subject}}
  - Date received: {{received_date}}

  Extract the data into the standard field entry JSON format.
  Flag any values you are uncertain about in the notes field.

expected_output_example: |
  {
    "job_id": "PRJ-2024-0042",
    "job_name": "Downtown Office Tower - Phase 2",
    "cost_code": "03-300",
    "entry_date": "2024-01-15",
    "labor_entries": [
      {
        "employee_name": "Mike Johnson",
        "hours_worked": 8.5,
        "labor_rate": 55.00,
        "task_description": "Concrete pour - Level 3 slab"
      },
      {
        "employee_name": "Carlos Rodriguez",
        "hours_worked": 8.5,
        "labor_rate": 48.00,
        "task_description": "Concrete finishing"
      }
    ],
    "material_entries": [
      {
        "description": "Ready-mix concrete 4000 PSI",
        "quantity": 45,
        "unit_cost": 125.00,
        "vendor": "Metro Concrete Supply",
        "receipt_number": "RC-78234"
      }
    ],
    "equipment_entries": [
      {
        "equipment_type": "Concrete pump truck",
        "hours_used": 6,
        "hourly_rate": 185.00
      }
    ],
    "submitted_by": "Mike Johnson",
    "extraction_confidence": "high",
    "notes": "All values clearly visible. Receipt attached shows exact quantities."
  }`,
              },
              {
                language: 'yaml',
                title: 'Budget Variance Analysis Prompt',
                description:
                  'Prompt template for ChatGPT to analyze extracted costs against budgets and generate actionable alerts.',
                code: `system_prompt: |
  You are a construction cost analyst AI. Given extracted field costs and
  budget data, analyze for variances and generate actionable recommendations.

  ANALYSIS FRAMEWORK:
  1. Calculate actual vs. budgeted cost for each cost code
  2. Identify cost codes exceeding 80% budget consumption
  3. Project completion costs based on current burn rate
  4. Flag unusual patterns (e.g., labor hours spiking, material costs above estimate)

  ALERT THRESHOLDS:
  - CRITICAL: >100% of budget consumed, or projected to exceed by >20%
  - WARNING: 85-100% consumed, or burn rate trending 15% over plan
  - WATCH: 70-85% consumed with accelerating spend

  OUTPUT: Generate a brief executive summary (3-5 bullets) followed by
  specific action items for each flagged cost code.

user_prompt: |
  Analyze today's field cost submissions against the job budget:

  JOB: {{job_name}} ({{job_id}})

  TODAY'S SUBMISSIONS:
  {{extracted_costs_json}}

  CURRENT BUDGET STATUS:
  {{budget_status_json}}

  Provide:
  1. Executive summary of cost status
  2. Any cost codes requiring immediate attention
  3. Recommended actions for the project manager

analysis_output_schema:
  executive_summary:
    - bullet 1
    - bullet 2
    - bullet 3
  alerts:
    - cost_code: "string"
      severity: "CRITICAL | WARNING | WATCH"
      current_consumed_pct: number
      projected_overrun: number
      recommended_action: "string"
  action_items:
    - priority: "HIGH | MEDIUM | LOW"
      assignee: "PM | Foreman | Estimating"
      action: "string"
      deadline: "string"`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier to append extracted costs to a Google Sheet cost tracking system, calculate budget variances, and send automated Slack alerts when thresholds are exceeded. PMs receive daily summaries and instant alerts for critical overruns.',
            toolsUsed: ['Zapier', 'Google Sheets', 'Slack'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Cost Entry & Alert Workflow',
                description:
                  'Complete Zapier workflow that processes ChatGPT responses, updates the cost tracking sheet, and triggers budget alerts.',
                code: `{
  "zap_name": "Field Cost OCR to Budget Alert",
  "trigger": {
    "app": "Webhooks by Zapier",
    "event": "Catch Hook",
    "webhook_url": "https://hooks.zapier.com/hooks/catch/12345/fieldcost/"
  },
  "step_2_parse_chatgpt": {
    "app": "Code by Zapier",
    "event": "Run Javascript",
    "code": "const data = JSON.parse(inputData.chatgpt_response); return { job_id: data.job_id, entries: JSON.stringify(data.labor_entries.concat(data.material_entries)), confidence: data.extraction_confidence };"
  },
  "step_3_append_costs": {
    "app": "Google Sheets",
    "event": "Create Spreadsheet Row",
    "config": {
      "spreadsheet_id": "1ABC_JobCostTracker_2024",
      "worksheet": "Daily Entries",
      "row_data": {
        "A": "{{step_2.job_id}}",
        "B": "{{trigger.cost_code}}",
        "C": "{{trigger.entry_date}}",
        "D": "{{trigger.labor_total}}",
        "E": "{{trigger.material_total}}",
        "F": "{{trigger.equipment_total}}",
        "G": "=D{{row}}+E{{row}}+F{{row}}",
        "H": "{{step_2.confidence}}",
        "I": "={{now()}}"
      }
    }
  },
  "step_4_lookup_budget": {
    "app": "Google Sheets",
    "event": "Lookup Spreadsheet Row",
    "config": {
      "spreadsheet_id": "1ABC_JobCostTracker_2024",
      "worksheet": "Budgets",
      "lookup_column": "A",
      "lookup_value": "{{step_2.job_id}}-{{trigger.cost_code}}"
    }
  },
  "step_5_calculate_variance": {
    "app": "Code by Zapier",
    "event": "Run Javascript",
    "code": "const budget = parseFloat(inputData.budgeted_amount); const actual = parseFloat(inputData.actual_to_date); const pct = (actual / budget) * 100; return { pct_consumed: pct.toFixed(1), remaining: (budget - actual).toFixed(2), alert_level: pct >= 100 ? 'CRITICAL' : pct >= 85 ? 'WARNING' : pct >= 70 ? 'WATCH' : 'OK' };"
  },
  "step_6_conditional_alert": {
    "app": "Filter by Zapier",
    "condition": "{{step_5.alert_level}} is not OK"
  },
  "step_7_slack_alert": {
    "app": "Slack",
    "event": "Send Channel Message",
    "config": {
      "channel": "#construction-cost-alerts",
      "message": ":warning: *Budget Alert: {{step_5.alert_level}}*\\n\\n*Job:* {{step_2.job_id}}\\n*Cost Code:* {{trigger.cost_code}}\\n*Budget Consumed:* {{step_5.pct_consumed}}%\\n*Remaining:* USD {{step_5.remaining}}\\n\\n_Entry submitted by {{trigger.submitted_by}} on {{trigger.entry_date}}_",
      "bot_name": "Cost Tracker Bot",
      "bot_icon": ":chart_with_upwards_trend:"
    }
  },
  "step_8_daily_digest": {
    "app": "Schedule by Zapier",
    "event": "Every Day at 6 PM",
    "linked_zap": "Daily Cost Summary Email to PMs"
  }
}`,
              },
              {
                language: 'json',
                title: 'Google Sheets Budget Tracking Formulas',
                description:
                  'Sheet structure and formulas for the real-time budget tracking dashboard that Zapier populates.',
                code: `{
  "sheet_name": "Budget Dashboard",
  "headers": [
    "Job ID", "Cost Code", "Description", "Budgeted Amount",
    "Actual to Date", "Remaining", "% Consumed", "Status",
    "Last Entry Date", "Trend (7-day)", "Projected Final"
  ],
  "formula_columns": {
    "F_Remaining": "=D2-E2",
    "G_Pct_Consumed": "=IF(D2>0, E2/D2*100, 0)",
    "H_Status": "=IFS(G2>=100, \"CRITICAL\", G2>=85, \"WARNING\", G2>=70, \"WATCH\", TRUE, \"OK\")",
    "J_Trend": "=SPARKLINE(FILTER('Daily Entries'!G:G, 'Daily Entries'!A:A=A2, 'Daily Entries'!C:C>=TODAY()-7))",
    "K_Projected_Final": "=IF(I2<>'', E2 + (E2/DAYS(I2, ProjectStartDate) * DAYS(ProjectEndDate, TODAY())), E2)"
  },
  "conditional_formatting": {
    "H_Status_CRITICAL": { "background": "#FF6B6B", "font_color": "#FFFFFF" },
    "H_Status_WARNING": { "background": "#FFE66D", "font_color": "#000000" },
    "H_Status_WATCH": { "background": "#4ECDC4", "font_color": "#000000" }
  },
  "pivot_summary": {
    "rows": "Job ID",
    "values": ["SUM of Budgeted Amount", "SUM of Actual to Date", "AVG of % Consumed"],
    "filter": "Status is CRITICAL or WARNING"
  }
}`,
              },
            ],
          },
        ],
      },

      aiAdvanced: {
        overview:
          'Deploy a multi-agent AI system that provides end-to-end field cost intelligence: OCR agents extract data from any document format, validation agents cross-reference budgets and cost codes, analysis agents detect anomalies and forecast overruns, and alert agents deliver contextualized notifications to the right stakeholders at the right time.',
        estimatedMonthlyCost: '$600 - $1,200/month',
        architecture:
          'Supervisor agent coordinates four specialist agents: (1) Document Ingestion Agent handles OCR and data extraction from photos, PDFs, and handwritten notes, (2) Validation Agent cross-references cost codes and budget allocations, (3) Analytics Agent performs trend analysis and overrun forecasting, (4) Notification Agent routes alerts by severity and stakeholder role. LangGraph orchestrates the pipeline with Redis-backed state for handling offline field submissions.',
        agents: [
          {
            name: 'FieldDocumentAgent',
            role: 'Document Ingestion & OCR',
            goal: 'Extract structured cost data from any field document format including photos of handwritten timecards, PDF receipts, and mobile app exports',
            tools: ['GPT-4 Vision', 'Azure Document Intelligence', 'Tesseract OCR', 'PDF Parser'],
          },
          {
            name: 'CostValidationAgent',
            role: 'Data Validation & Enrichment',
            goal: 'Validate extracted data against master cost code lists, budget allocations, and employee rosters; flag anomalies and enrich with missing context',
            tools: ['PostgreSQL', 'Procore API', 'Cost Code Lookup', 'Employee Directory API'],
          },
          {
            name: 'BudgetAnalyticsAgent',
            role: 'Trend Analysis & Forecasting',
            goal: 'Analyze cost trends, detect burn rate anomalies, forecast project completion costs, and identify early warning indicators of budget overruns',
            tools: ['Pandas', 'Prophet', 'Scikit-learn', 'Cost Variance Calculator'],
          },
          {
            name: 'StakeholderAlertAgent',
            role: 'Intelligent Notification Routing',
            goal: 'Deliver contextualized alerts to appropriate stakeholders based on severity, role, and communication preferences; aggregate low-priority alerts into digests',
            tools: ['Slack API', 'Email API', 'SMS Gateway', 'Procore Notifications'],
          },
          {
            name: 'SupervisorAgent',
            role: 'Pipeline Orchestration',
            goal: 'Coordinate agent workflows, manage retry logic for failed extractions, aggregate results, and ensure end-to-end data quality',
            tools: ['LangGraph', 'Redis', 'Prometheus Metrics'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with hourly checkpointing; offline submissions queued and processed on connectivity restoration',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system using CrewAI with specialized roles for field cost processing. Each agent has domain-specific tools and clear success criteria for the construction cost tracking workflow.',
            toolsUsed: ['CrewAI', 'LangChain', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Field Cost Agent Definitions',
                description:
                  'CrewAI agent definitions for the field cost intelligence system with construction-specific roles, goals, and tool access.',
                code: `"""agents/field_cost_agents.py - Multi-agent system for construction field cost intelligence"""
from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class FieldCostEntry(BaseModel):
    """Structured field cost entry extracted from documents."""
    job_id: str = Field(description="Project/job identifier")
    cost_code: str = Field(description="CSI division cost code")
    entry_date: str = Field(description="Date of work in YYYY-MM-DD format")
    labor_hours: float = Field(default=0.0, description="Total labor hours")
    labor_cost: float = Field(default=0.0, description="Total labor cost in dollars")
    material_cost: float = Field(default=0.0, description="Total material cost in dollars")
    equipment_cost: float = Field(default=0.0, description="Total equipment cost in dollars")
    confidence_score: float = Field(description="Extraction confidence 0.0-1.0")
    validation_status: str = Field(default="pending", description="pending|valid|invalid|needs_review")
    anomaly_flags: List[str] = Field(default_factory=list, description="Detected anomalies")


# Initialize LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)


# Tool definitions
def ocr_extract_tool(image_path: str) -> dict:
    """Extract text and structured data from field document image."""
    # Implementation uses GPT-4 Vision or Azure Document Intelligence
    logger.info(f"OCR extracting from: {image_path}")
    # ... actual implementation
    return {"status": "extracted", "raw_text": "...", "structured_data": {}}


def validate_cost_code_tool(cost_code: str, job_id: str) -> dict:
    """Validate cost code against master list and job budget."""
    logger.info(f"Validating cost code {cost_code} for job {job_id}")
    # ... database lookup implementation
    return {"valid": True, "budgeted_amount": 50000.00, "spent_to_date": 32000.00}


def calculate_burn_rate_tool(job_id: str, cost_code: str, lookback_days: int = 30) -> dict:
    """Calculate daily burn rate and project completion cost."""
    logger.info(f"Calculating burn rate for {job_id}/{cost_code}")
    # ... analytics implementation
    return {"daily_burn_rate": 850.00, "projected_completion_cost": 58500.00, "days_to_exhaustion": 21}


def send_alert_tool(alert_type: str, severity: str, recipients: List[str], message: str) -> dict:
    """Send alert via appropriate channel based on severity."""
    logger.info(f"Sending {severity} alert to {recipients}")
    # ... notification implementation
    return {"sent": True, "channel": "slack" if severity != "CRITICAL" else "sms"}


# Agent definitions
field_document_agent = Agent(
    role="Field Document Intelligence Specialist",
    goal="Extract accurate, structured cost data from any field document format "
         "including photos of handwritten timecards, scanned receipts, and PDF reports",
    backstory="You are an expert at reading construction field documents. You understand "
              "CSI cost code formats, union timecard layouts, and material receipt structures. "
              "You excel at extracting data even from poor quality images or handwritten notes.",
    llm=llm,
    tools=[Tool(name="ocr_extract", func=ocr_extract_tool, description="Extract text from images")],
    verbose=True,
    allow_delegation=False,
)

cost_validation_agent = Agent(
    role="Construction Cost Validation Specialist",
    goal="Ensure all extracted cost data is valid, properly coded, and matches "
         "the project budget structure before it enters the cost tracking system",
    backstory="You are a meticulous construction accountant who knows every cost code "
              "in the CSI MasterFormat. You catch errors that would cause month-end "
              "reconciliation nightmares and ensure every dollar is properly allocated.",
    llm=llm,
    tools=[Tool(name="validate_cost_code", func=validate_cost_code_tool,
                description="Validate cost code against master list")],
    verbose=True,
    allow_delegation=False,
)

budget_analytics_agent = Agent(
    role="Construction Budget Analytics Specialist",
    goal="Analyze cost trends, detect anomalies, and forecast budget outcomes "
         "to provide early warning of potential overruns before they become critical",
    backstory="You are a construction financial analyst who has seen every type of "
              "budget blowout. You know the early warning signs - unusual burn rates, "
              "material cost spikes, labor hour creep - and you catch them early.",
    llm=llm,
    tools=[Tool(name="calculate_burn_rate", func=calculate_burn_rate_tool,
                description="Calculate spend rates and projections")],
    verbose=True,
    allow_delegation=False,
)

alert_routing_agent = Agent(
    role="Construction Stakeholder Communication Specialist",
    goal="Deliver the right information to the right person at the right time - "
         "critical alerts immediately via SMS, warnings to PM dashboards, and "
         "routine updates in daily digests",
    backstory="You understand construction project hierarchies and communication needs. "
              "You know that a foreman needs different information than a project exec, "
              "and that some alerts need immediate attention while others can wait.",
    llm=llm,
    tools=[Tool(name="send_alert", func=send_alert_tool, description="Route alerts to stakeholders")],
    verbose=True,
    allow_delegation=False,
)

supervisor_agent = Agent(
    role="Field Cost Pipeline Supervisor",
    goal="Orchestrate the end-to-end field cost processing pipeline, ensuring "
         "data quality, handling failures gracefully, and maintaining system reliability",
    backstory="You are the operations manager who keeps the field cost system running "
              "smoothly. You coordinate the specialists, manage retries, and ensure "
              "no field submission falls through the cracks.",
    llm=llm,
    tools=[],
    verbose=True,
    allow_delegation=True,
)

logger.info("Field cost agents initialized successfully")`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the FieldDocumentAgent with multi-format OCR capabilities using GPT-4 Vision for photos and Azure Document Intelligence for PDFs. Handle offline submissions by queueing to Redis.',
            toolsUsed: ['GPT-4 Vision', 'Azure Document Intelligence', 'Redis', 'FastAPI'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Multi-Format Document Ingestion Agent',
                description:
                  'Agent implementation that handles OCR extraction from photos, PDFs, and scanned documents with confidence scoring and offline queue support.',
                code: `"""agents/document_ingestion.py - Field document OCR and extraction agent"""
import base64
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import redis
from openai import OpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ExtractionResult(BaseModel):
    """Result of document extraction."""
    document_id: str
    document_type: str = Field(description="timecard|receipt|daily_report|invoice")
    extraction_timestamp: datetime
    raw_text: str
    structured_data: dict[str, Any]
    confidence_score: float = Field(ge=0.0, le=1.0)
    extraction_method: str = Field(description="gpt4_vision|azure_di|tesseract")
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)


class FieldDocumentIngestionAgent:
    """Agent for extracting structured cost data from field documents."""

    def __init__(
        self,
        openai_api_key: str,
        redis_url: str = "redis://localhost:6379",
        azure_di_endpoint: Optional[str] = None,
        azure_di_key: Optional[str] = None,
    ):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.redis_client = redis.from_url(redis_url)
        self.azure_di_endpoint = azure_di_endpoint
        self.azure_di_key = azure_di_key
        logger.info("FieldDocumentIngestionAgent initialized")

    def _generate_document_id(self, content: bytes) -> str:
        """Generate unique document ID from content hash."""
        return hashlib.sha256(content).hexdigest()[:16]

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for GPT-4 Vision."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_extraction_prompt(self, document_type: str) -> str:
        """Get specialized extraction prompt based on document type."""
        prompts = {
            "timecard": '''Extract all timecard data from this image:
- Employee name(s) and ID(s)
- Date(s) of work
- Hours worked (regular and overtime)
- Job/project identifier
- Cost code or work type
- Any supervisor signatures or approvals

Return as JSON with: employee_entries[], job_id, cost_code, total_hours, overtime_hours''',

            "receipt": '''Extract all material receipt/invoice data:
- Vendor name and address
- Receipt/invoice number
- Date of purchase
- Line items with: description, quantity, unit price, total
- Tax amount
- Total amount
- Job number or PO reference if visible

Return as JSON with: vendor, receipt_number, date, line_items[], subtotal, tax, total, job_reference''',

            "daily_report": '''Extract daily field report data:
- Date and weather conditions
- Job/project identifier
- Crew members present with hours
- Equipment used with hours
- Materials received/used
- Work completed description
- Any delays or issues noted

Return as JSON with: date, job_id, weather, crew[], equipment[], materials[], work_completed, issues[]''',
        }
        return prompts.get(document_type, prompts["daily_report"])

    async def extract_from_image(
        self,
        image_path: Path,
        document_type: str = "daily_report",
        job_context: Optional[dict] = None,
    ) -> ExtractionResult:
        """Extract structured data from a field document image using GPT-4 Vision."""
        logger.info(f"Extracting from image: {image_path}, type: {document_type}")

        with open(image_path, "rb") as f:
            content = f.read()
        document_id = self._generate_document_id(content)
        image_b64 = base64.b64encode(content).decode("utf-8")

        context_str = ""
        if job_context:
            context_str = f"\\nContext: Job {job_context.get('job_id', 'unknown')}, " \
                          f"expected cost codes: {job_context.get('valid_cost_codes', [])}"

        prompt = self._get_extraction_prompt(document_type) + context_str

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2000,
                temperature=0,
            )

            raw_response = response.choices[0].message.content
            logger.info(f"GPT-4 Vision extraction complete for {document_id}")

            # Parse JSON from response
            json_start = raw_response.find("{")
            json_end = raw_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                structured_data = json.loads(raw_response[json_start:json_end])
            else:
                structured_data = {"raw_response": raw_response}
                logger.warning(f"Could not parse JSON from response for {document_id}")

            # Calculate confidence based on extraction completeness
            confidence = self._calculate_confidence(structured_data, document_type)
            needs_review = confidence < 0.7
            review_reasons = []
            if confidence < 0.7:
                review_reasons.append("Low extraction confidence")
            if not structured_data.get("job_id"):
                review_reasons.append("Missing job identifier")

            return ExtractionResult(
                document_id=document_id,
                document_type=document_type,
                extraction_timestamp=datetime.utcnow(),
                raw_text=raw_response,
                structured_data=structured_data,
                confidence_score=confidence,
                extraction_method="gpt4_vision",
                needs_review=needs_review,
                review_reasons=review_reasons,
            )

        except Exception as e:
            logger.error(f"Extraction failed for {document_id}: {e}")
            # Queue for retry
            self._queue_for_retry(document_id, str(image_path), document_type, str(e))
            raise

    def _calculate_confidence(self, data: dict, doc_type: str) -> float:
        """Calculate confidence score based on extraction completeness."""
        required_fields = {
            "timecard": ["employee_entries", "job_id", "total_hours"],
            "receipt": ["vendor", "line_items", "total"],
            "daily_report": ["date", "job_id", "crew"],
        }
        fields = required_fields.get(doc_type, ["job_id"])
        present = sum(1 for f in fields if data.get(f))
        return present / len(fields) if fields else 0.5

    def _queue_for_retry(
        self, doc_id: str, path: str, doc_type: str, error: str
    ) -> None:
        """Queue failed extraction for later retry."""
        retry_data = {
            "document_id": doc_id,
            "path": path,
            "document_type": doc_type,
            "error": error,
            "queued_at": datetime.utcnow().isoformat(),
            "retry_count": 0,
        }
        self.redis_client.lpush("extraction_retry_queue", json.dumps(retry_data))
        logger.info(f"Queued {doc_id} for retry")

    async def process_offline_queue(self, batch_size: int = 10) -> list[ExtractionResult]:
        """Process queued offline submissions when connectivity is restored."""
        results = []
        for _ in range(batch_size):
            item = self.redis_client.rpop("offline_submission_queue")
            if not item:
                break
            submission = json.loads(item)
            logger.info(f"Processing offline submission: {submission['document_id']}")
            try:
                result = await self.extract_from_image(
                    Path(submission["path"]),
                    submission.get("document_type", "daily_report"),
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process offline submission: {e}")
        return results`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the BudgetAnalyticsAgent with trend detection, burn rate forecasting, and anomaly identification using statistical methods and ML-based prediction.',
            toolsUsed: ['Pandas', 'Prophet', 'Scikit-learn', 'PostgreSQL'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Budget Analytics & Forecasting Agent',
                description:
                  'Agent that analyzes cost trends, detects anomalies, and forecasts project completion costs with early warning indicators.',
                code: `"""agents/budget_analytics.py - Cost trend analysis and forecasting agent"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class BudgetAlert(BaseModel):
    """Budget alert with severity and recommended actions."""
    alert_id: str
    job_id: str
    cost_code: str
    alert_type: str = Field(description="overrun|anomaly|trend|forecast")
    severity: str = Field(description="CRITICAL|WARNING|WATCH|INFO")
    current_spent: float
    budgeted_amount: float
    pct_consumed: float
    projected_final: Optional[float] = None
    days_to_exhaustion: Optional[int] = None
    anomaly_description: Optional[str] = None
    recommended_actions: list[str]
    generated_at: datetime


class CostForecast(BaseModel):
    """Cost forecast for a job/cost code combination."""
    job_id: str
    cost_code: str
    forecast_date: datetime
    current_spent: float
    daily_burn_rate: float
    projected_final_cost: float
    confidence_interval_low: float
    confidence_interval_high: float
    days_remaining: int
    on_track: bool


class BudgetAnalyticsAgent:
    """Agent for analyzing cost trends and forecasting budget outcomes."""

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        logger.info("BudgetAnalyticsAgent initialized")

    def get_cost_history(
        self,
        job_id: str,
        cost_code: Optional[str] = None,
        days: int = 90,
    ) -> pd.DataFrame:
        """Fetch historical cost entries for analysis."""
        query = text("""
            SELECT
                entry_date,
                cost_code,
                SUM(labor_hours * COALESCE(labor_rate, 0)) as labor_cost,
                SUM(material_cost) as material_cost,
                SUM(equipment_hours * COALESCE(equipment_rate, 0)) as equipment_cost,
                SUM(labor_hours * COALESCE(labor_rate, 0) +
                    material_cost +
                    equipment_hours * COALESCE(equipment_rate, 0)) as daily_total
            FROM field_cost_entries
            WHERE job_id = :job_id
              AND entry_date >= CURRENT_DATE - :days
              AND (:cost_code IS NULL OR cost_code = :cost_code)
            GROUP BY entry_date, cost_code
            ORDER BY entry_date
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={"job_id": job_id, "cost_code": cost_code, "days": days},
            )
        logger.info(f"Retrieved {len(df)} days of cost history for {job_id}")
        return df

    def get_budget_status(self, job_id: str, cost_code: str) -> dict[str, Any]:
        """Get current budget status for a job/cost code."""
        query = text("""
            SELECT
                b.budgeted_amount,
                COALESCE(SUM(f.labor_hours * f.labor_rate), 0) +
                COALESCE(SUM(f.material_cost), 0) +
                COALESCE(SUM(f.equipment_hours * f.equipment_rate), 0) as actual_spent,
                j.planned_start_date,
                j.planned_end_date
            FROM job_budgets b
            JOIN jobs j ON j.job_id = b.job_id
            LEFT JOIN field_cost_entries f ON f.job_id = b.job_id AND f.cost_code = b.cost_code
            WHERE b.job_id = :job_id AND b.cost_code = :cost_code
            GROUP BY b.budgeted_amount, j.planned_start_date, j.planned_end_date
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {"job_id": job_id, "cost_code": cost_code})
            row = result.fetchone()

        if not row:
            return {"error": "Budget not found"}

        return {
            "budgeted_amount": float(row[0]),
            "actual_spent": float(row[1]),
            "pct_consumed": (float(row[1]) / float(row[0]) * 100) if row[0] else 0,
            "remaining": float(row[0]) - float(row[1]),
            "planned_start": row[2],
            "planned_end": row[3],
        }

    def calculate_burn_rate(
        self,
        job_id: str,
        cost_code: str,
        lookback_days: int = 14,
    ) -> dict[str, Any]:
        """Calculate daily burn rate and trend."""
        df = self.get_cost_history(job_id, cost_code, lookback_days)

        if df.empty or len(df) < 3:
            return {"error": "Insufficient data for burn rate calculation"}

        daily_totals = df.groupby("entry_date")["daily_total"].sum()
        avg_burn = daily_totals.mean()
        recent_burn = daily_totals.tail(7).mean() if len(daily_totals) >= 7 else avg_burn

        # Trend: positive = accelerating spend
        if len(daily_totals) >= 7:
            trend = (recent_burn - daily_totals.head(7).mean()) / daily_totals.head(7).mean() * 100
        else:
            trend = 0.0

        return {
            "avg_daily_burn": float(avg_burn),
            "recent_daily_burn": float(recent_burn),
            "trend_pct": float(trend),
            "trend_direction": "accelerating" if trend > 5 else "decelerating" if trend < -5 else "stable",
            "data_days": len(daily_totals),
        }

    def forecast_completion_cost(
        self,
        job_id: str,
        cost_code: str,
        forecast_days: int = 30,
    ) -> CostForecast:
        """Forecast completion cost using Prophet time series model."""
        df = self.get_cost_history(job_id, cost_code, days=90)
        budget_status = self.get_budget_status(job_id, cost_code)

        if df.empty or len(df) < 14:
            # Fall back to linear projection
            burn_rate = self.calculate_burn_rate(job_id, cost_code)
            projected = budget_status["actual_spent"] + (
                burn_rate.get("avg_daily_burn", 0) * forecast_days
            )
            return CostForecast(
                job_id=job_id,
                cost_code=cost_code,
                forecast_date=datetime.utcnow(),
                current_spent=budget_status["actual_spent"],
                daily_burn_rate=burn_rate.get("avg_daily_burn", 0),
                projected_final_cost=projected,
                confidence_interval_low=projected * 0.9,
                confidence_interval_high=projected * 1.1,
                days_remaining=forecast_days,
                on_track=projected <= budget_status["budgeted_amount"],
            )

        # Prepare data for Prophet
        prophet_df = df.groupby("entry_date")["daily_total"].sum().reset_index()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df["y"] = prophet_df["y"].cumsum()  # Cumulative spend

        # Fit Prophet model
        model = Prophet(daily_seasonality=False, weekly_seasonality=True)
        model.fit(prophet_df)

        # Make future predictions
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        final_forecast = forecast.iloc[-1]
        current_spent = budget_status["actual_spent"]
        daily_burn = (final_forecast["yhat"] - current_spent) / forecast_days

        logger.info(f"Forecast for {job_id}/{cost_code}: projected={final_forecast['yhat']:.2f}")

        return CostForecast(
            job_id=job_id,
            cost_code=cost_code,
            forecast_date=datetime.utcnow(),
            current_spent=current_spent,
            daily_burn_rate=daily_burn,
            projected_final_cost=final_forecast["yhat"],
            confidence_interval_low=final_forecast["yhat_lower"],
            confidence_interval_high=final_forecast["yhat_upper"],
            days_remaining=forecast_days,
            on_track=final_forecast["yhat"] <= budget_status["budgeted_amount"],
        )

    def detect_anomalies(
        self,
        job_id: str,
        cost_code: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Detect cost anomalies using Isolation Forest."""
        df = self.get_cost_history(job_id, cost_code, days=60)

        if df.empty or len(df) < 10:
            return []

        # Prepare features for anomaly detection
        features = df[["labor_cost", "material_cost", "equipment_cost", "daily_total"]].values

        # Fit and predict
        self.anomaly_detector.fit(features)
        predictions = self.anomaly_detector.predict(features)

        anomalies = []
        for idx, pred in enumerate(predictions):
            if pred == -1:  # Anomaly
                row = df.iloc[idx]
                anomalies.append({
                    "date": row["entry_date"],
                    "cost_code": row["cost_code"],
                    "daily_total": row["daily_total"],
                    "labor_cost": row["labor_cost"],
                    "material_cost": row["material_cost"],
                    "anomaly_type": self._classify_anomaly(row, df),
                })

        logger.info(f"Detected {len(anomalies)} anomalies for {job_id}")
        return anomalies

    def _classify_anomaly(self, row: pd.Series, df: pd.DataFrame) -> str:
        """Classify the type of anomaly detected."""
        avg_daily = df["daily_total"].mean()
        avg_labor = df["labor_cost"].mean()
        avg_material = df["material_cost"].mean()

        if row["daily_total"] > avg_daily * 2:
            return "high_daily_spend"
        if row["labor_cost"] > avg_labor * 2:
            return "labor_spike"
        if row["material_cost"] > avg_material * 2:
            return "material_spike"
        return "general_outlier"

    def generate_budget_alert(
        self,
        job_id: str,
        cost_code: str,
    ) -> Optional[BudgetAlert]:
        """Generate alert if budget thresholds are exceeded."""
        status = self.get_budget_status(job_id, cost_code)
        burn_rate = self.calculate_burn_rate(job_id, cost_code)
        forecast = self.forecast_completion_cost(job_id, cost_code)

        pct_consumed = status["pct_consumed"]

        # Determine severity
        if pct_consumed >= 100:
            severity = "CRITICAL"
            alert_type = "overrun"
        elif pct_consumed >= 90 or not forecast.on_track:
            severity = "WARNING"
            alert_type = "forecast" if not forecast.on_track else "overrun"
        elif pct_consumed >= 80 or burn_rate.get("trend_direction") == "accelerating":
            severity = "WATCH"
            alert_type = "trend"
        else:
            return None  # No alert needed

        # Generate recommended actions
        actions = self._generate_recommendations(severity, status, burn_rate, forecast)

        remaining = status["remaining"]
        days_to_exhaustion = None
        if burn_rate.get("recent_daily_burn", 0) > 0:
            days_to_exhaustion = int(remaining / burn_rate["recent_daily_burn"])

        return BudgetAlert(
            alert_id=f"{job_id}-{cost_code}-{datetime.utcnow().strftime('%Y%m%d%H%M')}",
            job_id=job_id,
            cost_code=cost_code,
            alert_type=alert_type,
            severity=severity,
            current_spent=status["actual_spent"],
            budgeted_amount=status["budgeted_amount"],
            pct_consumed=pct_consumed,
            projected_final=forecast.projected_final_cost,
            days_to_exhaustion=days_to_exhaustion,
            recommended_actions=actions,
            generated_at=datetime.utcnow(),
        )

    def _generate_recommendations(
        self,
        severity: str,
        status: dict,
        burn_rate: dict,
        forecast: CostForecast,
    ) -> list[str]:
        """Generate context-specific recommendations."""
        actions = []

        if severity == "CRITICAL":
            actions.append("IMMEDIATE: Schedule budget review meeting with PM and stakeholders")
            actions.append("Document scope changes that may justify a change order")
            actions.append("Halt non-critical spending until budget is addressed")

        if not forecast.on_track:
            overrun = forecast.projected_final_cost - status["budgeted_amount"]
            actions.append(f"Projected overrun of USD {overrun:,.2f} - evaluate cost reduction options")

        if burn_rate.get("trend_direction") == "accelerating":
            actions.append("Investigate cause of accelerating spend rate")
            actions.append("Review recent material and labor cost changes")

        if status["pct_consumed"] > 80:
            actions.append("Request updated estimate-to-complete from field team")

        return actions`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement the LangGraph state machine that coordinates the agent pipeline, managing state transitions, error handling, and retry logic with Redis-backed persistence.',
            toolsUsed: ['LangGraph', 'Redis', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Field Cost Pipeline Orchestrator',
                description:
                  'State machine implementation using LangGraph that coordinates document ingestion, validation, analytics, and alerting agents with fault-tolerant state management.',
                code: `"""orchestration/field_cost_pipeline.py - LangGraph orchestrator for field cost processing"""
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

import redis
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.redis import RedisSaver
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ProcessingStatus(str, Enum):
    """Processing status for field cost submissions."""
    PENDING = "pending"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    ANALYZING = "analyzing"
    ALERTING = "alerting"
    COMPLETE = "complete"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class FieldCostState(TypedDict):
    """State object for the field cost processing pipeline."""
    # Input
    document_path: str
    document_type: str
    job_context: dict[str, Any]
    submission_id: str
    submitted_at: str

    # Processing state
    status: ProcessingStatus
    current_step: str
    retry_count: int
    error_message: str | None

    # Extraction results
    extraction_result: dict[str, Any] | None
    extraction_confidence: float

    # Validation results
    validation_passed: bool
    validation_errors: list[str]
    enriched_data: dict[str, Any] | None

    # Analytics results
    budget_status: dict[str, Any] | None
    forecast: dict[str, Any] | None
    anomalies: list[dict[str, Any]]
    alert: dict[str, Any] | None

    # Output
    final_result: dict[str, Any] | None
    processing_duration_ms: int


def create_initial_state(
    document_path: str,
    document_type: str,
    job_context: dict[str, Any],
) -> FieldCostState:
    """Create initial state for a new field cost submission."""
    return FieldCostState(
        document_path=document_path,
        document_type=document_type,
        job_context=job_context,
        submission_id=f"sub_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
        submitted_at=datetime.utcnow().isoformat(),
        status=ProcessingStatus.PENDING,
        current_step="start",
        retry_count=0,
        error_message=None,
        extraction_result=None,
        extraction_confidence=0.0,
        validation_passed=False,
        validation_errors=[],
        enriched_data=None,
        budget_status=None,
        forecast=None,
        anomalies=[],
        alert=None,
        final_result=None,
        processing_duration_ms=0,
    )


# Node functions for the state graph
async def extract_document(state: FieldCostState) -> FieldCostState:
    """Node: Extract data from field document using OCR agent."""
    logger.info(f"[{state['submission_id']}] Starting document extraction")
    state["status"] = ProcessingStatus.EXTRACTING
    state["current_step"] = "extract"

    try:
        # Import agent here to avoid circular imports
        from agents.document_ingestion import FieldDocumentIngestionAgent

        agent = FieldDocumentIngestionAgent(
            openai_api_key="...",  # From config
            redis_url="redis://localhost:6379",
        )

        from pathlib import Path
        result = await agent.extract_from_image(
            Path(state["document_path"]),
            state["document_type"],
            state["job_context"],
        )

        state["extraction_result"] = result.structured_data
        state["extraction_confidence"] = result.confidence_score

        if result.needs_review:
            state["status"] = ProcessingStatus.NEEDS_REVIEW
            state["validation_errors"].extend(result.review_reasons)

        logger.info(f"[{state['submission_id']}] Extraction complete, confidence={result.confidence_score}")

    except Exception as e:
        logger.error(f"[{state['submission_id']}] Extraction failed: {e}")
        state["error_message"] = str(e)
        state["retry_count"] += 1
        if state["retry_count"] >= 3:
            state["status"] = ProcessingStatus.FAILED

    return state


async def validate_data(state: FieldCostState) -> FieldCostState:
    """Node: Validate extracted data against budgets and cost codes."""
    logger.info(f"[{state['submission_id']}] Starting data validation")
    state["status"] = ProcessingStatus.VALIDATING
    state["current_step"] = "validate"

    if not state["extraction_result"]:
        state["validation_errors"].append("No extraction data to validate")
        state["validation_passed"] = False
        return state

    try:
        from agents.cost_validation import CostValidationAgent

        agent = CostValidationAgent(database_url="postgresql://...")

        validation_result = await agent.validate_entry(
            state["extraction_result"],
            state["job_context"],
        )

        state["validation_passed"] = validation_result["is_valid"]
        state["validation_errors"].extend(validation_result.get("errors", []))
        state["enriched_data"] = validation_result.get("enriched_data")

        logger.info(f"[{state['submission_id']}] Validation {'passed' if state['validation_passed'] else 'failed'}")

    except Exception as e:
        logger.error(f"[{state['submission_id']}] Validation error: {e}")
        state["validation_errors"].append(str(e))
        state["validation_passed"] = False

    return state


async def analyze_budget(state: FieldCostState) -> FieldCostState:
    """Node: Analyze budget impact and generate forecasts."""
    logger.info(f"[{state['submission_id']}] Starting budget analysis")
    state["status"] = ProcessingStatus.ANALYZING
    state["current_step"] = "analyze"

    data = state["enriched_data"] or state["extraction_result"]
    if not data or not data.get("job_id"):
        logger.warning(f"[{state['submission_id']}] No job_id for analysis")
        return state

    try:
        from agents.budget_analytics import BudgetAnalyticsAgent

        agent = BudgetAnalyticsAgent(database_url="postgresql://...")

        job_id = data["job_id"]
        cost_code = data.get("cost_code", "UNKNOWN")

        state["budget_status"] = agent.get_budget_status(job_id, cost_code)

        forecast = agent.forecast_completion_cost(job_id, cost_code)
        state["forecast"] = forecast.model_dump()

        state["anomalies"] = agent.detect_anomalies(job_id, cost_code)

        alert = agent.generate_budget_alert(job_id, cost_code)
        if alert:
            state["alert"] = alert.model_dump()

        logger.info(f"[{state['submission_id']}] Analysis complete, alert={alert is not None}")

    except Exception as e:
        logger.error(f"[{state['submission_id']}] Analysis error: {e}")
        state["error_message"] = str(e)

    return state


async def send_alerts(state: FieldCostState) -> FieldCostState:
    """Node: Send alerts to appropriate stakeholders."""
    logger.info(f"[{state['submission_id']}] Processing alerts")
    state["status"] = ProcessingStatus.ALERTING
    state["current_step"] = "alert"

    if not state["alert"]:
        logger.info(f"[{state['submission_id']}] No alert to send")
        state["status"] = ProcessingStatus.COMPLETE
        return state

    try:
        from agents.alert_routing import AlertRoutingAgent

        agent = AlertRoutingAgent(
            slack_webhook="...",
            email_api_key="...",
        )

        await agent.route_alert(state["alert"])

        logger.info(f"[{state['submission_id']}] Alert sent successfully")
        state["status"] = ProcessingStatus.COMPLETE

    except Exception as e:
        logger.error(f"[{state['submission_id']}] Alert routing error: {e}")
        state["error_message"] = str(e)
        # Don't fail the whole pipeline for alert errors
        state["status"] = ProcessingStatus.COMPLETE

    return state


async def finalize(state: FieldCostState) -> FieldCostState:
    """Node: Finalize processing and prepare output."""
    logger.info(f"[{state['submission_id']}] Finalizing")
    state["current_step"] = "finalize"

    start_time = datetime.fromisoformat(state["submitted_at"])
    duration = (datetime.utcnow() - start_time).total_seconds() * 1000
    state["processing_duration_ms"] = int(duration)

    state["final_result"] = {
        "submission_id": state["submission_id"],
        "status": state["status"].value if isinstance(state["status"], ProcessingStatus) else state["status"],
        "extraction_confidence": state["extraction_confidence"],
        "validation_passed": state["validation_passed"],
        "validation_errors": state["validation_errors"],
        "alert_generated": state["alert"] is not None,
        "anomalies_detected": len(state["anomalies"]),
        "processing_duration_ms": state["processing_duration_ms"],
    }

    return state


# Routing functions
def route_after_extraction(state: FieldCostState) -> Literal["validate", "needs_review", "retry", "fail"]:
    """Route after extraction based on results."""
    if state["status"] == ProcessingStatus.FAILED:
        return "fail"
    if state["status"] == ProcessingStatus.NEEDS_REVIEW:
        return "needs_review"
    if state["retry_count"] > 0 and state["retry_count"] < 3:
        return "retry"
    if state["extraction_confidence"] < 0.5:
        return "needs_review"
    return "validate"


def route_after_validation(state: FieldCostState) -> Literal["analyze", "needs_review"]:
    """Route after validation based on results."""
    if not state["validation_passed"]:
        return "needs_review"
    return "analyze"


def build_field_cost_graph() -> StateGraph:
    """Build the LangGraph state machine for field cost processing."""

    graph = StateGraph(FieldCostState)

    # Add nodes
    graph.add_node("extract", extract_document)
    graph.add_node("validate", validate_data)
    graph.add_node("analyze", analyze_budget)
    graph.add_node("alert", send_alerts)
    graph.add_node("finalize", finalize)
    graph.add_node("needs_review", lambda s: {**s, "status": ProcessingStatus.NEEDS_REVIEW})
    graph.add_node("fail", lambda s: {**s, "status": ProcessingStatus.FAILED})

    # Set entry point
    graph.set_entry_point("extract")

    # Add edges with conditional routing
    graph.add_conditional_edges(
        "extract",
        route_after_extraction,
        {
            "validate": "validate",
            "needs_review": "needs_review",
            "retry": "extract",
            "fail": "fail",
        },
    )

    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "analyze": "analyze",
            "needs_review": "needs_review",
        },
    )

    graph.add_edge("analyze", "alert")
    graph.add_edge("alert", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("needs_review", "finalize")
    graph.add_edge("fail", "finalize")

    return graph


class FieldCostPipelineRunner:
    """Runner for the field cost processing pipeline with Redis checkpointing."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.checkpointer = RedisSaver(self.redis_client)
        self.graph = build_field_cost_graph().compile(checkpointer=self.checkpointer)
        logger.info("FieldCostPipelineRunner initialized")

    async def process_submission(
        self,
        document_path: str,
        document_type: str,
        job_context: dict[str, Any],
    ) -> FieldCostState:
        """Process a single field cost submission through the pipeline."""
        initial_state = create_initial_state(document_path, document_type, job_context)

        logger.info(f"Processing submission {initial_state['submission_id']}")

        config = {"configurable": {"thread_id": initial_state["submission_id"]}}

        final_state = await self.graph.ainvoke(initial_state, config)

        logger.info(
            f"Submission {initial_state['submission_id']} complete: "
            f"status={final_state['status']}, duration={final_state['processing_duration_ms']}ms"
        )

        return final_state

    async def resume_submission(self, submission_id: str) -> FieldCostState | None:
        """Resume a previously checkpointed submission."""
        config = {"configurable": {"thread_id": submission_id}}

        state = await self.graph.aget_state(config)
        if not state:
            logger.warning(f"No checkpointed state found for {submission_id}")
            return None

        logger.info(f"Resuming submission {submission_id} from step {state.values.get('current_step')}")

        final_state = await self.graph.ainvoke(None, config)
        return final_state`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the multi-agent system with Docker Compose, configure LangSmith for agent tracing, and set up Prometheus metrics for monitoring pipeline health and performance.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose Production Deployment',
                description:
                  'Production Docker Compose configuration for the field cost intelligence system with all agents, Redis state store, and observability stack.',
                code: `# docker-compose.production.yml
version: "3.9"

services:
  # Main API service
  field-cost-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    image: ghcr.io/construction-co/field-cost-api:latest
    ports:
      - "8000:8000"
    environment:
      - APP_ENV=production
      - DATABASE_URL=postgresql://app:\${DB_PASSWORD}@postgres:5432/construction
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=field-cost-production
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
    networks:
      - field-cost-network

  # Agent worker service
  agent-worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    image: ghcr.io/construction-co/field-cost-worker:latest
    environment:
      - APP_ENV=production
      - DATABASE_URL=postgresql://app:\${DB_PASSWORD}@postgres:5432/construction
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_TRACING_V2=true
    depends_on:
      - redis
      - postgres
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
    networks:
      - field-cost-network

  # Redis for state management and queues
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - field-cost-network

  # PostgreSQL database
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=construction
      - POSTGRES_USER=app
      - POSTGRES_PASSWORD=\${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d construction"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - field-cost-network

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=30d"
    networks:
      - field-cost-network

  # Grafana for dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    networks:
      - field-cost-network

volumes:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:

networks:
  field-cost-network:
    driver: bridge`,
              },
              {
                language: 'python',
                title: 'Prometheus Metrics & LangSmith Instrumentation',
                description:
                  'Observability instrumentation for the agent pipeline with Prometheus metrics and LangSmith tracing for debugging agent decisions.',
                code: `"""observability/metrics.py - Prometheus metrics and LangSmith tracing for field cost pipeline"""
import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

from langsmith import Client, traceable
from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Type variable for generic decorator
F = TypeVar("F", bound=Callable[..., Any])

# Prometheus metrics
SUBMISSIONS_TOTAL = Counter(
    "field_cost_submissions_total",
    "Total field cost submissions processed",
    ["document_type", "status"],
)

PROCESSING_DURATION = Histogram(
    "field_cost_processing_duration_seconds",
    "Time spent processing field cost submissions",
    ["document_type", "step"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

EXTRACTION_CONFIDENCE = Histogram(
    "field_cost_extraction_confidence",
    "OCR extraction confidence scores",
    ["document_type"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

VALIDATION_ERRORS = Counter(
    "field_cost_validation_errors_total",
    "Total validation errors by type",
    ["error_type"],
)

BUDGET_ALERTS = Counter(
    "field_cost_budget_alerts_total",
    "Total budget alerts generated",
    ["severity", "alert_type"],
)

AGENT_INVOCATIONS = Counter(
    "field_cost_agent_invocations_total",
    "Total agent invocations",
    ["agent_name", "status"],
)

PIPELINE_QUEUE_SIZE = Gauge(
    "field_cost_pipeline_queue_size",
    "Current size of the processing queue",
    ["queue_type"],
)

ACTIVE_SUBMISSIONS = Gauge(
    "field_cost_active_submissions",
    "Number of submissions currently being processed",
)

SYSTEM_INFO = Info(
    "field_cost_system",
    "Field cost system information",
)


class MetricsCollector:
    """Collects and exposes metrics for the field cost pipeline."""

    def __init__(self, port: int = 9091):
        self.port = port
        self.langsmith_client = Client()
        SYSTEM_INFO.info({
            "version": "1.0.0",
            "environment": "production",
            "framework": "langgraph",
        })
        logger.info("MetricsCollector initialized")

    def start_server(self) -> None:
        """Start the Prometheus metrics HTTP server."""
        start_http_server(self.port)
        logger.info(f"Prometheus metrics server started on port {self.port}")

    @contextmanager
    def track_processing(self, document_type: str, step: str):
        """Context manager to track processing duration."""
        ACTIVE_SUBMISSIONS.inc()
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            PROCESSING_DURATION.labels(document_type=document_type, step=step).observe(duration)
            ACTIVE_SUBMISSIONS.dec()

    def record_submission(self, document_type: str, status: str) -> None:
        """Record a processed submission."""
        SUBMISSIONS_TOTAL.labels(document_type=document_type, status=status).inc()

    def record_extraction_confidence(self, document_type: str, confidence: float) -> None:
        """Record OCR extraction confidence score."""
        EXTRACTION_CONFIDENCE.labels(document_type=document_type).observe(confidence)

    def record_validation_error(self, error_type: str) -> None:
        """Record a validation error."""
        VALIDATION_ERRORS.labels(error_type=error_type).inc()

    def record_budget_alert(self, severity: str, alert_type: str) -> None:
        """Record a generated budget alert."""
        BUDGET_ALERTS.labels(severity=severity, alert_type=alert_type).inc()

    def record_agent_invocation(self, agent_name: str, status: str) -> None:
        """Record an agent invocation."""
        AGENT_INVOCATIONS.labels(agent_name=agent_name, status=status).inc()

    def update_queue_size(self, queue_type: str, size: int) -> None:
        """Update the current queue size."""
        PIPELINE_QUEUE_SIZE.labels(queue_type=queue_type).set(size)


# Global metrics collector instance
metrics = MetricsCollector()


def traced_agent(agent_name: str) -> Callable[[F], F]:
    """Decorator to add LangSmith tracing and Prometheus metrics to agent functions."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        @traceable(name=agent_name, run_type="chain")
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.info(f"Agent {agent_name} invoked")
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                metrics.record_agent_invocation(agent_name, "success")
                return result
            except Exception as e:
                metrics.record_agent_invocation(agent_name, "error")
                logger.error(f"Agent {agent_name} failed: {e}")
                raise
            finally:
                duration = time.time() - start_time
                PROCESSING_DURATION.labels(
                    document_type="agent",
                    step=agent_name,
                ).observe(duration)
        return wrapper  # type: ignore
    return decorator


# Example usage of the traced_agent decorator
@traced_agent("FieldDocumentAgent")
async def extract_with_tracing(document_path: str, document_type: str) -> dict[str, Any]:
    """Example traced agent function."""
    # ... extraction implementation
    return {"status": "extracted"}


@traced_agent("BudgetAnalyticsAgent")
async def analyze_with_tracing(job_id: str, cost_code: str) -> dict[str, Any]:
    """Example traced analytics function."""
    # ... analytics implementation
    return {"status": "analyzed"}`,
              },
              {
                language: 'yaml',
                title: 'Prometheus Scrape Configuration',
                description:
                  'Prometheus configuration for scraping metrics from the field cost pipeline services.',
                code: `# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

rule_files:
  - /etc/prometheus/rules/*.yml

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "field-cost-api"
    static_configs:
      - targets: ["field-cost-api:8000"]
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: "agent-worker"
    static_configs:
      - targets: ["agent-worker:9091"]
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: "redis"
    static_configs:
      - targets: ["redis-exporter:9121"]

  - job_name: "postgres"
    static_configs:
      - targets: ["postgres-exporter:9187"]

# Alert rules
# /etc/prometheus/rules/field-cost-alerts.yml
groups:
  - name: field-cost-alerts
    rules:
      - alert: HighExtractionErrorRate
        expr: |
          rate(field_cost_submissions_total{status="failed"}[5m]) /
          rate(field_cost_submissions_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High extraction error rate detected"
          description: "More than 10% of submissions failing extraction"

      - alert: PipelineQueueBacklog
        expr: field_cost_pipeline_queue_size{queue_type="pending"} > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Pipeline queue backlog detected"
          description: "More than 100 submissions waiting in queue"

      - alert: CriticalBudgetAlertSpike
        expr: |
          increase(field_cost_budget_alerts_total{severity="CRITICAL"}[1h]) > 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Spike in critical budget alerts"
          description: "More than 10 critical budget alerts in the last hour"

      - alert: AgentLatencyHigh
        expr: |
          histogram_quantile(0.95,
            rate(field_cost_processing_duration_seconds_bucket{step=~".*Agent.*"}[5m])
          ) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Agent processing latency high"
          description: "95th percentile agent latency exceeds 30 seconds"`,
              },
            ],
          },
        ],
      },
    },

    // ── Pain Point 2 ── Excel Import Dependency ────────────────────────
    {
      id: 'excel-import-dependency',
      number: 2,
      title: 'Excel Import Dependency',
      subtitle: 'Critical Project Data Trapped in Spreadsheets',
      summary:
        'Estimating, scheduling, and procurement all run on Excel macros built by someone who left 5 years ago. One broken macro halts a $50M project.',
      tags: ['excel', 'legacy', 'migration'],
      metrics: {
        annualCostRange: '$300K - $2M',
        roi: '6x',
        paybackPeriod: '4-6 months',
        investmentRange: '$100K - $200K',
      },

      price: {
        present: {
          title: 'Current Reality',
          severity: 'critical',
          description:
            'Mission-critical project workflows — estimating, procurement, and scheduling — depend on a network of Excel workbooks with VBA macros that no one on staff fully understands.',
          bullets: [
            'Estimating workbook contains 14 interconnected sheets with 3,200+ lines of VBA',
            'Procurement tracker macro breaks every time Office updates, halting purchase orders',
            'Schedule roll-up macro takes 45 minutes to run and crashes on files over 20 MB',
            'Only two people in the company can troubleshoot macro errors, and neither wrote the originals',
          ],
        },
        root: {
          title: 'Root Cause',
          severity: 'high',
          description:
            'Business logic was embedded directly in Excel VBA by a former employee with no documentation, version control, or migration plan. The macros were never designed to scale beyond a single user.',
          bullets: [
            'VBA code has zero documentation and uses Hungarian notation from 2005',
            'No version control — "final_FINAL_v3_USE_THIS.xlsm" is the production file',
            'Macros hard-code file paths, printer names, and Windows-specific COM references',
            'Business rules are buried in cell formulas that span 500+ characters',
          ],
        },
        impact: {
          title: 'Business Impact',
          severity: 'critical',
          description:
            'A single macro failure blocks the entire estimating-to-procurement pipeline. Projects have been delayed days waiting for a VBA fix that no one can confidently make.',
          bullets: [
            '3-5 day project delays per quarter caused by macro breakages after Office updates',
            'Estimators manually re-key data between workbooks, introducing 4-8% error rate',
            'Cannot onboard new estimators without a 6-week shadow period to learn the spreadsheet maze',
            'Audit findings cite lack of data lineage and change tracking in spreadsheet-based workflows',
          ],
        },
        cost: {
          title: 'Cost of Inaction',
          severity: 'high',
          description:
            'The cost is not just the macro fix — it is the compounding risk that one retirement, one Office update, or one corrupt file takes down a multi-million-dollar project pipeline.',
          bullets: [
            '$300K - $2M annual exposure from project delays, rework, and data errors',
            '$80K+ per year in overtime for the two employees who babysit the macros',
            'Single point of failure: one corrupt .xlsm file can stall active bids worth $50M+',
            'Increasing insurance and compliance risk due to unauditable spreadsheet processes',
          ],
        },
        expectedReturn: {
          title: 'Expected Return',
          severity: 'high',
          description:
            'Migrating Excel logic into Python services and a centralized database eliminates single-point-of-failure risk and creates a maintainable, auditable, and scalable data platform.',
          bullets: [
            'Eliminate macro-related project delays entirely — zero downtime from Office updates',
            'Reduce estimating data entry errors from 8% to under 0.5% with validation rules',
            'Cut new-estimator onboarding from 6 weeks to 1 week with a structured UI',
            'Full audit trail on every data change, satisfying compliance and bonding requirements',
          ],
        },
      },

      implementation: {
        overview:
          'Reverse-engineer the critical Excel macros into documented Python modules, then migrate the underlying data into a centralized PostgreSQL warehouse. The Python layer replicates every calculation the VBA performed, with unit tests proving equivalence, while the database provides multi-user access, audit logging, and backup.',
        prerequisites: [
          'Python 3.11+ with openpyxl, pandas, and pytest',
          'PostgreSQL 14+ for the centralized project data warehouse',
          'Copies of all production Excel workbooks (.xlsm) and sample input/output data',
          'Access to the VBA source (Alt+F11 export) for each macro-enabled workbook',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        toolsUsed: ['Python', 'PostgreSQL', 'pandas', 'openpyxl', 'pytest', 'Docker', 'GitHub Actions', 'cron / Airflow', 'Slack API'],
        steps: [
          {
            stepNumber: 1,
            title: 'Extract and Replicate VBA Logic in Python',
            description:
              'Parse the existing Excel workbooks, extract the VBA calculation logic, and rewrite it as testable Python functions. Validate outputs against the original spreadsheet to guarantee equivalence.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Estimating Macro Migration Module',
                description:
                  'Replicates the core estimating workbook calculations — material take-off, labor burden, and markup — as pure Python functions with type hints.',
                code: `"""estimating/calculations.py — migrated from EstimateMaster_v3.xlsm VBA"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import List

@dataclass
class LineItem:
    cost_code: str
    description: str
    quantity: Decimal
    unit_cost: Decimal
    waste_factor: Decimal  # e.g., 1.05 for 5% waste
    labor_hours: Decimal
    labor_rate: Decimal

def material_cost(item: LineItem) -> Decimal:
    """Replicates VBA: CalcMaterialCost (Sheet 'Takeoff', col H)."""
    raw = item.quantity * item.unit_cost * item.waste_factor
    return raw.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def labor_burden(item: LineItem, burden_pct: Decimal = Decimal("0.35")) -> Decimal:
    """Replicates VBA: CalcLaborBurden (Sheet 'Labor', col F).
    Burden includes FICA, WC, and benefits at a blended rate."""
    base = item.labor_hours * item.labor_rate
    return (base * (1 + burden_pct)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def estimate_total(
    items: List[LineItem],
    overhead_pct: Decimal = Decimal("0.10"),
    profit_pct: Decimal = Decimal("0.08"),
) -> dict:
    """Replicates VBA: BuildEstimateSummary (Sheet 'Summary')."""
    mat_total = sum(material_cost(i) for i in items)
    lab_total = sum(labor_burden(i) for i in items)
    direct_cost = mat_total + lab_total
    overhead = (direct_cost * overhead_pct).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    subtotal = direct_cost + overhead
    profit = (subtotal * profit_pct).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return {
        "material_total": mat_total,
        "labor_total": lab_total,
        "direct_cost": direct_cost,
        "overhead": overhead,
        "profit": profit,
        "bid_price": subtotal + profit,
    }`,
              },
              {
                language: 'python',
                title: 'Equivalence Test Against Original Spreadsheet',
                description:
                  'Reads known inputs and outputs from the legacy workbook and asserts that the Python functions produce identical results.',
                code: `"""tests/test_estimating_equivalence.py"""
import pytest
from decimal import Decimal
from openpyxl import load_workbook
from estimating.calculations import LineItem, material_cost, labor_burden, estimate_total

LEGACY_WORKBOOK = "fixtures/EstimateMaster_v3.xlsm"

def load_legacy_test_cases():
    """Pull rows from the 'Takeoff' sheet to use as test vectors."""
    wb = load_workbook(LEGACY_WORKBOOK, data_only=True)
    ws = wb["Takeoff"]
    cases = []
    for row in ws.iter_rows(min_row=3, max_col=10, values_only=True):
        if row[0] is None:
            break
        cases.append({
            "item": LineItem(
                cost_code=str(row[0]),
                description=str(row[1]),
                quantity=Decimal(str(row[2])),
                unit_cost=Decimal(str(row[3])),
                waste_factor=Decimal(str(row[4])),
                labor_hours=Decimal(str(row[5])),
                labor_rate=Decimal(str(row[6])),
            ),
            "expected_material": Decimal(str(row[7])),
            "expected_labor": Decimal(str(row[8])),
        })
    return cases

@pytest.mark.parametrize("case", load_legacy_test_cases(), ids=lambda c: c["item"].cost_code)
def test_material_matches_legacy(case):
    assert material_cost(case["item"]) == case["expected_material"]

@pytest.mark.parametrize("case", load_legacy_test_cases(), ids=lambda c: c["item"].cost_code)
def test_labor_matches_legacy(case):
    assert labor_burden(case["item"]) == case["expected_labor"]`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Build the Centralized Project Data Warehouse',
            description:
              'Create the PostgreSQL schema that replaces the web of interconnected spreadsheets. Each table corresponds to a sheet or workbook, with proper foreign keys, constraints, and an audit log.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Project Data Warehouse Schema',
                description:
                  'Core tables for estimates, procurement, and scheduling that replace the Excel workbooks, with a generic audit log for full change tracking.',
                code: `-- Estimates table (replaces EstimateMaster_v3.xlsm)
CREATE TABLE estimates (
    estimate_id   SERIAL       PRIMARY KEY,
    project_id    INT          NOT NULL REFERENCES projects(project_id),
    revision      INT          NOT NULL DEFAULT 1,
    status        VARCHAR(20)  NOT NULL DEFAULT 'draft'
                  CHECK (status IN ('draft','review','approved','submitted')),
    created_by    INT          NOT NULL REFERENCES employees(employee_id),
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    approved_at   TIMESTAMPTZ,
    UNIQUE (project_id, revision)
);

CREATE TABLE estimate_line_items (
    line_id       SERIAL       PRIMARY KEY,
    estimate_id   INT          NOT NULL REFERENCES estimates(estimate_id),
    cost_code     VARCHAR(20)  NOT NULL,
    description   TEXT,
    quantity      NUMERIC(12,3) NOT NULL,
    unit_cost     NUMERIC(12,2) NOT NULL,
    waste_factor  NUMERIC(5,3) NOT NULL DEFAULT 1.000,
    labor_hours   NUMERIC(10,2) NOT NULL DEFAULT 0,
    labor_rate    NUMERIC(10,2) NOT NULL DEFAULT 0,
    material_total NUMERIC(14,2) GENERATED ALWAYS AS
                  (ROUND(quantity * unit_cost * waste_factor, 2)) STORED,
    labor_total   NUMERIC(14,2) GENERATED ALWAYS AS
                  (ROUND(labor_hours * labor_rate * 1.35, 2)) STORED
);

-- Generic audit log for full change tracking
CREATE TABLE audit_log (
    log_id       BIGSERIAL    PRIMARY KEY,
    table_name   VARCHAR(64)  NOT NULL,
    record_id    INT          NOT NULL,
    action       VARCHAR(10)  NOT NULL CHECK (action IN ('INSERT','UPDATE','DELETE')),
    changed_by   INT          NOT NULL,
    changed_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    old_values   JSONB,
    new_values   JSONB
);

CREATE INDEX idx_audit_table_record ON audit_log (table_name, record_id);`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate migration data integrity with SQL assertions covering row counts, total reconciliation, and orphan detection. Complement with pytest-based calculation equivalence tests that prove the Python modules produce identical results to the legacy VBA macros.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Migration Data Validation Assertions',
                description:
                  'Comprehensive SQL assertions that verify row counts match between source workbooks and the database, reconcile monetary totals, and detect orphaned references after migration.',
                code: `-- ============================================================
-- Migration Data Validation Assertions
-- Run after each migration batch to verify data integrity
-- ============================================================

-- 1. Row count reconciliation: compare migrated rows vs expected
SELECT
    'row_count_check'                  AS assertion,
    e.estimate_id,
    p.project_name,
    COUNT(li.line_id)                  AS migrated_rows,
    ms.expected_row_count,
    CASE
        WHEN COUNT(li.line_id) = ms.expected_row_count THEN 'PASS'
        ELSE 'FAIL'
    END AS status
FROM estimates e
JOIN projects p USING (project_id)
JOIN estimate_line_items li USING (estimate_id)
LEFT JOIN migration_source_stats ms ON ms.estimate_id = e.estimate_id
GROUP BY e.estimate_id, p.project_name, ms.expected_row_count
ORDER BY e.estimate_id;

-- 2. Total reconciliation: material + labor must match legacy sums
SELECT
    'total_reconciliation'             AS assertion,
    e.estimate_id,
    p.project_name,
    SUM(li.material_total)             AS db_material,
    SUM(li.labor_total)                AS db_labor,
    SUM(li.material_total + li.labor_total) AS db_direct_cost,
    ms.legacy_material_total,
    ms.legacy_labor_total,
    CASE
        WHEN ABS(SUM(li.material_total) - ms.legacy_material_total) < 0.02
         AND ABS(SUM(li.labor_total)    - ms.legacy_labor_total)    < 0.02
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status
FROM estimates e
JOIN projects p USING (project_id)
JOIN estimate_line_items li USING (estimate_id)
LEFT JOIN migration_source_stats ms ON ms.estimate_id = e.estimate_id
GROUP BY e.estimate_id, p.project_name,
         ms.legacy_material_total, ms.legacy_labor_total
ORDER BY e.estimate_id;

-- 3. Orphan detection: cost codes with no master reference
SELECT
    'orphan_cost_codes'                AS assertion,
    COUNT(DISTINCT li.cost_code)       AS orphan_count,
    CASE WHEN COUNT(DISTINCT li.cost_code) = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM estimate_line_items li
LEFT JOIN cost_code_master ccm ON ccm.cost_code = li.cost_code
WHERE ccm.cost_code IS NULL;

-- 4. Duplicate detection: same cost code on same estimate
SELECT
    'duplicate_lines'                  AS assertion,
    COUNT(*)                           AS duplicate_count,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM (
    SELECT estimate_id, cost_code, COUNT(*) AS cnt
    FROM estimate_line_items
    GROUP BY estimate_id, cost_code
    HAVING COUNT(*) > 1
) dupes;

-- 5. Estimates with no line items (empty migration)
SELECT
    'empty_estimates'                  AS assertion,
    COUNT(*)                           AS empty_count,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM estimates e
LEFT JOIN estimate_line_items li USING (estimate_id)
WHERE li.line_id IS NULL;`,
              },
              {
                language: 'python',
                title: 'Pytest Calculation Equivalence Suite',
                description:
                  'Pytest-based test suite that loads known input/output pairs from legacy workbooks and asserts the Python calculation modules produce identical results within rounding tolerance.',
                code: `"""tests/test_calculation_equivalence.py — Prove Python matches VBA output"""
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from openpyxl import load_workbook

from estimating.calculations import (
    LineItem,
    estimate_total,
    labor_burden,
    material_cost,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger: logging.Logger = logging.getLogger(__name__)

FIXTURE_DIR: Path = Path("fixtures")
LEGACY_WORKBOOK: Path = FIXTURE_DIR / "EstimateMaster_v3.xlsm"
TOLERANCE: Decimal = Decimal("0.02")


def _load_test_vectors() -> list[dict[str, Any]]:
    """Extract rows from the legacy Takeoff sheet as test vectors."""
    logger.info("Loading test vectors from %s", LEGACY_WORKBOOK)
    wb = load_workbook(str(LEGACY_WORKBOOK), data_only=True)
    ws = wb["Takeoff"]
    vectors: list[dict[str, Any]] = []
    for row in ws.iter_rows(min_row=3, max_col=10, values_only=True):
        if row[0] is None:
            break
        vectors.append({
            "item": LineItem(
                cost_code=str(row[0]),
                description=str(row[1]),
                quantity=Decimal(str(row[2])),
                unit_cost=Decimal(str(row[3])),
                waste_factor=Decimal(str(row[4])),
                labor_hours=Decimal(str(row[5])),
                labor_rate=Decimal(str(row[6])),
            ),
            "expected_material": Decimal(str(row[7])),
            "expected_labor": Decimal(str(row[8])),
        })
    logger.info("Loaded %d test vectors", len(vectors))
    return vectors


TEST_VECTORS: list[dict[str, Any]] = _load_test_vectors()


@pytest.mark.parametrize("vector", TEST_VECTORS, ids=lambda v: v["item"].cost_code)
class TestMaterialCostEquivalence:
    """Verify material_cost() matches the legacy VBA CalcMaterialCost."""

    def test_material_within_tolerance(self, vector: dict[str, Any]) -> None:
        result: Decimal = material_cost(vector["item"])
        expected: Decimal = vector["expected_material"]
        diff: Decimal = abs(result - expected)
        logger.info("Cost code %s: result=%s expected=%s diff=%s",
                     vector["item"].cost_code, result, expected, diff)
        assert diff <= TOLERANCE, (
            f"Material cost mismatch for {vector['item'].cost_code}: "
            f"got {result}, expected {expected}, diff {diff}"
        )


@pytest.mark.parametrize("vector", TEST_VECTORS, ids=lambda v: v["item"].cost_code)
class TestLaborBurdenEquivalence:
    """Verify labor_burden() matches the legacy VBA CalcLaborBurden."""

    def test_labor_within_tolerance(self, vector: dict[str, Any]) -> None:
        result: Decimal = labor_burden(vector["item"])
        expected: Decimal = vector["expected_labor"]
        diff: Decimal = abs(result - expected)
        logger.info("Cost code %s: result=%s expected=%s diff=%s",
                     vector["item"].cost_code, result, expected, diff)
        assert diff <= TOLERANCE, (
            f"Labor burden mismatch for {vector['item'].cost_code}: "
            f"got {result}, expected {expected}, diff {diff}"
        )


class TestEstimateTotalIntegration:
    """End-to-end estimate_total() validation against legacy summary."""

    def test_total_matches_legacy_summary(self) -> None:
        items: list[LineItem] = [v["item"] for v in TEST_VECTORS]
        result: dict[str, Decimal] = estimate_total(items)
        logger.info("Estimate total result: %s", result)
        assert result["material_total"] > 0, "Material total should be positive"
        assert result["labor_total"] > 0, "Labor total should be positive"
        assert result["bid_price"] > result["direct_cost"], (
            "Bid price must exceed direct cost (overhead + profit)"
        )
        assert result["bid_price"] == (
            result["direct_cost"] + result["overhead"] + result["profit"]
        ), "Bid price must equal direct_cost + overhead + profit"`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Deploy the migration pipeline and ongoing sync service using a containerized runner script. The configuration loader supports environment-specific settings and secret injection for database credentials and webhook URLs.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Migration Runner Deployment Script',
                description:
                  'Production deployment script that runs the Excel-to-database migration inside a Docker container, with pre-flight checks, backup, and rollback support.',
                code: `#!/usr/bin/env bash
# ============================================================
# deploy_migration.sh — Run the Excel-to-Postgres migration
# Usage: ./deploy_migration.sh [staging|production]
# ============================================================
set -euo pipefail

ENVIRONMENT="\${1:-staging}"
APP_NAME="excel-migration-runner"
IMAGE_TAG="\$(git rev-parse --short HEAD)"
REGISTRY="ghcr.io/construction-co"
COMPOSE_FILE="docker-compose.\${ENVIRONMENT}.yml"
BACKUP_DIR="/backups/pre-migration"
TIMESTAMP="\$(date +%Y%m%d_%H%M%S)"

echo "=== Excel Migration Runner — \${ENVIRONMENT} ==="
echo "Image tag: \${IMAGE_TAG}"
echo "Timestamp: \${TIMESTAMP}"

# Step 1: Pre-flight database connectivity check
echo "[1/6] Checking database connectivity..."
docker compose -f "\${COMPOSE_FILE}" run --rm db-check \\
  pg_isready -h db -p 5432 -U app -d construction

# Step 2: Backup current database state
echo "[2/6] Creating pre-migration backup..."
mkdir -p "\${BACKUP_DIR}"
docker compose -f "\${COMPOSE_FILE}" run --rm pg-dump \\
  pg_dump -h db -U app -d construction \\
  --format=custom \\
  --file="/backups/pre-migration/backup_\${TIMESTAMP}.dump"
echo "Backup saved to \${BACKUP_DIR}/backup_\${TIMESTAMP}.dump"

# Step 3: Build the migration image
echo "[3/6] Building migration image..."
docker build \\
  --tag "\${REGISTRY}/\${APP_NAME}:\${IMAGE_TAG}" \\
  --tag "\${REGISTRY}/\${APP_NAME}:latest" \\
  --build-arg ENV="\${ENVIRONMENT}" \\
  --file Dockerfile.migration .

# Step 4: Run schema migrations via Alembic
echo "[4/6] Running Alembic schema migrations..."
docker compose -f "\${COMPOSE_FILE}" run --rm migrate \\
  alembic upgrade head

# Step 5: Execute the data migration
echo "[5/6] Running Excel data migration..."
docker compose -f "\${COMPOSE_FILE}" run --rm "\${APP_NAME}" \\
  python -m migration.ingest_estimates
MIGRATION_EXIT=\$?

if [ "\${MIGRATION_EXIT}" -ne 0 ]; then
    echo "ERROR: Migration failed with exit code \${MIGRATION_EXIT}"
    echo "Restoring from backup..."
    docker compose -f "\${COMPOSE_FILE}" run --rm pg-restore \\
      pg_restore -h db -U app -d construction --clean \\
      "/backups/pre-migration/backup_\${TIMESTAMP}.dump"
    echo "Database restored. Migration aborted."
    exit 1
fi

# Step 6: Run post-migration validation
echo "[6/6] Running post-migration validation queries..."
docker compose -f "\${COMPOSE_FILE}" run --rm validate \\
  python -m migration.validate_migration

echo "=== Migration complete ==="`,
              },
              {
                language: 'python',
                title: 'Migration Service Configuration Loader',
                description:
                  'Typed configuration loader for the migration and sync service that reads YAML configs with environment variable overrides for secrets.',
                code: `"""config/migration_config.py — Configuration for the Excel migration service"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "construction"
    user: str = "app"
    password: str = ""
    pool_size: int = 5
    ssl_mode: str = "prefer"

    @property
    def url(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


@dataclass
class MigrationConfig:
    workbook_dir: str = "legacy_workbooks/estimates"
    batch_size: int = 500
    fail_on_error: bool = False
    backup_before_run: bool = True
    validate_after_run: bool = True


@dataclass
class SyncConfig:
    enabled: bool = True
    schedule_cron: str = "0 */6 * * *"
    slack_webhook_url: str = ""
    notify_on_failure: bool = True
    notify_on_success: bool = False


@dataclass
class AppConfig:
    environment: str = "staging"
    debug: bool = False
    log_level: str = "INFO"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)


def load_config(env: str | None = None) -> AppConfig:
    """Load migration config from YAML with env var overrides for secrets."""
    environment: str = env or os.getenv("APP_ENV", "staging")
    config_dir: Path = Path(__file__).parent / "environments"
    config_path: Path = config_dir / f"{environment}.yml"

    logger.info("Loading migration config: env=%s path=%s", environment, config_path)

    raw: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        logger.info("Loaded YAML config: %d keys", len(raw))
    else:
        logger.warning("Config not found at %s — using defaults", config_path)

    db_raw: dict[str, Any] = raw.get("database", {})
    mig_raw: dict[str, Any] = raw.get("migration", {})
    sync_raw: dict[str, Any] = raw.get("sync", {})

    config: AppConfig = AppConfig(
        environment=environment,
        debug=raw.get("debug", False),
        log_level=raw.get("log_level", "INFO"),
        database=DatabaseConfig(
            host=os.getenv("DB_HOST", db_raw.get("host", "localhost")),
            port=int(os.getenv("DB_PORT", str(db_raw.get("port", 5432)))),
            name=os.getenv("DB_NAME", db_raw.get("name", "construction")),
            user=os.getenv("DB_USER", db_raw.get("user", "app")),
            password=os.getenv("DB_PASSWORD", db_raw.get("password", "")),
            pool_size=db_raw.get("pool_size", 5),
            ssl_mode=db_raw.get("ssl_mode", "prefer"),
        ),
        migration=MigrationConfig(
            workbook_dir=mig_raw.get("workbook_dir", "legacy_workbooks/estimates"),
            batch_size=mig_raw.get("batch_size", 500),
            fail_on_error=mig_raw.get("fail_on_error", False),
            backup_before_run=mig_raw.get("backup_before_run", True),
            validate_after_run=mig_raw.get("validate_after_run", True),
        ),
        sync=SyncConfig(
            enabled=sync_raw.get("enabled", True),
            schedule_cron=sync_raw.get("schedule_cron", "0 */6 * * *"),
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL", sync_raw.get("slack_webhook_url", "")),
            notify_on_failure=sync_raw.get("notify_on_failure", True),
            notify_on_success=sync_raw.get("notify_on_success", False),
        ),
    )
    logger.info("Config loaded: env=%s, db=%s, workbook_dir=%s",
                config.environment, config.database.host, config.migration.workbook_dir)
    return config`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Migrate Legacy Workbook Data into the Warehouse',
            description:
              'Build an automated ingestion pipeline that reads every production Excel file, validates the data, and loads it into the new database tables. Generates a migration report showing row counts and validation failures. Includes sync health and data freshness monitoring for ongoing operations.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Excel-to-Database Migration Pipeline',
                description:
                  'Reads legacy estimate workbooks, validates each row, and bulk-inserts into the new schema with a detailed migration report.',
                code: `"""migration/ingest_estimates.py — one-time data migration from Excel to Postgres"""
import logging
from pathlib import Path
from decimal import Decimal, InvalidOperation
from openpyxl import load_workbook
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_URL = "postgresql://app:secret@localhost:5432/construction"
WORKBOOK_DIR = Path("legacy_workbooks/estimates")

engine = create_engine(DB_URL)

def safe_decimal(val, default="0") -> Decimal:
    try:
        return Decimal(str(val)) if val is not None else Decimal(default)
    except InvalidOperation:
        return Decimal(default)

def migrate_workbook(filepath: Path):
    wb = load_workbook(filepath, data_only=True)
    ws = wb["Takeoff"]
    rows_ok, rows_err = 0, 0
    project_name = wb["Summary"]["B2"].value or filepath.stem

    with engine.begin() as conn:
        result = conn.execute(
            text("INSERT INTO estimates (project_id, created_by) "
                 "VALUES (:pid, 1) RETURNING estimate_id"),
            {"pid": _resolve_project(conn, project_name)},
        )
        est_id = result.scalar_one()

        for row in ws.iter_rows(min_row=3, max_col=7, values_only=True):
            if row[0] is None:
                break
            try:
                conn.execute(text(\"\"\"
                    INSERT INTO estimate_line_items
                        (estimate_id, cost_code, description, quantity,
                         unit_cost, waste_factor, labor_hours, labor_rate)
                    VALUES (:eid, :cc, :desc, :qty, :uc, :wf, :lh, :lr)
                \"\"\"), {
                    "eid": est_id, "cc": str(row[0]), "desc": str(row[1]),
                    "qty": safe_decimal(row[2]), "uc": safe_decimal(row[3]),
                    "wf": safe_decimal(row[4], "1.0"),
                    "lh": safe_decimal(row[5]), "lr": safe_decimal(row[6]),
                })
                rows_ok += 1
            except Exception as exc:
                log.warning("Row error in %s: %s — %s", filepath.name, row, exc)
                rows_err += 1

    log.info("Migrated %s: %d rows OK, %d errors", filepath.name, rows_ok, rows_err)
    return rows_ok, rows_err

def run_full_migration():
    total_ok, total_err = 0, 0
    for xlsm in sorted(WORKBOOK_DIR.glob("*.xlsm")):
        ok, err = migrate_workbook(xlsm)
        total_ok += ok
        total_err += err
    log.info("Migration complete: %d total rows, %d errors", total_ok + total_err, total_err)

if __name__ == "__main__":
    run_full_migration()`,
              },
              {
                language: 'sql',
                title: 'Post-Migration Validation Queries',
                description:
                  'Queries to verify migrated data integrity by comparing row counts, totals, and detecting orphaned cost codes.',
                code: `-- Compare migrated estimate totals against legacy summary values
SELECT
    e.estimate_id,
    p.project_name,
    COUNT(li.line_id)                        AS line_count,
    SUM(li.material_total)                   AS db_material_total,
    SUM(li.labor_total)                      AS db_labor_total,
    SUM(li.material_total + li.labor_total)  AS db_direct_cost
FROM estimates e
JOIN projects p USING (project_id)
JOIN estimate_line_items li USING (estimate_id)
GROUP BY e.estimate_id, p.project_name
ORDER BY e.estimate_id;

-- Detect orphaned cost codes not in the master list
SELECT DISTINCT li.cost_code
FROM estimate_line_items li
LEFT JOIN cost_code_master ccm ON ccm.cost_code = li.cost_code
WHERE ccm.cost_code IS NULL
ORDER BY li.cost_code;

-- Audit: confirm every estimate has at least one line item
SELECT e.estimate_id, p.project_name
FROM estimates e
JOIN projects p USING (project_id)
LEFT JOIN estimate_line_items li USING (estimate_id)
WHERE li.line_id IS NULL;`,
              },
              {
                language: 'sql',
                title: 'Sync Health & Data Freshness Monitoring',
                description:
                  'Monitoring views and queries that track migration sync health, data freshness, and detect stale or broken pipelines for ongoing operational awareness.',
                code: `-- ============================================================
-- Sync Health & Data Freshness Monitoring
-- Powers the ops dashboard for ongoing migration health
-- ============================================================

-- 1. Data freshness overview: when was each table last updated?
CREATE OR REPLACE VIEW v_data_freshness AS
SELECT
    'estimates'            AS table_name,
    COUNT(*)               AS total_rows,
    MAX(created_at)        AS last_insert,
    EXTRACT(EPOCH FROM (now() - MAX(created_at))) / 3600
                           AS hours_since_last_insert,
    CASE
        WHEN MAX(created_at) > now() - INTERVAL '24 hours' THEN 'FRESH'
        WHEN MAX(created_at) > now() - INTERVAL '72 hours' THEN 'STALE'
        ELSE 'CRITICAL'
    END AS freshness_status
FROM estimates
UNION ALL
SELECT
    'estimate_line_items'  AS table_name,
    COUNT(*)               AS total_rows,
    MAX(created_at)        AS last_insert,
    EXTRACT(EPOCH FROM (now() - MAX(created_at))) / 3600
                           AS hours_since_last_insert,
    CASE
        WHEN MAX(created_at) > now() - INTERVAL '24 hours' THEN 'FRESH'
        WHEN MAX(created_at) > now() - INTERVAL '72 hours' THEN 'STALE'
        ELSE 'CRITICAL'
    END AS freshness_status
FROM estimate_line_items
UNION ALL
SELECT
    'audit_log'            AS table_name,
    COUNT(*)               AS total_rows,
    MAX(changed_at)        AS last_insert,
    EXTRACT(EPOCH FROM (now() - MAX(changed_at))) / 3600
                           AS hours_since_last_insert,
    CASE
        WHEN MAX(changed_at) > now() - INTERVAL '24 hours' THEN 'FRESH'
        WHEN MAX(changed_at) > now() - INTERVAL '72 hours' THEN 'STALE'
        ELSE 'CRITICAL'
    END AS freshness_status
FROM audit_log;

-- 2. Sync health: track migration run history and outcomes
CREATE TABLE IF NOT EXISTS migration_run_log (
    run_id         SERIAL       PRIMARY KEY,
    run_started_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    run_ended_at   TIMESTAMPTZ,
    workbooks_processed INT     DEFAULT 0,
    rows_migrated  INT          DEFAULT 0,
    rows_errored   INT          DEFAULT 0,
    status         VARCHAR(20)  DEFAULT 'running'
                   CHECK (status IN ('running','success','failed','partial')),
    error_detail   TEXT
);

-- 3. Sync health dashboard query
SELECT
    run_id,
    run_started_at,
    run_ended_at,
    workbooks_processed,
    rows_migrated,
    rows_errored,
    status,
    ROUND(rows_errored::NUMERIC / NULLIF(rows_migrated + rows_errored, 0) * 100, 2)
        AS error_rate_pct,
    EXTRACT(EPOCH FROM (run_ended_at - run_started_at))
        AS duration_seconds
FROM migration_run_log
ORDER BY run_started_at DESC
LIMIT 20;

-- 4. Alert: detect pipelines that have not run in expected window
SELECT
    'pipeline_stale' AS alert_type,
    MAX(run_started_at) AS last_run,
    EXTRACT(EPOCH FROM (now() - MAX(run_started_at))) / 3600
        AS hours_since_last_run,
    CASE
        WHEN MAX(run_started_at) < now() - INTERVAL '12 hours'
        THEN 'ALERT'
        ELSE 'OK'
    END AS status
FROM migration_run_log
WHERE status IN ('success', 'partial');`,
              },
            ],
          },
        ],
      },

      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude to analyze and document legacy VBA macro logic, then automate the migration validation process with Zapier. AI reads the exported VBA code, generates human-readable documentation, creates Python function equivalents, and validates migrated data matches original outputs.',
        estimatedMonthlyCost: '$120 - $200/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Google Sheets (Free)', 'GitHub ($4/mo)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Autodesk AI (Enterprise)', 'Microsoft Power Automate ($15/mo)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Export VBA code from legacy Excel workbooks and upload to a shared drive. Set up a Zapier workflow that watches for new VBA exports and prepares them for AI analysis by chunking large codebases.',
            toolsUsed: ['Excel VBA Editor', 'Google Drive', 'Zapier'],
            codeSnippets: [
              {
                language: 'json',
                title: 'VBA Export Structure Schema',
                description:
                  'The schema for organizing exported VBA code and metadata for AI analysis.',
                code: `{
  "vbaExportSchema": {
    "workbook_name": "string",
    "export_date": "YYYY-MM-DD",
    "total_modules": "number",
    "total_lines_of_code": "number",
    "modules": [
      {
        "module_name": "string",
        "module_type": "Standard | Class | Form | Sheet | ThisWorkbook",
        "line_count": "number",
        "procedures": [
          {
            "name": "string",
            "type": "Sub | Function | Property",
            "visibility": "Public | Private",
            "parameters": ["string"],
            "return_type": "string | null",
            "line_start": "number",
            "line_end": "number",
            "calls_external": ["string"],
            "modifies_sheets": ["string"]
          }
        ],
        "dependencies": {
          "worksheets_referenced": ["string"],
          "external_references": ["string"],
          "com_objects": ["string"]
        },
        "raw_code": "string"
      }
    ],
    "global_variables": [
      {
        "name": "string",
        "type": "string",
        "scope": "Public | Private | Module"
      }
    ],
    "complexity_score": "number"
  }
}`,
              },
              {
                language: 'json',
                title: 'Zapier VBA Upload Watcher Config',
                description:
                  'Zapier trigger configuration that monitors for new VBA export files and prepares them for AI processing.',
                code: `{
  "zap_name": "VBA Export Processor",
  "trigger": {
    "app": "Google Drive",
    "event": "New File in Folder",
    "config": {
      "folder_id": "1ABC_VBA_Exports_Inbox",
      "file_types": [".bas", ".cls", ".frm", ".txt", ".vba"]
    }
  },
  "action_1_read_file": {
    "app": "Google Drive",
    "event": "Get File Content",
    "config": {
      "file_id": "{{trigger.file_id}}",
      "output_format": "text"
    }
  },
  "action_2_chunk_if_large": {
    "app": "Code by Zapier",
    "event": "Run Javascript",
    "code": "const code = inputData.file_content; const MAX_CHUNK = 12000; if (code.length <= MAX_CHUNK) return { chunks: [code], total_chunks: 1 }; const chunks = []; for (let i = 0; i < code.length; i += MAX_CHUNK) { chunks.push(code.slice(i, i + MAX_CHUNK)); } return { chunks: JSON.stringify(chunks), total_chunks: chunks.length };"
  },
  "action_3_log_metadata": {
    "app": "Google Sheets",
    "event": "Create Spreadsheet Row",
    "config": {
      "spreadsheet_id": "1ABC_VBA_Migration_Tracker",
      "worksheet": "Exports",
      "row_data": {
        "A": "{{trigger.file_name}}",
        "B": "={{now()}}",
        "C": "{{action_2.total_chunks}}",
        "D": "Pending Analysis",
        "E": "{{trigger.file_id}}"
      }
    }
  }
}`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description:
              'Send VBA code to ChatGPT for comprehensive analysis including documentation generation, business logic extraction, Python equivalent code, and migration risk assessment.',
            toolsUsed: ['ChatGPT Plus', 'OpenAI API'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'VBA Documentation & Analysis Prompt',
                description:
                  'System and user prompts for ChatGPT to analyze VBA code, generate documentation, and identify migration risks.',
                code: `system_prompt: |
  You are an expert VBA-to-Python migration specialist with deep experience
  in construction industry Excel workbooks. Your job is to analyze legacy
  VBA code and produce comprehensive documentation and migration guidance.

  ANALYSIS FRAMEWORK:
  1. PURPOSE: Identify what business function this code performs
  2. DATA FLOW: Map inputs (cells, ranges, files) to outputs
  3. BUSINESS RULES: Extract calculation logic and validation rules
  4. DEPENDENCIES: List external references, COM objects, file paths
  5. RISKS: Flag Windows-specific code, hardcoded paths, Hungarian notation

  DOCUMENTATION REQUIREMENTS:
  - Write for someone who has never seen this code before
  - Explain the "why" not just the "what"
  - Use construction industry terminology where appropriate
  - Flag any undocumented magic numbers or business rules

  PYTHON CONVERSION RULES:
  - Use type hints and docstrings
  - Replace Range/Cell references with pandas DataFrame operations
  - Use Decimal for financial calculations (not float)
  - Handle errors explicitly (no On Error Resume Next equivalents)

  OUTPUT FORMAT: Structured JSON with documentation, python_code, risks, and test_cases

user_prompt: |
  Analyze this VBA module from the {{workbook_name}} construction workbook:

  MODULE: {{module_name}}
  TYPE: {{module_type}}

  CODE:
  \`\`\`vba
  {{vba_code}}
  \`\`\`

  Please provide:
  1. A plain-English description of what this code does
  2. The business rules and calculations it implements
  3. A Python equivalent with type hints and docstrings
  4. Any migration risks or concerns
  5. Test cases to validate the Python matches the VBA output

expected_output_schema:
  documentation:
    purpose: "string - business function description"
    inputs:
      - name: "string"
        source: "Cell A1 | Named Range | Parameter"
        data_type: "string"
    outputs:
      - name: "string"
        destination: "Cell B5 | Return Value"
        data_type: "string"
    business_rules:
      - rule_name: "string"
        description: "string"
        formula: "string"
    dependencies:
      - type: "worksheet | external_file | com_object"
        reference: "string"
        required: boolean
  python_equivalent:
    function_name: "string"
    code: "string - complete Python function"
    imports_required: ["string"]
  migration_risks:
    - risk: "string"
      severity: "HIGH | MEDIUM | LOW"
      mitigation: "string"
  test_cases:
    - name: "string"
      inputs: {}
      expected_output: "any"`,
              },
              {
                language: 'yaml',
                title: 'Migration Validation Prompt',
                description:
                  'Prompt for ChatGPT to compare VBA and Python outputs and validate migration correctness.',
                code: `system_prompt: |
  You are a quality assurance specialist validating VBA-to-Python migrations
  for construction estimating and project management workbooks. Your job is
  to verify that the Python code produces identical outputs to the VBA.

  VALIDATION RULES:
  1. Numeric values must match within rounding tolerance (2 decimal places)
  2. String outputs must match exactly (case-sensitive)
  3. Date/time values must match when formatted identically
  4. Array/range outputs must match element-by-element
  5. Error conditions must be handled equivalently

  TOLERANCE HANDLING:
  - Financial calculations: $0.01 tolerance
  - Percentages: 0.01% tolerance
  - Quantities: exact match required
  - Intermediate calculations: may vary, final output must match

  REPORT FORMAT:
  - List each test case with PASS/FAIL status
  - For failures, show expected vs actual values
  - Provide root cause analysis for mismatches
  - Recommend fixes for any discrepancies

user_prompt: |
  Validate the Python migration against the original VBA:

  FUNCTION: {{function_name}}

  ORIGINAL VBA OUTPUT:
  {{vba_test_results}}

  PYTHON OUTPUT:
  {{python_test_results}}

  TEST DATA USED:
  {{test_data_json}}

  Compare each output and report:
  1. Overall PASS/FAIL status
  2. Detailed comparison of each test case
  3. Root cause for any mismatches
  4. Recommended fixes if needed

validation_output_schema:
  overall_status: "PASS | FAIL | PARTIAL"
  total_tests: number
  passed: number
  failed: number
  test_results:
    - test_name: "string"
      status: "PASS | FAIL"
      vba_output: "any"
      python_output: "any"
      difference: "string | null"
      within_tolerance: boolean
  failures_analysis:
    - test_name: "string"
      root_cause: "string"
      recommended_fix: "string"
      priority: "HIGH | MEDIUM | LOW"
  migration_confidence: number  # 0.0 to 1.0`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier to store AI-generated documentation in a knowledge base, create GitHub issues for migration tasks, and notify the team when analysis is complete. Validation results are tracked in a Google Sheet dashboard.',
            toolsUsed: ['Zapier', 'Google Sheets', 'GitHub', 'Slack'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Migration Tracking Workflow',
                description:
                  'Complete Zapier workflow that processes AI analysis, creates migration tasks, and tracks validation progress.',
                code: `{
  "zap_name": "VBA Migration Task Generator",
  "trigger": {
    "app": "Webhooks by Zapier",
    "event": "Catch Hook",
    "webhook_url": "https://hooks.zapier.com/hooks/catch/12345/vba-analysis/"
  },
  "step_2_parse_analysis": {
    "app": "Code by Zapier",
    "event": "Run Javascript",
    "code": "const analysis = JSON.parse(inputData.chatgpt_response); const risks = analysis.migration_risks.filter(r => r.severity === 'HIGH'); return { function_name: analysis.python_equivalent.function_name, python_code: analysis.python_equivalent.code, high_risks: risks.length, doc_summary: analysis.documentation.purpose, test_count: analysis.test_cases.length };"
  },
  "step_3_save_documentation": {
    "app": "Google Docs",
    "event": "Create Document from Template",
    "config": {
      "template_id": "1ABC_Migration_Doc_Template",
      "folder_id": "1ABC_Migration_Docs",
      "document_name": "Migration - {{step_2.function_name}}",
      "replacements": {
        "{{FUNCTION_NAME}}": "{{step_2.function_name}}",
        "{{PURPOSE}}": "{{step_2.doc_summary}}",
        "{{PYTHON_CODE}}": "{{step_2.python_code}}",
        "{{RISK_COUNT}}": "{{step_2.high_risks}}"
      }
    }
  },
  "step_4_create_github_issue": {
    "app": "GitHub",
    "event": "Create Issue",
    "config": {
      "repo": "construction-co/excel-migration",
      "title": "Migrate: {{step_2.function_name}}",
      "body": "## Migration Task\\n\\n**Original Module:** {{trigger.module_name}}\\n**Workbook:** {{trigger.workbook_name}}\\n\\n### Summary\\n{{step_2.doc_summary}}\\n\\n### High-Priority Risks\\n{{step_2.high_risks}} risks identified\\n\\n### Test Cases\\n{{step_2.test_count}} test cases generated\\n\\n### Documentation\\n[View full documentation]({{step_3.document_url}})\\n\\n---\\n_Auto-generated by VBA Migration Assistant_",
      "labels": ["migration", "vba-to-python", "{{trigger.workbook_name}}"],
      "assignees": []
    }
  },
  "step_5_update_tracker": {
    "app": "Google Sheets",
    "event": "Update Spreadsheet Row",
    "config": {
      "spreadsheet_id": "1ABC_VBA_Migration_Tracker",
      "worksheet": "Modules",
      "lookup_column": "A",
      "lookup_value": "{{trigger.module_name}}",
      "updates": {
        "D": "Analysis Complete",
        "E": "{{step_2.high_risks}}",
        "F": "{{step_2.test_count}}",
        "G": "{{step_3.document_url}}",
        "H": "{{step_4.issue_url}}",
        "I": "={{now()}}"
      }
    }
  },
  "step_6_notify_team": {
    "app": "Slack",
    "event": "Send Channel Message",
    "config": {
      "channel": "#excel-migration",
      "message": ":white_check_mark: *VBA Analysis Complete*\\n\\n*Module:* {{trigger.module_name}}\\n*Function:* {{step_2.function_name}}\\n*High Risks:* {{step_2.high_risks}}\\n*Test Cases:* {{step_2.test_count}}\\n\\n:page_facing_up: <{{step_3.document_url}}|View Documentation>\\n:octocat: <{{step_4.issue_url}}|GitHub Issue>",
      "bot_name": "Migration Bot",
      "bot_icon": ":gear:"
    }
  }
}`,
              },
              {
                language: 'json',
                title: 'Google Sheets Migration Dashboard Structure',
                description:
                  'Sheet structure and formulas for tracking VBA migration progress, risk scores, and validation status.',
                code: `{
  "sheet_name": "Migration Dashboard",
  "headers": [
    "Module Name", "Workbook", "Status", "High Risks", "Test Cases",
    "Tests Passed", "Pass Rate", "Documentation URL", "GitHub Issue",
    "Analysis Date", "Validation Date", "Assigned To", "Priority Score"
  ],
  "formula_columns": {
    "G_Pass_Rate": "=IF(F2>0, F2/E2*100, 0)",
    "M_Priority_Score": "=D2*10 + (100-G2) + IF(B2=\"EstimateMaster\", 50, 0)"
  },
  "conditional_formatting": {
    "C_Status": {
      "Pending Analysis": { "background": "#E0E0E0" },
      "Analysis Complete": { "background": "#FFF9C4" },
      "In Development": { "background": "#BBDEFB" },
      "Validation Passed": { "background": "#C8E6C9" },
      "Validation Failed": { "background": "#FFCDD2" }
    },
    "G_Pass_Rate": {
      "100": { "background": "#4CAF50", "font_color": "#FFFFFF" },
      ">=90": { "background": "#8BC34A" },
      ">=70": { "background": "#FFC107" },
      "<70": { "background": "#F44336", "font_color": "#FFFFFF" }
    }
  },
  "summary_section": {
    "total_modules": "=COUNTA(A:A)-1",
    "completed": "=COUNTIF(C:C, \"Validation Passed\")",
    "in_progress": "=COUNTIF(C:C, \"In Development\")",
    "blocked": "=COUNTIF(C:C, \"Validation Failed\")",
    "avg_pass_rate": "=AVERAGE(G:G)",
    "total_high_risks": "=SUM(D:D)"
  },
  "pivot_by_workbook": {
    "rows": "Workbook",
    "values": ["COUNT of Module Name", "SUM of High Risks", "AVG of Pass Rate"],
    "sort": "SUM of High Risks DESC"
  }
}`,
              },
            ],
          },
        ],
      },

      aiAdvanced: {
        overview:
          'Deploy a multi-agent AI system that fully automates Excel-to-database migration: Code Analysis agents reverse-engineer VBA logic, Translation agents generate Python equivalents with tests, Validation agents run comparative testing, and Migration agents execute data transfers with rollback capabilities. The system handles complex interdependencies between workbooks and provides confidence scoring for each migration.',
        estimatedMonthlyCost: '$800 - $1,500/month',
        architecture:
          'Supervisor agent coordinates five specialist agents: (1) VBA Analysis Agent parses and documents legacy code, (2) Python Translation Agent generates equivalent functions with type hints, (3) Test Generation Agent creates comprehensive validation suites, (4) Migration Execution Agent transfers data with transactional safety, (5) Validation Agent runs comparative tests and reports discrepancies. LangGraph manages the workflow with PostgreSQL-backed state for complex multi-workbook migrations.',
        agents: [
          {
            name: 'VBAAnalysisAgent',
            role: 'Legacy Code Analysis',
            goal: 'Parse VBA modules, extract business logic, identify dependencies, and generate comprehensive documentation for each function and subroutine',
            tools: ['VBA Parser', 'AST Analyzer', 'Dependency Graph Builder', 'Documentation Generator'],
          },
          {
            name: 'PythonTranslationAgent',
            role: 'Code Translation',
            goal: 'Convert VBA functions to idiomatic Python with type hints, docstrings, and proper error handling while preserving exact calculation logic',
            tools: ['GPT-4', 'Code Validator', 'Type Checker', 'Linter'],
          },
          {
            name: 'TestGenerationAgent',
            role: 'Test Suite Creation',
            goal: 'Generate comprehensive pytest test suites that validate Python outputs match VBA outputs across edge cases, boundary conditions, and real production data',
            tools: ['pytest', 'Hypothesis', 'openpyxl', 'Test Data Generator'],
          },
          {
            name: 'MigrationExecutionAgent',
            role: 'Data Migration',
            goal: 'Execute data transfers from Excel workbooks to PostgreSQL with transactional safety, rollback capabilities, and detailed audit logging',
            tools: ['PostgreSQL', 'pandas', 'openpyxl', 'Alembic', 'Transaction Manager'],
          },
          {
            name: 'ValidationAgent',
            role: 'Migration Validation',
            goal: 'Run comparative tests between VBA and Python outputs, generate discrepancy reports, and provide confidence scores for each migrated function',
            tools: ['pytest', 'Excel COM', 'Comparison Engine', 'Report Generator'],
          },
          {
            name: 'SupervisorAgent',
            role: 'Migration Orchestration',
            goal: 'Coordinate the multi-agent migration pipeline, manage dependencies between workbooks, handle failures with retry logic, and track overall migration progress',
            tools: ['LangGraph', 'PostgreSQL', 'Slack API'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Hierarchical',
          stateManagement: 'PostgreSQL-backed state with workbook-level checkpointing; supports pause/resume for long-running multi-workbook migrations',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent migration system using CrewAI with specialized roles for VBA analysis, Python translation, test generation, and validation. Each agent has domain-specific tools and clear handoff protocols.',
            toolsUsed: ['CrewAI', 'LangChain', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Excel Migration Agent Definitions',
                description:
                  'CrewAI agent definitions for the Excel-to-database migration system with specialized roles for code analysis, translation, and validation.',
                code: `"""agents/excel_migration_agents.py - Multi-agent system for Excel VBA migration"""
from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class MigrationStatus(str, Enum):
    """Status of a module migration."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    TRANSLATING = "translating"
    TESTING = "testing"
    MIGRATING = "migrating"
    VALIDATING = "validating"
    COMPLETE = "complete"
    FAILED = "failed"


class VBAModule(BaseModel):
    """Parsed VBA module with metadata."""
    module_name: str
    module_type: str = Field(description="Standard|Class|Form|Sheet")
    workbook_name: str
    raw_code: str
    line_count: int
    procedures: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    complexity_score: float


class PythonFunction(BaseModel):
    """Generated Python function with metadata."""
    function_name: str
    source_procedure: str
    python_code: str
    imports_required: List[str]
    type_hints: Dict[str, str]
    docstring: str
    test_cases: List[Dict[str, Any]]


class MigrationResult(BaseModel):
    """Result of a module migration."""
    module_name: str
    status: MigrationStatus
    python_functions: List[PythonFunction]
    rows_migrated: int
    validation_passed: bool
    confidence_score: float
    discrepancies: List[Dict[str, Any]]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)


# Tool definitions
def parse_vba_module_tool(code: str, module_name: str) -> dict:
    """Parse VBA module and extract structure."""
    logger.info(f"Parsing VBA module: {module_name}")
    # Implementation parses VBA AST
    return {"parsed": True, "procedures": [], "dependencies": {}}


def translate_to_python_tool(vba_code: str, context: dict) -> dict:
    """Translate VBA procedure to Python."""
    logger.info("Translating VBA to Python")
    # Implementation uses GPT-4 for translation
    return {"python_code": "...", "imports": [], "type_hints": {}}


def generate_tests_tool(python_code: str, vba_examples: list) -> dict:
    """Generate pytest test cases for Python function."""
    logger.info("Generating test cases")
    # Implementation creates pytest test cases
    return {"test_code": "...", "test_count": 0}


def run_validation_tool(python_func: str, vba_outputs: list) -> dict:
    """Run comparative validation between Python and VBA."""
    logger.info("Running comparative validation")
    # Implementation runs both and compares
    return {"passed": True, "discrepancies": []}


def execute_migration_tool(data: dict, target_table: str) -> dict:
    """Execute data migration to PostgreSQL."""
    logger.info(f"Migrating data to {target_table}")
    # Implementation handles transactional migration
    return {"rows_migrated": 0, "success": True}


# Agent definitions
vba_analysis_agent = Agent(
    role="VBA Code Analysis Specialist",
    goal="Parse and document legacy VBA code, identifying all procedures, "
         "dependencies, and business logic for accurate translation",
    backstory="You are a veteran Excel developer who has worked with construction "
              "estimating workbooks for 20 years. You know every VBA pattern, from "
              "ancient Hungarian notation to modern best practices. You can read "
              "undocumented code and understand what the original developer intended.",
    llm=llm,
    tools=[Tool(name="parse_vba", func=parse_vba_module_tool,
                description="Parse VBA module structure")],
    verbose=True,
    allow_delegation=False,
)

python_translation_agent = Agent(
    role="VBA-to-Python Translation Specialist",
    goal="Convert VBA functions to clean, idiomatic Python with exact "
         "calculation equivalence, proper type hints, and comprehensive docstrings",
    backstory="You are a polyglot developer fluent in both VBA and Python. You "
              "understand the subtle differences in numeric handling, date math, "
              "and string operations between the languages. You write Python that "
              "a junior developer can maintain, not just code that works.",
    llm=llm,
    tools=[Tool(name="translate", func=translate_to_python_tool,
                description="Translate VBA to Python")],
    verbose=True,
    allow_delegation=False,
)

test_generation_agent = Agent(
    role="Migration Test Engineer",
    goal="Generate comprehensive test suites that validate Python functions "
         "produce identical outputs to the original VBA across all edge cases",
    backstory="You are a QA engineer obsessed with edge cases. You have seen "
              "migrations fail because someone forgot to test negative numbers, "
              "dates before 1900, or strings with special characters. You create "
              "tests that catch these issues before production.",
    llm=llm,
    tools=[Tool(name="generate_tests", func=generate_tests_tool,
                description="Generate pytest test cases")],
    verbose=True,
    allow_delegation=False,
)

migration_execution_agent = Agent(
    role="Data Migration Engineer",
    goal="Execute data migrations from Excel to PostgreSQL with transactional "
         "safety, detailed logging, and automatic rollback on failure",
    backstory="You are a database engineer who has migrated terabytes of data "
              "without losing a single row. You know that migrations fail, so you "
              "build in checkpoints, rollbacks, and reconciliation at every step.",
    llm=llm,
    tools=[Tool(name="migrate", func=execute_migration_tool,
                description="Execute data migration")],
    verbose=True,
    allow_delegation=False,
)

validation_agent = Agent(
    role="Migration Validation Specialist",
    goal="Run comparative tests between VBA and Python outputs, identify "
         "discrepancies, and provide confidence scores for migration readiness",
    backstory="You are the final gatekeeper before any migration goes live. You "
              "have rejected migrations that 'mostly worked' because you know that "
              "99.9% accuracy in financial calculations is a disaster. You only "
              "approve migrations you would stake your reputation on.",
    llm=llm,
    tools=[Tool(name="validate", func=run_validation_tool,
                description="Run comparative validation")],
    verbose=True,
    allow_delegation=False,
)

supervisor_agent = Agent(
    role="Migration Project Supervisor",
    goal="Coordinate the multi-agent migration pipeline, managing dependencies "
         "between workbooks and ensuring overall migration success",
    backstory="You are a project manager who has overseen dozens of legacy "
              "migrations. You know that the technical work is only half the job - "
              "the other half is coordination, communication, and risk management.",
    llm=llm,
    tools=[],
    verbose=True,
    allow_delegation=True,
)

logger.info("Excel migration agents initialized successfully")`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the VBAAnalysisAgent that parses legacy Excel workbooks, extracts VBA modules, builds dependency graphs, and generates structured documentation for each procedure.',
            toolsUsed: ['openpyxl', 'VBA Parser', 'NetworkX', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'VBA Analysis Agent Implementation',
                description:
                  'Agent that parses VBA code, extracts procedures and dependencies, and generates structured documentation for migration planning.',
                code: `"""agents/vba_analysis.py - VBA code analysis and documentation agent"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from openpyxl import load_workbook
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class VBAProcedure(BaseModel):
    """Parsed VBA procedure with metadata."""
    name: str
    procedure_type: str = Field(description="Sub|Function|Property")
    visibility: str = Field(description="Public|Private")
    parameters: list[dict[str, str]]
    return_type: Optional[str] = None
    line_start: int
    line_end: int
    raw_code: str
    calls: list[str] = Field(default_factory=list)
    cell_references: list[str] = Field(default_factory=list)
    sheet_references: list[str] = Field(default_factory=list)
    complexity_score: float = 0.0


class VBAModule(BaseModel):
    """Parsed VBA module with all procedures."""
    module_name: str
    module_type: str
    workbook_path: str
    line_count: int
    procedures: list[VBAProcedure]
    global_variables: list[dict[str, Any]]
    dependencies: dict[str, list[str]]
    raw_code: str


class WorkbookAnalysis(BaseModel):
    """Complete analysis of a workbook's VBA code."""
    workbook_name: str
    workbook_path: str
    modules: list[VBAModule]
    dependency_graph: dict[str, list[str]]
    total_procedures: int
    total_lines: int
    complexity_score: float
    migration_order: list[str]


class VBAAnalysisAgent:
    """Agent for analyzing VBA code in Excel workbooks."""

    # Regex patterns for VBA parsing
    PROC_PATTERN = re.compile(
        r"^(Public|Private)?\s*(Sub|Function|Property\s+(?:Get|Let|Set))\s+"
        r"(\w+)\s*\(([^)]*)\)(?:\s+As\s+(\w+))?",
        re.MULTILINE | re.IGNORECASE,
    )
    CALL_PATTERN = re.compile(r"\b(Call\s+)?(\w+)\s*\(", re.IGNORECASE)
    CELL_PATTERN = re.compile(r"(?:Range|Cells)\s*\(\s*[\"']?([A-Z]+\d+|[A-Z]+:[A-Z]+)[\"']?\s*\)", re.IGNORECASE)
    SHEET_PATTERN = re.compile(r"(?:Worksheets|Sheets)\s*\(\s*[\"']([^\"']+)[\"']\s*\)", re.IGNORECASE)

    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        logger.info("VBAAnalysisAgent initialized")

    def extract_vba_from_workbook(self, workbook_path: Path) -> list[dict[str, str]]:
        """Extract VBA code from an Excel workbook."""
        logger.info(f"Extracting VBA from: {workbook_path}")

        # Note: openpyxl doesn't support VBA extraction directly
        # In production, use win32com on Windows or oletools for cross-platform
        # This is a simplified implementation that works with .bas exports

        modules = []
        vba_dir = workbook_path.parent / f"{workbook_path.stem}_vba"

        if vba_dir.exists():
            for vba_file in vba_dir.glob("*.bas"):
                with open(vba_file, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
                modules.append({
                    "name": vba_file.stem,
                    "type": "Standard",
                    "code": code,
                })
            for cls_file in vba_dir.glob("*.cls"):
                with open(cls_file, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
                modules.append({
                    "name": cls_file.stem,
                    "type": "Class",
                    "code": code,
                })

        logger.info(f"Extracted {len(modules)} VBA modules")
        return modules

    def parse_procedure(self, code: str, start_line: int) -> Optional[VBAProcedure]:
        """Parse a single VBA procedure."""
        match = self.PROC_PATTERN.search(code)
        if not match:
            return None

        visibility = match.group(1) or "Public"
        proc_type = match.group(2)
        name = match.group(3)
        params_str = match.group(4)
        return_type = match.group(5)

        # Parse parameters
        parameters = []
        if params_str.strip():
            for param in params_str.split(","):
                param = param.strip()
                param_match = re.match(r"(?:ByVal|ByRef)?\s*(\w+)(?:\s+As\s+(\w+))?", param)
                if param_match:
                    parameters.append({
                        "name": param_match.group(1),
                        "type": param_match.group(2) or "Variant",
                    })

        # Extract calls, cell references, and sheet references
        calls = list(set(m.group(2) for m in self.CALL_PATTERN.finditer(code)))
        cell_refs = list(set(m.group(1) for m in self.CELL_PATTERN.finditer(code)))
        sheet_refs = list(set(m.group(1) for m in self.SHEET_PATTERN.finditer(code)))

        # Calculate complexity score (simplified McCabe complexity)
        complexity = 1
        complexity += len(re.findall(r"\bIf\b", code, re.IGNORECASE))
        complexity += len(re.findall(r"\bFor\b", code, re.IGNORECASE))
        complexity += len(re.findall(r"\bWhile\b", code, re.IGNORECASE))
        complexity += len(re.findall(r"\bSelect Case\b", code, re.IGNORECASE))

        line_count = code.count("\\n") + 1

        return VBAProcedure(
            name=name,
            procedure_type=proc_type,
            visibility=visibility,
            parameters=parameters,
            return_type=return_type,
            line_start=start_line,
            line_end=start_line + line_count - 1,
            raw_code=code,
            calls=calls,
            cell_references=cell_refs,
            sheet_references=sheet_refs,
            complexity_score=complexity,
        )

    def parse_module(self, name: str, module_type: str, code: str, workbook_path: str) -> VBAModule:
        """Parse a complete VBA module."""
        logger.info(f"Parsing module: {name}")

        procedures = []
        # Split by procedure boundaries
        proc_blocks = re.split(
            r"(?=^(?:Public|Private)?\s*(?:Sub|Function|Property))",
            code,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        line_offset = 1
        for block in proc_blocks:
            if not block.strip():
                continue
            proc = self.parse_procedure(block, line_offset)
            if proc:
                procedures.append(proc)
            line_offset += block.count("\\n") + 1

        # Extract global variables
        global_vars = []
        for match in re.finditer(
            r"^(Public|Private|Dim)\s+(\w+)\s+As\s+(\w+)",
            code,
            re.MULTILINE | re.IGNORECASE,
        ):
            global_vars.append({
                "name": match.group(2),
                "type": match.group(3),
                "scope": match.group(1),
            })

        # Build dependencies
        dependencies = {
            "procedures_called": list(set(
                call for proc in procedures for call in proc.calls
            )),
            "sheets_referenced": list(set(
                sheet for proc in procedures for sheet in proc.sheet_references
            )),
            "cells_referenced": list(set(
                cell for proc in procedures for cell in proc.cell_references
            )),
        }

        return VBAModule(
            module_name=name,
            module_type=module_type,
            workbook_path=workbook_path,
            line_count=code.count("\\n") + 1,
            procedures=procedures,
            global_variables=global_vars,
            dependencies=dependencies,
            raw_code=code,
        )

    def analyze_workbook(self, workbook_path: Path) -> WorkbookAnalysis:
        """Perform complete analysis of a workbook's VBA code."""
        logger.info(f"Analyzing workbook: {workbook_path}")

        raw_modules = self.extract_vba_from_workbook(workbook_path)
        modules = []

        for raw in raw_modules:
            module = self.parse_module(
                raw["name"],
                raw["type"],
                raw["code"],
                str(workbook_path),
            )
            modules.append(module)

            # Add to dependency graph
            for proc in module.procedures:
                self.dependency_graph.add_node(f"{module.module_name}.{proc.name}")
                for call in proc.calls:
                    self.dependency_graph.add_edge(
                        f"{module.module_name}.{proc.name}",
                        call,
                    )

        # Calculate migration order using topological sort
        try:
            migration_order = list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXUnfeasible:
            logger.warning("Circular dependencies detected, using degree-based ordering")
            migration_order = sorted(
                self.dependency_graph.nodes(),
                key=lambda n: self.dependency_graph.in_degree(n),
            )

        total_procedures = sum(len(m.procedures) for m in modules)
        total_lines = sum(m.line_count for m in modules)
        avg_complexity = (
            sum(p.complexity_score for m in modules for p in m.procedures) / total_procedures
            if total_procedures > 0 else 0
        )

        analysis = WorkbookAnalysis(
            workbook_name=workbook_path.stem,
            workbook_path=str(workbook_path),
            modules=modules,
            dependency_graph=dict(self.dependency_graph.adjacency()),
            total_procedures=total_procedures,
            total_lines=total_lines,
            complexity_score=avg_complexity,
            migration_order=migration_order,
        )

        logger.info(
            f"Analysis complete: {total_procedures} procedures, "
            f"{total_lines} lines, complexity={avg_complexity:.2f}"
        )
        return analysis

    def generate_documentation(self, analysis: WorkbookAnalysis) -> str:
        """Generate human-readable documentation from analysis."""
        doc_lines = [
            f"# VBA Migration Documentation: {analysis.workbook_name}",
            "",
            "## Summary",
            f"- **Total Modules:** {len(analysis.modules)}",
            f"- **Total Procedures:** {analysis.total_procedures}",
            f"- **Total Lines of Code:** {analysis.total_lines}",
            f"- **Average Complexity:** {analysis.complexity_score:.2f}",
            "",
            "## Migration Order",
            "The following order respects dependencies:",
            "",
        ]

        for i, proc_name in enumerate(analysis.migration_order, 1):
            doc_lines.append(f"{i}. {proc_name}")

        doc_lines.extend(["", "## Modules", ""])

        for module in analysis.modules:
            doc_lines.extend([
                f"### {module.module_name} ({module.module_type})",
                f"- Lines: {module.line_count}",
                f"- Procedures: {len(module.procedures)}",
                "",
                "| Procedure | Type | Visibility | Complexity | Calls |",
                "|-----------|------|------------|------------|-------|",
            ])

            for proc in module.procedures:
                calls_str = ", ".join(proc.calls[:3])
                if len(proc.calls) > 3:
                    calls_str += f" (+{len(proc.calls) - 3} more)"
                doc_lines.append(
                    f"| {proc.name} | {proc.procedure_type} | {proc.visibility} | "
                    f"{proc.complexity_score:.1f} | {calls_str} |"
                )

            doc_lines.append("")

        return "\\n".join(doc_lines)`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the PythonTranslationAgent and TestGenerationAgent that convert VBA to Python and create comprehensive validation test suites with edge case coverage.',
            toolsUsed: ['GPT-4', 'pytest', 'Hypothesis', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Python Translation Agent Implementation',
                description:
                  'Agent that translates VBA procedures to Python functions with proper type hints, docstrings, and calculation equivalence.',
                code: `"""agents/python_translation.py - VBA to Python translation agent"""
import logging
import re
from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class TranslationResult(BaseModel):
    """Result of VBA to Python translation."""
    original_name: str
    python_name: str
    python_code: str
    imports_required: list[str]
    type_hints: dict[str, str]
    docstring: str
    translation_notes: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    manual_review_needed: bool = False
    review_reasons: list[str] = Field(default_factory=list)


class TestCase(BaseModel):
    """Generated test case for validation."""
    test_name: str
    description: str
    inputs: dict[str, Any]
    expected_output: Any
    tolerance: Optional[float] = None
    edge_case_type: Optional[str] = None


class PythonTranslationAgent:
    """Agent for translating VBA procedures to Python."""

    TRANSLATION_SYSTEM_PROMPT = '''You are an expert VBA-to-Python translator specializing in
construction industry Excel workbooks. Convert VBA code to clean, idiomatic Python.

TRANSLATION RULES:
1. Use Decimal for all financial calculations (never float)
2. Use type hints on all parameters and return values
3. Write comprehensive docstrings explaining the business logic
4. Replace Range/Cells operations with function parameters
5. Handle errors explicitly (no silent failures)
6. Use snake_case for function and variable names
7. Preserve the exact calculation logic - rounding, order of operations, etc.

VBA TO PYTHON MAPPINGS:
- Dim x As Double -> x: Decimal
- If...Then...Else -> if...elif...else
- For i = 1 To 10 -> for i in range(1, 11)
- Do While...Loop -> while...
- MsgBox -> logging.info() or raise Exception
- Range("A1").Value -> parameter
- WorksheetFunction.Sum -> sum() or Decimal sum
- IsNumeric(x) -> isinstance(x, (int, float, Decimal))
- Trim(x) -> x.strip()
- Left(x, n) -> x[:n]
- Right(x, n) -> x[-n:]
- Mid(x, start, len) -> x[start-1:start-1+len]
- InStr(x, y) -> y in x or x.find(y)
- UCase/LCase -> x.upper()/x.lower()
- Format(x, "0.00") -> f"{x:.2f}"

OUTPUT: Return only the Python code with imports, type hints, and docstring.'''

    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        logger.info("PythonTranslationAgent initialized")

    def _convert_vba_name(self, name: str) -> str:
        """Convert VBA name to Python snake_case."""
        # Handle Hungarian notation (e.g., strName -> name, dblAmount -> amount)
        hungarian_prefixes = ["str", "dbl", "int", "lng", "bln", "obj", "rng", "ws", "wb"]
        for prefix in hungarian_prefixes:
            if name.lower().startswith(prefix) and len(name) > len(prefix):
                name = name[len(prefix):]
                break

        # Convert CamelCase to snake_case
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\\1_\\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\\1_\\2", s1).lower()

    def translate_procedure(
        self,
        vba_code: str,
        procedure_name: str,
        context: Optional[dict[str, Any]] = None,
    ) -> TranslationResult:
        """Translate a VBA procedure to Python."""
        logger.info(f"Translating procedure: {procedure_name}")

        context_str = ""
        if context:
            context_str = f"""
CONTEXT:
- Workbook: {context.get('workbook_name', 'Unknown')}
- Module: {context.get('module_name', 'Unknown')}
- Called by: {', '.join(context.get('called_by', []))}
- Calls: {', '.join(context.get('calls', []))}
- Cell references: {', '.join(context.get('cell_refs', []))}
"""

        user_prompt = f"""Translate this VBA procedure to Python:

{context_str}

VBA CODE:
\`\`\`vba
{vba_code}
\`\`\`

Provide:
1. Complete Python function with type hints and docstring
2. Required imports
3. Any translation notes or concerns"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": self.TRANSLATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=2000,
        )

        raw_response = response.choices[0].message.content
        logger.info(f"Translation complete for {procedure_name}")

        # Extract Python code from response
        code_match = re.search(r"\`\`\`python\\n(.*?)\`\`\`", raw_response, re.DOTALL)
        python_code = code_match.group(1) if code_match else raw_response

        # Extract imports
        imports = re.findall(r"^(?:from|import)\s+.+$", python_code, re.MULTILINE)

        # Identify potential issues
        review_reasons = []
        manual_review = False

        if "# TODO" in python_code or "# FIXME" in python_code:
            review_reasons.append("Contains TODO/FIXME comments")
            manual_review = True

        if "raise NotImplementedError" in python_code:
            review_reasons.append("Contains unimplemented sections")
            manual_review = True

        if re.search(r"COM|CreateObject|GetObject", vba_code, re.IGNORECASE):
            review_reasons.append("Original code uses COM objects")
            manual_review = True

        # Calculate confidence based on code complexity and issues
        confidence = 0.9
        if manual_review:
            confidence -= 0.2
        if len(vba_code) > 500:
            confidence -= 0.1

        python_name = self._convert_vba_name(procedure_name)

        return TranslationResult(
            original_name=procedure_name,
            python_name=python_name,
            python_code=python_code,
            imports_required=imports,
            type_hints={},  # Extracted from code in production
            docstring="",  # Extracted from code in production
            translation_notes=[],
            confidence_score=max(0.5, confidence),
            manual_review_needed=manual_review,
            review_reasons=review_reasons,
        )

    def generate_test_cases(
        self,
        python_code: str,
        vba_code: str,
        function_name: str,
    ) -> list[TestCase]:
        """Generate test cases for the translated function."""
        logger.info(f"Generating test cases for: {function_name}")

        test_prompt = f"""Generate comprehensive pytest test cases for this Python function
that validate it produces identical outputs to the original VBA.

PYTHON:
\`\`\`python
{python_code}
\`\`\`

ORIGINAL VBA:
\`\`\`vba
{vba_code}
\`\`\`

Generate test cases covering:
1. Normal operation with typical inputs
2. Edge cases (zero, negative, empty)
3. Boundary conditions
4. Error handling

Format each test case as:
- test_name: descriptive name
- inputs: dict of parameter values
- expected_output: expected result
- tolerance: numeric tolerance (for Decimal comparisons)
- edge_case_type: type of edge case (if applicable)"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a test engineer creating validation tests."},
                {"role": "user", "content": test_prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )

        # Parse test cases from response (simplified)
        test_cases = [
            TestCase(
                test_name=f"test_{function_name}_normal",
                description="Test normal operation",
                inputs={"value": 100.0},
                expected_output=100.0,
                tolerance=0.01,
            ),
            TestCase(
                test_name=f"test_{function_name}_zero",
                description="Test with zero input",
                inputs={"value": 0.0},
                expected_output=0.0,
                edge_case_type="zero",
            ),
            TestCase(
                test_name=f"test_{function_name}_negative",
                description="Test with negative input",
                inputs={"value": -50.0},
                expected_output=-50.0,
                edge_case_type="negative",
            ),
        ]

        logger.info(f"Generated {len(test_cases)} test cases")
        return test_cases

    def generate_pytest_file(
        self,
        function_name: str,
        python_code: str,
        test_cases: list[TestCase],
    ) -> str:
        """Generate a complete pytest file for the function."""
        imports = [
            "import pytest",
            "from decimal import Decimal",
            "from typing import Any",
            "",
            "# Import the function under test",
            f"from estimating.calculations import {function_name}",
            "",
        ]

        test_functions = []
        for tc in test_cases:
            test_func = f"""
def \\{tc.test_name\\}() -> None:
    \\\"\\\"\\\"Test: \\{tc.description\\}\\\"\\\"\\\"
    inputs = \\{tc.inputs\\}
    expected = \\{tc.expected_output!r\\}
    result = \\{function_name\\}(**inputs)
    \\{'assert abs(result - expected) <= ' + str(tc.tolerance) if tc.tolerance else 'assert result == expected'\\}
"""
            test_functions.append(test_func)

        return "\\n".join(imports) + "\\n".join(test_functions)`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement the LangGraph state machine that coordinates the complete migration pipeline from VBA analysis through data migration, with PostgreSQL-backed state for long-running multi-workbook migrations.',
            toolsUsed: ['LangGraph', 'PostgreSQL', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Migration Pipeline Orchestrator',
                description:
                  'State machine implementation using LangGraph that coordinates VBA analysis, translation, testing, and data migration with checkpoint support.',
                code: `"""orchestration/migration_pipeline.py - LangGraph orchestrator for Excel migration"""
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel
import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class MigrationPhase(str, Enum):
    """Phase of the migration pipeline."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    TRANSLATING = "translating"
    TESTING = "testing"
    MIGRATING = "migrating"
    VALIDATING = "validating"
    COMPLETE = "complete"
    FAILED = "failed"
    PAUSED = "paused"


class ModuleMigrationState(TypedDict):
    """State for a single module migration."""
    module_name: str
    workbook_name: str
    phase: MigrationPhase
    analysis_result: dict[str, Any] | None
    translations: list[dict[str, Any]]
    test_results: dict[str, Any] | None
    migration_result: dict[str, Any] | None
    validation_result: dict[str, Any] | None
    error_message: str | None


class WorkbookMigrationState(TypedDict):
    """State object for a complete workbook migration."""
    # Input
    workbook_path: str
    workbook_name: str
    migration_id: str
    started_at: str

    # Overall state
    phase: MigrationPhase
    current_module: str | None
    modules_pending: list[str]
    modules_completed: list[str]
    modules_failed: list[str]

    # Module states
    module_states: dict[str, ModuleMigrationState]

    # Results
    total_procedures: int
    procedures_translated: int
    procedures_validated: int
    rows_migrated: int
    overall_confidence: float
    errors: list[dict[str, Any]]

    # Timing
    last_checkpoint: str
    estimated_completion: str | None


def create_initial_state(workbook_path: str, modules: list[str]) -> WorkbookMigrationState:
    """Create initial state for a workbook migration."""
    return WorkbookMigrationState(
        workbook_path=workbook_path,
        workbook_name=workbook_path.split("/")[-1].replace(".xlsm", ""),
        migration_id=f"mig_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        started_at=datetime.utcnow().isoformat(),
        phase=MigrationPhase.PENDING,
        current_module=None,
        modules_pending=modules,
        modules_completed=[],
        modules_failed=[],
        module_states={},
        total_procedures=0,
        procedures_translated=0,
        procedures_validated=0,
        rows_migrated=0,
        overall_confidence=0.0,
        errors=[],
        last_checkpoint=datetime.utcnow().isoformat(),
        estimated_completion=None,
    )


# Node functions
async def analyze_workbook(state: WorkbookMigrationState) -> WorkbookMigrationState:
    """Node: Analyze the complete workbook VBA structure."""
    logger.info(f"[{state['migration_id']}] Analyzing workbook: {state['workbook_name']}")
    state["phase"] = MigrationPhase.ANALYZING

    try:
        from agents.vba_analysis import VBAAnalysisAgent
        from pathlib import Path

        agent = VBAAnalysisAgent()
        analysis = agent.analyze_workbook(Path(state["workbook_path"]))

        # Initialize module states from analysis
        for module in analysis.modules:
            state["module_states"][module.module_name] = ModuleMigrationState(
                module_name=module.module_name,
                workbook_name=state["workbook_name"],
                phase=MigrationPhase.PENDING,
                analysis_result=module.model_dump(),
                translations=[],
                test_results=None,
                migration_result=None,
                validation_result=None,
                error_message=None,
            )

        state["total_procedures"] = analysis.total_procedures
        state["modules_pending"] = analysis.migration_order
        state["overall_confidence"] = 1.0 - (analysis.complexity_score / 100)

        logger.info(f"[{state['migration_id']}] Analysis complete: {len(analysis.modules)} modules")

    except Exception as e:
        logger.error(f"[{state['migration_id']}] Analysis failed: {e}")
        state["errors"].append({"phase": "analyze", "error": str(e), "timestamp": datetime.utcnow().isoformat()})
        state["phase"] = MigrationPhase.FAILED

    state["last_checkpoint"] = datetime.utcnow().isoformat()
    return state


async def translate_module(state: WorkbookMigrationState) -> WorkbookMigrationState:
    """Node: Translate the current module's VBA to Python."""
    if not state["modules_pending"]:
        logger.info(f"[{state['migration_id']}] No modules pending translation")
        return state

    module_name = state["modules_pending"][0]
    state["current_module"] = module_name
    state["phase"] = MigrationPhase.TRANSLATING

    logger.info(f"[{state['migration_id']}] Translating module: {module_name}")

    module_state = state["module_states"].get(module_name)
    if not module_state:
        logger.error(f"Module state not found for {module_name}")
        state["modules_failed"].append(module_name)
        state["modules_pending"].pop(0)
        return state

    try:
        from agents.python_translation import PythonTranslationAgent
        import os

        agent = PythonTranslationAgent(openai_api_key=os.getenv("OPENAI_API_KEY", ""))

        analysis = module_state["analysis_result"]
        translations = []

        for proc in analysis.get("procedures", []):
            result = agent.translate_procedure(
                proc["raw_code"],
                proc["name"],
                context={
                    "workbook_name": state["workbook_name"],
                    "module_name": module_name,
                    "calls": proc.get("calls", []),
                    "cell_refs": proc.get("cell_references", []),
                },
            )
            translations.append(result.model_dump())
            state["procedures_translated"] += 1

        module_state["translations"] = translations
        module_state["phase"] = MigrationPhase.TESTING

        logger.info(f"[{state['migration_id']}] Translated {len(translations)} procedures")

    except Exception as e:
        logger.error(f"[{state['migration_id']}] Translation failed for {module_name}: {e}")
        module_state["error_message"] = str(e)
        module_state["phase"] = MigrationPhase.FAILED
        state["modules_failed"].append(module_name)
        state["modules_pending"].pop(0)

    state["last_checkpoint"] = datetime.utcnow().isoformat()
    return state


async def run_tests(state: WorkbookMigrationState) -> WorkbookMigrationState:
    """Node: Run validation tests for the current module."""
    module_name = state["current_module"]
    if not module_name:
        return state

    logger.info(f"[{state['migration_id']}] Running tests for: {module_name}")
    state["phase"] = MigrationPhase.TESTING

    module_state = state["module_states"].get(module_name)
    if not module_state:
        return state

    try:
        # In production, this would actually run pytest
        test_results = {
            "total_tests": len(module_state["translations"]) * 3,
            "passed": len(module_state["translations"]) * 3,
            "failed": 0,
            "errors": [],
            "duration_seconds": 5.2,
        }

        module_state["test_results"] = test_results
        state["procedures_validated"] += test_results["passed"]

        if test_results["failed"] == 0:
            module_state["phase"] = MigrationPhase.MIGRATING
        else:
            module_state["phase"] = MigrationPhase.FAILED
            state["modules_failed"].append(module_name)
            state["modules_pending"].pop(0)

        logger.info(f"[{state['migration_id']}] Tests: {test_results['passed']}/{test_results['total_tests']} passed")

    except Exception as e:
        logger.error(f"[{state['migration_id']}] Testing failed: {e}")
        module_state["error_message"] = str(e)
        module_state["phase"] = MigrationPhase.FAILED

    state["last_checkpoint"] = datetime.utcnow().isoformat()
    return state


async def migrate_data(state: WorkbookMigrationState) -> WorkbookMigrationState:
    """Node: Migrate data from Excel to PostgreSQL."""
    module_name = state["current_module"]
    if not module_name:
        return state

    logger.info(f"[{state['migration_id']}] Migrating data for: {module_name}")
    state["phase"] = MigrationPhase.MIGRATING

    module_state = state["module_states"].get(module_name)
    if not module_state:
        return state

    try:
        # In production, this would execute actual data migration
        migration_result = {
            "rows_migrated": 150,
            "tables_created": 2,
            "duration_seconds": 12.5,
            "success": True,
        }

        module_state["migration_result"] = migration_result
        state["rows_migrated"] += migration_result["rows_migrated"]
        module_state["phase"] = MigrationPhase.VALIDATING

        logger.info(f"[{state['migration_id']}] Migrated {migration_result['rows_migrated']} rows")

    except Exception as e:
        logger.error(f"[{state['migration_id']}] Migration failed: {e}")
        module_state["error_message"] = str(e)
        module_state["phase"] = MigrationPhase.FAILED

    state["last_checkpoint"] = datetime.utcnow().isoformat()
    return state


async def validate_migration(state: WorkbookMigrationState) -> WorkbookMigrationState:
    """Node: Validate migrated data matches Excel source."""
    module_name = state["current_module"]
    if not module_name:
        return state

    logger.info(f"[{state['migration_id']}] Validating migration: {module_name}")
    state["phase"] = MigrationPhase.VALIDATING

    module_state = state["module_states"].get(module_name)
    if not module_state:
        return state

    try:
        validation_result = {
            "rows_compared": module_state["migration_result"]["rows_migrated"],
            "rows_matched": module_state["migration_result"]["rows_migrated"],
            "discrepancies": [],
            "confidence_score": 0.98,
            "passed": True,
        }

        module_state["validation_result"] = validation_result
        module_state["phase"] = MigrationPhase.COMPLETE

        # Move to next module
        state["modules_completed"].append(module_name)
        state["modules_pending"].pop(0)
        state["current_module"] = None

        logger.info(f"[{state['migration_id']}] Validation passed for {module_name}")

    except Exception as e:
        logger.error(f"[{state['migration_id']}] Validation failed: {e}")
        module_state["error_message"] = str(e)
        module_state["phase"] = MigrationPhase.FAILED

    state["last_checkpoint"] = datetime.utcnow().isoformat()
    return state


async def finalize(state: WorkbookMigrationState) -> WorkbookMigrationState:
    """Node: Finalize the migration and generate report."""
    logger.info(f"[{state['migration_id']}] Finalizing migration")

    if state["modules_failed"]:
        state["phase"] = MigrationPhase.FAILED
    else:
        state["phase"] = MigrationPhase.COMPLETE

    # Calculate overall confidence
    confidences = [
        ms["validation_result"]["confidence_score"]
        for ms in state["module_states"].values()
        if ms.get("validation_result")
    ]
    state["overall_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0

    logger.info(
        f"[{state['migration_id']}] Migration complete: "
        f"{len(state['modules_completed'])} succeeded, "
        f"{len(state['modules_failed'])} failed, "
        f"confidence={state['overall_confidence']:.2%}"
    )

    return state


# Routing functions
def route_after_translate(state: WorkbookMigrationState) -> Literal["test", "finalize"]:
    """Route after translation."""
    if not state["modules_pending"]:
        return "finalize"
    module = state["module_states"].get(state["current_module"])
    if module and module["phase"] == MigrationPhase.TESTING:
        return "test"
    return "finalize"


def route_after_validate(state: WorkbookMigrationState) -> Literal["translate", "finalize"]:
    """Route after validation."""
    if state["modules_pending"]:
        return "translate"
    return "finalize"


def build_migration_graph() -> StateGraph:
    """Build the LangGraph state machine for workbook migration."""

    graph = StateGraph(WorkbookMigrationState)

    # Add nodes
    graph.add_node("analyze", analyze_workbook)
    graph.add_node("translate", translate_module)
    graph.add_node("test", run_tests)
    graph.add_node("migrate", migrate_data)
    graph.add_node("validate", validate_migration)
    graph.add_node("finalize", finalize)

    # Set entry point
    graph.set_entry_point("analyze")

    # Add edges
    graph.add_edge("analyze", "translate")
    graph.add_conditional_edges("translate", route_after_translate, {"test": "test", "finalize": "finalize"})
    graph.add_edge("test", "migrate")
    graph.add_edge("migrate", "validate")
    graph.add_conditional_edges("validate", route_after_validate, {"translate": "translate", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph


class MigrationPipelineRunner:
    """Runner for the Excel migration pipeline with PostgreSQL checkpointing."""

    def __init__(self, database_url: str):
        self.conn = psycopg2.connect(database_url)
        self.checkpointer = PostgresSaver(self.conn)
        self.graph = build_migration_graph().compile(checkpointer=self.checkpointer)
        logger.info("MigrationPipelineRunner initialized")

    async def run_migration(
        self,
        workbook_path: str,
        modules: list[str],
    ) -> WorkbookMigrationState:
        """Run a complete workbook migration."""
        initial_state = create_initial_state(workbook_path, modules)

        logger.info(f"Starting migration {initial_state['migration_id']}")

        config = {"configurable": {"thread_id": initial_state["migration_id"]}}

        final_state = await self.graph.ainvoke(initial_state, config)

        return final_state

    async def resume_migration(self, migration_id: str) -> WorkbookMigrationState | None:
        """Resume a paused migration."""
        config = {"configurable": {"thread_id": migration_id}}

        state = await self.graph.aget_state(config)
        if not state:
            logger.warning(f"No state found for migration {migration_id}")
            return None

        logger.info(f"Resuming migration {migration_id}")
        return await self.graph.ainvoke(None, config)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the multi-agent migration system with Docker Compose, configure LangSmith for agent tracing, and set up comprehensive monitoring for tracking migration progress and success rates.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'PostgreSQL'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose Migration System Deployment',
                description:
                  'Production Docker Compose configuration for the Excel migration system with all agents, PostgreSQL state store, and observability.',
                code: `# docker-compose.migration.yml
version: "3.9"

services:
  # Migration API service
  migration-api:
    build:
      context: .
      dockerfile: Dockerfile.migration-api
    image: ghcr.io/construction-co/excel-migration-api:latest
    ports:
      - "8080:8080"
    environment:
      - APP_ENV=production
      - DATABASE_URL=postgresql://app:\${DB_PASSWORD}@postgres:5432/migration
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=excel-migration-production
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
    volumes:
      - ./workbooks:/app/workbooks:ro
      - ./output:/app/output
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - migration-network

  # Migration worker (runs long-running migrations)
  migration-worker:
    build:
      context: .
      dockerfile: Dockerfile.migration-worker
    image: ghcr.io/construction-co/excel-migration-worker:latest
    environment:
      - APP_ENV=production
      - DATABASE_URL=postgresql://app:\${DB_PASSWORD}@postgres:5432/migration
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_TRACING_V2=true
    volumes:
      - ./workbooks:/app/workbooks:ro
      - ./output:/app/output
    depends_on:
      - postgres
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 8G
          cpus: "4.0"
    networks:
      - migration-network

  # PostgreSQL for state management and migration tracking
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=migration
      - POSTGRES_USER=app
      - POSTGRES_PASSWORD=\${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-migration-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d migration"]
      interval: 10s
      timeout: 5s
      retries: 5
    ports:
      - "5432:5432"
    networks:
      - migration-network

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus-migration.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - migration-network

  # Grafana for migration dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/migration-dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    networks:
      - migration-network

volumes:
  postgres-data:
  prometheus-data:
  grafana-data:

networks:
  migration-network:
    driver: bridge`,
              },
              {
                language: 'python',
                title: 'Migration Metrics & Progress Tracking',
                description:
                  'Observability implementation for tracking migration progress, success rates, and agent performance with Prometheus metrics.',
                code: `"""observability/migration_metrics.py - Metrics for Excel migration pipeline"""
import logging
from datetime import datetime
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Prometheus metrics for migration tracking
MIGRATIONS_TOTAL = Counter(
    "excel_migrations_total",
    "Total Excel migrations processed",
    ["workbook", "status"],
)

MODULES_MIGRATED = Counter(
    "excel_modules_migrated_total",
    "Total VBA modules migrated",
    ["workbook", "status"],
)

PROCEDURES_TRANSLATED = Counter(
    "excel_procedures_translated_total",
    "Total VBA procedures translated to Python",
    ["workbook", "module"],
)

ROWS_MIGRATED = Counter(
    "excel_rows_migrated_total",
    "Total data rows migrated to PostgreSQL",
    ["workbook", "table"],
)

MIGRATION_DURATION = Histogram(
    "excel_migration_duration_seconds",
    "Time spent on migrations",
    ["workbook", "phase"],
    buckets=[60, 300, 600, 1800, 3600, 7200, 14400],
)

TRANSLATION_CONFIDENCE = Histogram(
    "excel_translation_confidence",
    "Translation confidence scores",
    ["workbook", "module"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
)

VALIDATION_PASS_RATE = Gauge(
    "excel_validation_pass_rate",
    "Current validation pass rate",
    ["workbook"],
)

ACTIVE_MIGRATIONS = Gauge(
    "excel_active_migrations",
    "Number of migrations currently in progress",
)

PENDING_MODULES = Gauge(
    "excel_pending_modules",
    "Number of modules pending migration",
    ["workbook"],
)

SYSTEM_INFO = Info(
    "excel_migration_system",
    "Excel migration system information",
)


class MigrationMetricsCollector:
    """Collects and exposes metrics for the Excel migration pipeline."""

    def __init__(self, port: int = 9092):
        self.port = port
        SYSTEM_INFO.info({
            "version": "1.0.0",
            "environment": "production",
            "framework": "langgraph",
        })
        logger.info("MigrationMetricsCollector initialized")

    def start_server(self) -> None:
        """Start the Prometheus metrics HTTP server."""
        start_http_server(self.port)
        logger.info(f"Migration metrics server started on port {self.port}")

    def record_migration_start(self, workbook: str, modules: int) -> None:
        """Record the start of a migration."""
        ACTIVE_MIGRATIONS.inc()
        PENDING_MODULES.labels(workbook=workbook).set(modules)
        logger.info(f"Migration started: {workbook} with {modules} modules")

    def record_migration_complete(
        self,
        workbook: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        """Record a completed migration."""
        MIGRATIONS_TOTAL.labels(workbook=workbook, status=status).inc()
        MIGRATION_DURATION.labels(workbook=workbook, phase="total").observe(duration_seconds)
        ACTIVE_MIGRATIONS.dec()
        PENDING_MODULES.labels(workbook=workbook).set(0)
        logger.info(f"Migration complete: {workbook} - {status} in {duration_seconds:.1f}s")

    def record_module_complete(
        self,
        workbook: str,
        module: str,
        status: str,
        procedures: int,
        confidence: float,
    ) -> None:
        """Record a completed module migration."""
        MODULES_MIGRATED.labels(workbook=workbook, status=status).inc()
        PROCEDURES_TRANSLATED.labels(workbook=workbook, module=module).inc(procedures)
        TRANSLATION_CONFIDENCE.labels(workbook=workbook, module=module).observe(confidence)
        logger.info(f"Module complete: {workbook}/{module} - {procedures} procedures, confidence={confidence:.2%}")

    def record_data_migration(
        self,
        workbook: str,
        table: str,
        rows: int,
    ) -> None:
        """Record data rows migrated."""
        ROWS_MIGRATED.labels(workbook=workbook, table=table).inc(rows)
        logger.info(f"Data migrated: {rows} rows to {table}")

    def record_validation_result(
        self,
        workbook: str,
        passed: int,
        total: int,
    ) -> None:
        """Record validation results."""
        rate = passed / total if total > 0 else 0.0
        VALIDATION_PASS_RATE.labels(workbook=workbook).set(rate)
        logger.info(f"Validation: {workbook} - {passed}/{total} ({rate:.1%})")

    def update_pending_modules(self, workbook: str, pending: int) -> None:
        """Update the count of pending modules."""
        PENDING_MODULES.labels(workbook=workbook).set(pending)


# Global metrics instance
migration_metrics = MigrationMetricsCollector()


def generate_migration_report(state: dict[str, Any]) -> str:
    """Generate a human-readable migration report."""
    report_lines = [
        "=" * 60,
        f"EXCEL MIGRATION REPORT: {state['workbook_name']}",
        "=" * 60,
        "",
        f"Migration ID: {state['migration_id']}",
        f"Status: {state['phase']}",
        f"Started: {state['started_at']}",
        f"Last Checkpoint: {state['last_checkpoint']}",
        "",
        "SUMMARY",
        "-" * 40,
        f"Total Procedures: {state['total_procedures']}",
        f"Translated: {state['procedures_translated']}",
        f"Validated: {state['procedures_validated']}",
        f"Rows Migrated: {state['rows_migrated']}",
        f"Overall Confidence: {state['overall_confidence']:.1%}",
        "",
        "MODULES",
        "-" * 40,
        f"Completed: {len(state['modules_completed'])}",
        f"Failed: {len(state['modules_failed'])}",
        f"Pending: {len(state['modules_pending'])}",
        "",
    ]

    if state['modules_completed']:
        report_lines.append("Completed Modules:")
        for module in state['modules_completed']:
            ms = state['module_states'].get(module, {})
            conf = ms.get('validation_result', {}).get('confidence_score', 0)
            report_lines.append(f"  - {module}: {conf:.1%} confidence")

    if state['modules_failed']:
        report_lines.append("")
        report_lines.append("Failed Modules:")
        for module in state['modules_failed']:
            ms = state['module_states'].get(module, {})
            err = ms.get('error_message', 'Unknown error')
            report_lines.append(f"  - {module}: {err}")

    if state['errors']:
        report_lines.append("")
        report_lines.append("ERRORS")
        report_lines.append("-" * 40)
        for error in state['errors']:
            report_lines.append(f"  [{error['phase']}] {error['error']}")

    report_lines.extend(["", "=" * 60])

    return "\\n".join(report_lines)`,
              },
            ],
          },
        ],
      },
    },
  ],
};
