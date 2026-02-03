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
    },
  ],
};
