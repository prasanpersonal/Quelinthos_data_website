import type { Category } from '../types.ts';

export const customerServiceCategory: Category = {
  id: 'customer-service',
  number: 3,
  title: 'Customer Service & Experience',
  shortTitle: 'Customer Service',
  description:
    'Build unified customer views, close feedback loops, and eliminate the WISMO crisis drowning your support team.',
  icon: 'Headphones',
  accentColor: 'neon-gold',
  painPoints: [
    /* ──────────────────────────────────────────────
       Pain Point 1 — The Unified Customer View
       ────────────────────────────────────────────── */
    {
      id: 'unified-customer-view',
      number: 1,
      title: 'The Unified Customer View',
      subtitle: 'Fragmented Customer Data Across Systems',
      summary:
        'Your support agents toggle between 6 tabs to understand one customer. Average handle time is 3x what it should be.',
      tags: ['customer-360', 'data-unification', 'support'],
      metrics: {
        annualCostRange: '$800K - $4M',
        roi: '7x',
        paybackPeriod: '3-4 months',
        investmentRange: '$120K - $250K',
      },
      price: {
        present: {
          title: 'Present — The Symptom',
          description:
            'Support agents alt-tab through CRM, billing, shipping, product, loyalty, and returns systems for every interaction.',
          bullets: [
            'Average handle time exceeds 12 minutes per ticket — industry benchmark is 4 minutes',
            'Agents ask customers to repeat information already captured in other systems',
            'Escalation rate sits at 35% because Tier-1 agents lack the full picture',
            'Customer satisfaction (CSAT) scores have declined 18% over the past two quarters',
          ],
          severity: 'critical',
        },
        root: {
          title: 'Root Cause — Why It Persists',
          description:
            'Customer data lives in siloed systems that were purchased independently and never integrated at the identity layer.',
          bullets: [
            'No canonical customer identifier spans all platforms — email, phone, and account IDs conflict',
            'Each department owns its own database with no shared schema or CDC pipeline',
            'Previous integration attempts produced brittle point-to-point ETL jobs that break silently',
            'IT backlog for integration projects averages 9-14 months across enterprise teams',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Impact — Business Damage',
          description:
            'Fragmented customer data inflates support costs, erodes loyalty, and blinds leadership to churn risk.',
          bullets: [
            'Support labor costs are $800K-$4M higher annually than a unified-view benchmark',
            'Repeat contacts on the same issue cost $22 per unnecessary interaction',
            'Churn-risk customers go undetected because no single view aggregates warning signals',
            'Cross-sell and upsell opportunities are invisible to the agents who speak with customers daily',
          ],
          severity: 'critical',
        },
        cost: {
          title: 'Cost of Inaction — 12-Month Horizon',
          description:
            'Without unification, handle times rise with every new system added and agent turnover compounds the knowledge gap.',
          bullets: [
            'Each new SaaS tool added to the stack increases average handle time by another 45 seconds',
            'Agent onboarding takes 8 weeks instead of 3 when tribal knowledge spans 6 tools',
            'Annual agent attrition of 40% means the training cost repeats every cycle',
            'Competitors with unified views are resolving tickets 3x faster and winning on experience',
          ],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Expected Return — Post-Implementation',
          description:
            'A single customer-360 view cuts handle time, reduces escalations, and unlocks revenue signals buried in support data.',
          bullets: [
            'Handle time drops from 12 minutes to 4 minutes — a 65% reduction',
            'Escalation rate falls from 35% to 12% as Tier-1 agents gain full context',
            'Agent onboarding shrinks from 8 weeks to 3 weeks with one unified interface',
            'Proactive churn detection enabled through aggregated behavioral signals',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Build a real-time customer-360 view by resolving identities across systems, materializing a unified profile in your warehouse, and exposing it through a low-latency API for agent tooling.',
        prerequisites: [
          'Read access to CRM, billing, shipping, product, and loyalty databases',
          'A data warehouse (Snowflake, BigQuery, or Redshift) with CDC ingestion enabled',
          'Python 3.10+ with FastAPI for the profile API layer',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        toolsUsed: ['SQL', 'Python', 'FastAPI', 'dbt', 'pytest', 'Docker', 'GitHub Actions', 'cron / Airflow', 'Slack API'],
        steps: [
          {
            stepNumber: 1,
            title: 'Create the Customer 360 Base View',
            description:
              'Resolve customer identities across systems using email, phone, and account ID matching, then materialize a single canonical profile per customer.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Customer 360 View Creation',
                description:
                  'Builds a unified customer profile by joining CRM, billing, and order data with deterministic identity resolution.',
                code: `-- customer_360_base.sql
-- Deterministic identity resolution across core systems
CREATE OR REPLACE TABLE analytics.customer_360 AS
WITH identity_graph AS (
  SELECT
    COALESCE(c.canonical_id, b.customer_id, o.account_id) AS customer_id,
    LOWER(TRIM(COALESCE(c.email, b.email, o.email)))      AS email,
    COALESCE(c.phone, b.phone)                             AS phone,
    c.full_name,
    c.created_at                                           AS first_seen_at,
    c.segment
  FROM crm.customers        c
  FULL OUTER JOIN billing.accounts b
    ON LOWER(TRIM(c.email)) = LOWER(TRIM(b.email))
  FULL OUTER JOIN orders.customers o
    ON LOWER(TRIM(c.email)) = LOWER(TRIM(o.email))
),
enriched AS (
  SELECT
    ig.*,
    COUNT(DISTINCT o.order_id)                     AS lifetime_orders,
    SUM(o.total_amount)                            AS lifetime_revenue,
    MAX(o.created_at)                              AS last_order_at,
    AVG(r.rating)                                  AS avg_satisfaction,
    COUNT(DISTINCT t.ticket_id)                    AS open_tickets,
    lp.points_balance                              AS loyalty_points
  FROM identity_graph ig
  LEFT JOIN orders.orders        o  ON ig.customer_id = o.customer_id
  LEFT JOIN support.tickets      t  ON ig.customer_id = t.customer_id
    AND t.status IN ('open', 'pending')
  LEFT JOIN surveys.responses    r  ON ig.customer_id = r.customer_id
  LEFT JOIN loyalty.members      lp ON ig.customer_id = lp.customer_id
  GROUP BY 1,2,3,4,5,6
)
SELECT * FROM enriched;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Cross-System Identity Resolution',
            description:
              'Handle fuzzy and probabilistic identity matching for records that cannot be joined on exact email. This step catches roughly 15-20% of customer records that slip through deterministic matching.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Probabilistic Identity Matching',
                description:
                  'Uses phone normalization, name similarity, and address hashing to link records that share no common email.',
                code: `-- identity_resolution_fuzzy.sql
-- Catch non-deterministic matches via phone + name + address
WITH phone_normalized AS (
  SELECT
    customer_id,
    REGEXP_REPLACE(phone, '[^0-9]', '')            AS clean_phone,
    LOWER(TRIM(full_name))                         AS clean_name,
    MD5(CONCAT(LOWER(TRIM(address_line_1)),
               LOWER(TRIM(zip_code))))             AS address_hash
  FROM analytics.customer_360
  WHERE email IS NULL
),
candidate_pairs AS (
  SELECT
    a.customer_id  AS id_a,
    b.customer_id  AS id_b,
    CASE WHEN a.clean_phone = b.clean_phone        THEN 0.45 ELSE 0 END
    + CASE WHEN JAROWINKLER_SIMILARITY(a.clean_name, b.clean_name) > 0.88
           THEN 0.35 ELSE 0 END
    + CASE WHEN a.address_hash = b.address_hash    THEN 0.20 ELSE 0 END
      AS confidence_score
  FROM phone_normalized a
  JOIN phone_normalized b
    ON a.customer_id < b.customer_id
   AND (a.clean_phone = b.clean_phone
        OR a.address_hash = b.address_hash)
)
SELECT id_a, id_b, confidence_score
FROM candidate_pairs
WHERE confidence_score >= 0.70
ORDER BY confidence_score DESC;`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate data quality in the customer-360 pipeline using SQL assertions and a pytest suite that checks identity resolution accuracy, profile completeness, and freshness guarantees.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Customer 360 Data Quality Assertions',
                description:
                  'SQL assertion queries that verify identity resolution produced no orphan records, no duplicate canonical IDs, and that profile completeness meets SLA thresholds.',
                code: `-- test_customer_360_quality.sql
-- Assertion 1: No duplicate canonical customer IDs
SELECT 'duplicate_customer_ids' AS assertion,
       COUNT(*) AS violations
FROM (
  SELECT customer_id, COUNT(*) AS cnt
  FROM analytics.customer_360
  GROUP BY customer_id
  HAVING COUNT(*) > 1
) dupes
HAVING COUNT(*) > 0;

-- Assertion 2: Email fill-rate exceeds 90%
SELECT 'email_fill_rate_below_90pct' AS assertion,
       ROUND(100.0 * SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END)
             / COUNT(*), 2) AS null_pct
FROM analytics.customer_360
HAVING SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END)
       / COUNT(*)::float > 0.10;

-- Assertion 3: No customer_id is NULL
SELECT 'null_customer_id' AS assertion,
       COUNT(*) AS violations
FROM analytics.customer_360
WHERE customer_id IS NULL
HAVING COUNT(*) > 0;

-- Assertion 4: Freshness — table refreshed within last 2 hours
SELECT 'stale_customer_360' AS assertion,
       MAX(last_order_at)::text AS most_recent
FROM analytics.customer_360
HAVING MAX(last_order_at) < CURRENT_TIMESTAMP - INTERVAL '2 hours';`,
              },
              {
                language: 'python',
                title: 'Pytest Pipeline Validation Suite',
                description:
                  'A pytest-based validation suite with typed test classes that verify identity resolution, profile schema integrity, and row-count drift between refreshes.',
                code: `# tests/test_customer_360_pipeline.py
"""Validation suite for the customer-360 identity resolution pipeline."""
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

@dataclass
class QualityThresholds:
    """Typed thresholds for data-quality checks."""
    max_duplicate_ids: int = 0
    min_email_fill_rate: float = 0.90
    max_row_count_drift_pct: float = 0.05
    max_staleness_hours: int = 2


class TestCustomer360Quality:
    """Data-quality tests for analytics.customer_360."""

    engine = create_engine("postgresql://localhost/warehouse")
    thresholds = QualityThresholds()

    # ── helpers ──────────────────────────────────────
    def _query_scalar(self, sql: str) -> Any:
        with self.engine.connect() as conn:
            return conn.execute(text(sql)).scalar()

    # ── tests ────────────────────────────────────────
    def test_no_duplicate_customer_ids(self) -> None:
        dupes: int = self._query_scalar(
            "SELECT COUNT(*) FROM ("
            "  SELECT customer_id FROM analytics.customer_360"
            "  GROUP BY customer_id HAVING COUNT(*) > 1"
            ") t"
        )
        logger.info("Duplicate customer IDs found: %d", dupes)
        assert dupes <= self.thresholds.max_duplicate_ids

    def test_email_fill_rate(self) -> None:
        rate: float = self._query_scalar(
            "SELECT AVG(CASE WHEN email IS NOT NULL"
            "  THEN 1.0 ELSE 0.0 END)"
            " FROM analytics.customer_360"
        )
        logger.info("Email fill rate: %.2f%%", rate * 100)
        assert rate >= self.thresholds.min_email_fill_rate

    def test_row_count_drift(self) -> None:
        current: int = self._query_scalar(
            "SELECT COUNT(*) FROM analytics.customer_360"
        )
        previous: int = self._query_scalar(
            "SELECT row_count FROM analytics.pipeline_metadata"
            " WHERE table_name = 'customer_360'"
            " ORDER BY snapshot_at DESC LIMIT 1"
        )
        drift: float = abs(current - previous) / max(previous, 1)
        logger.info("Row-count drift: %.2f%%", drift * 100)
        assert drift <= self.thresholds.max_row_count_drift_pct

    def test_freshness(self) -> None:
        hours: float = self._query_scalar(
            "SELECT EXTRACT(EPOCH FROM"
            "  (CURRENT_TIMESTAMP - MAX(last_order_at)))"
            " / 3600.0 FROM analytics.customer_360"
        )
        logger.info("Table staleness: %.1f hours", hours)
        assert hours <= self.thresholds.max_staleness_hours`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Containerize the customer-360 refresh pipeline and profile API with Docker, then deploy via a bash script that validates the build, runs migrations, and starts services behind a health-check.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Customer 360 Deployment Script',
                description:
                  'End-to-end deployment script that builds the Docker image, runs database migrations, executes the pytest validation suite, and starts the customer-360 API service.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────
APP_NAME="customer-360"
IMAGE_TAG="\${APP_NAME}:\$(git rev-parse --short HEAD)"
COMPOSE_FILE="docker-compose.customer360.yml"
MIGRATION_DIR="./migrations/customer_360"
HEALTHCHECK_URL="http://localhost:8000/healthz"
MAX_WAIT=60

echo "==> Building Docker image \${IMAGE_TAG}"
docker build -t "\${IMAGE_TAG}" -f Dockerfile.customer360 .

echo "==> Running database migrations"
docker run --rm --env-file .env "\${IMAGE_TAG}" \
  python -m alembic -c "\${MIGRATION_DIR}/alembic.ini" upgrade head

echo "==> Running pytest validation suite"
docker run --rm --env-file .env "\${IMAGE_TAG}" \
  python -m pytest tests/test_customer_360_pipeline.py -v --tb=short

echo "==> Starting services via docker-compose"
docker-compose -f "\${COMPOSE_FILE}" up -d

echo "==> Waiting for health-check at \${HEALTHCHECK_URL}"
elapsed=0
until curl -sf "\${HEALTHCHECK_URL}" > /dev/null 2>&1; do
  sleep 2
  elapsed=\$((elapsed + 2))
  if [ "\${elapsed}" -ge "\${MAX_WAIT}" ]; then
    echo "ERROR: Health-check failed after \${MAX_WAIT}s" >&2
    docker-compose -f "\${COMPOSE_FILE}" logs --tail=50
    exit 1
  fi
done

echo "==> Deployment complete — \${APP_NAME} is healthy"`,
              },
              {
                language: 'python',
                title: 'Typed Configuration Loader',
                description:
                  'A dataclass-based configuration loader that reads environment variables with validation, type coercion, and sensible defaults for the customer-360 service.',
                code: `# config/customer360_config.py
"""Typed configuration loader for the customer-360 service."""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabaseConfig:
    """Connection settings for the analytics warehouse."""
    url: str = ""
    pool_min: int = 5
    pool_max: int = 20
    statement_timeout_ms: int = 30_000


@dataclass(frozen=True)
class ApiConfig:
    """Settings for the FastAPI profile service."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    log_level: str = "info"


@dataclass(frozen=True)
class SchedulerConfig:
    """Refresh-pipeline scheduling settings."""
    cron_expression: str = "0 */2 * * *"
    max_retries: int = 3
    retry_delay_seconds: int = 60


@dataclass(frozen=True)
class Customer360Config:
    """Top-level configuration for the customer-360 stack."""
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    slack_webhook: str = ""

    @classmethod
    def from_env(cls) -> Customer360Config:
        """Build config from environment variables with validation."""
        cfg = cls(
            db=DatabaseConfig(
                url=os.environ["DATABASE_URL"],
                pool_min=int(os.getenv("DB_POOL_MIN", "5")),
                pool_max=int(os.getenv("DB_POOL_MAX", "20")),
                statement_timeout_ms=int(os.getenv("DB_STMT_TIMEOUT", "30000")),
            ),
            api=ApiConfig(
                host=os.getenv("API_HOST", "0.0.0.0"),
                port=int(os.getenv("API_PORT", "8000")),
                workers=int(os.getenv("API_WORKERS", "4")),
                log_level=os.getenv("API_LOG_LEVEL", "info"),
            ),
            scheduler=SchedulerConfig(
                cron_expression=os.getenv("REFRESH_CRON", "0 */2 * * *"),
                max_retries=int(os.getenv("REFRESH_MAX_RETRIES", "3")),
                retry_delay_seconds=int(os.getenv("REFRESH_RETRY_DELAY", "60")),
            ),
            slack_webhook=os.getenv("SLACK_WEBHOOK_URL", ""),
        )
        logger.info("Loaded Customer360Config — DB pool %d-%d, API :%d",
                     cfg.db.pool_min, cfg.db.pool_max, cfg.api.port)
        return cfg`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Expose the unified profile through a low-latency FastAPI service so agent desktops can render the full customer context in under 200ms, and add continuous monitoring to detect profile staleness and identity-resolution anomalies.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Customer Profile API Endpoint',
                description:
                  'A FastAPI service that queries the customer-360 table and returns a structured profile with caching for sub-200ms responses.',
                code: `# customer_profile_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from functools import lru_cache
import asyncpg
import os

app = FastAPI(title="Customer 360 API")

class CustomerProfile(BaseModel):
    customer_id: str
    email: str | None
    full_name: str | None
    segment: str | None
    lifetime_orders: int
    lifetime_revenue: float
    last_order_at: str | None
    avg_satisfaction: float | None
    open_tickets: int
    loyalty_points: int | None

async def get_pool():
    return await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=5, max_size=20)

@app.on_event("startup")
async def startup():
    app.state.pool = await get_pool()

@app.get("/api/v1/customer/{customer_id}", response_model=CustomerProfile)
async def get_customer_profile(customer_id: str):
    query = """
        SELECT customer_id, email, full_name, segment,
               lifetime_orders, lifetime_revenue,
               last_order_at::text, avg_satisfaction,
               open_tickets, loyalty_points
        FROM analytics.customer_360
        WHERE customer_id = $1
    """
    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow(query, customer_id)
    if not row:
        raise HTTPException(status_code=404, detail="Customer not found")
    return CustomerProfile(**dict(row))`,
              },
              {
                language: 'python',
                title: 'Profile Staleness Monitor and Slack Alerter',
                description:
                  'Periodically checks the customer-360 table for staleness and identity-resolution anomalies, then sends Slack alerts when thresholds are breached.',
                code: `# monitor_customer360.py
"""Monitoring and alerting for the customer-360 pipeline."""
import logging
from dataclasses import dataclass
from datetime import datetime

import requests
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


@dataclass
class AlertThresholds:
    max_staleness_hours: float = 2.0
    max_duplicate_rate: float = 0.001
    min_row_count: int = 1000


def check_customer360_health(
    db_url: str,
    slack_webhook: str,
    thresholds: AlertThresholds | None = None,
) -> list[str]:
    """Run health checks and send Slack alerts for any violations."""
    thresholds = thresholds or AlertThresholds()
    engine = create_engine(db_url)
    alerts: list[str] = []

    with engine.connect() as conn:
        # Check staleness
        staleness_h: float = conn.execute(text(
            "SELECT EXTRACT(EPOCH FROM"
            " (CURRENT_TIMESTAMP - MAX(last_order_at)))"
            " / 3600.0 FROM analytics.customer_360"
        )).scalar() or 0.0

        if staleness_h > thresholds.max_staleness_hours:
            alerts.append(
                f"*Staleness alert:* customer_360 last refreshed"
                f" {staleness_h:.1f}h ago (limit: {thresholds.max_staleness_hours}h)"
            )

        # Check duplicates
        total: int = conn.execute(text(
            "SELECT COUNT(*) FROM analytics.customer_360"
        )).scalar() or 0
        dupes: int = conn.execute(text(
            "SELECT COUNT(*) FROM ("
            " SELECT customer_id FROM analytics.customer_360"
            " GROUP BY customer_id HAVING COUNT(*) > 1) t"
        )).scalar() or 0
        dup_rate: float = dupes / max(total, 1)

        if dup_rate > thresholds.max_duplicate_rate:
            alerts.append(
                f"*Duplicate-ID alert:* {dupes} duplicate IDs"
                f" ({dup_rate:.4%}) exceed {thresholds.max_duplicate_rate:.4%} limit"
            )

        # Check row count
        if total < thresholds.min_row_count:
            alerts.append(
                f"*Row-count alert:* only {total:,} rows"
                f" (minimum: {thresholds.min_row_count:,})"
            )

    if alerts and slack_webhook:
        payload = {
            "text": (
                ":warning: *Customer-360 Health Check Failed*\\n"
                + "\\n".join(alerts)
                + f"\\n_Checked at {datetime.utcnow():%Y-%m-%d %H:%M UTC}_"
            )
        }
        requests.post(slack_webhook, json=payload, timeout=10)
        logger.warning("Sent %d alerts to Slack", len(alerts))

    return alerts`,
              },
            ],
          },
        ],
      },
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude combined with Zapier to automatically consolidate customer data from multiple sources, generate unified customer summaries, and deliver them to agent dashboards without custom code.',
        estimatedMonthlyCost: '$120 - $200/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Intercom ($74/mo)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Zendesk AI ($55/mo)', 'Freshdesk AI ($35/mo)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Set up Zapier to extract customer data from CRM, billing, and support systems whenever an agent opens a ticket, formatting it into a structured JSON payload for AI analysis.',
            toolsUsed: ['Zapier', 'Webhooks', 'Intercom API'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Trigger Configuration',
                description: 'Configure Zapier to trigger on new ticket creation and fetch related customer data from connected systems.',
                code: `{
  "trigger": {
    "app": "Intercom",
    "event": "New Conversation",
    "filters": {
      "conversation_type": "support"
    }
  },
  "actions": [
    {
      "step": 1,
      "app": "Webhooks by Zapier",
      "action": "GET",
      "url": "{{CRM_API_URL}}/customers/{{contact_email}}",
      "headers": {
        "Authorization": "Bearer {{CRM_API_KEY}}"
      }
    },
    {
      "step": 2,
      "app": "Webhooks by Zapier",
      "action": "GET",
      "url": "{{BILLING_API_URL}}/accounts?email={{contact_email}}",
      "headers": {
        "Authorization": "Bearer {{BILLING_API_KEY}}"
      }
    },
    {
      "step": 3,
      "app": "Webhooks by Zapier",
      "action": "GET",
      "url": "{{ORDERS_API_URL}}/orders?customer_email={{contact_email}}&limit=10",
      "headers": {
        "Authorization": "Bearer {{ORDERS_API_KEY}}"
      }
    }
  ],
  "output": {
    "customer_profile": {
      "email": "{{contact_email}}",
      "crm_data": "{{step1_response}}",
      "billing_data": "{{step2_response}}",
      "order_history": "{{step3_response}}"
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
              'Send the consolidated customer data to ChatGPT or Claude to generate a unified customer summary with key insights, risk indicators, and recommended actions for the support agent.',
            toolsUsed: ['ChatGPT API', 'Claude API'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Customer 360 Summary Prompt Template',
                description: 'A structured prompt that instructs the AI to analyze customer data and produce an actionable summary.',
                code: `system_prompt: |
  You are a customer intelligence assistant for support agents.
  Your job is to analyze customer data from multiple systems and
  produce a concise, actionable summary.

  Always include:
  1. Customer tier and lifetime value
  2. Recent order status and any issues
  3. Open support tickets and sentiment
  4. Churn risk assessment (Low/Medium/High)
  5. Recommended talking points

user_prompt_template: |
  Analyze this customer's data and provide a unified profile summary:

  ## CRM Data
  {{crm_data}}

  ## Billing History
  {{billing_data}}

  ## Recent Orders
  {{order_history}}

  ## Current Ticket
  Subject: {{ticket_subject}}
  Message: {{ticket_body}}

  ---
  Provide a structured summary with:
  - **Customer Overview**: Name, tier, LTV, tenure
  - **Recent Activity**: Last 3 orders, any delivery issues
  - **Support History**: Open tickets, avg satisfaction
  - **Risk Assessment**: Churn probability and reasoning
  - **Agent Recommendations**: 2-3 talking points for this interaction

output_format: |
  Return the summary in markdown format suitable for display
  in the agent dashboard. Keep it scannable - use bullet points
  and bold key metrics.

parameters:
  model: "gpt-4-turbo"
  temperature: 0.3
  max_tokens: 800`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier to post the AI-generated customer summary directly into the support ticket or agent dashboard, making it available within seconds of ticket creation.',
            toolsUsed: ['Zapier', 'Intercom API', 'Slack'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Delivery Workflow',
                description: 'Complete Zapier workflow that delivers the AI summary to Intercom and optionally alerts on high-risk customers.',
                code: `{
  "workflow_name": "Customer 360 AI Summary",
  "steps": [
    {
      "step": 4,
      "app": "OpenAI (ChatGPT)",
      "action": "Send Prompt",
      "config": {
        "model": "gpt-4-turbo",
        "system_message": "{{system_prompt}}",
        "user_message": "{{formatted_user_prompt}}",
        "temperature": 0.3,
        "max_tokens": 800
      }
    },
    {
      "step": 5,
      "app": "Intercom",
      "action": "Add Note to Conversation",
      "config": {
        "conversation_id": "{{trigger_conversation_id}}",
        "note_body": "## AI Customer Summary\\n\\n{{chatgpt_response}}",
        "admin_id": "{{bot_admin_id}}"
      }
    },
    {
      "step": 6,
      "app": "Filter by Zapier",
      "condition": "chatgpt_response CONTAINS 'High' AND chatgpt_response CONTAINS 'Churn'"
    },
    {
      "step": 7,
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#high-risk-customers",
        "message": ":warning: *High Churn Risk Customer*\\n*Email:* {{contact_email}}\\n*Ticket:* {{ticket_subject}}\\n*Summary:* {{chatgpt_response | truncate: 500}}\\n<{{intercom_conversation_url}}|View in Intercom>"
      }
    }
  ],
  "error_handling": {
    "on_chatgpt_error": {
      "action": "Add Note to Conversation",
      "note": "AI summary unavailable - please check systems manually"
    }
  }
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent system that continuously resolves customer identities across systems, enriches profiles with behavioral signals, and serves unified views through a real-time API with sub-100ms latency.',
        estimatedMonthlyCost: '$800 - $1,500/month',
        architecture:
          'A supervisor agent coordinates four specialist agents: Identity Resolver, Profile Enricher, Risk Scorer, and API Server. State is managed in Redis with PostgreSQL for persistence.',
        agents: [
          {
            name: 'IdentityResolverAgent',
            role: 'Identity Resolution Specialist',
            goal: 'Match and merge customer records across CRM, billing, orders, and support systems using deterministic and probabilistic matching',
            tools: ['PostgreSQL', 'Redis', 'FuzzyWuzzy', 'RecordLinkage'],
          },
          {
            name: 'ProfileEnricherAgent',
            role: 'Customer Data Enricher',
            goal: 'Aggregate behavioral signals, compute lifetime metrics, and enrich profiles with external data sources',
            tools: ['SQL', 'Pandas', 'Clearbit API', 'FullContact API'],
          },
          {
            name: 'RiskScorerAgent',
            role: 'Churn Risk Analyst',
            goal: 'Calculate real-time churn risk scores based on behavioral patterns, support history, and engagement metrics',
            tools: ['Scikit-learn', 'XGBoost', 'Feature Store'],
          },
          {
            name: 'ProfileAPIAgent',
            role: 'Real-Time Profile Server',
            goal: 'Serve unified customer profiles to agent tools with sub-100ms latency and intelligent caching',
            tools: ['FastAPI', 'Redis Cache', 'AsyncPG'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with hourly checkpointing to PostgreSQL',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system with CrewAI, establishing clear roles, goals, and tool assignments for each specialist agent in the customer-360 pipeline.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Customer 360 Agent Definitions',
                description: 'CrewAI agent definitions for the unified customer view multi-agent system.',
                code: `# agents/customer360_agents.py
"""Multi-agent system for unified customer view generation."""
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Customer360AgentFactory:
    """Factory for creating customer-360 specialist agents."""

    def __init__(self, llm_model: str = "gpt-4-turbo"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)

    def create_identity_resolver(self) -> Agent:
        """Agent specialized in cross-system identity resolution."""
        return Agent(
            role="Identity Resolution Specialist",
            goal="Match and merge customer records across all systems "
                 "using deterministic and probabilistic matching techniques",
            backstory="""You are an expert in customer identity resolution
            with deep knowledge of data matching algorithms. You understand
            how to handle fuzzy matching on names, normalize phone numbers
            and emails, and calculate confidence scores for potential matches.
            You prioritize precision over recall to avoid false merges.""",
            llm=self.llm,
            tools=[],  # Tools injected at runtime
            verbose=True,
            allow_delegation=False,
        )

    def create_profile_enricher(self) -> Agent:
        """Agent specialized in profile data enrichment."""
        return Agent(
            role="Customer Data Enricher",
            goal="Aggregate behavioral signals and compute lifetime metrics "
                 "to create comprehensive customer profiles",
            backstory="""You are a customer analytics expert who understands
            how to synthesize data from multiple touchpoints. You calculate
            LTV, identify purchase patterns, and flag anomalies that might
            indicate fraud or churn risk. You know how to handle missing
            data gracefully.""",
            llm=self.llm,
            tools=[],
            verbose=True,
            allow_delegation=False,
        )

    def create_risk_scorer(self) -> Agent:
        """Agent specialized in churn risk assessment."""
        return Agent(
            role="Churn Risk Analyst",
            goal="Calculate real-time churn risk scores based on behavioral "
                 "patterns, support history, and engagement metrics",
            backstory="""You are a predictive analytics specialist focused
            on customer retention. You understand the signals that precede
            churn: declining engagement, negative support interactions,
            and competitive research behavior. You provide actionable
            risk scores with clear explanations.""",
            llm=self.llm,
            tools=[],
            verbose=True,
            allow_delegation=False,
        )

    def create_profile_api_agent(self) -> Agent:
        """Agent specialized in serving unified profiles."""
        return Agent(
            role="Real-Time Profile Server",
            goal="Serve unified customer profiles with sub-100ms latency "
                 "using intelligent caching and query optimization",
            backstory="""You are a performance optimization expert who
            understands caching strategies, query patterns, and latency
            budgets. You know how to structure data for fast retrieval
            and when to refresh cached profiles.""",
            llm=self.llm,
            tools=[],
            verbose=True,
            allow_delegation=False,
        )

    def create_crew(self) -> Crew:
        """Assemble the complete customer-360 agent crew."""
        agents = [
            self.create_identity_resolver(),
            self.create_profile_enricher(),
            self.create_risk_scorer(),
            self.create_profile_api_agent(),
        ]

        tasks = [
            Task(
                description="Resolve customer identity across systems",
                agent=agents[0],
                expected_output="Canonical customer ID with confidence score",
            ),
            Task(
                description="Enrich profile with behavioral metrics",
                agent=agents[1],
                expected_output="Enriched profile with LTV and patterns",
            ),
            Task(
                description="Calculate churn risk score",
                agent=agents[2],
                expected_output="Risk score 0-100 with explanation",
            ),
            Task(
                description="Prepare profile for API serving",
                agent=agents[3],
                expected_output="Cached profile ready for retrieval",
            ),
        ]

        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the Identity Resolver agent with tools for deterministic email/phone matching and probabilistic name/address matching using record linkage techniques.',
            toolsUsed: ['CrewAI Tools', 'RecordLinkage', 'PostgreSQL'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Identity Resolution Tools',
                description: 'Custom tools for the Identity Resolver agent to match and merge customer records.',
                code: `# tools/identity_resolution_tools.py
"""Identity resolution tools for customer data matching."""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import recordlinkage
import pandas as pd
from sqlalchemy import create_engine, text
import os
import logging

logger = logging.getLogger(__name__)


class CustomerMatchInput(BaseModel):
    """Input schema for customer matching."""
    email: Optional[str] = Field(None, description="Customer email address")
    phone: Optional[str] = Field(None, description="Customer phone number")
    name: Optional[str] = Field(None, description="Customer full name")
    address: Optional[str] = Field(None, description="Customer address")


class DeterministicMatchTool(BaseTool):
    """Tool for exact matching on email and normalized phone."""

    name: str = "deterministic_match"
    description: str = (
        "Find exact customer matches using email or normalized phone number. "
        "Use this first before attempting fuzzy matching."
    )
    args_schema: type[BaseModel] = CustomerMatchInput

    def __init__(self):
        super().__init__()
        self.engine = create_engine(os.environ["DATABASE_URL"])

    def _normalize_phone(self, phone: str) -> str:
        """Strip non-numeric characters from phone."""
        if not phone:
            return ""
        return "".join(c for c in phone if c.isdigit())[-10:]

    def _run(
        self,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute deterministic matching."""
        matches = []

        with self.engine.connect() as conn:
            if email:
                result = conn.execute(
                    text("""
                        SELECT customer_id, email, phone, full_name,
                               'email' as match_type, 1.0 as confidence
                        FROM analytics.customer_360
                        WHERE LOWER(TRIM(email)) = LOWER(TRIM(:email))
                    """),
                    {"email": email}
                )
                matches.extend([dict(row._mapping) for row in result])

            if phone and not matches:
                normalized = self._normalize_phone(phone)
                result = conn.execute(
                    text("""
                        SELECT customer_id, email, phone, full_name,
                               'phone' as match_type, 0.95 as confidence
                        FROM analytics.customer_360
                        WHERE REGEXP_REPLACE(phone, '[^0-9]', '', 'g')
                              LIKE '%' || :phone
                    """),
                    {"phone": normalized}
                )
                matches.extend([dict(row._mapping) for row in result])

        return {
            "match_count": len(matches),
            "matches": matches[:5],  # Top 5 matches
            "match_method": "deterministic",
        }


class ProbabilisticMatchTool(BaseTool):
    """Tool for fuzzy matching using record linkage."""

    name: str = "probabilistic_match"
    description: str = (
        "Find potential customer matches using fuzzy name and address matching. "
        "Use when deterministic matching returns no results."
    )
    args_schema: type[BaseModel] = CustomerMatchInput

    def __init__(self):
        super().__init__()
        self.engine = create_engine(os.environ["DATABASE_URL"])

    def _run(
        self,
        name: Optional[str] = None,
        address: Optional[str] = None,
        phone: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute probabilistic matching with record linkage."""
        if not name and not address:
            return {"match_count": 0, "matches": [], "error": "Need name or address"}

        # Load candidate records
        with self.engine.connect() as conn:
            candidates = pd.read_sql(
                """
                SELECT customer_id, full_name, phone,
                       address_line_1, city, zip_code
                FROM analytics.customer_360
                WHERE full_name IS NOT NULL
                LIMIT 10000
                """,
                conn
            )

        # Create query record
        query_df = pd.DataFrame([{
            "customer_id": "QUERY",
            "full_name": name or "",
            "phone": phone or "",
            "address_line_1": address or "",
        }])

        # Build comparison index
        indexer = recordlinkage.Index()
        indexer.block(left_on="full_name", right_on="full_name")
        candidate_pairs = indexer.index(query_df, candidates)

        # Compare records
        compare = recordlinkage.Compare()
        compare.string("full_name", "full_name", method="jarowinkler",
                       threshold=0.85, label="name_score")
        if phone:
            compare.exact("phone", "phone", label="phone_score")

        features = compare.compute(candidate_pairs, query_df, candidates)

        # Score and rank matches
        features["total_score"] = features.sum(axis=1)
        top_matches = features[features["total_score"] > 0.7].nlargest(5, "total_score")

        matches = []
        for (_, idx), row in top_matches.iterrows():
            candidate = candidates.iloc[idx]
            matches.append({
                "customer_id": candidate["customer_id"],
                "full_name": candidate["full_name"],
                "confidence": round(row["total_score"], 3),
                "match_type": "probabilistic",
            })

        return {
            "match_count": len(matches),
            "matches": matches,
            "match_method": "probabilistic",
        }`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Profile Enricher and Risk Scorer agents with tools for computing lifetime metrics, behavioral patterns, and ML-based churn prediction.',
            toolsUsed: ['CrewAI Tools', 'Scikit-learn', 'XGBoost', 'Pandas'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Profile Enrichment and Risk Scoring Tools',
                description: 'Tools for enriching customer profiles and calculating churn risk scores.',
                code: `# tools/profile_analysis_tools.py
"""Profile enrichment and risk scoring tools."""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import joblib
import os
import logging

logger = logging.getLogger(__name__)


class CustomerIdInput(BaseModel):
    """Input schema requiring customer ID."""
    customer_id: str = Field(..., description="Canonical customer ID")


class ProfileEnrichmentTool(BaseTool):
    """Tool for enriching customer profiles with behavioral metrics."""

    name: str = "enrich_profile"
    description: str = (
        "Compute lifetime metrics and behavioral patterns for a customer. "
        "Returns LTV, purchase frequency, average order value, and engagement scores."
    )
    args_schema: type[BaseModel] = CustomerIdInput

    def __init__(self):
        super().__init__()
        self.engine = create_engine(os.environ["DATABASE_URL"])

    def _run(self, customer_id: str) -> Dict[str, Any]:
        """Compute enriched profile metrics."""
        with self.engine.connect() as conn:
            # Fetch aggregated metrics
            metrics = conn.execute(
                text("""
                    WITH order_stats AS (
                        SELECT
                            customer_id,
                            COUNT(*) as total_orders,
                            SUM(total_amount) as lifetime_revenue,
                            AVG(total_amount) as avg_order_value,
                            MAX(created_at) as last_order_date,
                            MIN(created_at) as first_order_date,
                            COUNT(*) FILTER (
                                WHERE created_at > CURRENT_DATE - 90
                            ) as orders_last_90d
                        FROM orders.orders
                        WHERE customer_id = :cid
                        GROUP BY customer_id
                    ),
                    support_stats AS (
                        SELECT
                            customer_id,
                            COUNT(*) as total_tickets,
                            AVG(CASE
                                WHEN satisfaction_score IS NOT NULL
                                THEN satisfaction_score
                            END) as avg_csat,
                            COUNT(*) FILTER (
                                WHERE status = 'open'
                            ) as open_tickets
                        FROM support.tickets
                        WHERE customer_id = :cid
                        GROUP BY customer_id
                    )
                    SELECT
                        o.*,
                        s.total_tickets,
                        s.avg_csat,
                        s.open_tickets,
                        EXTRACT(DAY FROM CURRENT_DATE - o.last_order_date)
                            as days_since_last_order,
                        EXTRACT(DAY FROM o.last_order_date - o.first_order_date)
                            / NULLIF(o.total_orders - 1, 0) as avg_days_between_orders
                    FROM order_stats o
                    LEFT JOIN support_stats s ON o.customer_id = s.customer_id
                """),
                {"cid": customer_id}
            ).fetchone()

            if not metrics:
                return {"error": f"No data found for customer {customer_id}"}

            metrics_dict = dict(metrics._mapping)

            # Calculate derived metrics
            tenure_days = (
                datetime.now() - metrics_dict.get("first_order_date", datetime.now())
            ).days if metrics_dict.get("first_order_date") else 0

            # Compute engagement score (0-100)
            recency_score = max(0, 100 - metrics_dict.get("days_since_last_order", 100))
            frequency_score = min(100, (metrics_dict.get("orders_last_90d", 0) / 3) * 100)
            monetary_score = min(100, (metrics_dict.get("lifetime_revenue", 0) / 5000) * 100)
            engagement_score = (recency_score * 0.4 + frequency_score * 0.35 + monetary_score * 0.25)

            return {
                "customer_id": customer_id,
                "lifetime_revenue": float(metrics_dict.get("lifetime_revenue", 0)),
                "total_orders": int(metrics_dict.get("total_orders", 0)),
                "avg_order_value": float(metrics_dict.get("avg_order_value", 0)),
                "days_since_last_order": int(metrics_dict.get("days_since_last_order", 0)),
                "avg_days_between_orders": float(metrics_dict.get("avg_days_between_orders") or 0),
                "orders_last_90d": int(metrics_dict.get("orders_last_90d", 0)),
                "total_support_tickets": int(metrics_dict.get("total_tickets") or 0),
                "avg_csat": float(metrics_dict.get("avg_csat") or 0),
                "open_tickets": int(metrics_dict.get("open_tickets") or 0),
                "tenure_days": tenure_days,
                "engagement_score": round(engagement_score, 1),
            }


class ChurnRiskScoringTool(BaseTool):
    """Tool for calculating ML-based churn risk scores."""

    name: str = "calculate_churn_risk"
    description: str = (
        "Calculate churn risk score (0-100) using ML model based on "
        "behavioral patterns, support history, and engagement metrics."
    )
    args_schema: type[BaseModel] = CustomerIdInput

    def __init__(self, model_path: str = "/models/churn_model.joblib"):
        super().__init__()
        self.engine = create_engine(os.environ["DATABASE_URL"])
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        """Lazy-load the churn prediction model."""
        if self._model is None:
            try:
                self._model = joblib.load(self.model_path)
            except FileNotFoundError:
                logger.warning("Churn model not found, using rule-based fallback")
                self._model = None
        return self._model

    def _rule_based_score(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based scoring when ML model unavailable."""
        score = 50  # Base score
        reasons = []

        # Recency factor
        days_inactive = features.get("days_since_last_order", 0)
        if days_inactive > 90:
            score += 25
            reasons.append(f"Inactive for {days_inactive} days")
        elif days_inactive > 60:
            score += 15
            reasons.append(f"No orders in {days_inactive} days")
        elif days_inactive < 30:
            score -= 15
            reasons.append("Recent purchase activity")

        # Support sentiment
        avg_csat = features.get("avg_csat", 0)
        if avg_csat and avg_csat < 3:
            score += 20
            reasons.append(f"Low satisfaction ({avg_csat:.1f}/5)")
        elif avg_csat and avg_csat > 4:
            score -= 10
            reasons.append("High satisfaction score")

        # Open issues
        open_tickets = features.get("open_tickets", 0)
        if open_tickets > 2:
            score += 15
            reasons.append(f"{open_tickets} unresolved tickets")

        # Engagement trend
        engagement = features.get("engagement_score", 50)
        if engagement < 30:
            score += 10
            reasons.append("Declining engagement")
        elif engagement > 70:
            score -= 10
            reasons.append("Strong engagement")

        return {
            "churn_risk_score": max(0, min(100, score)),
            "risk_level": "High" if score > 70 else "Medium" if score > 40 else "Low",
            "risk_factors": reasons,
            "model_type": "rule_based",
        }

    def _run(self, customer_id: str) -> Dict[str, Any]:
        """Calculate churn risk score."""
        # First get enriched profile
        enricher = ProfileEnrichmentTool()
        profile = enricher._run(customer_id)

        if "error" in profile:
            return profile

        # Prepare features for model
        features = {
            "days_since_last_order": profile["days_since_last_order"],
            "orders_last_90d": profile["orders_last_90d"],
            "avg_order_value": profile["avg_order_value"],
            "total_support_tickets": profile["total_support_tickets"],
            "avg_csat": profile["avg_csat"],
            "open_tickets": profile["open_tickets"],
            "engagement_score": profile["engagement_score"],
            "tenure_days": profile["tenure_days"],
        }

        if self.model:
            # Use ML model
            feature_array = np.array([[
                features["days_since_last_order"],
                features["orders_last_90d"],
                features["avg_order_value"],
                features["total_support_tickets"],
                features["avg_csat"],
                features["engagement_score"],
            ]])

            proba = self.model.predict_proba(feature_array)[0][1]
            score = int(proba * 100)

            return {
                "customer_id": customer_id,
                "churn_risk_score": score,
                "risk_level": "High" if score > 70 else "Medium" if score > 40 else "Low",
                "model_type": "ml_xgboost",
                "features_used": features,
            }
        else:
            # Fallback to rules
            result = self._rule_based_score(features)
            result["customer_id"] = customer_id
            return result`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement LangGraph-based orchestration that coordinates the agents, manages state transitions, and handles real-time profile refresh requests.',
            toolsUsed: ['LangGraph', 'Redis', 'PostgreSQL'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Orchestration for Customer 360',
                description: 'State machine orchestration for the customer-360 multi-agent pipeline.',
                code: `# orchestration/customer360_graph.py
"""LangGraph orchestration for Customer 360 multi-agent system."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence, Literal
from pydantic import BaseModel
import redis
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Customer360State(TypedDict):
    """State schema for customer-360 pipeline."""
    # Input
    lookup_key: str  # email, phone, or customer_id
    lookup_type: str  # 'email', 'phone', 'customer_id'

    # Identity resolution
    canonical_id: str | None
    identity_confidence: float
    identity_method: str

    # Enriched profile
    profile_data: dict | None
    enrichment_timestamp: str | None

    # Risk assessment
    churn_risk_score: int | None
    risk_level: str | None
    risk_factors: list[str]

    # Output
    unified_profile: dict | None
    processing_status: str
    errors: list[str]


class Customer360Orchestrator:
    """LangGraph-based orchestrator for customer-360 pipeline."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the customer-360 state graph."""
        graph = StateGraph(Customer360State)

        # Add nodes
        graph.add_node("resolve_identity", self._resolve_identity)
        graph.add_node("enrich_profile", self._enrich_profile)
        graph.add_node("score_risk", self._score_risk)
        graph.add_node("assemble_profile", self._assemble_profile)
        graph.add_node("cache_profile", self._cache_profile)

        # Add edges
        graph.add_edge("resolve_identity", "enrich_profile")
        graph.add_edge("enrich_profile", "score_risk")
        graph.add_edge("score_risk", "assemble_profile")
        graph.add_edge("assemble_profile", "cache_profile")
        graph.add_edge("cache_profile", END)

        # Set entry point
        graph.set_entry_point("resolve_identity")

        return graph.compile(checkpointer=self.checkpointer)

    def _resolve_identity(self, state: Customer360State) -> Customer360State:
        """Node: Resolve customer identity across systems."""
        from tools.identity_resolution_tools import (
            DeterministicMatchTool, ProbabilisticMatchTool
        )

        lookup_key = state["lookup_key"]
        lookup_type = state["lookup_type"]

        # Try deterministic match first
        det_tool = DeterministicMatchTool()
        if lookup_type == "email":
            result = det_tool._run(email=lookup_key)
        elif lookup_type == "phone":
            result = det_tool._run(phone=lookup_key)
        else:
            result = {"matches": [{"customer_id": lookup_key, "confidence": 1.0}]}

        if result["match_count"] > 0:
            best_match = result["matches"][0]
            return {
                **state,
                "canonical_id": best_match["customer_id"],
                "identity_confidence": best_match.get("confidence", 1.0),
                "identity_method": "deterministic",
                "processing_status": "identity_resolved",
            }

        # Fallback to probabilistic
        prob_tool = ProbabilisticMatchTool()
        result = prob_tool._run(name=lookup_key)

        if result["match_count"] > 0:
            best_match = result["matches"][0]
            return {
                **state,
                "canonical_id": best_match["customer_id"],
                "identity_confidence": best_match["confidence"],
                "identity_method": "probabilistic",
                "processing_status": "identity_resolved",
            }

        return {
            **state,
            "canonical_id": None,
            "identity_confidence": 0.0,
            "identity_method": "none",
            "processing_status": "identity_not_found",
            "errors": state.get("errors", []) + ["No matching customer found"],
        }

    def _enrich_profile(self, state: Customer360State) -> Customer360State:
        """Node: Enrich profile with behavioral metrics."""
        if not state.get("canonical_id"):
            return state

        from tools.profile_analysis_tools import ProfileEnrichmentTool

        tool = ProfileEnrichmentTool()
        profile = tool._run(state["canonical_id"])

        return {
            **state,
            "profile_data": profile,
            "enrichment_timestamp": datetime.utcnow().isoformat(),
            "processing_status": "profile_enriched",
        }

    def _score_risk(self, state: Customer360State) -> Customer360State:
        """Node: Calculate churn risk score."""
        if not state.get("canonical_id"):
            return state

        from tools.profile_analysis_tools import ChurnRiskScoringTool

        tool = ChurnRiskScoringTool()
        risk = tool._run(state["canonical_id"])

        return {
            **state,
            "churn_risk_score": risk.get("churn_risk_score"),
            "risk_level": risk.get("risk_level"),
            "risk_factors": risk.get("risk_factors", []),
            "processing_status": "risk_scored",
        }

    def _assemble_profile(self, state: Customer360State) -> Customer360State:
        """Node: Assemble unified profile from all components."""
        unified = {
            "customer_id": state.get("canonical_id"),
            "identity": {
                "confidence": state.get("identity_confidence"),
                "method": state.get("identity_method"),
            },
            "profile": state.get("profile_data"),
            "risk": {
                "score": state.get("churn_risk_score"),
                "level": state.get("risk_level"),
                "factors": state.get("risk_factors", []),
            },
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "version": "1.0",
            },
        }

        return {
            **state,
            "unified_profile": unified,
            "processing_status": "assembled",
        }

    def _cache_profile(self, state: Customer360State) -> Customer360State:
        """Node: Cache unified profile in Redis."""
        if not state.get("unified_profile") or not state.get("canonical_id"):
            return state

        cache_key = f"customer360:{state['canonical_id']}"
        self.redis.setex(
            cache_key,
            3600,  # 1 hour TTL
            json.dumps(state["unified_profile"])
        )

        logger.info(f"Cached profile for {state['canonical_id']}")

        return {
            **state,
            "processing_status": "completed",
        }

    async def get_profile(
        self,
        lookup_key: str,
        lookup_type: str = "email"
    ) -> dict:
        """Get unified customer profile, using cache when available."""
        # Check cache first if we have customer_id
        if lookup_type == "customer_id":
            cached = self.redis.get(f"customer360:{lookup_key}")
            if cached:
                logger.info(f"Cache hit for {lookup_key}")
                return json.loads(cached)

        # Run the graph
        initial_state: Customer360State = {
            "lookup_key": lookup_key,
            "lookup_type": lookup_type,
            "canonical_id": None,
            "identity_confidence": 0.0,
            "identity_method": "",
            "profile_data": None,
            "enrichment_timestamp": None,
            "churn_risk_score": None,
            "risk_level": None,
            "risk_factors": [],
            "unified_profile": None,
            "processing_status": "started",
            "errors": [],
        }

        config = {"configurable": {"thread_id": lookup_key}}
        result = await self.graph.ainvoke(initial_state, config)

        return result.get("unified_profile", {"error": "Profile generation failed"})`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the multi-agent system with Docker, implement health checks, and set up LangSmith tracing for observability and debugging.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'FastAPI'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Customer 360 API with Observability',
                description: 'FastAPI service exposing the multi-agent customer-360 system with LangSmith tracing.',
                code: `# api/customer360_service.py
"""Customer 360 API service with observability."""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
from langsmith import traceable
import structlog
import time
import os

from orchestration.customer360_graph import Customer360Orchestrator

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Prometheus metrics
PROFILE_REQUESTS = Counter(
    "customer360_profile_requests_total",
    "Total profile requests",
    ["lookup_type", "status"]
)
PROFILE_LATENCY = Histogram(
    "customer360_profile_latency_seconds",
    "Profile generation latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
CACHE_HITS = Counter(
    "customer360_cache_hits_total",
    "Cache hit count"
)

app = FastAPI(
    title="Customer 360 Multi-Agent API",
    description="Unified customer profile service powered by AI agents",
    version="2.0.0"
)

orchestrator = Customer360Orchestrator(
    redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379")
)


class ProfileRequest(BaseModel):
    """Request schema for profile lookup."""
    lookup_key: str = Field(..., description="Email, phone, or customer ID")
    lookup_type: str = Field(
        default="email",
        description="Type of lookup: email, phone, or customer_id"
    )


class ProfileResponse(BaseModel):
    """Response schema for unified profile."""
    customer_id: str | None
    identity: dict
    profile: dict | None
    risk: dict
    metadata: dict


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "customer360"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@app.post("/api/v2/profile", response_model=ProfileResponse)
@traceable(name="get_customer_profile")
async def get_customer_profile(request: ProfileRequest):
    """
    Get unified customer profile using multi-agent system.

    The system will:
    1. Resolve customer identity across systems
    2. Enrich profile with behavioral metrics
    3. Calculate churn risk score
    4. Return unified profile
    """
    start_time = time.time()

    logger.info(
        "profile_request_received",
        lookup_key=request.lookup_key[:3] + "***",
        lookup_type=request.lookup_type
    )

    try:
        profile = await orchestrator.get_profile(
            lookup_key=request.lookup_key,
            lookup_type=request.lookup_type
        )

        if "error" in profile:
            PROFILE_REQUESTS.labels(
                lookup_type=request.lookup_type,
                status="not_found"
            ).inc()
            raise HTTPException(status_code=404, detail=profile["error"])

        # Record metrics
        latency = time.time() - start_time
        PROFILE_LATENCY.observe(latency)
        PROFILE_REQUESTS.labels(
            lookup_type=request.lookup_type,
            status="success"
        ).inc()

        logger.info(
            "profile_request_completed",
            customer_id=profile.get("customer_id"),
            latency_ms=round(latency * 1000, 2)
        )

        return ProfileResponse(**profile)

    except HTTPException:
        raise
    except Exception as e:
        PROFILE_REQUESTS.labels(
            lookup_type=request.lookup_type,
            status="error"
        ).inc()
        logger.error("profile_request_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Profile generation failed")


@app.post("/api/v2/profile/refresh")
@traceable(name="refresh_customer_profile")
async def refresh_profile(
    request: ProfileRequest,
    background_tasks: BackgroundTasks
):
    """
    Force refresh a customer profile, bypassing cache.
    """
    # Invalidate cache
    if request.lookup_type == "customer_id":
        orchestrator.redis.delete(f"customer360:{request.lookup_key}")

    # Trigger async refresh
    background_tasks.add_task(
        orchestrator.get_profile,
        request.lookup_key,
        request.lookup_type
    )

    return {"status": "refresh_scheduled", "lookup_key": request.lookup_key}`,
              },
              {
                language: 'yaml',
                title: 'Docker Compose for Multi-Agent Deployment',
                description: 'Docker Compose configuration for deploying the customer-360 multi-agent system.',
                code: `# docker-compose.customer360-agents.yml
version: '3.8'

services:
  customer360-api:
    build:
      context: .
      dockerfile: Dockerfile.customer360
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=\${DATABASE_URL}
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=customer360-agents
    depends_on:
      - redis
      - postgres
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=warehouse
      - POSTGRES_USER=\${DB_USER}
      - POSTGRES_PASSWORD=\${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U \${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards

volumes:
  redis-data:
  postgres-data:
  grafana-data:`,
              },
            ],
          },
        ],
      },
    },

    /* ──────────────────────────────────────────────
       Pain Point 2 — Feedback Loop Failure
       ────────────────────────────────────────────── */
    {
      id: 'feedback-loop-failure',
      number: 2,
      title: 'Feedback Loop Failure',
      subtitle: 'Customer Feedback Data Never Reaches Product',
      summary:
        'Thousands of support tickets contain product insights that never reach your product team. You are solving the same bugs twice.',
      tags: ['feedback', 'product-alignment', 'nlp'],
      metrics: {
        annualCostRange: '$400K - $1.5M',
        roi: '6x',
        paybackPeriod: '3-5 months',
        investmentRange: '$60K - $100K',
      },
      price: {
        present: {
          title: 'Present — The Symptom',
          description:
            'Support tickets pile up with recurring product complaints, but the product team learns about issues weeks later — or never.',
          bullets: [
            'Product roadmap is driven by stakeholder opinion, not quantified customer pain',
            'The same defect generates 200+ tickets before engineering hears about it',
            'Support managers manually forward "important" tickets — an unreliable, subjective filter',
            'NPS detractors cite issues that already have open tickets dating back months',
          ],
          severity: 'high',
        },
        root: {
          title: 'Root Cause — Why It Persists',
          description:
            'No automated pipeline classifies, quantifies, and routes support feedback into the product backlog.',
          bullets: [
            'Ticket taxonomies are free-text or shallow dropdown categories that lack product-level granularity',
            'Support and product teams use separate tools with no integration layer',
            'NLP classification has never been applied to the ticket corpus',
            'No one owns the feedback-to-product handoff as a formal process',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Impact — Business Damage',
          description:
            'Product teams build features customers did not ask for while ignoring defects customers scream about.',
          bullets: [
            'Wasted engineering sprints on low-signal features cost $400K-$1.5M annually',
            'Customer churn driven by unresolved known issues that were never escalated',
            'Support agents lose morale when they see the same bug reopened quarter after quarter',
            'Competitor products fix identical issues faster because they close the feedback loop',
          ],
          severity: 'high',
        },
        cost: {
          title: 'Cost of Inaction — 12-Month Horizon',
          description:
            'Every quarter without a feedback pipeline means another cycle of misallocated engineering effort and preventable churn.',
          bullets: [
            'Engineering spends 20% of sprint capacity on features with no customer signal backing them',
            'Repeat-ticket volume grows 8% quarter-over-quarter as product gaps widen',
            'Support headcount must grow linearly with ticket volume since root causes go unfixed',
            'Customer trust erodes as reported issues go unacknowledged by the product org',
          ],
          severity: 'medium',
        },
        expectedReturn: {
          title: 'Expected Return — Post-Implementation',
          description:
            'An automated NLP pipeline turns every ticket into a structured product signal, cutting wasted sprints and repeat tickets.',
          bullets: [
            'Product backlog is ranked by real customer pain volume, not gut feel',
            'Repeat-ticket clusters drop 40% within two quarters as root causes get prioritized',
            'Engineering allocates 15% more capacity to high-signal work',
            'Time from customer report to engineering awareness falls from weeks to hours',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Deploy an NLP classification and sentiment pipeline over your ticket data, then materialize a feedback-to-product dashboard that auto-ranks issues by frequency, severity, and revenue impact.',
        prerequisites: [
          'Access to support ticket data (Zendesk, Freshdesk, or database export)',
          'Python 3.10+ with transformers and scikit-learn',
          'A data warehouse for the classified ticket output',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        toolsUsed: ['Python', 'Hugging Face Transformers', 'SQL', 'scikit-learn', 'pytest', 'Docker', 'GitHub Actions', 'cron / Airflow', 'Slack API'],
        steps: [
          {
            stepNumber: 1,
            title: 'NLP Ticket Classification',
            description:
              'Classify raw support tickets into product-area categories and defect types using a fine-tuned transformer model, producing structured labels from unstructured text.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Ticket Classification Pipeline',
                description:
                  'Uses a zero-shot classification model to tag tickets with product-area labels and defect types without manual labeling.',
                code: `# ticket_classifier.py
from transformers import pipeline
from dataclasses import dataclass
import json

PRODUCT_AREAS = [
    "checkout-flow", "search-results", "payment-processing",
    "account-management", "shipping-delivery", "returns-refunds",
    "mobile-app", "notifications", "pricing-discounts",
]

DEFECT_TYPES = ["bug", "ux-friction", "feature-request", "documentation-gap", "performance"]

@dataclass
class ClassifiedTicket:
    ticket_id: str
    product_area: str
    product_area_confidence: float
    defect_type: str
    defect_type_confidence: float
    sentiment_score: float

def build_classifiers():
    area_clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return area_clf, sentiment

def classify_ticket(ticket_id: str, text: str, area_clf, sentiment_clf) -> ClassifiedTicket:
    area_result = area_clf(text, candidate_labels=PRODUCT_AREAS)
    defect_result = area_clf(text, candidate_labels=DEFECT_TYPES)
    sent_result = sentiment_clf(text[:512])[0]
    sentiment_score = sent_result["score"] if sent_result["label"] == "POSITIVE" else -sent_result["score"]

    return ClassifiedTicket(
        ticket_id=ticket_id,
        product_area=area_result["labels"][0],
        product_area_confidence=round(area_result["scores"][0], 3),
        defect_type=defect_result["labels"][0],
        defect_type_confidence=round(defect_result["scores"][0], 3),
        sentiment_score=round(sentiment_score, 3),
    )`,
              },
              {
                language: 'python',
                title: 'Batch Processing and Warehouse Loading',
                description:
                  'Processes tickets in batches and writes classified output to the warehouse for downstream analytics.',
                code: `# batch_classify.py
import pandas as pd
from sqlalchemy import create_engine
from ticket_classifier import build_classifiers, classify_ticket
import os

def process_ticket_batch(batch_size: int = 500):
    engine = create_engine(os.environ["DATABASE_URL"])
    area_clf, sentiment_clf = build_classifiers()

    unclassified = pd.read_sql("""
        SELECT t.ticket_id, t.subject || ' ' || t.body AS full_text
        FROM support.tickets t
        LEFT JOIN analytics.classified_tickets ct ON t.ticket_id = ct.ticket_id
        WHERE ct.ticket_id IS NULL
        ORDER BY t.created_at DESC
        LIMIT %(limit)s
    """, engine, params={"limit": batch_size})

    results = []
    for _, row in unclassified.iterrows():
        classified = classify_ticket(row["ticket_id"], row["full_text"], area_clf, sentiment_clf)
        results.append(classified.__dict__)

    if results:
        df = pd.DataFrame(results)
        df.to_sql("classified_tickets", engine, schema="analytics",
                   if_exists="append", index=False, method="multi")
        print(f"Classified and loaded {len(results)} tickets")
    return len(results)`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Feedback-to-Product Pipeline',
            description:
              'Aggregate classified tickets into a product-team-ready view that ranks issues by frequency, customer revenue impact, and sentiment severity.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Product Feedback Aggregation View',
                description:
                  'Materializes a ranked view of product issues combining ticket volume, affected revenue, and average sentiment.',
                code: `-- feedback_to_product_pipeline.sql
-- Aggregated product feedback ranked by business impact
CREATE OR REPLACE VIEW analytics.product_feedback_ranked AS
WITH ticket_impact AS (
  SELECT
    ct.product_area,
    ct.defect_type,
    COUNT(*)                                          AS ticket_count,
    COUNT(DISTINCT t.customer_id)                     AS affected_customers,
    SUM(c360.lifetime_revenue)                        AS affected_revenue,
    AVG(ct.sentiment_score)                           AS avg_sentiment,
    MIN(t.created_at)                                 AS first_reported,
    MAX(t.created_at)                                 AS last_reported
  FROM analytics.classified_tickets ct
  JOIN support.tickets t            ON ct.ticket_id = t.ticket_id
  LEFT JOIN analytics.customer_360 c360 ON t.customer_id = c360.customer_id
  WHERE t.created_at >= CURRENT_DATE - INTERVAL '90 days'
  GROUP BY ct.product_area, ct.defect_type
),
scored AS (
  SELECT
    *,
    NTILE(100) OVER (ORDER BY ticket_count)           AS volume_pctl,
    NTILE(100) OVER (ORDER BY affected_revenue)       AS revenue_pctl,
    NTILE(100) OVER (ORDER BY avg_sentiment ASC)      AS severity_pctl,
    ROUND((0.4 * NTILE(100) OVER (ORDER BY ticket_count)
         + 0.35 * NTILE(100) OVER (ORDER BY affected_revenue)
         + 0.25 * NTILE(100) OVER (ORDER BY avg_sentiment ASC)), 1)
      AS priority_score
  FROM ticket_impact
)
SELECT * FROM scored
ORDER BY priority_score DESC;`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate the NLP classification pipeline with SQL data-quality assertions on the classified-ticket output and a pytest suite that verifies classification accuracy, label distribution, and sentiment consistency.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Classified Ticket Data Quality Assertions',
                description:
                  'SQL assertion queries that verify the NLP pipeline produced valid labels, acceptable confidence scores, and no orphaned ticket references.',
                code: `-- test_classified_tickets_quality.sql
-- Assertion 1: Every classified ticket exists in the source table
SELECT 'orphaned_classified_tickets' AS assertion,
       COUNT(*) AS violations
FROM analytics.classified_tickets ct
LEFT JOIN support.tickets t ON ct.ticket_id = t.ticket_id
WHERE t.ticket_id IS NULL
HAVING COUNT(*) > 0;

-- Assertion 2: No NULL product areas
SELECT 'null_product_area' AS assertion,
       COUNT(*) AS violations
FROM analytics.classified_tickets
WHERE product_area IS NULL
HAVING COUNT(*) > 0;

-- Assertion 3: Confidence scores within [0, 1]
SELECT 'invalid_confidence_scores' AS assertion,
       COUNT(*) AS violations
FROM analytics.classified_tickets
WHERE product_area_confidence < 0
   OR product_area_confidence > 1
   OR defect_type_confidence < 0
   OR defect_type_confidence > 1
HAVING COUNT(*) > 0;

-- Assertion 4: Classification coverage — at least 95% of recent tickets classified
SELECT 'low_classification_coverage' AS assertion,
       ROUND(100.0 * classified / total, 2) AS coverage_pct
FROM (
  SELECT COUNT(*) AS total,
         COUNT(ct.ticket_id) AS classified
  FROM support.tickets t
  LEFT JOIN analytics.classified_tickets ct ON t.ticket_id = ct.ticket_id
  WHERE t.created_at >= CURRENT_DATE - INTERVAL '7 days'
) sub
WHERE classified::float / GREATEST(total, 1) < 0.95;`,
              },
              {
                language: 'python',
                title: 'Pytest NLP Pipeline Validation Suite',
                description:
                  'A pytest-based validation suite with typed test classes that verify classification label distribution, confidence thresholds, and sentiment-score consistency.',
                code: `# tests/test_ticket_classification_pipeline.py
"""Validation suite for the NLP ticket classification pipeline."""
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


@dataclass
class ClassificationThresholds:
    """Typed thresholds for classification quality checks."""
    min_coverage_rate: float = 0.95
    min_avg_confidence: float = 0.55
    max_label_skew_ratio: float = 0.60
    sentiment_bound: float = 1.0


class TestTicketClassificationQuality:
    """Data-quality tests for analytics.classified_tickets."""

    engine = create_engine("postgresql://localhost/warehouse")
    thresholds = ClassificationThresholds()

    def _query_scalar(self, sql: str) -> Any:
        with self.engine.connect() as conn:
            return conn.execute(text(sql)).scalar()

    def _query_df(self, sql: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn)

    def test_classification_coverage(self) -> None:
        total: int = self._query_scalar(
            "SELECT COUNT(*) FROM support.tickets"
            " WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'"
        )
        classified: int = self._query_scalar(
            "SELECT COUNT(*) FROM analytics.classified_tickets ct"
            " JOIN support.tickets t ON ct.ticket_id = t.ticket_id"
            " WHERE t.created_at >= CURRENT_DATE - INTERVAL '7 days'"
        )
        rate: float = classified / max(total, 1)
        logger.info("Classification coverage: %.2f%%", rate * 100)
        assert rate >= self.thresholds.min_coverage_rate

    def test_average_confidence(self) -> None:
        avg_conf: float = self._query_scalar(
            "SELECT AVG(product_area_confidence)"
            " FROM analytics.classified_tickets"
            " WHERE product_area_confidence IS NOT NULL"
        )
        logger.info("Average product-area confidence: %.3f", avg_conf)
        assert avg_conf >= self.thresholds.min_avg_confidence

    def test_label_distribution_not_skewed(self) -> None:
        df = self._query_df(
            "SELECT product_area, COUNT(*) AS cnt"
            " FROM analytics.classified_tickets"
            " GROUP BY product_area ORDER BY cnt DESC"
        )
        if len(df) > 0:
            top_share: float = df.iloc[0]["cnt"] / df["cnt"].sum()
            logger.info("Top-label share: %.2f%%", top_share * 100)
            assert top_share <= self.thresholds.max_label_skew_ratio

    def test_sentiment_scores_bounded(self) -> None:
        violations: int = self._query_scalar(
            "SELECT COUNT(*) FROM analytics.classified_tickets"
            " WHERE ABS(sentiment_score) > 1.0"
        )
        logger.info("Sentiment out-of-bound violations: %d", violations)
        assert violations == 0`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Containerize the NLP classification pipeline with Docker, deploy via a scripted workflow that validates model artifacts and runs integration tests, and manage configuration with typed dataclasses.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Feedback Pipeline Deployment Script',
                description:
                  'End-to-end deployment script that builds the NLP classification Docker image, validates model weights exist, runs the pytest suite, and schedules the batch classifier.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────
APP_NAME="feedback-classifier"
IMAGE_TAG="\${APP_NAME}:\$(git rev-parse --short HEAD)"
COMPOSE_FILE="docker-compose.feedback.yml"
MODEL_CACHE="/models/bart-large-mnli"
CRON_SCHEDULE="0 */4 * * *"

echo "==> Building Docker image \${IMAGE_TAG}"
docker build -t "\${IMAGE_TAG}" -f Dockerfile.feedback .

echo "==> Verifying model artifacts in \${MODEL_CACHE}"
if [ ! -d "\${MODEL_CACHE}" ]; then
  echo "ERROR: Model cache not found at \${MODEL_CACHE}" >&2
  echo "Run: python -c 'from transformers import pipeline; pipeline(\"zero-shot-classification\", model=\"facebook/bart-large-mnli\")'"
  exit 1
fi

echo "==> Running pytest validation suite"
docker run --rm --env-file .env \
  -v "\${MODEL_CACHE}:/models/bart-large-mnli:ro" \
  "\${IMAGE_TAG}" \
  python -m pytest tests/test_ticket_classification_pipeline.py -v --tb=short

echo "==> Starting services via docker-compose"
docker-compose -f "\${COMPOSE_FILE}" up -d

echo "==> Registering cron schedule: \${CRON_SCHEDULE}"
CRON_CMD="docker run --rm --env-file /opt/\${APP_NAME}/.env \${IMAGE_TAG} python batch_classify.py"
(crontab -l 2>/dev/null | grep -v "\${APP_NAME}"; echo "\${CRON_SCHEDULE} \${CRON_CMD} # \${APP_NAME}") | crontab -

echo "==> Deployment complete — \${APP_NAME} scheduled"`,
              },
              {
                language: 'python',
                title: 'Typed Configuration Loader',
                description:
                  'A dataclass-based configuration loader that reads environment variables with validation and defaults for the NLP feedback classification service.',
                code: `# config/feedback_config.py
"""Typed configuration loader for the feedback classification pipeline."""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabaseConfig:
    """Connection settings for the analytics warehouse."""
    url: str = ""
    pool_min: int = 2
    pool_max: int = 10


@dataclass(frozen=True)
class ModelConfig:
    """NLP model settings."""
    area_model: str = "facebook/bart-large-mnli"
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    batch_size: int = 500
    confidence_threshold: float = 0.40
    cache_dir: str = "/models"


@dataclass(frozen=True)
class SchedulerConfig:
    """Batch classification scheduling settings."""
    cron_expression: str = "0 */4 * * *"
    max_retries: int = 3
    retry_delay_seconds: int = 120


@dataclass(frozen=True)
class FeedbackPipelineConfig:
    """Top-level configuration for the feedback classification pipeline."""
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    slack_webhook: str = ""

    @classmethod
    def from_env(cls) -> FeedbackPipelineConfig:
        """Build config from environment variables with validation."""
        cfg = cls(
            db=DatabaseConfig(
                url=os.environ["DATABASE_URL"],
                pool_min=int(os.getenv("DB_POOL_MIN", "2")),
                pool_max=int(os.getenv("DB_POOL_MAX", "10")),
            ),
            model=ModelConfig(
                area_model=os.getenv("AREA_MODEL", "facebook/bart-large-mnli"),
                sentiment_model=os.getenv(
                    "SENTIMENT_MODEL",
                    "distilbert-base-uncased-finetuned-sst-2-english",
                ),
                batch_size=int(os.getenv("CLASSIFY_BATCH_SIZE", "500")),
                confidence_threshold=float(os.getenv("MIN_CONFIDENCE", "0.40")),
                cache_dir=os.getenv("MODEL_CACHE_DIR", "/models"),
            ),
            scheduler=SchedulerConfig(
                cron_expression=os.getenv("CLASSIFY_CRON", "0 */4 * * *"),
                max_retries=int(os.getenv("CLASSIFY_MAX_RETRIES", "3")),
                retry_delay_seconds=int(os.getenv("CLASSIFY_RETRY_DELAY", "120")),
            ),
            slack_webhook=os.getenv("SLACK_PRODUCT_WEBHOOK", ""),
        )
        logger.info(
            "Loaded FeedbackPipelineConfig — model=%s, batch=%d, cron=%s",
            cfg.model.area_model, cfg.model.batch_size,
            cfg.scheduler.cron_expression,
        )
        return cfg`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Detect sudden ticket spikes per product area and alert the product team via Slack before the problem compounds, and continuously monitor classification pipeline health with anomaly detection.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Feedback Spike Detection and Alerting',
                description:
                  'Monitors classified ticket inflow and sends Slack alerts when a product area exceeds its rolling-average threshold.',
                code: `# feedback_spike_alert.py
import pandas as pd
from sqlalchemy import create_engine
import requests, os
from datetime import datetime, timedelta

SLACK_WEBHOOK = os.environ["SLACK_PRODUCT_WEBHOOK"]
SPIKE_THRESHOLD = 2.5  # alert if today > 2.5x the 14-day average

def check_feedback_spikes():
    engine = create_engine(os.environ["DATABASE_URL"])
    df = pd.read_sql("""
        SELECT product_area,
               COUNT(*) FILTER (WHERE t.created_at >= CURRENT_DATE)       AS today_count,
               COUNT(*) FILTER (WHERE t.created_at >= CURRENT_DATE - 14)
                 / 14.0                                                    AS daily_avg_14d
        FROM analytics.classified_tickets ct
        JOIN support.tickets t ON ct.ticket_id = t.ticket_id
        WHERE t.created_at >= CURRENT_DATE - INTERVAL '14 days'
        GROUP BY product_area
    """, engine)

    spikes = df[df["today_count"] > df["daily_avg_14d"] * SPIKE_THRESHOLD]

    for _, row in spikes.iterrows():
        payload = {
            "text": (
                f":rotating_light: *Feedback Spike Detected*\\n"
                f"*Area:* {row['product_area']}\\n"
                f"*Today:* {int(row['today_count'])} tickets "
                f"(avg: {row['daily_avg_14d']:.1f}/day)\\n"
                f"*Multiplier:* {row['today_count'] / row['daily_avg_14d']:.1f}x\\n"
                f"<{os.environ['DASHBOARD_URL']}|View Dashboard>"
            )
        }
        requests.post(SLACK_WEBHOOK, json=payload, timeout=10)

    return len(spikes)`,
              },
              {
                language: 'python',
                title: 'Classification Pipeline Health Monitor',
                description:
                  'Monitors the NLP pipeline for classification drift, confidence degradation, and backlog growth, sending Slack alerts when thresholds are breached.',
                code: `# monitor_feedback_pipeline.py
"""Monitoring and alerting for the NLP feedback classification pipeline."""
import logging
from dataclasses import dataclass
from datetime import datetime

import requests
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


@dataclass
class PipelineThresholds:
    max_unclassified_backlog: int = 200
    min_avg_confidence: float = 0.50
    max_staleness_hours: float = 6.0


def check_feedback_pipeline_health(
    db_url: str,
    slack_webhook: str,
    thresholds: PipelineThresholds | None = None,
) -> list[str]:
    """Run health checks and send Slack alerts for any violations."""
    thresholds = thresholds or PipelineThresholds()
    engine = create_engine(db_url)
    alerts: list[str] = []

    with engine.connect() as conn:
        # Check unclassified backlog
        backlog: int = conn.execute(text(
            "SELECT COUNT(*) FROM support.tickets t"
            " LEFT JOIN analytics.classified_tickets ct"
            "   ON t.ticket_id = ct.ticket_id"
            " WHERE ct.ticket_id IS NULL"
            "   AND t.created_at >= CURRENT_DATE - INTERVAL '7 days'"
        )).scalar() or 0

        if backlog > thresholds.max_unclassified_backlog:
            alerts.append(
                f"*Backlog alert:* {backlog:,} unclassified tickets"
                f" (limit: {thresholds.max_unclassified_backlog:,})"
            )

        # Check confidence degradation
        avg_conf: float = conn.execute(text(
            "SELECT AVG(product_area_confidence)"
            " FROM analytics.classified_tickets"
            " WHERE product_area_confidence IS NOT NULL"
        )).scalar() or 0.0

        if avg_conf < thresholds.min_avg_confidence:
            alerts.append(
                f"*Confidence alert:* avg confidence {avg_conf:.3f}"
                f" below {thresholds.min_avg_confidence:.3f} threshold"
            )

        # Check pipeline freshness
        hours: float = conn.execute(text(
            "SELECT EXTRACT(EPOCH FROM"
            "  (CURRENT_TIMESTAMP - MAX(ct.classified_at)))"
            " / 3600.0 FROM analytics.classified_tickets ct"
        )).scalar() or 999.0

        if hours > thresholds.max_staleness_hours:
            alerts.append(
                f"*Staleness alert:* last classification"
                f" {hours:.1f}h ago (limit: {thresholds.max_staleness_hours}h)"
            )

    if alerts and slack_webhook:
        payload = {
            "text": (
                ":microscope: *Feedback Pipeline Health Check Failed*\\n"
                + "\\n".join(alerts)
                + f"\\n_Checked at {datetime.utcnow():%Y-%m-%d %H:%M UTC}_"
            )
        }
        requests.post(slack_webhook, json=payload, timeout=10)
        logger.warning("Sent %d alerts to Slack", len(alerts))

    return alerts`,
              },
            ],
          },
        ],
      },
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to automatically classify support tickets, extract product feedback themes, and route insights to product teams via Slack or Notion without building custom NLP models.',
        estimatedMonthlyCost: '$100 - $180/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Zendesk ($55/mo)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Freshdesk AI ($35/mo)', 'Intercom AI ($74/mo)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Configure Zapier to trigger on new support tickets from Zendesk or Freshdesk, extracting ticket subject, body, customer segment, and historical context for AI analysis.',
            toolsUsed: ['Zapier', 'Zendesk API', 'Webhooks'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Ticket Extraction Configuration',
                description: 'Configure Zapier to extract ticket data and customer context when a new ticket is created.',
                code: `{
  "trigger": {
    "app": "Zendesk",
    "event": "New Ticket",
    "filters": {
      "status": ["new", "open"],
      "exclude_spam": true
    }
  },
  "actions": [
    {
      "step": 1,
      "app": "Zendesk",
      "action": "Find Ticket",
      "config": {
        "ticket_id": "{{trigger_ticket_id}}",
        "include_comments": true,
        "include_requester": true
      }
    },
    {
      "step": 2,
      "app": "Webhooks by Zapier",
      "action": "GET",
      "url": "{{CUSTOMER_API_URL}}/customers/{{requester_email}}",
      "headers": {
        "Authorization": "Bearer {{CUSTOMER_API_KEY}}"
      }
    },
    {
      "step": 3,
      "app": "Zendesk",
      "action": "Find Tickets",
      "config": {
        "query": "requester:{{requester_email}} created>30daysAgo",
        "limit": 10
      }
    }
  ],
  "output": {
    "ticket_data": {
      "ticket_id": "{{trigger_ticket_id}}",
      "subject": "{{ticket_subject}}",
      "description": "{{ticket_description}}",
      "priority": "{{ticket_priority}}",
      "requester_email": "{{requester_email}}",
      "requester_name": "{{requester_name}}",
      "customer_tier": "{{step2_customer_tier}}",
      "customer_ltv": "{{step2_lifetime_value}}",
      "recent_tickets": "{{step3_ticket_subjects}}"
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
              'Send ticket content to ChatGPT or Claude for multi-dimensional classification: product area, defect type, sentiment, urgency, and extraction of actionable product feedback.',
            toolsUsed: ['ChatGPT API', 'Claude API'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Ticket Classification Prompt Template',
                description: 'A structured prompt that instructs the AI to classify tickets and extract product insights.',
                code: `system_prompt: |
  You are a support ticket analyst specializing in product feedback extraction.
  Your job is to classify tickets and identify actionable product insights.

  Classification taxonomy:
  - Product Areas: checkout-flow, search-results, payment-processing,
    account-management, shipping-delivery, returns-refunds, mobile-app,
    notifications, pricing-discounts, performance
  - Issue Types: bug, ux-friction, feature-request, documentation-gap,
    performance-issue, integration-problem
  - Sentiment: very-negative, negative, neutral, positive, very-positive
  - Urgency: critical, high, medium, low

user_prompt_template: |
  Analyze this support ticket and provide structured classification:

  ## Ticket Information
  **Subject:** {{ticket_subject}}
  **Description:** {{ticket_description}}
  **Customer Tier:** {{customer_tier}}
  **Customer LTV:** \${{customer_ltv}}

  ## Recent Ticket History
  {{recent_tickets}}

  ---
  Provide analysis in this exact JSON format:
  {
    "classification": {
      "product_area": "<primary area>",
      "product_area_secondary": "<secondary area if applicable>",
      "issue_type": "<issue type>",
      "sentiment": "<sentiment>",
      "sentiment_score": <-1.0 to 1.0>,
      "urgency": "<urgency level>"
    },
    "product_feedback": {
      "is_actionable": <true/false>,
      "feedback_summary": "<1-2 sentence summary of product insight>",
      "affected_feature": "<specific feature or flow>",
      "user_impact": "<how this affects user experience>",
      "suggested_fix": "<potential solution if apparent>"
    },
    "routing": {
      "escalate_to_product": <true/false>,
      "escalation_reason": "<why product team should see this>",
      "priority_score": <1-100 based on LTV and severity>
    }
  }

parameters:
  model: "gpt-4-turbo"
  temperature: 0.2
  max_tokens: 600
  response_format: { "type": "json_object" }`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Route AI-classified tickets to appropriate channels: update Zendesk tags, post high-priority product feedback to Slack, and aggregate insights in a Notion database for product team review.',
            toolsUsed: ['Zapier', 'Zendesk API', 'Slack', 'Notion'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Feedback Routing Workflow',
                description: 'Complete workflow that routes classified feedback to product teams and updates the ticketing system.',
                code: `{
  "workflow_name": "Ticket Classification & Product Feedback Router",
  "steps": [
    {
      "step": 4,
      "app": "OpenAI (ChatGPT)",
      "action": "Send Prompt",
      "config": {
        "model": "gpt-4-turbo",
        "system_message": "{{system_prompt}}",
        "user_message": "{{formatted_ticket_prompt}}",
        "temperature": 0.2,
        "response_format": "json"
      }
    },
    {
      "step": 5,
      "app": "Code by Zapier",
      "action": "Run JavaScript",
      "config": {
        "code": "const response = JSON.parse(inputData.chatgpt_response); return { product_area: response.classification.product_area, issue_type: response.classification.issue_type, sentiment: response.classification.sentiment, escalate: response.routing.escalate_to_product, priority_score: response.routing.priority_score, feedback_summary: response.product_feedback.feedback_summary };"
      }
    },
    {
      "step": 6,
      "app": "Zendesk",
      "action": "Update Ticket",
      "config": {
        "ticket_id": "{{trigger_ticket_id}}",
        "tags_add": ["ai-classified", "{{product_area}}", "{{issue_type}}"],
        "custom_fields": {
          "ai_sentiment": "{{sentiment}}",
          "ai_priority_score": "{{priority_score}}"
        }
      }
    },
    {
      "step": 7,
      "app": "Filter by Zapier",
      "condition": "escalate EQUALS true AND priority_score GREATER_THAN 60"
    },
    {
      "step": 8,
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#product-feedback",
        "message": ":bulb: *New Product Feedback*\\n\\n*Area:* {{product_area}}\\n*Type:* {{issue_type}}\\n*Priority:* {{priority_score}}/100\\n*Sentiment:* {{sentiment}}\\n\\n*Summary:*\\n{{feedback_summary}}\\n\\n*Customer:* {{customer_tier}} (LTV: \${{customer_ltv}})\\n<{{zendesk_ticket_url}}|View Ticket>"
      }
    },
    {
      "step": 9,
      "app": "Notion",
      "action": "Create Database Item",
      "config": {
        "database_id": "{{PRODUCT_FEEDBACK_DB_ID}}",
        "properties": {
          "Title": "{{ticket_subject}}",
          "Product Area": { "select": "{{product_area}}" },
          "Issue Type": { "select": "{{issue_type}}" },
          "Priority Score": { "number": "{{priority_score}}" },
          "Sentiment": { "select": "{{sentiment}}" },
          "Feedback Summary": { "rich_text": "{{feedback_summary}}" },
          "Customer Tier": { "select": "{{customer_tier}}" },
          "Ticket URL": { "url": "{{zendesk_ticket_url}}" },
          "Status": { "select": "New" },
          "Created": { "date": "{{current_timestamp}}" }
        }
      }
    }
  ],
  "error_handling": {
    "on_chatgpt_error": {
      "action": "Update Ticket",
      "tags_add": ["ai-classification-failed"],
      "internal_note": "AI classification failed - manual review needed"
    }
  }
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent system that continuously analyzes support tickets, identifies emerging product issues, clusters feedback themes, and auto-generates prioritized product backlog items with supporting evidence.',
        estimatedMonthlyCost: '$600 - $1,200/month',
        architecture:
          'A supervisor agent coordinates four specialist agents: Ticket Classifier, Sentiment Analyzer, Theme Clusterer, and Backlog Generator. State is managed in Redis with vector embeddings stored in Pinecone.',
        agents: [
          {
            name: 'TicketClassifierAgent',
            role: 'Support Ticket Classifier',
            goal: 'Classify incoming tickets by product area, issue type, and urgency using fine-tuned models',
            tools: ['Hugging Face Transformers', 'Zero-Shot Classifier', 'PostgreSQL'],
          },
          {
            name: 'SentimentAnalyzerAgent',
            role: 'Customer Sentiment Analyst',
            goal: 'Analyze customer sentiment, detect frustration patterns, and identify at-risk customers',
            tools: ['VADER', 'RoBERTa Sentiment', 'Customer 360 API'],
          },
          {
            name: 'ThemeClustererAgent',
            role: 'Feedback Theme Identifier',
            goal: 'Cluster similar feedback into themes, detect emerging issues, and track theme velocity',
            tools: ['Sentence Transformers', 'HDBSCAN', 'Pinecone'],
          },
          {
            name: 'BacklogGeneratorAgent',
            role: 'Product Backlog Curator',
            goal: 'Generate prioritized backlog items from feedback clusters with supporting evidence and impact estimates',
            tools: ['GPT-4', 'Jira API', 'Linear API'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with vector embeddings in Pinecone for semantic clustering',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system with CrewAI, establishing specialized roles for ticket classification, sentiment analysis, theme clustering, and backlog generation.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Feedback Loop Agent Definitions',
                description: 'CrewAI agent definitions for the feedback-to-product multi-agent system.',
                code: `# agents/feedback_loop_agents.py
"""Multi-agent system for feedback loop automation."""
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FeedbackLoopAgentFactory:
    """Factory for creating feedback loop specialist agents."""

    def __init__(self, llm_model: str = "gpt-4-turbo"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)

    def create_ticket_classifier(self) -> Agent:
        """Agent specialized in multi-label ticket classification."""
        return Agent(
            role="Support Ticket Classifier",
            goal="Accurately classify tickets by product area, issue type, "
                 "and urgency to enable proper routing and analysis",
            backstory="""You are an expert in support ticket triage with
            deep understanding of product taxonomies. You can identify
            the root issue even when customers describe symptoms rather
            than causes. You understand the difference between bugs,
            UX friction, feature requests, and documentation gaps.""",
            llm=self.llm,
            tools=[],  # Tools injected at runtime
            verbose=True,
            allow_delegation=False,
        )

    def create_sentiment_analyzer(self) -> Agent:
        """Agent specialized in customer sentiment analysis."""
        return Agent(
            role="Customer Sentiment Analyst",
            goal="Analyze customer sentiment to identify frustration, "
                 "detect churn risk, and prioritize responses",
            backstory="""You are a customer experience expert who can
            read between the lines. You detect subtle signals of
            frustration, urgency, and satisfaction. You understand
            that a calm message might hide deep frustration and that
            exclamation points do not always mean anger.""",
            llm=self.llm,
            tools=[],
            verbose=True,
            allow_delegation=False,
        )

    def create_theme_clusterer(self) -> Agent:
        """Agent specialized in feedback theme identification."""
        return Agent(
            role="Feedback Theme Identifier",
            goal="Cluster similar feedback into coherent themes and "
                 "detect emerging issues before they become crises",
            backstory="""You are a pattern recognition expert who sees
            connections across thousands of tickets. You identify when
            multiple customers report variants of the same issue and
            can distinguish between isolated incidents and systemic
            problems. You track theme velocity to spot emerging issues.""",
            llm=self.llm,
            tools=[],
            verbose=True,
            allow_delegation=True,
        )

    def create_backlog_generator(self) -> Agent:
        """Agent specialized in generating product backlog items."""
        return Agent(
            role="Product Backlog Curator",
            goal="Transform feedback clusters into well-structured "
                 "backlog items with clear acceptance criteria",
            backstory="""You are a product management expert who writes
            clear, actionable user stories. You understand how to
            translate customer pain into engineering tasks. You
            include relevant evidence from tickets and estimate
            impact based on customer segment and frequency.""",
            llm=self.llm,
            tools=[],
            verbose=True,
            allow_delegation=False,
        )

    def create_crew(self) -> Crew:
        """Assemble the complete feedback loop agent crew."""
        classifier = self.create_ticket_classifier()
        sentiment = self.create_sentiment_analyzer()
        clusterer = self.create_theme_clusterer()
        backlog = self.create_backlog_generator()

        tasks = [
            Task(
                description="Classify the incoming ticket batch",
                agent=classifier,
                expected_output="Classification labels for each ticket",
            ),
            Task(
                description="Analyze sentiment for classified tickets",
                agent=sentiment,
                expected_output="Sentiment scores and risk flags",
            ),
            Task(
                description="Cluster tickets into feedback themes",
                agent=clusterer,
                expected_output="Theme clusters with ticket membership",
            ),
            Task(
                description="Generate backlog items from themes",
                agent=backlog,
                expected_output="Prioritized backlog items with evidence",
            ),
        ]

        return Crew(
            agents=[classifier, sentiment, clusterer, backlog],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the Ticket Classifier agent with tools for zero-shot classification using transformer models, handling multi-label classification across product areas and issue types.',
            toolsUsed: ['CrewAI Tools', 'Hugging Face', 'PostgreSQL'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Ticket Classification Tools',
                description: 'Custom tools for the Ticket Classifier agent to classify support tickets.',
                code: `# tools/ticket_classification_tools.py
"""Ticket classification tools using transformer models."""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from transformers import pipeline
from sqlalchemy import create_engine, text
import os
import logging

logger = logging.getLogger(__name__)


class TicketInput(BaseModel):
    """Input schema for ticket classification."""
    ticket_id: str = Field(..., description="Ticket ID")
    subject: str = Field(..., description="Ticket subject line")
    body: str = Field(..., description="Ticket body content")


class BatchTicketInput(BaseModel):
    """Input schema for batch ticket classification."""
    tickets: List[TicketInput] = Field(..., description="List of tickets to classify")


class ZeroShotClassifierTool(BaseTool):
    """Tool for zero-shot ticket classification."""

    name: str = "classify_ticket"
    description: str = (
        "Classify a support ticket into product area, issue type, and urgency "
        "using zero-shot classification without requiring labeled training data."
    )
    args_schema: type[BaseModel] = TicketInput

    PRODUCT_AREAS = [
        "checkout-flow", "search-results", "payment-processing",
        "account-management", "shipping-delivery", "returns-refunds",
        "mobile-app", "notifications", "pricing-discounts", "performance"
    ]

    ISSUE_TYPES = [
        "bug", "ux-friction", "feature-request",
        "documentation-gap", "performance-issue", "integration-problem"
    ]

    URGENCY_LEVELS = ["critical", "high", "medium", "low"]

    def __init__(self):
        super().__init__()
        self._classifier = None

    @property
    def classifier(self):
        """Lazy-load the zero-shot classifier."""
        if self._classifier is None:
            self._classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # CPU, use 0 for GPU
            )
        return self._classifier

    def _run(
        self,
        ticket_id: str,
        subject: str,
        body: str
    ) -> Dict[str, Any]:
        """Classify a single ticket."""
        text = f"{subject}\\n\\n{body}"[:1024]  # Truncate for model limit

        # Classify product area
        area_result = self.classifier(
            text,
            candidate_labels=self.PRODUCT_AREAS,
            multi_label=False
        )

        # Classify issue type
        type_result = self.classifier(
            text,
            candidate_labels=self.ISSUE_TYPES,
            multi_label=False
        )

        # Classify urgency
        urgency_result = self.classifier(
            text,
            candidate_labels=self.URGENCY_LEVELS,
            multi_label=False
        )

        return {
            "ticket_id": ticket_id,
            "classification": {
                "product_area": area_result["labels"][0],
                "product_area_confidence": round(area_result["scores"][0], 3),
                "issue_type": type_result["labels"][0],
                "issue_type_confidence": round(type_result["scores"][0], 3),
                "urgency": urgency_result["labels"][0],
                "urgency_confidence": round(urgency_result["scores"][0], 3),
            }
        }


class BatchClassifierTool(BaseTool):
    """Tool for batch ticket classification with database persistence."""

    name: str = "classify_ticket_batch"
    description: str = (
        "Classify a batch of tickets and persist results to the database. "
        "More efficient than classifying tickets individually."
    )
    args_schema: type[BaseModel] = BatchTicketInput

    def __init__(self):
        super().__init__()
        self.single_classifier = ZeroShotClassifierTool()
        self.engine = create_engine(os.environ["DATABASE_URL"])

    def _run(self, tickets: List[Dict[str, str]]) -> Dict[str, Any]:
        """Classify batch of tickets and persist results."""
        results = []

        for ticket in tickets:
            try:
                classification = self.single_classifier._run(
                    ticket_id=ticket["ticket_id"],
                    subject=ticket["subject"],
                    body=ticket["body"]
                )
                results.append(classification)
            except Exception as e:
                logger.error(f"Failed to classify ticket {ticket['ticket_id']}: {e}")
                results.append({
                    "ticket_id": ticket["ticket_id"],
                    "error": str(e)
                })

        # Persist to database
        successful = [r for r in results if "classification" in r]
        if successful:
            with self.engine.connect() as conn:
                for result in successful:
                    conn.execute(
                        text("""
                            INSERT INTO analytics.classified_tickets
                            (ticket_id, product_area, product_area_confidence,
                             issue_type, issue_type_confidence, urgency, classified_at)
                            VALUES (:tid, :area, :area_conf, :type, :type_conf, :urg, NOW())
                            ON CONFLICT (ticket_id) DO UPDATE SET
                                product_area = EXCLUDED.product_area,
                                product_area_confidence = EXCLUDED.product_area_confidence,
                                issue_type = EXCLUDED.issue_type,
                                issue_type_confidence = EXCLUDED.issue_type_confidence,
                                urgency = EXCLUDED.urgency,
                                classified_at = NOW()
                        """),
                        {
                            "tid": result["ticket_id"],
                            "area": result["classification"]["product_area"],
                            "area_conf": result["classification"]["product_area_confidence"],
                            "type": result["classification"]["issue_type"],
                            "type_conf": result["classification"]["issue_type_confidence"],
                            "urg": result["classification"]["urgency"],
                        }
                    )
                conn.commit()

        return {
            "total_processed": len(tickets),
            "successful": len(successful),
            "failed": len(tickets) - len(successful),
            "results": results[:10],  # Return sample
        }`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Sentiment Analyzer and Theme Clusterer agents with tools for deep sentiment analysis and semantic clustering using embeddings.',
            toolsUsed: ['CrewAI Tools', 'Sentence Transformers', 'HDBSCAN', 'Pinecone'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Sentiment Analysis and Theme Clustering Tools',
                description: 'Tools for analyzing sentiment and clustering feedback into themes.',
                code: `# tools/feedback_analysis_tools.py
"""Sentiment analysis and theme clustering tools."""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import hdbscan
import numpy as np
from pinecone import Pinecone
from sqlalchemy import create_engine, text
from collections import Counter
import os
import logging

logger = logging.getLogger(__name__)


class TextInput(BaseModel):
    """Input schema for text analysis."""
    ticket_id: str = Field(..., description="Ticket ID")
    text: str = Field(..., description="Text to analyze")


class SentimentAnalysisTool(BaseTool):
    """Tool for deep sentiment analysis with nuance detection."""

    name: str = "analyze_sentiment"
    description: str = (
        "Analyze sentiment of ticket text, detecting frustration levels, "
        "urgency signals, and churn risk indicators."
    )
    args_schema: type[BaseModel] = TextInput

    def __init__(self):
        super().__init__()
        self._sentiment_pipe = None
        self._emotion_pipe = None

    @property
    def sentiment_pipe(self):
        if self._sentiment_pipe is None:
            self._sentiment_pipe = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
        return self._sentiment_pipe

    @property
    def emotion_pipe(self):
        if self._emotion_pipe is None:
            self._emotion_pipe = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=3,
                device=-1
            )
        return self._emotion_pipe

    def _detect_frustration_signals(self, text: str) -> Dict[str, Any]:
        """Detect linguistic signals of frustration."""
        text_lower = text.lower()

        frustration_phrases = [
            "still waiting", "again", "multiple times", "no response",
            "very frustrated", "unacceptable", "terrible", "worst",
            "cancel", "refund", "never again", "been days", "been weeks"
        ]

        urgency_phrases = [
            "urgent", "asap", "immediately", "emergency", "critical",
            "deadline", "need this today", "time sensitive"
        ]

        signals = {
            "frustration_count": sum(1 for p in frustration_phrases if p in text_lower),
            "urgency_count": sum(1 for p in urgency_phrases if p in text_lower),
            "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "exclamation_count": text.count("!"),
            "question_count": text.count("?"),
        }

        # Calculate frustration score 0-100
        frustration_score = min(100, (
            signals["frustration_count"] * 15 +
            signals["caps_ratio"] * 50 +
            signals["exclamation_count"] * 5
        ))

        return {
            "signals": signals,
            "frustration_score": round(frustration_score, 1),
            "has_urgency": signals["urgency_count"] > 0,
        }

    def _run(self, ticket_id: str, text: str) -> Dict[str, Any]:
        """Analyze sentiment with multiple dimensions."""
        truncated = text[:512]

        # Base sentiment
        sentiment = self.sentiment_pipe(truncated)[0]

        # Emotion detection
        emotions = self.emotion_pipe(truncated)[0]

        # Frustration signals
        frustration = self._detect_frustration_signals(text)

        # Calculate composite risk score
        sentiment_score = (
            -1.0 if sentiment["label"] == "negative"
            else 1.0 if sentiment["label"] == "positive"
            else 0.0
        ) * sentiment["score"]

        churn_risk = min(100, max(0, (
            50 - (sentiment_score * 30) +
            frustration["frustration_score"] * 0.3
        )))

        return {
            "ticket_id": ticket_id,
            "sentiment": {
                "label": sentiment["label"],
                "score": round(sentiment["score"], 3),
                "normalized_score": round(sentiment_score, 3),
            },
            "emotions": [
                {"emotion": e["label"], "score": round(e["score"], 3)}
                for e in emotions
            ],
            "frustration": frustration,
            "churn_risk_score": round(churn_risk, 1),
            "risk_level": (
                "high" if churn_risk > 70
                else "medium" if churn_risk > 40
                else "low"
            ),
        }


class ThemeClusteringTool(BaseTool):
    """Tool for clustering feedback into semantic themes."""

    name: str = "cluster_feedback_themes"
    description: str = (
        "Cluster feedback tickets into semantic themes using embeddings. "
        "Identifies emerging issues and tracks theme velocity."
    )

    def __init__(self, pinecone_index: str = "feedback-themes"):
        super().__init__()
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.engine = create_engine(os.environ["DATABASE_URL"])

        # Initialize Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = pc.Index(pinecone_index)

    def _run(
        self,
        product_area: Optional[str] = None,
        days_back: int = 7,
        min_cluster_size: int = 5
    ) -> Dict[str, Any]:
        """Cluster recent tickets into themes."""
        # Fetch recent classified tickets
        query = """
            SELECT ct.ticket_id, t.subject, t.body,
                   ct.product_area, ct.issue_type
            FROM analytics.classified_tickets ct
            JOIN support.tickets t ON ct.ticket_id = t.ticket_id
            WHERE ct.classified_at >= CURRENT_DATE - :days
        """
        params = {"days": days_back}

        if product_area:
            query += " AND ct.product_area = :area"
            params["area"] = product_area

        with self.engine.connect() as conn:
            tickets = conn.execute(text(query), params).fetchall()

        if len(tickets) < min_cluster_size:
            return {"error": "Not enough tickets for clustering", "count": len(tickets)}

        # Generate embeddings
        texts = [f"{t.subject} {t.body[:500]}" for t in tickets]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)

        # Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric="euclidean"
        )
        labels = clusterer.fit_predict(embeddings)

        # Analyze clusters
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue  # Skip noise
            if label not in clusters:
                clusters[label] = {
                    "tickets": [],
                    "product_areas": [],
                    "issue_types": [],
                }
            clusters[label]["tickets"].append(tickets[i].ticket_id)
            clusters[label]["product_areas"].append(tickets[i].product_area)
            clusters[label]["issue_types"].append(tickets[i].issue_type)

        # Generate cluster summaries
        themes = []
        for label, data in clusters.items():
            # Get representative ticket (closest to centroid)
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            centroid = np.mean([embeddings[i] for i in cluster_indices], axis=0)
            distances = [np.linalg.norm(embeddings[i] - centroid) for i in cluster_indices]
            representative_idx = cluster_indices[np.argmin(distances)]

            themes.append({
                "theme_id": f"theme_{label}",
                "ticket_count": len(data["tickets"]),
                "primary_product_area": Counter(data["product_areas"]).most_common(1)[0][0],
                "primary_issue_type": Counter(data["issue_types"]).most_common(1)[0][0],
                "representative_subject": tickets[representative_idx].subject,
                "ticket_ids": data["tickets"][:10],
            })

        # Sort by ticket count
        themes.sort(key=lambda x: x["ticket_count"], reverse=True)

        # Store in Pinecone for tracking
        for theme in themes[:10]:
            idx = [i for i, t in enumerate(tickets) if t.ticket_id == theme["ticket_ids"][0]][0]
            self.index.upsert(vectors=[{
                "id": theme["theme_id"],
                "values": embeddings[idx].tolist(),
                "metadata": {
                    "ticket_count": theme["ticket_count"],
                    "product_area": theme["primary_product_area"],
                    "issue_type": theme["primary_issue_type"],
                }
            }])

        return {
            "total_tickets": len(tickets),
            "clustered_tickets": sum(1 for l in labels if l != -1),
            "noise_tickets": sum(1 for l in labels if l == -1),
            "theme_count": len(themes),
            "themes": themes[:10],
        }`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement LangGraph-based orchestration that processes ticket batches through classification, sentiment analysis, clustering, and backlog generation in a coordinated pipeline.',
            toolsUsed: ['LangGraph', 'Redis', 'Pinecone'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Orchestration for Feedback Loop',
                description: 'State machine orchestration for the feedback-to-product multi-agent pipeline.',
                code: `# orchestration/feedback_loop_graph.py
"""LangGraph orchestration for Feedback Loop multi-agent system."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict, Any, Optional
import redis
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FeedbackLoopState(TypedDict):
    """State schema for feedback loop pipeline."""
    # Input
    product_area: Optional[str]
    days_back: int
    min_cluster_size: int

    # Classification results
    classified_tickets: List[Dict[str, Any]]
    classification_stats: Dict[str, int]

    # Sentiment results
    sentiment_results: List[Dict[str, Any]]
    high_risk_tickets: List[str]

    # Clustering results
    themes: List[Dict[str, Any]]
    emerging_themes: List[Dict[str, Any]]

    # Backlog items
    backlog_items: List[Dict[str, Any]]

    # Pipeline metadata
    processing_status: str
    started_at: str
    completed_at: Optional[str]
    errors: List[str]


class FeedbackLoopOrchestrator:
    """LangGraph-based orchestrator for feedback loop pipeline."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the feedback loop state graph."""
        graph = StateGraph(FeedbackLoopState)

        # Add nodes
        graph.add_node("fetch_and_classify", self._fetch_and_classify)
        graph.add_node("analyze_sentiment", self._analyze_sentiment)
        graph.add_node("cluster_themes", self._cluster_themes)
        graph.add_node("detect_emerging", self._detect_emerging)
        graph.add_node("generate_backlog", self._generate_backlog)
        graph.add_node("persist_results", self._persist_results)

        # Add edges
        graph.add_edge("fetch_and_classify", "analyze_sentiment")
        graph.add_edge("analyze_sentiment", "cluster_themes")
        graph.add_edge("cluster_themes", "detect_emerging")
        graph.add_edge("detect_emerging", "generate_backlog")
        graph.add_edge("generate_backlog", "persist_results")
        graph.add_edge("persist_results", END)

        graph.set_entry_point("fetch_and_classify")

        return graph.compile(checkpointer=self.checkpointer)

    def _fetch_and_classify(self, state: FeedbackLoopState) -> FeedbackLoopState:
        """Node: Fetch unclassified tickets and classify them."""
        from tools.ticket_classification_tools import BatchClassifierTool
        from sqlalchemy import create_engine, text
        import os

        engine = create_engine(os.environ["DATABASE_URL"])

        # Fetch unclassified tickets
        query = """
            SELECT t.ticket_id, t.subject, t.body
            FROM support.tickets t
            LEFT JOIN analytics.classified_tickets ct ON t.ticket_id = ct.ticket_id
            WHERE ct.ticket_id IS NULL
              AND t.created_at >= CURRENT_DATE - :days
        """
        params = {"days": state["days_back"]}

        if state.get("product_area"):
            # If filtering by area, we need already-classified tickets
            query = """
                SELECT t.ticket_id, t.subject, t.body
                FROM support.tickets t
                JOIN analytics.classified_tickets ct ON t.ticket_id = ct.ticket_id
                WHERE ct.product_area = :area
                  AND t.created_at >= CURRENT_DATE - :days
            """
            params["area"] = state["product_area"]

        with engine.connect() as conn:
            tickets = conn.execute(text(query), params).fetchall()

        if not tickets:
            return {
                **state,
                "classified_tickets": [],
                "classification_stats": {"total": 0},
                "processing_status": "no_tickets",
            }

        # Classify tickets
        classifier = BatchClassifierTool()
        ticket_list = [
            {"ticket_id": t.ticket_id, "subject": t.subject, "body": t.body}
            for t in tickets
        ]
        result = classifier._run(ticket_list)

        return {
            **state,
            "classified_tickets": result.get("results", []),
            "classification_stats": {
                "total": result["total_processed"],
                "successful": result["successful"],
                "failed": result["failed"],
            },
            "processing_status": "classified",
        }

    def _analyze_sentiment(self, state: FeedbackLoopState) -> FeedbackLoopState:
        """Node: Analyze sentiment for classified tickets."""
        from tools.feedback_analysis_tools import SentimentAnalysisTool
        from sqlalchemy import create_engine, text
        import os

        if not state.get("classified_tickets"):
            return state

        engine = create_engine(os.environ["DATABASE_URL"])
        sentiment_tool = SentimentAnalysisTool()

        # Get ticket texts
        ticket_ids = [t["ticket_id"] for t in state["classified_tickets"] if "classification" in t]

        with engine.connect() as conn:
            tickets = conn.execute(
                text("SELECT ticket_id, subject, body FROM support.tickets WHERE ticket_id = ANY(:ids)"),
                {"ids": ticket_ids}
            ).fetchall()

        results = []
        high_risk = []

        for ticket in tickets:
            sentiment = sentiment_tool._run(
                ticket_id=ticket.ticket_id,
                text=f"{ticket.subject} {ticket.body}"
            )
            results.append(sentiment)

            if sentiment.get("risk_level") == "high":
                high_risk.append(ticket.ticket_id)

        return {
            **state,
            "sentiment_results": results,
            "high_risk_tickets": high_risk,
            "processing_status": "sentiment_analyzed",
        }

    def _cluster_themes(self, state: FeedbackLoopState) -> FeedbackLoopState:
        """Node: Cluster tickets into feedback themes."""
        from tools.feedback_analysis_tools import ThemeClusteringTool

        clusterer = ThemeClusteringTool()
        result = clusterer._run(
            product_area=state.get("product_area"),
            days_back=state["days_back"],
            min_cluster_size=state["min_cluster_size"]
        )

        if "error" in result:
            return {
                **state,
                "themes": [],
                "errors": state.get("errors", []) + [result["error"]],
            }

        return {
            **state,
            "themes": result.get("themes", []),
            "processing_status": "themes_clustered",
        }

    def _detect_emerging(self, state: FeedbackLoopState) -> FeedbackLoopState:
        """Node: Detect emerging themes by comparing to historical baselines."""
        themes = state.get("themes", [])

        # Compare current theme sizes to historical averages
        emerging = []
        for theme in themes:
            # In production, compare to stored historical baseline
            # For now, flag themes with > 10 tickets as potentially emerging
            if theme["ticket_count"] > 10:
                emerging.append({
                    **theme,
                    "emergence_signal": "high_volume",
                    "velocity": "accelerating",
                })

        return {
            **state,
            "emerging_themes": emerging,
            "processing_status": "emerging_detected",
        }

    def _generate_backlog(self, state: FeedbackLoopState) -> FeedbackLoopState:
        """Node: Generate product backlog items from themes."""
        from langchain_openai import ChatOpenAI

        themes = state.get("themes", [])
        if not themes:
            return {**state, "backlog_items": []}

        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)
        backlog_items = []

        for theme in themes[:5]:  # Top 5 themes
            prompt = f"""
            Generate a product backlog item for this feedback theme:

            Theme: {theme['representative_subject']}
            Product Area: {theme['primary_product_area']}
            Issue Type: {theme['primary_issue_type']}
            Affected Tickets: {theme['ticket_count']}

            Return a JSON object with:
            - title: Clear, actionable title
            - description: User story format
            - acceptance_criteria: List of testable criteria
            - priority: P0/P1/P2/P3 based on volume
            - estimated_impact: Number of affected customers
            """

            response = llm.invoke(prompt)
            try:
                item = json.loads(response.content)
                item["theme_id"] = theme["theme_id"]
                item["ticket_count"] = theme["ticket_count"]
                backlog_items.append(item)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse backlog item for {theme['theme_id']}")

        return {
            **state,
            "backlog_items": backlog_items,
            "processing_status": "backlog_generated",
        }

    def _persist_results(self, state: FeedbackLoopState) -> FeedbackLoopState:
        """Node: Persist pipeline results to database and cache."""
        run_id = f"feedback_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Cache results in Redis
        self.redis.setex(
            f"feedback_pipeline:{run_id}",
            86400,  # 24 hour TTL
            json.dumps({
                "themes": state.get("themes", []),
                "backlog_items": state.get("backlog_items", []),
                "high_risk_tickets": state.get("high_risk_tickets", []),
                "stats": state.get("classification_stats", {}),
            })
        )

        return {
            **state,
            "completed_at": datetime.utcnow().isoformat(),
            "processing_status": "completed",
        }

    async def run_pipeline(
        self,
        product_area: Optional[str] = None,
        days_back: int = 7,
        min_cluster_size: int = 5
    ) -> Dict[str, Any]:
        """Execute the feedback loop pipeline."""
        initial_state: FeedbackLoopState = {
            "product_area": product_area,
            "days_back": days_back,
            "min_cluster_size": min_cluster_size,
            "classified_tickets": [],
            "classification_stats": {},
            "sentiment_results": [],
            "high_risk_tickets": [],
            "themes": [],
            "emerging_themes": [],
            "backlog_items": [],
            "processing_status": "started",
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "errors": [],
        }

        config = {"configurable": {"thread_id": f"feedback_{datetime.utcnow().timestamp()}"}}
        result = await self.graph.ainvoke(initial_state, config)

        return {
            "status": result["processing_status"],
            "themes_found": len(result.get("themes", [])),
            "backlog_items_generated": len(result.get("backlog_items", [])),
            "high_risk_tickets": len(result.get("high_risk_tickets", [])),
            "themes": result.get("themes", []),
            "backlog_items": result.get("backlog_items", []),
        }`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the multi-agent feedback pipeline with Docker, implement scheduled execution, and set up monitoring dashboards for theme velocity and backlog generation.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'FastAPI'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Feedback Loop API with Scheduling',
                description: 'FastAPI service exposing the feedback loop pipeline with scheduled execution.',
                code: `# api/feedback_loop_service.py
"""Feedback Loop API service with scheduling and observability."""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from langsmith import traceable
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import structlog
import time
import os
from typing import Optional, List

from orchestration.feedback_loop_graph import FeedbackLoopOrchestrator

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Prometheus metrics
PIPELINE_RUNS = Counter(
    "feedback_pipeline_runs_total",
    "Total pipeline runs",
    ["status"]
)
THEMES_DETECTED = Gauge(
    "feedback_themes_detected",
    "Number of themes detected in last run"
)
BACKLOG_ITEMS_GENERATED = Gauge(
    "feedback_backlog_items_generated",
    "Backlog items generated in last run"
)
HIGH_RISK_TICKETS = Gauge(
    "feedback_high_risk_tickets",
    "High risk tickets detected"
)
PIPELINE_DURATION = Histogram(
    "feedback_pipeline_duration_seconds",
    "Pipeline execution duration",
    buckets=[30, 60, 120, 300, 600, 1200]
)

app = FastAPI(
    title="Feedback Loop Multi-Agent API",
    description="Automated feedback classification and backlog generation",
    version="2.0.0"
)

orchestrator = FeedbackLoopOrchestrator(
    redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379")
)

scheduler = AsyncIOScheduler()


class PipelineRequest(BaseModel):
    """Request schema for pipeline execution."""
    product_area: Optional[str] = Field(None, description="Filter by product area")
    days_back: int = Field(default=7, ge=1, le=90, description="Days to analyze")
    min_cluster_size: int = Field(default=5, ge=2, le=50, description="Minimum cluster size")


class PipelineResponse(BaseModel):
    """Response schema for pipeline results."""
    status: str
    themes_found: int
    backlog_items_generated: int
    high_risk_tickets: int
    themes: List[dict]
    backlog_items: List[dict]


@app.on_event("startup")
async def startup():
    """Start the scheduler for automated runs."""
    # Run pipeline daily at 6 AM UTC
    scheduler.add_job(
        run_scheduled_pipeline,
        "cron",
        hour=6,
        minute=0,
        id="daily_feedback_pipeline"
    )
    scheduler.start()
    logger.info("scheduler_started", job="daily_feedback_pipeline")


@app.on_event("shutdown")
async def shutdown():
    """Shutdown the scheduler."""
    scheduler.shutdown()


async def run_scheduled_pipeline():
    """Scheduled pipeline execution."""
    logger.info("scheduled_pipeline_started")
    start_time = time.time()

    try:
        result = await orchestrator.run_pipeline(
            product_area=None,
            days_back=7,
            min_cluster_size=5
        )

        duration = time.time() - start_time
        PIPELINE_DURATION.observe(duration)
        PIPELINE_RUNS.labels(status="success").inc()
        THEMES_DETECTED.set(result["themes_found"])
        BACKLOG_ITEMS_GENERATED.set(result["backlog_items_generated"])
        HIGH_RISK_TICKETS.set(result["high_risk_tickets"])

        logger.info(
            "scheduled_pipeline_completed",
            duration_seconds=round(duration, 2),
            themes=result["themes_found"],
            backlog_items=result["backlog_items_generated"]
        )

    except Exception as e:
        PIPELINE_RUNS.labels(status="error").inc()
        logger.error("scheduled_pipeline_failed", error=str(e))


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "feedback-loop"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@app.post("/api/v2/pipeline/run", response_model=PipelineResponse)
@traceable(name="run_feedback_pipeline")
async def run_pipeline(request: PipelineRequest):
    """
    Execute the feedback loop pipeline on-demand.

    The pipeline will:
    1. Classify unprocessed tickets
    2. Analyze sentiment and detect high-risk customers
    3. Cluster feedback into themes
    4. Generate prioritized backlog items
    """
    start_time = time.time()

    logger.info(
        "pipeline_request_received",
        product_area=request.product_area,
        days_back=request.days_back
    )

    try:
        result = await orchestrator.run_pipeline(
            product_area=request.product_area,
            days_back=request.days_back,
            min_cluster_size=request.min_cluster_size
        )

        duration = time.time() - start_time
        PIPELINE_DURATION.observe(duration)
        PIPELINE_RUNS.labels(status="success").inc()

        logger.info(
            "pipeline_request_completed",
            duration_seconds=round(duration, 2),
            themes=result["themes_found"]
        )

        return PipelineResponse(**result)

    except Exception as e:
        PIPELINE_RUNS.labels(status="error").inc()
        logger.error("pipeline_request_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Pipeline execution failed")


@app.get("/api/v2/themes")
async def get_recent_themes(days: int = 7, limit: int = 20):
    """Get recently detected feedback themes."""
    import json

    # Fetch from Redis cache
    keys = orchestrator.redis.keys("feedback_pipeline:*")
    all_themes = []

    for key in sorted(keys, reverse=True)[:5]:
        data = orchestrator.redis.get(key)
        if data:
            parsed = json.loads(data)
            all_themes.extend(parsed.get("themes", []))

    # Dedupe and sort by ticket count
    seen = set()
    unique_themes = []
    for theme in all_themes:
        if theme["theme_id"] not in seen:
            seen.add(theme["theme_id"])
            unique_themes.append(theme)

    unique_themes.sort(key=lambda x: x["ticket_count"], reverse=True)

    return {"themes": unique_themes[:limit]}


@app.get("/api/v2/backlog")
async def get_generated_backlog(limit: int = 20):
    """Get recently generated backlog items."""
    import json

    keys = orchestrator.redis.keys("feedback_pipeline:*")
    all_items = []

    for key in sorted(keys, reverse=True)[:5]:
        data = orchestrator.redis.get(key)
        if data:
            parsed = json.loads(data)
            all_items.extend(parsed.get("backlog_items", []))

    return {"backlog_items": all_items[:limit]}`,
              },
              {
                language: 'yaml',
                title: 'Docker Compose for Feedback Loop Deployment',
                description: 'Docker Compose configuration for deploying the feedback loop multi-agent system.',
                code: `# docker-compose.feedback-loop.yml
version: '3.8'

services:
  feedback-api:
    build:
      context: .
      dockerfile: Dockerfile.feedback
    ports:
      - "8010:8000"
    environment:
      - DATABASE_URL=\${DATABASE_URL}
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - PINECONE_API_KEY=\${PINECONE_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=feedback-loop-agents
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  feedback-worker:
    build:
      context: .
      dockerfile: Dockerfile.feedback
    command: python -m celery -A workers.feedback_worker worker -l info
    environment:
      - DATABASE_URL=\${DATABASE_URL}
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - PINECONE_API_KEY=\${PINECONE_API_KEY}
    depends_on:
      - redis
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G

  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    volumes:
      - feedback-redis-data:/data
    command: redis-server --appendonly yes

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus-feedback.yml:/etc/prometheus/prometheus.yml

volumes:
  feedback-redis-data:`,
              },
            ],
          },
        ],
      },
    },

    /* ──────────────────────────────────────────────
       Pain Point 3 — The WISMO Crisis
       ────────────────────────────────────────────── */
    {
      id: 'wismo-crisis',
      number: 3,
      title: 'The WISMO Crisis',
      subtitle: 'Where Is My Order — 40% of All Support Tickets',
      summary:
        'WISMO tickets consume 40% of your support bandwidth because order tracking data is delayed, incomplete, or inaccessible to agents.',
      tags: ['wismo', 'order-tracking', 'logistics'],
      metrics: {
        annualCostRange: '$600K - $2.5M',
        roi: '9x',
        paybackPeriod: '2-3 months',
        investmentRange: '$70K - $140K',
      },
      price: {
        present: {
          title: 'Present — The Symptom',
          description:
            'Four out of every ten support tickets are customers asking "Where is my order?" because self-service tracking is broken or absent.',
          bullets: [
            'WISMO tickets represent 40% of total support volume across phone, chat, and email',
            'Agents spend an average of 6 minutes per WISMO ticket manually looking up carrier data',
            'Order status pages show stale data — updates lag by 12-48 hours behind carrier scans',
            'Customers escalate to social media when tracking pages show no movement for 3+ days',
          ],
          severity: 'critical',
        },
        root: {
          title: 'Root Cause — Why It Persists',
          description:
            'Order, fulfillment, and carrier tracking data live in separate systems with no real-time synchronization layer.',
          bullets: [
            'Carrier tracking APIs are polled in batch every 6-24 hours instead of via webhook',
            'OMS, WMS, and carrier platforms use different order/shipment ID formats with no mapping table',
            'The customer-facing tracking page queries a stale replica that refreshes overnight',
            'No proactive notification system exists to alert customers before they ask',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Impact — Business Damage',
          description:
            'WISMO volume drives up support costs, crushes CSAT, and diverts agents from revenue-generating interactions.',
          bullets: [
            'Direct support cost of WISMO tickets is $600K-$2.5M annually at $8-$12 per contact',
            'CSAT for WISMO interactions averages 2.1/5 — the lowest of any ticket category',
            'Agent time spent on WISMO is time not spent on retention, upsell, or complex issue resolution',
            'Social media complaints about shipping visibility damage brand perception at scale',
          ],
          severity: 'critical',
        },
        cost: {
          title: 'Cost of Inaction — 12-Month Horizon',
          description:
            'WISMO volume scales linearly with order volume. As sales grow, the support team drowns without a self-service solution.',
          bullets: [
            'A 20% YoY order growth means 20% more WISMO tickets without any product change',
            'Hiring to cover WISMO volume requires 8-12 additional agents at $50K each annually',
            'Stale tracking pages generate repeat contacts — the same customer calls 2-3 times per order',
            'Competitors with proactive tracking notifications report 60% fewer inbound WISMO contacts',
          ],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Expected Return — Post-Implementation',
          description:
            'Real-time tracking views and proactive notifications can eliminate 70-80% of WISMO tickets within 90 days.',
          bullets: [
            'WISMO ticket volume drops by 75%, freeing agent capacity for high-value work',
            'Customer-facing tracking page shows real-time status with sub-hour latency',
            'Proactive SMS/email notifications reduce repeat contacts by 65%',
            'CSAT for order-tracking interactions rises from 2.1 to 4.3 out of 5',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Build a real-time order tracking view that unifies OMS, WMS, and carrier data, then deploy a proactive notification system that messages customers before they need to ask.',
        prerequisites: [
          'Access to OMS, WMS, and carrier tracking APIs (FedEx, UPS, USPS, etc.)',
          'A data warehouse or operational database for the unified tracking view',
          'Python 3.10+ with an async HTTP client for carrier webhook ingestion',
          'SMS/email provider (Twilio, SendGrid, or equivalent)',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        toolsUsed: ['SQL', 'Python', 'FastAPI', 'Twilio', 'pytest', 'Docker', 'GitHub Actions', 'cron / Airflow', 'Slack API'],
        steps: [
          {
            stepNumber: 1,
            title: 'Real-Time Order Tracking View',
            description:
              'Create a unified view that joins OMS order data, WMS fulfillment events, and carrier tracking milestones into a single timeline per order — refreshed in near-real-time.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Unified Order Tracking View',
                description:
                  'Joins order, fulfillment, and carrier milestone data into a single real-time view with computed delivery estimates.',
                code: `-- real_time_order_tracking.sql
-- Unified order tracking view across OMS, WMS, and carrier systems
CREATE OR REPLACE VIEW logistics.order_tracking_live AS
WITH carrier_latest AS (
  SELECT
    shipment_id,
    carrier_code,
    tracking_number,
    status                          AS carrier_status,
    location                        AS last_known_location,
    scanned_at                      AS last_scan_at,
    estimated_delivery_date,
    ROW_NUMBER() OVER (
      PARTITION BY shipment_id ORDER BY scanned_at DESC
    ) AS rn
  FROM logistics.carrier_events
),
fulfillment AS (
  SELECT
    order_id,
    shipment_id,
    warehouse_code,
    picked_at,
    packed_at,
    shipped_at,
    CASE
      WHEN shipped_at IS NOT NULL THEN 'shipped'
      WHEN packed_at  IS NOT NULL THEN 'packed'
      WHEN picked_at  IS NOT NULL THEN 'picking'
      ELSE 'processing'
    END AS fulfillment_stage
  FROM wms.fulfillment_events
)
SELECT
  o.order_id,
  o.customer_id,
  o.placed_at,
  f.fulfillment_stage,
  f.shipped_at,
  cl.carrier_code,
  cl.tracking_number,
  cl.carrier_status,
  cl.last_known_location,
  cl.last_scan_at,
  cl.estimated_delivery_date,
  DATEDIFF('hour', cl.last_scan_at, CURRENT_TIMESTAMP) AS hours_since_last_scan
FROM orders.orders o
LEFT JOIN fulfillment f       ON o.order_id = f.order_id
LEFT JOIN carrier_latest cl   ON f.shipment_id = cl.shipment_id AND cl.rn = 1
WHERE o.status NOT IN ('cancelled', 'returned');`,
              },
              {
                language: 'sql',
                title: 'Stale Shipment Detection Query',
                description:
                  'Identifies shipments with no carrier scan in 48+ hours that are likely to trigger WISMO tickets, enabling proactive outreach.',
                code: `-- stale_shipment_detection.sql
-- Flag shipments at risk of generating WISMO tickets
CREATE OR REPLACE VIEW logistics.stale_shipments AS
SELECT
  ot.order_id,
  ot.customer_id,
  ot.tracking_number,
  ot.carrier_code,
  ot.carrier_status,
  ot.last_known_location,
  ot.last_scan_at,
  ot.hours_since_last_scan,
  c360.email,
  c360.phone,
  CASE
    WHEN ot.hours_since_last_scan > 96 THEN 'critical'
    WHEN ot.hours_since_last_scan > 72 THEN 'high'
    WHEN ot.hours_since_last_scan > 48 THEN 'medium'
    ELSE 'normal'
  END AS staleness_tier,
  ot.estimated_delivery_date,
  CASE
    WHEN ot.estimated_delivery_date < CURRENT_DATE THEN TRUE
    ELSE FALSE
  END AS is_past_due
FROM logistics.order_tracking_live ot
JOIN analytics.customer_360 c360 ON ot.customer_id = c360.customer_id
WHERE ot.carrier_status NOT IN ('delivered', 'returned_to_sender')
  AND ot.hours_since_last_scan > 48
ORDER BY ot.hours_since_last_scan DESC;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Carrier Webhook Ingestion Service',
            description:
              'Replace batch polling with a webhook receiver that captures carrier scan events in real time, keeping the tracking view fresh to within minutes.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Carrier Webhook Receiver',
                description:
                  'A FastAPI service that receives carrier tracking webhooks, normalizes events across carriers, and writes them to the warehouse.',
                code: `# carrier_webhook_receiver.py
from fastapi import FastAPI, Request, HTTPException
from datetime import datetime
import asyncpg, os, hmac, hashlib

app = FastAPI(title="Carrier Webhook Receiver")

CARRIER_SECRETS = {
    "fedex": os.environ["FEDEX_WEBHOOK_SECRET"],
    "ups": os.environ["UPS_WEBHOOK_SECRET"],
    "usps": os.environ["USPS_WEBHOOK_SECRET"],
}

STATUS_MAP = {
    "fedex": {"PU": "picked_up", "IT": "in_transit", "DL": "delivered", "DE": "exception"},
    "ups":   {"M": "picked_up", "I": "in_transit", "D": "delivered", "X": "exception"},
    "usps":  {"AC": "picked_up", "OF": "in_transit", "01": "delivered", "09": "exception"},
}

def verify_signature(carrier: str, payload: bytes, signature: str) -> bool:
    secret = CARRIER_SECRETS.get(carrier, "")
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)

@app.post("/webhooks/carrier/{carrier}")
async def receive_carrier_event(carrier: str, request: Request):
    body = await request.body()
    sig = request.headers.get("x-webhook-signature", "")
    if not verify_signature(carrier, body, sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    data = await request.json()
    normalized_status = STATUS_MAP.get(carrier, {}).get(data.get("status_code"), "unknown")

    pool = app.state.pool
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO logistics.carrier_events
              (shipment_id, carrier_code, tracking_number, status, location, scanned_at)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, data["shipment_id"], carrier, data["tracking_number"],
           normalized_status, data.get("location", ""), datetime.fromisoformat(data["timestamp"]))

    return {"status": "accepted", "carrier": carrier, "normalized_status": normalized_status}`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate the order-tracking pipeline with SQL assertions on carrier data integrity and a pytest suite that checks shipment freshness, status mapping correctness, and notification delivery accuracy.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Order Tracking Data Quality Assertions',
                description:
                  'SQL assertion queries that verify carrier events contain valid statuses, no orphaned shipment references, and that the tracking view freshness meets SLA.',
                code: `-- test_order_tracking_quality.sql
-- Assertion 1: No carrier events with unknown status
SELECT 'unknown_carrier_status' AS assertion,
       COUNT(*) AS violations
FROM logistics.carrier_events
WHERE status NOT IN (
  'picked_up', 'in_transit', 'delivered',
  'exception', 'returned_to_sender', 'unknown'
)
HAVING COUNT(*) > 0;

-- Assertion 2: Every carrier event maps to a valid shipment
SELECT 'orphaned_carrier_events' AS assertion,
       COUNT(*) AS violations
FROM logistics.carrier_events ce
LEFT JOIN wms.fulfillment_events fe ON ce.shipment_id = fe.shipment_id
WHERE fe.shipment_id IS NULL
HAVING COUNT(*) > 0;

-- Assertion 3: No shipments stuck in processing for > 48 hours
SELECT 'stuck_shipments' AS assertion,
       COUNT(*) AS violations
FROM logistics.order_tracking_live
WHERE fulfillment_stage = 'processing'
  AND placed_at < CURRENT_TIMESTAMP - INTERVAL '48 hours'
HAVING COUNT(*) > 0;

-- Assertion 4: Carrier event freshness — newest event within last hour
SELECT 'stale_carrier_feed' AS assertion,
       MAX(scanned_at)::text AS most_recent
FROM logistics.carrier_events
HAVING MAX(scanned_at) < CURRENT_TIMESTAMP - INTERVAL '1 hour';`,
              },
              {
                language: 'python',
                title: 'Pytest Order Tracking Validation Suite',
                description:
                  'A pytest-based validation suite with typed test classes that verify carrier-status mapping, shipment-to-order join integrity, and stale-shipment detection accuracy.',
                code: `# tests/test_order_tracking_pipeline.py
"""Validation suite for the order-tracking and carrier data pipeline."""
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


@dataclass
class TrackingThresholds:
    """Typed thresholds for order-tracking quality checks."""
    max_orphaned_events: int = 0
    max_stuck_shipments: int = 5
    max_carrier_feed_staleness_hours: float = 1.0
    min_delivery_rate: float = 0.85


class TestOrderTrackingQuality:
    """Data-quality tests for logistics.order_tracking_live."""

    engine = create_engine("postgresql://localhost/warehouse")
    thresholds = TrackingThresholds()

    def _query_scalar(self, sql: str) -> Any:
        with self.engine.connect() as conn:
            return conn.execute(text(sql)).scalar()

    def test_no_orphaned_carrier_events(self) -> None:
        orphans: int = self._query_scalar(
            "SELECT COUNT(*) FROM logistics.carrier_events ce"
            " LEFT JOIN wms.fulfillment_events fe"
            "   ON ce.shipment_id = fe.shipment_id"
            " WHERE fe.shipment_id IS NULL"
        )
        logger.info("Orphaned carrier events: %d", orphans)
        assert orphans <= self.thresholds.max_orphaned_events

    def test_no_unknown_statuses(self) -> None:
        unknowns: int = self._query_scalar(
            "SELECT COUNT(*) FROM logistics.carrier_events"
            " WHERE status NOT IN ("
            "   'picked_up','in_transit','delivered',"
            "   'exception','returned_to_sender','unknown')"
        )
        logger.info("Unknown carrier statuses: %d", unknowns)
        assert unknowns == 0

    def test_stuck_shipments_within_limit(self) -> None:
        stuck: int = self._query_scalar(
            "SELECT COUNT(*) FROM logistics.order_tracking_live"
            " WHERE fulfillment_stage = 'processing'"
            "   AND placed_at < CURRENT_TIMESTAMP - INTERVAL '48 hours'"
        )
        logger.info("Stuck shipments: %d", stuck)
        assert stuck <= self.thresholds.max_stuck_shipments

    def test_carrier_feed_freshness(self) -> None:
        hours: float = self._query_scalar(
            "SELECT EXTRACT(EPOCH FROM"
            "  (CURRENT_TIMESTAMP - MAX(scanned_at)))"
            " / 3600.0 FROM logistics.carrier_events"
        )
        logger.info("Carrier feed staleness: %.1f hours", hours)
        assert hours <= self.thresholds.max_carrier_feed_staleness_hours`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Containerize the carrier webhook receiver and notification dispatcher with Docker, deploy via a bash script that validates carrier API connectivity and starts all services, and manage configuration with typed dataclasses.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'WISMO Stack Deployment Script',
                description:
                  'End-to-end deployment script that builds the WISMO Docker images, verifies carrier API credentials, runs the pytest suite, and starts the webhook receiver and notification services.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────
APP_NAME="wismo-stack"
IMAGE_TAG="\${APP_NAME}:\$(git rev-parse --short HEAD)"
COMPOSE_FILE="docker-compose.wismo.yml"
WEBHOOK_HEALTH="http://localhost:8001/healthz"
NOTIFIER_HEALTH="http://localhost:8002/healthz"
MAX_WAIT=60

echo "==> Building Docker image \${IMAGE_TAG}"
docker build -t "\${IMAGE_TAG}" -f Dockerfile.wismo .

echo "==> Verifying carrier API credentials"
for CARRIER in fedex ups usps; do
  VAR_NAME="\$(echo "\${CARRIER}" | tr '[:lower:]' '[:upper:]')_WEBHOOK_SECRET"
  if [ -z "\${!VAR_NAME:-}" ]; then
    echo "ERROR: \${VAR_NAME} is not set" >&2
    exit 1
  fi
done
echo "    All carrier secrets present"

echo "==> Running pytest validation suite"
docker run --rm --env-file .env "\${IMAGE_TAG}" \
  python -m pytest tests/test_order_tracking_pipeline.py -v --tb=short

echo "==> Starting services via docker-compose"
docker-compose -f "\${COMPOSE_FILE}" up -d

echo "==> Waiting for webhook receiver at \${WEBHOOK_HEALTH}"
elapsed=0
until curl -sf "\${WEBHOOK_HEALTH}" > /dev/null 2>&1; do
  sleep 2
  elapsed=\$((elapsed + 2))
  if [ "\${elapsed}" -ge "\${MAX_WAIT}" ]; then
    echo "ERROR: Webhook health-check failed after \${MAX_WAIT}s" >&2
    docker-compose -f "\${COMPOSE_FILE}" logs --tail=50 webhook-receiver
    exit 1
  fi
done

echo "==> Waiting for notification service at \${NOTIFIER_HEALTH}"
elapsed=0
until curl -sf "\${NOTIFIER_HEALTH}" > /dev/null 2>&1; do
  sleep 2
  elapsed=\$((elapsed + 2))
  if [ "\${elapsed}" -ge "\${MAX_WAIT}" ]; then
    echo "ERROR: Notifier health-check failed after \${MAX_WAIT}s" >&2
    docker-compose -f "\${COMPOSE_FILE}" logs --tail=50 notifier
    exit 1
  fi
done

echo "==> Deployment complete — \${APP_NAME} is healthy"`,
              },
              {
                language: 'python',
                title: 'Typed Configuration Loader',
                description:
                  'A dataclass-based configuration loader that reads environment variables with validation and defaults for the WISMO tracking and notification stack.',
                code: `# config/wismo_config.py
"""Typed configuration loader for the WISMO tracking stack."""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabaseConfig:
    """Connection settings for the logistics database."""
    url: str = ""
    pool_min: int = 5
    pool_max: int = 20


@dataclass(frozen=True)
class CarrierConfig:
    """Carrier webhook and API settings."""
    fedex_secret: str = ""
    ups_secret: str = ""
    usps_secret: str = ""
    webhook_port: int = 8001


@dataclass(frozen=True)
class NotificationConfig:
    """Twilio and SendGrid settings for proactive notifications."""
    twilio_sid: str = ""
    twilio_token: str = ""
    twilio_from: str = ""
    sendgrid_api_key: str = ""
    from_email: str = ""
    notifier_port: int = 8002


@dataclass(frozen=True)
class SchedulerConfig:
    """Notification dispatch scheduling."""
    cron_expression: str = "*/15 * * * *"
    max_retries: int = 3
    retry_delay_seconds: int = 30


@dataclass(frozen=True)
class WismoConfig:
    """Top-level configuration for the WISMO stack."""
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    carrier: CarrierConfig = field(default_factory=CarrierConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    slack_webhook: str = ""

    @classmethod
    def from_env(cls) -> WismoConfig:
        """Build config from environment variables with validation."""
        cfg = cls(
            db=DatabaseConfig(
                url=os.environ["DATABASE_URL"],
                pool_min=int(os.getenv("DB_POOL_MIN", "5")),
                pool_max=int(os.getenv("DB_POOL_MAX", "20")),
            ),
            carrier=CarrierConfig(
                fedex_secret=os.environ["FEDEX_WEBHOOK_SECRET"],
                ups_secret=os.environ["UPS_WEBHOOK_SECRET"],
                usps_secret=os.environ["USPS_WEBHOOK_SECRET"],
                webhook_port=int(os.getenv("WEBHOOK_PORT", "8001")),
            ),
            notification=NotificationConfig(
                twilio_sid=os.environ["TWILIO_SID"],
                twilio_token=os.environ["TWILIO_TOKEN"],
                twilio_from=os.environ["TWILIO_FROM"],
                sendgrid_api_key=os.environ["SENDGRID_API_KEY"],
                from_email=os.environ["FROM_EMAIL"],
                notifier_port=int(os.getenv("NOTIFIER_PORT", "8002")),
            ),
            scheduler=SchedulerConfig(
                cron_expression=os.getenv("DISPATCH_CRON", "*/15 * * * *"),
                max_retries=int(os.getenv("DISPATCH_MAX_RETRIES", "3")),
                retry_delay_seconds=int(os.getenv("DISPATCH_RETRY_DELAY", "30")),
            ),
            slack_webhook=os.getenv("SLACK_WEBHOOK_URL", ""),
        )
        logger.info(
            "Loaded WismoConfig — webhook:%d, notifier:%d, cron=%s",
            cfg.carrier.webhook_port, cfg.notification.notifier_port,
            cfg.scheduler.cron_expression,
        )
        return cfg`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Automatically notify customers about shipment milestones and delays before they reach out to support, and monitor the WISMO stack for carrier feed outages, notification delivery failures, and stale-shipment surges.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Proactive Notification Dispatcher',
                description:
                  'Monitors the tracking view for meaningful status changes and stale shipments, then sends SMS and email notifications through Twilio and SendGrid.',
                code: `# proactive_wismo_notifier.py
import pandas as pd
from sqlalchemy import create_engine
from twilio.rest import Client as TwilioClient
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import os

twilio = TwilioClient(os.environ["TWILIO_SID"], os.environ["TWILIO_TOKEN"])
sendgrid = SendGridAPIClient(os.environ["SENDGRID_API_KEY"])
engine = create_engine(os.environ["DATABASE_URL"])

TEMPLATES = {
    "shipped":   "Great news! Your order {order_id} has shipped. Track it here: {tracking_url}",
    "in_transit": "Update: Your order {order_id} is on the move — last seen in {location}.",
    "delayed":   "We know you are waiting on order {order_id}. It is taking longer than expected. "
                 "Estimated delivery: {est_date}. We are on it.",
    "delivered": "Your order {order_id} has been delivered! We hope you love it.",
}

def send_sms(phone: str, message: str):
    twilio.messages.create(body=message, from_=os.environ["TWILIO_FROM"], to=phone)

def send_email(email: str, subject: str, body: str):
    msg = Mail(from_email=os.environ["FROM_EMAIL"], to_emails=email,
               subject=subject, plain_text_content=body)
    sendgrid.send(msg)

def dispatch_notifications():
    pending = pd.read_sql("""
        SELECT ot.order_id, ot.customer_id, ot.carrier_status,
               ot.last_known_location, ot.estimated_delivery_date,
               ot.tracking_number, ot.carrier_code,
               c.email, c.phone, ot.hours_since_last_scan
        FROM logistics.order_tracking_live ot
        JOIN analytics.customer_360 c ON ot.customer_id = c.customer_id
        LEFT JOIN notifications.sent_log sl
          ON ot.order_id = sl.order_id AND ot.carrier_status = sl.event_type
        WHERE sl.id IS NULL
          AND ot.carrier_status IN ('shipped','in_transit','delivered')
        UNION ALL
        SELECT s.order_id, s.customer_id, 'delayed' AS carrier_status,
               s.last_known_location, s.estimated_delivery_date,
               s.tracking_number, s.carrier_code,
               s.email, s.phone, s.hours_since_last_scan
        FROM logistics.stale_shipments s
        LEFT JOIN notifications.sent_log sl
          ON s.order_id = sl.order_id AND sl.event_type = 'delayed'
        WHERE sl.id IS NULL AND s.staleness_tier IN ('high','critical')
    """, engine)

    for _, row in pending.iterrows():
        tracking_url = f"https://track.example.com/{row['carrier_code']}/{row['tracking_number']}"
        msg = TEMPLATES[row["carrier_status"]].format(
            order_id=row["order_id"], location=row["last_known_location"],
            est_date=row["estimated_delivery_date"], tracking_url=tracking_url,
        )
        if row.get("phone"):
            send_sms(row["phone"], msg)
        if row.get("email"):
            send_email(row["email"], f"Order {row['order_id']} — {row['carrier_status'].replace('_',' ').title()}", msg)

    return len(pending)`,
              },
              {
                language: 'sql',
                title: 'WISMO Stack Operational Health Queries',
                description:
                  'Monitoring queries that detect carrier feed outages, notification delivery failures, and surges in stale shipments requiring ops intervention.',
                code: `-- monitor_wismo_health.sql
-- Monitor 1: Carrier feed outage detection per carrier
SELECT carrier_code,
       MAX(scanned_at) AS last_event,
       EXTRACT(EPOCH FROM
         (CURRENT_TIMESTAMP - MAX(scanned_at))) / 3600.0
         AS hours_since_last_event,
       CASE
         WHEN EXTRACT(EPOCH FROM
           (CURRENT_TIMESTAMP - MAX(scanned_at))) / 3600.0 > 2 THEN 'OUTAGE'
         WHEN EXTRACT(EPOCH FROM
           (CURRENT_TIMESTAMP - MAX(scanned_at))) / 3600.0 > 1 THEN 'WARNING'
         ELSE 'OK'
       END AS feed_status
FROM logistics.carrier_events
GROUP BY carrier_code
ORDER BY hours_since_last_event DESC;

-- Monitor 2: Notification delivery failure rate (last 24h)
SELECT
  COUNT(*) AS total_attempted,
  SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) AS delivered,
  SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
  ROUND(100.0 * SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)
        / GREATEST(COUNT(*), 1), 2) AS failure_rate_pct
FROM notifications.sent_log
WHERE sent_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours';

-- Monitor 3: Stale shipment surge detection
SELECT
  CURRENT_DATE AS check_date,
  COUNT(*) AS stale_shipments_today,
  LAG(COUNT(*)) OVER (ORDER BY CURRENT_DATE) AS stale_yesterday,
  CASE
    WHEN COUNT(*) > LAG(COUNT(*)) OVER (ORDER BY CURRENT_DATE) * 1.5
    THEN 'SPIKE'
    ELSE 'NORMAL'
  END AS trend
FROM logistics.stale_shipments
GROUP BY CURRENT_DATE;`,
              },
            ],
          },
        ],
      },
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to automatically monitor shipment statuses, detect delays, compose personalized customer notifications, and reduce WISMO tickets without building a custom tracking system.',
        estimatedMonthlyCost: '$130 - $220/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Twilio ($20/mo)', 'AfterShip ($29/mo)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Freshdesk AI ($35/mo)', 'Shippo ($10/mo)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Configure Zapier to monitor shipping status changes from carriers via AfterShip or direct carrier APIs, extracting order details and customer information for AI-powered notification generation.',
            toolsUsed: ['Zapier', 'AfterShip API', 'Webhooks'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Shipment Monitoring Configuration',
                description: 'Configure Zapier to trigger on shipment status changes and fetch customer context.',
                code: `{
  "trigger": {
    "app": "AfterShip",
    "event": "Tracking Update",
    "filters": {
      "tag": ["InTransit", "OutForDelivery", "Delivered", "Exception", "FailedAttempt"]
    }
  },
  "actions": [
    {
      "step": 1,
      "app": "Webhooks by Zapier",
      "action": "GET",
      "url": "{{ORDER_API_URL}}/orders/{{tracking_number}}",
      "headers": {
        "Authorization": "Bearer {{ORDER_API_KEY}}"
      }
    },
    {
      "step": 2,
      "app": "Webhooks by Zapier",
      "action": "GET",
      "url": "{{CUSTOMER_API_URL}}/customers/{{order_customer_email}}",
      "headers": {
        "Authorization": "Bearer {{CUSTOMER_API_KEY}}"
      }
    },
    {
      "step": 3,
      "app": "Code by Zapier",
      "action": "Run JavaScript",
      "config": {
        "code": "const now = new Date(); const eta = new Date(inputData.expected_delivery); const daysLate = Math.ceil((now - eta) / (1000 * 60 * 60 * 24)); const isDelayed = daysLate > 0; const lastScan = new Date(inputData.last_checkpoint_time); const hoursSinceLastScan = Math.ceil((now - lastScan) / (1000 * 60 * 60)); return { is_delayed: isDelayed, days_late: Math.max(0, daysLate), hours_since_scan: hoursSinceLastScan, is_stale: hoursSinceLastScan > 48 };"
      }
    }
  ],
  "output": {
    "shipment_data": {
      "tracking_number": "{{tracking_number}}",
      "carrier": "{{courier_name}}",
      "status": "{{tag}}",
      "status_detail": "{{message}}",
      "last_location": "{{last_checkpoint_location}}",
      "last_scan_time": "{{last_checkpoint_time}}",
      "expected_delivery": "{{expected_delivery}}",
      "is_delayed": "{{step3_is_delayed}}",
      "days_late": "{{step3_days_late}}",
      "hours_since_scan": "{{step3_hours_since_scan}}",
      "is_stale": "{{step3_is_stale}}"
    },
    "order_data": {
      "order_id": "{{step1_order_id}}",
      "order_date": "{{step1_order_date}}",
      "items": "{{step1_line_items}}"
    },
    "customer_data": {
      "email": "{{step2_email}}",
      "phone": "{{step2_phone}}",
      "name": "{{step2_first_name}}",
      "tier": "{{step2_customer_tier}}",
      "preferred_channel": "{{step2_notification_preference}}"
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
              'Send shipment context to ChatGPT or Claude to compose personalized, empathetic customer notifications tailored to the situation (delay, exception, delivery) and customer tier.',
            toolsUsed: ['ChatGPT API', 'Claude API'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Shipment Notification Prompt Template',
                description: 'A structured prompt that generates personalized shipping notifications.',
                code: `system_prompt: |
  You are a customer communication specialist for an e-commerce company.
  Your job is to compose shipping status notifications that are:
  1. Clear and informative
  2. Empathetic when there are delays
  3. Proactive in setting expectations
  4. Tailored to the customer tier (VIP gets more detail)

  Tone guidelines:
  - Delays: Apologetic but solution-focused
  - On-track: Friendly and informative
  - Delivered: Celebratory and inviting feedback
  - Exception: Urgent but reassuring

user_prompt_template: |
  Compose a shipping notification for this customer:

  ## Shipment Status
  **Carrier:** {{carrier}}
  **Status:** {{status}} - {{status_detail}}
  **Last Location:** {{last_location}}
  **Last Scan:** {{hours_since_scan}} hours ago
  **Expected Delivery:** {{expected_delivery}}
  **Days Late:** {{days_late}}

  ## Order Details
  **Order ID:** {{order_id}}
  **Order Date:** {{order_date}}
  **Items:** {{items}}

  ## Customer
  **Name:** {{customer_name}}
  **Tier:** {{customer_tier}}
  **Preferred Channel:** {{preferred_channel}}

  ---
  Generate a notification message appropriate for {{preferred_channel}}
  (SMS should be under 160 characters, email can be longer).

  Return JSON:
  {
    "subject": "<email subject if email>",
    "message": "<notification body>",
    "urgency": "low|medium|high",
    "include_tracking_link": true|false,
    "offer_support_contact": true|false,
    "compensation_suggested": true|false,
    "compensation_reason": "<if applicable>"
  }

parameters:
  model: "gpt-4-turbo"
  temperature: 0.4
  max_tokens: 400
  response_format: { "type": "json_object" }`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier to send AI-generated notifications via the customer preferred channel (SMS via Twilio, email via SendGrid), update the order system, and alert support for high-risk shipments.',
            toolsUsed: ['Zapier', 'Twilio', 'SendGrid', 'Slack'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Proactive Notification Workflow',
                description: 'Complete workflow that sends personalized notifications and handles exceptions.',
                code: `{
  "workflow_name": "WISMO Proactive Notification System",
  "steps": [
    {
      "step": 4,
      "app": "OpenAI (ChatGPT)",
      "action": "Send Prompt",
      "config": {
        "model": "gpt-4-turbo",
        "system_message": "{{system_prompt}}",
        "user_message": "{{formatted_shipment_prompt}}",
        "temperature": 0.4,
        "response_format": "json"
      }
    },
    {
      "step": 5,
      "app": "Code by Zapier",
      "action": "Run JavaScript",
      "config": {
        "code": "const response = JSON.parse(inputData.chatgpt_response); return { subject: response.subject || 'Shipping Update', message: response.message, urgency: response.urgency, include_link: response.include_tracking_link, offer_support: response.offer_support_contact, suggest_comp: response.compensation_suggested };"
      }
    },
    {
      "step": 6,
      "app": "Paths by Zapier",
      "paths": [
        {
          "name": "SMS Notification",
          "condition": "preferred_channel EQUALS sms",
          "steps": [
            {
              "app": "Twilio",
              "action": "Send SMS",
              "config": {
                "to": "{{customer_phone}}",
                "body": "{{message}} Track: {{tracking_url}}",
                "from": "{{TWILIO_FROM_NUMBER}}"
              }
            }
          ]
        },
        {
          "name": "Email Notification",
          "condition": "preferred_channel EQUALS email",
          "steps": [
            {
              "app": "SendGrid",
              "action": "Send Email",
              "config": {
                "to": "{{customer_email}}",
                "from": "shipping@example.com",
                "subject": "{{subject}}",
                "template_id": "{{SHIPPING_UPDATE_TEMPLATE}}",
                "dynamic_template_data": {
                  "customer_name": "{{customer_name}}",
                  "message": "{{message}}",
                  "tracking_url": "{{tracking_url}}",
                  "order_id": "{{order_id}}",
                  "carrier": "{{carrier}}",
                  "show_support_cta": "{{offer_support}}"
                }
              }
            }
          ]
        }
      ]
    },
    {
      "step": 7,
      "app": "Filter by Zapier",
      "condition": "urgency EQUALS high OR suggest_comp EQUALS true"
    },
    {
      "step": 8,
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#shipping-alerts",
        "message": ":package: *High-Priority Shipment Alert*\\n\\n*Order:* {{order_id}}\\n*Customer:* {{customer_name}} ({{customer_tier}})\\n*Status:* {{status}} - {{status_detail}}\\n*Days Late:* {{days_late}}\\n*Compensation Suggested:* {{suggest_comp}}\\n\\n*AI Message Sent:*\\n> {{message}}\\n\\n<{{tracking_url}}|Track Shipment> | <{{order_admin_url}}|View Order>"
      }
    },
    {
      "step": 9,
      "app": "Webhooks by Zapier",
      "action": "POST",
      "url": "{{ORDER_API_URL}}/orders/{{order_id}}/notes",
      "headers": {
        "Authorization": "Bearer {{ORDER_API_KEY}}",
        "Content-Type": "application/json"
      },
      "body": {
        "note_type": "shipping_notification",
        "channel": "{{preferred_channel}}",
        "message_sent": "{{message}}",
        "status_at_send": "{{status}}",
        "ai_generated": true,
        "timestamp": "{{zap_meta_human_now}}"
      }
    }
  ],
  "error_handling": {
    "on_twilio_error": {
      "fallback": "SendGrid Email",
      "log_to": "#shipping-errors"
    },
    "on_chatgpt_error": {
      "use_template": "default_{{status}}_template"
    }
  }
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent system that ingests carrier tracking data in real-time, predicts delivery delays before they happen, orchestrates proactive customer communications, and automatically handles WISMO ticket deflection.',
        estimatedMonthlyCost: '$700 - $1,500/month',
        architecture:
          'A supervisor agent coordinates four specialist agents: Carrier Ingestion Agent, Delay Prediction Agent, Notification Orchestrator, and Ticket Deflection Agent. Real-time data flows through Kafka with state in Redis.',
        agents: [
          {
            name: 'CarrierIngestionAgent',
            role: 'Carrier Data Ingestion Specialist',
            goal: 'Ingest and normalize tracking data from multiple carriers in real-time via webhooks and APIs',
            tools: ['FastAPI Webhooks', 'Kafka', 'Carrier APIs', 'PostgreSQL'],
          },
          {
            name: 'DelayPredictionAgent',
            role: 'Delivery Delay Predictor',
            goal: 'Predict shipment delays before they happen using ML models trained on historical carrier performance',
            tools: ['XGBoost', 'Weather API', 'Carrier Performance DB', 'Feature Store'],
          },
          {
            name: 'NotificationOrchestratorAgent',
            role: 'Proactive Notification Manager',
            goal: 'Orchestrate multi-channel customer notifications with optimal timing and personalized messaging',
            tools: ['GPT-4', 'Twilio', 'SendGrid', 'Customer Preferences API'],
          },
          {
            name: 'TicketDeflectionAgent',
            role: 'WISMO Ticket Deflection Specialist',
            goal: 'Intercept WISMO tickets and provide instant, accurate order status via AI chatbot integration',
            tools: ['Intercom API', 'Zendesk API', 'Order Tracking API', 'GPT-4'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Kafka for event streaming, Redis for real-time state, PostgreSQL for persistence',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system with CrewAI, establishing specialized roles for carrier data ingestion, delay prediction, notification orchestration, and WISMO ticket deflection.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'WISMO Agent Definitions',
                description: 'CrewAI agent definitions for the WISMO crisis multi-agent system.',
                code: `# agents/wismo_agents.py
"""Multi-agent system for WISMO crisis resolution."""
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class WISMOAgentFactory:
    """Factory for creating WISMO resolution specialist agents."""

    def __init__(self, llm_model: str = "gpt-4-turbo"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)

    def create_carrier_ingestion_agent(self) -> Agent:
        """Agent specialized in carrier data ingestion and normalization."""
        return Agent(
            role="Carrier Data Ingestion Specialist",
            goal="Ingest and normalize tracking data from all major carriers "
                 "in real-time, ensuring consistent status mapping",
            backstory="""You are an expert in logistics data integration
            with deep knowledge of carrier APIs and webhook formats. You
            understand the nuances of different carrier status codes and
            can map them to a unified tracking taxonomy. You handle rate
            limits, retries, and data quality issues gracefully.""",
            llm=self.llm,
            tools=[],  # Tools injected at runtime
            verbose=True,
            allow_delegation=False,
        )

    def create_delay_prediction_agent(self) -> Agent:
        """Agent specialized in predicting delivery delays."""
        return Agent(
            role="Delivery Delay Predictor",
            goal="Predict shipment delays before customers notice them, "
                 "enabling proactive communication",
            backstory="""You are a predictive analytics expert specializing
            in logistics forecasting. You understand how weather, carrier
            performance patterns, and shipping lane congestion affect
            delivery times. You calculate confidence intervals and know
            when predictions are reliable vs uncertain.""",
            llm=self.llm,
            tools=[],
            verbose=True,
            allow_delegation=False,
        )

    def create_notification_orchestrator(self) -> Agent:
        """Agent specialized in customer notification orchestration."""
        return Agent(
            role="Proactive Notification Manager",
            goal="Orchestrate personalized, well-timed notifications that "
                 "keep customers informed and reduce inbound contacts",
            backstory="""You are a customer communication expert who
            understands notification fatigue and optimal timing. You craft
            messages that are informative without being overwhelming, and
            you know when to use SMS vs email vs push notifications. You
            personalize tone based on customer history and preferences.""",
            llm=self.llm,
            tools=[],
            verbose=True,
            allow_delegation=True,
        )

    def create_ticket_deflection_agent(self) -> Agent:
        """Agent specialized in WISMO ticket deflection."""
        return Agent(
            role="WISMO Ticket Deflection Specialist",
            goal="Intercept and resolve WISMO inquiries instantly via AI, "
                 "deflecting tickets before they reach human agents",
            backstory="""You are a conversational AI expert who can
            understand customer intent and provide accurate order status
            information instantly. You know when to provide self-service
            answers and when to escalate to human agents. You handle
            frustrated customers with empathy while being efficient.""",
            llm=self.llm,
            tools=[],
            verbose=True,
            allow_delegation=False,
        )

    def create_crew(self) -> Crew:
        """Assemble the complete WISMO resolution agent crew."""
        ingestion = self.create_carrier_ingestion_agent()
        prediction = self.create_delay_prediction_agent()
        notification = self.create_notification_orchestrator()
        deflection = self.create_ticket_deflection_agent()

        tasks = [
            Task(
                description="Ingest and normalize carrier tracking updates",
                agent=ingestion,
                expected_output="Normalized tracking events with unified status",
            ),
            Task(
                description="Predict potential delays for active shipments",
                agent=prediction,
                expected_output="Delay predictions with confidence scores",
            ),
            Task(
                description="Orchestrate proactive customer notifications",
                agent=notification,
                expected_output="Notification queue with personalized messages",
            ),
            Task(
                description="Handle incoming WISMO inquiries via AI",
                agent=deflection,
                expected_output="Resolution or escalation decision",
            ),
        ]

        return Crew(
            agents=[ingestion, prediction, notification, deflection],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the Carrier Ingestion agent with tools for receiving webhooks from multiple carriers, normalizing status codes, and publishing events to Kafka for downstream processing.',
            toolsUsed: ['FastAPI', 'Kafka', 'PostgreSQL'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Carrier Webhook Ingestion Tools',
                description: 'Tools for the Carrier Ingestion agent to normalize and publish tracking events.',
                code: `# tools/carrier_ingestion_tools.py
"""Carrier data ingestion and normalization tools."""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from kafka import KafkaProducer
from sqlalchemy import create_engine, text
import json
import hashlib
import hmac
import os
import logging

logger = logging.getLogger(__name__)


class CarrierEventInput(BaseModel):
    """Input schema for carrier tracking events."""
    carrier: str = Field(..., description="Carrier code (fedex, ups, usps, etc.)")
    tracking_number: str = Field(..., description="Tracking number")
    raw_status: str = Field(..., description="Raw carrier status code")
    location: Optional[str] = Field(None, description="Last scan location")
    timestamp: str = Field(..., description="Event timestamp ISO format")
    raw_payload: Dict[str, Any] = Field(default_factory=dict, description="Raw carrier payload")


# Unified status taxonomy
UNIFIED_STATUS_MAP = {
    # FedEx mappings
    "fedex": {
        "PU": "picked_up",
        "IT": "in_transit",
        "OD": "out_for_delivery",
        "DL": "delivered",
        "DE": "exception",
        "SE": "exception",
        "CA": "cancelled",
    },
    # UPS mappings
    "ups": {
        "M": "label_created",
        "P": "picked_up",
        "I": "in_transit",
        "O": "out_for_delivery",
        "D": "delivered",
        "X": "exception",
        "RS": "returned",
    },
    # USPS mappings
    "usps": {
        "MA": "label_created",
        "AC": "picked_up",
        "OF": "in_transit",
        "OD": "out_for_delivery",
        "01": "delivered",
        "09": "exception",
        "04": "returned",
    },
    # DHL mappings
    "dhl": {
        "PU": "picked_up",
        "TD": "in_transit",
        "WC": "out_for_delivery",
        "OK": "delivered",
        "OH": "exception",
    },
}


class CarrierEventNormalizerTool(BaseTool):
    """Tool for normalizing carrier tracking events."""

    name: str = "normalize_carrier_event"
    description: str = (
        "Normalize a raw carrier tracking event into unified status taxonomy. "
        "Maps carrier-specific status codes to standard statuses."
    )
    args_schema: type[BaseModel] = CarrierEventInput

    def _run(
        self,
        carrier: str,
        tracking_number: str,
        raw_status: str,
        timestamp: str,
        location: Optional[str] = None,
        raw_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Normalize carrier event to unified format."""
        carrier_lower = carrier.lower()
        status_map = UNIFIED_STATUS_MAP.get(carrier_lower, {})
        unified_status = status_map.get(raw_status, "unknown")

        # Parse timestamp
        try:
            event_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            event_time = datetime.utcnow()

        # Generate event ID
        event_id = hashlib.sha256(
            f"{tracking_number}:{raw_status}:{timestamp}".encode()
        ).hexdigest()[:16]

        normalized = {
            "event_id": event_id,
            "tracking_number": tracking_number,
            "carrier": carrier_lower,
            "raw_status": raw_status,
            "unified_status": unified_status,
            "location": location or "unknown",
            "event_timestamp": event_time.isoformat(),
            "ingested_at": datetime.utcnow().isoformat(),
            "is_terminal": unified_status in ["delivered", "returned", "cancelled"],
        }

        return normalized


class KafkaEventPublisherTool(BaseTool):
    """Tool for publishing normalized events to Kafka."""

    name: str = "publish_tracking_event"
    description: str = (
        "Publish normalized tracking event to Kafka for downstream processing. "
        "Events are partitioned by tracking number for ordering guarantees."
    )

    def __init__(self, bootstrap_servers: str = None):
        super().__init__()
        self.bootstrap_servers = bootstrap_servers or os.environ.get(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self._producer = None

    @property
    def producer(self):
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
            )
        return self._producer

    def _run(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Publish event to Kafka topic."""
        topic = "tracking-events"
        key = event["tracking_number"]

        try:
            future = self.producer.send(topic, key=key, value=event)
            record_metadata = future.get(timeout=10)

            return {
                "status": "published",
                "topic": record_metadata.topic,
                "partition": record_metadata.partition,
                "offset": record_metadata.offset,
                "event_id": event["event_id"],
            }
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return {"status": "failed", "error": str(e)}


class ShipmentStatePersisterTool(BaseTool):
    """Tool for persisting shipment state to database."""

    name: str = "persist_shipment_state"
    description: str = (
        "Persist or update shipment state in the database. "
        "Maintains current status, location, and tracking history."
    )

    def __init__(self):
        super().__init__()
        self.engine = create_engine(os.environ["DATABASE_URL"])

    def _run(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Persist shipment state to database."""
        with self.engine.connect() as conn:
            # Upsert shipment state
            conn.execute(
                text("""
                    INSERT INTO logistics.shipment_states
                    (tracking_number, carrier, current_status, last_location,
                     last_event_at, is_delivered, updated_at)
                    VALUES (:tn, :carrier, :status, :location, :event_at, :delivered, NOW())
                    ON CONFLICT (tracking_number) DO UPDATE SET
                        current_status = EXCLUDED.current_status,
                        last_location = EXCLUDED.last_location,
                        last_event_at = EXCLUDED.last_event_at,
                        is_delivered = EXCLUDED.is_delivered,
                        updated_at = NOW()
                """),
                {
                    "tn": event["tracking_number"],
                    "carrier": event["carrier"],
                    "status": event["unified_status"],
                    "location": event["location"],
                    "event_at": event["event_timestamp"],
                    "delivered": event["is_terminal"] and event["unified_status"] == "delivered",
                }
            )

            # Insert into event history
            conn.execute(
                text("""
                    INSERT INTO logistics.tracking_events
                    (event_id, tracking_number, carrier, status, location, event_at, raw_status)
                    VALUES (:eid, :tn, :carrier, :status, :location, :event_at, :raw)
                    ON CONFLICT (event_id) DO NOTHING
                """),
                {
                    "eid": event["event_id"],
                    "tn": event["tracking_number"],
                    "carrier": event["carrier"],
                    "status": event["unified_status"],
                    "location": event["location"],
                    "event_at": event["event_timestamp"],
                    "raw": event["raw_status"],
                }
            )
            conn.commit()

        return {
            "status": "persisted",
            "tracking_number": event["tracking_number"],
            "unified_status": event["unified_status"],
        }`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Delay Prediction agent with ML-based forecasting and the Notification Orchestrator agent for personalized message generation and channel selection.',
            toolsUsed: ['XGBoost', 'Weather API', 'GPT-4', 'Twilio', 'SendGrid'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Delay Prediction and Notification Tools',
                description: 'Tools for predicting delays and orchestrating customer notifications.',
                code: `# tools/wismo_analysis_tools.py
"""Delay prediction and notification orchestration tools."""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from twilio.rest import Client as TwilioClient
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import os
import logging

logger = logging.getLogger(__name__)


class ShipmentInput(BaseModel):
    """Input schema for shipment analysis."""
    tracking_number: str = Field(..., description="Tracking number")
    carrier: str = Field(..., description="Carrier code")
    origin_zip: Optional[str] = Field(None, description="Origin ZIP code")
    destination_zip: Optional[str] = Field(None, description="Destination ZIP code")


class DelayPredictionTool(BaseTool):
    """Tool for predicting shipment delivery delays."""

    name: str = "predict_delivery_delay"
    description: str = (
        "Predict if a shipment will be delayed and by how much. "
        "Uses ML model trained on historical carrier performance."
    )
    args_schema: type[BaseModel] = ShipmentInput

    def __init__(self, model_path: str = "/models/delay_predictor.joblib"):
        super().__init__()
        self.engine = create_engine(os.environ["DATABASE_URL"])
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                self._model = joblib.load(self.model_path)
            except FileNotFoundError:
                logger.warning("Delay model not found, using rule-based fallback")
        return self._model

    def _get_carrier_performance(self, carrier: str) -> Dict[str, float]:
        """Get historical carrier performance metrics."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        AVG(CASE WHEN actual_delivery <= expected_delivery THEN 1 ELSE 0 END)
                            as on_time_rate,
                        AVG(EXTRACT(DAY FROM actual_delivery - expected_delivery))
                            as avg_delay_days,
                        STDDEV(EXTRACT(DAY FROM actual_delivery - expected_delivery))
                            as delay_stddev
                    FROM logistics.delivery_history
                    WHERE carrier = :carrier
                      AND actual_delivery IS NOT NULL
                      AND shipped_at >= CURRENT_DATE - INTERVAL '90 days'
                """),
                {"carrier": carrier}
            ).fetchone()

            if result:
                return {
                    "on_time_rate": float(result.on_time_rate or 0.85),
                    "avg_delay_days": float(result.avg_delay_days or 0.5),
                    "delay_stddev": float(result.delay_stddev or 1.0),
                }
            return {"on_time_rate": 0.85, "avg_delay_days": 0.5, "delay_stddev": 1.0}

    def _get_shipment_features(self, tracking_number: str) -> Dict[str, Any]:
        """Get current shipment features for prediction."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        ss.current_status,
                        ss.last_event_at,
                        EXTRACT(EPOCH FROM (NOW() - ss.last_event_at)) / 3600 as hours_since_scan,
                        o.expected_delivery,
                        EXTRACT(DAY FROM o.expected_delivery - CURRENT_DATE) as days_until_eta
                    FROM logistics.shipment_states ss
                    JOIN orders.orders o ON ss.tracking_number = o.tracking_number
                    WHERE ss.tracking_number = :tn
                """),
                {"tn": tracking_number}
            ).fetchone()

            if result:
                return {
                    "status": result.current_status,
                    "hours_since_scan": float(result.hours_since_scan or 0),
                    "days_until_eta": float(result.days_until_eta or 0),
                    "expected_delivery": result.expected_delivery,
                }
            return None

    def _run(
        self,
        tracking_number: str,
        carrier: str,
        origin_zip: Optional[str] = None,
        destination_zip: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Predict delivery delay for shipment."""
        features = self._get_shipment_features(tracking_number)
        if not features:
            return {"error": f"Shipment {tracking_number} not found"}

        carrier_perf = self._get_carrier_performance(carrier)

        # Feature engineering
        hours_since_scan = features["hours_since_scan"]
        days_until_eta = features["days_until_eta"]
        status = features["status"]

        # Rule-based scoring when no ML model
        delay_score = 0
        risk_factors = []

        # Hours since last scan
        if hours_since_scan > 72:
            delay_score += 40
            risk_factors.append(f"No scan in {int(hours_since_scan)} hours")
        elif hours_since_scan > 48:
            delay_score += 25
            risk_factors.append(f"Last scan {int(hours_since_scan)} hours ago")

        # Status-based risk
        if status == "exception":
            delay_score += 50
            risk_factors.append("Shipment has exception status")
        elif status == "in_transit" and days_until_eta <= 0:
            delay_score += 30
            risk_factors.append("Past expected delivery date")

        # Carrier performance
        if carrier_perf["on_time_rate"] < 0.80:
            delay_score += 15
            risk_factors.append(f"Carrier on-time rate: {carrier_perf['on_time_rate']:.0%}")

        # Predict delay days
        if delay_score > 50:
            predicted_delay = carrier_perf["avg_delay_days"] + carrier_perf["delay_stddev"]
        elif delay_score > 25:
            predicted_delay = carrier_perf["avg_delay_days"]
        else:
            predicted_delay = 0

        return {
            "tracking_number": tracking_number,
            "delay_risk_score": min(100, delay_score),
            "delay_probability": min(0.95, delay_score / 100),
            "predicted_delay_days": round(predicted_delay, 1),
            "new_eta": (
                (features["expected_delivery"] + timedelta(days=predicted_delay)).isoformat()
                if features["expected_delivery"] and predicted_delay > 0
                else features["expected_delivery"].isoformat() if features["expected_delivery"] else None
            ),
            "risk_level": "high" if delay_score > 60 else "medium" if delay_score > 30 else "low",
            "risk_factors": risk_factors,
            "carrier_on_time_rate": carrier_perf["on_time_rate"],
        }


class NotificationGeneratorTool(BaseTool):
    """Tool for generating personalized shipping notifications."""

    name: str = "generate_notification"
    description: str = (
        "Generate a personalized shipping notification based on shipment status, "
        "delay prediction, and customer preferences."
    )

    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.4)
        self.engine = create_engine(os.environ["DATABASE_URL"])

    def _get_customer_context(self, tracking_number: str) -> Dict[str, Any]:
        """Get customer context for personalization."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT
                        c.customer_id, c.first_name, c.email, c.phone,
                        c.notification_preference, c.customer_tier,
                        o.order_id, o.items_summary
                    FROM orders.orders o
                    JOIN customers.customers c ON o.customer_id = c.customer_id
                    WHERE o.tracking_number = :tn
                """),
                {"tn": tracking_number}
            ).fetchone()

            if result:
                return {
                    "customer_id": result.customer_id,
                    "first_name": result.first_name,
                    "email": result.email,
                    "phone": result.phone,
                    "channel": result.notification_preference or "email",
                    "tier": result.customer_tier,
                    "order_id": result.order_id,
                    "items": result.items_summary,
                }
            return None

    def _run(
        self,
        tracking_number: str,
        status: str,
        delay_prediction: Dict[str, Any],
        location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate personalized notification."""
        customer = self._get_customer_context(tracking_number)
        if not customer:
            return {"error": "Customer context not found"}

        # Build prompt
        prompt = f"""
        Generate a shipping notification for this customer:

        Customer: {customer['first_name']} ({customer['tier']} tier)
        Channel: {customer['channel']} (SMS max 160 chars)
        Order: {customer['order_id']} - {customer['items']}

        Shipment Status: {status}
        Location: {location or 'In transit'}
        Delay Risk: {delay_prediction['risk_level']}
        Predicted Delay: {delay_prediction['predicted_delay_days']} days
        Risk Factors: {', '.join(delay_prediction.get('risk_factors', []))}

        Generate an appropriate message. Be empathetic for delays, celebratory for delivery.
        For VIP customers, offer expedited resolution options.

        Return JSON:
        {{
            "subject": "<email subject>",
            "message": "<notification body>",
            "tone": "informative|apologetic|celebratory",
            "offer_compensation": true/false,
            "escalate_to_human": true/false
        }}
        """

        response = self.llm.invoke(prompt)

        try:
            notification = json.loads(response.content)
        except:
            # Fallback template
            notification = {
                "subject": f"Update on your order {customer['order_id']}",
                "message": f"Hi {customer['first_name']}, your order is {status}.",
                "tone": "informative",
                "offer_compensation": False,
                "escalate_to_human": False,
            }

        return {
            "tracking_number": tracking_number,
            "customer_id": customer["customer_id"],
            "channel": customer["channel"],
            "recipient": customer["phone"] if customer["channel"] == "sms" else customer["email"],
            "notification": notification,
        }


class NotificationSenderTool(BaseTool):
    """Tool for sending notifications via appropriate channel."""

    name: str = "send_notification"
    description: str = (
        "Send notification to customer via their preferred channel (SMS or email)."
    )

    def __init__(self):
        super().__init__()
        self.twilio = TwilioClient(
            os.environ["TWILIO_SID"],
            os.environ["TWILIO_TOKEN"]
        )
        self.sendgrid = SendGridAPIClient(os.environ["SENDGRID_API_KEY"])
        self.engine = create_engine(os.environ["DATABASE_URL"])

    def _run(
        self,
        channel: str,
        recipient: str,
        notification: Dict[str, Any],
        tracking_number: str,
    ) -> Dict[str, Any]:
        """Send notification and log result."""
        try:
            if channel == "sms":
                message = self.twilio.messages.create(
                    body=notification["message"][:160],
                    from_=os.environ["TWILIO_FROM"],
                    to=recipient
                )
                delivery_status = "sent"
                message_id = message.sid
            else:
                mail = Mail(
                    from_email=os.environ["FROM_EMAIL"],
                    to_emails=recipient,
                    subject=notification["subject"],
                    plain_text_content=notification["message"]
                )
                response = self.sendgrid.send(mail)
                delivery_status = "sent" if response.status_code < 400 else "failed"
                message_id = response.headers.get("X-Message-Id", "unknown")

            # Log notification
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO notifications.sent_log
                        (tracking_number, channel, recipient, message_id,
                         status, sent_at, notification_type)
                        VALUES (:tn, :channel, :recipient, :mid, :status, NOW(), 'shipping')
                    """),
                    {
                        "tn": tracking_number,
                        "channel": channel,
                        "recipient": recipient[:50],
                        "mid": message_id,
                        "status": delivery_status,
                    }
                )
                conn.commit()

            return {
                "status": delivery_status,
                "channel": channel,
                "message_id": message_id,
            }

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return {"status": "failed", "error": str(e)}`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement LangGraph-based orchestration that processes tracking events through prediction, notification, and ticket deflection in a coordinated real-time pipeline.',
            toolsUsed: ['LangGraph', 'Kafka', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Orchestration for WISMO',
                description: 'State machine orchestration for the WISMO resolution multi-agent pipeline.',
                code: `# orchestration/wismo_graph.py
"""LangGraph orchestration for WISMO multi-agent system."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict, Any, Optional, Literal
import redis
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class WISMOState(TypedDict):
    """State schema for WISMO pipeline."""
    # Input event
    tracking_number: str
    carrier: str
    raw_status: str
    location: Optional[str]
    event_timestamp: str

    # Normalized event
    normalized_event: Optional[Dict[str, Any]]

    # Delay prediction
    delay_prediction: Optional[Dict[str, Any]]
    requires_notification: bool

    # Notification
    notification: Optional[Dict[str, Any]]
    notification_sent: bool
    notification_result: Optional[Dict[str, Any]]

    # Ticket deflection context
    deflection_context: Optional[Dict[str, Any]]

    # Pipeline metadata
    processing_status: str
    errors: List[str]


class WISMOOrchestrator:
    """LangGraph-based orchestrator for WISMO pipeline."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the WISMO state graph."""
        graph = StateGraph(WISMOState)

        # Add nodes
        graph.add_node("normalize_event", self._normalize_event)
        graph.add_node("persist_state", self._persist_state)
        graph.add_node("predict_delay", self._predict_delay)
        graph.add_node("decide_notification", self._decide_notification)
        graph.add_node("generate_notification", self._generate_notification)
        graph.add_node("send_notification", self._send_notification)
        graph.add_node("update_deflection_context", self._update_deflection_context)

        # Add edges
        graph.add_edge("normalize_event", "persist_state")
        graph.add_edge("persist_state", "predict_delay")
        graph.add_edge("predict_delay", "decide_notification")

        # Conditional edge for notification
        graph.add_conditional_edges(
            "decide_notification",
            self._should_notify,
            {
                "notify": "generate_notification",
                "skip": "update_deflection_context",
            }
        )
        graph.add_edge("generate_notification", "send_notification")
        graph.add_edge("send_notification", "update_deflection_context")
        graph.add_edge("update_deflection_context", END)

        graph.set_entry_point("normalize_event")

        return graph.compile(checkpointer=self.checkpointer)

    def _normalize_event(self, state: WISMOState) -> WISMOState:
        """Node: Normalize carrier event."""
        from tools.carrier_ingestion_tools import CarrierEventNormalizerTool

        normalizer = CarrierEventNormalizerTool()
        normalized = normalizer._run(
            carrier=state["carrier"],
            tracking_number=state["tracking_number"],
            raw_status=state["raw_status"],
            timestamp=state["event_timestamp"],
            location=state.get("location"),
        )

        return {
            **state,
            "normalized_event": normalized,
            "processing_status": "normalized",
        }

    def _persist_state(self, state: WISMOState) -> WISMOState:
        """Node: Persist shipment state to database."""
        from tools.carrier_ingestion_tools import ShipmentStatePersisterTool

        if not state.get("normalized_event"):
            return state

        persister = ShipmentStatePersisterTool()
        persister._run(state["normalized_event"])

        return {
            **state,
            "processing_status": "persisted",
        }

    def _predict_delay(self, state: WISMOState) -> WISMOState:
        """Node: Predict delivery delay."""
        from tools.wismo_analysis_tools import DelayPredictionTool

        if not state.get("normalized_event"):
            return state

        event = state["normalized_event"]

        # Skip prediction for terminal events
        if event.get("is_terminal"):
            return {
                **state,
                "delay_prediction": {"risk_level": "none", "predicted_delay_days": 0},
                "processing_status": "terminal_event",
            }

        predictor = DelayPredictionTool()
        prediction = predictor._run(
            tracking_number=state["tracking_number"],
            carrier=state["carrier"],
        )

        return {
            **state,
            "delay_prediction": prediction,
            "processing_status": "delay_predicted",
        }

    def _decide_notification(self, state: WISMOState) -> WISMOState:
        """Node: Decide if notification is needed."""
        event = state.get("normalized_event", {})
        prediction = state.get("delay_prediction", {})

        # Notification triggers
        should_notify = False

        # Always notify for key status changes
        notify_statuses = ["picked_up", "out_for_delivery", "delivered", "exception"]
        if event.get("unified_status") in notify_statuses:
            should_notify = True

        # Notify for high delay risk
        if prediction.get("risk_level") == "high":
            should_notify = True

        # Check if we already notified for this status
        cache_key = f"notified:{state['tracking_number']}:{event.get('unified_status')}"
        if self.redis.get(cache_key):
            should_notify = False
        elif should_notify:
            # Mark as notified
            self.redis.setex(cache_key, 86400, "1")

        return {
            **state,
            "requires_notification": should_notify,
            "processing_status": "notification_decided",
        }

    def _should_notify(self, state: WISMOState) -> Literal["notify", "skip"]:
        """Conditional: Check if notification should be sent."""
        return "notify" if state.get("requires_notification") else "skip"

    def _generate_notification(self, state: WISMOState) -> WISMOState:
        """Node: Generate personalized notification."""
        from tools.wismo_analysis_tools import NotificationGeneratorTool

        generator = NotificationGeneratorTool()
        notification = generator._run(
            tracking_number=state["tracking_number"],
            status=state["normalized_event"]["unified_status"],
            delay_prediction=state.get("delay_prediction", {}),
            location=state.get("location"),
        )

        return {
            **state,
            "notification": notification,
            "processing_status": "notification_generated",
        }

    def _send_notification(self, state: WISMOState) -> WISMOState:
        """Node: Send notification to customer."""
        from tools.wismo_analysis_tools import NotificationSenderTool

        if not state.get("notification") or "error" in state.get("notification", {}):
            return {
                **state,
                "notification_sent": False,
                "errors": state.get("errors", []) + ["No notification to send"],
            }

        sender = NotificationSenderTool()
        result = sender._run(
            channel=state["notification"]["channel"],
            recipient=state["notification"]["recipient"],
            notification=state["notification"]["notification"],
            tracking_number=state["tracking_number"],
        )

        return {
            **state,
            "notification_sent": result["status"] == "sent",
            "notification_result": result,
            "processing_status": "notification_sent",
        }

    def _update_deflection_context(self, state: WISMOState) -> WISMOState:
        """Node: Update context for ticket deflection bot."""
        event = state.get("normalized_event", {})
        prediction = state.get("delay_prediction", {})

        # Cache context for chatbot deflection
        context = {
            "tracking_number": state["tracking_number"],
            "carrier": state["carrier"],
            "status": event.get("unified_status"),
            "location": event.get("location"),
            "last_update": event.get("event_timestamp"),
            "delay_risk": prediction.get("risk_level"),
            "predicted_delay": prediction.get("predicted_delay_days"),
            "new_eta": prediction.get("new_eta"),
            "updated_at": datetime.utcnow().isoformat(),
        }

        cache_key = f"shipment_context:{state['tracking_number']}"
        self.redis.setex(cache_key, 86400 * 7, json.dumps(context))

        return {
            **state,
            "deflection_context": context,
            "processing_status": "completed",
        }

    async def process_tracking_event(
        self,
        tracking_number: str,
        carrier: str,
        raw_status: str,
        event_timestamp: str,
        location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a tracking event through the WISMO pipeline."""
        initial_state: WISMOState = {
            "tracking_number": tracking_number,
            "carrier": carrier,
            "raw_status": raw_status,
            "location": location,
            "event_timestamp": event_timestamp,
            "normalized_event": None,
            "delay_prediction": None,
            "requires_notification": False,
            "notification": None,
            "notification_sent": False,
            "notification_result": None,
            "deflection_context": None,
            "processing_status": "started",
            "errors": [],
        }

        config = {"configurable": {"thread_id": f"wismo_{tracking_number}_{datetime.utcnow().timestamp()}"}}
        result = await self.graph.ainvoke(initial_state, config)

        return {
            "tracking_number": tracking_number,
            "status": result["processing_status"],
            "unified_status": result.get("normalized_event", {}).get("unified_status"),
            "delay_risk": result.get("delay_prediction", {}).get("risk_level"),
            "notification_sent": result.get("notification_sent"),
        }

    def get_deflection_context(self, tracking_number: str) -> Optional[Dict[str, Any]]:
        """Get cached context for ticket deflection."""
        cache_key = f"shipment_context:{tracking_number}"
        data = self.redis.get(cache_key)
        return json.loads(data) if data else None`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the multi-agent WISMO system with Docker, implement webhook endpoints for carrier integration, and set up monitoring for notification delivery and deflection rates.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'FastAPI', 'Kafka'],
            codeSnippets: [
              {
                language: 'python',
                title: 'WISMO API with Webhook Receivers',
                description: 'FastAPI service exposing webhook endpoints for carriers and ticket deflection.',
                code: `# api/wismo_service.py
"""WISMO API service with carrier webhooks and observability."""
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from langsmith import traceable
import structlog
import time
import hmac
import hashlib
import os
from typing import Optional, Dict, Any

from orchestration.wismo_graph import WISMOOrchestrator

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Prometheus metrics
TRACKING_EVENTS = Counter(
    "wismo_tracking_events_total",
    "Total tracking events processed",
    ["carrier", "status"]
)
NOTIFICATIONS_SENT = Counter(
    "wismo_notifications_sent_total",
    "Total notifications sent",
    ["channel", "status"]
)
DEFLECTION_REQUESTS = Counter(
    "wismo_deflection_requests_total",
    "Total deflection requests",
    ["result"]
)
EVENT_PROCESSING_TIME = Histogram(
    "wismo_event_processing_seconds",
    "Event processing duration",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)
ACTIVE_SHIPMENTS = Gauge(
    "wismo_active_shipments",
    "Number of active shipments being tracked"
)

app = FastAPI(
    title="WISMO Multi-Agent API",
    description="Proactive shipment tracking and notification system",
    version="2.0.0"
)

orchestrator = WISMOOrchestrator(
    redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379")
)

CARRIER_SECRETS = {
    "fedex": os.environ.get("FEDEX_WEBHOOK_SECRET", ""),
    "ups": os.environ.get("UPS_WEBHOOK_SECRET", ""),
    "usps": os.environ.get("USPS_WEBHOOK_SECRET", ""),
    "dhl": os.environ.get("DHL_WEBHOOK_SECRET", ""),
}


def verify_webhook_signature(carrier: str, payload: bytes, signature: str) -> bool:
    """Verify carrier webhook signature."""
    secret = CARRIER_SECRETS.get(carrier, "")
    if not secret:
        return False
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


class TrackingEventRequest(BaseModel):
    """Request schema for tracking events."""
    tracking_number: str
    status_code: str
    location: Optional[str] = None
    timestamp: str
    shipment_id: Optional[str] = None


class DeflectionRequest(BaseModel):
    """Request schema for ticket deflection."""
    tracking_number: Optional[str] = None
    order_id: Optional[str] = None
    customer_message: str


class DeflectionResponse(BaseModel):
    """Response schema for ticket deflection."""
    can_deflect: bool
    response_message: str
    confidence: float
    shipment_status: Optional[Dict[str, Any]] = None
    escalate_reason: Optional[str] = None


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "wismo"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@app.post("/webhooks/carrier/{carrier}")
@traceable(name="process_carrier_webhook")
async def receive_carrier_webhook(
    carrier: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Receive carrier tracking webhook and process asynchronously.
    """
    body = await request.body()
    signature = request.headers.get("x-webhook-signature", "")

    if not verify_webhook_signature(carrier, body, signature):
        logger.warning("webhook_signature_invalid", carrier=carrier)
        raise HTTPException(status_code=401, detail="Invalid signature")

    data = await request.json()

    logger.info(
        "webhook_received",
        carrier=carrier,
        tracking_number=data.get("tracking_number"),
        status=data.get("status_code")
    )

    # Process asynchronously
    background_tasks.add_task(
        process_tracking_event,
        carrier=carrier,
        tracking_number=data["tracking_number"],
        raw_status=data["status_code"],
        event_timestamp=data["timestamp"],
        location=data.get("location"),
    )

    return {"status": "accepted", "tracking_number": data["tracking_number"]}


async def process_tracking_event(
    carrier: str,
    tracking_number: str,
    raw_status: str,
    event_timestamp: str,
    location: Optional[str] = None,
):
    """Process tracking event through orchestrator."""
    start_time = time.time()

    try:
        result = await orchestrator.process_tracking_event(
            tracking_number=tracking_number,
            carrier=carrier,
            raw_status=raw_status,
            event_timestamp=event_timestamp,
            location=location,
        )

        duration = time.time() - start_time
        EVENT_PROCESSING_TIME.observe(duration)
        TRACKING_EVENTS.labels(
            carrier=carrier,
            status=result.get("unified_status", "unknown")
        ).inc()

        if result.get("notification_sent"):
            NOTIFICATIONS_SENT.labels(
                channel="auto",
                status="success"
            ).inc()

        logger.info(
            "tracking_event_processed",
            tracking_number=tracking_number,
            duration_ms=round(duration * 1000, 2),
            notification_sent=result.get("notification_sent")
        )

    except Exception as e:
        logger.error(
            "tracking_event_failed",
            tracking_number=tracking_number,
            error=str(e)
        )


@app.post("/api/v2/deflect", response_model=DeflectionResponse)
@traceable(name="deflect_wismo_inquiry")
async def deflect_wismo_inquiry(request: DeflectionRequest):
    """
    Attempt to deflect a WISMO inquiry using cached shipment context.
    Returns a response the chatbot can use, or escalates to human agent.
    """
    from langchain_openai import ChatOpenAI

    # Get shipment context
    context = None
    if request.tracking_number:
        context = orchestrator.get_deflection_context(request.tracking_number)
    elif request.order_id:
        # Look up tracking number from order
        # Implementation depends on your order system
        pass

    if not context:
        DEFLECTION_REQUESTS.labels(result="no_context").inc()
        return DeflectionResponse(
            can_deflect=False,
            response_message="",
            confidence=0.0,
            escalate_reason="Shipment not found in tracking system"
        )

    # Generate response using LLM
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)

    prompt = f"""
    Customer inquiry: "{request.customer_message}"

    Shipment context:
    - Tracking: {context['tracking_number']}
    - Carrier: {context['carrier']}
    - Status: {context['status']}
    - Location: {context['location']}
    - Last update: {context['last_update']}
    - Delay risk: {context['delay_risk']}
    - Predicted delay: {context['predicted_delay']} days
    - New ETA: {context['new_eta']}

    Generate a helpful response. If the customer is very frustrated or the
    situation is complex, recommend escalation. Return JSON:
    {{
        "response": "<helpful message>",
        "confidence": <0.0-1.0>,
        "should_escalate": true/false,
        "escalate_reason": "<if applicable>"
    }}
    """

    response = llm.invoke(prompt)

    try:
        result = json.loads(response.content)
        can_deflect = not result.get("should_escalate", False)

        DEFLECTION_REQUESTS.labels(
            result="deflected" if can_deflect else "escalated"
        ).inc()

        return DeflectionResponse(
            can_deflect=can_deflect,
            response_message=result["response"],
            confidence=result.get("confidence", 0.8),
            shipment_status=context,
            escalate_reason=result.get("escalate_reason")
        )

    except Exception as e:
        logger.error("deflection_generation_failed", error=str(e))
        DEFLECTION_REQUESTS.labels(result="error").inc()

        return DeflectionResponse(
            can_deflect=False,
            response_message="",
            confidence=0.0,
            escalate_reason="Unable to generate response"
        )


@app.get("/api/v2/shipment/{tracking_number}")
async def get_shipment_status(tracking_number: str):
    """Get current shipment status and context."""
    context = orchestrator.get_deflection_context(tracking_number)

    if not context:
        raise HTTPException(status_code=404, detail="Shipment not found")

    return {
        "tracking_number": tracking_number,
        "context": context
    }`,
              },
              {
                language: 'yaml',
                title: 'Docker Compose for WISMO Deployment',
                description: 'Docker Compose configuration for deploying the WISMO multi-agent system.',
                code: `# docker-compose.wismo.yml
version: '3.8'

services:
  wismo-api:
    build:
      context: .
      dockerfile: Dockerfile.wismo
    ports:
      - "8020:8000"
    environment:
      - DATABASE_URL=\${DATABASE_URL}
      - REDIS_URL=redis://redis:6379
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - TWILIO_SID=\${TWILIO_SID}
      - TWILIO_TOKEN=\${TWILIO_TOKEN}
      - TWILIO_FROM=\${TWILIO_FROM}
      - SENDGRID_API_KEY=\${SENDGRID_API_KEY}
      - FROM_EMAIL=\${FROM_EMAIL}
      - FEDEX_WEBHOOK_SECRET=\${FEDEX_WEBHOOK_SECRET}
      - UPS_WEBHOOK_SECRET=\${UPS_WEBHOOK_SECRET}
      - USPS_WEBHOOK_SECRET=\${USPS_WEBHOOK_SECRET}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=wismo-agents
    depends_on:
      - redis
      - kafka
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3

  wismo-consumer:
    build:
      context: .
      dockerfile: Dockerfile.wismo
    command: python -m consumers.tracking_event_consumer
    environment:
      - DATABASE_URL=\${DATABASE_URL}
      - REDIS_URL=redis://redis:6379
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - TWILIO_SID=\${TWILIO_SID}
      - TWILIO_TOKEN=\${TWILIO_TOKEN}
    depends_on:
      - kafka
      - redis
    deploy:
      replicas: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6381:6379"
    volumes:
      - wismo-redis-data:/data
    command: redis-server --appendonly yes

  kafka:
    image: confluentinc/cp-kafka:latest
    ports:
      - "9092:9092"
    environment:
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092
      - KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1
    depends_on:
      - zookeeper

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      - ZOOKEEPER_CLIENT_PORT=2181

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9092:9090"
    volumes:
      - ./prometheus-wismo.yml:/etc/prometheus/prometheus.yml

volumes:
  wismo-redis-data:`,
              },
            ],
          },
        ],
      },
    },
  ],
};
