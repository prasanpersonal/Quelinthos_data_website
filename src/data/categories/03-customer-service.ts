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
    },
  ],
};
