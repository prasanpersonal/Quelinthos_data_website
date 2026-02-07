import type { Category } from '../types.ts';

export const salesCategory: Category = {
  id: 'sales',
  number: 2,
  title: 'Sales Operations & Forecasting',
  shortTitle: 'Sales',
  description:
    'Eliminate pipeline data rot, lead scoring delays, and lost WhatsApp relationships that kill your revenue.',
  icon: 'TrendingUp',
  accentColor: 'neon-purple',
  painPoints: [
    // ── Pain Point 1: Pipeline Data Validity ────────────────────────────
    {
      id: 'pipeline-data-validity',
      number: 1,
      title: 'Pipeline Data Validity Crisis',
      subtitle: 'CRM Data Rot & Forecast Unreliability',
      summary:
        'Your pipeline is 40% phantom deals. Stale opportunities, missing fields, and duplicate contacts make every forecast a guess.',
      tags: ['crm', 'pipeline', 'forecasting'],
      metrics: {
        annualCostRange: '$1M - $5M',
        roi: '8x',
        paybackPeriod: '2-3 months',
        investmentRange: '$100K - $200K',
      },
      price: {
        present: {
          title: 'Present Reality',
          severity: 'critical',
          description:
            'CRM data is decaying faster than your team can maintain it. Reps skip fields, managers distrust dashboards, and forecasts swing wildly week to week.',
          bullets: [
            'Over 40% of pipeline opportunities have not been updated in 30+ days',
            'Duplicate contact records inflate deal counts by 15-25%',
            'Required fields are blank on 35% of open opportunities',
            'Stage progression timestamps are missing or manually backdated',
            'Forecasted close dates are pushed quarter after quarter with no notes',
          ],
        },
        root: {
          title: 'Root Cause',
          severity: 'high',
          description:
            'No data governance layer exists between sales reps and the CRM. Entry is optional, validation is absent, and there is zero accountability for data quality.',
          bullets: [
            'CRM has no field-level validation rules or mandatory workflows',
            'No automated staleness detection flags aging opportunities',
            'Deduplication runs are manual and happen once a quarter at best',
            'Sales managers review pipeline by gut feel, not data health scores',
            'No integration between email/calendar activity and deal progression',
          ],
        },
        impact: {
          title: 'Business Impact',
          severity: 'critical',
          description:
            'Forecast accuracy drops below 60%, causing misallocated resources, missed targets, and eroded board confidence.',
          bullets: [
            'Quarterly forecast variance exceeds 30%, triggering firefighting',
            'Sales leaders spend 8+ hours per week manually scrubbing pipeline',
            'Marketing cannot attribute ROI because lead source data is unreliable',
            'Finance builds revenue models on fundamentally flawed input data',
            'Board loses confidence in growth projections and tightens investment',
          ],
        },
        cost: {
          title: 'Financial Cost',
          severity: 'high',
          description:
            'Revenue leakage from phantom deals, wasted rep time on dead leads, and forecast-driven misallocation compound into millions annually.',
          bullets: [
            '$1M-$3M in revenue slippage from undetected deal decay',
            '$200K-$500K in rep productivity lost to manual data cleanup',
            '$300K-$800K in misallocated sales resources from bad forecasts',
            'Customer acquisition cost inflated 20-30% by duplicate outreach',
            'Opportunity cost of pursuing phantom deals instead of real prospects',
          ],
        },
        expectedReturn: {
          title: 'Expected Return',
          severity: 'high',
          description:
            'A clean, governed pipeline delivers forecast accuracy above 85% and recovers millions in hidden revenue leakage.',
          bullets: [
            'Forecast accuracy improves from 55% to 85%+ within two quarters',
            'Rep time spent on data entry and cleanup drops by 60%',
            'Duplicate contacts eliminated, reducing wasted outreach by 25%',
            'Pipeline reviews shift from data gathering to strategic coaching',
            'Board confidence restored with reliable revenue projections',
          ],
        },
      },
      implementation: {
        overview:
          'Deploy a CRM data quality framework combining SQL-based auditing with Python-powered automated hygiene scoring to eliminate pipeline rot.',
        prerequisites: [
          'Read access to CRM database (Salesforce, HubSpot, or equivalent)',
          'Python 3.9+ with pandas and scikit-learn',
          'Scheduled job runner (Airflow, cron, or CRM-native automation)',
          'pytest >= 7.0 with pytest-asyncio',
          'Docker and docker-compose for containerized deployment',
          'cron, Airflow, or Prefect for job scheduling',
          'Slack incoming webhook URL for operational alerts',
        ],
        toolsUsed: ['SQL', 'Python', 'CRM API', 'Airflow', 'pytest', 'Docker', 'GitHub Actions', 'cron', 'Slack API', 'Prometheus'],
        steps: [
          {
            stepNumber: 1,
            title: 'CRM Data Quality Audit',
            description:
              'Run a comprehensive SQL audit against your CRM to quantify data rot across every pipeline stage. This reveals the exact scope of missing fields, stale records, and duplicate contacts.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Pipeline Staleness & Completeness Audit',
                description:
                  'Identifies stale opportunities, missing required fields, and calculates a per-record health score across the entire pipeline.',
                code: `-- Pipeline Data Quality Audit: Staleness & Completeness
-- Flags opportunities with missing fields, stale updates, and assigns health scores

WITH opportunity_health AS (
  SELECT
    o.id                          AS opportunity_id,
    o.name                        AS deal_name,
    o.owner_id,
    u.full_name                   AS rep_name,
    o.stage_name,
    o.amount,
    o.close_date,
    o.last_modified_date,
    DATEDIFF(day, o.last_modified_date, CURRENT_DATE) AS days_since_update,
    CASE
      WHEN o.amount IS NULL OR o.amount = 0           THEN 1 ELSE 0
    END AS missing_amount,
    CASE WHEN o.close_date IS NULL                     THEN 1 ELSE 0 END AS missing_close_date,
    CASE WHEN o.lead_source IS NULL OR o.lead_source = '' THEN 1 ELSE 0 END AS missing_lead_source,
    CASE WHEN o.next_step IS NULL OR o.next_step = ''  THEN 1 ELSE 0 END AS missing_next_step,
    CASE
      WHEN DATEDIFF(day, o.last_modified_date, CURRENT_DATE) > 30 THEN 'critical'
      WHEN DATEDIFF(day, o.last_modified_date, CURRENT_DATE) > 14 THEN 'warning'
      ELSE 'healthy'
    END AS staleness_status
  FROM opportunities o
  JOIN users u ON o.owner_id = u.id
  WHERE o.is_closed = FALSE
    AND o.is_deleted = FALSE
)
SELECT
  rep_name,
  COUNT(*)                                          AS total_open_deals,
  SUM(CASE WHEN staleness_status = 'critical' THEN 1 ELSE 0 END) AS critical_stale,
  SUM(CASE WHEN staleness_status = 'warning'  THEN 1 ELSE 0 END) AS warning_stale,
  ROUND(AVG(days_since_update), 1)                  AS avg_days_since_update,
  SUM(missing_amount + missing_close_date + missing_lead_source + missing_next_step)
    AS total_missing_fields,
  ROUND(
    100.0 * SUM(CASE WHEN staleness_status = 'critical' THEN 1 ELSE 0 END) / COUNT(*), 1
  ) AS pct_critical
FROM opportunity_health
GROUP BY rep_name
ORDER BY pct_critical DESC;`,
              },
              {
                language: 'sql',
                title: 'Duplicate Contact Detection',
                description:
                  'Finds duplicate contacts using fuzzy matching on name, email domain, and company to quantify inflated pipeline counts.',
                code: `-- Duplicate Contact Detection
-- Identifies probable duplicates by normalized email, name similarity, and company match

WITH normalized_contacts AS (
  SELECT
    c.id,
    c.first_name,
    c.last_name,
    LOWER(TRIM(c.email))                       AS email_clean,
    LOWER(SPLIT_PART(TRIM(c.email), '@', 2))   AS email_domain,
    LOWER(TRIM(c.first_name || ' ' || c.last_name)) AS full_name_lower,
    LOWER(TRIM(c.company))                     AS company_lower,
    c.created_date
  FROM contacts c
  WHERE c.is_deleted = FALSE
),
duplicate_pairs AS (
  SELECT
    a.id          AS contact_a,
    b.id          AS contact_b,
    a.full_name_lower,
    a.email_clean AS email_a,
    b.email_clean AS email_b,
    a.company_lower,
    CASE
      WHEN a.email_clean = b.email_clean                    THEN 'exact_email'
      WHEN a.full_name_lower = b.full_name_lower
        AND a.email_domain = b.email_domain                 THEN 'name_domain_match'
      WHEN a.full_name_lower = b.full_name_lower
        AND a.company_lower = b.company_lower               THEN 'name_company_match'
    END AS match_type
  FROM normalized_contacts a
  JOIN normalized_contacts b
    ON a.id < b.id
    AND (
      a.email_clean = b.email_clean
      OR (a.full_name_lower = b.full_name_lower AND a.email_domain = b.email_domain)
      OR (a.full_name_lower = b.full_name_lower AND a.company_lower = b.company_lower)
    )
)
SELECT
  match_type,
  COUNT(*) AS duplicate_pairs_found,
  COUNT(DISTINCT contact_a) + COUNT(DISTINCT contact_b) AS contacts_involved
FROM duplicate_pairs
GROUP BY match_type
ORDER BY duplicate_pairs_found DESC;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Automated Pipeline Hygiene Scoring',
            description:
              'Build a Python service that continuously scores every opportunity on data completeness, freshness, and engagement signals, then pushes health grades back into the CRM.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Pipeline Health Scoring Engine',
                description:
                  'Calculates a composite health score for each opportunity based on field completeness, update recency, and activity signals.',
                code: `"""
Pipeline Health Scoring Engine
Scores every open opportunity on completeness, freshness, and engagement.
Pushes a letter grade (A-F) back into the CRM for dashboard filtering.
"""

import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass

REQUIRED_FIELDS = [
    "amount", "close_date", "lead_source",
    "next_step", "decision_maker", "stage_name",
]

WEIGHTS = {
    "completeness": 0.35,
    "freshness": 0.35,
    "engagement": 0.30,
}


@dataclass
class HealthScore:
    opportunity_id: str
    completeness: float
    freshness: float
    engagement: float
    composite: float
    grade: str


def score_completeness(row: pd.Series) -> float:
    filled = sum(1 for f in REQUIRED_FIELDS if pd.notna(row.get(f)) and row.get(f) != "")
    return round(filled / len(REQUIRED_FIELDS), 2)


def score_freshness(last_modified: datetime, stage: str) -> float:
    days_stale = (datetime.utcnow() - last_modified).days
    threshold = 7 if stage in ("Negotiation", "Proposal") else 14
    if days_stale <= threshold:
        return 1.0
    elif days_stale <= threshold * 2:
        return 0.6
    elif days_stale <= threshold * 3:
        return 0.3
    return 0.0


def score_engagement(activities: int, emails_opened: int, meetings: int) -> float:
    activity_score = min(activities / 10, 1.0)
    email_score = min(emails_opened / 5, 1.0)
    meeting_score = min(meetings / 2, 1.0)
    return round((activity_score + email_score + meeting_score) / 3, 2)


def compute_grade(composite: float) -> str:
    if composite >= 0.85:
        return "A"
    elif composite >= 0.70:
        return "B"
    elif composite >= 0.50:
        return "C"
    elif composite >= 0.30:
        return "D"
    return "F"


def score_pipeline(opportunities_df: pd.DataFrame) -> list[HealthScore]:
    results = []
    for _, row in opportunities_df.iterrows():
        comp = score_completeness(row)
        fresh = score_freshness(row["last_modified_date"], row["stage_name"])
        eng = score_engagement(
            row.get("activity_count", 0),
            row.get("emails_opened", 0),
            row.get("meetings_held", 0),
        )
        composite = round(
            comp * WEIGHTS["completeness"]
            + fresh * WEIGHTS["freshness"]
            + eng * WEIGHTS["engagement"],
            2,
        )
        results.append(HealthScore(
            opportunity_id=row["id"],
            completeness=comp,
            freshness=fresh,
            engagement=eng,
            composite=composite,
            grade=compute_grade(composite),
        ))
    return results`,
              },
              {
                language: 'python',
                title: 'CRM Writeback & Alerting',
                description:
                  'Pushes health scores back into the CRM and sends Slack alerts for opportunities that drop below grade C.',
                code: `"""
CRM Writeback & Alerting
Syncs health scores to CRM custom fields and alerts reps via Slack
when their deals decay below acceptable thresholds.
"""

import requests
from typing import Any

CRM_API_URL = "https://api.crm.example.com/v2"
SLACK_WEBHOOK = "https://hooks.slack.com/services/T00/B00/xxx"


def push_scores_to_crm(
    scores: list[dict], api_key: str, batch_size: int = 50
) -> dict[str, int]:
    success, failed = 0, 0
    for i in range(0, len(scores), batch_size):
        batch = scores[i : i + batch_size]
        payload = {
            "records": [
                {
                    "id": s["opportunity_id"],
                    "custom_fields": {
                        "pipeline_health_score__c": s["composite"],
                        "pipeline_health_grade__c": s["grade"],
                        "health_scored_at__c": s["scored_at"],
                    },
                }
                for s in batch
            ]
        }
        resp = requests.patch(
            f"{CRM_API_URL}/opportunities/batch",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        if resp.status_code == 200:
            success += len(batch)
        else:
            failed += len(batch)
    return {"success": success, "failed": failed}


def alert_decaying_deals(
    scores: list[dict], threshold_grade: str = "C"
) -> None:
    grade_order = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
    flagged = [
        s for s in scores
        if grade_order.get(s["grade"], 0) < grade_order[threshold_grade]
    ]
    if not flagged:
        return

    rep_groups: dict[str, list] = {}
    for s in flagged:
        rep_groups.setdefault(s["rep_name"], []).append(s)

    for rep, deals in rep_groups.items():
        deal_lines = "\\n".join(
            f"  - {d['deal_name']} (Grade {d['grade']}, "
            f"{d['days_stale']}d stale, \\\${d['amount']:,.0f})"
            for d in deals[:10]
        )
        message = (
            f":warning: *Pipeline Health Alert for {rep}*\\n"
            f"{len(deals)} deal(s) below grade {threshold_grade}:\\n"
            f"{deal_lines}"
        )
        requests.post(
            SLACK_WEBHOOK,
            json={"text": message},
            timeout=10,
        )`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate CRM data quality assertions and pipeline health scoring accuracy with automated SQL checks and pytest-based integration tests before promoting to production.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'CRM Data Quality Assertion Queries',
                description:
                  'Automated data quality gates: row count thresholds, null checks on critical fields, referential integrity between contacts and opportunities, and data freshness validation.',
                code: `-- CRM Data Quality Assertions
-- Run before each pipeline scoring cycle to ensure data integrity

-- 1. Row count sanity: open opportunities should not drop more than 20% overnight
WITH daily_counts AS (
  SELECT
    DATE_TRUNC('day', snapshot_date) AS snap_day,
    COUNT(*)                         AS opp_count
  FROM opportunity_snapshots
  WHERE snapshot_date >= CURRENT_DATE - INTERVAL '2 days'
  GROUP BY DATE_TRUNC('day', snapshot_date)
)
SELECT
  CASE
    WHEN MIN(opp_count) < MAX(opp_count) * 0.8
    THEN 'FAIL: opportunity count dropped >20% day-over-day'
    ELSE 'PASS'
  END AS row_count_check
FROM daily_counts;

-- 2. Null checks on critical pipeline fields
SELECT
  'FAIL: ' || COUNT(*) || ' opportunities missing amount or close_date'
    AS null_check
FROM opportunities
WHERE is_closed = FALSE
  AND is_deleted = FALSE
  AND (amount IS NULL OR close_date IS NULL)
HAVING COUNT(*) > 0
UNION ALL
SELECT 'PASS: no critical nulls found'
WHERE NOT EXISTS (
  SELECT 1 FROM opportunities
  WHERE is_closed = FALSE AND is_deleted = FALSE
    AND (amount IS NULL OR close_date IS NULL)
);

-- 3. Referential integrity: every opportunity must link to a valid owner
SELECT
  CASE
    WHEN COUNT(*) > 0
    THEN 'FAIL: ' || COUNT(*) || ' opportunities reference non-existent users'
    ELSE 'PASS'
  END AS ref_integrity_check
FROM opportunities o
LEFT JOIN users u ON o.owner_id = u.id
WHERE o.is_closed = FALSE AND u.id IS NULL;

-- 4. Freshness: scoring must have run within the last 25 hours
SELECT
  CASE
    WHEN MAX(scored_at) < NOW() - INTERVAL '25 hours'
    THEN 'FAIL: last scoring run is stale (' ||
         EXTRACT(HOUR FROM NOW() - MAX(scored_at)) || 'h ago)'
    ELSE 'PASS: scoring data is fresh'
  END AS freshness_check
FROM opportunities
WHERE pipeline_health_score__c IS NOT NULL;`,
              },
              {
                language: 'python',
                title: 'Pipeline Scoring Validation Tests',
                description:
                  'pytest suite that validates health score calculations, grade boundaries, completeness scoring logic, and CRM writeback integration.',
                code: `"""
Pipeline Health Scoring — Validation Test Suite
Runs as part of CI and pre-deployment gate to catch regressions.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import the scoring engine under test
from pipeline_health_scoring import (
    score_completeness,
    score_freshness,
    score_engagement,
    compute_grade,
    score_pipeline,
    REQUIRED_FIELDS,
    WEIGHTS,
)


@pytest.fixture
def sample_opportunity():
    return pd.Series({
        "id": "opp_001",
        "amount": 50000,
        "close_date": "2025-06-15",
        "lead_source": "Website",
        "next_step": "Demo scheduled",
        "decision_maker": "VP Sales",
        "stage_name": "Proposal",
        "last_modified_date": datetime.utcnow() - timedelta(days=3),
        "activity_count": 8,
        "emails_opened": 4,
        "meetings_held": 2,
    })


@pytest.fixture
def stale_opportunity():
    return pd.Series({
        "id": "opp_002",
        "amount": None,
        "close_date": None,
        "lead_source": "",
        "next_step": "",
        "decision_maker": None,
        "stage_name": "Qualification",
        "last_modified_date": datetime.utcnow() - timedelta(days=45),
        "activity_count": 0,
        "emails_opened": 0,
        "meetings_held": 0,
    })


class TestCompletenessScoring:
    def test_fully_complete_record(self, sample_opportunity):
        score = score_completeness(sample_opportunity)
        assert score == 1.0

    def test_empty_fields_reduce_score(self, stale_opportunity):
        score = score_completeness(stale_opportunity)
        assert score < 0.5

    def test_score_is_bounded(self, sample_opportunity):
        score = score_completeness(sample_opportunity)
        assert 0.0 <= score <= 1.0


class TestFreshnessScoring:
    def test_recently_updated_is_fresh(self):
        recent = datetime.utcnow() - timedelta(days=2)
        assert score_freshness(recent, "Proposal") == 1.0

    def test_stale_proposal_scores_low(self):
        stale = datetime.utcnow() - timedelta(days=25)
        assert score_freshness(stale, "Proposal") <= 0.3

    def test_qualification_has_longer_threshold(self):
        mid_age = datetime.utcnow() - timedelta(days=10)
        assert score_freshness(mid_age, "Qualification") == 1.0


class TestEngagementScoring:
    def test_high_engagement(self):
        score = score_engagement(activities=12, emails_opened=6, meetings=3)
        assert score >= 0.9

    def test_zero_engagement(self):
        score = score_engagement(activities=0, emails_opened=0, meetings=0)
        assert score == 0.0


class TestGradeComputation:
    @pytest.mark.parametrize("composite,expected", [
        (0.90, "A"), (0.85, "A"),
        (0.75, "B"), (0.70, "B"),
        (0.55, "C"), (0.50, "C"),
        (0.35, "D"), (0.30, "D"),
        (0.20, "F"), (0.0, "F"),
    ])
    def test_grade_boundaries(self, composite, expected):
        assert compute_grade(composite) == expected


class TestPipelineScoring:
    def test_scores_all_opportunities(self, sample_opportunity):
        df = pd.DataFrame([sample_opportunity])
        results = score_pipeline(df)
        assert len(results) == 1
        assert results[0].grade in ("A", "B", "C", "D", "F")

    def test_composite_uses_correct_weights(self, sample_opportunity):
        df = pd.DataFrame([sample_opportunity])
        results = score_pipeline(df)
        r = results[0]
        expected = round(
            r.completeness * WEIGHTS["completeness"]
            + r.freshness * WEIGHTS["freshness"]
            + r.engagement * WEIGHTS["engagement"],
            2,
        )
        assert r.composite == expected`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Deploy the pipeline health scoring service with a repeatable bash deployment script, environment-aware configuration, database migrations, and cron-based scheduling for continuous CRM sync.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Pipeline Health Service Deployment Script',
                description:
                  'Idempotent deployment script with environment checks, Python dependency installation, database migration, and cron scheduler setup for the CRM health scoring pipeline.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ── Pipeline Health Scoring Service — Deployment Script ──────────
APP_NAME="pipeline-health-scorer"
APP_DIR="/opt/\${APP_NAME}"
VENV_DIR="\${APP_DIR}/venv"
LOG_DIR="/var/log/\${APP_NAME}"
REQUIRED_VARS=("CRM_API_KEY" "CRM_API_URL" "DATABASE_URL" "SLACK_WEBHOOK_URL")
PYTHON_VERSION="3.11"

echo "==> Deploying \${APP_NAME}..."

# 1. Environment variable checks
for var in "\${REQUIRED_VARS[@]}"; do
  if [[ -z "\${!var:-}" ]]; then
    echo "ERROR: Required env var \${var} is not set." >&2
    exit 1
  fi
done
echo "    Environment variables validated."

# 2. System dependencies
if ! command -v "python\${PYTHON_VERSION}" &>/dev/null; then
  echo "ERROR: python\${PYTHON_VERSION} not found. Install it first." >&2
  exit 1
fi
echo "    Python \${PYTHON_VERSION} found."

# 3. Create app directory and virtualenv
sudo mkdir -p "\${APP_DIR}" "\${LOG_DIR}"
sudo chown "\$(whoami):" "\${APP_DIR}" "\${LOG_DIR}"

if [[ ! -d "\${VENV_DIR}" ]]; then
  "python\${PYTHON_VERSION}" -m venv "\${VENV_DIR}"
  echo "    Virtual environment created."
fi

# 4. Install / upgrade dependencies
"\${VENV_DIR}/bin/pip" install --upgrade pip --quiet
"\${VENV_DIR}/bin/pip" install -r "\${APP_DIR}/requirements.txt" --quiet
echo "    Dependencies installed."

# 5. Run database migration
"\${VENV_DIR}/bin/python" "\${APP_DIR}/migrate.py" --apply
echo "    Database migration applied."

# 6. Register cron schedule: run health scoring every 6 hours
CRON_EXPR="0 */6 * * *"
CRON_CMD="\${VENV_DIR}/bin/python \${APP_DIR}/run_scoring.py >> \${LOG_DIR}/scoring.log 2>&1"
( crontab -l 2>/dev/null | grep -v "\${APP_NAME}" ; echo "\${CRON_EXPR} \${CRON_CMD}  # \${APP_NAME}" ) | crontab -
echo "    Cron job registered: \${CRON_EXPR}"

# 7. Run initial scoring pass
"\${VENV_DIR}/bin/python" "\${APP_DIR}/run_scoring.py"
echo "==> Deployment complete. Logs at \${LOG_DIR}/scoring.log"`,
              },
              {
                language: 'python',
                title: 'Environment-Aware Configuration Loader',
                description:
                  'Loads configuration from environment variables with validation, manages secrets securely, and initializes connection pools for the CRM API and scoring database.',
                code: `"""
Pipeline Health Scorer — Configuration Loader
Reads env vars, validates required secrets, and builds connection pools.
"""

import os
import sys
from dataclasses import dataclass, field
from urllib.parse import urlparse

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool


@dataclass(frozen=True)
class ScorerConfig:
    """Immutable configuration loaded once at startup."""
    crm_api_url: str
    crm_api_key: str
    database_url: str
    slack_webhook_url: str
    scoring_interval_hours: int = 6
    batch_size: int = 50
    stale_threshold_days: int = 30
    log_level: str = "INFO"


def load_config() -> ScorerConfig:
    required = {
        "CRM_API_URL": os.getenv("CRM_API_URL"),
        "CRM_API_KEY": os.getenv("CRM_API_KEY"),
        "DATABASE_URL": os.getenv("DATABASE_URL"),
        "SLACK_WEBHOOK_URL": os.getenv("SLACK_WEBHOOK_URL"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"FATAL: Missing env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    # Basic URL validation
    for key in ("CRM_API_URL", "DATABASE_URL", "SLACK_WEBHOOK_URL"):
        parsed = urlparse(required[key])
        if not parsed.scheme or not parsed.netloc:
            print(f"FATAL: {key} is not a valid URL", file=sys.stderr)
            sys.exit(1)

    return ScorerConfig(
        crm_api_url=required["CRM_API_URL"],
        crm_api_key=required["CRM_API_KEY"],
        database_url=required["DATABASE_URL"],
        slack_webhook_url=required["SLACK_WEBHOOK_URL"],
        scoring_interval_hours=int(os.getenv("SCORING_INTERVAL_HOURS", "6")),
        batch_size=int(os.getenv("BATCH_SIZE", "50")),
        stale_threshold_days=int(os.getenv("STALE_THRESHOLD_DAYS", "30")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


def create_db_pool(config: ScorerConfig) -> sqlalchemy.engine.Engine:
    return create_engine(
        config.database_url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True,
    )


# Module-level singletons
config = load_config()
db_engine = create_db_pool(config)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Track pipeline hygiene trends over time with weekly health dashboards, and send proactive Slack alerts when data quality degrades or scoring freshness lapses.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Weekly Pipeline Health Dashboard Query',
                description:
                  'Powers a weekly data quality dashboard showing health trends by team, stage, and time period.',
                code: `-- Weekly Pipeline Health Trend Report
-- Tracks data quality metrics over time for executive dashboards

WITH weekly_snapshots AS (
  SELECT
    DATE_TRUNC('week', scored_at)           AS week_start,
    team_name,
    COUNT(*)                                AS total_opportunities,
    ROUND(AVG(pipeline_health_score__c), 2) AS avg_health_score,
    SUM(CASE WHEN pipeline_health_grade__c = 'A' THEN 1 ELSE 0 END) AS grade_a,
    SUM(CASE WHEN pipeline_health_grade__c = 'B' THEN 1 ELSE 0 END) AS grade_b,
    SUM(CASE WHEN pipeline_health_grade__c IN ('D','F') THEN 1 ELSE 0 END) AS at_risk,
    SUM(CASE WHEN pipeline_health_grade__c IN ('D','F') THEN amount ELSE 0 END) AS at_risk_revenue,
    SUM(amount) AS total_pipeline_value
  FROM opportunities o
  JOIN users u ON o.owner_id = u.id
  JOIN teams t ON u.team_id = t.id
  WHERE o.is_closed = FALSE
    AND scored_at >= CURRENT_DATE - INTERVAL '12 weeks'
  GROUP BY DATE_TRUNC('week', scored_at), team_name
)
SELECT
  week_start,
  team_name,
  total_opportunities,
  avg_health_score,
  grade_a + grade_b                                       AS healthy_deals,
  at_risk,
  ROUND(100.0 * at_risk / NULLIF(total_opportunities, 0), 1) AS pct_at_risk,
  at_risk_revenue,
  total_pipeline_value,
  ROUND(100.0 * at_risk_revenue / NULLIF(total_pipeline_value, 0), 1)
    AS pct_revenue_at_risk
FROM weekly_snapshots
ORDER BY week_start DESC, team_name;`,
              },
              {
                language: 'python',
                title: 'Pipeline Health Slack Alerting',
                description:
                  'Monitors pipeline health metrics against configurable thresholds and sends Slack alerts when data quality degrades, scoring becomes stale, or at-risk revenue exceeds limits.',
                code: `"""
Pipeline Health — Slack Alerting Service
Checks key health thresholds and fires Slack notifications when
pipeline data quality drops below acceptable levels.
"""

import os
import requests
from datetime import datetime
from dataclasses import dataclass
from sqlalchemy import create_engine, text

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
engine = create_engine(DATABASE_URL)


@dataclass
class AlertThresholds:
    max_pct_at_risk: float = 25.0
    max_scoring_age_hours: int = 25
    max_stale_critical_pct: float = 20.0
    min_avg_health_score: float = 0.5


def check_pipeline_health(thresholds: AlertThresholds) -> list[dict]:
    alerts: list[dict] = []
    with engine.connect() as conn:
        # Check scoring freshness
        row = conn.execute(text(
            "SELECT MAX(scored_at) AS last_scored FROM opportunities "
            "WHERE pipeline_health_score__c IS NOT NULL"
        )).fetchone()
        if row and row.last_scored:
            age_hours = (datetime.utcnow() - row.last_scored).total_seconds() / 3600
            if age_hours > thresholds.max_scoring_age_hours:
                alerts.append({
                    "type": "staleness",
                    "message": f"Scoring data is {age_hours:.1f}h old "
                               f"(threshold: {thresholds.max_scoring_age_hours}h)",
                })

        # Check at-risk percentage
        stats = conn.execute(text(\"\"\"
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN pipeline_health_grade__c IN ('D','F')
                  THEN 1 ELSE 0 END) AS at_risk,
              ROUND(AVG(pipeline_health_score__c), 3) AS avg_score
            FROM opportunities WHERE is_closed = FALSE
        \"\"\")).fetchone()
        if stats and stats.total > 0:
            pct_risk = 100.0 * stats.at_risk / stats.total
            if pct_risk > thresholds.max_pct_at_risk:
                alerts.append({
                    "type": "quality_degradation",
                    "message": f"{pct_risk:.1f}% of pipeline is at-risk "
                               f"(threshold: {thresholds.max_pct_at_risk}%)",
                })
            if stats.avg_score < thresholds.min_avg_health_score:
                alerts.append({
                    "type": "low_avg_score",
                    "message": f"Avg health score {stats.avg_score:.3f} "
                               f"below minimum {thresholds.min_avg_health_score}",
                })
    return alerts


def send_slack_alerts(alerts: list[dict]) -> None:
    if not alerts or not SLACK_WEBHOOK:
        return
    severity = "critical" if len(alerts) >= 2 else "warning"
    icon = ":rotating_light:" if severity == "critical" else ":warning:"
    lines = [f"  - [{a['type']}] {a['message']}" for a in alerts]
    payload = {
        "text": (
            f"{icon} *Pipeline Health Alert* ({severity})\\n"
            + "\\n".join(lines)
            + f"\\nChecked at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        ),
    }
    requests.post(SLACK_WEBHOOK, json=payload, timeout=10)


if __name__ == "__main__":
    thresholds = AlertThresholds()
    alerts = check_pipeline_health(thresholds)
    if alerts:
        print(f"Found {len(alerts)} alert(s), sending to Slack...")
        send_slack_alerts(alerts)
    else:
        print("All pipeline health checks passed.")`,
              },
            ],
          },
        ],
      },
      aiEasyWin: {
        overview:
          'Use ChatGPT/Claude with Zapier to automatically analyze CRM data exports, identify stale deals and data quality issues, and send enrichment suggestions without writing custom code.',
        estimatedMonthlyCost: '$100 - $180/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Clay ($149/mo for enrichment)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Apollo AI ($49/mo)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Set up automated CRM data exports via Zapier that extract pipeline opportunities with key fields for AI analysis. Configure scheduled triggers to pull fresh data daily.',
            toolsUsed: ['Zapier', 'CRM (Salesforce/HubSpot)', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier CRM Export Trigger Configuration',
                description:
                  'Configures a Zapier trigger to export open opportunities with staleness indicators to Google Sheets for AI analysis.',
                code: `{
  "trigger": {
    "app": "salesforce",
    "event": "find_records_search",
    "config": {
      "object": "Opportunity",
      "search_criteria": {
        "StageName": { "not_equals": "Closed Won" },
        "StageName": { "not_equals": "Closed Lost" },
        "IsClosed": false
      },
      "fields": [
        "Id", "Name", "StageName", "Amount", "CloseDate",
        "LastModifiedDate", "OwnerId", "Owner.Name",
        "LeadSource", "NextStep", "Description"
      ],
      "schedule": {
        "type": "daily",
        "time": "06:00",
        "timezone": "America/New_York"
      }
    }
  },
  "actions": [
    {
      "app": "google_sheets",
      "event": "create_spreadsheet_row",
      "config": {
        "spreadsheet_id": "{{env.PIPELINE_SHEET_ID}}",
        "worksheet": "Raw_Pipeline_Data",
        "row_data": {
          "Opportunity_ID": "{{trigger.Id}}",
          "Deal_Name": "{{trigger.Name}}",
          "Stage": "{{trigger.StageName}}",
          "Amount": "{{trigger.Amount}}",
          "Close_Date": "{{trigger.CloseDate}}",
          "Last_Modified": "{{trigger.LastModifiedDate}}",
          "Owner": "{{trigger.Owner.Name}}",
          "Lead_Source": "{{trigger.LeadSource}}",
          "Next_Step": "{{trigger.NextStep}}",
          "Days_Since_Update": "={{DATEDIF(trigger.LastModifiedDate, TODAY(), \\"D\\")}}"
        }
      }
    }
  ]
}`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description:
              'Use ChatGPT or Claude to analyze the exported pipeline data, identify stale deals, missing fields, and data quality issues. Generate prioritized action items for each sales rep.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro', 'Zapier AI Actions'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Pipeline Health Analysis Prompt Template',
                description:
                  'Structured prompt for AI to analyze CRM data quality and generate actionable insights for sales managers.',
                code: `system_prompt: |
  You are a Sales Operations AI analyst specializing in CRM data quality
  and pipeline health assessment. Analyze the provided pipeline data and
  identify issues that require immediate attention.

user_prompt_template: |
  ## Pipeline Data Quality Analysis Request

  **Analysis Date:** {{current_date}}
  **Sales Team:** {{team_name}}

  ### Pipeline Data (CSV format):
  \`\`\`csv
  {{pipeline_data_csv}}
  \`\`\`

  ### Analysis Instructions:

  1. **Staleness Detection:**
     - Flag deals not updated in 14+ days as "Warning"
     - Flag deals not updated in 30+ days as "Critical"
     - Consider stage context (Negotiation stage should update weekly)

  2. **Data Completeness Check:**
     - Identify deals missing: Amount, Close Date, Next Step, Lead Source
     - Calculate completeness score per deal (0-100%)

  3. **Anomaly Detection:**
     - Close dates in the past but deal still open
     - Deals stuck in same stage for 60+ days
     - Unusually high or low amounts for the stage

  4. **Owner Analysis:**
     - Summarize data quality score per rep
     - Identify reps with highest % of stale deals

  ### Required Output Format:

  \`\`\`json
  {
    "summary": {
      "total_deals_analyzed": <number>,
      "critical_issues": <number>,
      "warning_issues": <number>,
      "overall_health_score": "<A/B/C/D/F>",
      "total_pipeline_at_risk": "<dollar amount>"
    },
    "critical_deals": [
      {
        "deal_name": "<name>",
        "owner": "<rep name>",
        "issue": "<description>",
        "days_stale": <number>,
        "amount": "<dollar amount>",
        "recommended_action": "<specific action>"
      }
    ],
    "rep_scorecards": [
      {
        "rep_name": "<name>",
        "total_deals": <number>,
        "health_score": "<A-F>",
        "stale_deals": <number>,
        "missing_fields_count": <number>,
        "priority_actions": ["<action 1>", "<action 2>"]
      }
    ],
    "enrichment_suggestions": [
      {
        "deal_name": "<name>",
        "missing_data": ["<field1>", "<field2>"],
        "suggested_sources": ["Clay", "Apollo", "LinkedIn"]
      }
    ]
  }
  \`\`\`

output_instructions: |
  - Be specific with deal names and owner names
  - Prioritize by revenue impact (Amount * probability of loss)
  - Include actionable next steps, not just observations
  - Keep recommended actions to 1-2 sentences each`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier to automatically run the AI analysis, parse results, and distribute personalized reports to sales reps and managers via Slack or email.',
            toolsUsed: ['Zapier', 'Slack', 'Gmail', 'Clay'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier AI Analysis and Distribution Workflow',
                description:
                  'Complete Zapier workflow that triggers AI analysis, parses results, and sends personalized alerts to sales team members.',
                code: `{
  "workflow_name": "Daily Pipeline Health AI Analysis",
  "trigger": {
    "type": "schedule",
    "config": {
      "frequency": "daily",
      "time": "07:00",
      "timezone": "America/New_York"
    }
  },
  "steps": [
    {
      "step_number": 1,
      "app": "google_sheets",
      "action": "get_all_rows",
      "config": {
        "spreadsheet_id": "{{env.PIPELINE_SHEET_ID}}",
        "worksheet": "Raw_Pipeline_Data",
        "output_format": "csv"
      }
    },
    {
      "step_number": 2,
      "app": "chatgpt",
      "action": "conversation",
      "config": {
        "model": "gpt-4",
        "system_message": "{{prompts.pipeline_analysis.system}}",
        "user_message": "{{prompts.pipeline_analysis.user | replace('{{pipeline_data_csv}}', step1.csv_data)}}",
        "max_tokens": 4000,
        "temperature": 0.3
      }
    },
    {
      "step_number": 3,
      "app": "code",
      "action": "run_javascript",
      "config": {
        "code": "const result = JSON.parse(inputData.ai_response); return { summary: result.summary, critical_deals: result.critical_deals, rep_scorecards: result.rep_scorecards };"
      }
    },
    {
      "step_number": 4,
      "app": "slack",
      "action": "send_channel_message",
      "config": {
        "channel": "#sales-ops-alerts",
        "message_blocks": [
          {
            "type": "header",
            "text": ":chart_with_upwards_trend: Daily Pipeline Health Report"
          },
          {
            "type": "section",
            "text": "*Overall Health: {{step3.summary.overall_health_score}}*\\n:warning: {{step3.summary.critical_issues}} critical issues | :eyes: {{step3.summary.warning_issues}} warnings\\n:moneybag: Pipeline at risk: {{step3.summary.total_pipeline_at_risk}}"
          },
          {
            "type": "divider"
          },
          {
            "type": "section",
            "text": "*Critical Deals Requiring Immediate Action:*\\n{{#each step3.critical_deals}}• *{{deal_name}}* ({{owner}}): {{issue}} - {{recommended_action}}\\n{{/each}}"
          }
        ]
      }
    },
    {
      "step_number": 5,
      "app": "loop",
      "action": "for_each",
      "config": {
        "items": "{{step3.rep_scorecards}}",
        "sub_steps": [
          {
            "app": "slack",
            "action": "send_direct_message",
            "config": {
              "user_email": "{{loop.item.rep_email}}",
              "message": ":wave: Hi {{loop.item.rep_name}}, your daily pipeline health score is *{{loop.item.health_score}}*.\\n\\n*Priority Actions:*\\n{{#each loop.item.priority_actions}}• {{this}}\\n{{/each}}\\n\\nYou have {{loop.item.stale_deals}} stale deals that need updates today."
            }
          }
        ]
      }
    },
    {
      "step_number": 6,
      "app": "clay",
      "action": "enrich_records",
      "condition": "{{step3.enrichment_suggestions.length > 0}}",
      "config": {
        "table_id": "{{env.CLAY_ENRICHMENT_TABLE}}",
        "records": "{{step3.enrichment_suggestions}}",
        "enrichment_sources": ["linkedin", "apollo", "clearbit"]
      }
    }
  ]
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent system where specialized AI agents continuously monitor CRM data quality, auto-enrich stale records, detect anomalies, and proactively alert sales managers with prescriptive actions.',
        estimatedMonthlyCost: '$600 - $1,200/month',
        architecture:
          'A Supervisor agent orchestrates four specialist agents: DataAuditor scans for quality issues, EnrichmentAgent fills gaps using external APIs, AnomalyDetector identifies unusual patterns, and AlertDispatcher routes actionable insights to the right stakeholders.',
        agents: [
          {
            name: 'CRMDataAuditorAgent',
            role: 'Data Quality Auditor',
            goal: 'Continuously scan CRM records for staleness, missing fields, duplicates, and data inconsistencies.',
            tools: ['CRM API', 'SQL Database', 'Pandas'],
          },
          {
            name: 'DataEnrichmentAgent',
            role: 'Record Enrichment Specialist',
            goal: 'Automatically enrich incomplete records using external data sources like Clay, Apollo, and LinkedIn.',
            tools: ['Clay API', 'Apollo API', 'LinkedIn Sales Navigator', 'Clearbit'],
          },
          {
            name: 'AnomalyDetectorAgent',
            role: 'Pipeline Anomaly Detector',
            goal: 'Identify unusual patterns like stuck deals, outlier amounts, and suspicious stage progressions.',
            tools: ['Scikit-learn', 'Pandas', 'Statistical Models'],
          },
          {
            name: 'AlertDispatcherAgent',
            role: 'Notification Router',
            goal: 'Route prioritized alerts to the right people via Slack, email, or CRM tasks based on severity and ownership.',
            tools: ['Slack API', 'Email Service', 'CRM Task API'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with hourly checkpointing and 30-day audit trail',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system with CrewAI, establishing clear roles, goals, and tool access for each specialist agent in the pipeline health monitoring crew.',
            toolsUsed: ['CrewAI', 'LangChain', 'Python'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Pipeline Health Agent Crew Definition',
                description:
                  'CrewAI configuration defining the four specialist agents with their roles, goals, backstories, and tool assignments.',
                code: `"""
Pipeline Health Multi-Agent System
CrewAI-based agent definitions for CRM data quality monitoring.
"""

from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import os

# Initialize LLM with appropriate model
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
)


class PipelineHealthAgents:
    """Factory class for creating pipeline health monitoring agents."""

    @staticmethod
    def create_data_auditor_agent(tools: List[Any]) -> Agent:
        return Agent(
            role="CRM Data Quality Auditor",
            goal="""Systematically audit CRM pipeline data to identify:
                1. Stale opportunities (no updates in 14+ days)
                2. Missing required fields (Amount, Close Date, Next Step)
                3. Duplicate contacts and opportunities
                4. Data inconsistencies (past close dates, invalid stages)""",
            backstory="""You are a meticulous data quality specialist with 10 years
                of experience in sales operations. You've seen how bad data destroys
                forecasts and wastes rep time. You catch issues others miss and
                quantify the revenue impact of every data problem you find.""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=15,
        )

    @staticmethod
    def create_enrichment_agent(tools: List[Any]) -> Agent:
        return Agent(
            role="Data Enrichment Specialist",
            goal="""Automatically enrich incomplete CRM records by:
                1. Finding missing contact information via Apollo/Clay
                2. Updating company firmographic data
                3. Identifying decision-makers for accounts lacking them
                4. Validating and correcting outdated information""",
            backstory="""You are a research expert who can find information on any
                company or contact. You know which enrichment sources are most
                reliable for different data types and always verify before updating.
                You prioritize high-value opportunities for enrichment.""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=20,
        )

    @staticmethod
    def create_anomaly_detector_agent(tools: List[Any]) -> Agent:
        return Agent(
            role="Pipeline Anomaly Detector",
            goal="""Detect unusual patterns in pipeline data including:
                1. Deals stuck in same stage beyond normal duration
                2. Unusual amount values for deal stage or segment
                3. Sudden changes in close dates or deal values
                4. Reps with abnormal win/loss patterns""",
            backstory="""You are a data scientist specializing in anomaly detection.
                You've built models that caught millions in revenue leakage from
                phantom deals and gaming behaviors. You balance statistical rigor
                with business context to avoid false positives.""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=10,
        )

    @staticmethod
    def create_alert_dispatcher_agent(tools: List[Any]) -> Agent:
        return Agent(
            role="Alert Routing Specialist",
            goal="""Route pipeline health alerts to the right stakeholders:
                1. Critical issues to sales managers immediately via Slack
                2. Rep-specific issues as personalized daily digests
                3. Enrichment completions back to opportunity owners
                4. Weekly summaries to sales leadership""",
            backstory="""You are a communications expert who knows that the right
                message to the wrong person is noise. You craft concise, actionable
                alerts that drive behavior change. You never cry wolf and always
                include the 'so what' and 'now what'.""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=10,
        )


def create_pipeline_health_crew(
    auditor_tools: List[Any],
    enrichment_tools: List[Any],
    anomaly_tools: List[Any],
    alert_tools: List[Any],
) -> Crew:
    """Create the full pipeline health monitoring crew."""
    agents_factory = PipelineHealthAgents()

    data_auditor = agents_factory.create_data_auditor_agent(auditor_tools)
    enrichment_agent = agents_factory.create_enrichment_agent(enrichment_tools)
    anomaly_detector = agents_factory.create_anomaly_detector_agent(anomaly_tools)
    alert_dispatcher = agents_factory.create_alert_dispatcher_agent(alert_tools)

    return Crew(
        agents=[data_auditor, enrichment_agent, anomaly_detector, alert_dispatcher],
        process=Process.sequential,
        verbose=True,
        memory=True,
        cache=True,
        max_rpm=30,
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the CRM Data Auditor agent with tools to connect to Salesforce/HubSpot APIs, query for pipeline data, and calculate health metrics for each opportunity.',
            toolsUsed: ['LangChain Tools', 'Salesforce API', 'HubSpot API', 'Pandas'],
            codeSnippets: [
              {
                language: 'python',
                title: 'CRM Data Auditor Agent Tools',
                description:
                  'LangChain tool implementations for the Data Auditor agent to query CRM data and assess pipeline health.',
                code: `"""
CRM Data Auditor Agent — Tool Implementations
Tools for querying CRM data and calculating health metrics.
"""

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import httpx
import os


class CRMQueryInput(BaseModel):
    query_type: str = Field(
        description="Type of query: 'stale_deals', 'missing_fields', 'duplicates', 'all_open'"
    )
    days_threshold: int = Field(
        default=14,
        description="Number of days to consider a deal stale",
    )
    owner_filter: Optional[str] = Field(
        default=None,
        description="Filter by opportunity owner email",
    )


class CRMQueryTool(BaseTool):
    name: str = "crm_query"
    description: str = """Query CRM for pipeline opportunities with health indicators.
        Use query_type='stale_deals' to find opportunities not updated recently.
        Use query_type='missing_fields' to find opportunities with incomplete data.
        Use query_type='duplicates' to find potential duplicate records.
        Use query_type='all_open' to get all open opportunities."""
    args_schema: type[BaseModel] = CRMQueryInput

    def _run(
        self,
        query_type: str,
        days_threshold: int = 14,
        owner_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        crm_api_key = os.getenv("CRM_API_KEY")
        crm_api_url = os.getenv("CRM_API_URL", "https://api.salesforce.com")

        # Build query based on type
        base_query = {
            "object": "Opportunity",
            "fields": [
                "Id", "Name", "StageName", "Amount", "CloseDate",
                "LastModifiedDate", "OwnerId", "Owner.Name", "Owner.Email",
                "LeadSource", "NextStep", "AccountId", "Account.Name",
                "CreatedDate", "Probability",
            ],
            "filters": {"IsClosed": False},
        }

        if owner_filter:
            base_query["filters"]["Owner.Email"] = owner_filter

        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{crm_api_url}/services/data/v58.0/query",
                headers={"Authorization": f"Bearer {crm_api_key}"},
                json=base_query,
            )
            response.raise_for_status()
            raw_data = response.json()["records"]

        df = pd.DataFrame(raw_data)
        df["LastModifiedDate"] = pd.to_datetime(df["LastModifiedDate"])
        df["DaysSinceUpdate"] = (
            datetime.utcnow() - df["LastModifiedDate"]
        ).dt.days

        if query_type == "stale_deals":
            stale = df[df["DaysSinceUpdate"] >= days_threshold]
            return {
                "query_type": "stale_deals",
                "threshold_days": days_threshold,
                "total_found": len(stale),
                "total_value_at_risk": float(stale["Amount"].sum()),
                "deals": stale[[
                    "Id", "Name", "StageName", "Amount", "DaysSinceUpdate",
                    "Owner.Name", "Account.Name",
                ]].to_dict(orient="records"),
            }

        elif query_type == "missing_fields":
            required_fields = ["Amount", "CloseDate", "NextStep", "LeadSource"]
            missing_mask = df[required_fields].isna().any(axis=1)
            incomplete = df[missing_mask].copy()
            incomplete["MissingFields"] = df[required_fields].isna().apply(
                lambda row: [f for f in required_fields if row[f]], axis=1
            )
            return {
                "query_type": "missing_fields",
                "total_found": len(incomplete),
                "deals": incomplete[[
                    "Id", "Name", "StageName", "Amount", "Owner.Name", "MissingFields",
                ]].to_dict(orient="records"),
            }

        elif query_type == "duplicates":
            # Simple duplicate detection by account + similar name
            df["NameNormalized"] = df["Name"].str.lower().str.strip()
            duplicates = df[df.duplicated(
                subset=["AccountId", "NameNormalized"], keep=False
            )]
            return {
                "query_type": "duplicates",
                "total_found": len(duplicates),
                "duplicate_groups": duplicates.groupby(
                    ["AccountId", "NameNormalized"]
                ).apply(lambda g: g[["Id", "Name", "Amount"]].to_dict(orient="records"))
                .to_dict(),
            }

        else:  # all_open
            df["HealthScore"] = self._calculate_health_score(df)
            return {
                "query_type": "all_open",
                "total_deals": len(df),
                "total_pipeline_value": float(df["Amount"].sum()),
                "avg_health_score": float(df["HealthScore"].mean()),
                "deals": df[[
                    "Id", "Name", "StageName", "Amount", "DaysSinceUpdate",
                    "HealthScore", "Owner.Name",
                ]].to_dict(orient="records"),
            }

    def _calculate_health_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate a 0-100 health score for each opportunity."""
        scores = pd.Series(100.0, index=df.index)

        # Staleness penalty
        scores -= df["DaysSinceUpdate"].clip(0, 60) * 1.5

        # Missing fields penalty
        required = ["Amount", "CloseDate", "NextStep", "LeadSource"]
        missing_count = df[required].isna().sum(axis=1)
        scores -= missing_count * 10

        # Past close date penalty
        df["CloseDate"] = pd.to_datetime(df["CloseDate"])
        past_due = df["CloseDate"] < datetime.utcnow()
        scores[past_due] -= 25

        return scores.clip(0, 100)`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Anomaly Detector and Enrichment agents that analyze pipeline patterns, identify outliers, and automatically enrich records using external data sources.',
            toolsUsed: ['LangChain Tools', 'Clay API', 'Apollo API', 'Scikit-learn'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Anomaly Detection and Enrichment Agent Tools',
                description:
                  'Tool implementations for detecting pipeline anomalies and enriching incomplete CRM records.',
                code: `"""
Pipeline Analysis Agents — Anomaly Detection & Data Enrichment Tools
"""

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import httpx
import os


class AnomalyDetectionInput(BaseModel):
    pipeline_data: List[Dict[str, Any]] = Field(
        description="List of opportunity records to analyze for anomalies"
    )
    sensitivity: str = Field(
        default="medium",
        description="Anomaly sensitivity: 'low', 'medium', 'high'",
    )


class PipelineAnomalyDetectorTool(BaseTool):
    name: str = "detect_pipeline_anomalies"
    description: str = """Analyze pipeline data for anomalies including:
        - Deals stuck in stage beyond normal duration
        - Unusual deal amounts for the stage
        - Abnormal win/loss patterns by rep
        - Sudden changes in deal values or dates"""
    args_schema: type[BaseModel] = AnomalyDetectionInput

    def _run(
        self,
        pipeline_data: List[Dict[str, Any]],
        sensitivity: str = "medium",
    ) -> Dict[str, Any]:
        df = pd.DataFrame(pipeline_data)
        df["LastModifiedDate"] = pd.to_datetime(df["LastModifiedDate"])
        df["CreatedDate"] = pd.to_datetime(df["CreatedDate"])
        df["DaysInStage"] = (datetime.utcnow() - df["LastModifiedDate"]).dt.days

        threshold_multiplier = {"low": 2.5, "medium": 2.0, "high": 1.5}[sensitivity]
        anomalies = []

        # 1. Stuck deals: days in stage > stage-specific threshold
        stage_thresholds = {
            "Qualification": 14, "Discovery": 21, "Proposal": 14,
            "Negotiation": 7, "Closed Won": 0, "Closed Lost": 0,
        }
        for stage, threshold in stage_thresholds.items():
            stuck = df[
                (df["StageName"] == stage) &
                (df["DaysInStage"] > threshold * threshold_multiplier)
            ]
            for _, row in stuck.iterrows():
                anomalies.append({
                    "type": "stuck_deal",
                    "severity": "high" if row["DaysInStage"] > threshold * 3 else "medium",
                    "opportunity_id": row["Id"],
                    "deal_name": row["Name"],
                    "owner": row.get("Owner.Name", "Unknown"),
                    "details": f"Stuck in {stage} for {row['DaysInStage']} days (threshold: {threshold})",
                    "amount_at_risk": float(row.get("Amount", 0) or 0),
                })

        # 2. Amount outliers by stage using IQR method
        for stage in df["StageName"].unique():
            stage_df = df[df["StageName"] == stage]
            if len(stage_df) < 5:
                continue
            q1, q3 = stage_df["Amount"].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            outliers = stage_df[
                (stage_df["Amount"] < lower_bound) |
                (stage_df["Amount"] > upper_bound)
            ]
            for _, row in outliers.iterrows():
                direction = "unusually_high" if row["Amount"] > upper_bound else "unusually_low"
                anomalies.append({
                    "type": "amount_outlier",
                    "severity": "medium",
                    "opportunity_id": row["Id"],
                    "deal_name": row["Name"],
                    "owner": row.get("Owner.Name", "Unknown"),
                    "details": f"Amount \${row['Amount']:,.0f} is {direction} for {stage} stage (range: \${lower_bound:,.0f}-\${upper_bound:,.0f})",
                    "amount_at_risk": float(row.get("Amount", 0) or 0),
                })

        # 3. Past close date anomaly
        df["CloseDate"] = pd.to_datetime(df["CloseDate"])
        past_due = df[df["CloseDate"] < datetime.utcnow()]
        for _, row in past_due.iterrows():
            days_past = (datetime.utcnow() - row["CloseDate"]).days
            anomalies.append({
                "type": "past_close_date",
                "severity": "high" if days_past > 30 else "medium",
                "opportunity_id": row["Id"],
                "deal_name": row["Name"],
                "owner": row.get("Owner.Name", "Unknown"),
                "details": f"Close date {row['CloseDate'].strftime('%Y-%m-%d')} is {days_past} days in the past",
                "amount_at_risk": float(row.get("Amount", 0) or 0),
            })

        total_at_risk = sum(a["amount_at_risk"] for a in anomalies)

        return {
            "total_anomalies": len(anomalies),
            "total_value_at_risk": total_at_risk,
            "anomalies_by_type": pd.DataFrame(anomalies).groupby("type").size().to_dict() if anomalies else {},
            "anomalies": sorted(anomalies, key=lambda x: x["amount_at_risk"], reverse=True),
        }


class EnrichmentInput(BaseModel):
    records_to_enrich: List[Dict[str, Any]] = Field(
        description="List of records with missing data to enrich"
    )
    enrichment_fields: List[str] = Field(
        default=["company_size", "industry", "decision_maker", "email", "phone"],
        description="Fields to attempt to enrich",
    )


class DataEnrichmentTool(BaseTool):
    name: str = "enrich_crm_records"
    description: str = """Enrich incomplete CRM records using external data sources.
        Attempts to fill missing company firmographics, contact details, and decision-maker info."""
    args_schema: type[BaseModel] = EnrichmentInput

    def _run(
        self,
        records_to_enrich: List[Dict[str, Any]],
        enrichment_fields: List[str] = None,
    ) -> Dict[str, Any]:
        enrichment_fields = enrichment_fields or [
            "company_size", "industry", "decision_maker", "email", "phone"
        ]
        clay_api_key = os.getenv("CLAY_API_KEY")
        clay_api_url = os.getenv("CLAY_API_URL", "https://api.clay.com/v1")

        enriched_records = []
        failed_records = []

        for record in records_to_enrich:
            try:
                with httpx.Client(timeout=30) as client:
                    # Query Clay for company enrichment
                    company_name = record.get("Account.Name") or record.get("company_name")
                    if not company_name:
                        failed_records.append({
                            "record_id": record.get("Id"),
                            "reason": "No company name available for enrichment",
                        })
                        continue

                    response = client.post(
                        f"{clay_api_url}/enrich/company",
                        headers={"Authorization": f"Bearer {clay_api_key}"},
                        json={
                            "company_name": company_name,
                            "domain": record.get("website"),
                            "requested_fields": enrichment_fields,
                        },
                    )

                    if response.status_code == 200:
                        enrichment_data = response.json()
                        enriched_records.append({
                            "record_id": record.get("Id"),
                            "deal_name": record.get("Name"),
                            "original_data": record,
                            "enriched_data": enrichment_data,
                            "fields_enriched": [
                                f for f in enrichment_fields
                                if enrichment_data.get(f) and not record.get(f)
                            ],
                        })
                    else:
                        failed_records.append({
                            "record_id": record.get("Id"),
                            "reason": f"API error: {response.status_code}",
                        })

            except Exception as e:
                failed_records.append({
                    "record_id": record.get("Id"),
                    "reason": str(e),
                })

        return {
            "total_processed": len(records_to_enrich),
            "successfully_enriched": len(enriched_records),
            "failed": len(failed_records),
            "enriched_records": enriched_records,
            "failed_records": failed_records,
        }`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement LangGraph state machine to orchestrate the multi-agent workflow with conditional routing, error handling, and state persistence.',
            toolsUsed: ['LangGraph', 'Redis', 'Python asyncio'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Pipeline Health Orchestrator',
                description:
                  'State machine implementation that orchestrates the pipeline health agents with conditional logic and Redis-backed persistence.',
                code: `"""
Pipeline Health Monitoring — LangGraph Orchestration
Supervisor pattern with conditional routing and state persistence.
"""

from typing import TypedDict, Annotated, Sequence, Literal
from datetime import datetime
import operator
import json
import redis

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class PipelineHealthState(TypedDict):
    """State schema for the pipeline health monitoring workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    pipeline_data: dict
    audit_results: dict
    anomalies: list
    enrichment_results: dict
    alerts_to_send: list
    workflow_status: str
    error_log: list
    run_timestamp: str


class RedisCheckpointer(BaseCheckpointSaver):
    """Redis-backed state persistence for workflow checkpointing."""

    def __init__(self, redis_url: str, ttl_days: int = 30):
        self.client = redis.from_url(redis_url)
        self.ttl_seconds = ttl_days * 24 * 60 * 60

    def get(self, thread_id: str) -> dict | None:
        data = self.client.get(f"pipeline_health:{thread_id}")
        return json.loads(data) if data else None

    def put(self, thread_id: str, state: dict) -> None:
        self.client.setex(
            f"pipeline_health:{thread_id}",
            self.ttl_seconds,
            json.dumps(state, default=str),
        )


def create_pipeline_health_graph(
    auditor_agent,
    anomaly_agent,
    enrichment_agent,
    alert_agent,
    redis_url: str,
) -> StateGraph:
    """Create the LangGraph workflow for pipeline health monitoring."""

    async def run_data_audit(state: PipelineHealthState) -> dict:
        """Execute the CRM data audit agent."""
        try:
            result = await auditor_agent.ainvoke({
                "input": "Audit all open pipeline opportunities for staleness, "
                         "missing fields, and duplicates.",
                "pipeline_data": state.get("pipeline_data", {}),
            })
            return {
                "audit_results": result,
                "messages": [AIMessage(content=f"Audit complete: {result.get('summary', {})}")],
                "workflow_status": "audit_complete",
            }
        except Exception as e:
            return {
                "error_log": [{"step": "audit", "error": str(e), "timestamp": datetime.utcnow().isoformat()}],
                "workflow_status": "audit_failed",
            }

    async def run_anomaly_detection(state: PipelineHealthState) -> dict:
        """Execute the anomaly detection agent."""
        try:
            audit_data = state.get("audit_results", {}).get("deals", [])
            result = await anomaly_agent.ainvoke({
                "input": "Analyze the audited pipeline data for anomalies.",
                "pipeline_data": audit_data,
            })
            return {
                "anomalies": result.get("anomalies", []),
                "messages": [AIMessage(content=f"Found {len(result.get('anomalies', []))} anomalies")],
                "workflow_status": "anomaly_detection_complete",
            }
        except Exception as e:
            return {
                "error_log": [{"step": "anomaly_detection", "error": str(e), "timestamp": datetime.utcnow().isoformat()}],
                "workflow_status": "anomaly_detection_failed",
            }

    async def run_enrichment(state: PipelineHealthState) -> dict:
        """Execute the data enrichment agent for records with missing data."""
        try:
            incomplete_records = state.get("audit_results", {}).get("incomplete_records", [])
            if not incomplete_records:
                return {
                    "enrichment_results": {"skipped": True, "reason": "No records need enrichment"},
                    "workflow_status": "enrichment_skipped",
                }

            result = await enrichment_agent.ainvoke({
                "input": "Enrich the following incomplete CRM records.",
                "records": incomplete_records[:50],  # Limit batch size
            })
            return {
                "enrichment_results": result,
                "messages": [AIMessage(content=f"Enriched {result.get('successfully_enriched', 0)} records")],
                "workflow_status": "enrichment_complete",
            }
        except Exception as e:
            return {
                "error_log": [{"step": "enrichment", "error": str(e), "timestamp": datetime.utcnow().isoformat()}],
                "workflow_status": "enrichment_failed",
            }

    async def prepare_alerts(state: PipelineHealthState) -> dict:
        """Compile alerts from audit results and anomalies."""
        alerts = []

        # Critical audit findings
        audit = state.get("audit_results", {})
        if audit.get("critical_stale_deals", 0) > 0:
            alerts.append({
                "severity": "critical",
                "type": "stale_deals",
                "message": f"{audit['critical_stale_deals']} deals haven't been updated in 30+ days",
                "value_at_risk": audit.get("stale_pipeline_value", 0),
            })

        # Anomalies
        for anomaly in state.get("anomalies", [])[:10]:  # Top 10
            if anomaly.get("severity") == "high":
                alerts.append({
                    "severity": "high",
                    "type": anomaly["type"],
                    "message": anomaly["details"],
                    "deal_name": anomaly.get("deal_name"),
                    "value_at_risk": anomaly.get("amount_at_risk", 0),
                })

        return {
            "alerts_to_send": alerts,
            "workflow_status": "alerts_prepared",
        }

    async def dispatch_alerts(state: PipelineHealthState) -> dict:
        """Send alerts via the alert dispatcher agent."""
        try:
            alerts = state.get("alerts_to_send", [])
            if not alerts:
                return {"workflow_status": "complete_no_alerts"}

            result = await alert_agent.ainvoke({
                "input": "Route these pipeline health alerts to the appropriate stakeholders.",
                "alerts": alerts,
            })
            return {
                "messages": [AIMessage(content=f"Dispatched {len(alerts)} alerts")],
                "workflow_status": "complete",
            }
        except Exception as e:
            return {
                "error_log": [{"step": "dispatch", "error": str(e), "timestamp": datetime.utcnow().isoformat()}],
                "workflow_status": "dispatch_failed",
            }

    def should_run_enrichment(state: PipelineHealthState) -> Literal["enrich", "skip_enrich"]:
        """Conditional: only run enrichment if there are incomplete records."""
        incomplete = state.get("audit_results", {}).get("incomplete_records", [])
        return "enrich" if len(incomplete) > 0 else "skip_enrich"

    def should_alert(state: PipelineHealthState) -> Literal["alert", "no_alert"]:
        """Conditional: only dispatch alerts if there are issues to report."""
        alerts = state.get("alerts_to_send", [])
        return "alert" if len(alerts) > 0 else "no_alert"

    # Build the graph
    workflow = StateGraph(PipelineHealthState)

    # Add nodes
    workflow.add_node("audit", run_data_audit)
    workflow.add_node("detect_anomalies", run_anomaly_detection)
    workflow.add_node("enrich", run_enrichment)
    workflow.add_node("prepare_alerts", prepare_alerts)
    workflow.add_node("dispatch_alerts", dispatch_alerts)

    # Add edges
    workflow.set_entry_point("audit")
    workflow.add_edge("audit", "detect_anomalies")
    workflow.add_conditional_edges(
        "detect_anomalies",
        should_run_enrichment,
        {"enrich": "enrich", "skip_enrich": "prepare_alerts"},
    )
    workflow.add_edge("enrich", "prepare_alerts")
    workflow.add_conditional_edges(
        "prepare_alerts",
        should_alert,
        {"alert": "dispatch_alerts", "no_alert": END},
    )
    workflow.add_edge("dispatch_alerts", END)

    # Compile with checkpointing
    checkpointer = RedisCheckpointer(redis_url)
    return workflow.compile(checkpointer=checkpointer)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the multi-agent system with Docker, configure LangSmith for tracing, and set up Prometheus metrics for production monitoring.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose Deployment Configuration',
                description:
                  'Production deployment configuration for the pipeline health monitoring system with Redis, Prometheus, and the agent service.',
                code: `version: '3.8'

services:
  pipeline-health-agents:
    build:
      context: .
      dockerfile: Dockerfile.agents
    container_name: pipeline-health-agents
    restart: unless-stopped
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - CRM_API_KEY=\${CRM_API_KEY}
      - CRM_API_URL=\${CRM_API_URL}
      - CLAY_API_KEY=\${CLAY_API_KEY}
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
      - REDIS_URL=redis://redis:6379/0
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=\${LANGSMITH_API_KEY}
      - LANGCHAIN_PROJECT=pipeline-health-monitor
      - LOG_LEVEL=INFO
    ports:
      - "8080:8080"
    depends_on:
      - redis
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: pipeline-health-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: pipeline-health-prometheus
    restart: unless-stopped
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:10.0.0
    container_name: pipeline-health-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_ADMIN_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile.scheduler
    container_name: pipeline-health-scheduler
    restart: unless-stopped
    environment:
      - AGENT_SERVICE_URL=http://pipeline-health-agents:8080
      - SCHEDULE_CRON=0 */6 * * *
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
    depends_on:
      - pipeline-health-agents

volumes:
  redis_data:
  prometheus_data:
  grafana_data:`,
              },
              {
                language: 'python',
                title: 'Prometheus Metrics and Health Endpoints',
                description:
                  'FastAPI endpoints for health checks and Prometheus metrics collection for the agent system.',
                code: `"""
Pipeline Health Agents — Observability Module
Prometheus metrics, health checks, and LangSmith tracing integration.
"""

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.responses import Response
from datetime import datetime
import asyncio
import os

app = FastAPI(title="Pipeline Health Agent Service")

# Prometheus metrics
WORKFLOW_RUNS = Counter(
    "pipeline_health_workflow_runs_total",
    "Total workflow runs",
    ["status"],
)
WORKFLOW_DURATION = Histogram(
    "pipeline_health_workflow_duration_seconds",
    "Workflow execution duration",
    buckets=[30, 60, 120, 300, 600, 1200],
)
ANOMALIES_DETECTED = Gauge(
    "pipeline_health_anomalies_detected",
    "Number of anomalies detected in last run",
    ["severity"],
)
PIPELINE_AT_RISK = Gauge(
    "pipeline_health_value_at_risk_dollars",
    "Total pipeline value at risk",
)
ENRICHMENT_SUCCESS_RATE = Gauge(
    "pipeline_health_enrichment_success_rate",
    "Enrichment success rate (0-1)",
)
LAST_RUN_TIMESTAMP = Gauge(
    "pipeline_health_last_run_timestamp",
    "Unix timestamp of last successful run",
)

# Health state
health_state = {
    "last_run": None,
    "last_status": "unknown",
    "consecutive_failures": 0,
}


@app.get("/health")
async def health_check():
    """Kubernetes-compatible health check endpoint."""
    if health_state["consecutive_failures"] >= 3:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {health_state['consecutive_failures']} consecutive failures",
        )
    return {
        "status": "healthy",
        "last_run": health_state["last_run"],
        "last_status": health_state["last_status"],
    }


@app.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes."""
    # Check Redis connection
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        r.ping()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis not ready: {e}")

    # Check OpenAI API
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


async def record_workflow_metrics(
    status: str,
    duration_seconds: float,
    anomalies: dict,
    value_at_risk: float,
    enrichment_rate: float,
):
    """Record metrics after a workflow run."""
    WORKFLOW_RUNS.labels(status=status).inc()
    WORKFLOW_DURATION.observe(duration_seconds)

    for severity, count in anomalies.items():
        ANOMALIES_DETECTED.labels(severity=severity).set(count)

    PIPELINE_AT_RISK.set(value_at_risk)
    ENRICHMENT_SUCCESS_RATE.set(enrichment_rate)
    LAST_RUN_TIMESTAMP.set(datetime.utcnow().timestamp())

    # Update health state
    health_state["last_run"] = datetime.utcnow().isoformat()
    health_state["last_status"] = status
    if status == "success":
        health_state["consecutive_failures"] = 0
    else:
        health_state["consecutive_failures"] += 1


@app.post("/run")
async def trigger_workflow():
    """Manually trigger the pipeline health workflow."""
    from pipeline_orchestrator import create_pipeline_health_graph
    from agents import create_all_agents

    start_time = datetime.utcnow()

    try:
        agents = create_all_agents()
        graph = create_pipeline_health_graph(
            auditor_agent=agents["auditor"],
            anomaly_agent=agents["anomaly"],
            enrichment_agent=agents["enrichment"],
            alert_agent=agents["alert"],
            redis_url=os.getenv("REDIS_URL"),
        )

        result = await graph.ainvoke({
            "messages": [],
            "run_timestamp": start_time.isoformat(),
        })

        duration = (datetime.utcnow() - start_time).total_seconds()

        await record_workflow_metrics(
            status="success",
            duration_seconds=duration,
            anomalies={
                "high": len([a for a in result.get("anomalies", []) if a.get("severity") == "high"]),
                "medium": len([a for a in result.get("anomalies", []) if a.get("severity") == "medium"]),
            },
            value_at_risk=sum(a.get("amount_at_risk", 0) for a in result.get("anomalies", [])),
            enrichment_rate=result.get("enrichment_results", {}).get("success_rate", 0),
        )

        return {
            "status": "success",
            "duration_seconds": duration,
            "anomalies_found": len(result.get("anomalies", [])),
            "alerts_sent": len(result.get("alerts_to_send", [])),
        }

    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds()
        await record_workflow_metrics(
            status="failed",
            duration_seconds=duration,
            anomalies={},
            value_at_risk=0,
            enrichment_rate=0,
        )
        raise HTTPException(status_code=500, detail=str(e))`,
              },
            ],
          },
        ],
      },
    },

    // ── Pain Point 2: Lead Scoring Latency ──────────────────────────────
    {
      id: 'lead-scoring-latency',
      number: 2,
      title: 'Lead Scoring Latency',
      subtitle: 'Delayed Lead Qualification Killing Conversions',
      summary:
        'Your leads go cold waiting for manual qualification. By the time sales calls, the prospect bought from a competitor.',
      tags: ['lead-scoring', 'automation', 'conversion'],
      metrics: {
        annualCostRange: '$400K - $2M',
        roi: '7x',
        paybackPeriod: '3-4 months',
        investmentRange: '$70K - $130K',
      },
      price: {
        present: {
          title: 'Present Reality',
          severity: 'high',
          description:
            'Leads sit in a queue for 24-72 hours before anyone reviews them. Manual scoring is inconsistent, subjective, and bottlenecked by a single SDR manager.',
          bullets: [
            'Average lead response time exceeds 36 hours across all channels',
            'Only 30% of inbound leads receive a response within 4 hours',
            'SDRs cherry-pick leads by company name instead of data-driven signals',
            'No behavioral scoring — a whitepaper download and a pricing page visit are treated equally',
            'Lead handoff from marketing to sales has no SLA or quality gate',
          ],
        },
        root: {
          title: 'Root Cause',
          severity: 'high',
          description:
            'Lead qualification is an entirely manual process with no real-time scoring model. Behavioral data exists in the marketing platform but never reaches the CRM in a usable form.',
          bullets: [
            'Marketing automation and CRM operate as disconnected silos',
            'No predictive model exists to rank leads by conversion likelihood',
            'Engagement signals (page views, email clicks, content downloads) are not aggregated',
            'Lead scoring rules were set once three years ago and never updated',
            'No feedback loop connects closed-won outcomes back to lead attributes',
          ],
        },
        impact: {
          title: 'Business Impact',
          severity: 'critical',
          description:
            'Conversion rates on high-intent leads drop 80% after the first hour. Every hour of delay hands revenue to faster-responding competitors.',
          bullets: [
            'Lead-to-opportunity conversion rate is 40% below industry benchmark',
            'Hot leads go cold, requiring 5x more touches to re-engage',
            'Sales reps waste cycles on low-intent leads that will never convert',
            'Marketing ROI is impossible to prove because attribution is broken',
            'Competitor win rate increases as response time degrades',
          ],
        },
        cost: {
          title: 'Financial Cost',
          severity: 'high',
          description:
            'Lost conversions, wasted SDR time, and inflated customer acquisition costs compound into significant revenue loss.',
          bullets: [
            '$200K-$800K in lost deals from leads that went cold before contact',
            '$100K-$300K in SDR labor wasted on leads below qualification threshold',
            '$50K-$150K in marketing spend with unattributable ROI',
            'Customer acquisition cost inflated 35% by inefficient lead routing',
            'Pipeline velocity slows as unqualified leads clog the funnel',
          ],
        },
        expectedReturn: {
          title: 'Expected Return',
          severity: 'high',
          description:
            'Real-time ML-based lead scoring cuts response time to under 5 minutes and lifts conversion rates by 40-60%.',
          bullets: [
            'Lead response time drops from 36 hours to under 5 minutes',
            'Conversion rate on scored leads improves 40-60%',
            'SDR productivity increases 50% by focusing only on high-score leads',
            'Marketing can finally prove ROI with closed-loop attribution',
            'Model continuously improves via automated outcome feedback loops',
          ],
        },
      },
      implementation: {
        overview:
          'Build a real-time lead scoring pipeline that combines SQL-based engagement aggregation with a Python ML model to score and route leads within seconds of capture.',
        prerequisites: [
          'Access to marketing automation platform data (HubSpot, Marketo, or equivalent)',
          'CRM database with historical lead-to-opportunity conversion data',
          'Python 3.9+ with scikit-learn, pandas, and FastAPI',
          'pytest >= 7.0 with pytest-asyncio',
          'Docker and docker-compose for containerized deployment',
          'cron, Airflow, or Prefect for job scheduling',
          'Slack incoming webhook URL for operational alerts',
        ],
        toolsUsed: ['SQL', 'Python', 'scikit-learn', 'FastAPI', 'pytest', 'Docker', 'GitHub Actions', 'cron', 'Slack API', 'Prometheus'],
        steps: [
          {
            stepNumber: 1,
            title: 'Lead Engagement Scoring with SQL',
            description:
              'Aggregate behavioral signals from your marketing platform into a single engagement score per lead, creating the feature set for the ML model.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Multi-Signal Lead Engagement Score',
                description:
                  'Combines web activity, email engagement, content downloads, and form submissions into a weighted engagement score per lead.',
                code: `-- Lead Engagement Score: Multi-Signal Aggregation
-- Combines behavioral data into a single score for ML model input

WITH web_activity AS (
  SELECT
    lead_id,
    COUNT(*) FILTER (WHERE page_url LIKE '%/pricing%')    AS pricing_views,
    COUNT(*) FILTER (WHERE page_url LIKE '%/demo%')       AS demo_views,
    COUNT(*) FILTER (WHERE page_url LIKE '%/case-study%') AS case_study_views,
    COUNT(DISTINCT session_id)                            AS total_sessions,
    MAX(visited_at)                                       AS last_visit
  FROM web_visits
  WHERE visited_at >= CURRENT_DATE - INTERVAL '30 days'
  GROUP BY lead_id
),
email_engagement AS (
  SELECT
    lead_id,
    SUM(CASE WHEN opened THEN 1 ELSE 0 END)   AS emails_opened,
    SUM(CASE WHEN clicked THEN 1 ELSE 0 END)   AS emails_clicked,
    COUNT(*)                                    AS emails_sent
  FROM email_events
  WHERE sent_at >= CURRENT_DATE - INTERVAL '30 days'
  GROUP BY lead_id
),
content_activity AS (
  SELECT
    lead_id,
    COUNT(*)                                                AS downloads,
    COUNT(*) FILTER (WHERE asset_type = 'whitepaper')       AS whitepapers,
    COUNT(*) FILTER (WHERE asset_type = 'roi_calculator')   AS roi_tools
  FROM content_downloads
  WHERE downloaded_at >= CURRENT_DATE - INTERVAL '60 days'
  GROUP BY lead_id
)
SELECT
  l.id                                              AS lead_id,
  l.email,
  l.company,
  l.job_title,
  l.employee_count,
  COALESCE(w.pricing_views, 0)   * 10              AS pricing_signal,
  COALESCE(w.demo_views, 0)      * 15              AS demo_signal,
  COALESCE(w.total_sessions, 0)  * 2               AS session_signal,
  COALESCE(e.emails_clicked, 0)  * 5               AS click_signal,
  COALESCE(c.roi_tools, 0)       * 20              AS tool_signal,
  (
    COALESCE(w.pricing_views, 0)  * 10 +
    COALESCE(w.demo_views, 0)     * 15 +
    COALESCE(w.total_sessions, 0) * 2  +
    COALESCE(e.emails_clicked, 0) * 5  +
    COALESCE(c.downloads, 0)      * 3  +
    COALESCE(c.roi_tools, 0)      * 20
  ) AS raw_engagement_score
FROM leads l
LEFT JOIN web_activity       w ON l.id = w.lead_id
LEFT JOIN email_engagement   e ON l.id = e.lead_id
LEFT JOIN content_activity   c ON l.id = c.lead_id
WHERE l.status = 'open'
ORDER BY raw_engagement_score DESC;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'ML-Based Lead Scoring Model',
            description:
              'Train a gradient boosting classifier on historical conversion data, then deploy it as a real-time scoring API that the CRM calls on every new lead.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Lead Scoring Model Training',
                description:
                  'Trains a GradientBoosting model on historical lead data with engagement features, firmographic attributes, and conversion outcomes.',
                code: `"""
Lead Scoring Model — Training Pipeline
Trains on historical lead-to-opportunity conversion data and exports
a serialized model for real-time inference.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib

FEATURE_COLS = [
    "pricing_signal", "demo_signal", "session_signal",
    "click_signal", "tool_signal", "raw_engagement_score",
    "employee_count_bucket", "job_title_encoded",
    "days_since_created", "form_submissions",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    df["job_title_encoded"] = le.fit_transform(
        df["job_title_level"].fillna("unknown")
    )
    df["employee_count_bucket"] = pd.cut(
        df["employee_count"].fillna(0),
        bins=[0, 50, 200, 1000, 5000, 100000],
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    df["days_since_created"] = (
        pd.Timestamp.utcnow() - pd.to_datetime(df["created_date"])
    ).dt.days
    return df


def train_model(df: pd.DataFrame) -> GradientBoostingClassifier:
    df = prepare_features(df)
    X = df[FEATURE_COLS].fillna(0)
    y = df["converted"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"CV AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    joblib.dump(model, "lead_scoring_model.pkl")
    return model`,
              },
              {
                language: 'python',
                title: 'Real-Time Scoring API',
                description:
                  'FastAPI service that scores incoming leads in real-time and pushes the score back to the CRM for immediate routing.',
                code: `"""
Real-Time Lead Scoring API
Receives lead data via webhook, scores instantly, and updates CRM.
Designed for sub-200ms response times.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import httpx
from datetime import datetime

app = FastAPI(title="Lead Scoring Service")
model = joblib.load("lead_scoring_model.pkl")

CRM_API = "https://api.crm.example.com/v2"
SCORE_THRESHOLDS = {"hot": 0.75, "warm": 0.45, "cold": 0.0}


class LeadPayload(BaseModel):
    lead_id: str
    pricing_signal: float = 0
    demo_signal: float = 0
    session_signal: float = 0
    click_signal: float = 0
    tool_signal: float = 0
    raw_engagement_score: float = 0
    employee_count_bucket: int = 1
    job_title_encoded: int = 0
    days_since_created: int = 0
    form_submissions: int = 0


class ScoreResponse(BaseModel):
    lead_id: str
    score: float
    category: str
    scored_at: str


def classify_lead(probability: float) -> str:
    if probability >= SCORE_THRESHOLDS["hot"]:
        return "hot"
    elif probability >= SCORE_THRESHOLDS["warm"]:
        return "warm"
    return "cold"


@app.post("/score", response_model=ScoreResponse)
async def score_lead(payload: LeadPayload):
    features = np.array([[
        payload.pricing_signal, payload.demo_signal,
        payload.session_signal, payload.click_signal,
        payload.tool_signal, payload.raw_engagement_score,
        payload.employee_count_bucket, payload.job_title_encoded,
        payload.days_since_created, payload.form_submissions,
    ]])

    probability = float(model.predict_proba(features)[0][1])
    category = classify_lead(probability)
    scored_at = datetime.utcnow().isoformat()

    async with httpx.AsyncClient() as client:
        await client.patch(
            f"{CRM_API}/leads/{payload.lead_id}",
            json={
                "lead_score__c": round(probability * 100, 1),
                "lead_category__c": category,
                "scored_at__c": scored_at,
            },
            headers={"Authorization": "Bearer $CRM_API_KEY"},
            timeout=10,
        )

    return ScoreResponse(
        lead_id=payload.lead_id,
        score=round(probability, 4),
        category=category,
        scored_at=scored_at,
    )`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate model accuracy, detect feature drift, and test the scoring API end-to-end with SQL data quality assertions and pytest-based integration tests.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Lead Scoring Data Quality Assertions',
                description:
                  'Automated checks for feature completeness, label correctness, scoring table freshness, and referential integrity between leads and engagement data.',
                code: `-- Lead Scoring Data Quality Assertions
-- Run before model training and after each scoring cycle

-- 1. Row count: ensure training set has enough converted leads
SELECT
  CASE
    WHEN COUNT(*) FILTER (WHERE converted = TRUE) < 200
    THEN 'FAIL: insufficient positive samples ('
         || COUNT(*) FILTER (WHERE converted = TRUE) || ' < 200)'
    ELSE 'PASS: ' || COUNT(*) FILTER (WHERE converted = TRUE)
         || ' positive samples available'
  END AS training_data_check
FROM leads
WHERE scored_at__c IS NOT NULL
  AND created_date >= CURRENT_DATE - INTERVAL '6 months';

-- 2. Null check: engagement features must be populated
SELECT
  CASE
    WHEN COUNT(*) > 0
    THEN 'FAIL: ' || COUNT(*) || ' leads missing engagement scores'
    ELSE 'PASS: all leads have engagement data'
  END AS feature_null_check
FROM leads
WHERE status = 'open'
  AND raw_engagement_score IS NULL;

-- 3. Referential integrity: scored leads must link to valid campaigns
SELECT
  CASE
    WHEN COUNT(*) > 0
    THEN 'FAIL: ' || COUNT(*) || ' leads reference non-existent campaigns'
    ELSE 'PASS'
  END AS ref_integrity_check
FROM leads l
LEFT JOIN campaigns c ON l.campaign_id = c.id
WHERE l.campaign_id IS NOT NULL AND c.id IS NULL;

-- 4. Freshness: scores must be current
SELECT
  CASE
    WHEN MAX(scored_at__c) < NOW() - INTERVAL '2 hours'
    THEN 'FAIL: scoring is stale ('
         || EXTRACT(HOUR FROM NOW() - MAX(scored_at__c)) || 'h ago)'
    ELSE 'PASS: scoring data is fresh'
  END AS freshness_check
FROM leads
WHERE scored_at__c IS NOT NULL;`,
              },
              {
                language: 'python',
                title: 'Lead Scoring Model & API Validation Tests',
                description:
                  'pytest suite that validates model accuracy thresholds, detects feature distribution drift, and tests the FastAPI scoring endpoint end-to-end.',
                code: `"""
Lead Scoring — Validation Test Suite
Tests model accuracy, feature drift detection, and scoring API responses.
"""

import pytest
import numpy as np
import joblib
from datetime import datetime
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient
from sklearn.metrics import roc_auc_score

from lead_scoring_api import app, classify_lead, SCORE_THRESHOLDS


@pytest.fixture
def trained_model():
    return joblib.load("lead_scoring_model.pkl")


@pytest.fixture
def validation_data():
    """Load held-out validation set."""
    import pandas as pd
    return pd.read_parquet("validation_leads.parquet")


class TestModelAccuracy:
    def test_auc_above_threshold(self, trained_model, validation_data):
        X = validation_data.drop(columns=["converted", "lead_id"])
        y = validation_data["converted"].astype(int)
        y_prob = trained_model.predict_proba(X.fillna(0))[:, 1]
        auc = roc_auc_score(y, y_prob)
        assert auc >= 0.72, f"Model AUC {auc:.3f} below minimum 0.72"

    def test_hot_leads_convert_above_baseline(self, trained_model, validation_data):
        X = validation_data.drop(columns=["converted", "lead_id"])
        probs = trained_model.predict_proba(X.fillna(0))[:, 1]
        hot_mask = probs >= SCORE_THRESHOLDS["hot"]
        hot_conversion = validation_data.loc[hot_mask, "converted"].mean()
        assert hot_conversion >= 0.30, (
            f"Hot lead conversion rate {hot_conversion:.2%} is below 30%"
        )


class TestFeatureDrift:
    def test_engagement_distribution_stable(self, validation_data):
        current_mean = validation_data["raw_engagement_score"].mean()
        historical_mean = 45.0  # baseline from training window
        drift_pct = abs(current_mean - historical_mean) / historical_mean * 100
        assert drift_pct < 30, (
            f"Engagement score drifted {drift_pct:.1f}% from baseline"
        )

    def test_no_new_null_features(self, validation_data):
        null_pcts = validation_data.isnull().mean()
        high_null = null_pcts[null_pcts > 0.5]
        assert high_null.empty, (
            f"Features with >50% nulls: {list(high_null.index)}"
        )


class TestClassification:
    @pytest.mark.parametrize("prob,expected", [
        (0.80, "hot"), (0.75, "hot"),
        (0.50, "warm"), (0.45, "warm"),
        (0.30, "cold"), (0.0, "cold"),
    ])
    def test_lead_classification(self, prob, expected):
        assert classify_lead(prob) == expected


@pytest.mark.asyncio
class TestScoringAPI:
    async def test_score_endpoint_returns_valid_response(self):
        async with AsyncClient(app=app, base_url="http://test") as client:
            with patch("lead_scoring_api.httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__ = AsyncMock(
                    return_value=AsyncMock(
                        patch=AsyncMock(return_value=AsyncMock(status_code=200))
                    )
                )
                resp = await client.post("/score", json={
                    "lead_id": "test_001",
                    "raw_engagement_score": 60.0,
                    "pricing_signal": 20.0,
                    "demo_signal": 15.0,
                })
                assert resp.status_code == 200
                body = resp.json()
                assert "score" in body
                assert 0.0 <= body["score"] <= 1.0
                assert body["category"] in ("hot", "warm", "cold")`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Deploy the lead scoring API as a containerized service with model registry, database migrations, and scheduled retraining via a battle-tested deployment script.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Lead Scoring API Deployment Script',
                description:
                  'Deploys the scoring API container, runs DB migrations, registers the model artifact, and configures cron-based retraining and health checks.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ── Lead Scoring API — Containerized Deployment ─────────────────
APP_NAME="lead-scoring-api"
IMAGE_TAG="\${APP_NAME}:\$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')"
MODEL_DIR="/opt/models/\${APP_NAME}"
LOG_DIR="/var/log/\${APP_NAME}"
REQUIRED_VARS=("CRM_API_KEY" "DATABASE_URL" "SLACK_WEBHOOK_URL" "MODEL_REGISTRY_URL")

echo "==> Deploying \${APP_NAME} (\${IMAGE_TAG})..."

# 1. Validate required environment variables
for var in "\${REQUIRED_VARS[@]}"; do
  if [[ -z "\${!var:-}" ]]; then
    echo "ERROR: Required env var \${var} is not set." >&2
    exit 1
  fi
done
echo "    Environment validated."

# 2. Verify Docker is running
if ! docker info &>/dev/null; then
  echo "ERROR: Docker daemon is not running." >&2
  exit 1
fi
echo "    Docker daemon is healthy."

# 3. Build container image
docker build -t "\${IMAGE_TAG}" -f Dockerfile.scoring .
echo "    Container image built: \${IMAGE_TAG}"

# 4. Run database migrations
docker run --rm \\
  -e DATABASE_URL="\${DATABASE_URL}" \\
  "\${IMAGE_TAG}" python migrate.py --apply
echo "    Database migrations applied."

# 5. Pull latest model from registry
sudo mkdir -p "\${MODEL_DIR}" "\${LOG_DIR}"
curl -fsSL "\${MODEL_REGISTRY_URL}/latest/lead_scoring_model.pkl" \\
  -o "\${MODEL_DIR}/lead_scoring_model.pkl"
echo "    Latest model downloaded to \${MODEL_DIR}."

# 6. Stop existing container and start new one
docker stop "\${APP_NAME}" 2>/dev/null || true
docker rm "\${APP_NAME}" 2>/dev/null || true
docker run -d \\
  --name "\${APP_NAME}" \\
  --restart unless-stopped \\
  -p 8080:8080 \\
  -v "\${MODEL_DIR}:/app/models:ro" \\
  -e CRM_API_KEY="\${CRM_API_KEY}" \\
  -e DATABASE_URL="\${DATABASE_URL}" \\
  -e SLACK_WEBHOOK_URL="\${SLACK_WEBHOOK_URL}" \\
  "\${IMAGE_TAG}"
echo "    Container started on port 8080."

# 7. Schedule weekly model retraining
RETRAIN_EXPR="0 3 * * 0"
RETRAIN_CMD="docker exec \${APP_NAME} python retrain.py >> \${LOG_DIR}/retrain.log 2>&1"
( crontab -l 2>/dev/null | grep -v "\${APP_NAME}" ; echo "\${RETRAIN_EXPR} \${RETRAIN_CMD}  # \${APP_NAME}-retrain" ) | crontab -
echo "    Retraining cron set: \${RETRAIN_EXPR}"

# 8. Health check
sleep 3
if curl -sf http://localhost:8080/health > /dev/null; then
  echo "==> Deployment successful. API is healthy."
else
  echo "ERROR: Health check failed. Check logs: docker logs \${APP_NAME}" >&2
  exit 1
fi`,
              },
              {
                language: 'python',
                title: 'Scoring Service Configuration Loader',
                description:
                  'Environment-aware configuration for the lead scoring service: loads secrets, validates model paths, and initializes database and HTTP connection pools.',
                code: `"""
Lead Scoring API — Configuration Loader
Manages env-based config, model registry paths, and connection pooling.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool


@dataclass(frozen=True)
class ScoringConfig:
    """Immutable service configuration loaded at startup."""
    crm_api_key: str
    crm_api_url: str
    database_url: str
    slack_webhook_url: str
    model_path: str
    model_registry_url: str
    hot_threshold: float = 0.75
    warm_threshold: float = 0.45
    retrain_interval_days: int = 7
    api_port: int = 8080
    log_level: str = "INFO"


def load_config() -> ScoringConfig:
    required = {
        "CRM_API_KEY": os.getenv("CRM_API_KEY"),
        "CRM_API_URL": os.getenv("CRM_API_URL", "https://api.crm.example.com/v2"),
        "DATABASE_URL": os.getenv("DATABASE_URL"),
        "SLACK_WEBHOOK_URL": os.getenv("SLACK_WEBHOOK_URL"),
        "MODEL_REGISTRY_URL": os.getenv("MODEL_REGISTRY_URL", ""),
    }
    missing = [k for k, v in required.items()
               if not v and k not in ("CRM_API_URL", "MODEL_REGISTRY_URL")]
    if missing:
        print(f"FATAL: Missing env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    model_path = os.getenv("MODEL_PATH", "/app/models/lead_scoring_model.pkl")
    if not Path(model_path).exists():
        print(f"FATAL: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    return ScoringConfig(
        crm_api_key=required["CRM_API_KEY"],
        crm_api_url=required["CRM_API_URL"],
        database_url=required["DATABASE_URL"],
        slack_webhook_url=required["SLACK_WEBHOOK_URL"],
        model_path=model_path,
        model_registry_url=required["MODEL_REGISTRY_URL"],
        hot_threshold=float(os.getenv("HOT_THRESHOLD", "0.75")),
        warm_threshold=float(os.getenv("WARM_THRESHOLD", "0.45")),
        retrain_interval_days=int(os.getenv("RETRAIN_INTERVAL_DAYS", "7")),
        api_port=int(os.getenv("API_PORT", "8080")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


def create_db_pool(config: ScoringConfig) -> sqlalchemy.engine.Engine:
    return create_engine(
        config.database_url,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_pre_ping=True,
    )


config = load_config()
db_engine = create_db_pool(config)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Close the feedback loop by tracking model performance against actual conversion outcomes, monitoring score distribution drift, and alerting via Slack when model accuracy degrades.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Conversion Outcome Feedback Query',
                description:
                  'Extracts closed-loop data connecting original lead scores to actual conversion outcomes for model retraining and performance monitoring.',
                code: `-- Conversion Feedback Loop: Score vs. Outcome
-- Feeds actual outcomes back to improve the scoring model

SELECT
  l.id                             AS lead_id,
  l.lead_score__c                  AS predicted_score,
  l.lead_category__c               AS predicted_category,
  l.scored_at__c                   AS scored_at,
  o.stage_name                     AS final_stage,
  CASE
    WHEN o.is_won = TRUE  THEN 'converted'
    WHEN o.is_won = FALSE THEN 'lost'
    ELSE 'open'
  END                              AS outcome,
  o.amount                         AS deal_value,
  DATEDIFF(day, l.created_date, o.close_date) AS days_to_close,
  l.pricing_signal,
  l.demo_signal,
  l.session_signal,
  l.click_signal,
  l.raw_engagement_score,
  l.employee_count,
  l.job_title_level
FROM leads l
LEFT JOIN opportunities o
  ON l.converted_opportunity_id = o.id
WHERE l.scored_at__c IS NOT NULL
  AND l.created_date >= CURRENT_DATE - INTERVAL '6 months'
ORDER BY l.scored_at__c DESC;`,
              },
              {
                language: 'python',
                title: 'Model Performance Slack Alerting',
                description:
                  'Monitors lead scoring model performance, tracks conversion rate by predicted category, detects score distribution drift, and sends Slack alerts when thresholds are breached.',
                code: `"""
Lead Scoring — Model Performance Monitoring & Slack Alerts
Tracks predicted vs. actual conversion rates and alerts on drift.
"""

import os
import requests
from datetime import datetime
from dataclasses import dataclass
from sqlalchemy import create_engine, text

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
engine = create_engine(DATABASE_URL)


@dataclass
class ModelThresholds:
    min_hot_conversion_rate: float = 0.30
    min_overall_auc_proxy: float = 0.15
    max_cold_conversion_rate: float = 0.10
    max_score_staleness_hours: int = 4
    min_daily_scored_leads: int = 10


def check_model_performance(thresholds: ModelThresholds) -> list[dict]:
    alerts: list[dict] = []
    with engine.connect() as conn:
        # Conversion rates by predicted category (last 30 days)
        rows = conn.execute(text(\"\"\"
            SELECT
              l.lead_category__c AS category,
              COUNT(*) AS total,
              SUM(CASE WHEN o.is_won = TRUE THEN 1 ELSE 0 END) AS converted
            FROM leads l
            LEFT JOIN opportunities o ON l.converted_opportunity_id = o.id
            WHERE l.scored_at__c >= NOW() - INTERVAL '30 days'
              AND l.lead_category__c IS NOT NULL
            GROUP BY l.lead_category__c
        \"\"\")).fetchall()

        rates = {r.category: r.converted / max(r.total, 1) for r in rows}

        if rates.get("hot", 1.0) < thresholds.min_hot_conversion_rate:
            alerts.append({
                "type": "hot_conversion_drop",
                "message": f"Hot lead conversion rate "
                           f"{rates['hot']:.1%} below "
                           f"{thresholds.min_hot_conversion_rate:.0%} threshold",
            })

        if rates.get("cold", 0.0) > thresholds.max_cold_conversion_rate:
            alerts.append({
                "type": "cold_leakage",
                "message": f"Cold leads converting at "
                           f"{rates['cold']:.1%} — model may be "
                           f"under-scoring good leads",
            })

        # Score distribution check — gap between hot and cold rates
        gap = rates.get("hot", 0) - rates.get("cold", 0)
        if gap < thresholds.min_overall_auc_proxy:
            alerts.append({
                "type": "model_degradation",
                "message": f"Hot-cold conversion gap is only "
                           f"{gap:.1%} — model losing discriminative power",
            })

        # Freshness check
        fresh = conn.execute(text(
            "SELECT MAX(scored_at__c) AS last_scored FROM leads"
        )).fetchone()
        if fresh and fresh.last_scored:
            age_h = (datetime.utcnow() - fresh.last_scored).total_seconds() / 3600
            if age_h > thresholds.max_score_staleness_hours:
                alerts.append({
                    "type": "scoring_stale",
                    "message": f"Lead scoring is {age_h:.1f}h old "
                               f"(max {thresholds.max_score_staleness_hours}h)",
                })
    return alerts


def send_slack_alerts(alerts: list[dict]) -> None:
    if not alerts or not SLACK_WEBHOOK:
        return
    severity = "critical" if len(alerts) >= 2 else "warning"
    icon = ":rotating_light:" if severity == "critical" else ":warning:"
    lines = [f"  - [{a['type']}] {a['message']}" for a in alerts]
    payload = {
        "text": (
            f"{icon} *Lead Scoring Model Alert* ({severity})\\n"
            + "\\n".join(lines)
            + f"\\nChecked at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        ),
    }
    requests.post(SLACK_WEBHOOK, json=payload, timeout=10)


if __name__ == "__main__":
    thresholds = ModelThresholds()
    alerts = check_model_performance(thresholds)
    if alerts:
        print(f"Found {len(alerts)} alert(s), notifying Slack...")
        send_slack_alerts(alerts)
    else:
        print("Lead scoring model health checks passed.")`,
              },
            ],
          },
        ],
      },
      aiEasyWin: {
        overview:
          'Use ChatGPT/Claude with Zapier to build a no-code lead scoring system that analyzes engagement signals in real-time and automatically routes hot leads to sales reps.',
        estimatedMonthlyCost: '$120 - $200/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'HubSpot or Salesforce (existing)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Apollo AI ($49/mo)', 'Gong AI (varies)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Set up Zapier triggers to capture lead engagement signals in real-time: form submissions, email opens, page views, and content downloads. Aggregate signals into a scoring-ready format.',
            toolsUsed: ['Zapier', 'HubSpot/Salesforce', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Multi-Trigger Lead Signal Aggregation',
                description:
                  'Configures multiple Zapier triggers to capture engagement signals and aggregate them for AI-powered scoring.',
                code: `{
  "workflow_name": "Lead Signal Aggregator",
  "triggers": [
    {
      "id": "form_submission",
      "app": "hubspot",
      "event": "new_form_submission",
      "config": {
        "form_types": ["contact_us", "demo_request", "content_download"],
        "output_mapping": {
          "lead_email": "{{trigger.email}}",
          "signal_type": "form_submission",
          "signal_value": "{{trigger.form_name}}",
          "signal_weight": "{{#if trigger.form_name == 'demo_request'}}20{{else}}5{{/if}}",
          "timestamp": "{{trigger.submitted_at}}"
        }
      }
    },
    {
      "id": "email_engagement",
      "app": "hubspot",
      "event": "email_opened_or_clicked",
      "config": {
        "filter": {
          "campaign_type": ["nurture", "sales_outreach", "product_update"]
        },
        "output_mapping": {
          "lead_email": "{{trigger.recipient_email}}",
          "signal_type": "{{#if trigger.clicked}}email_click{{else}}email_open{{/if}}",
          "signal_value": "{{trigger.email_subject}}",
          "signal_weight": "{{#if trigger.clicked}}8{{else}}2{{/if}}",
          "timestamp": "{{trigger.event_timestamp}}"
        }
      }
    },
    {
      "id": "page_view",
      "app": "hubspot",
      "event": "page_viewed",
      "config": {
        "filter": {
          "page_url_contains": ["/pricing", "/demo", "/case-study", "/product"]
        },
        "output_mapping": {
          "lead_email": "{{trigger.contact_email}}",
          "signal_type": "page_view",
          "signal_value": "{{trigger.page_url}}",
          "signal_weight": "{{#if trigger.page_url contains '/pricing'}}15{{else if trigger.page_url contains '/demo'}}12{{else}}5{{/if}}",
          "timestamp": "{{trigger.viewed_at}}"
        }
      }
    }
  ],
  "aggregation_action": {
    "app": "google_sheets",
    "action": "append_or_update_row",
    "config": {
      "spreadsheet_id": "{{env.LEAD_SIGNALS_SHEET_ID}}",
      "worksheet": "Lead_Signals",
      "lookup_column": "Lead_Email",
      "row_data": {
        "Lead_Email": "{{trigger.lead_email}}",
        "Last_Signal_Type": "{{trigger.signal_type}}",
        "Last_Signal_Value": "{{trigger.signal_value}}",
        "Last_Signal_Timestamp": "{{trigger.timestamp}}",
        "Total_Signals": "={{COUNTIF(A:A, trigger.lead_email)}}",
        "Cumulative_Score": "={{SUMIF(A:A, trigger.lead_email, D:D)}}",
        "Last_Updated": "={{NOW()}}"
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
              'Use ChatGPT or Claude to analyze lead signals, calculate conversion probability, and categorize leads as hot/warm/cold with specific routing recommendations.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro', 'Zapier AI Actions'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Lead Scoring AI Prompt Template',
                description:
                  'Structured prompt for AI to score leads based on engagement signals and recommend immediate actions.',
                code: `system_prompt: |
  You are an expert Sales Development AI that scores leads based on
  behavioral signals. You understand that timing is critical - a lead
  viewing pricing pages should be contacted within 5 minutes.

  Your scoring philosophy:
  - Demo requests and pricing page views indicate high intent
  - Multiple signals in 24 hours indicate active evaluation
  - Email clicks on product content show genuine interest
  - Content downloads without follow-up activity may indicate research phase

user_prompt_template: |
  ## Lead Scoring Request

  **Lead Email:** {{lead_email}}
  **Company:** {{company_name}}
  **Job Title:** {{job_title}}
  **Lead Source:** {{lead_source}}
  **Days Since Created:** {{days_since_created}}

  ### Engagement Signals (Last 30 Days):
  \`\`\`json
  {{engagement_signals_json}}
  \`\`\`

  ### Firmographic Data:
  - Company Size: {{employee_count}}
  - Industry: {{industry}}
  - Annual Revenue: {{annual_revenue}}

  ### Scoring Instructions:

  1. **Calculate Engagement Score (0-100):**
     - Weight recent signals (last 7 days) at 2x
     - High-intent pages (pricing, demo) score higher
     - Multiple sessions indicate active evaluation

  2. **Calculate Fit Score (0-100):**
     - Company size alignment with ICP
     - Industry match
     - Job title decision-making authority

  3. **Determine Lead Category:**
     - HOT (combined > 150): Contact within 5 minutes
     - WARM (combined 80-150): Contact within 4 hours
     - COLD (combined < 80): Add to nurture sequence

  4. **Generate Action Recommendation:**
     - Specific outreach message suggestion
     - Recommended channel (call vs email)
     - Key talking points based on content consumed

  ### Required Output Format:

  \`\`\`json
  {
    "lead_email": "{{lead_email}}",
    "scores": {
      "engagement_score": <0-100>,
      "fit_score": <0-100>,
      "combined_score": <0-200>,
      "conversion_probability": <0.0-1.0>
    },
    "category": "<HOT|WARM|COLD>",
    "urgency": "<IMMEDIATE|SAME_DAY|NEXT_DAY|NURTURE>",
    "reasoning": {
      "engagement_factors": ["<factor 1>", "<factor 2>"],
      "fit_factors": ["<factor 1>", "<factor 2>"],
      "risk_factors": ["<any concerns>"]
    },
    "recommended_action": {
      "channel": "<call|email|linkedin>",
      "timing": "<specific timeframe>",
      "message_suggestion": "<personalized opening line>",
      "talking_points": ["<point 1>", "<point 2>"],
      "content_to_reference": "<specific content they engaged with>"
    },
    "routing": {
      "assigned_to": "<SDR|AE|nurture_automation>",
      "queue_priority": <1-10>
    }
  }
  \`\`\`

output_instructions: |
  - Be specific about WHY the lead scored as they did
  - Personalize the message suggestion based on actual content consumed
  - If lead is HOT, emphasize urgency in routing
  - Include any red flags (e.g., competitor employee, student email)`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier to automatically score leads in real-time, update CRM records, and route hot leads to sales reps via Slack with personalized talking points.',
            toolsUsed: ['Zapier', 'Slack', 'HubSpot/Salesforce', 'Calendar'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Real-Time Lead Scoring and Routing Workflow',
                description:
                  'Complete Zapier workflow that scores leads with AI, updates CRM, and routes hot leads to available reps instantly.',
                code: `{
  "workflow_name": "Real-Time AI Lead Scoring & Routing",
  "trigger": {
    "app": "google_sheets",
    "event": "new_or_updated_row",
    "config": {
      "spreadsheet_id": "{{env.LEAD_SIGNALS_SHEET_ID}}",
      "worksheet": "Lead_Signals",
      "filter": {
        "Cumulative_Score": { "greater_than": 20 },
        "Last_Scored_At": { "older_than_minutes": 30 }
      }
    }
  },
  "steps": [
    {
      "step_number": 1,
      "name": "Fetch Lead Details from CRM",
      "app": "hubspot",
      "action": "get_contact",
      "config": {
        "email": "{{trigger.Lead_Email}}",
        "properties": [
          "firstname", "lastname", "company", "jobtitle",
          "industry", "numberofemployees", "annualrevenue",
          "hs_lead_status", "lifecyclestage", "createdate"
        ]
      }
    },
    {
      "step_number": 2,
      "name": "Fetch Engagement History",
      "app": "hubspot",
      "action": "get_contact_activity",
      "config": {
        "contact_id": "{{step1.contact_id}}",
        "activity_types": ["PAGE_VIEW", "EMAIL_OPEN", "EMAIL_CLICK", "FORM_SUBMISSION"],
        "days_back": 30,
        "limit": 50
      }
    },
    {
      "step_number": 3,
      "name": "AI Lead Scoring",
      "app": "chatgpt",
      "action": "conversation",
      "config": {
        "model": "gpt-4",
        "system_message": "{{prompts.lead_scoring.system}}",
        "user_message": "{{prompts.lead_scoring.user | replace_all(trigger, step1, step2)}}",
        "max_tokens": 2000,
        "temperature": 0.2
      }
    },
    {
      "step_number": 4,
      "name": "Parse AI Response",
      "app": "code",
      "action": "run_javascript",
      "config": {
        "code": "const result = JSON.parse(inputData.ai_response); return result;"
      }
    },
    {
      "step_number": 5,
      "name": "Update CRM with Score",
      "app": "hubspot",
      "action": "update_contact",
      "config": {
        "contact_id": "{{step1.contact_id}}",
        "properties": {
          "lead_score_ai": "{{step4.scores.combined_score}}",
          "lead_category_ai": "{{step4.category}}",
          "conversion_probability": "{{step4.scores.conversion_probability}}",
          "ai_scored_at": "{{now}}",
          "recommended_action": "{{step4.recommended_action.message_suggestion}}"
        }
      }
    },
    {
      "step_number": 6,
      "name": "Route HOT Leads to Slack",
      "condition": "{{step4.category == 'HOT'}}",
      "app": "slack",
      "action": "send_channel_message",
      "config": {
        "channel": "#hot-leads",
        "message_blocks": [
          {
            "type": "header",
            "text": ":fire: HOT LEAD - Contact Within 5 Minutes!"
          },
          {
            "type": "section",
            "text": "*{{step1.firstname}} {{step1.lastname}}* at *{{step1.company}}*\\nTitle: {{step1.jobtitle}}\\nScore: {{step4.scores.combined_score}}/200 ({{step4.scores.conversion_probability | multiply:100}}% conversion probability)"
          },
          {
            "type": "section",
            "text": "*Why they're hot:*\\n{{#each step4.reasoning.engagement_factors}}• {{this}}\\n{{/each}}"
          },
          {
            "type": "section",
            "text": "*Recommended approach:*\\n:phone: {{step4.recommended_action.channel}} - {{step4.recommended_action.timing}}\\n\\n_\\"{{step4.recommended_action.message_suggestion}}\\"_"
          },
          {
            "type": "section",
            "text": "*Talking points:*\\n{{#each step4.recommended_action.talking_points}}• {{this}}\\n{{/each}}"
          },
          {
            "type": "actions",
            "elements": [
              {
                "type": "button",
                "text": "Claim This Lead",
                "action_id": "claim_lead_{{step1.contact_id}}",
                "style": "primary"
              },
              {
                "type": "button",
                "text": "View in HubSpot",
                "url": "https://app.hubspot.com/contacts/{{env.HUBSPOT_PORTAL_ID}}/contact/{{step1.contact_id}}"
              }
            ]
          }
        ]
      }
    },
    {
      "step_number": 7,
      "name": "Create Task for WARM Leads",
      "condition": "{{step4.category == 'WARM'}}",
      "app": "hubspot",
      "action": "create_task",
      "config": {
        "contact_id": "{{step1.contact_id}}",
        "subject": "Follow up with {{step1.firstname}} - AI Score: {{step4.scores.combined_score}}",
        "body": "{{step4.recommended_action.message_suggestion}}\\n\\nTalking Points:\\n{{step4.recommended_action.talking_points | join('\\n')}}",
        "due_date": "{{#if step4.urgency == 'SAME_DAY'}}{{today}}{{else}}{{tomorrow}}{{/if}}",
        "priority": "HIGH"
      }
    },
    {
      "step_number": 8,
      "name": "Update Signal Sheet with Score",
      "app": "google_sheets",
      "action": "update_row",
      "config": {
        "spreadsheet_id": "{{env.LEAD_SIGNALS_SHEET_ID}}",
        "worksheet": "Lead_Signals",
        "lookup_column": "Lead_Email",
        "lookup_value": "{{trigger.Lead_Email}}",
        "row_data": {
          "AI_Score": "{{step4.scores.combined_score}}",
          "AI_Category": "{{step4.category}}",
          "Conversion_Probability": "{{step4.scores.conversion_probability}}",
          "Last_Scored_At": "{{now}}",
          "Routed_To": "{{#if step4.category == 'HOT'}}hot-leads-channel{{else}}task-queue{{/if}}"
        }
      }
    }
  ]
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent ML system that ingests engagement signals in real-time, trains adaptive scoring models, and routes leads through an intelligent assignment engine that matches leads to the best available rep.',
        estimatedMonthlyCost: '$800 - $1,500/month',
        architecture:
          'A Supervisor agent coordinates four specialists: SignalCollector aggregates engagement data, ScoringModelAgent maintains and retrains the ML model, LeadAnalyzer generates scores with explanations, and RoutingAgent matches leads to optimal reps based on expertise and availability.',
        agents: [
          {
            name: 'SignalCollectorAgent',
            role: 'Engagement Signal Aggregator',
            goal: 'Collect and normalize engagement signals from multiple sources in real-time, maintaining a unified lead activity timeline.',
            tools: ['HubSpot API', 'Salesforce API', 'Segment', 'Webhook Receiver'],
          },
          {
            name: 'ScoringModelAgent',
            role: 'ML Model Trainer',
            goal: 'Train and continuously improve lead scoring models using historical conversion data, detecting feature drift and triggering retraining.',
            tools: ['Scikit-learn', 'MLflow', 'Feature Store', 'Model Registry'],
          },
          {
            name: 'LeadAnalyzerAgent',
            role: 'Real-Time Lead Scorer',
            goal: 'Score incoming leads in sub-second latency, generate human-readable explanations, and flag high-intent signals for immediate action.',
            tools: ['ML Model', 'SHAP Explainer', 'Redis Cache'],
          },
          {
            name: 'IntelligentRouterAgent',
            role: 'Lead-to-Rep Matcher',
            goal: 'Match scored leads to the optimal sales rep based on expertise, current workload, timezone, and historical performance with similar leads.',
            tools: ['Rep Profile Database', 'Calendar API', 'Workload Balancer'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Sequential',
          stateManagement: 'Redis-backed real-time state with Kafka for signal streaming',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the real-time lead scoring multi-agent system with CrewAI, establishing clear roles for signal collection, model management, scoring, and intelligent routing.',
            toolsUsed: ['CrewAI', 'LangChain', 'Python'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Lead Scoring Agent Crew Definition',
                description:
                  'CrewAI configuration for the real-time lead scoring system with four specialist agents.',
                code: `"""
Real-Time Lead Scoring Multi-Agent System
CrewAI-based agent definitions for sub-second lead scoring and routing.
"""

from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import os

llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
)


class LeadScoringAgents:
    """Factory class for creating lead scoring agents."""

    @staticmethod
    def create_signal_collector_agent(tools: List[Any]) -> Agent:
        return Agent(
            role="Engagement Signal Aggregator",
            goal="""Collect and unify engagement signals from all sources:
                1. Website activity (page views, time on site, scroll depth)
                2. Email engagement (opens, clicks, replies)
                3. Content consumption (downloads, video views, webinar attendance)
                4. Social signals (LinkedIn profile views, ad clicks)
                5. Sales touchpoints (calls, meetings, proposals sent)""",
            backstory="""You are a data integration specialist who built the
                customer data platform at a unicorn SaaS company. You understand
                that a single missed signal can mean a lost deal. You normalize
                data from disparate sources into a unified timeline that tells
                the complete story of a lead's journey.""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=10,
        )

    @staticmethod
    def create_scoring_model_agent(tools: List[Any]) -> Agent:
        return Agent(
            role="ML Model Manager",
            goal="""Maintain and improve the lead scoring ML model:
                1. Monitor model performance against actual conversions
                2. Detect feature drift and data quality issues
                3. Trigger retraining when accuracy drops below threshold
                4. A/B test new features and model architectures
                5. Ensure model fairness across segments""",
            backstory="""You are an ML engineer who deployed scoring models at
                scale for a Fortune 500 sales organization. You know that models
                decay over time and that the best model is one that's continuously
                learning. You balance model sophistication with interpretability
                because sales teams don't trust black boxes.""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=15,
        )

    @staticmethod
    def create_lead_analyzer_agent(tools: List[Any]) -> Agent:
        return Agent(
            role="Real-Time Lead Scorer",
            goal="""Score leads with sub-second latency and full explainability:
                1. Generate scores using the latest ML model
                2. Provide SHAP-based explanations for each score
                3. Flag leads showing buying signals for immediate action
                4. Compare lead to historical conversion patterns
                5. Predict optimal contact timing and channel""",
            backstory="""You are a sales intelligence analyst who can read intent
                from behavioral patterns. You've studied thousands of won deals
                and know the signals that separate tire-kickers from buyers.
                You explain your reasoning clearly so reps trust and act on
                your recommendations.""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=10,
        )

    @staticmethod
    def create_routing_agent(tools: List[Any]) -> Agent:
        return Agent(
            role="Intelligent Lead Router",
            goal="""Match leads to the optimal sales rep considering:
                1. Rep expertise with the lead's industry/company size
                2. Current workload and capacity
                3. Timezone alignment for faster response
                4. Historical win rate with similar leads
                5. Round-robin fairness when factors are equal""",
            backstory="""You are a sales operations leader who increased team
                conversion rates 40% through intelligent routing. You know that
                the right rep for a lead isn't always the next in queue - it's
                the one most likely to close. You balance optimization with
                fairness to keep the team motivated.""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=10,
        )


def create_lead_scoring_crew(
    signal_tools: List[Any],
    model_tools: List[Any],
    analyzer_tools: List[Any],
    routing_tools: List[Any],
) -> Crew:
    """Create the complete lead scoring crew."""
    factory = LeadScoringAgents()

    signal_collector = factory.create_signal_collector_agent(signal_tools)
    model_manager = factory.create_scoring_model_agent(model_tools)
    lead_analyzer = factory.create_lead_analyzer_agent(analyzer_tools)
    router = factory.create_routing_agent(routing_tools)

    return Crew(
        agents=[signal_collector, model_manager, lead_analyzer, router],
        process=Process.sequential,
        verbose=True,
        memory=True,
        cache=True,
        max_rpm=60,  # Higher rate for real-time scoring
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the Signal Collector agent with tools to ingest engagement data from multiple sources and maintain a unified lead activity timeline.',
            toolsUsed: ['LangChain Tools', 'HubSpot API', 'Segment', 'Redis Streams'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Signal Collector Agent Tools',
                description:
                  'Tool implementations for real-time engagement signal collection and normalization.',
                code: `"""
Signal Collector Agent — Tool Implementations
Real-time engagement signal ingestion from multiple sources.
"""

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import httpx
import redis
import json
import os


class SignalIngestionInput(BaseModel):
    lead_email: str = Field(description="Email address of the lead")
    lookback_days: int = Field(default=30, description="Days of history to fetch")


class EngagementSignalCollectorTool(BaseTool):
    name: str = "collect_engagement_signals"
    description: str = """Collect all engagement signals for a lead from multiple sources.
        Returns a unified timeline of website visits, email engagement, content downloads,
        and sales touchpoints."""
    args_schema: type[BaseModel] = SignalIngestionInput

    def __init__(self):
        super().__init__()
        self.redis_client = redis.from_url(os.getenv("REDIS_URL"))
        self.hubspot_key = os.getenv("HUBSPOT_API_KEY")
        self.segment_key = os.getenv("SEGMENT_WRITE_KEY")

    def _run(self, lead_email: str, lookback_days: int = 30) -> Dict[str, Any]:
        signals = []
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        # 1. Fetch from HubSpot
        hubspot_signals = self._fetch_hubspot_signals(lead_email, cutoff_date)
        signals.extend(hubspot_signals)

        # 2. Fetch from Redis cache (real-time website events)
        redis_signals = self._fetch_redis_signals(lead_email, cutoff_date)
        signals.extend(redis_signals)

        # 3. Calculate signal aggregates
        signals_sorted = sorted(signals, key=lambda x: x["timestamp"], reverse=True)

        return {
            "lead_email": lead_email,
            "signal_count": len(signals_sorted),
            "signals": signals_sorted[:100],  # Cap at 100 most recent
            "aggregates": self._calculate_aggregates(signals_sorted),
            "recency_score": self._calculate_recency_score(signals_sorted),
            "velocity_score": self._calculate_velocity_score(signals_sorted),
        }

    def _fetch_hubspot_signals(
        self, email: str, cutoff: datetime
    ) -> List[Dict[str, Any]]:
        signals = []
        with httpx.Client(timeout=30) as client:
            # Get contact ID
            contact_resp = client.get(
                f"https://api.hubapi.com/crm/v3/objects/contacts/{email}",
                params={"idProperty": "email"},
                headers={"Authorization": f"Bearer {self.hubspot_key}"},
            )
            if contact_resp.status_code != 200:
                return signals

            contact_id = contact_resp.json()["id"]

            # Fetch engagement timeline
            timeline_resp = client.get(
                f"https://api.hubapi.com/crm/v3/objects/contacts/{contact_id}/associations/engagements",
                headers={"Authorization": f"Bearer {self.hubspot_key}"},
            )
            if timeline_resp.status_code == 200:
                for engagement in timeline_resp.json().get("results", []):
                    eng_type = engagement.get("type", "unknown")
                    timestamp = datetime.fromisoformat(
                        engagement.get("timestamp", "").replace("Z", "+00:00")
                    )
                    if timestamp >= cutoff:
                        signals.append({
                            "source": "hubspot",
                            "type": self._map_hubspot_type(eng_type),
                            "timestamp": timestamp.isoformat(),
                            "details": engagement.get("metadata", {}),
                            "weight": self._get_signal_weight(eng_type),
                        })

        return signals

    def _fetch_redis_signals(
        self, email: str, cutoff: datetime
    ) -> List[Dict[str, Any]]:
        signals = []
        stream_key = f"signals:{email.lower()}"
        cutoff_ms = int(cutoff.timestamp() * 1000)

        # Read from Redis stream
        entries = self.redis_client.xrange(
            stream_key, min=cutoff_ms, max="+", count=500
        )
        for entry_id, data in entries:
            signals.append({
                "source": "website",
                "type": data.get(b"type", b"page_view").decode(),
                "timestamp": datetime.fromtimestamp(int(entry_id.decode().split("-")[0]) / 1000).isoformat(),
                "details": json.loads(data.get(b"details", b"{}").decode()),
                "weight": int(data.get(b"weight", b"1").decode()),
            })

        return signals

    def _map_hubspot_type(self, hubspot_type: str) -> str:
        mapping = {
            "EMAIL": "email_sent",
            "EMAIL_OPEN": "email_open",
            "EMAIL_CLICK": "email_click",
            "FORM_SUBMISSION": "form_submission",
            "MEETING": "meeting_scheduled",
            "CALL": "call",
            "NOTE": "note",
        }
        return mapping.get(hubspot_type, "other")

    def _get_signal_weight(self, signal_type: str) -> int:
        weights = {
            "form_submission": 15,
            "meeting_scheduled": 25,
            "email_click": 8,
            "email_open": 2,
            "call": 20,
            "page_view_pricing": 18,
            "page_view_demo": 15,
            "page_view": 3,
            "content_download": 10,
        }
        return weights.get(signal_type, 1)

    def _calculate_aggregates(self, signals: List[Dict]) -> Dict[str, int]:
        agg = {}
        for s in signals:
            agg[s["type"]] = agg.get(s["type"], 0) + 1
        return agg

    def _calculate_recency_score(self, signals: List[Dict]) -> float:
        if not signals:
            return 0.0
        latest = datetime.fromisoformat(signals[0]["timestamp"].replace("Z", "+00:00"))
        hours_ago = (datetime.utcnow() - latest.replace(tzinfo=None)).total_seconds() / 3600
        if hours_ago <= 1:
            return 100.0
        elif hours_ago <= 24:
            return 80.0
        elif hours_ago <= 72:
            return 50.0
        elif hours_ago <= 168:
            return 30.0
        return 10.0

    def _calculate_velocity_score(self, signals: List[Dict]) -> float:
        if len(signals) < 2:
            return 0.0
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_signals = [
            s for s in signals
            if datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None) >= week_ago
        ]
        # More signals in recent week = higher velocity
        return min(len(recent_signals) * 10, 100.0)`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Lead Analyzer and ML Model agents that score leads in real-time with SHAP explanations and continuously improve model accuracy.',
            toolsUsed: ['Scikit-learn', 'SHAP', 'MLflow', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Lead Scoring ML Model and Analyzer Tools',
                description:
                  'Tools for real-time ML scoring with explainability and model performance monitoring.',
                code: `"""
Lead Scoring ML Model — Training and Real-Time Inference
"""

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import shap
import mlflow
import redis
import json
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve


class LeadScoringInput(BaseModel):
    lead_email: str = Field(description="Email of lead to score")
    signals: Dict[str, Any] = Field(description="Aggregated engagement signals")
    firmographics: Dict[str, Any] = Field(description="Company and contact attributes")


class RealTimeLeadScorerTool(BaseTool):
    name: str = "score_lead"
    description: str = """Score a lead in real-time using the ML model.
        Returns conversion probability, category, and SHAP-based explanation."""
    args_schema: type[BaseModel] = LeadScoringInput

    def __init__(self):
        super().__init__()
        self.redis_client = redis.from_url(os.getenv("REDIS_URL"))
        self.model = self._load_model()
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = self._get_feature_names()

    def _load_model(self):
        model_path = os.getenv("MODEL_PATH", "/app/models/lead_scorer.pkl")
        return joblib.load(model_path)

    def _get_feature_names(self) -> List[str]:
        return [
            "total_signals", "recency_score", "velocity_score",
            "email_opens", "email_clicks", "page_views_pricing",
            "page_views_demo", "form_submissions", "content_downloads",
            "employee_count_bucket", "industry_score", "title_score",
            "days_since_created", "total_sessions", "avg_session_duration",
        ]

    def _run(
        self,
        lead_email: str,
        signals: Dict[str, Any],
        firmographics: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Build feature vector
        features = self._extract_features(signals, firmographics)
        feature_array = np.array([features])

        # Get prediction
        probability = float(self.model.predict_proba(feature_array)[0][1])

        # Calculate SHAP values for explanation
        shap_values = self.explainer.shap_values(feature_array)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification

        # Generate explanation
        explanation = self._generate_explanation(features, shap_values[0])

        # Determine category
        if probability >= 0.75:
            category = "HOT"
            urgency = "IMMEDIATE"
        elif probability >= 0.45:
            category = "WARM"
            urgency = "SAME_DAY"
        elif probability >= 0.25:
            category = "COOL"
            urgency = "NEXT_DAY"
        else:
            category = "COLD"
            urgency = "NURTURE"

        # Calculate engagement and fit scores
        engagement_score = min(100, (
            signals.get("recency_score", 0) * 0.4 +
            signals.get("velocity_score", 0) * 0.3 +
            min(signals.get("signal_count", 0) * 2, 30)
        ))
        fit_score = self._calculate_fit_score(firmographics)

        # Cache the score
        self._cache_score(lead_email, probability, category)

        return {
            "lead_email": lead_email,
            "scores": {
                "conversion_probability": round(probability, 4),
                "engagement_score": round(engagement_score, 1),
                "fit_score": round(fit_score, 1),
                "combined_score": round(engagement_score + fit_score, 1),
            },
            "category": category,
            "urgency": urgency,
            "explanation": explanation,
            "top_positive_factors": [
                f for f in explanation["factors"] if f["impact"] == "positive"
            ][:3],
            "top_negative_factors": [
                f for f in explanation["factors"] if f["impact"] == "negative"
            ][:2],
            "model_version": os.getenv("MODEL_VERSION", "v1.0"),
            "scored_at": datetime.utcnow().isoformat(),
        }

    def _extract_features(
        self, signals: Dict, firmographics: Dict
    ) -> List[float]:
        aggregates = signals.get("aggregates", {})
        return [
            signals.get("signal_count", 0),
            signals.get("recency_score", 0),
            signals.get("velocity_score", 0),
            aggregates.get("email_open", 0),
            aggregates.get("email_click", 0),
            aggregates.get("page_view_pricing", 0),
            aggregates.get("page_view_demo", 0),
            aggregates.get("form_submission", 0),
            aggregates.get("content_download", 0),
            self._bucket_employee_count(firmographics.get("employee_count", 0)),
            self._score_industry(firmographics.get("industry", "")),
            self._score_title(firmographics.get("job_title", "")),
            firmographics.get("days_since_created", 0),
            aggregates.get("sessions", 1),
            signals.get("avg_session_duration", 0),
        ]

    def _bucket_employee_count(self, count: int) -> int:
        if count <= 50:
            return 1
        elif count <= 200:
            return 2
        elif count <= 1000:
            return 3
        elif count <= 5000:
            return 4
        return 5

    def _score_industry(self, industry: str) -> float:
        high_value = ["technology", "finance", "healthcare", "saas"]
        medium_value = ["manufacturing", "retail", "professional_services"]
        industry_lower = industry.lower()
        if any(i in industry_lower for i in high_value):
            return 1.0
        elif any(i in industry_lower for i in medium_value):
            return 0.6
        return 0.3

    def _score_title(self, title: str) -> float:
        executive = ["ceo", "cfo", "cto", "coo", "president", "founder"]
        director = ["director", "vp", "head of", "chief"]
        manager = ["manager", "lead", "senior"]
        title_lower = title.lower()
        if any(t in title_lower for t in executive):
            return 1.0
        elif any(t in title_lower for t in director):
            return 0.8
        elif any(t in title_lower for t in manager):
            return 0.5
        return 0.2

    def _calculate_fit_score(self, firmographics: Dict) -> float:
        industry_score = self._score_industry(firmographics.get("industry", "")) * 40
        title_score = self._score_title(firmographics.get("job_title", "")) * 35
        size_score = min(self._bucket_employee_count(firmographics.get("employee_count", 0)) * 5, 25)
        return industry_score + title_score + size_score

    def _generate_explanation(
        self, features: List[float], shap_values: np.ndarray
    ) -> Dict[str, Any]:
        factors = []
        for i, (name, value, shap_val) in enumerate(
            zip(self.feature_names, features, shap_values)
        ):
            if abs(shap_val) > 0.01:  # Only significant factors
                factors.append({
                    "feature": name,
                    "value": value,
                    "shap_value": round(float(shap_val), 4),
                    "impact": "positive" if shap_val > 0 else "negative",
                    "description": self._describe_factor(name, value, shap_val),
                })

        factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return {
            "base_probability": round(float(self.explainer.expected_value), 4),
            "factors": factors,
        }

    def _describe_factor(self, name: str, value: float, shap: float) -> str:
        impact = "increases" if shap > 0 else "decreases"
        descriptions = {
            "recency_score": f"Recent activity (score: {value:.0f}) {impact} likelihood",
            "velocity_score": f"Activity velocity (score: {value:.0f}) {impact} likelihood",
            "page_views_pricing": f"Pricing page views ({value:.0f}) {impact} likelihood",
            "page_views_demo": f"Demo page views ({value:.0f}) {impact} likelihood",
            "email_clicks": f"Email clicks ({value:.0f}) {impact} likelihood",
            "form_submissions": f"Form submissions ({value:.0f}) {impact} likelihood",
            "title_score": f"Job title seniority {impact} likelihood",
            "industry_score": f"Industry fit {impact} likelihood",
        }
        return descriptions.get(name, f"{name} = {value:.1f} {impact} likelihood")

    def _cache_score(self, email: str, probability: float, category: str) -> None:
        cache_data = {
            "probability": probability,
            "category": category,
            "scored_at": datetime.utcnow().isoformat(),
        }
        self.redis_client.setex(
            f"lead_score:{email.lower()}",
            3600,  # 1 hour TTL
            json.dumps(cache_data),
        )`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement LangGraph state machine for the real-time lead scoring pipeline with sub-second latency targets and intelligent routing logic.',
            toolsUsed: ['LangGraph', 'Redis', 'Kafka'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Lead Scoring Orchestrator',
                description:
                  'Real-time workflow orchestration for lead scoring with intelligent routing.',
                code: `"""
Lead Scoring — LangGraph Real-Time Orchestration
Sequential workflow optimized for sub-second scoring latency.
"""

from typing import TypedDict, Annotated, Literal
from datetime import datetime
import operator
import json
import redis
import asyncio

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage


class LeadScoringState(TypedDict):
    """State for real-time lead scoring workflow."""
    lead_email: str
    signals: dict
    firmographics: dict
    score_result: dict
    routing_decision: dict
    notifications_sent: list
    latency_ms: float
    workflow_status: str


class LeadScoringOrchestrator:
    def __init__(
        self,
        signal_agent,
        scorer_agent,
        router_agent,
        redis_url: str,
    ):
        self.signal_agent = signal_agent
        self.scorer_agent = scorer_agent
        self.router_agent = router_agent
        self.redis = redis.from_url(redis_url)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the lead scoring workflow graph."""

        async def collect_signals(state: LeadScoringState) -> dict:
            """Collect engagement signals for the lead."""
            start = datetime.utcnow()
            try:
                result = await self.signal_agent.ainvoke({
                    "input": f"Collect all engagement signals for {state['lead_email']}",
                    "lead_email": state["lead_email"],
                })
                return {
                    "signals": result,
                    "workflow_status": "signals_collected",
                    "latency_ms": (datetime.utcnow() - start).total_seconds() * 1000,
                }
            except Exception as e:
                return {"workflow_status": f"signal_error: {e}"}

        async def score_lead(state: LeadScoringState) -> dict:
            """Score the lead using ML model."""
            start = datetime.utcnow()
            try:
                result = await self.scorer_agent.ainvoke({
                    "input": "Score this lead with full explanation",
                    "lead_email": state["lead_email"],
                    "signals": state["signals"],
                    "firmographics": state.get("firmographics", {}),
                })
                return {
                    "score_result": result,
                    "workflow_status": "scored",
                    "latency_ms": state.get("latency_ms", 0) + (datetime.utcnow() - start).total_seconds() * 1000,
                }
            except Exception as e:
                return {"workflow_status": f"scoring_error: {e}"}

        async def route_lead(state: LeadScoringState) -> dict:
            """Route the lead to the optimal rep."""
            start = datetime.utcnow()
            try:
                score_result = state.get("score_result", {})
                if score_result.get("category") in ["HOT", "WARM"]:
                    result = await self.router_agent.ainvoke({
                        "input": "Find the best available rep for this lead",
                        "lead_email": state["lead_email"],
                        "score": score_result.get("scores", {}),
                        "category": score_result.get("category"),
                        "firmographics": state.get("firmographics", {}),
                    })
                    return {
                        "routing_decision": result,
                        "workflow_status": "routed",
                        "latency_ms": state.get("latency_ms", 0) + (datetime.utcnow() - start).total_seconds() * 1000,
                    }
                else:
                    return {
                        "routing_decision": {
                            "action": "nurture",
                            "sequence": "general_nurture_v2",
                        },
                        "workflow_status": "routed_to_nurture",
                    }
            except Exception as e:
                return {"workflow_status": f"routing_error: {e}"}

        async def send_notifications(state: LeadScoringState) -> dict:
            """Send notifications based on routing decision."""
            notifications = []
            routing = state.get("routing_decision", {})
            score = state.get("score_result", {})

            if score.get("category") == "HOT":
                # Send immediate Slack notification
                notifications.append({
                    "channel": "slack",
                    "target": routing.get("assigned_rep_slack", "#hot-leads"),
                    "sent_at": datetime.utcnow().isoformat(),
                })

            if routing.get("action") != "nurture":
                # Update CRM
                notifications.append({
                    "channel": "crm_update",
                    "lead_email": state["lead_email"],
                    "sent_at": datetime.utcnow().isoformat(),
                })

            return {
                "notifications_sent": notifications,
                "workflow_status": "complete",
            }

        def should_route_to_rep(state: LeadScoringState) -> Literal["route", "nurture"]:
            """Determine if lead should be routed to rep or nurture."""
            category = state.get("score_result", {}).get("category", "COLD")
            return "route" if category in ["HOT", "WARM"] else "nurture"

        # Build graph
        workflow = StateGraph(LeadScoringState)

        workflow.add_node("collect_signals", collect_signals)
        workflow.add_node("score", score_lead)
        workflow.add_node("route", route_lead)
        workflow.add_node("notify", send_notifications)

        workflow.set_entry_point("collect_signals")
        workflow.add_edge("collect_signals", "score")
        workflow.add_conditional_edges(
            "score",
            should_route_to_rep,
            {"route": "route", "nurture": "notify"},
        )
        workflow.add_edge("route", "notify")
        workflow.add_edge("notify", END)

        return workflow.compile()

    async def score_lead(
        self,
        lead_email: str,
        firmographics: dict = None,
    ) -> dict:
        """Score a lead end-to-end with latency tracking."""
        start_time = datetime.utcnow()

        result = await self.graph.ainvoke({
            "lead_email": lead_email,
            "signals": {},
            "firmographics": firmographics or {},
            "score_result": {},
            "routing_decision": {},
            "notifications_sent": [],
            "latency_ms": 0,
            "workflow_status": "started",
        })

        total_latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        result["total_latency_ms"] = total_latency

        # Log latency for monitoring
        self.redis.lpush(
            "lead_scoring_latencies",
            json.dumps({
                "email": lead_email,
                "latency_ms": total_latency,
                "category": result.get("score_result", {}).get("category"),
                "timestamp": datetime.utcnow().isoformat(),
            })
        )
        self.redis.ltrim("lead_scoring_latencies", 0, 9999)

        return result`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the real-time scoring system with Docker, configure MLflow for model tracking, and set up comprehensive monitoring for latency and model performance.',
            toolsUsed: ['Docker', 'MLflow', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose for Lead Scoring System',
                description:
                  'Production deployment with real-time scoring API, ML model serving, and monitoring stack.',
                code: `version: '3.8'

services:
  lead-scoring-api:
    build:
      context: .
      dockerfile: Dockerfile.scoring
    container_name: lead-scoring-api
    restart: unless-stopped
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - HUBSPOT_API_KEY=\${HUBSPOT_API_KEY}
      - REDIS_URL=redis://redis:6379/0
      - MODEL_PATH=/models/lead_scorer.pkl
      - MODEL_VERSION=\${MODEL_VERSION:-v1.0}
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - LOG_LEVEL=INFO
    ports:
      - "8080:8080"
    depends_on:
      - redis
      - mlflow
    volumes:
      - ./models:/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  redis:
    image: redis:7-alpine
    container_name: lead-scoring-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.0
    container_name: lead-scoring-mlflow
    restart: unless-stopped
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://\${POSTGRES_USER}:\${POSTGRES_PASSWORD}@postgres:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    ports:
      - "5000:5000"
    depends_on:
      - postgres
    command: mlflow server --host 0.0.0.0 --port 5000

  postgres:
    image: postgres:15-alpine
    container_name: lead-scoring-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=\${POSTGRES_USER}
      - POSTGRES_PASSWORD=\${POSTGRES_PASSWORD}
      - POSTGRES_DB=mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data

  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: lead-scoring-prometheus
    restart: unless-stopped
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:10.0.0
    container_name: lead-scoring-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_ADMIN_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"

volumes:
  redis_data:
  mlflow_artifacts:
  postgres_data:
  prometheus_data:
  grafana_data:`,
              },
              {
                language: 'python',
                title: 'Real-Time Scoring API with Latency Monitoring',
                description:
                  'FastAPI service for real-time lead scoring with comprehensive metrics.',
                code: `"""
Lead Scoring API — Real-Time Inference with Monitoring
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
from datetime import datetime
import asyncio
import os

app = FastAPI(title="Real-Time Lead Scoring API")

# Prometheus metrics
SCORING_REQUESTS = Counter(
    "lead_scoring_requests_total",
    "Total scoring requests",
    ["category", "status"],
)
SCORING_LATENCY = Histogram(
    "lead_scoring_latency_seconds",
    "Scoring latency in seconds",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)
MODEL_ACCURACY = Gauge(
    "lead_scoring_model_accuracy",
    "Model accuracy from last evaluation",
)
HOT_LEADS_ROUTED = Counter(
    "lead_scoring_hot_leads_routed_total",
    "Total hot leads routed to reps",
)
P95_LATENCY = Gauge(
    "lead_scoring_p95_latency_seconds",
    "95th percentile latency",
)


class ScoringRequest(BaseModel):
    lead_email: str
    firmographics: dict = {}
    async_mode: bool = False


class ScoringResponse(BaseModel):
    lead_email: str
    conversion_probability: float
    category: str
    urgency: str
    explanation: dict
    routing: dict
    latency_ms: float


@app.post("/score", response_model=ScoringResponse)
async def score_lead(request: ScoringRequest, background_tasks: BackgroundTasks):
    """Score a lead in real-time."""
    start_time = datetime.utcnow()

    try:
        from orchestrator import LeadScoringOrchestrator
        orchestrator = get_orchestrator()

        if request.async_mode:
            # Fire and forget for batch processing
            background_tasks.add_task(
                orchestrator.score_lead,
                request.lead_email,
                request.firmographics,
            )
            return ScoringResponse(
                lead_email=request.lead_email,
                conversion_probability=0,
                category="PENDING",
                urgency="ASYNC",
                explanation={},
                routing={"status": "queued"},
                latency_ms=0,
            )

        result = await orchestrator.score_lead(
            request.lead_email,
            request.firmographics,
        )

        latency_seconds = (datetime.utcnow() - start_time).total_seconds()
        category = result.get("score_result", {}).get("category", "UNKNOWN")

        # Record metrics
        SCORING_REQUESTS.labels(category=category, status="success").inc()
        SCORING_LATENCY.observe(latency_seconds)

        if category == "HOT":
            HOT_LEADS_ROUTED.inc()

        return ScoringResponse(
            lead_email=request.lead_email,
            conversion_probability=result.get("score_result", {}).get("scores", {}).get("conversion_probability", 0),
            category=category,
            urgency=result.get("score_result", {}).get("urgency", "UNKNOWN"),
            explanation=result.get("score_result", {}).get("explanation", {}),
            routing=result.get("routing_decision", {}),
            latency_ms=latency_seconds * 1000,
        )

    except Exception as e:
        latency_seconds = (datetime.utcnow() - start_time).total_seconds()
        SCORING_REQUESTS.labels(category="ERROR", status="failed").inc()
        SCORING_LATENCY.observe(latency_seconds)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "model_version": os.getenv("MODEL_VERSION")}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")


_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from orchestrator import LeadScoringOrchestrator
        from agents import create_scoring_agents
        agents = create_scoring_agents()
        _orchestrator = LeadScoringOrchestrator(
            signal_agent=agents["signal"],
            scorer_agent=agents["scorer"],
            router_agent=agents["router"],
            redis_url=os.getenv("REDIS_URL"),
        )
    return _orchestrator`,
              },
            ],
          },
        ],
      },
    },

    // ── Pain Point 3: WhatsApp Relationship Gap ─────────────────────────
    {
      id: 'whatsapp-relationship-gap',
      number: 3,
      title: 'The WhatsApp Relationship Gap',
      subtitle: 'Untracked Conversational Sales Data',
      summary:
        'Half your deals close over WhatsApp, but none of that context makes it into your CRM. Critical relationship data lives in personal phones.',
      tags: ['whatsapp', 'crm-integration', 'relationship-data'],
      metrics: {
        annualCostRange: '$300K - $1.5M',
        roi: '5x',
        paybackPeriod: '4-5 months',
        investmentRange: '$80K - $150K',
      },
      price: {
        present: {
          title: 'Present Reality',
          severity: 'high',
          description:
            'Sales reps conduct 40-60% of client communication over WhatsApp, yet zero conversation data flows into the CRM. When a rep leaves, entire relationship histories walk out the door.',
          bullets: [
            'WhatsApp is the primary sales channel in 50%+ of deals but completely untracked',
            'Deal context, pricing discussions, and verbal commitments exist only on personal phones',
            'When reps churn, successor reps start from scratch with no conversation history',
            'Managers cannot coach on conversational selling because they have no visibility',
            'Compliance and audit teams cannot review client communications on WhatsApp',
          ],
        },
        root: {
          title: 'Root Cause',
          severity: 'high',
          description:
            'The organization has not adopted WhatsApp Business API and has no integration layer between messaging platforms and the CRM.',
          bullets: [
            'WhatsApp Business API is not configured or connected to any system',
            'Reps use personal WhatsApp accounts with no company oversight',
            'No middleware exists to capture, parse, and route messages to the CRM',
            'CRM has no data model for conversational interaction logging',
            'IT/Ops teams are unaware of how much revenue flows through WhatsApp',
          ],
        },
        impact: {
          title: 'Business Impact',
          severity: 'critical',
          description:
            'Institutional relationship knowledge is fragile and siloed. Deal context evaporates, handoffs fail, and customer experience suffers.',
          bullets: [
            'Rep turnover causes 15-25% pipeline disruption as relationships are lost',
            'Deal handoffs take 3x longer because successors lack conversation context',
            'Cross-sell opportunities are missed because interaction history is invisible',
            'Customer complaints about repeating themselves to new reps increase churn',
            'Competitive intelligence shared in chats never reaches strategy teams',
          ],
        },
        cost: {
          title: 'Financial Cost',
          severity: 'high',
          description:
            'Revenue loss from broken handoffs, missed context, and unrecoverable relationship data adds up across the sales organization.',
          bullets: [
            '$100K-$500K in pipeline value lost per departing senior rep',
            '$50K-$200K annually in extended ramp time for replacement reps',
            '$80K-$300K in missed cross-sell/upsell from invisible conversation signals',
            '$30K-$100K in compliance risk from unmonitored client communications',
            'Unmeasurable loss of competitive intelligence shared informally in chats',
          ],
        },
        expectedReturn: {
          title: 'Expected Return',
          severity: 'high',
          description:
            'Integrating WhatsApp into the CRM preserves relationship data, accelerates handoffs, and surfaces hidden revenue signals from conversational data.',
          bullets: [
            'Full conversation history persisted in CRM, surviving rep transitions',
            'Deal handoff time reduced by 60% with complete context transfer',
            'Cross-sell signals extracted from chat patterns increase upsell rate by 20%',
            'Compliance team gains audit trail over all client communications',
            'Conversational analytics reveal winning messaging patterns for coaching',
          ],
        },
      },
      implementation: {
        overview:
          'Integrate WhatsApp Business API with your CRM through a Python middleware that captures messages, extracts deal signals using NLP, and maps conversations to CRM records.',
        prerequisites: [
          'WhatsApp Business API account (Meta-approved)',
          'CRM with API access and custom object support',
          'Python 3.9+ with httpx, spaCy, and SQLAlchemy',
          'PostgreSQL or equivalent database for message storage',
          'pytest >= 7.0 with pytest-asyncio',
          'Docker and docker-compose for containerized deployment',
          'cron, Airflow, or Prefect for job scheduling',
          'Slack incoming webhook URL for operational alerts',
        ],
        toolsUsed: ['Python', 'WhatsApp Business API', 'SQL', 'spaCy', 'pytest', 'Docker', 'GitHub Actions', 'cron', 'Slack API', 'Prometheus'],
        steps: [
          {
            stepNumber: 1,
            title: 'WhatsApp Business API Integration',
            description:
              'Build a webhook receiver that captures incoming and outgoing WhatsApp messages, normalizes them, and persists them to a conversation store linked to CRM contacts.',
            codeSnippets: [
              {
                language: 'python',
                title: 'WhatsApp Webhook Receiver',
                description:
                  'FastAPI webhook that receives WhatsApp Business API callbacks, validates them, extracts message data, and stores conversations with CRM contact linkage.',
                code: `"""
WhatsApp Business API Webhook Receiver
Captures messages, matches to CRM contacts, and persists to database.
"""

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from datetime import datetime
import hashlib
import hmac
import httpx
from sqlalchemy import create_engine, text

app = FastAPI(title="WhatsApp CRM Bridge")
engine = create_engine("postgresql://user:pass@localhost/whatsapp_crm")

VERIFY_TOKEN = "whatsapp_verify_token_2024"
APP_SECRET = "whatsapp_app_secret"
CRM_API = "https://api.crm.example.com/v2"


def verify_signature(payload: bytes, signature: str) -> bool:
    expected = hmac.new(
        APP_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


@app.get("/webhook")
async def verify_webhook(mode: str = "", token: str = "", challenge: str = ""):
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return int(challenge)
    raise HTTPException(403, "Verification failed")


@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.body()
    signature = request.headers.get("x-hub-signature-256", "")
    if not verify_signature(body, signature):
        raise HTTPException(403, "Invalid signature")

    data = await request.json()

    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])
            for msg in messages:
                await process_message(
                    phone=msg["from"],
                    text=msg.get("text", {}).get("body", ""),
                    timestamp=msg["timestamp"],
                    msg_id=msg["id"],
                    direction="inbound",
                )
    return {"status": "ok"}


async def process_message(
    phone: str, text: str, timestamp: str, msg_id: str, direction: str
):
    contact_id = await match_crm_contact(phone)
    with engine.connect() as conn:
        conn.execute(
            text(\"\"\"
                INSERT INTO whatsapp_messages
                  (message_id, phone, contact_id, direction, body, received_at)
                VALUES (:msg_id, :phone, :cid, :dir, :body, :ts)
                ON CONFLICT (message_id) DO NOTHING
            \"\"\"),
            {
                "msg_id": msg_id, "phone": phone, "cid": contact_id,
                "dir": direction, "body": text,
                "ts": datetime.fromtimestamp(int(timestamp)),
            },
        )
        conn.commit()


async def match_crm_contact(phone: str) -> str | None:
    normalized = phone.lstrip("+").replace(" ", "").replace("-", "")
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{CRM_API}/contacts/search",
            params={"phone": normalized},
            headers={"Authorization": "Bearer $CRM_API_KEY"},
            timeout=10,
        )
        results = resp.json().get("results", [])
        return results[0]["id"] if results else None`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Conversation-to-CRM Mapping',
            description:
              'Build SQL views and Python NLP pipelines that extract deal signals from WhatsApp conversations and link them to CRM opportunities as activity records.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Conversation-to-Opportunity Mapping',
                description:
                  'Links WhatsApp conversations to CRM opportunities and surfaces key conversation metrics for each deal.',
                code: `-- WhatsApp Conversation <-> CRM Opportunity Mapping
-- Links message threads to open deals for relationship context

CREATE OR REPLACE VIEW vw_deal_conversations AS
WITH conversation_threads AS (
  SELECT
    wm.contact_id,
    c.full_name                             AS contact_name,
    c.company,
    COUNT(*)                                AS total_messages,
    SUM(CASE WHEN wm.direction = 'inbound'  THEN 1 ELSE 0 END) AS inbound_msgs,
    SUM(CASE WHEN wm.direction = 'outbound' THEN 1 ELSE 0 END) AS outbound_msgs,
    MIN(wm.received_at)                     AS first_message,
    MAX(wm.received_at)                     AS last_message,
    EXTRACT(EPOCH FROM MAX(wm.received_at) - MIN(wm.received_at)) / 86400
      AS conversation_span_days,
    ROUND(
      SUM(CASE WHEN wm.direction = 'inbound' THEN 1 ELSE 0 END)::numeric
      / NULLIF(COUNT(*), 0), 2
    ) AS engagement_ratio
  FROM whatsapp_messages wm
  JOIN crm_contacts c ON wm.contact_id = c.id
  WHERE wm.contact_id IS NOT NULL
  GROUP BY wm.contact_id, c.full_name, c.company
),
deal_links AS (
  SELECT
    ocr.contact_id,
    o.id          AS opportunity_id,
    o.name        AS deal_name,
    o.stage_name,
    o.amount,
    o.close_date
  FROM opportunity_contact_roles ocr
  JOIN opportunities o ON ocr.opportunity_id = o.id
  WHERE o.is_closed = FALSE
)
SELECT
  ct.contact_name,
  ct.company,
  dl.deal_name,
  dl.stage_name,
  dl.amount,
  ct.total_messages,
  ct.engagement_ratio,
  ct.conversation_span_days,
  ct.last_message,
  CASE
    WHEN ct.last_message >= NOW() - INTERVAL '3 days'  THEN 'active'
    WHEN ct.last_message >= NOW() - INTERVAL '14 days' THEN 'cooling'
    ELSE 'dormant'
  END AS conversation_status
FROM conversation_threads ct
JOIN deal_links dl ON ct.contact_id = dl.contact_id
ORDER BY ct.last_message DESC;`,
              },
              {
                language: 'python',
                title: 'Deal Signal Extraction with NLP',
                description:
                  'Uses spaCy NLP to extract deal-relevant signals from WhatsApp messages — pricing mentions, competitor references, urgency indicators, and commitment language.',
                code: `"""
WhatsApp Deal Signal Extractor
Parses conversation text to identify deal-relevant signals
and logs them as CRM activity records.
"""

import re
import spacy
from dataclasses import dataclass, field
from datetime import datetime

nlp = spacy.load("en_core_web_sm")

PRICING_PATTERNS = [
    r"\\$[\\d,]+(?:\\.\\d{2})?",
    r"\\b\\d+[kK]\\b",
    r"(?:price|cost|budget|quote|invoice|discount|offer)\\w*",
]
URGENCY_PATTERNS = [
    r"\\b(?:asap|urgent|deadline|by (?:monday|tuesday|wednesday|thursday|friday)|end of (?:week|month|quarter))\\b",
    r"\\b(?:running out of time|need this done|cannot wait)\\b",
]
COMPETITOR_KEYWORDS = [
    "competitor_a", "competitor_b", "competitor_c",
    "alternative", "other vendor", "looking at options",
]
COMMITMENT_PATTERNS = [
    r"\\b(?:let'?s? (?:go ahead|proceed|do it|move forward|sign))\\b",
    r"\\b(?:approved|greenlight|ready to start|confirmed)\\b",
]


@dataclass
class DealSignal:
    signal_type: str
    confidence: float
    excerpt: str
    detected_at: datetime = field(default_factory=datetime.utcnow)


def extract_signals(message_text: str) -> list[DealSignal]:
    signals: list[DealSignal] = []
    text_lower = message_text.lower()

    for pattern in PRICING_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            signals.append(DealSignal(
                signal_type="pricing_mention",
                confidence=0.85,
                excerpt=matches[0],
            ))

    for pattern in URGENCY_PATTERNS:
        if re.search(pattern, text_lower):
            signals.append(DealSignal(
                signal_type="urgency_indicator",
                confidence=0.80,
                excerpt=re.search(pattern, text_lower).group(),
            ))

    for keyword in COMPETITOR_KEYWORDS:
        if keyword in text_lower:
            signals.append(DealSignal(
                signal_type="competitor_mention",
                confidence=0.75,
                excerpt=keyword,
            ))

    for pattern in COMMITMENT_PATTERNS:
        if re.search(pattern, text_lower):
            signals.append(DealSignal(
                signal_type="commitment_language",
                confidence=0.90,
                excerpt=re.search(pattern, text_lower).group(),
            ))

    doc = nlp(message_text)
    for ent in doc.ents:
        if ent.label_ == "DATE":
            signals.append(DealSignal(
                signal_type="timeline_reference",
                confidence=0.70,
                excerpt=ent.text,
            ))
        elif ent.label_ == "MONEY":
            signals.append(DealSignal(
                signal_type="budget_reference",
                confidence=0.88,
                excerpt=ent.text,
            ))

    return signals`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate WhatsApp message parsing accuracy, CRM contact matching logic, and deal signal extraction with SQL data quality assertions and pytest-based integration tests.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'WhatsApp Data Quality Assertion Queries',
                description:
                  'Automated checks for message ingestion completeness, CRM contact match rates, referential integrity between messages and contacts, and sync freshness.',
                code: `-- WhatsApp Data Quality Assertions
-- Run before each sync cycle to validate data pipeline health

-- 1. Row count: messages should be ingesting continuously
SELECT
  CASE
    WHEN COUNT(*) < 10
    THEN 'FAIL: only ' || COUNT(*) || ' messages in last 24h (expected 10+)'
    ELSE 'PASS: ' || COUNT(*) || ' messages ingested in last 24h'
  END AS ingestion_check
FROM whatsapp_messages
WHERE received_at >= NOW() - INTERVAL '24 hours';

-- 2. Null check: messages must have phone and body populated
SELECT
  CASE
    WHEN COUNT(*) > 0
    THEN 'FAIL: ' || COUNT(*) || ' messages missing phone or body'
    ELSE 'PASS: all messages have required fields'
  END AS null_check
FROM whatsapp_messages
WHERE (phone IS NULL OR phone = '' OR body IS NULL)
  AND received_at >= NOW() - INTERVAL '7 days';

-- 3. CRM contact match rate should be above 70%
SELECT
  CASE
    WHEN matched_pct < 70.0
    THEN 'FAIL: contact match rate is ' || matched_pct || '% (threshold 70%)'
    ELSE 'PASS: contact match rate is ' || matched_pct || '%'
  END AS match_rate_check
FROM (
  SELECT ROUND(
    100.0 * COUNT(*) FILTER (WHERE contact_id IS NOT NULL)
    / NULLIF(COUNT(*), 0), 1
  ) AS matched_pct
  FROM whatsapp_messages
  WHERE received_at >= NOW() - INTERVAL '30 days'
) sub;

-- 4. Referential integrity: contact_id must exist in CRM
SELECT
  CASE
    WHEN COUNT(*) > 0
    THEN 'FAIL: ' || COUNT(*) || ' messages reference non-existent CRM contacts'
    ELSE 'PASS'
  END AS ref_integrity_check
FROM whatsapp_messages wm
LEFT JOIN crm_contacts c ON wm.contact_id = c.id
WHERE wm.contact_id IS NOT NULL AND c.id IS NULL
  AND wm.received_at >= NOW() - INTERVAL '7 days';

-- 5. Sync freshness: last sync should be within 30 minutes
SELECT
  CASE
    WHEN MAX(synced_at) < NOW() - INTERVAL '30 minutes'
    THEN 'FAIL: last CRM sync was ' ||
         EXTRACT(MINUTE FROM NOW() - MAX(synced_at)) || ' minutes ago'
    ELSE 'PASS: sync is current'
  END AS freshness_check
FROM whatsapp_messages
WHERE synced_to_crm = TRUE;`,
              },
              {
                language: 'python',
                title: 'WhatsApp Pipeline Validation Tests',
                description:
                  'pytest suite that validates message parsing accuracy, CRM contact matching, deal signal extraction, and webhook signature verification.',
                code: `"""
WhatsApp CRM Bridge — Validation Test Suite
Tests message parsing, contact matching, signal extraction, and webhook security.
"""

import pytest
import hashlib
import hmac
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock

from whatsapp_webhook import (
    verify_signature,
    process_message,
    match_crm_contact,
    APP_SECRET,
)
from deal_signal_extractor import extract_signals, DealSignal


class TestSignatureVerification:
    def test_valid_signature_passes(self):
        payload = b'{"test": "data"}'
        expected = hmac.new(
            APP_SECRET.encode(), payload, hashlib.sha256
        ).hexdigest()
        assert verify_signature(payload, f"sha256={expected}") is True

    def test_invalid_signature_fails(self):
        assert verify_signature(b"data", "sha256=invalid") is False

    def test_empty_signature_fails(self):
        assert verify_signature(b"data", "") is False


class TestContactMatching:
    @pytest.mark.asyncio
    async def test_normalized_phone_lookup(self):
        with patch("whatsapp_webhook.httpx.AsyncClient") as mock:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "results": [{"id": "contact_001"}]
            }
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            result = await match_crm_contact("+1-555-123-4567")
            call_args = mock_client.get.call_args
            assert call_args[1]["params"]["phone"] == "15551234567"

    @pytest.mark.asyncio
    async def test_unmatched_phone_returns_none(self):
        with patch("whatsapp_webhook.httpx.AsyncClient") as mock:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"results": []}
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            result = await match_crm_contact("+0000000000")
            assert result is None


class TestDealSignalExtraction:
    def test_pricing_mention_detected(self):
        signals = extract_signals("The total cost would be $25,000 for this package")
        types = [s.signal_type for s in signals]
        assert "pricing_mention" in types or "budget_reference" in types

    def test_urgency_detected(self):
        signals = extract_signals("We need this done ASAP before end of quarter")
        types = [s.signal_type for s in signals]
        assert "urgency_indicator" in types

    def test_commitment_language_detected(self):
        signals = extract_signals("Looks good, let's go ahead and proceed")
        types = [s.signal_type for s in signals]
        assert "commitment_language" in types

    def test_competitor_mention_detected(self):
        signals = extract_signals("We are also looking at options from competitor_a")
        types = [s.signal_type for s in signals]
        assert "competitor_mention" in types

    def test_neutral_message_no_signals(self):
        signals = extract_signals("Thanks for the update, talk soon")
        assert len(signals) == 0

    def test_all_signals_have_confidence(self):
        signals = extract_signals("Budget is $50k, need it asap, let's proceed")
        for signal in signals:
            assert 0.0 < signal.confidence <= 1.0
            assert signal.excerpt != ""`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Deploy the WhatsApp webhook service as a container with message queue setup, database migrations, and scheduled CRM sync via a battle-tested deployment script.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'WhatsApp Webhook Service Deployment Script',
                description:
                  'Deploys the WhatsApp CRM bridge container, runs DB migrations, configures message queue, and sets up cron-based sync scheduling.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ── WhatsApp CRM Bridge — Deployment Script ─────────────────────
APP_NAME="whatsapp-crm-bridge"
IMAGE_TAG="\${APP_NAME}:\$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')"
LOG_DIR="/var/log/\${APP_NAME}"
DATA_DIR="/opt/\${APP_NAME}/data"
REQUIRED_VARS=(
  "WHATSAPP_APP_SECRET" "WHATSAPP_VERIFY_TOKEN" "WHATSAPP_ACCESS_TOKEN"
  "CRM_API_KEY" "DATABASE_URL" "SLACK_WEBHOOK_URL"
)

echo "==> Deploying \${APP_NAME} (\${IMAGE_TAG})..."

# 1. Validate required environment variables
for var in "\${REQUIRED_VARS[@]}"; do
  if [[ -z "\${!var:-}" ]]; then
    echo "ERROR: Required env var \${var} is not set." >&2
    exit 1
  fi
done
echo "    Environment validated (\${#REQUIRED_VARS[@]} vars checked)."

# 2. Verify Docker and docker-compose are available
if ! docker info &>/dev/null; then
  echo "ERROR: Docker daemon is not running." >&2
  exit 1
fi
if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null; then
  echo "ERROR: docker-compose not found." >&2
  exit 1
fi
echo "    Docker and docker-compose verified."

# 3. Create data directories
sudo mkdir -p "\${LOG_DIR}" "\${DATA_DIR}"
sudo chown "\$(whoami):" "\${LOG_DIR}" "\${DATA_DIR}"

# 4. Build container image
docker build -t "\${IMAGE_TAG}" -f Dockerfile.whatsapp .
echo "    Container image built: \${IMAGE_TAG}"

# 5. Run database migrations
docker run --rm \\
  -e DATABASE_URL="\${DATABASE_URL}" \\
  "\${IMAGE_TAG}" python migrate.py --apply
echo "    Database migrations applied."

# 6. Stop existing containers and start fresh
docker stop "\${APP_NAME}" "\${APP_NAME}-worker" 2>/dev/null || true
docker rm "\${APP_NAME}" "\${APP_NAME}-worker" 2>/dev/null || true

# 7. Start webhook receiver (public-facing)
docker run -d \\
  --name "\${APP_NAME}" \\
  --restart unless-stopped \\
  -p 443:8443 \\
  -e WHATSAPP_APP_SECRET="\${WHATSAPP_APP_SECRET}" \\
  -e WHATSAPP_VERIFY_TOKEN="\${WHATSAPP_VERIFY_TOKEN}" \\
  -e WHATSAPP_ACCESS_TOKEN="\${WHATSAPP_ACCESS_TOKEN}" \\
  -e DATABASE_URL="\${DATABASE_URL}" \\
  -e CRM_API_KEY="\${CRM_API_KEY}" \\
  "\${IMAGE_TAG}" uvicorn whatsapp_webhook:app --host 0.0.0.0 --port 8443
echo "    Webhook receiver started on port 443."

# 8. Schedule CRM sync every 15 minutes
SYNC_EXPR="*/15 * * * *"
SYNC_CMD="docker exec \${APP_NAME} python sync_to_crm.py >> \${LOG_DIR}/sync.log 2>&1"
( crontab -l 2>/dev/null | grep -v "\${APP_NAME}" ; echo "\${SYNC_EXPR} \${SYNC_CMD}  # \${APP_NAME}-sync" ) | crontab -
echo "    CRM sync cron set: \${SYNC_EXPR}"

# 9. Health check
sleep 3
if curl -sf http://localhost:8443/health > /dev/null; then
  echo "==> Deployment successful. Webhook receiver is healthy."
else
  echo "ERROR: Health check failed. Check logs: docker logs \${APP_NAME}" >&2
  exit 1
fi`,
              },
              {
                language: 'python',
                title: 'WhatsApp Service Configuration Loader',
                description:
                  'Loads WhatsApp API credentials, CRM connection details, and message queue settings from environment variables with validation and connection pooling.',
                code: `"""
WhatsApp CRM Bridge — Configuration Loader
Manages WhatsApp API secrets, CRM credentials, and DB connection pools.
"""

import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool


@dataclass(frozen=True)
class WhatsAppBridgeConfig:
    """Immutable configuration for the WhatsApp CRM bridge service."""
    whatsapp_app_secret: str
    whatsapp_verify_token: str
    whatsapp_access_token: str
    crm_api_key: str
    crm_api_url: str
    database_url: str
    slack_webhook_url: str
    sync_interval_minutes: int = 15
    contact_match_threshold: float = 0.70
    max_message_age_days: int = 90
    webhook_port: int = 8443
    log_level: str = "INFO"


def load_config() -> WhatsAppBridgeConfig:
    required = {
        "WHATSAPP_APP_SECRET": os.getenv("WHATSAPP_APP_SECRET"),
        "WHATSAPP_VERIFY_TOKEN": os.getenv("WHATSAPP_VERIFY_TOKEN"),
        "WHATSAPP_ACCESS_TOKEN": os.getenv("WHATSAPP_ACCESS_TOKEN"),
        "CRM_API_KEY": os.getenv("CRM_API_KEY"),
        "DATABASE_URL": os.getenv("DATABASE_URL"),
        "SLACK_WEBHOOK_URL": os.getenv("SLACK_WEBHOOK_URL"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"FATAL: Missing env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    # Validate URL-type variables
    for key in ("DATABASE_URL", "SLACK_WEBHOOK_URL"):
        parsed = urlparse(required[key])
        if not parsed.scheme or not parsed.netloc:
            print(f"FATAL: {key} is not a valid URL", file=sys.stderr)
            sys.exit(1)

    return WhatsAppBridgeConfig(
        whatsapp_app_secret=required["WHATSAPP_APP_SECRET"],
        whatsapp_verify_token=required["WHATSAPP_VERIFY_TOKEN"],
        whatsapp_access_token=required["WHATSAPP_ACCESS_TOKEN"],
        crm_api_key=required["CRM_API_KEY"],
        crm_api_url=os.getenv("CRM_API_URL", "https://api.crm.example.com/v2"),
        database_url=required["DATABASE_URL"],
        slack_webhook_url=required["SLACK_WEBHOOK_URL"],
        sync_interval_minutes=int(os.getenv("SYNC_INTERVAL_MINUTES", "15")),
        contact_match_threshold=float(os.getenv("CONTACT_MATCH_THRESHOLD", "0.70")),
        max_message_age_days=int(os.getenv("MAX_MESSAGE_AGE_DAYS", "90")),
        webhook_port=int(os.getenv("WEBHOOK_PORT", "8443")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


def create_db_pool(config: WhatsAppBridgeConfig) -> sqlalchemy.engine.Engine:
    return create_engine(
        config.database_url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True,
    )


config = load_config()
db_engine = create_db_pool(config)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Track WhatsApp message sync health, CRM activity creation rates, and conversation coverage with dashboards and proactive Slack alerts when sync lags or match rates drop.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'WhatsApp Engagement Dashboard Query',
                description:
                  'Powers a dashboard showing WhatsApp conversation health alongside CRM pipeline metrics per rep and deal.',
                code: `-- WhatsApp Engagement Dashboard
-- Combines CRM pipeline data with WhatsApp conversation metrics

WITH rep_whatsapp_stats AS (
  SELECT
    u.id                                  AS rep_id,
    u.full_name                           AS rep_name,
    COUNT(DISTINCT wm.contact_id)         AS contacts_messaged,
    COUNT(wm.id)                          AS total_messages,
    COUNT(wm.id) FILTER (
      WHERE wm.received_at >= NOW() - INTERVAL '7 days'
    ) AS messages_this_week,
    COUNT(DISTINCT ds.id)                 AS deal_signals_captured,
    COUNT(DISTINCT ds.id) FILTER (
      WHERE ds.signal_type = 'commitment_language'
    ) AS commitment_signals
  FROM users u
  LEFT JOIN opportunities o ON u.id = o.owner_id AND o.is_closed = FALSE
  LEFT JOIN opportunity_contact_roles ocr ON o.id = ocr.opportunity_id
  LEFT JOIN whatsapp_messages wm ON ocr.contact_id = wm.contact_id
  LEFT JOIN deal_signals ds ON wm.id = ds.message_id
  GROUP BY u.id, u.full_name
),
rep_pipeline AS (
  SELECT
    o.owner_id                            AS rep_id,
    COUNT(*)                              AS open_deals,
    SUM(o.amount)                         AS total_pipeline_value,
    AVG(o.amount)                         AS avg_deal_size
  FROM opportunities o
  WHERE o.is_closed = FALSE
  GROUP BY o.owner_id
)
SELECT
  rws.rep_name,
  COALESCE(rp.open_deals, 0)              AS open_deals,
  COALESCE(rp.total_pipeline_value, 0)     AS pipeline_value,
  rws.contacts_messaged,
  rws.total_messages,
  rws.messages_this_week,
  rws.deal_signals_captured,
  rws.commitment_signals,
  CASE
    WHEN rws.messages_this_week > 20 THEN 'highly_active'
    WHEN rws.messages_this_week > 5  THEN 'active'
    WHEN rws.messages_this_week > 0  THEN 'low_activity'
    ELSE 'no_whatsapp_usage'
  END AS whatsapp_engagement_level,
  ROUND(
    rws.deal_signals_captured::numeric
    / NULLIF(rws.total_messages, 0) * 100, 1
  ) AS signal_density_pct
FROM rep_whatsapp_stats rws
LEFT JOIN rep_pipeline rp ON rws.rep_id = rp.rep_id
ORDER BY rws.deal_signals_captured DESC;`,
              },
              {
                language: 'python',
                title: 'WhatsApp Sync Health Slack Alerting',
                description:
                  'Monitors message ingestion rates, CRM contact match ratios, sync lag, and activity creation success — sends Slack alerts when any metric breaches its threshold.',
                code: `"""
WhatsApp CRM Bridge — Sync Health Monitoring & Slack Alerts
Tracks message ingestion, contact matching, and CRM sync health.
"""

import os
import requests
from datetime import datetime
from dataclasses import dataclass
from sqlalchemy import create_engine, text

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
engine = create_engine(DATABASE_URL)


@dataclass
class SyncThresholds:
    min_messages_per_day: int = 10
    min_contact_match_pct: float = 70.0
    max_sync_lag_minutes: int = 30
    max_unsynced_messages: int = 100
    min_signal_extraction_pct: float = 5.0


def check_sync_health(thresholds: SyncThresholds) -> list[dict]:
    alerts: list[dict] = []
    with engine.connect() as conn:
        # Message ingestion rate
        row = conn.execute(text(
            "SELECT COUNT(*) AS cnt FROM whatsapp_messages "
            "WHERE received_at >= NOW() - INTERVAL '24 hours'"
        )).fetchone()
        if row.cnt < thresholds.min_messages_per_day:
            alerts.append({
                "type": "low_ingestion",
                "message": f"Only {row.cnt} messages in 24h "
                           f"(threshold: {thresholds.min_messages_per_day})",
            })

        # Contact match rate
        match_row = conn.execute(text(\"\"\"
            SELECT
              COUNT(*) AS total,
              COUNT(*) FILTER (WHERE contact_id IS NOT NULL) AS matched
            FROM whatsapp_messages
            WHERE received_at >= NOW() - INTERVAL '7 days'
        \"\"\")).fetchone()
        if match_row.total > 0:
            match_pct = 100.0 * match_row.matched / match_row.total
            if match_pct < thresholds.min_contact_match_pct:
                alerts.append({
                    "type": "low_match_rate",
                    "message": f"Contact match rate is {match_pct:.1f}% "
                               f"(threshold: {thresholds.min_contact_match_pct}%)",
                })

        # Sync lag
        lag_row = conn.execute(text(\"\"\"
            SELECT COUNT(*) AS unsynced,
                   MIN(received_at) AS oldest_unsynced
            FROM whatsapp_messages
            WHERE synced_to_crm = FALSE AND contact_id IS NOT NULL
        \"\"\")).fetchone()
        if lag_row.unsynced > thresholds.max_unsynced_messages:
            alerts.append({
                "type": "sync_backlog",
                "message": f"{lag_row.unsynced} messages pending CRM sync "
                           f"(threshold: {thresholds.max_unsynced_messages})",
            })
        if lag_row.oldest_unsynced:
            lag_min = (datetime.utcnow() - lag_row.oldest_unsynced).total_seconds() / 60
            if lag_min > thresholds.max_sync_lag_minutes:
                alerts.append({
                    "type": "sync_lag",
                    "message": f"Oldest unsynced message is {lag_min:.0f}m old "
                               f"(threshold: {thresholds.max_sync_lag_minutes}m)",
                })

        # Signal extraction health
        sig_row = conn.execute(text(\"\"\"
            SELECT
              COUNT(DISTINCT wm.id) AS messages_with_contact,
              COUNT(DISTINCT ds.message_id) AS messages_with_signals
            FROM whatsapp_messages wm
            LEFT JOIN deal_signals ds ON wm.id = ds.message_id
            WHERE wm.received_at >= NOW() - INTERVAL '7 days'
              AND wm.contact_id IS NOT NULL
        \"\"\")).fetchone()
        if sig_row.messages_with_contact > 0:
            sig_pct = 100.0 * sig_row.messages_with_signals / sig_row.messages_with_contact
            if sig_pct < thresholds.min_signal_extraction_pct:
                alerts.append({
                    "type": "low_signal_rate",
                    "message": f"Signal extraction rate {sig_pct:.1f}% "
                               f"(threshold: {thresholds.min_signal_extraction_pct}%)",
                })
    return alerts


def send_slack_alerts(alerts: list[dict]) -> None:
    if not alerts or not SLACK_WEBHOOK:
        return
    severity = "critical" if len(alerts) >= 3 else "warning"
    icon = ":rotating_light:" if severity == "critical" else ":warning:"
    lines = [f"  - [{a['type']}] {a['message']}" for a in alerts]
    payload = {
        "text": (
            f"{icon} *WhatsApp CRM Bridge Alert* ({severity})\\n"
            + "\\n".join(lines)
            + f"\\nChecked at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        ),
    }
    requests.post(SLACK_WEBHOOK, json=payload, timeout=10)


if __name__ == "__main__":
    thresholds = SyncThresholds()
    alerts = check_sync_health(thresholds)
    if alerts:
        print(f"Found {len(alerts)} alert(s), notifying Slack...")
        send_slack_alerts(alerts)
    else:
        print("WhatsApp sync health checks passed.")`,
              },
            ],
          },
        ],
      },
      aiEasyWin: {
        overview:
          'Use ChatGPT/Claude with Zapier to capture WhatsApp conversations via WhatsApp Business API, extract deal signals, and sync key conversation summaries to your CRM without building custom middleware.',
        estimatedMonthlyCost: '$150 - $250/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Twilio for WhatsApp ($0.005-0.05/msg)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'MessageBird', 'Gong AI (for call analysis)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Set up Twilio WhatsApp Business API sandbox or production account, configure webhook to capture all inbound and outbound messages, and route them to Zapier for processing.',
            toolsUsed: ['Twilio', 'Zapier Webhooks', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Twilio WhatsApp Webhook to Zapier Configuration',
                description:
                  'Configures Twilio to send WhatsApp messages to Zapier webhook for AI processing and CRM sync.',
                code: `{
  "twilio_webhook_config": {
    "account_sid": "{{env.TWILIO_ACCOUNT_SID}}",
    "whatsapp_number": "+14155238886",
    "webhook_url": "https://hooks.zapier.com/hooks/catch/{{zapier_webhook_id}}/",
    "events": ["onMessageSent", "onMessageReceived"],
    "method": "POST",
    "content_type": "application/json"
  },
  "zapier_trigger": {
    "type": "webhook",
    "event": "catch_raw_hook",
    "output_mapping": {
      "message_sid": "{{trigger.MessageSid}}",
      "from_number": "{{trigger.From}}",
      "to_number": "{{trigger.To}}",
      "body": "{{trigger.Body}}",
      "direction": "{{#if trigger.From contains 'whatsapp:+1'}}outbound{{else}}inbound{{/if}}",
      "timestamp": "{{trigger.DateCreated}}",
      "media_url": "{{trigger.MediaUrl0}}",
      "num_media": "{{trigger.NumMedia}}"
    }
  },
  "storage_action": {
    "app": "google_sheets",
    "action": "create_spreadsheet_row",
    "config": {
      "spreadsheet_id": "{{env.WHATSAPP_LOG_SHEET_ID}}",
      "worksheet": "Messages_Raw",
      "row_data": {
        "Message_SID": "{{trigger.message_sid}}",
        "From": "{{trigger.from_number | replace('whatsapp:', '')}}",
        "To": "{{trigger.to_number | replace('whatsapp:', '')}}",
        "Direction": "{{trigger.direction}}",
        "Body": "{{trigger.body}}",
        "Timestamp": "{{trigger.timestamp}}",
        "Has_Media": "{{#if trigger.num_media > 0}}Yes{{else}}No{{/if}}",
        "Processed": "No",
        "CRM_Contact_ID": "",
        "Deal_Signals": ""
      }
    }
  },
  "contact_lookup": {
    "app": "salesforce",
    "action": "find_record",
    "config": {
      "object": "Contact",
      "search_field": "Phone",
      "search_value": "{{trigger.from_number | replace('whatsapp:', '') | replace('+', '')}}",
      "fallback_search": {
        "field": "MobilePhone",
        "value": "{{trigger.from_number | replace('whatsapp:', '')}}"
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
              'Use ChatGPT or Claude to analyze WhatsApp conversations, extract deal signals (pricing discussions, objections, commitments), and generate concise summaries for CRM logging.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro', 'Zapier AI Actions'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'WhatsApp Conversation Analysis Prompt Template',
                description:
                  'Structured prompt for AI to extract deal signals and generate CRM-ready summaries from WhatsApp conversations.',
                code: `system_prompt: |
  You are a Sales Intelligence AI that analyzes WhatsApp conversations between
  sales reps and prospects. Your job is to extract actionable deal signals and
  create concise summaries that help the sales team and preserve institutional
  knowledge.

  Key signals to detect:
  - Pricing/budget discussions (amounts, discount requests, budget constraints)
  - Timeline/urgency indicators (deadlines, "need this by", "asap")
  - Competitor mentions (alternative vendors, "also looking at")
  - Objections (concerns, pushback, hesitation)
  - Commitment language ("let's proceed", "approved", "ready to go")
  - Stakeholder mentions (decision-makers, approvals needed)
  - Next steps agreed upon

user_prompt_template: |
  ## WhatsApp Conversation Analysis

  **Contact Phone:** {{contact_phone}}
  **CRM Contact:** {{crm_contact_name}} at {{crm_company}}
  **Associated Deal:** {{deal_name}} (Stage: {{deal_stage}})
  **Conversation Date Range:** {{first_message_date}} to {{last_message_date}}

  ### Conversation Thread:
  \`\`\`
  {{conversation_thread}}
  \`\`\`

  ### Analysis Instructions:

  1. **Extract Deal Signals:** Identify all buying signals and objections
  2. **Detect Commitments:** Note any verbal agreements or next steps
  3. **Flag Risks:** Identify competitor mentions, budget concerns, or delays
  4. **Summarize for CRM:** Create a 2-3 sentence summary for the activity log
  5. **Suggest Actions:** What should the rep do next based on this conversation?

  ### Required Output Format:

  \`\`\`json
  {
    "conversation_summary": "<2-3 sentence summary for CRM activity log>",
    "sentiment": "<positive|neutral|negative|mixed>",
    "deal_signals": [
      {
        "type": "<pricing|urgency|commitment|objection|competitor|stakeholder|next_step>",
        "confidence": <0.0-1.0>,
        "excerpt": "<exact quote from conversation>",
        "implication": "<what this means for the deal>"
      }
    ],
    "key_topics_discussed": ["<topic 1>", "<topic 2>"],
    "commitments_made": [
      {
        "by": "<rep|prospect>",
        "commitment": "<what was committed>",
        "deadline": "<if mentioned>"
      }
    ],
    "risks_identified": [
      {
        "type": "<competitor|budget|timeline|stakeholder|other>",
        "description": "<description of risk>",
        "severity": "<high|medium|low>"
      }
    ],
    "recommended_actions": [
      {
        "action": "<specific action to take>",
        "priority": "<high|medium|low>",
        "timing": "<immediate|today|this_week>"
      }
    ],
    "crm_update": {
      "activity_subject": "<short subject line for CRM task>",
      "activity_description": "<full description for CRM>",
      "next_step_field": "<suggested update for Next Step field>",
      "stage_recommendation": "<should deal stage change? which stage?>"
    }
  }
  \`\`\`

output_instructions: |
  - Use exact quotes for excerpts when possible
  - Be specific about next steps - don't just say "follow up"
  - If pricing is mentioned, extract the exact amounts
  - Flag any competitor mentions prominently
  - Keep CRM summary concise but actionable`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier to process conversations in batches, sync summaries and signals to CRM as activities, and alert reps when hot signals are detected.',
            toolsUsed: ['Zapier', 'Salesforce/HubSpot', 'Slack'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier WhatsApp-to-CRM Sync Workflow',
                description:
                  'Complete workflow that analyzes WhatsApp conversations with AI and syncs results to CRM with intelligent alerting.',
                code: `{
  "workflow_name": "WhatsApp Conversation Intelligence Sync",
  "trigger": {
    "app": "schedule",
    "event": "every_15_minutes"
  },
  "steps": [
    {
      "step_number": 1,
      "name": "Fetch Unprocessed Messages",
      "app": "google_sheets",
      "action": "find_rows",
      "config": {
        "spreadsheet_id": "{{env.WHATSAPP_LOG_SHEET_ID}}",
        "worksheet": "Messages_Raw",
        "filter": {
          "Processed": "No"
        },
        "limit": 100
      }
    },
    {
      "step_number": 2,
      "name": "Group Messages by Contact",
      "app": "code",
      "action": "run_javascript",
      "config": {
        "code": "const messages = inputData.rows; const grouped = {}; messages.forEach(m => { const key = m.From; if (!grouped[key]) grouped[key] = []; grouped[key].push(m); }); return Object.entries(grouped).map(([phone, msgs]) => ({ phone, messages: msgs.sort((a,b) => new Date(a.Timestamp) - new Date(b.Timestamp)), thread: msgs.map(m => m.Direction === 'inbound' ? '[Customer]: ' + m.Body : '[Rep]: ' + m.Body).join('\\n') }));"
      }
    },
    {
      "step_number": 3,
      "name": "Lookup CRM Contact and Deal",
      "app": "loop",
      "action": "for_each",
      "config": {
        "items": "{{step2.grouped_conversations}}",
        "sub_steps": [
          {
            "app": "salesforce",
            "action": "find_record",
            "config": {
              "object": "Contact",
              "soql": "SELECT Id, Name, AccountId, Account.Name, (SELECT Id, Name, StageName, Amount FROM Opportunities WHERE IsClosed = false LIMIT 1) FROM Contact WHERE Phone LIKE '%{{loop.item.phone | last_digits:10}}%' OR MobilePhone LIKE '%{{loop.item.phone | last_digits:10}}%' LIMIT 1"
            }
          }
        ]
      }
    },
    {
      "step_number": 4,
      "name": "AI Conversation Analysis",
      "app": "loop",
      "action": "for_each",
      "config": {
        "items": "{{step2.grouped_conversations}}",
        "sub_steps": [
          {
            "app": "chatgpt",
            "action": "conversation",
            "config": {
              "model": "gpt-4",
              "system_message": "{{prompts.whatsapp_analysis.system}}",
              "user_message": "{{prompts.whatsapp_analysis.user | replace_vars(loop.item, step3.contact_data)}}",
              "max_tokens": 2500,
              "temperature": 0.3
            }
          }
        ]
      }
    },
    {
      "step_number": 5,
      "name": "Create CRM Activity",
      "app": "loop",
      "action": "for_each",
      "config": {
        "items": "{{step4.analysis_results}}",
        "condition": "{{loop.item.crm_contact_id is not empty}}",
        "sub_steps": [
          {
            "app": "salesforce",
            "action": "create_record",
            "config": {
              "object": "Task",
              "fields": {
                "WhoId": "{{loop.item.crm_contact_id}}",
                "WhatId": "{{loop.item.deal_id}}",
                "Subject": "WhatsApp: {{loop.item.ai_analysis.crm_update.activity_subject}}",
                "Description": "{{loop.item.ai_analysis.crm_update.activity_description}}\\n\\n---\\nDeal Signals: {{loop.item.ai_analysis.deal_signals | map:'type' | join:', '}}\\nSentiment: {{loop.item.ai_analysis.sentiment}}\\nMessages Analyzed: {{loop.item.message_count}}",
                "Status": "Completed",
                "Priority": "{{#if loop.item.ai_analysis.deal_signals contains 'commitment'}}High{{else}}Normal{{/if}}",
                "ActivityDate": "{{today}}"
              }
            }
          }
        ]
      }
    },
    {
      "step_number": 6,
      "name": "Alert on Hot Signals",
      "app": "filter",
      "config": {
        "condition": "{{step4.analysis_results | filter: 'has_hot_signal' | length > 0}}"
      },
      "sub_steps": [
        {
          "app": "slack",
          "action": "send_channel_message",
          "config": {
            "channel": "#sales-whatsapp-intel",
            "message_blocks": [
              {
                "type": "header",
                "text": ":speech_balloon: WhatsApp Deal Signal Detected"
              },
              {
                "type": "section",
                "text": "*{{loop.item.crm_contact_name}}* at *{{loop.item.company}}*\\nDeal: {{loop.item.deal_name}} ({{loop.item.deal_stage}})"
              },
              {
                "type": "section",
                "text": "*Signals Detected:*\\n{{#each loop.item.ai_analysis.deal_signals}}:sparkles: *{{type}}*: \\"{{excerpt}}\\"\\n{{/each}}"
              },
              {
                "type": "section",
                "text": "*Recommended Action:*\\n{{loop.item.ai_analysis.recommended_actions[0].action}} ({{loop.item.ai_analysis.recommended_actions[0].timing}})"
              },
              {
                "type": "actions",
                "elements": [
                  {
                    "type": "button",
                    "text": "View in CRM",
                    "url": "{{env.CRM_BASE_URL}}/{{loop.item.deal_id}}"
                  }
                ]
              }
            ]
          }
        }
      ]
    },
    {
      "step_number": 7,
      "name": "Mark Messages as Processed",
      "app": "google_sheets",
      "action": "update_rows",
      "config": {
        "spreadsheet_id": "{{env.WHATSAPP_LOG_SHEET_ID}}",
        "worksheet": "Messages_Raw",
        "filter": {
          "Message_SID": { "in": "{{step1.rows | map:'Message_SID'}}" }
        },
        "updates": {
          "Processed": "Yes",
          "Processed_At": "{{now}}",
          "CRM_Activity_ID": "{{step5.created_activity_id}}"
        }
      }
    }
  ]
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent system that captures WhatsApp conversations in real-time, uses NLP to extract deal intelligence, auto-maps conversations to CRM records, and generates actionable insights for sales coaching.',
        estimatedMonthlyCost: '$700 - $1,400/month',
        architecture:
          'A Supervisor agent orchestrates four specialists: MessageIngestion captures and stores messages via WhatsApp API, ConversationAnalyzer extracts deal signals using NLP, CRMMapper links conversations to contacts and opportunities, and InsightGenerator produces coaching reports and alerts.',
        agents: [
          {
            name: 'MessageIngestionAgent',
            role: 'WhatsApp Message Collector',
            goal: 'Capture, validate, and persist all WhatsApp messages in real-time with proper contact matching and deduplication.',
            tools: ['WhatsApp Business API', 'PostgreSQL', 'Redis Streams'],
          },
          {
            name: 'ConversationAnalyzerAgent',
            role: 'NLP Deal Signal Extractor',
            goal: 'Analyze conversation text to extract pricing mentions, competitor references, objections, commitments, and sentiment.',
            tools: ['spaCy', 'OpenAI GPT-4', 'Custom NER Models'],
          },
          {
            name: 'CRMMapperAgent',
            role: 'Contact and Opportunity Matcher',
            goal: 'Accurately link WhatsApp contacts to CRM records using phone matching, fuzzy name matching, and company inference.',
            tools: ['Salesforce API', 'HubSpot API', 'Fuzzy Matching'],
          },
          {
            name: 'InsightGeneratorAgent',
            role: 'Sales Intelligence Synthesizer',
            goal: 'Aggregate conversation signals into deal health scores, rep coaching reports, and competitive intelligence summaries.',
            tools: ['Analytics Engine', 'Report Generator', 'Slack API'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'PostgreSQL for messages, Redis for real-time processing state, 90-day retention',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system for WhatsApp conversation intelligence with CrewAI, establishing roles for message capture, NLP analysis, CRM integration, and insight generation.',
            toolsUsed: ['CrewAI', 'LangChain', 'Python'],
            codeSnippets: [
              {
                language: 'python',
                title: 'WhatsApp Intelligence Agent Crew Definition',
                description:
                  'CrewAI configuration for the WhatsApp conversation intelligence system with four specialist agents.',
                code: `"""
WhatsApp Conversation Intelligence Multi-Agent System
CrewAI-based agent definitions for real-time conversation capture and analysis.
"""

from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import os

llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
)


class WhatsAppIntelligenceAgents:
    """Factory class for WhatsApp conversation intelligence agents."""

    @staticmethod
    def create_ingestion_agent(tools: List[Any]) -> Agent:
        return Agent(
            role="WhatsApp Message Collector",
            goal="""Capture and persist every WhatsApp message with full metadata:
                1. Validate incoming webhook signatures for security
                2. Deduplicate messages using message SID
                3. Normalize phone numbers for consistent matching
                4. Store messages with timestamps and direction
                5. Queue messages for downstream processing""",
            backstory="""You are a data integration engineer who built messaging
                infrastructure at scale. You understand that every lost message is
                lost institutional knowledge. You handle edge cases gracefully -
                duplicate webhooks, missing fields, rate limits - and ensure
                99.99% message capture reliability.""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=10,
        )

    @staticmethod
    def create_analyzer_agent(tools: List[Any]) -> Agent:
        return Agent(
            role="Conversation NLP Analyst",
            goal="""Extract actionable deal intelligence from conversations:
                1. Identify pricing discussions and exact amounts mentioned
                2. Detect competitor references and comparison language
                3. Extract urgency indicators and timeline mentions
                4. Recognize commitment language and verbal agreements
                5. Assess overall sentiment and relationship health""",
            backstory="""You are a conversational AI specialist who has analyzed
                millions of sales conversations. You can read between the lines -
                detecting hesitation, enthusiasm, and buying signals that others
                miss. You understand that 'let me think about it' often means 'no'
                while 'when can we start?' means 'close me now'.""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=15,
        )

    @staticmethod
    def create_crm_mapper_agent(tools: List[Any]) -> Agent:
        return Agent(
            role="CRM Contact Matcher",
            goal="""Accurately link WhatsApp contacts to CRM records:
                1. Match phone numbers with normalization (country codes, formats)
                2. Use fuzzy matching on names when phone match fails
                3. Infer company from email domain or conversation context
                4. Link conversations to relevant open opportunities
                5. Create new contact records when no match exists""",
            backstory="""You are a data quality expert who has cleaned CRM databases
                for Fortune 500 sales teams. You know that phone formats vary wildly
                across countries, that people use nicknames, and that company names
                have many variations. You achieve 95%+ match accuracy while
                minimizing false positives that would pollute CRM data.""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=10,
        )

    @staticmethod
    def create_insight_agent(tools: List[Any]) -> Agent:
        return Agent(
            role="Sales Intelligence Synthesizer",
            goal="""Generate actionable insights from conversation patterns:
                1. Calculate conversation health scores for each deal
                2. Identify deals going cold based on response patterns
                3. Surface competitive threats across the pipeline
                4. Generate coaching reports on rep messaging patterns
                5. Predict deals likely to close based on conversation signals""",
            backstory="""You are a sales analytics leader who has driven 40%
                improvement in win rates through conversation intelligence. You
                see patterns across thousands of conversations that reveal what
                winning looks like. You translate data into specific coaching
                actions that change rep behavior.""",
            tools=tools,
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_iter=15,
        )


def create_whatsapp_intelligence_crew(
    ingestion_tools: List[Any],
    analyzer_tools: List[Any],
    mapper_tools: List[Any],
    insight_tools: List[Any],
) -> Crew:
    """Create the WhatsApp conversation intelligence crew."""
    factory = WhatsAppIntelligenceAgents()

    ingestion_agent = factory.create_ingestion_agent(ingestion_tools)
    analyzer_agent = factory.create_analyzer_agent(analyzer_tools)
    crm_mapper = factory.create_crm_mapper_agent(mapper_tools)
    insight_generator = factory.create_insight_agent(insight_tools)

    return Crew(
        agents=[ingestion_agent, analyzer_agent, crm_mapper, insight_generator],
        process=Process.sequential,
        verbose=True,
        memory=True,
        cache=True,
        max_rpm=40,
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the Message Ingestion agent with tools to capture WhatsApp messages via webhook, validate signatures, and persist to the conversation store.',
            toolsUsed: ['FastAPI', 'WhatsApp Business API', 'PostgreSQL', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'WhatsApp Message Ingestion Agent Tools',
                description:
                  'Tool implementations for secure message capture, deduplication, and storage.',
                code: `"""
WhatsApp Message Ingestion Agent — Tool Implementations
Secure webhook handling, message normalization, and persistence.
"""

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib
import hmac
import phonenumbers
import asyncio
import httpx
import os

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool


class MessageIngestionInput(BaseModel):
    webhook_payload: Dict[str, Any] = Field(description="Raw webhook payload from WhatsApp")
    signature: str = Field(description="X-Hub-Signature-256 header value")


class WhatsAppIngestionTool(BaseTool):
    name: str = "ingest_whatsapp_message"
    description: str = """Process incoming WhatsApp webhook, validate signature,
        extract message data, and persist to database with deduplication."""
    args_schema: type[BaseModel] = MessageIngestionInput

    def __init__(self):
        super().__init__()
        self.app_secret = os.getenv("WHATSAPP_APP_SECRET")
        self.db_engine = create_engine(
            os.getenv("DATABASE_URL"),
            poolclass=QueuePool,
            pool_size=10,
        )

    def _run(
        self,
        webhook_payload: Dict[str, Any],
        signature: str,
    ) -> Dict[str, Any]:
        # 1. Validate webhook signature
        if not self._verify_signature(webhook_payload, signature):
            return {"status": "error", "reason": "Invalid signature"}

        # 2. Extract messages from payload
        messages = self._extract_messages(webhook_payload)
        if not messages:
            return {"status": "ok", "messages_processed": 0}

        # 3. Process and store each message
        results = []
        for msg in messages:
            result = self._process_message(msg)
            results.append(result)

        return {
            "status": "ok",
            "messages_processed": len(results),
            "messages": results,
        }

    def _verify_signature(self, payload: Dict, signature: str) -> bool:
        import json
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
        expected = hmac.new(
            self.app_secret.encode(),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(f"sha256={expected}", signature)

    def _extract_messages(self, payload: Dict) -> List[Dict]:
        messages = []
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                for msg in value.get("messages", []):
                    messages.append({
                        "message_id": msg["id"],
                        "from_phone": msg["from"],
                        "timestamp": msg["timestamp"],
                        "type": msg["type"],
                        "text": msg.get("text", {}).get("body", ""),
                        "direction": "inbound",
                    })
                # Also capture status updates for outbound tracking
                for status in value.get("statuses", []):
                    if status.get("status") == "sent":
                        messages.append({
                            "message_id": status["id"],
                            "to_phone": status["recipient_id"],
                            "timestamp": status["timestamp"],
                            "direction": "outbound",
                            "status": status["status"],
                        })
        return messages

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to E.164 format."""
        try:
            parsed = phonenumbers.parse(phone, None)
            return phonenumbers.format_number(
                parsed, phonenumbers.PhoneNumberFormat.E164
            )
        except Exception:
            # Fallback: strip non-digits
            return "+" + "".join(c for c in phone if c.isdigit())

    def _process_message(self, msg: Dict) -> Dict:
        phone_field = "from_phone" if msg["direction"] == "inbound" else "to_phone"
        normalized_phone = self._normalize_phone(msg.get(phone_field, ""))

        message_record = {
            "message_id": msg["message_id"],
            "phone_normalized": normalized_phone,
            "direction": msg["direction"],
            "body": msg.get("text", ""),
            "message_type": msg.get("type", "text"),
            "received_at": datetime.fromtimestamp(int(msg["timestamp"])),
            "processed_at": datetime.utcnow(),
        }

        # Insert with deduplication
        with self.db_engine.connect() as conn:
            result = conn.execute(
                text(\"\"\"
                    INSERT INTO whatsapp_messages
                    (message_id, phone_normalized, direction, body, message_type, received_at, processed_at)
                    VALUES (:message_id, :phone_normalized, :direction, :body, :message_type, :received_at, :processed_at)
                    ON CONFLICT (message_id) DO NOTHING
                    RETURNING id, message_id
                \"\"\"),
                message_record,
            )
            conn.commit()
            inserted = result.fetchone()

        return {
            "message_id": msg["message_id"],
            "phone": normalized_phone,
            "direction": msg["direction"],
            "stored": inserted is not None,
            "db_id": inserted[0] if inserted else None,
        }


class PhoneContactMatchInput(BaseModel):
    phone_normalized: str = Field(description="E.164 formatted phone number")


class CRMContactMatchTool(BaseTool):
    name: str = "match_crm_contact"
    description: str = """Find CRM contact matching a phone number.
        Uses exact match first, then fuzzy matching on name."""
    args_schema: type[BaseModel] = PhoneContactMatchInput

    def __init__(self):
        super().__init__()
        self.crm_api_url = os.getenv("CRM_API_URL")
        self.crm_api_key = os.getenv("CRM_API_KEY")

    def _run(self, phone_normalized: str) -> Dict[str, Any]:
        # Try multiple phone formats
        phone_variants = self._generate_phone_variants(phone_normalized)

        with httpx.Client(timeout=30) as client:
            for variant in phone_variants:
                response = client.get(
                    f"{self.crm_api_url}/contacts/search",
                    params={"phone": variant},
                    headers={"Authorization": f"Bearer {self.crm_api_key}"},
                )
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    if results:
                        contact = results[0]
                        # Fetch associated opportunities
                        opps = self._fetch_opportunities(client, contact["id"])
                        return {
                            "matched": True,
                            "match_type": "phone_exact",
                            "contact_id": contact["id"],
                            "contact_name": contact.get("name", ""),
                            "company": contact.get("company", ""),
                            "opportunities": opps,
                        }

        return {
            "matched": False,
            "match_type": None,
            "contact_id": None,
            "phone_searched": phone_normalized,
        }

    def _generate_phone_variants(self, phone: str) -> List[str]:
        """Generate common phone format variations."""
        digits = "".join(c for c in phone if c.isdigit())
        return [
            phone,  # Original E.164
            digits,  # Just digits
            digits[-10:],  # Last 10 digits (US format)
            f"+{digits}",  # With plus
            f"({digits[:3]}) {digits[3:6]}-{digits[6:10]}" if len(digits) >= 10 else digits,
        ]

    def _fetch_opportunities(
        self, client: httpx.Client, contact_id: str
    ) -> List[Dict]:
        response = client.get(
            f"{self.crm_api_url}/contacts/{contact_id}/opportunities",
            params={"status": "open"},
            headers={"Authorization": f"Bearer {self.crm_api_key}"},
        )
        if response.status_code == 200:
            return response.json().get("results", [])[:5]
        return []`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Conversation Analyzer agent with NLP tools to extract deal signals, sentiment, and actionable insights from message content.',
            toolsUsed: ['spaCy', 'OpenAI GPT-4', 'Custom NER', 'Sentiment Analysis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Conversation NLP Analysis Tools',
                description:
                  'Tools for extracting deal signals, sentiment, and conversation intelligence using NLP.',
                code: `"""
Conversation Analyzer Agent — NLP Analysis Tools
Extract deal signals, sentiment, and actionable insights.
"""

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import re
import spacy
from openai import OpenAI
import os


class ConversationAnalysisInput(BaseModel):
    conversation_thread: List[Dict[str, Any]] = Field(
        description="List of messages in chronological order"
    )
    contact_context: Dict[str, Any] = Field(
        default={},
        description="CRM context: contact name, company, deal info",
    )


class DealSignalExtractorTool(BaseTool):
    name: str = "extract_deal_signals"
    description: str = """Analyze conversation to extract deal signals:
        pricing mentions, competitor references, urgency, commitments, objections."""
    args_schema: type[BaseModel] = ConversationAnalysisInput

    def __init__(self):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._init_patterns()

    def _init_patterns(self):
        self.patterns = {
            "pricing": [
                r"\\$[\\d,]+(?:\\.\\d{2})?",
                r"\\b\\d+(?:\\.\\d+)?\\s*[kK]\\b",
                r"(?:price|cost|budget|quote|proposal|discount|offer|rate)\\w*",
                r"(?:per\\s+(?:month|year|seat|user|license))",
            ],
            "urgency": [
                r"\\b(?:asap|urgent|immediately|right away)\\b",
                r"\\b(?:deadline|by\\s+(?:monday|tuesday|wednesday|thursday|friday|tomorrow))\\b",
                r"\\b(?:end\\s+of\\s+(?:week|month|quarter|year))\\b",
                r"\\b(?:running\\s+out|time\\s+sensitive|cannot\\s+wait)\\b",
            ],
            "commitment": [
                r"\\b(?:let'?s\\s+(?:go\\s+ahead|proceed|do\\s+it|move\\s+forward|sign))\\b",
                r"\\b(?:approved|greenlight|ready\\s+to\\s+start|confirmed|deal)\\b",
                r"\\b(?:send\\s+(?:the\\s+)?(?:contract|agreement|paperwork))\\b",
            ],
            "objection": [
                r"\\b(?:too\\s+expensive|over\\s+budget|can'?t\\s+afford)\\b",
                r"\\b(?:not\\s+(?:sure|ready|convinced)|need\\s+to\\s+think)\\b",
                r"\\b(?:talk\\s+to\\s+(?:my\\s+)?(?:boss|team|manager|cfo))\\b",
                r"\\b(?:competitor|alternative|other\\s+(?:option|vendor))\\b",
            ],
            "next_step": [
                r"\\b(?:schedule|book|set\\s+up)\\s+(?:a\\s+)?(?:call|meeting|demo)\\b",
                r"\\b(?:send\\s+(?:me|over|through))\\b",
                r"\\b(?:follow\\s+up|get\\s+back|touch\\s+base)\\b",
            ],
        }

    def _run(
        self,
        conversation_thread: List[Dict[str, Any]],
        contact_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        contact_context = contact_context or {}

        # Combine all messages into analyzable text
        full_text = "\\n".join(
            f"[{m.get('direction', 'unknown').upper()}] {m.get('body', '')}"
            for m in conversation_thread
        )

        # Extract pattern-based signals
        signals = []
        for signal_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, full_text.lower())
                for match in matches:
                    signals.append({
                        "type": signal_type,
                        "pattern_match": match,
                        "confidence": 0.85,
                    })

        # Use spaCy for entity extraction
        doc = self.nlp(full_text)
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                signals.append({
                    "type": "pricing",
                    "entity": ent.text,
                    "confidence": 0.90,
                })
            elif ent.label_ == "DATE":
                signals.append({
                    "type": "timeline",
                    "entity": ent.text,
                    "confidence": 0.75,
                })
            elif ent.label_ == "ORG":
                # Could be competitor mention
                if ent.text.lower() not in contact_context.get("company", "").lower():
                    signals.append({
                        "type": "org_mention",
                        "entity": ent.text,
                        "confidence": 0.60,
                    })

        # Use GPT-4 for deeper analysis
        gpt_analysis = self._gpt_analyze(full_text, contact_context)

        # Deduplicate and merge signals
        final_signals = self._merge_signals(signals, gpt_analysis.get("signals", []))

        return {
            "signal_count": len(final_signals),
            "signals": final_signals,
            "sentiment": gpt_analysis.get("sentiment", "neutral"),
            "summary": gpt_analysis.get("summary", ""),
            "recommended_actions": gpt_analysis.get("actions", []),
            "conversation_health": self._calculate_health_score(final_signals),
        }

    def _gpt_analyze(
        self, conversation: str, context: Dict
    ) -> Dict[str, Any]:
        prompt = f\"\"\"Analyze this sales conversation and extract:
1. Deal signals (pricing, urgency, commitment, objection)
2. Overall sentiment (positive/neutral/negative)
3. Key summary (2-3 sentences)
4. Recommended next actions

Context:
- Contact: {context.get('contact_name', 'Unknown')}
- Company: {context.get('company', 'Unknown')}
- Deal: {context.get('deal_name', 'Unknown')} ({context.get('deal_stage', 'Unknown')})

Conversation:
{conversation[:3000]}

Return JSON format:
{{
  "signals": [{{"type": "...", "excerpt": "...", "confidence": 0.0-1.0}}],
  "sentiment": "positive|neutral|negative",
  "summary": "...",
  "actions": [{{"action": "...", "priority": "high|medium|low"}}]
}}\"\"\"

        response = self.openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        import json
        return json.loads(response.choices[0].message.content)

    def _merge_signals(
        self,
        pattern_signals: List[Dict],
        gpt_signals: List[Dict],
    ) -> List[Dict]:
        merged = {}
        for sig in pattern_signals + gpt_signals:
            key = f"{sig['type']}:{sig.get('excerpt', sig.get('pattern_match', sig.get('entity', '')))[:50]}"
            if key not in merged or sig.get("confidence", 0) > merged[key].get("confidence", 0):
                merged[key] = sig
        return list(merged.values())

    def _calculate_health_score(self, signals: List[Dict]) -> Dict[str, Any]:
        score = 50  # Baseline

        for sig in signals:
            if sig["type"] == "commitment":
                score += 15
            elif sig["type"] == "urgency":
                score += 10
            elif sig["type"] == "pricing":
                score += 5
            elif sig["type"] == "objection":
                score -= 10
            elif sig["type"] == "competitor":
                score -= 15

        score = max(0, min(100, score))

        if score >= 70:
            status = "healthy"
        elif score >= 40:
            status = "needs_attention"
        else:
            status = "at_risk"

        return {
            "score": score,
            "status": status,
        }`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement LangGraph state machine to orchestrate the WhatsApp intelligence pipeline from message ingestion through CRM sync and alerting.',
            toolsUsed: ['LangGraph', 'Redis', 'PostgreSQL'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph WhatsApp Intelligence Orchestrator',
                description:
                  'State machine for the WhatsApp conversation intelligence workflow.',
                code: `"""
WhatsApp Intelligence — LangGraph Orchestration
Real-time message processing with conversation aggregation.
"""

from typing import TypedDict, Annotated, Literal, List
from datetime import datetime, timedelta
import operator
import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
import redis


class WhatsAppIntelState(TypedDict):
    """State for WhatsApp intelligence workflow."""
    messages_to_process: List[dict]
    processed_messages: List[dict]
    contact_matches: dict
    conversation_analyses: List[dict]
    crm_updates: List[dict]
    alerts_generated: List[dict]
    workflow_status: str


class WhatsAppIntelOrchestrator:
    def __init__(
        self,
        ingestion_agent,
        analyzer_agent,
        mapper_agent,
        insight_agent,
        redis_url: str,
    ):
        self.ingestion_agent = ingestion_agent
        self.analyzer_agent = analyzer_agent
        self.mapper_agent = mapper_agent
        self.insight_agent = insight_agent
        self.redis = redis.from_url(redis_url)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:

        async def process_messages(state: WhatsAppIntelState) -> dict:
            """Process incoming messages through ingestion agent."""
            processed = []
            for msg in state.get("messages_to_process", []):
                result = await self.ingestion_agent.ainvoke({
                    "input": "Process and store this WhatsApp message",
                    "message": msg,
                })
                processed.append(result)

            return {
                "processed_messages": processed,
                "workflow_status": "messages_processed",
            }

        async def match_contacts(state: WhatsAppIntelState) -> dict:
            """Match messages to CRM contacts."""
            matches = {}
            unique_phones = set(
                m.get("phone_normalized")
                for m in state.get("processed_messages", [])
                if m.get("phone_normalized")
            )

            for phone in unique_phones:
                result = await self.mapper_agent.ainvoke({
                    "input": f"Find CRM contact for phone {phone}",
                    "phone": phone,
                })
                matches[phone] = result

            return {
                "contact_matches": matches,
                "workflow_status": "contacts_matched",
            }

        async def aggregate_conversations(state: WhatsAppIntelState) -> dict:
            """Group messages into conversation threads for analysis."""
            # Group by contact
            conversations = {}
            for msg in state.get("processed_messages", []):
                phone = msg.get("phone_normalized")
                if phone:
                    if phone not in conversations:
                        conversations[phone] = {
                            "phone": phone,
                            "contact": state.get("contact_matches", {}).get(phone, {}),
                            "messages": [],
                        }
                    conversations[phone]["messages"].append(msg)

            # Only analyze conversations with 3+ messages
            for phone, conv in conversations.items():
                conv["messages"].sort(key=lambda x: x.get("received_at", ""))

            return {
                "conversations": [
                    c for c in conversations.values()
                    if len(c["messages"]) >= 3
                ],
                "workflow_status": "conversations_aggregated",
            }

        async def analyze_conversations(state: WhatsAppIntelState) -> dict:
            """Run NLP analysis on conversation threads."""
            analyses = []
            for conv in state.get("conversations", []):
                if not conv.get("contact", {}).get("matched"):
                    continue

                result = await self.analyzer_agent.ainvoke({
                    "input": "Analyze this conversation for deal signals",
                    "conversation_thread": conv["messages"],
                    "contact_context": conv["contact"],
                })
                analyses.append({
                    "phone": conv["phone"],
                    "contact_id": conv["contact"].get("contact_id"),
                    "analysis": result,
                })

            return {
                "conversation_analyses": analyses,
                "workflow_status": "analysis_complete",
            }

        async def generate_crm_updates(state: WhatsAppIntelState) -> dict:
            """Create CRM activity records from analyses."""
            updates = []
            for analysis in state.get("conversation_analyses", []):
                if analysis.get("analysis", {}).get("signal_count", 0) > 0:
                    update = {
                        "contact_id": analysis["contact_id"],
                        "activity_type": "WhatsApp Conversation",
                        "subject": f"WhatsApp: {len(analysis['analysis'].get('signals', []))} deal signals detected",
                        "description": analysis["analysis"].get("summary", ""),
                        "signals": analysis["analysis"].get("signals", []),
                        "health_score": analysis["analysis"].get("conversation_health", {}),
                    }
                    updates.append(update)

            return {
                "crm_updates": updates,
                "workflow_status": "crm_updates_prepared",
            }

        async def generate_alerts(state: WhatsAppIntelState) -> dict:
            """Generate alerts for high-priority signals."""
            alerts = []
            for analysis in state.get("conversation_analyses", []):
                signals = analysis.get("analysis", {}).get("signals", [])

                # Alert on commitment signals
                commitment_signals = [s for s in signals if s.get("type") == "commitment"]
                if commitment_signals:
                    alerts.append({
                        "type": "commitment_detected",
                        "priority": "high",
                        "contact_id": analysis["contact_id"],
                        "message": f"Commitment language detected: {commitment_signals[0].get('excerpt', '')}",
                    })

                # Alert on competitor mentions
                competitor_signals = [s for s in signals if s.get("type") == "competitor"]
                if competitor_signals:
                    alerts.append({
                        "type": "competitor_mention",
                        "priority": "high",
                        "contact_id": analysis["contact_id"],
                        "message": f"Competitor mentioned: {competitor_signals[0].get('excerpt', '')}",
                    })

                # Alert on deals going cold
                health = analysis.get("analysis", {}).get("conversation_health", {})
                if health.get("status") == "at_risk":
                    alerts.append({
                        "type": "deal_at_risk",
                        "priority": "medium",
                        "contact_id": analysis["contact_id"],
                        "message": f"Conversation health score dropped to {health.get('score')}",
                    })

            return {
                "alerts_generated": alerts,
                "workflow_status": "complete",
            }

        def should_analyze(state: WhatsAppIntelState) -> Literal["analyze", "skip"]:
            """Only analyze if we have matched contacts."""
            matches = state.get("contact_matches", {})
            has_matches = any(m.get("matched") for m in matches.values())
            return "analyze" if has_matches else "skip"

        # Build graph
        workflow = StateGraph(WhatsAppIntelState)

        workflow.add_node("process_messages", process_messages)
        workflow.add_node("match_contacts", match_contacts)
        workflow.add_node("aggregate", aggregate_conversations)
        workflow.add_node("analyze", analyze_conversations)
        workflow.add_node("crm_updates", generate_crm_updates)
        workflow.add_node("alerts", generate_alerts)

        workflow.set_entry_point("process_messages")
        workflow.add_edge("process_messages", "match_contacts")
        workflow.add_edge("match_contacts", "aggregate")
        workflow.add_conditional_edges(
            "aggregate",
            should_analyze,
            {"analyze": "analyze", "skip": END},
        )
        workflow.add_edge("analyze", "crm_updates")
        workflow.add_edge("crm_updates", "alerts")
        workflow.add_edge("alerts", END)

        return workflow.compile()

    async def process_batch(
        self, messages: List[dict]
    ) -> dict:
        """Process a batch of WhatsApp messages."""
        result = await self.graph.ainvoke({
            "messages_to_process": messages,
            "processed_messages": [],
            "contact_matches": {},
            "conversation_analyses": [],
            "crm_updates": [],
            "alerts_generated": [],
            "workflow_status": "started",
        })
        return result`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the WhatsApp intelligence system with Docker, configure webhooks, and set up monitoring for message processing and CRM sync health.',
            toolsUsed: ['Docker', 'Prometheus', 'Grafana', 'PostgreSQL'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose for WhatsApp Intelligence System',
                description:
                  'Production deployment with webhook receiver, message processor, and monitoring.',
                code: `version: '3.8'

services:
  whatsapp-webhook:
    build:
      context: .
      dockerfile: Dockerfile.webhook
    container_name: whatsapp-webhook
    restart: unless-stopped
    environment:
      - WHATSAPP_APP_SECRET=\${WHATSAPP_APP_SECRET}
      - WHATSAPP_VERIFY_TOKEN=\${WHATSAPP_VERIFY_TOKEN}
      - DATABASE_URL=postgresql://\${POSTGRES_USER}:\${POSTGRES_PASSWORD}@postgres:5432/whatsapp
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
    ports:
      - "8443:8443"
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8443/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  message-processor:
    build:
      context: .
      dockerfile: Dockerfile.processor
    container_name: whatsapp-processor
    restart: unless-stopped
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - CRM_API_KEY=\${CRM_API_KEY}
      - CRM_API_URL=\${CRM_API_URL}
      - DATABASE_URL=postgresql://\${POSTGRES_USER}:\${POSTGRES_PASSWORD}@postgres:5432/whatsapp
      - REDIS_URL=redis://redis:6379/0
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
      - BATCH_SIZE=50
      - PROCESS_INTERVAL_SECONDS=30
    depends_on:
      - postgres
      - redis
    volumes:
      - ./models:/app/models:ro

  postgres:
    image: postgres:15-alpine
    container_name: whatsapp-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=\${POSTGRES_USER}
      - POSTGRES_PASSWORD=\${POSTGRES_PASSWORD}
      - POSTGRES_DB=whatsapp
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U \${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: whatsapp-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: whatsapp-prometheus
    restart: unless-stopped
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:10.0.0
    container_name: whatsapp-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_ADMIN_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:`,
              },
              {
                language: 'python',
                title: 'WhatsApp Intelligence API with Metrics',
                description:
                  'FastAPI service for webhook handling and message processing with Prometheus metrics.',
                code: `"""
WhatsApp Intelligence API — Webhook Handler with Monitoring
"""

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
from datetime import datetime
import hashlib
import hmac
import os

app = FastAPI(title="WhatsApp Intelligence Service")

# Prometheus metrics
MESSAGES_RECEIVED = Counter(
    "whatsapp_messages_received_total",
    "Total messages received",
    ["direction"],
)
MESSAGES_PROCESSED = Counter(
    "whatsapp_messages_processed_total",
    "Total messages processed",
    ["status"],
)
PROCESSING_LATENCY = Histogram(
    "whatsapp_processing_latency_seconds",
    "Message processing latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
CRM_MATCHES = Counter(
    "whatsapp_crm_matches_total",
    "CRM contact matches",
    ["match_type"],
)
SIGNALS_DETECTED = Counter(
    "whatsapp_signals_detected_total",
    "Deal signals detected",
    ["signal_type"],
)
ALERTS_SENT = Counter(
    "whatsapp_alerts_sent_total",
    "Alerts sent",
    ["alert_type"],
)
PENDING_MESSAGES = Gauge(
    "whatsapp_pending_messages",
    "Messages pending processing",
)


@app.get("/webhook")
async def verify_webhook(
    mode: str = "",
    token: str = "",
    challenge: str = "",
):
    """WhatsApp webhook verification endpoint."""
    verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN")
    if mode == "subscribe" and token == verify_token:
        return int(challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Receive and queue WhatsApp messages."""
    body = await request.body()
    signature = request.headers.get("x-hub-signature-256", "")

    if not verify_signature(body, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    data = await request.json()

    # Extract and queue messages
    messages = extract_messages(data)
    for msg in messages:
        MESSAGES_RECEIVED.labels(direction=msg.get("direction", "unknown")).inc()

    # Process in background
    background_tasks.add_task(process_messages_batch, messages)

    return {"status": "ok", "messages_queued": len(messages)}


def verify_signature(payload: bytes, signature: str) -> bool:
    app_secret = os.getenv("WHATSAPP_APP_SECRET", "")
    expected = hmac.new(
        app_secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


def extract_messages(data: dict) -> list:
    messages = []
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                messages.append({
                    "message_id": msg["id"],
                    "from_phone": msg["from"],
                    "body": msg.get("text", {}).get("body", ""),
                    "timestamp": msg["timestamp"],
                    "direction": "inbound",
                })
    return messages


async def process_messages_batch(messages: list):
    """Background task to process messages."""
    from orchestrator import WhatsAppIntelOrchestrator

    start_time = datetime.utcnow()
    orchestrator = get_orchestrator()

    try:
        result = await orchestrator.process_batch(messages)

        # Record metrics
        MESSAGES_PROCESSED.labels(status="success").inc(len(messages))
        PROCESSING_LATENCY.observe(
            (datetime.utcnow() - start_time).total_seconds()
        )

        # Record signal metrics
        for analysis in result.get("conversation_analyses", []):
            for signal in analysis.get("analysis", {}).get("signals", []):
                SIGNALS_DETECTED.labels(signal_type=signal.get("type", "unknown")).inc()

        # Record alert metrics
        for alert in result.get("alerts_generated", []):
            ALERTS_SENT.labels(alert_type=alert.get("type", "unknown")).inc()

    except Exception as e:
        MESSAGES_PROCESSED.labels(status="failed").inc(len(messages))
        raise


_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from orchestrator import WhatsAppIntelOrchestrator
        from agents import create_whatsapp_agents
        agents = create_whatsapp_agents()
        _orchestrator = WhatsAppIntelOrchestrator(
            ingestion_agent=agents["ingestion"],
            analyzer_agent=agents["analyzer"],
            mapper_agent=agents["mapper"],
            insight_agent=agents["insight"],
            redis_url=os.getenv("REDIS_URL"),
        )
    return _orchestrator


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")`,
              },
            ],
          },
        ],
      },
    },
  ],
};
