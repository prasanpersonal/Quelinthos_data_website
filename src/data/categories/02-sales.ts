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
    },
  ],
};
