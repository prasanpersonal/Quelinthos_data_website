import type { Category } from '../types.ts';

export const marketingCategory: Category = {
  id: 'marketing',
  number: 1,
  title: 'Marketing & Customer Acquisition',
  shortTitle: 'Marketing',
  description:
    'Fix broken attribution, consent gaps, and platform fragmentation draining your marketing budget.',
  icon: 'Megaphone',
  accentColor: 'neon-blue',

  painPoints: [
    /* ------------------------------------------------------------------ */
    /*  1 — The Attribution Void                                          */
    /* ------------------------------------------------------------------ */
    {
      id: 'attribution-void',
      number: 1,
      title: 'The Attribution Void',
      subtitle: 'Multi-Touch Attribution Failure',
      summary:
        'Your marketing spend is flying blind. Last-click attribution ignores 80% of the customer journey, misallocating millions annually.',
      tags: ['attribution', 'analytics', 'marketing-spend'],
      metrics: {
        annualCostRange: '$500K - $3M',
        roi: '6x',
        paybackPeriod: '3-4 months',
        investmentRange: '$80K - $150K',
      },

      price: {
        present: {
          title: 'Current Attribution Blindspot',
          severity: 'critical',
          description:
            'Marketing teams rely on last-click or single-touch attribution models, which systematically misrepresent the contribution of upper-funnel and mid-funnel channels. Budget allocation decisions are based on incomplete journey data.',
          bullets: [
            'Last-click attribution credits only the final touchpoint before conversion, ignoring 80%+ of the journey',
            'Paid search cannibalises organic and brand traffic credit, inflating perceived ROAS',
            'Display and content marketing appear underperforming because they rarely receive last-click credit',
            'Marketing mix decisions are driven by gut feel rather than data-backed models',
          ],
        },
        root: {
          title: 'Fragmented Tracking & Missing Identity Resolution',
          severity: 'high',
          description:
            'Attribution fails because touchpoint data lives in disconnected systems with no unified customer identity layer. Cross-device and cross-channel journeys break when there is no deterministic or probabilistic ID graph linking interactions.',
          bullets: [
            'No unified customer ID spanning web sessions, CRM events, and ad platform impressions',
            'UTM parameters are inconsistently applied across campaigns, emails, and partners',
            'Cookie deprecation and iOS ATT have eroded client-side tracking by 30-40%',
            'Server-side event capture exists only for checkout, missing mid-funnel engagement signals',
          ],
        },
        impact: {
          title: 'Misallocated Spend & Stalled Growth',
          severity: 'critical',
          description:
            'Without accurate attribution, marketing budgets over-invest in bottom-funnel channels and under-invest in awareness and consideration tactics. The compounding effect is a shrinking top-of-funnel and rising CAC.',
          bullets: [
            '25-40% of paid media budget is misallocated to channels with inflated last-click credit',
            'Customer acquisition cost (CAC) increases quarter-over-quarter as upper funnel starves',
            'Brand campaigns get cut first in budget reviews despite driving long-term pipeline',
            'Executive trust in marketing reporting erodes, leading to arbitrary budget cuts',
          ],
        },
        cost: {
          title: 'Direct Financial Exposure',
          severity: 'high',
          description:
            'The cost of attribution failure is not just wasted ad spend — it includes opportunity cost of under-funded channels, analyst time spent reconciling conflicting reports, and vendor fees for overlapping, poorly integrated tools.',
          bullets: [
            '$500K-$3M annually in misallocated media spend for mid-market companies',
            '15-20 analyst hours per week spent manually reconciling attribution data across platforms',
            '$50K-$150K/year in redundant analytics and attribution tool licenses',
            'Opportunity cost of 2-3 growth experiments per quarter that never launch due to budget misallocation',
          ],
        },
        expectedReturn: {
          title: 'Data-Driven Attribution ROI',
          severity: 'medium',
          description:
            'Implementing a multi-touch attribution model with unified identity resolution typically delivers a 6x return within the first year by reallocating spend to higher-performing channels and reducing waste.',
          bullets: [
            '15-25% improvement in marketing-sourced pipeline within 6 months',
            'CAC reduction of 20-30% through better channel mix optimisation',
            'Single source of truth for attribution eliminates 15+ hours/week of manual reconciliation',
            'Executive confidence in marketing data unlocks faster budget approval cycles',
          ],
        },
      },

      implementation: {
        overview:
          'Build a multi-touch attribution system by unifying touchpoint data into a single events table, then apply a Markov chain model to assign fractional credit across all channels in the customer journey.',
        prerequisites: [
          'Access to web analytics event stream (GA4 BigQuery export or similar)',
          'CRM or CDP with customer identity mapping',
          'Ad platform APIs configured for server-side conversion tracking',
          'Python 3.9+ with pandas, numpy, and scikit-learn',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Unify Touchpoint Data into an Attribution Events Table',
            description:
              'Create a consolidated events table that merges web analytics, CRM, and ad platform touchpoints under a single customer ID. This becomes the foundation for any attribution model.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Create Unified Attribution Events Table',
                description:
                  'Merges web sessions, CRM touches, and ad impressions into a single touchpoint timeline per customer.',
                code: `CREATE TABLE attribution_events AS
WITH web_touches AS (
  SELECT
    user_pseudo_id AS customer_id,
    event_timestamp,
    traffic_source.medium AS channel,
    traffic_source.source AS source,
    traffic_source.campaign AS campaign,
    'web_session' AS touch_type
  FROM analytics_events
  WHERE event_name = 'session_start'
),
crm_touches AS (
  SELECT
    contact_id AS customer_id,
    activity_date AS event_timestamp,
    channel_name AS channel,
    source_name AS source,
    campaign_name AS campaign,
    'crm_activity' AS touch_type
  FROM crm_activities
  WHERE activity_type IN ('email_open', 'email_click', 'demo_request')
)
SELECT * FROM web_touches
UNION ALL
SELECT * FROM crm_touches
ORDER BY customer_id, event_timestamp;`,
              },
              {
                language: 'sql',
                title: 'Build Multi-Touch Journey Sequences',
                description:
                  'Aggregates touchpoints into ordered journey paths per customer, which are required input for Markov chain attribution.',
                code: `SELECT
  customer_id,
  STRING_AGG(channel, ' > ' ORDER BY event_timestamp) AS journey_path,
  COUNT(*) AS touchpoint_count,
  MIN(event_timestamp) AS first_touch,
  MAX(event_timestamp) AS last_touch,
  CASE
    WHEN customer_id IN (SELECT customer_id FROM conversions) THEN 1
    ELSE 0
  END AS converted
FROM attribution_events
GROUP BY customer_id
HAVING touchpoint_count >= 2;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Implement Markov Chain Attribution Model',
            description:
              'Use a Markov chain approach to compute removal effects for each channel. This measures how much total conversions would drop if a channel were removed, giving fractional credit proportional to actual influence.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Markov Chain Attribution Calculator',
                description:
                  'Builds a transition matrix from journey paths and calculates each channel removal effect to assign fractional attribution credit.',
                code: `import pandas as pd
import numpy as np
from collections import defaultdict

def build_transition_matrix(journeys: pd.DataFrame) -> dict:
    transitions = defaultdict(lambda: defaultdict(int))
    for _, row in journeys.iterrows():
        path = ['start'] + row['journey_path'].split(' > ')
        path.append('conversion' if row['converted'] else 'null')
        for i in range(len(path) - 1):
            transitions[path[i]][path[i + 1]] += 1

    matrix = {}
    for state, next_states in transitions.items():
        total = sum(next_states.values())
        matrix[state] = {k: v / total for k, v in next_states.items()}
    return matrix

def removal_effect(matrix: dict, channel: str) -> float:
    reduced = {k: {nk: nv for nk, nv in v.items() if nk != channel}
                for k, v in matrix.items() if k != channel}
    for state in reduced:
        total = sum(reduced[state].values())
        if total > 0:
            reduced[state] = {k: v / total for k, v in reduced[state].items()}
    base_cr = simulate_conversion_rate(matrix)
    reduced_cr = simulate_conversion_rate(reduced)
    return (base_cr - reduced_cr) / base_cr if base_cr > 0 else 0

def simulate_conversion_rate(matrix: dict, n_sim: int = 10000) -> float:
    conversions = 0
    for _ in range(n_sim):
        state = 'start'
        for _ in range(50):
            if state in ('conversion', 'null') or state not in matrix:
                break
            probs = matrix[state]
            state = np.random.choice(list(probs.keys()), p=list(probs.values()))
        if state == 'conversion':
            conversions += 1
    return conversions / n_sim`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate the attribution pipeline end-to-end with data quality assertions in SQL and a pytest-based test suite in Python to catch regressions before they reach production dashboards.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Attribution Data Quality Assertions',
                description:
                  'Runs null checks, referential integrity, freshness thresholds, and value range checks against the attribution events table.',
                code: `-- Null check: every row must have a customer_id and channel
SELECT 'null_customer_id' AS check_name,
       COUNT(*) AS failures
FROM attribution_events
WHERE customer_id IS NULL
UNION ALL
SELECT 'null_channel',
       COUNT(*)
FROM attribution_events
WHERE channel IS NULL
UNION ALL
-- Referential integrity: all converted customers exist in conversions table
SELECT 'orphan_conversions',
       COUNT(*)
FROM (
  SELECT DISTINCT ae.customer_id
  FROM attribution_events ae
  LEFT JOIN conversions c ON ae.customer_id = c.customer_id
  WHERE c.customer_id IS NULL
    AND ae.customer_id IN (
      SELECT customer_id FROM attribution_events
      GROUP BY customer_id
      HAVING MAX(CASE WHEN channel = 'conversion' THEN 1 ELSE 0 END) = 1
    )
) orphans
UNION ALL
-- Freshness: data must be no older than 26 hours
SELECT 'stale_data',
       CASE WHEN MAX(event_timestamp) < CURRENT_TIMESTAMP - INTERVAL '26 hours'
            THEN 1 ELSE 0 END
FROM attribution_events
UNION ALL
-- Value range: touchpoint_count in journey paths must be >= 1
SELECT 'invalid_touchpoint_count',
       COUNT(*)
FROM (
  SELECT customer_id, COUNT(*) AS cnt
  FROM attribution_events
  GROUP BY customer_id
  HAVING COUNT(*) < 1
) bad_counts;`,
              },
              {
                language: 'python',
                title: 'Pytest Attribution Pipeline Validation Suite',
                description:
                  'A pytest-based test class that validates schema conformance, Markov weight bounds, and journey path integrity.',
                code: `import logging
from typing import Dict, List

import pandas as pd
import pytest

logger = logging.getLogger(__name__)


class TestAttributionPipeline:
    """Validate the attribution pipeline outputs before promotion."""

    @pytest.fixture(autouse=True)
    def setup(self, db_conn) -> None:
        self.conn = db_conn
        self.events: pd.DataFrame = pd.read_sql(
            "SELECT * FROM attribution_events LIMIT 50000", db_conn
        )
        self.weights: pd.DataFrame = pd.read_sql(
            "SELECT * FROM markov_attribution_weights", db_conn
        )
        logger.info("Loaded %d events, %d weights", len(self.events), len(self.weights))

    def test_no_null_customer_ids(self) -> None:
        nulls: int = int(self.events["customer_id"].isna().sum())
        assert nulls == 0, f"Found {nulls} null customer_id rows"

    def test_no_null_channels(self) -> None:
        nulls: int = int(self.events["channel"].isna().sum())
        assert nulls == 0, f"Found {nulls} null channel rows"

    def test_markov_weights_sum_to_one(self) -> None:
        total: float = float(self.weights["markov_weight"].sum())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"

    def test_markov_weights_positive(self) -> None:
        negatives: int = int((self.weights["markov_weight"] < 0).sum())
        assert negatives == 0, f"Found {negatives} negative Markov weights"

    def test_journey_paths_have_minimum_touches(self) -> None:
        journey_counts: pd.Series = self.events.groupby("customer_id").size()
        single_touch: int = int((journey_counts < 2).sum())
        logger.info("Customers with < 2 touches: %d", single_touch)
        assert single_touch == 0, (
            f"{single_touch} customers have fewer than 2 touchpoints"
        )

    def test_event_timestamps_within_range(self) -> None:
        max_ts = pd.to_datetime(self.events["event_timestamp"]).max()
        staleness = pd.Timestamp.utcnow() - max_ts
        assert staleness.total_seconds() < 93600, (
            f"Latest event is {staleness} old, exceeds 26h threshold"
        )`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Package the attribution pipeline as a containerised service with automated deployment, database migrations, scheduled execution, and Slack notifications on success or failure.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Attribution Pipeline Deployment Script',
                description:
                  'End-to-end deployment: validates environment, builds Docker image, runs tests, applies migrations, registers cron schedule, and sends Slack notification.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ── Environment variable validation ──────────────────────────────
REQUIRED_VARS=(
  DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD
  SLACK_WEBHOOK_URL DOCKER_REGISTRY
)
for var in "\${REQUIRED_VARS[@]}"; do
  if [[ -z "\${!var:-}" ]]; then
    echo "ERROR: \${var} is not set" >&2
    exit 1
  fi
done

APP_NAME="attribution-pipeline"
IMAGE_TAG="\${DOCKER_REGISTRY}/\${APP_NAME}:\$(git rev-parse --short HEAD)"

echo "==> Building Docker image \${IMAGE_TAG}"
docker build -t "\${IMAGE_TAG}" -f Dockerfile.attribution .

echo "==> Running pytest suite inside container"
docker run --rm \\
  -e DB_HOST -e DB_PORT -e DB_NAME -e DB_USER -e DB_PASSWORD \\
  "\${IMAGE_TAG}" pytest tests/test_attribution.py -v --tb=short

echo "==> Applying database migrations"
docker run --rm \\
  -e DB_HOST -e DB_PORT -e DB_NAME -e DB_USER -e DB_PASSWORD \\
  "\${IMAGE_TAG}" python manage.py migrate

echo "==> Pushing image to registry"
docker push "\${IMAGE_TAG}"

echo "==> Registering daily cron job"
CRON_EXPR="30 6 * * *"
(crontab -l 2>/dev/null | grep -v "\${APP_NAME}"; \\
 echo "\${CRON_EXPR} docker run --rm -e DB_HOST -e DB_PORT -e DB_NAME -e DB_USER -e DB_PASSWORD \${IMAGE_TAG} python run_attribution.py >> /var/log/\${APP_NAME}.log 2>&1") | crontab -

echo "==> Sending Slack notification"
curl -sf -X POST "\${SLACK_WEBHOOK_URL}" \\
  -H 'Content-Type: application/json' \\
  -d "{
    \\"text\\": \\"Attribution pipeline deployed successfully.\\\\nImage: \${IMAGE_TAG}\\\\nCron: \${CRON_EXPR}\\"
  }"

echo "==> Deployment complete"`,
              },
              {
                language: 'python',
                title: 'Typed Configuration Loader for Attribution Pipeline',
                description:
                  'Reads environment variables with sensible defaults, validates settings, and initialises a connection pool using dataclasses.',
                code: `import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DbConfig:
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME", "attribution"))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "pipeline"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    pool_min: int = field(default_factory=lambda: int(os.getenv("DB_POOL_MIN", "2")))
    pool_max: int = field(default_factory=lambda: int(os.getenv("DB_POOL_MAX", "10")))

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


@dataclass(frozen=True)
class AppConfig:
    db: DbConfig = field(default_factory=DbConfig)
    slack_webhook: str = field(
        default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL", "")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    simulation_count: int = field(
        default_factory=lambda: int(os.getenv("SIMULATION_COUNT", "10000"))
    )


def init_connection_pool(config: DbConfig):
    """Create a connection pool from the typed config."""
    try:
        import psycopg2.pool as pool  # type: ignore[import-untyped]

        conn_pool = pool.ThreadedConnectionPool(
            minconn=config.pool_min,
            maxconn=config.pool_max,
            dsn=config.dsn,
        )
        logger.info(
            "Connection pool created: %s:%d/%s (min=%d, max=%d)",
            config.host, config.port, config.name,
            config.pool_min, config.pool_max,
        )
        return conn_pool
    except Exception:
        logger.exception("Failed to create connection pool")
        raise`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Combine the Markov attribution weights with actual spend data to produce actionable budget reallocation recommendations, and layer on operational monitoring to detect anomalies and enforce SLA thresholds.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Attribution-Weighted Channel Performance Report',
                description:
                  'Joins Markov attribution weights with spend and revenue data to calculate true ROAS per channel.',
                code: `SELECT
  c.channel,
  c.total_spend,
  c.last_click_conversions,
  m.markov_weight,
  ROUND(m.markov_weight * t.total_revenue, 2) AS attributed_revenue,
  ROUND((m.markov_weight * t.total_revenue) / NULLIF(c.total_spend, 0), 2)
    AS true_roas,
  ROUND(c.last_click_conversions * t.avg_order_value / NULLIF(c.total_spend, 0), 2)
    AS last_click_roas,
  ROUND(
    ((m.markov_weight * t.total_revenue) / NULLIF(c.total_spend, 0))
    - (c.last_click_conversions * t.avg_order_value / NULLIF(c.total_spend, 0)),
  2) AS roas_delta
FROM channel_spend c
JOIN markov_attribution_weights m ON c.channel = m.channel
CROSS JOIN (
  SELECT SUM(revenue) AS total_revenue, AVG(revenue) AS avg_order_value
  FROM conversions
) t
ORDER BY roas_delta DESC;`,
              },
              {
                language: 'python',
                title: 'Attribution SLA Monitor & Anomaly Alerter',
                description:
                  'Checks pipeline freshness, detects anomalous weight shifts, and sends Slack alerts when SLA thresholds are breached.',
                code: `import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

SLACK_WEBHOOK: str = os.getenv("SLACK_WEBHOOK_URL", "")
FRESHNESS_SLA_HOURS: int = 26
WEIGHT_SHIFT_THRESHOLD: float = 0.15


@dataclass
class SLACheckResult:
    check_name: str
    passed: bool
    detail: str


def check_freshness(db_conn) -> SLACheckResult:
    row = db_conn.execute(
        "SELECT MAX(event_timestamp) AS latest FROM attribution_events"
    ).fetchone()
    if row is None or row[0] is None:
        return SLACheckResult("freshness", False, "No events found")
    staleness: timedelta = datetime.now(timezone.utc) - row[0]
    passed: bool = staleness.total_seconds() < FRESHNESS_SLA_HOURS * 3600
    return SLACheckResult(
        "freshness", passed,
        f"Latest event: {row[0].isoformat()}, staleness: {staleness}"
    )


def check_weight_anomalies(db_conn) -> SLACheckResult:
    current: pd.DataFrame = pd.read_sql(
        "SELECT channel, markov_weight FROM markov_attribution_weights", db_conn
    )
    previous: pd.DataFrame = pd.read_sql(
        "SELECT channel, markov_weight FROM markov_attribution_weights_previous", db_conn
    )
    merged: pd.DataFrame = current.merge(previous, on="channel", suffixes=("_cur", "_prev"))
    merged["shift"] = abs(merged["markov_weight_cur"] - merged["markov_weight_prev"])
    anomalies: pd.DataFrame = merged[merged["shift"] > WEIGHT_SHIFT_THRESHOLD]
    if anomalies.empty:
        return SLACheckResult("weight_anomaly", True, "No anomalous shifts")
    detail: str = "; ".join(
        f"{r['channel']}: {r['shift']:.2%}" for _, r in anomalies.iterrows()
    )
    return SLACheckResult("weight_anomaly", False, f"Anomalies: {detail}")


def send_alert(results: List[SLACheckResult]) -> None:
    failures: List[SLACheckResult] = [r for r in results if not r.passed]
    if not failures:
        logger.info("All SLA checks passed")
        return
    blocks: List[str] = [
        f"*{r.check_name}*: {r.detail}" for r in failures
    ]
    payload = {
        "text": f"Attribution Pipeline Alert — {len(failures)} SLA breach(es):\\n"
        + "\\n".join(blocks)
    }
    if SLACK_WEBHOOK:
        requests.post(SLACK_WEBHOOK, json=payload, timeout=10)
    logger.warning("Alerts sent: %s", payload["text"])


def run_monitoring(db_conn) -> List[SLACheckResult]:
    results: List[SLACheckResult] = [
        check_freshness(db_conn),
        check_weight_anomalies(db_conn),
    ]
    send_alert(results)
    return results`,
              },
            ],
          },
        ],
        toolsUsed: [
          'BigQuery / Snowflake / Redshift (SQL warehouse)',
          'Python (pandas, numpy)',
          'GA4 BigQuery Export',
          'CRM API (HubSpot / Salesforce)',
          'dbt (for transformation orchestration)',
          'pytest',
          'Docker',
          'GitHub Actions',
          'cron / Airflow',
          'Slack API',
        ],
      },
    },

    /* ------------------------------------------------------------------ */
    /*  2 — The GDPR Consent Gap                                          */
    /* ------------------------------------------------------------------ */
    {
      id: 'gdpr-consent-gap',
      number: 2,
      title: 'The GDPR Consent Gap',
      subtitle: 'Consent Management Data Chaos',
      summary:
        'Your consent records are scattered across 5+ systems. One audit could cost you 4% of global revenue.',
      tags: ['gdpr', 'compliance', 'consent-management'],
      metrics: {
        annualCostRange: '$200K - $2M',
        roi: '10x',
        paybackPeriod: '2-3 months',
        investmentRange: '$50K - $100K',
      },

      price: {
        present: {
          title: 'Fragmented Consent Records',
          severity: 'critical',
          description:
            'Consent data is siloed across your CMP, CRM, email platform, analytics tools, and ad tech stack. No single system can answer the question: "What did this user consent to, and when?" Audit readiness is near zero.',
          bullets: [
            'Consent preferences stored in 5-8 disconnected systems with no single source of truth',
            'CMP consent signals are not propagated to downstream data processors in real time',
            'Email marketing continues sending to users who revoked consent on the website CMP',
            'No audit trail showing consent state at the time data was collected or processed',
          ],
        },
        root: {
          title: 'Missing Centralised Consent Ledger',
          severity: 'high',
          description:
            'The root cause is the absence of a centralised, immutable consent ledger that acts as the authoritative record. Systems were integrated independently, each storing consent locally without a synchronisation layer.',
          bullets: [
            'CMP was deployed as a frontend widget with no backend integration to the data warehouse',
            'CRM consent fields are updated manually by sales reps, not synced from the CMP',
            'Ad platforms rely on their own consent signals, disconnected from your first-party records',
            'No event-driven architecture to propagate consent changes across all processors within seconds',
          ],
        },
        impact: {
          title: 'Regulatory Exposure & Data Trust Erosion',
          severity: 'critical',
          description:
            'GDPR fines can reach 4% of global annual revenue. Beyond fines, consent failures erode customer trust, trigger data subject access request (DSAR) bottlenecks, and create legal liability for every data-sharing agreement.',
          bullets: [
            'GDPR Article 83 penalties up to 4% of global turnover or EUR 20M, whichever is higher',
            'Average DSAR response takes 12-15 business days instead of the required 30-day maximum, with fragmented consent data being the primary bottleneck',
            'Data sharing agreements with partners become legally void without provable consent chains',
            'Customer trust scores drop measurably after consent-related incidents become public',
          ],
        },
        cost: {
          title: 'Compliance & Operational Costs',
          severity: 'high',
          description:
            'The ongoing cost of consent fragmentation includes legal counsel, manual DSAR processing, redundant CMP licenses, and the opportunity cost of marketing campaigns that cannot run due to consent uncertainty.',
          bullets: [
            '$200K-$2M annually in combined legal, compliance, and operational overhead',
            '40+ hours per month spent manually processing DSARs across fragmented systems',
            '$30K-$80K/year in overlapping consent management tool subscriptions',
            '10-15% of planned marketing campaigns delayed or cancelled due to consent data gaps',
          ],
        },
        expectedReturn: {
          title: 'Unified Consent Infrastructure ROI',
          severity: 'medium',
          description:
            'A centralised consent ledger with real-time propagation eliminates audit risk, automates DSAR fulfilment, and unlocks marketing campaigns that were previously blocked by consent uncertainty.',
          bullets: [
            'DSAR response time drops from 12 days to under 24 hours with automated consent lookups',
            'Audit readiness goes from weeks of preparation to instant, always-on compliance reporting',
            '10-15% increase in marketable audience as consent records are properly unified and validated',
            'Legal and compliance team capacity freed up by 30-40% for strategic work',
          ],
        },
      },

      implementation: {
        overview:
          'Build a centralised consent ledger in your data warehouse that ingests consent events from all sources, maintains an immutable audit trail, and exposes a real-time sync pipeline to propagate consent changes to all downstream processors.',
        prerequisites: [
          'Access to CMP webhook or API (OneTrust, Cookiebot, or similar)',
          'CRM API access for consent field synchronisation',
          'Data warehouse with append-only table support (BigQuery, Snowflake)',
          'Python 3.9+ with requests library for API integration',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Create the Centralised Consent Ledger',
            description:
              'Design an immutable, append-only consent events table that records every consent change with full context. This becomes the single source of truth for all consent queries and audits.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Consent Ledger Schema and Audit View',
                description:
                  'Creates the append-only consent events table and a view that returns the current consent state per user per purpose.',
                code: `CREATE TABLE consent_ledger (
  event_id STRING NOT NULL,
  user_id STRING NOT NULL,
  consent_purpose STRING NOT NULL,
  consent_status STRING NOT NULL,  -- 'granted' | 'denied' | 'withdrawn'
  source_system STRING NOT NULL,
  collection_method STRING NOT NULL,
  ip_country STRING,
  recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  raw_payload JSON
);

-- Current consent state: latest event per user per purpose
CREATE VIEW current_consent AS
SELECT DISTINCT ON (user_id, consent_purpose)
  user_id,
  consent_purpose,
  consent_status,
  source_system,
  collection_method,
  recorded_at AS last_updated
FROM consent_ledger
ORDER BY user_id, consent_purpose, recorded_at DESC;`,
              },
              {
                language: 'sql',
                title: 'Consent Audit Report Query',
                description:
                  'Generates a compliance audit report showing consent coverage gaps and systems with stale consent records.',
                code: `SELECT
  s.system_name,
  COUNT(DISTINCT cl.user_id) AS users_with_consent,
  COUNT(DISTINCT u.user_id) AS total_active_users,
  ROUND(
    COUNT(DISTINCT cl.user_id) * 100.0
    / NULLIF(COUNT(DISTINCT u.user_id), 0), 1
  ) AS consent_coverage_pct,
  COUNT(CASE WHEN cl.consent_status = 'granted' THEN 1 END) AS granted,
  COUNT(CASE WHEN cl.consent_status = 'denied' THEN 1 END) AS denied,
  COUNT(CASE WHEN cl.consent_status = 'withdrawn' THEN 1 END) AS withdrawn,
  MAX(cl.recorded_at) AS latest_record,
  CASE
    WHEN MAX(cl.recorded_at) < CURRENT_TIMESTAMP - INTERVAL '90 days'
    THEN 'STALE'
    ELSE 'CURRENT'
  END AS freshness_status
FROM downstream_systems s
LEFT JOIN consent_ledger cl ON s.system_name = cl.source_system
LEFT JOIN active_users u ON u.primary_system = s.system_name
GROUP BY s.system_name
ORDER BY consent_coverage_pct ASC;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Build the Real-Time Consent Sync Pipeline',
            description:
              'Create a Python service that listens for CMP webhook events, writes to the consent ledger, and propagates consent changes to all downstream systems (CRM, email platform, ad platforms) within seconds.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Consent Event Ingestion and Propagation Service',
                description:
                  'Receives consent webhook events from the CMP, normalises them, writes to the ledger, and fans out updates to downstream systems via their APIs.',
                code: `import uuid
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Literal

import requests

logger = logging.getLogger(__name__)

DOWNSTREAM_ENDPOINTS = {
    "crm": "https://crm.internal/api/consent",
    "email": "https://email.internal/api/consent",
    "analytics": "https://analytics.internal/api/consent",
}

@dataclass
class ConsentEvent:
    user_id: str
    purpose: str
    status: Literal["granted", "denied", "withdrawn"]
    source: str
    method: str
    ip_country: str
    timestamp: datetime

def process_cmp_webhook(payload: dict) -> ConsentEvent:
    return ConsentEvent(
        user_id=payload["userId"],
        purpose=payload["purpose"],
        status=payload["action"],
        source="cmp_webhook",
        method=payload.get("collectionMethod", "banner"),
        ip_country=payload.get("country", "unknown"),
        timestamp=datetime.now(timezone.utc),
    )

def write_to_ledger(event: ConsentEvent, db_conn) -> str:
    event_id = str(uuid.uuid4())
    db_conn.execute(
        "INSERT INTO consent_ledger VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NULL)",
        (event_id, event.user_id, event.purpose, event.status,
         event.source, event.method, event.ip_country, event.timestamp),
    )
    return event_id

def propagate_to_downstream(event: ConsentEvent) -> dict:
    results = {}
    for system, url in DOWNSTREAM_ENDPOINTS.items():
        try:
            resp = requests.post(url, json={
                "userId": event.user_id,
                "purpose": event.purpose,
                "status": event.status,
                "updatedAt": event.timestamp.isoformat(),
            }, timeout=5)
            results[system] = resp.status_code
        except requests.RequestException as exc:
            logger.error("Failed to sync %s: %s", system, exc)
            results[system] = "error"
    return results`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate the consent pipeline with SQL data quality assertions and a pytest suite that ensures ledger integrity, propagation correctness, and DSAR readiness.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Consent Ledger Data Quality Assertions',
                description:
                  'Runs null checks on mandatory fields, referential integrity between ledger and downstream systems, freshness thresholds, and valid consent status values.',
                code: `-- Null checks on mandatory columns
SELECT 'null_user_id' AS check_name, COUNT(*) AS failures
FROM consent_ledger WHERE user_id IS NULL
UNION ALL
SELECT 'null_consent_purpose', COUNT(*)
FROM consent_ledger WHERE consent_purpose IS NULL
UNION ALL
SELECT 'null_consent_status', COUNT(*)
FROM consent_ledger WHERE consent_status IS NULL
UNION ALL
-- Referential integrity: every source_system must exist in downstream_systems
SELECT 'orphan_source_system', COUNT(*)
FROM (
  SELECT DISTINCT cl.source_system
  FROM consent_ledger cl
  LEFT JOIN downstream_systems ds ON cl.source_system = ds.system_name
  WHERE ds.system_name IS NULL
) orphans
UNION ALL
-- Freshness: ledger must have received events within the last 4 hours
SELECT 'stale_ledger',
       CASE WHEN MAX(recorded_at) < CURRENT_TIMESTAMP - INTERVAL '4 hours'
            THEN 1 ELSE 0 END
FROM consent_ledger
UNION ALL
-- Value range: consent_status must be one of the allowed values
SELECT 'invalid_consent_status', COUNT(*)
FROM consent_ledger
WHERE consent_status NOT IN ('granted', 'denied', 'withdrawn');`,
              },
              {
                language: 'python',
                title: 'Pytest Consent Pipeline Validation Suite',
                description:
                  'A pytest-based test class that validates ledger schema, consent propagation latency, and DSAR query correctness.',
                code: `import logging
from typing import List, Optional

import pandas as pd
import pytest

logger = logging.getLogger(__name__)


class TestConsentPipeline:
    """Validate consent ledger integrity and downstream sync."""

    @pytest.fixture(autouse=True)
    def setup(self, db_conn) -> None:
        self.conn = db_conn
        self.ledger: pd.DataFrame = pd.read_sql(
            "SELECT * FROM consent_ledger ORDER BY recorded_at DESC LIMIT 50000",
            db_conn,
        )
        logger.info("Loaded %d ledger rows for validation", len(self.ledger))

    def test_no_null_user_ids(self) -> None:
        nulls: int = int(self.ledger["user_id"].isna().sum())
        assert nulls == 0, f"Found {nulls} null user_id rows"

    def test_no_null_consent_purpose(self) -> None:
        nulls: int = int(self.ledger["consent_purpose"].isna().sum())
        assert nulls == 0, f"Found {nulls} null consent_purpose rows"

    def test_valid_consent_statuses(self) -> None:
        allowed: set = {"granted", "denied", "withdrawn"}
        invalid: pd.Series = self.ledger[
            ~self.ledger["consent_status"].isin(allowed)
        ]["consent_status"]
        assert invalid.empty, f"Invalid statuses found: {invalid.unique().tolist()}"

    def test_ledger_freshness(self) -> None:
        max_ts = pd.to_datetime(self.ledger["recorded_at"]).max()
        staleness = pd.Timestamp.utcnow() - max_ts
        assert staleness.total_seconds() < 14400, (
            f"Ledger stale: latest record is {staleness} old (>4h)"
        )

    def test_source_system_referential_integrity(self) -> None:
        systems: pd.DataFrame = pd.read_sql(
            "SELECT system_name FROM downstream_systems", self.conn
        )
        known: set = set(systems["system_name"].tolist())
        ledger_sources: set = set(self.ledger["source_system"].unique())
        orphans: set = ledger_sources - known
        assert not orphans, f"Unknown source systems: {orphans}"

    def test_dsar_query_returns_results(self) -> None:
        sample_user: Optional[str] = self.ledger["user_id"].iloc[0] if len(self.ledger) > 0 else None
        if sample_user is None:
            pytest.skip("No data to test DSAR query")
        result: pd.DataFrame = pd.read_sql(
            f"SELECT * FROM consent_ledger WHERE user_id = %s",
            self.conn,
            params=(sample_user,),
        )
        assert len(result) > 0, f"DSAR query returned 0 rows for {sample_user}"`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Package the consent sync service as a containerised application with automated deployment, database migrations, scheduled ledger reconciliation, and Slack notifications.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Consent Pipeline Deployment Script',
                description:
                  'End-to-end deployment: validates environment, builds Docker image, runs tests, applies migrations, registers cron schedule, and sends Slack notification.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ── Environment variable validation ──────────────────────────────
REQUIRED_VARS=(
  DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD
  CMP_WEBHOOK_SECRET SLACK_WEBHOOK_URL DOCKER_REGISTRY
)
for var in "\${REQUIRED_VARS[@]}"; do
  if [[ -z "\${!var:-}" ]]; then
    echo "ERROR: \${var} is not set" >&2
    exit 1
  fi
done

APP_NAME="consent-pipeline"
IMAGE_TAG="\${DOCKER_REGISTRY}/\${APP_NAME}:\$(git rev-parse --short HEAD)"

echo "==> Building Docker image \${IMAGE_TAG}"
docker build -t "\${IMAGE_TAG}" -f Dockerfile.consent .

echo "==> Running pytest suite inside container"
docker run --rm \\
  -e DB_HOST -e DB_PORT -e DB_NAME -e DB_USER -e DB_PASSWORD \\
  "\${IMAGE_TAG}" pytest tests/test_consent.py -v --tb=short

echo "==> Applying database migrations"
docker run --rm \\
  -e DB_HOST -e DB_PORT -e DB_NAME -e DB_USER -e DB_PASSWORD \\
  "\${IMAGE_TAG}" python manage.py migrate

echo "==> Pushing image to registry"
docker push "\${IMAGE_TAG}"

echo "==> Registering hourly consent reconciliation cron job"
CRON_EXPR="0 * * * *"
(crontab -l 2>/dev/null | grep -v "\${APP_NAME}"; \\
 echo "\${CRON_EXPR} docker run --rm -e DB_HOST -e DB_PORT -e DB_NAME -e DB_USER -e DB_PASSWORD -e CMP_WEBHOOK_SECRET \${IMAGE_TAG} python reconcile_consent.py >> /var/log/\${APP_NAME}.log 2>&1") | crontab -

echo "==> Sending Slack notification"
curl -sf -X POST "\${SLACK_WEBHOOK_URL}" \\
  -H 'Content-Type: application/json' \\
  -d "{
    \\"text\\": \\"Consent pipeline deployed successfully.\\\\nImage: \${IMAGE_TAG}\\\\nCron: \${CRON_EXPR}\\"
  }"

echo "==> Deployment complete"`,
              },
              {
                language: 'python',
                title: 'Typed Configuration Loader for Consent Pipeline',
                description:
                  'Reads environment variables with sensible defaults, validates settings, and initialises a connection pool using dataclasses.',
                code: `import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DbConfig:
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME", "consent"))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "pipeline"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    pool_min: int = field(default_factory=lambda: int(os.getenv("DB_POOL_MIN", "2")))
    pool_max: int = field(default_factory=lambda: int(os.getenv("DB_POOL_MAX", "10")))

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


@dataclass(frozen=True)
class ConsentConfig:
    db: DbConfig = field(default_factory=DbConfig)
    cmp_webhook_secret: str = field(
        default_factory=lambda: os.getenv("CMP_WEBHOOK_SECRET", "")
    )
    slack_webhook: str = field(
        default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL", "")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    propagation_timeout_sec: int = field(
        default_factory=lambda: int(os.getenv("PROPAGATION_TIMEOUT", "5"))
    )
    downstream_endpoints: Dict[str, str] = field(default_factory=lambda: {
        "crm": os.getenv("CRM_CONSENT_URL", "https://crm.internal/api/consent"),
        "email": os.getenv("EMAIL_CONSENT_URL", "https://email.internal/api/consent"),
        "analytics": os.getenv("ANALYTICS_CONSENT_URL", "https://analytics.internal/api/consent"),
    })


def init_connection_pool(config: DbConfig):
    """Create a connection pool from the typed config."""
    try:
        import psycopg2.pool as pool  # type: ignore[import-untyped]

        conn_pool = pool.ThreadedConnectionPool(
            minconn=config.pool_min,
            maxconn=config.pool_max,
            dsn=config.dsn,
        )
        logger.info(
            "Connection pool created: %s:%d/%s (min=%d, max=%d)",
            config.host, config.port, config.name,
            config.pool_min, config.pool_max,
        )
        return conn_pool
    except Exception:
        logger.exception("Failed to create connection pool")
        raise`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Use the centralised ledger to auto-generate DSAR response packages and layer on operational monitoring to detect consent propagation failures, coverage gaps, and SLA breaches.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'DSAR Consent History Export',
                description:
                  'Retrieves the complete consent history for a given user, formatted for regulatory disclosure.',
                code: `SELECT
  event_id,
  consent_purpose,
  consent_status,
  source_system,
  collection_method,
  ip_country,
  recorded_at,
  LEAD(recorded_at) OVER (
    PARTITION BY consent_purpose ORDER BY recorded_at
  ) AS superseded_at,
  CASE
    WHEN LEAD(recorded_at) OVER (
      PARTITION BY consent_purpose ORDER BY recorded_at
    ) IS NULL THEN 'CURRENT'
    ELSE 'SUPERSEDED'
  END AS record_status
FROM consent_ledger
WHERE user_id = :requested_user_id
ORDER BY consent_purpose, recorded_at;`,
              },
              {
                language: 'python',
                title: 'Consent SLA Monitor & Propagation Alerter',
                description:
                  'Monitors consent propagation latency, detects coverage gaps across downstream systems, and sends Slack alerts when SLA thresholds are breached.',
                code: `import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List

import requests
import pandas as pd

logger = logging.getLogger(__name__)

SLACK_WEBHOOK: str = os.getenv("SLACK_WEBHOOK_URL", "")
PROPAGATION_SLA_SECONDS: int = 30
COVERAGE_THRESHOLD_PCT: float = 95.0


@dataclass
class SLACheckResult:
    check_name: str
    passed: bool
    detail: str


def check_propagation_latency(db_conn) -> SLACheckResult:
    """Verify consent events propagate to all downstream systems within SLA."""
    df: pd.DataFrame = pd.read_sql(
        """
        SELECT source_system,
               AVG(EXTRACT(EPOCH FROM (propagated_at - recorded_at))) AS avg_latency,
               MAX(EXTRACT(EPOCH FROM (propagated_at - recorded_at))) AS max_latency
        FROM consent_propagation_log
        WHERE recorded_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
        GROUP BY source_system
        """,
        db_conn,
    )
    breaches: pd.DataFrame = df[df["max_latency"] > PROPAGATION_SLA_SECONDS]
    if breaches.empty:
        return SLACheckResult("propagation_latency", True, "All systems within SLA")
    detail: str = "; ".join(
        f"{r['source_system']}: max {r['max_latency']:.0f}s"
        for _, r in breaches.iterrows()
    )
    return SLACheckResult("propagation_latency", False, f"SLA breaches: {detail}")


def check_consent_coverage(db_conn) -> SLACheckResult:
    """Ensure consent coverage across active users meets the threshold."""
    row = db_conn.execute(
        """
        SELECT
          COUNT(DISTINCT cc.user_id) * 100.0 / NULLIF(COUNT(DISTINCT au.user_id), 0)
            AS coverage_pct
        FROM active_users au
        LEFT JOIN current_consent cc ON au.user_id = cc.user_id
        """
    ).fetchone()
    pct: float = float(row[0]) if row and row[0] else 0.0
    passed: bool = pct >= COVERAGE_THRESHOLD_PCT
    return SLACheckResult(
        "consent_coverage", passed, f"Coverage: {pct:.1f}% (threshold: {COVERAGE_THRESHOLD_PCT}%)"
    )


def send_alert(results: List[SLACheckResult]) -> None:
    failures: List[SLACheckResult] = [r for r in results if not r.passed]
    if not failures:
        logger.info("All consent SLA checks passed")
        return
    blocks: List[str] = [f"*{r.check_name}*: {r.detail}" for r in failures]
    payload = {
        "text": f"Consent Pipeline Alert — {len(failures)} SLA breach(es):\\n"
        + "\\n".join(blocks)
    }
    if SLACK_WEBHOOK:
        requests.post(SLACK_WEBHOOK, json=payload, timeout=10)
    logger.warning("Alerts sent: %s", payload["text"])


def run_consent_monitoring(db_conn) -> List[SLACheckResult]:
    results: List[SLACheckResult] = [
        check_propagation_latency(db_conn),
        check_consent_coverage(db_conn),
    ]
    send_alert(results)
    return results`,
              },
            ],
          },
        ],
        toolsUsed: [
          'BigQuery / Snowflake (consent ledger)',
          'Python (webhook service)',
          'OneTrust / Cookiebot CMP (webhook source)',
          'CRM API (HubSpot / Salesforce)',
          'Cloud Functions / AWS Lambda (event processing)',
          'pytest',
          'Docker',
          'GitHub Actions',
          'cron / Airflow',
          'Slack API',
        ],
      },
    },

    /* ------------------------------------------------------------------ */
    /*  3 — Platform Fragmentation                                        */
    /* ------------------------------------------------------------------ */
    {
      id: 'platform-fragmentation',
      number: 3,
      title: 'Platform Fragmentation',
      subtitle: 'Siloed Marketing Data Across Platforms',
      summary:
        'Google Ads, Meta, TikTok, LinkedIn — each platform tells a different story. You have no single source of truth for marketing performance.',
      tags: ['integration', 'marketing-platforms', 'data-consolidation'],
      metrics: {
        annualCostRange: '$300K - $1.5M',
        roi: '5x',
        paybackPeriod: '2-3 months',
        investmentRange: '$60K - $120K',
      },

      price: {
        present: {
          title: 'Conflicting Platform Metrics',
          severity: 'high',
          description:
            'Every ad platform reports its own version of conversions, reach, and ROAS. Totals never reconcile. Weekly reporting requires manual exports from 4-6 dashboards, and the numbers presented to leadership are best guesses.',
          bullets: [
            'Google Ads, Meta, TikTok, and LinkedIn each claim credit for the same conversions',
            'Sum of platform-reported conversions exceeds actual conversions by 40-60%',
            'Reporting requires 8-12 hours per week of manual CSV exports and spreadsheet merging',
            'No standardised taxonomy for campaign naming, making cross-platform comparison impossible',
          ],
        },
        root: {
          title: 'No Unified Data Layer Across Platforms',
          severity: 'high',
          description:
            'Each marketing platform was onboarded independently with its own tracking pixel, conversion event definitions, and attribution window. There is no shared data model or automated ingestion pipeline to normalise metrics.',
          bullets: [
            'Platform pixels fire independently with no deduplication against server-side events',
            'Conversion definitions differ: Google counts 7-day click-through, Meta counts 1-day view-through',
            'Campaign naming conventions vary by platform and by the team member who created the campaign',
            'No automated data pipeline — teams rely on native platform dashboards or manual exports',
          ],
        },
        impact: {
          title: 'Slow Decisions & Budget Waste',
          severity: 'high',
          description:
            'Without consolidated reporting, budget reallocation decisions lag by weeks. High-performing channels are underfunded while underperformers continue spending. Cross-channel synergies are invisible.',
          bullets: [
            'Budget reallocation decisions are delayed by 2-3 weeks due to manual reporting cycles',
            'Cross-channel synergies (e.g., YouTube awareness driving search conversions) are invisible',
            'A/B test results from one platform cannot be correlated with lift on another',
            'CMO-level reporting is inconsistent, eroding confidence in the marketing function',
          ],
        },
        cost: {
          title: 'Operational and Opportunity Costs',
          severity: 'medium',
          description:
            'The direct cost includes analyst time, redundant tooling, and wasted spend on underperforming channels. The opportunity cost is the inability to execute real-time cross-channel optimisation.',
          bullets: [
            '$300K-$1.5M annually in wasted spend from delayed optimisation and double-counted conversions',
            '50-80 analyst hours per month on manual data consolidation and reporting',
            '$20K-$60K/year on BI tools and connectors that partially solve the problem',
            'Missed 10-20% efficiency gains from cross-channel budget optimisation',
          ],
        },
        expectedReturn: {
          title: 'Consolidated Marketing Data Platform ROI',
          severity: 'medium',
          description:
            'A unified marketing data model with automated ingestion from all platforms delivers a single source of truth, enabling real-time optimisation and eliminating hours of manual reporting.',
          bullets: [
            'Reporting time drops from 8-12 hours/week to 30 minutes with automated dashboards',
            '15-20% improvement in blended ROAS through cross-channel budget optimisation',
            'Single source of truth eliminates conflicting numbers in leadership meetings',
            'Enables real-time alerts when platform spend pacing deviates from targets',
          ],
        },
      },

      implementation: {
        overview:
          'Build an automated pipeline that ingests campaign data from all marketing platforms into a unified data model with standardised metrics, enabling cross-platform comparison and consolidated reporting.',
        prerequisites: [
          'API access to Google Ads, Meta Marketing, TikTok Ads, and LinkedIn Campaign Manager',
          'Data warehouse (BigQuery, Snowflake, or Redshift)',
          'Python 3.9+ with platform SDK libraries',
          'Orchestration tool (Airflow, Prefect, or cron for simple setups)',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Design the Unified Marketing Data Model',
            description:
              'Create a normalised schema that maps each platform\'s metrics to a common taxonomy. This model must handle differences in attribution windows, metric definitions, and campaign hierarchy.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Unified Marketing Performance Schema',
                description:
                  'Creates the standardised tables for cross-platform marketing data, including a platform-agnostic campaign hierarchy and normalised metrics.',
                code: `CREATE TABLE unified_campaigns (
  campaign_key STRING NOT NULL,
  platform STRING NOT NULL,
  platform_campaign_id STRING NOT NULL,
  campaign_name STRING,
  normalised_name STRING,
  objective STRING,
  channel_group STRING,
  geo_target STRING,
  start_date DATE,
  end_date DATE,
  PRIMARY KEY (campaign_key)
);

CREATE TABLE unified_daily_metrics (
  campaign_key STRING NOT NULL,
  metric_date DATE NOT NULL,
  impressions BIGINT DEFAULT 0,
  clicks BIGINT DEFAULT 0,
  spend NUMERIC(12,2) DEFAULT 0,
  platform_conversions BIGINT DEFAULT 0,
  platform_revenue NUMERIC(12,2) DEFAULT 0,
  verified_conversions BIGINT DEFAULT 0,
  verified_revenue NUMERIC(12,2) DEFAULT 0,
  ctr NUMERIC(8,6) GENERATED ALWAYS AS (
    clicks::NUMERIC / NULLIF(impressions, 0)
  ) STORED,
  cpc NUMERIC(10,2) GENERATED ALWAYS AS (
    spend / NULLIF(clicks, 0)
  ) STORED,
  true_roas NUMERIC(10,2) GENERATED ALWAYS AS (
    verified_revenue / NULLIF(spend, 0)
  ) STORED,
  PRIMARY KEY (campaign_key, metric_date)
);`,
              },
              {
                language: 'sql',
                title: 'Cross-Platform Performance Dashboard Query',
                description:
                  'Aggregates metrics across all platforms with standardised calculations for executive reporting.',
                code: `SELECT
  uc.platform,
  uc.channel_group,
  COUNT(DISTINCT uc.campaign_key) AS active_campaigns,
  SUM(m.impressions) AS total_impressions,
  SUM(m.clicks) AS total_clicks,
  SUM(m.spend) AS total_spend,
  SUM(m.platform_conversions) AS platform_reported_conv,
  SUM(m.verified_conversions) AS deduplicated_conv,
  ROUND(
    (SUM(m.platform_conversions) - SUM(m.verified_conversions))
    * 100.0 / NULLIF(SUM(m.platform_conversions), 0), 1
  ) AS overcount_pct,
  ROUND(SUM(m.verified_revenue) / NULLIF(SUM(m.spend), 0), 2)
    AS true_blended_roas,
  ROUND(SUM(m.spend) / NULLIF(SUM(m.verified_conversions), 0), 2)
    AS true_cpa
FROM unified_daily_metrics m
JOIN unified_campaigns uc ON m.campaign_key = uc.campaign_key
WHERE m.metric_date BETWEEN :start_date AND :end_date
GROUP BY uc.platform, uc.channel_group
ORDER BY total_spend DESC;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Build the Multi-Platform Ingestion Pipeline',
            description:
              'Create a Python pipeline that connects to each platform API, extracts campaign and performance data, normalises it to the unified schema, and loads it into the data warehouse on a daily schedule.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Multi-Platform Data Ingestion Pipeline',
                description:
                  'Modular pipeline that fetches data from Google Ads, Meta, TikTok, and LinkedIn APIs, normalises metrics, and loads into the unified schema.',
                code: `import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import date
from typing import Iterator

logger = logging.getLogger(__name__)

@dataclass
class UnifiedMetricRow:
    campaign_key: str
    platform: str
    platform_campaign_id: str
    campaign_name: str
    metric_date: date
    impressions: int
    clicks: int
    spend: float
    platform_conversions: int
    platform_revenue: float

class PlatformConnector(ABC):
    @abstractmethod
    def fetch_daily_metrics(self, report_date: date) -> Iterator[UnifiedMetricRow]:
        pass

class GoogleAdsConnector(PlatformConnector):
    def __init__(self, client, customer_id: str):
        self.client = client
        self.customer_id = customer_id

    def fetch_daily_metrics(self, report_date: date) -> Iterator[UnifiedMetricRow]:
        query = f"""
            SELECT campaign.id, campaign.name, metrics.impressions,
                   metrics.clicks, metrics.cost_micros, metrics.conversions,
                   metrics.conversions_value
            FROM campaign
            WHERE segments.date = '{report_date.isoformat()}'
        """
        response = self.client.search(customer_id=self.customer_id, query=query)
        for row in response:
            cid = str(row.campaign.id)
            yield UnifiedMetricRow(
                campaign_key=hashlib.md5(f"google_{cid}".encode()).hexdigest(),
                platform="google_ads",
                platform_campaign_id=cid,
                campaign_name=row.campaign.name,
                metric_date=report_date,
                impressions=row.metrics.impressions,
                clicks=row.metrics.clicks,
                spend=row.metrics.cost_micros / 1_000_000,
                platform_conversions=int(row.metrics.conversions),
                platform_revenue=float(row.metrics.conversions_value),
            )`,
              },
              {
                language: 'python',
                title: 'Pipeline Orchestrator with Deduplication',
                description:
                  'Coordinates ingestion across all platforms, applies campaign name normalisation, and deduplicates conversions against server-side event data.',
                code: `import re
from datetime import date, timedelta
from typing import List

def normalise_campaign_name(raw_name: str) -> str:
    """Standardise campaign names across platforms for grouping."""
    name = raw_name.lower().strip()
    name = re.sub(r'[^a-z0-9_\\-/]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

def deduplicate_conversions(
    platform_rows: List[UnifiedMetricRow],
    server_events: dict,
) -> List[dict]:
    """Cross-reference platform conversions against server-side events."""
    results = []
    for row in platform_rows:
        row_dict = asdict(row)
        row_dict['normalised_name'] = normalise_campaign_name(row.campaign_name)
        key = (row.platform, row.platform_campaign_id, str(row.metric_date))
        verified = server_events.get(key, {})
        row_dict['verified_conversions'] = verified.get('conversions', 0)
        row_dict['verified_revenue'] = verified.get('revenue', 0.0)
        results.append(row_dict)
    return results

def run_daily_pipeline(
    connectors: List[PlatformConnector],
    server_events: dict,
    db_conn,
    report_date: date = None,
):
    report_date = report_date or date.today() - timedelta(days=1)
    all_rows = []
    for connector in connectors:
        try:
            rows = list(connector.fetch_daily_metrics(report_date))
            logger.info("%s: fetched %d rows", type(connector).__name__, len(rows))
            all_rows.extend(rows)
        except Exception as exc:
            logger.error("Failed: %s — %s", type(connector).__name__, exc)

    enriched = deduplicate_conversions(all_rows, server_events)
    db_conn.bulk_insert("unified_daily_metrics", enriched)
    logger.info("Loaded %d rows for %s", len(enriched), report_date)`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate the cross-platform ingestion pipeline with SQL data quality assertions and a pytest suite that ensures schema conformance, deduplication accuracy, and metric consistency.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Unified Metrics Data Quality Assertions',
                description:
                  'Runs null checks on key columns, referential integrity between campaigns and metrics, freshness thresholds, and value range checks on spend and conversion figures.',
                code: `-- Null checks on required columns
SELECT 'null_campaign_key' AS check_name, COUNT(*) AS failures
FROM unified_daily_metrics WHERE campaign_key IS NULL
UNION ALL
SELECT 'null_metric_date', COUNT(*)
FROM unified_daily_metrics WHERE metric_date IS NULL
UNION ALL
-- Referential integrity: every metric row must have a matching campaign
SELECT 'orphan_metrics', COUNT(*)
FROM (
  SELECT DISTINCT m.campaign_key
  FROM unified_daily_metrics m
  LEFT JOIN unified_campaigns c ON m.campaign_key = c.campaign_key
  WHERE c.campaign_key IS NULL
) orphans
UNION ALL
-- Freshness: data must be no older than 26 hours
SELECT 'stale_metrics',
       CASE WHEN MAX(metric_date) < CURRENT_DATE - INTERVAL '1 day'
            THEN 1 ELSE 0 END
FROM unified_daily_metrics
UNION ALL
-- Value range: spend must be non-negative
SELECT 'negative_spend', COUNT(*)
FROM unified_daily_metrics WHERE spend < 0
UNION ALL
-- Value range: verified conversions must not exceed platform conversions
SELECT 'verified_exceeds_platform', COUNT(*)
FROM unified_daily_metrics
WHERE verified_conversions > platform_conversions;`,
              },
              {
                language: 'python',
                title: 'Pytest Cross-Platform Pipeline Validation Suite',
                description:
                  'A pytest-based test class that validates schema conformance, deduplication logic, campaign name normalisation, and metric consistency across platforms.',
                code: `import logging
from typing import Dict, List, Set

import pandas as pd
import pytest

logger = logging.getLogger(__name__)


class TestCrossPlatformPipeline:
    """Validate the unified marketing data pipeline outputs."""

    @pytest.fixture(autouse=True)
    def setup(self, db_conn) -> None:
        self.conn = db_conn
        self.campaigns: pd.DataFrame = pd.read_sql(
            "SELECT * FROM unified_campaigns", db_conn
        )
        self.metrics: pd.DataFrame = pd.read_sql(
            "SELECT * FROM unified_daily_metrics ORDER BY metric_date DESC LIMIT 50000",
            db_conn,
        )
        logger.info(
            "Loaded %d campaigns, %d metric rows",
            len(self.campaigns), len(self.metrics),
        )

    def test_no_null_campaign_keys(self) -> None:
        nulls: int = int(self.metrics["campaign_key"].isna().sum())
        assert nulls == 0, f"Found {nulls} null campaign_key rows"

    def test_referential_integrity(self) -> None:
        campaign_keys: Set[str] = set(self.campaigns["campaign_key"].tolist())
        metric_keys: Set[str] = set(self.metrics["campaign_key"].unique())
        orphans: Set[str] = metric_keys - campaign_keys
        assert not orphans, f"Orphan campaign_keys in metrics: {orphans}"

    def test_no_negative_spend(self) -> None:
        negatives: int = int((self.metrics["spend"] < 0).sum())
        assert negatives == 0, f"Found {negatives} rows with negative spend"

    def test_verified_does_not_exceed_platform(self) -> None:
        violations: int = int(
            (self.metrics["verified_conversions"] > self.metrics["platform_conversions"]).sum()
        )
        assert violations == 0, (
            f"{violations} rows have verified_conversions > platform_conversions"
        )

    def test_all_platforms_present(self) -> None:
        expected: Set[str] = {"google_ads", "meta", "tiktok", "linkedin"}
        present: Set[str] = set(self.campaigns["platform"].unique())
        missing: Set[str] = expected - present
        assert not missing, f"Missing platforms: {missing}"

    def test_metrics_freshness(self) -> None:
        max_date = pd.to_datetime(self.metrics["metric_date"]).max()
        staleness = pd.Timestamp.now() - max_date
        assert staleness.days <= 1, (
            f"Latest metric_date is {max_date}, staleness: {staleness.days} days"
        )`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Package the multi-platform ingestion pipeline as a containerised service with automated deployment, database migrations, daily scheduling, and Slack notifications.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Cross-Platform Pipeline Deployment Script',
                description:
                  'End-to-end deployment: validates environment, builds Docker image, runs tests, applies migrations, registers cron schedule, and sends Slack notification.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ── Environment variable validation ──────────────────────────────
REQUIRED_VARS=(
  DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD
  GOOGLE_ADS_DEVELOPER_TOKEN META_ACCESS_TOKEN
  TIKTOK_ACCESS_TOKEN LINKEDIN_ACCESS_TOKEN
  SLACK_WEBHOOK_URL DOCKER_REGISTRY
)
for var in "\${REQUIRED_VARS[@]}"; do
  if [[ -z "\${!var:-}" ]]; then
    echo "ERROR: \${var} is not set" >&2
    exit 1
  fi
done

APP_NAME="platform-ingestion"
IMAGE_TAG="\${DOCKER_REGISTRY}/\${APP_NAME}:\$(git rev-parse --short HEAD)"

echo "==> Building Docker image \${IMAGE_TAG}"
docker build -t "\${IMAGE_TAG}" -f Dockerfile.platform .

echo "==> Running pytest suite inside container"
docker run --rm \\
  -e DB_HOST -e DB_PORT -e DB_NAME -e DB_USER -e DB_PASSWORD \\
  "\${IMAGE_TAG}" pytest tests/test_platform_pipeline.py -v --tb=short

echo "==> Applying database migrations"
docker run --rm \\
  -e DB_HOST -e DB_PORT -e DB_NAME -e DB_USER -e DB_PASSWORD \\
  "\${IMAGE_TAG}" python manage.py migrate

echo "==> Pushing image to registry"
docker push "\${IMAGE_TAG}"

echo "==> Registering daily cron job (7 AM UTC)"
CRON_EXPR="0 7 * * *"
(crontab -l 2>/dev/null | grep -v "\${APP_NAME}"; \\
 echo "\${CRON_EXPR} docker run --rm -e DB_HOST -e DB_PORT -e DB_NAME -e DB_USER -e DB_PASSWORD -e GOOGLE_ADS_DEVELOPER_TOKEN -e META_ACCESS_TOKEN -e TIKTOK_ACCESS_TOKEN -e LINKEDIN_ACCESS_TOKEN \${IMAGE_TAG} python run_ingestion.py >> /var/log/\${APP_NAME}.log 2>&1") | crontab -

echo "==> Sending Slack notification"
curl -sf -X POST "\${SLACK_WEBHOOK_URL}" \\
  -H 'Content-Type: application/json' \\
  -d "{
    \\"text\\": \\"Platform ingestion pipeline deployed successfully.\\\\nImage: \${IMAGE_TAG}\\\\nCron: \${CRON_EXPR}\\"
  }"

echo "==> Deployment complete"`,
              },
              {
                language: 'python',
                title: 'Typed Configuration Loader for Platform Pipeline',
                description:
                  'Reads environment variables with sensible defaults, validates platform API credentials, and initialises a connection pool using dataclasses.',
                code: `import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DbConfig:
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME", "marketing"))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "pipeline"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    pool_min: int = field(default_factory=lambda: int(os.getenv("DB_POOL_MIN", "2")))
    pool_max: int = field(default_factory=lambda: int(os.getenv("DB_POOL_MAX", "10")))

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


@dataclass(frozen=True)
class PlatformCredentials:
    google_ads_token: str = field(
        default_factory=lambda: os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN", "")
    )
    meta_access_token: str = field(
        default_factory=lambda: os.getenv("META_ACCESS_TOKEN", "")
    )
    tiktok_access_token: str = field(
        default_factory=lambda: os.getenv("TIKTOK_ACCESS_TOKEN", "")
    )
    linkedin_access_token: str = field(
        default_factory=lambda: os.getenv("LINKEDIN_ACCESS_TOKEN", "")
    )


@dataclass(frozen=True)
class PipelineConfig:
    db: DbConfig = field(default_factory=DbConfig)
    credentials: PlatformCredentials = field(default_factory=PlatformCredentials)
    slack_webhook: str = field(
        default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL", "")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    lookback_days: int = field(
        default_factory=lambda: int(os.getenv("LOOKBACK_DAYS", "1"))
    )


def init_connection_pool(config: DbConfig):
    """Create a connection pool from the typed config."""
    try:
        import psycopg2.pool as pool  # type: ignore[import-untyped]

        conn_pool = pool.ThreadedConnectionPool(
            minconn=config.pool_min,
            maxconn=config.pool_max,
            dsn=config.dsn,
        )
        logger.info(
            "Connection pool created: %s:%d/%s (min=%d, max=%d)",
            config.host, config.port, config.name,
            config.pool_min, config.pool_max,
        )
        return conn_pool
    except Exception:
        logger.exception("Failed to create connection pool")
        raise`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Layer operational monitoring on the unified marketing data model to detect ingestion failures, metric anomalies, and spend pacing deviations across all platforms.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Cross-Platform SLA & Anomaly Dashboard',
                description:
                  'Detects per-platform ingestion gaps, spend pacing anomalies, and overcount deviations that exceed alerting thresholds.',
                code: `-- Per-platform ingestion freshness check
SELECT
  uc.platform,
  MAX(m.metric_date) AS latest_date,
  CURRENT_DATE - MAX(m.metric_date) AS days_stale,
  CASE
    WHEN CURRENT_DATE - MAX(m.metric_date) > 1 THEN 'STALE'
    ELSE 'CURRENT'
  END AS freshness_status
FROM unified_daily_metrics m
JOIN unified_campaigns uc ON m.campaign_key = uc.campaign_key
GROUP BY uc.platform

UNION ALL

-- Spend pacing anomaly: flag platforms where yesterday spend
-- deviates > 30% from 7-day rolling average
SELECT
  sub.platform,
  sub.metric_date::TEXT AS latest_date,
  NULL AS days_stale,
  CASE
    WHEN ABS(sub.daily_spend - sub.avg_7d_spend) / NULLIF(sub.avg_7d_spend, 0) > 0.3
    THEN 'SPEND_ANOMALY'
    ELSE 'NORMAL'
  END AS freshness_status
FROM (
  SELECT
    uc.platform,
    m.metric_date,
    SUM(m.spend) AS daily_spend,
    AVG(SUM(m.spend)) OVER (
      PARTITION BY uc.platform
      ORDER BY m.metric_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS avg_7d_spend
  FROM unified_daily_metrics m
  JOIN unified_campaigns uc ON m.campaign_key = uc.campaign_key
  WHERE m.metric_date >= CURRENT_DATE - INTERVAL '8 days'
  GROUP BY uc.platform, m.metric_date
) sub
WHERE sub.metric_date = CURRENT_DATE - INTERVAL '1 day';`,
              },
              {
                language: 'python',
                title: 'Platform Pipeline SLA Monitor & Anomaly Alerter',
                description:
                  'Checks per-platform ingestion freshness, detects spend pacing anomalies, and sends Slack alerts when SLA thresholds are breached.',
                code: `import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List

import requests
import pandas as pd

logger = logging.getLogger(__name__)

SLACK_WEBHOOK: str = os.getenv("SLACK_WEBHOOK_URL", "")
SPEND_DEVIATION_THRESHOLD: float = 0.30
OVERCOUNT_ALERT_PCT: float = 50.0


@dataclass
class SLACheckResult:
    check_name: str
    passed: bool
    detail: str


def check_platform_freshness(db_conn) -> List[SLACheckResult]:
    """Verify each platform has delivered data within the last 26 hours."""
    df: pd.DataFrame = pd.read_sql(
        """
        SELECT uc.platform, MAX(m.metric_date) AS latest_date
        FROM unified_daily_metrics m
        JOIN unified_campaigns uc ON m.campaign_key = uc.campaign_key
        GROUP BY uc.platform
        """,
        db_conn,
    )
    results: List[SLACheckResult] = []
    yesterday: date = date.today() - timedelta(days=1)
    for _, row in df.iterrows():
        latest = row["latest_date"]
        passed: bool = latest >= yesterday
        results.append(SLACheckResult(
            f"freshness_{row['platform']}", passed,
            f"Latest: {latest} (expected >= {yesterday})"
        ))
    return results


def check_spend_anomalies(db_conn) -> SLACheckResult:
    """Detect spend pacing anomalies across platforms."""
    df: pd.DataFrame = pd.read_sql(
        """
        SELECT uc.platform,
               SUM(CASE WHEN m.metric_date = CURRENT_DATE - INTERVAL '1 day'
                        THEN m.spend ELSE 0 END) AS yesterday_spend,
               AVG(m.spend) AS avg_spend
        FROM unified_daily_metrics m
        JOIN unified_campaigns uc ON m.campaign_key = uc.campaign_key
        WHERE m.metric_date >= CURRENT_DATE - INTERVAL '8 days'
        GROUP BY uc.platform
        """,
        db_conn,
    )
    anomalies: List[str] = []
    for _, row in df.iterrows():
        if row["avg_spend"] and row["avg_spend"] > 0:
            deviation: float = abs(row["yesterday_spend"] - row["avg_spend"]) / row["avg_spend"]
            if deviation > SPEND_DEVIATION_THRESHOLD:
                anomalies.append(f"{row['platform']}: {deviation:.0%} deviation")
    if not anomalies:
        return SLACheckResult("spend_anomaly", True, "No spend pacing anomalies")
    return SLACheckResult("spend_anomaly", False, "; ".join(anomalies))


def send_alert(results: List[SLACheckResult]) -> None:
    failures: List[SLACheckResult] = [r for r in results if not r.passed]
    if not failures:
        logger.info("All platform SLA checks passed")
        return
    blocks: List[str] = [f"*{r.check_name}*: {r.detail}" for r in failures]
    payload = {
        "text": f"Platform Pipeline Alert — {len(failures)} issue(s):\\n"
        + "\\n".join(blocks)
    }
    if SLACK_WEBHOOK:
        requests.post(SLACK_WEBHOOK, json=payload, timeout=10)
    logger.warning("Alerts sent: %s", payload["text"])


def run_platform_monitoring(db_conn) -> List[SLACheckResult]:
    results: List[SLACheckResult] = check_platform_freshness(db_conn)
    results.append(check_spend_anomalies(db_conn))
    send_alert(results)
    return results`,
              },
            ],
          },
        ],
        toolsUsed: [
          'BigQuery / Snowflake (unified data model)',
          'Python (platform API connectors)',
          'Google Ads API',
          'Meta Marketing API',
          'TikTok Business API',
          'LinkedIn Campaign Manager API',
          'Apache Airflow / Prefect (orchestration)',
          'pytest',
          'Docker',
          'GitHub Actions',
          'cron / Airflow',
          'Slack API',
        ],
      },
    },
  ],
};
