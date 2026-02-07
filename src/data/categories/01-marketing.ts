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

      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to analyze marketing touchpoint data exported from Google Analytics and ad platforms, then automate weekly attribution insights delivery to Slack or email without writing custom code.',
        estimatedMonthlyCost: '$70 - $150/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Google Sheets (Free)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Supermetrics ($39/mo)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Export multi-touch journey data from Google Analytics 4 and ad platforms into Google Sheets using native connectors or Supermetrics. Structure the data with customer journey paths for AI analysis.',
            toolsUsed: ['Google Analytics 4', 'Google Sheets', 'Supermetrics'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Google Sheets Data Structure for Attribution Analysis',
                description:
                  'Schema for organizing touchpoint journey data in Google Sheets for AI processing.',
                code: `{
  "sheet_name": "Attribution_Journeys",
  "columns": [
    {"name": "customer_id", "type": "string", "description": "Unique customer identifier"},
    {"name": "journey_path", "type": "string", "description": "Ordered touchpoints: 'paid_search > email > organic > direct'"},
    {"name": "touchpoint_count", "type": "integer", "description": "Number of interactions before conversion"},
    {"name": "first_touch_channel", "type": "string", "description": "Initial acquisition channel"},
    {"name": "last_touch_channel", "type": "string", "description": "Final converting channel"},
    {"name": "conversion_value", "type": "number", "description": "Revenue from conversion"},
    {"name": "days_to_convert", "type": "integer", "description": "Journey duration in days"},
    {"name": "converted", "type": "boolean", "description": "Whether journey resulted in conversion"}
  ],
  "sample_row": {
    "customer_id": "usr_abc123",
    "journey_path": "paid_social > organic_search > email > direct",
    "touchpoint_count": 4,
    "first_touch_channel": "paid_social",
    "last_touch_channel": "direct",
    "conversion_value": 149.99,
    "days_to_convert": 12,
    "converted": true
  }
}`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description:
              'Use ChatGPT or Claude to analyze journey patterns, calculate position-based attribution weights, and identify undervalued channels. The AI interprets complex multi-touch paths and provides actionable budget recommendations.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Attribution Analysis Prompt Template',
                description:
                  'Structured prompt for AI to analyze multi-touch attribution data and generate budget recommendations.',
                code: `system_prompt: |
  You are a marketing analytics expert specializing in multi-touch attribution.
  Analyze customer journey data to identify channel contribution patterns and
  provide actionable budget reallocation recommendations.

user_prompt_template: |
  ## Attribution Analysis Request

  ### Data Summary
  - Total journeys analyzed: {{total_journeys}}
  - Conversion rate: {{conversion_rate}}%
  - Average journey length: {{avg_touchpoints}} touchpoints
  - Date range: {{start_date}} to {{end_date}}

  ### Journey Data (sample of top 100 converting paths)
  {{journey_data_csv}}

  ### Current Budget Allocation
  {{current_budget_by_channel}}

  ### Analysis Tasks
  1. **Channel Contribution Analysis**: For each channel, calculate:
     - First-touch frequency (awareness contribution)
     - Mid-journey frequency (consideration contribution)
     - Last-touch frequency (conversion contribution)
     - Position-weighted attribution score

  2. **Undervalued Channel Detection**: Identify channels that:
     - Appear frequently in mid-journey but rarely get last-click credit
     - Have high first-touch presence but low budget allocation
     - Show strong assist patterns in converting journeys

  3. **Budget Recommendations**: Provide specific reallocation suggestions:
     - Channels to increase investment (with % recommendation)
     - Channels to reduce investment (with % recommendation)
     - Expected impact on blended ROAS

  ### Output Format
  Provide analysis in structured sections with specific percentages and
  actionable recommendations. Include confidence levels for each suggestion.

expected_output_sections:
  - channel_contribution_table
  - undervalued_channels_list
  - budget_reallocation_recommendations
  - expected_roas_impact
  - confidence_assessment`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier to automatically extract fresh data weekly, send it to ChatGPT for analysis via the OpenAI API, and deliver formatted attribution insights to Slack and stakeholder emails.',
            toolsUsed: ['Zapier Pro', 'OpenAI API', 'Slack', 'Gmail'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Attribution Insights Workflow',
                description:
                  'Automated workflow configuration for weekly attribution analysis and delivery.',
                code: `{
  "workflow_name": "Weekly Attribution Insights",
  "trigger": {
    "app": "Schedule by Zapier",
    "event": "Every Week",
    "config": {
      "day_of_week": "Monday",
      "time": "08:00",
      "timezone": "America/New_York"
    }
  },
  "actions": [
    {
      "step": 1,
      "app": "Google Sheets",
      "action": "Get Many Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{ATTRIBUTION_SHEET_ID}}",
        "worksheet": "Attribution_Journeys",
        "filter_formula": "=AND(converted=TRUE, journey_date>=TODAY()-7)"
      }
    },
    {
      "step": 2,
      "app": "Formatter by Zapier",
      "action": "Text Transform",
      "config": {
        "operation": "Convert to CSV",
        "input": "{{step1_rows}}",
        "include_headers": true
      }
    },
    {
      "step": 3,
      "app": "OpenAI (ChatGPT)",
      "action": "Send Prompt",
      "config": {
        "model": "gpt-4-turbo",
        "system_message": "You are a marketing analytics expert...",
        "user_message": "Analyze these {{step1_row_count}} customer journeys:\\n{{step2_csv}}\\n\\nProvide attribution insights and budget recommendations.",
        "temperature": 0.3,
        "max_tokens": 2000
      }
    },
    {
      "step": 4,
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#marketing-insights",
        "message_format": "blocks",
        "blocks": [
          {
            "type": "header",
            "text": "Weekly Attribution Analysis - {{current_date}}"
          },
          {
            "type": "section",
            "text": "{{step3_response}}"
          },
          {
            "type": "context",
            "elements": [
              {"type": "mrkdwn", "text": "Based on {{step1_row_count}} converting journeys"}
            ]
          }
        ]
      }
    },
    {
      "step": 5,
      "app": "Gmail",
      "action": "Send Email",
      "config": {
        "to": "marketing-leadership@company.com",
        "subject": "Weekly Attribution Insights - {{current_date}}",
        "body_type": "html",
        "body": "<h2>Multi-Touch Attribution Analysis</h2>{{step3_response_html}}"
      }
    }
  ],
  "error_handling": {
    "on_error": "notify",
    "notification_channel": "#zapier-alerts"
  }
}`,
              },
            ],
          },
        ],
      },

      aiAdvanced: {
        overview:
          'Deploy a multi-agent system using CrewAI and LangGraph to continuously analyze attribution data, build and update Markov chain models, optimize spend allocation, and autonomously execute budget recommendations across ad platforms.',
        estimatedMonthlyCost: '$500 - $1,500/month',
        architecture:
          'A Supervisor agent coordinates four specialist agents: Data Ingestion Agent pulls from GA4/ad APIs, Markov Modeler Agent calculates removal effects, Optimization Agent generates budget recommendations, and Execution Agent pushes approved changes to ad platforms via APIs.',
        agents: [
          {
            name: 'Attribution Data Collector',
            role: 'Data Ingestion Specialist',
            goal: 'Continuously ingest touchpoint data from GA4, Google Ads, Meta, and CRM systems into a unified journey dataset',
            tools: ['GA4 BigQuery Connector', 'Google Ads API', 'Meta Marketing API', 'HubSpot API'],
          },
          {
            name: 'Markov Attribution Modeler',
            role: 'Statistical Analysis Specialist',
            goal: 'Build transition matrices from journey paths and calculate channel removal effects to determine true attribution weights',
            tools: ['Python NumPy', 'Pandas', 'Custom Markov Library', 'BigQuery ML'],
          },
          {
            name: 'Spend Optimization Strategist',
            role: 'Budget Allocation Specialist',
            goal: 'Generate optimal budget allocation recommendations based on Markov weights, historical performance, and business constraints',
            tools: ['Optimization Solver', 'Constraint Engine', 'ROI Calculator', 'Scenario Simulator'],
          },
          {
            name: 'Campaign Execution Agent',
            role: 'Ad Platform Integration Specialist',
            goal: 'Execute approved budget changes across ad platforms and monitor for anomalies post-adjustment',
            tools: ['Google Ads API', 'Meta Marketing API', 'TikTok Ads API', 'LinkedIn Campaign Manager API'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with daily checkpointing and 30-day audit trail',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system with CrewAI, establishing clear roles, goals, and tool access for each agent. The supervisor coordinates workflow and handles inter-agent communication.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'CrewAI Attribution Agent Definitions',
                description:
                  'Production-ready agent definitions for the multi-touch attribution system.',
                code: `from crewai import Agent, Crew, Task, Process
from crewai.tools import BaseTool
from typing import List, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class AttributionAgentConfig(BaseModel):
    """Configuration for attribution analysis agents."""
    llm_model: str = Field(default="gpt-4-turbo")
    temperature: float = Field(default=0.1)
    max_iterations: int = Field(default=10)
    verbose: bool = Field(default=True)


def create_data_collector_agent(
    config: AttributionAgentConfig,
    tools: List[BaseTool],
) -> Agent:
    """Create the data ingestion specialist agent."""
    return Agent(
        role="Attribution Data Collector",
        goal="""Continuously ingest touchpoint data from GA4, Google Ads,
        Meta, and CRM systems. Ensure data quality and freshness for
        downstream attribution modeling.""",
        backstory="""You are an expert data engineer specializing in marketing
        data pipelines. You understand the nuances of different platform APIs,
        handle rate limits gracefully, and ensure data consistency across sources.
        You validate data quality before passing to analysis agents.""",
        tools=tools,
        llm=config.llm_model,
        verbose=config.verbose,
        allow_delegation=False,
        max_iter=config.max_iterations,
    )


def create_markov_modeler_agent(
    config: AttributionAgentConfig,
    tools: List[BaseTool],
) -> Agent:
    """Create the statistical analysis specialist agent."""
    return Agent(
        role="Markov Attribution Modeler",
        goal="""Build accurate transition matrices from customer journey paths
        and calculate channel removal effects. Provide statistically rigorous
        attribution weights that reflect true channel contribution.""",
        backstory="""You are a quantitative marketing scientist with deep expertise
        in Markov chain modeling and probabilistic attribution. You understand
        the limitations of traditional attribution and use removal effects to
        measure true incremental value of each channel.""",
        tools=tools,
        llm=config.llm_model,
        verbose=config.verbose,
        allow_delegation=False,
        max_iter=config.max_iterations,
    )


def create_optimization_agent(
    config: AttributionAgentConfig,
    tools: List[BaseTool],
) -> Agent:
    """Create the budget allocation specialist agent."""
    return Agent(
        role="Spend Optimization Strategist",
        goal="""Generate optimal budget allocation recommendations that maximize
        ROAS while respecting business constraints. Balance short-term performance
        with long-term brand building.""",
        backstory="""You are a marketing operations strategist who combines
        data-driven insights with practical business constraints. You understand
        that optimization is not just about numbers—it requires considering
        seasonality, competitive dynamics, and organizational readiness.""",
        tools=tools,
        llm=config.llm_model,
        verbose=config.verbose,
        allow_delegation=True,
        max_iter=config.max_iterations,
    )


def create_execution_agent(
    config: AttributionAgentConfig,
    tools: List[BaseTool],
) -> Agent:
    """Create the ad platform integration specialist agent."""
    return Agent(
        role="Campaign Execution Agent",
        goal="""Execute approved budget changes across ad platforms safely.
        Implement changes incrementally, monitor for anomalies, and rollback
        if performance deviates beyond acceptable thresholds.""",
        backstory="""You are a campaign operations expert who has managed
        millions in ad spend across platforms. You know that execution is
        as important as strategy—you implement changes carefully, validate
        results, and maintain detailed audit trails.""",
        tools=tools,
        llm=config.llm_model,
        verbose=config.verbose,
        allow_delegation=False,
        max_iter=config.max_iterations,
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the data collection agent with tools for connecting to GA4 BigQuery exports, ad platform APIs, and CRM systems. The agent validates data quality and maintains a unified journey dataset.',
            toolsUsed: ['GA4 BigQuery Connector', 'Google Ads API', 'Meta Marketing API', 'HubSpot API'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Data Ingestion Agent Tools',
                description:
                  'Custom tools for the data collector agent to pull from multiple marketing platforms.',
                code: `from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import date, timedelta
from google.cloud import bigquery
import logging

logger = logging.getLogger(__name__)


class GA4QueryInput(BaseModel):
    """Input schema for GA4 BigQuery queries."""
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    project_id: str = Field(description="GCP project ID")
    dataset_id: str = Field(description="BigQuery dataset ID")


class GA4JourneyExtractorTool(BaseTool):
    """Tool to extract customer journey data from GA4 BigQuery exports."""

    name: str = "ga4_journey_extractor"
    description: str = """Extracts customer journey paths from GA4 BigQuery exports.
    Returns touchpoint sequences with timestamps, channels, and conversion data."""
    args_schema: type[BaseModel] = GA4QueryInput

    def __init__(self, bq_client: Optional[bigquery.Client] = None):
        super().__init__()
        self._client = bq_client or bigquery.Client()

    def _run(
        self,
        start_date: str,
        end_date: str,
        project_id: str,
        dataset_id: str,
    ) -> Dict[str, Any]:
        """Execute the GA4 journey extraction query."""
        query = f"""
        WITH sessions AS (
            SELECT
                user_pseudo_id,
                event_timestamp,
                traffic_source.medium AS channel,
                traffic_source.source AS source,
                traffic_source.name AS campaign,
                (SELECT value.int_value FROM UNNEST(event_params)
                 WHERE key = 'ga_session_id') AS session_id
            FROM \`{project_id}.{dataset_id}.events_*\`
            WHERE _TABLE_SUFFIX BETWEEN
                FORMAT_DATE('%Y%m%d', DATE('{start_date}'))
                AND FORMAT_DATE('%Y%m%d', DATE('{end_date}'))
                AND event_name = 'session_start'
        ),
        conversions AS (
            SELECT
                user_pseudo_id,
                event_timestamp AS conversion_time,
                (SELECT value.double_value FROM UNNEST(event_params)
                 WHERE key = 'value') AS conversion_value
            FROM \`{project_id}.{dataset_id}.events_*\`
            WHERE _TABLE_SUFFIX BETWEEN
                FORMAT_DATE('%Y%m%d', DATE('{start_date}'))
                AND FORMAT_DATE('%Y%m%d', DATE('{end_date}'))
                AND event_name = 'purchase'
        )
        SELECT
            s.user_pseudo_id AS customer_id,
            STRING_AGG(s.channel, ' > ' ORDER BY s.event_timestamp) AS journey_path,
            COUNT(DISTINCT s.session_id) AS touchpoint_count,
            MIN(s.event_timestamp) AS first_touch_time,
            MAX(s.event_timestamp) AS last_touch_time,
            CASE WHEN c.user_pseudo_id IS NOT NULL THEN TRUE ELSE FALSE END AS converted,
            COALESCE(c.conversion_value, 0) AS conversion_value
        FROM sessions s
        LEFT JOIN conversions c ON s.user_pseudo_id = c.user_pseudo_id
            AND s.event_timestamp <= c.conversion_time
        GROUP BY s.user_pseudo_id, c.user_pseudo_id, c.conversion_value
        HAVING touchpoint_count >= 2
        """

        try:
            result = self._client.query(query).result()
            journeys = [dict(row) for row in result]
            logger.info(f"Extracted {len(journeys)} journeys from GA4")
            return {
                "status": "success",
                "journey_count": len(journeys),
                "date_range": f"{start_date} to {end_date}",
                "journeys": journeys[:1000],  # Limit for LLM context
            }
        except Exception as e:
            logger.error(f"GA4 extraction failed: {e}")
            return {"status": "error", "message": str(e)}


class AdPlatformMetricsTool(BaseTool):
    """Tool to fetch campaign metrics from ad platforms."""

    name: str = "ad_platform_metrics"
    description: str = """Fetches campaign performance metrics from Google Ads,
    Meta, TikTok, and LinkedIn. Returns spend, impressions, clicks, and conversions."""

    def __init__(self, platform_clients: Dict[str, Any]):
        super().__init__()
        self._clients = platform_clients

    def _run(self, platform: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch metrics from the specified ad platform."""
        if platform not in self._clients:
            return {"status": "error", "message": f"Unknown platform: {platform}"}

        client = self._clients[platform]
        try:
            # Platform-specific API calls would go here
            metrics = client.get_campaign_metrics(start_date, end_date)
            logger.info(f"Fetched {len(metrics)} campaigns from {platform}")
            return {
                "status": "success",
                "platform": platform,
                "campaign_count": len(metrics),
                "metrics": metrics,
            }
        except Exception as e:
            logger.error(f"{platform} fetch failed: {e}")
            return {"status": "error", "platform": platform, "message": str(e)}`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Markov modeler and optimization agents with tools for building transition matrices, calculating removal effects, and generating constrained budget recommendations.',
            toolsUsed: ['NumPy', 'Pandas', 'SciPy Optimize', 'Custom Markov Library'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Markov Attribution Analysis Tools',
                description:
                  'Tools for the Markov modeler agent to calculate attribution weights.',
                code: `from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Any
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class JourneyDataInput(BaseModel):
    """Input schema for journey analysis."""
    journeys: List[Dict[str, Any]] = Field(
        description="List of journey records with journey_path and converted fields"
    )
    simulation_count: int = Field(default=10000, description="Monte Carlo simulations")


class MarkovTransitionMatrixTool(BaseTool):
    """Tool to build Markov chain transition matrices from journey data."""

    name: str = "markov_transition_matrix"
    description: str = """Builds a Markov chain transition matrix from customer
    journey paths. Calculates transition probabilities between channels and
    to conversion/null states."""
    args_schema: type[BaseModel] = JourneyDataInput

    def _run(self, journeys: List[Dict], simulation_count: int = 10000) -> Dict:
        """Build the transition matrix from journey data."""
        transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for journey in journeys:
            path = ['(start)'] + journey['journey_path'].split(' > ')
            end_state = '(conversion)' if journey.get('converted') else '(null)'
            path.append(end_state)

            for i in range(len(path) - 1):
                transitions[path[i]][path[i + 1]] += 1

        # Convert counts to probabilities
        matrix: Dict[str, Dict[str, float]] = {}
        channels: set = set()

        for state, next_states in transitions.items():
            total = sum(next_states.values())
            matrix[state] = {k: v / total for k, v in next_states.items()}
            if state not in ('(start)', '(conversion)', '(null)'):
                channels.add(state)

        logger.info(f"Built transition matrix with {len(channels)} channels")

        return {
            "status": "success",
            "matrix": matrix,
            "channels": list(channels),
            "total_journeys": len(journeys),
            "conversion_rate": sum(1 for j in journeys if j.get('converted')) / len(journeys),
        }


class RemovalEffectCalculatorTool(BaseTool):
    """Tool to calculate channel removal effects for attribution."""

    name: str = "removal_effect_calculator"
    description: str = """Calculates the removal effect for each channel by
    simulating conversion rates with and without each channel. Returns
    attribution weights proportional to each channel's incremental contribution."""

    def _run(
        self,
        matrix: Dict[str, Dict[str, float]],
        channels: List[str],
        simulations: int = 10000,
    ) -> Dict[str, Any]:
        """Calculate removal effects and attribution weights."""

        def simulate_conversion_rate(
            trans_matrix: Dict[str, Dict[str, float]],
            n_sim: int,
        ) -> float:
            """Monte Carlo simulation of conversion rate."""
            conversions = 0
            for _ in range(n_sim):
                state = '(start)'
                for _ in range(50):  # Max path length
                    if state in ('(conversion)', '(null)'):
                        break
                    if state not in trans_matrix:
                        break
                    probs = trans_matrix[state]
                    states = list(probs.keys())
                    probabilities = list(probs.values())
                    state = np.random.choice(states, p=probabilities)

                if state == '(conversion)':
                    conversions += 1

            return conversions / n_sim

        # Calculate baseline conversion rate
        base_cr = simulate_conversion_rate(matrix, simulations)
        logger.info(f"Baseline conversion rate: {base_cr:.4f}")

        # Calculate removal effect for each channel
        removal_effects: Dict[str, float] = {}

        for channel in channels:
            # Create matrix without this channel
            reduced_matrix: Dict[str, Dict[str, float]] = {}
            for state, next_states in matrix.items():
                if state == channel:
                    continue
                filtered = {k: v for k, v in next_states.items() if k != channel}
                if filtered:
                    total = sum(filtered.values())
                    reduced_matrix[state] = {k: v / total for k, v in filtered.items()}

            reduced_cr = simulate_conversion_rate(reduced_matrix, simulations)
            removal_effect = (base_cr - reduced_cr) / base_cr if base_cr > 0 else 0
            removal_effects[channel] = max(0, removal_effect)
            logger.debug(f"{channel}: removal effect = {removal_effect:.4f}")

        # Normalize to attribution weights
        total_effect = sum(removal_effects.values())
        weights: Dict[str, float] = {
            ch: effect / total_effect if total_effect > 0 else 1 / len(channels)
            for ch, effect in removal_effects.items()
        }

        return {
            "status": "success",
            "baseline_conversion_rate": base_cr,
            "removal_effects": removal_effects,
            "attribution_weights": weights,
            "simulations_run": simulations,
        }


class BudgetOptimizationTool(BaseTool):
    """Tool to generate optimal budget allocation recommendations."""

    name: str = "budget_optimizer"
    description: str = """Generates optimal budget allocation based on attribution
    weights, current spend, and business constraints. Uses convex optimization
    to maximize expected ROAS."""

    def _run(
        self,
        attribution_weights: Dict[str, float],
        current_spend: Dict[str, float],
        total_budget: float,
        constraints: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Optimize budget allocation across channels."""
        constraints = constraints or {}
        min_spend_pct = constraints.get('min_spend_pct', 0.05)
        max_spend_pct = constraints.get('max_spend_pct', 0.40)
        max_change_pct = constraints.get('max_change_pct', 0.25)

        channels = list(attribution_weights.keys())
        weights = np.array([attribution_weights[ch] for ch in channels])
        current = np.array([current_spend.get(ch, 0) for ch in channels])
        current_total = current.sum()

        # Simple weighted allocation with constraints
        raw_allocation = weights * total_budget

        # Apply min/max constraints
        min_spend = total_budget * min_spend_pct
        max_spend = total_budget * max_spend_pct
        constrained = np.clip(raw_allocation, min_spend, max_spend)

        # Apply change rate constraints
        if current_total > 0:
            current_pct = current / current_total
            proposed_pct = constrained / constrained.sum()
            max_delta = max_change_pct

            final_pct = np.clip(
                proposed_pct,
                current_pct - max_delta,
                current_pct + max_delta,
            )
            final_pct = np.maximum(final_pct, 0)
            final_pct = final_pct / final_pct.sum()
            final_allocation = final_pct * total_budget
        else:
            final_allocation = constrained / constrained.sum() * total_budget

        recommendations = {}
        for i, channel in enumerate(channels):
            curr = current[i]
            proposed = final_allocation[i]
            change = proposed - curr
            change_pct = (change / curr * 100) if curr > 0 else 0

            recommendations[channel] = {
                "current_spend": round(curr, 2),
                "recommended_spend": round(proposed, 2),
                "change_amount": round(change, 2),
                "change_percent": round(change_pct, 1),
                "attribution_weight": round(attribution_weights[channel], 4),
            }

        return {
            "status": "success",
            "total_budget": total_budget,
            "recommendations": recommendations,
            "expected_roas_lift": f"{(weights @ final_allocation / total_budget - 1) * 100:.1f}%",
        }`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement the LangGraph state machine that coordinates agent execution, manages state transitions, handles approval workflows, and maintains audit trails for all attribution decisions.',
            toolsUsed: ['LangGraph', 'Redis', 'PostgreSQL'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Attribution Workflow Orchestration',
                description:
                  'State machine implementation for coordinating the multi-agent attribution system.',
                code: `from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, List, Dict, Any, Literal
from datetime import datetime, timezone
import operator
import logging
import json

logger = logging.getLogger(__name__)


class AttributionState(TypedDict):
    """State schema for the attribution workflow."""
    # Workflow metadata
    run_id: str
    started_at: str
    status: Literal["running", "completed", "failed", "awaiting_approval"]

    # Data collection outputs
    journeys: List[Dict[str, Any]]
    journey_count: int
    platform_metrics: Dict[str, Any]
    data_quality_score: float

    # Markov modeling outputs
    transition_matrix: Dict[str, Dict[str, float]]
    removal_effects: Dict[str, float]
    attribution_weights: Dict[str, float]
    baseline_conversion_rate: float

    # Optimization outputs
    current_spend: Dict[str, float]
    recommended_spend: Dict[str, float]
    expected_roas_lift: str

    # Execution outputs
    approval_status: Literal["pending", "approved", "rejected"]
    execution_results: Dict[str, Any]

    # Audit trail
    messages: Annotated[List[str], operator.add]
    errors: List[str]


def create_attribution_workflow(
    data_collector_agent,
    markov_modeler_agent,
    optimization_agent,
    execution_agent,
    require_approval: bool = True,
) -> StateGraph:
    """Create the LangGraph workflow for attribution analysis."""

    workflow = StateGraph(AttributionState)

    # Node: Data Collection
    def collect_data(state: AttributionState) -> AttributionState:
        """Execute data collection agent."""
        logger.info("Starting data collection...")
        try:
            result = data_collector_agent.execute_task(
                "Extract customer journey data from GA4 and ad platforms for the last 30 days."
            )
            state["journeys"] = result.get("journeys", [])
            state["journey_count"] = len(state["journeys"])
            state["platform_metrics"] = result.get("platform_metrics", {})
            state["data_quality_score"] = result.get("quality_score", 0.0)
            state["messages"] = [f"Collected {state['journey_count']} journeys"]
        except Exception as e:
            state["errors"] = [f"Data collection failed: {str(e)}"]
            state["status"] = "failed"
        return state

    # Node: Markov Modeling
    def build_attribution_model(state: AttributionState) -> AttributionState:
        """Execute Markov modeler agent."""
        logger.info("Building attribution model...")
        try:
            result = markov_modeler_agent.execute_task(
                f"Build Markov chain model from {state['journey_count']} journeys "
                f"and calculate removal effects for each channel."
            )
            state["transition_matrix"] = result.get("matrix", {})
            state["removal_effects"] = result.get("removal_effects", {})
            state["attribution_weights"] = result.get("attribution_weights", {})
            state["baseline_conversion_rate"] = result.get("baseline_cr", 0.0)
            state["messages"] = [
                f"Calculated attribution weights for {len(state['attribution_weights'])} channels"
            ]
        except Exception as e:
            state["errors"] = [f"Markov modeling failed: {str(e)}"]
            state["status"] = "failed"
        return state

    # Node: Budget Optimization
    def optimize_budget(state: AttributionState) -> AttributionState:
        """Execute optimization agent."""
        logger.info("Optimizing budget allocation...")
        try:
            result = optimization_agent.execute_task(
                f"Generate budget recommendations based on attribution weights: "
                f"{json.dumps(state['attribution_weights'])}"
            )
            state["recommended_spend"] = result.get("recommendations", {})
            state["expected_roas_lift"] = result.get("expected_roas_lift", "0%")
            state["messages"] = [
                f"Generated recommendations with {state['expected_roas_lift']} expected lift"
            ]
            state["approval_status"] = "pending"
        except Exception as e:
            state["errors"] = [f"Optimization failed: {str(e)}"]
            state["status"] = "failed"
        return state

    # Node: Approval Check
    def check_approval(state: AttributionState) -> AttributionState:
        """Check if changes are approved for execution."""
        if state["approval_status"] == "approved":
            state["messages"] = ["Budget changes approved, proceeding to execution"]
        elif state["approval_status"] == "rejected":
            state["messages"] = ["Budget changes rejected, workflow complete"]
            state["status"] = "completed"
        else:
            state["status"] = "awaiting_approval"
            state["messages"] = ["Awaiting approval for budget changes"]
        return state

    # Node: Execution
    def execute_changes(state: AttributionState) -> AttributionState:
        """Execute approved budget changes."""
        logger.info("Executing budget changes...")
        try:
            result = execution_agent.execute_task(
                f"Execute the following budget changes across ad platforms: "
                f"{json.dumps(state['recommended_spend'])}"
            )
            state["execution_results"] = result
            state["status"] = "completed"
            state["messages"] = ["Budget changes executed successfully"]
        except Exception as e:
            state["errors"] = [f"Execution failed: {str(e)}"]
            state["status"] = "failed"
        return state

    # Routing function
    def route_after_approval(state: AttributionState) -> str:
        if state["approval_status"] == "approved":
            return "execute"
        elif state["approval_status"] == "rejected":
            return "end"
        else:
            return "wait"

    # Add nodes
    workflow.add_node("collect_data", collect_data)
    workflow.add_node("build_model", build_attribution_model)
    workflow.add_node("optimize", optimize_budget)
    workflow.add_node("check_approval", check_approval)
    workflow.add_node("execute", execute_changes)

    # Add edges
    workflow.set_entry_point("collect_data")
    workflow.add_edge("collect_data", "build_model")
    workflow.add_edge("build_model", "optimize")

    if require_approval:
        workflow.add_edge("optimize", "check_approval")
        workflow.add_conditional_edges(
            "check_approval",
            route_after_approval,
            {"execute": "execute", "end": END, "wait": END},
        )
    else:
        workflow.add_edge("optimize", "execute")

    workflow.add_edge("execute", END)

    return workflow.compile(checkpointer=SqliteSaver.from_conn_string(":memory:"))`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the multi-agent system with Docker, implement comprehensive observability using LangSmith for agent tracing, and set up Prometheus metrics for performance monitoring.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose for Attribution Agent System',
                description:
                  'Production deployment configuration for the multi-agent attribution system.',
                code: `version: '3.8'

services:
  attribution-agents:
    build:
      context: .
      dockerfile: Dockerfile.agents
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=attribution-agents
      - REDIS_URL=redis://redis:6379/0
      - POSTGRES_DSN=postgresql://user:pass@postgres:5432/attribution
      - GA4_PROJECT_ID=\${GA4_PROJECT_ID}
      - GOOGLE_ADS_DEVELOPER_TOKEN=\${GOOGLE_ADS_DEVELOPER_TOKEN}
      - META_ACCESS_TOKEN=\${META_ACCESS_TOKEN}
      - LOG_LEVEL=INFO
    depends_on:
      - redis
      - postgres
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=attribution
    volumes:
      - postgres_data:/var/lib/postgresql/data

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:`,
              },
              {
                language: 'python',
                title: 'LangSmith Observability Integration',
                description:
                  'Comprehensive tracing and monitoring setup for agent performance analysis.',
                code: `import os
from functools import wraps
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timezone
import logging

from langsmith import Client
from langsmith.run_trees import RunTree
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = logging.getLogger(__name__)

# Prometheus metrics
AGENT_RUNS = Counter(
    'attribution_agent_runs_total',
    'Total agent task executions',
    ['agent_name', 'status']
)
AGENT_DURATION = Histogram(
    'attribution_agent_duration_seconds',
    'Agent task execution duration',
    ['agent_name'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]
)
ATTRIBUTION_WEIGHTS = Gauge(
    'attribution_channel_weight',
    'Current attribution weight by channel',
    ['channel']
)
BUDGET_RECOMMENDATIONS = Gauge(
    'attribution_budget_recommendation',
    'Recommended budget by channel',
    ['channel']
)


class AgentObserver:
    """Observability wrapper for CrewAI agents with LangSmith and Prometheus."""

    def __init__(
        self,
        project_name: str = "attribution-agents",
        enable_langsmith: bool = True,
        enable_prometheus: bool = True,
        prometheus_port: int = 8000,
    ):
        self.project_name = project_name
        self.enable_langsmith = enable_langsmith and os.getenv("LANGSMITH_API_KEY")
        self.enable_prometheus = enable_prometheus

        if self.enable_langsmith:
            self.langsmith_client = Client()
            logger.info(f"LangSmith enabled for project: {project_name}")

        if self.enable_prometheus:
            start_http_server(prometheus_port)
            logger.info(f"Prometheus metrics server started on port {prometheus_port}")

    def trace_agent_task(
        self,
        agent_name: str,
        task_description: str,
    ) -> Callable:
        """Decorator to trace agent task execution."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = datetime.now(timezone.utc)
                run_tree: Optional[RunTree] = None

                # Start LangSmith trace
                if self.enable_langsmith:
                    run_tree = RunTree(
                        name=f"{agent_name}_task",
                        run_type="chain",
                        project_name=self.project_name,
                        inputs={"task": task_description, "kwargs": str(kwargs)},
                    )

                try:
                    result = func(*args, **kwargs)
                    status = "success"

                    if run_tree:
                        run_tree.end(outputs={"result": str(result)[:1000]})
                        run_tree.post()

                    return result

                except Exception as e:
                    status = "error"
                    logger.error(f"{agent_name} task failed: {e}")

                    if run_tree:
                        run_tree.end(error=str(e))
                        run_tree.post()

                    raise

                finally:
                    duration = (datetime.now(timezone.utc) - start_time).total_seconds()

                    if self.enable_prometheus:
                        AGENT_RUNS.labels(agent_name=agent_name, status=status).inc()
                        AGENT_DURATION.labels(agent_name=agent_name).observe(duration)

                    logger.info(
                        f"{agent_name} completed in {duration:.2f}s with status: {status}"
                    )

            return wrapper
        return decorator

    def record_attribution_weights(self, weights: Dict[str, float]) -> None:
        """Record current attribution weights to Prometheus."""
        if self.enable_prometheus:
            for channel, weight in weights.items():
                ATTRIBUTION_WEIGHTS.labels(channel=channel).set(weight)

    def record_budget_recommendations(self, recommendations: Dict[str, float]) -> None:
        """Record budget recommendations to Prometheus."""
        if self.enable_prometheus:
            for channel, amount in recommendations.items():
                BUDGET_RECOMMENDATIONS.labels(channel=channel).set(amount)`,
              },
            ],
          },
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

      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to monitor consent status across systems, automate DSAR response generation, and deliver weekly compliance reports without custom code development.',
        estimatedMonthlyCost: '$80 - $140/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Airtable ($20/mo)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Google Sheets (Free)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Consolidate consent records from your CMP, CRM, and email platform into a central Airtable or Google Sheets database. Use Zapier to automatically sync consent changes in real-time.',
            toolsUsed: ['Airtable', 'Zapier', 'OneTrust', 'HubSpot'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Consent Ledger Schema for Airtable/Sheets',
                description:
                  'Data structure for tracking consent records across multiple systems.',
                code: `{
  "table_name": "Consent_Ledger",
  "fields": [
    {"name": "record_id", "type": "autonumber", "description": "Unique record identifier"},
    {"name": "user_id", "type": "string", "description": "Customer identifier (email hash or ID)"},
    {"name": "user_email", "type": "email", "description": "User email for DSAR lookups"},
    {"name": "consent_purpose", "type": "single_select", "options": ["marketing_email", "analytics", "personalization", "third_party_sharing", "profiling"]},
    {"name": "consent_status", "type": "single_select", "options": ["granted", "denied", "withdrawn"]},
    {"name": "source_system", "type": "single_select", "options": ["website_cmp", "crm", "email_platform", "mobile_app"]},
    {"name": "collection_method", "type": "single_select", "options": ["explicit_opt_in", "soft_opt_in", "preference_center", "dsar_request"]},
    {"name": "recorded_at", "type": "datetime", "description": "Timestamp of consent action"},
    {"name": "ip_country", "type": "string", "description": "Country code for jurisdictional compliance"},
    {"name": "gdpr_lawful_basis", "type": "single_select", "options": ["consent", "legitimate_interest", "contract", "legal_obligation"]},
    {"name": "retention_expiry", "type": "date", "description": "Date when this consent record expires"}
  ],
  "views": [
    {
      "name": "Active Consents",
      "filter": "consent_status = 'granted' AND retention_expiry > TODAY()"
    },
    {
      "name": "DSAR Queue",
      "filter": "consent_purpose = 'dsar_request' AND status = 'pending'"
    },
    {
      "name": "Compliance Gaps",
      "filter": "source_system != 'website_cmp'"
    }
  ],
  "sample_record": {
    "record_id": 12345,
    "user_id": "usr_abc123",
    "user_email": "john.doe@example.com",
    "consent_purpose": "marketing_email",
    "consent_status": "granted",
    "source_system": "website_cmp",
    "collection_method": "explicit_opt_in",
    "recorded_at": "2024-01-15T14:30:00Z",
    "ip_country": "DE",
    "gdpr_lawful_basis": "consent",
    "retention_expiry": "2026-01-15"
  }
}`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description:
              'Use ChatGPT or Claude to analyze consent patterns, identify compliance gaps, generate DSAR response packages, and draft audit-ready compliance reports.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'GDPR Compliance Analysis Prompt Template',
                description:
                  'Structured prompts for AI to analyze consent data and generate compliance insights.',
                code: `system_prompt: |
  You are a GDPR compliance expert specializing in consent management and
  data protection. Analyze consent records to identify compliance gaps,
  generate DSAR responses, and provide actionable remediation recommendations.
  Always reference specific GDPR articles when applicable.

dsar_response_prompt: |
  ## DSAR Response Generation

  ### Subject Request Details
  - Requester Email: {{requester_email}}
  - Request Type: {{request_type}} (access/rectification/erasure/portability)
  - Request Date: {{request_date}}
  - Response Deadline: {{deadline}} (30 days from request)

  ### Consent History for Subject
  {{consent_records_csv}}

  ### Data Holdings Summary
  {{data_systems_list}}

  ### Generate DSAR Response Package
  1. **Acknowledgment Letter**: Draft a formal acknowledgment confirming
     receipt of the DSAR and expected response timeline.

  2. **Data Inventory**: List all personal data held about the subject,
     organized by processing purpose and lawful basis.

  3. **Consent Timeline**: Provide a chronological record of all consent
     actions taken by the subject.

  4. **Third-Party Disclosures**: List any third parties to whom data
     has been disclosed, with dates and purposes.

  5. **Response Letter**: Draft the formal response letter including:
     - Summary of data held
     - Lawful basis for each processing activity
     - Retention periods
     - Subject's rights under GDPR Articles 15-22

  Format the response as a professional legal document suitable for
  regulatory review.

compliance_audit_prompt: |
  ## Weekly Consent Compliance Audit

  ### Consent Records Summary (Last 7 Days)
  - New consent grants: {{new_grants}}
  - Consent withdrawals: {{withdrawals}}
  - Pending DSARs: {{pending_dsars}}

  ### Consent Coverage by System
  {{system_coverage_table}}

  ### Analysis Tasks
  1. **Gap Analysis**: Identify systems where consent coverage is below 95%

  2. **Propagation Audit**: Flag records where consent changes took >4 hours
     to propagate across systems

  3. **Jurisdiction Review**: Highlight any consent records from EU subjects
     that lack explicit opt-in (required under GDPR)

  4. **Expiring Consents**: List consents expiring in the next 30 days
     that require re-consent campaigns

  5. **Remediation Priorities**: Rank compliance issues by risk severity
     (GDPR fine potential) and provide specific remediation steps

  ### Output Format
  Provide a structured audit report with:
  - Executive summary (3-5 bullet points)
  - Detailed findings by category
  - Risk-prioritized action items with owners and deadlines`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier to trigger AI-powered DSAR responses, sync consent changes across systems, and deliver weekly compliance reports to legal and marketing teams.',
            toolsUsed: ['Zapier Pro', 'OpenAI API', 'Slack', 'Gmail'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier GDPR Compliance Automation Workflow',
                description:
                  'Automated workflows for consent monitoring, DSAR processing, and compliance reporting.',
                code: `{
  "workflows": [
    {
      "name": "Consent Change Propagation",
      "trigger": {
        "app": "Webhooks by Zapier",
        "event": "Catch Hook",
        "config": {
          "webhook_url": "{{CONSENT_WEBHOOK_URL}}",
          "description": "Receives consent change events from CMP"
        }
      },
      "actions": [
        {
          "step": 1,
          "app": "Airtable",
          "action": "Create Record",
          "config": {
            "base_id": "{{CONSENT_BASE_ID}}",
            "table": "Consent_Ledger",
            "fields": {
              "user_id": "{{trigger.userId}}",
              "consent_purpose": "{{trigger.purpose}}",
              "consent_status": "{{trigger.action}}",
              "source_system": "website_cmp",
              "recorded_at": "{{trigger.timestamp}}"
            }
          }
        },
        {
          "step": 2,
          "app": "HubSpot",
          "action": "Update Contact",
          "config": {
            "contact_email": "{{trigger.email}}",
            "properties": {
              "gdpr_consent_{{trigger.purpose}}": "{{trigger.action}}",
              "gdpr_consent_date": "{{trigger.timestamp}}"
            }
          }
        },
        {
          "step": 3,
          "app": "Mailchimp",
          "action": "Update Subscriber",
          "config": {
            "email": "{{trigger.email}}",
            "status": "{{trigger.action == 'granted' ? 'subscribed' : 'unsubscribed'}}"
          }
        }
      ]
    },
    {
      "name": "DSAR Auto-Response Generator",
      "trigger": {
        "app": "Gmail",
        "event": "New Email Matching Search",
        "config": {
          "search_query": "subject:(DSAR OR 'data subject' OR 'access request' OR 'right to erasure')"
        }
      },
      "actions": [
        {
          "step": 1,
          "app": "Airtable",
          "action": "Find Records",
          "config": {
            "base_id": "{{CONSENT_BASE_ID}}",
            "table": "Consent_Ledger",
            "formula": "SEARCH('{{trigger.from_email}}', {user_email})"
          }
        },
        {
          "step": 2,
          "app": "OpenAI (ChatGPT)",
          "action": "Send Prompt",
          "config": {
            "model": "gpt-4-turbo",
            "system_message": "You are a GDPR compliance expert...",
            "user_message": "Generate DSAR response package for:\\nEmail: {{trigger.from_email}}\\nRequest: {{trigger.body}}\\nConsent History:\\n{{step1_records}}",
            "temperature": 0.2,
            "max_tokens": 3000
          }
        },
        {
          "step": 3,
          "app": "Airtable",
          "action": "Create Record",
          "config": {
            "base_id": "{{CONSENT_BASE_ID}}",
            "table": "DSAR_Requests",
            "fields": {
              "requester_email": "{{trigger.from_email}}",
              "request_date": "{{trigger.date}}",
              "deadline": "{{trigger.date + 30 days}}",
              "ai_draft_response": "{{step2_response}}",
              "status": "pending_review"
            }
          }
        },
        {
          "step": 4,
          "app": "Slack",
          "action": "Send Channel Message",
          "config": {
            "channel": "#legal-compliance",
            "message": "New DSAR received from {{trigger.from_email}}\\nDeadline: {{step3.deadline}}\\nAI draft response ready for review in Airtable"
          }
        }
      ]
    },
    {
      "name": "Weekly Compliance Report",
      "trigger": {
        "app": "Schedule by Zapier",
        "event": "Every Week",
        "config": {
          "day_of_week": "Monday",
          "time": "09:00"
        }
      },
      "actions": [
        {
          "step": 1,
          "app": "Airtable",
          "action": "Find Records",
          "config": {
            "base_id": "{{CONSENT_BASE_ID}}",
            "table": "Consent_Ledger",
            "formula": "IS_AFTER({recorded_at}, DATEADD(TODAY(), -7, 'days'))"
          }
        },
        {
          "step": 2,
          "app": "OpenAI (ChatGPT)",
          "action": "Send Prompt",
          "config": {
            "model": "gpt-4-turbo",
            "user_message": "Generate weekly GDPR compliance audit report:\\n{{step1_records_summary}}",
            "temperature": 0.3
          }
        },
        {
          "step": 3,
          "app": "Gmail",
          "action": "Send Email",
          "config": {
            "to": "legal@company.com, marketing-ops@company.com",
            "subject": "Weekly GDPR Compliance Report - {{current_date}}",
            "body_type": "html",
            "body": "{{step2_response_html}}"
          }
        }
      ]
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
          'Deploy a multi-agent system using CrewAI and LangGraph to continuously monitor consent across all systems, automatically process DSAR requests, detect compliance violations in real-time, and maintain an immutable audit trail for regulatory readiness.',
        estimatedMonthlyCost: '$600 - $1,200/month',
        architecture:
          'A Compliance Supervisor agent coordinates four specialist agents: Consent Monitor Agent tracks consent changes across systems, DSAR Processor Agent generates compliant response packages, Audit Agent validates consent propagation, and Remediation Agent triggers automated fixes for compliance gaps.',
        agents: [
          {
            name: 'Consent Monitor',
            role: 'Real-Time Consent Surveillance Specialist',
            goal: 'Continuously monitor consent events across CMP, CRM, email platform, and ad systems to ensure synchronization and detect discrepancies within seconds',
            tools: ['OneTrust Webhook Listener', 'HubSpot API', 'Mailchimp API', 'Google Ads API'],
          },
          {
            name: 'DSAR Processor',
            role: 'Data Subject Request Specialist',
            goal: 'Automatically process DSAR requests by aggregating all personal data holdings, generating compliant response packages, and tracking response deadlines',
            tools: ['Data Catalog API', 'Document Generator', 'Email Parser', 'Deadline Tracker'],
          },
          {
            name: 'Compliance Auditor',
            role: 'GDPR Compliance Validation Specialist',
            goal: 'Continuously audit consent records for compliance gaps, validate propagation latency, and generate regulatory-ready audit reports',
            tools: ['SQL Query Engine', 'Compliance Rules Engine', 'Report Generator', 'Risk Scorer'],
          },
          {
            name: 'Remediation Agent',
            role: 'Automated Compliance Fix Specialist',
            goal: 'Automatically remediate detected compliance gaps by triggering consent re-sync, updating suppression lists, and escalating critical issues to legal',
            tools: ['System Sync APIs', 'Suppression List Manager', 'Escalation Workflow', 'Slack Notifier'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Immutable append-only PostgreSQL ledger with Redis caching for real-time queries',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the GDPR compliance multi-agent system with CrewAI, establishing specialized roles for consent monitoring, DSAR processing, audit validation, and automated remediation.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'CrewAI GDPR Compliance Agent Definitions',
                description:
                  'Production-ready agent definitions for the consent management and DSAR processing system.',
                code: `from crewai import Agent, Crew, Task, Process
from crewai.tools import BaseTool
from typing import List, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class GDPRAgentConfig(BaseModel):
    """Configuration for GDPR compliance agents."""
    llm_model: str = Field(default="gpt-4-turbo")
    temperature: float = Field(default=0.1)
    max_iterations: int = Field(default=15)
    verbose: bool = Field(default=True)


def create_consent_monitor_agent(
    config: GDPRAgentConfig,
    tools: List[BaseTool],
) -> Agent:
    """Create the real-time consent surveillance agent."""
    return Agent(
        role="Consent Monitor",
        goal="""Continuously monitor consent events across all integrated systems
        (CMP, CRM, email platform, ad platforms). Detect consent changes within
        seconds and ensure synchronization across all data processors. Flag any
        discrepancies or propagation failures immediately.""",
        backstory="""You are a vigilant data protection specialist with expertise
        in real-time event processing. You understand that consent is the cornerstone
        of GDPR compliance, and even a few minutes of unsynchronized consent state
        can expose the organization to regulatory risk. You monitor webhooks, poll
        APIs, and correlate events across systems to maintain a single source of truth.""",
        tools=tools,
        llm=config.llm_model,
        verbose=config.verbose,
        allow_delegation=False,
        max_iter=config.max_iterations,
    )


def create_dsar_processor_agent(
    config: GDPRAgentConfig,
    tools: List[BaseTool],
) -> Agent:
    """Create the data subject request processing agent."""
    return Agent(
        role="DSAR Processor",
        goal="""Process Data Subject Access Requests (DSARs) efficiently and
        compliantly. Aggregate personal data from all systems, generate legally
        compliant response packages, and ensure responses are delivered within
        the 30-day GDPR deadline.""",
        backstory="""You are a GDPR legal expert who has processed thousands of
        DSARs. You know exactly what data must be disclosed under Article 15,
        how to format responses for regulatory review, and how to handle complex
        requests involving rectification (Article 16), erasure (Article 17), and
        data portability (Article 20). You work methodically to ensure no data
        is missed and no deadline is breached.""",
        tools=tools,
        llm=config.llm_model,
        verbose=config.verbose,
        allow_delegation=True,
        max_iter=config.max_iterations,
    )


def create_compliance_auditor_agent(
    config: GDPRAgentConfig,
    tools: List[BaseTool],
) -> Agent:
    """Create the GDPR compliance validation agent."""
    return Agent(
        role="Compliance Auditor",
        goal="""Continuously audit consent records and data processing activities
        for GDPR compliance. Validate that consent propagation meets SLA thresholds,
        identify coverage gaps, and generate regulatory-ready audit reports that
        can withstand DPA scrutiny.""",
        backstory="""You are a former Data Protection Authority auditor who now
        helps organizations achieve proactive compliance. You know exactly what
        regulators look for during inspections, and you build audit trails that
        demonstrate accountability under Article 5(2). You score compliance risks
        and prioritize remediation based on potential fine exposure.""",
        tools=tools,
        llm=config.llm_model,
        verbose=config.verbose,
        allow_delegation=True,
        max_iter=config.max_iterations,
    )


def create_remediation_agent(
    config: GDPRAgentConfig,
    tools: List[BaseTool],
) -> Agent:
    """Create the automated compliance fix agent."""
    return Agent(
        role="Remediation Agent",
        goal="""Automatically remediate detected compliance gaps by triggering
        consent re-synchronization, updating suppression lists, and escalating
        critical issues to legal counsel. Minimize human intervention while
        maintaining full audit trails.""",
        backstory="""You are an automation specialist who believes that compliance
        should be continuous, not reactive. You have built integrations with every
        system in the martech stack and can trigger fixes within seconds of
        detecting an issue. You know when to auto-remediate and when to escalate—
        you never take actions that could make compliance worse.""",
        tools=tools,
        llm=config.llm_model,
        verbose=config.verbose,
        allow_delegation=False,
        max_iter=config.max_iterations,
    )


def create_gdpr_compliance_crew(
    config: GDPRAgentConfig,
    monitor_tools: List[BaseTool],
    dsar_tools: List[BaseTool],
    audit_tools: List[BaseTool],
    remediation_tools: List[BaseTool],
) -> Crew:
    """Assemble the full GDPR compliance crew."""
    monitor = create_consent_monitor_agent(config, monitor_tools)
    dsar_processor = create_dsar_processor_agent(config, dsar_tools)
    auditor = create_compliance_auditor_agent(config, audit_tools)
    remediator = create_remediation_agent(config, remediation_tools)

    return Crew(
        agents=[monitor, dsar_processor, auditor, remediator],
        process=Process.hierarchical,
        manager_llm=config.llm_model,
        verbose=config.verbose,
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the consent monitoring agent with tools for listening to CMP webhooks, polling CRM APIs, and correlating consent events across all integrated systems.',
            toolsUsed: ['OneTrust API', 'HubSpot API', 'Mailchimp API', 'PostgreSQL'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Consent Monitoring Agent Tools',
                description:
                  'Custom tools for real-time consent event ingestion and correlation.',
                code: `from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Literal
from datetime import datetime, timezone, timedelta
import hashlib
import logging
import asyncio

logger = logging.getLogger(__name__)


class ConsentEvent(BaseModel):
    """Schema for normalized consent events."""
    event_id: str
    user_id: str
    user_email: str
    consent_purpose: str
    consent_status: Literal["granted", "denied", "withdrawn"]
    source_system: str
    collection_method: str
    ip_country: str
    recorded_at: datetime
    raw_payload: Dict[str, Any]


class CMPWebhookListenerTool(BaseTool):
    """Tool to process incoming CMP webhook events."""

    name: str = "cmp_webhook_listener"
    description: str = """Processes consent change events from the CMP webhook.
    Normalizes events and writes to the consent ledger."""

    def __init__(self, db_conn, downstream_systems: List[str]):
        super().__init__()
        self._db = db_conn
        self._downstream = downstream_systems

    def _run(self, webhook_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process a CMP webhook payload."""
        try:
            # Normalize the event
            event = ConsentEvent(
                event_id=hashlib.sha256(
                    f"{webhook_payload['userId']}_{webhook_payload['timestamp']}".encode()
                ).hexdigest()[:32],
                user_id=webhook_payload["userId"],
                user_email=webhook_payload.get("email", ""),
                consent_purpose=webhook_payload["purpose"],
                consent_status=webhook_payload["action"],
                source_system="cmp_webhook",
                collection_method=webhook_payload.get("method", "banner"),
                ip_country=webhook_payload.get("country", "unknown"),
                recorded_at=datetime.now(timezone.utc),
                raw_payload=webhook_payload,
            )

            # Write to ledger
            self._db.execute(
                """
                INSERT INTO consent_ledger
                (event_id, user_id, user_email, consent_purpose, consent_status,
                 source_system, collection_method, ip_country, recorded_at, raw_payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    event.event_id, event.user_id, event.user_email,
                    event.consent_purpose, event.consent_status,
                    event.source_system, event.collection_method,
                    event.ip_country, event.recorded_at, event.raw_payload,
                ),
            )

            logger.info(f"Recorded consent event {event.event_id} for {event.user_id}")

            return {
                "status": "success",
                "event_id": event.event_id,
                "user_id": event.user_id,
                "consent_status": event.consent_status,
                "pending_propagation": self._downstream,
            }

        except Exception as e:
            logger.error(f"Failed to process CMP webhook: {e}")
            return {"status": "error", "message": str(e)}


class ConsentPropagationTool(BaseTool):
    """Tool to propagate consent changes to downstream systems."""

    name: str = "consent_propagator"
    description: str = """Propagates consent changes to CRM, email platform,
    and ad systems. Tracks propagation latency and reports failures."""

    def __init__(self, system_clients: Dict[str, Any], db_conn):
        super().__init__()
        self._clients = system_clients
        self._db = db_conn

    async def _propagate_to_system(
        self,
        system: str,
        event: ConsentEvent,
    ) -> Dict[str, Any]:
        """Propagate consent to a single downstream system."""
        client = self._clients.get(system)
        if not client:
            return {"system": system, "status": "error", "message": "Client not configured"}

        start_time = datetime.now(timezone.utc)
        try:
            await client.update_consent(
                user_id=event.user_id,
                email=event.user_email,
                purpose=event.consent_purpose,
                status=event.consent_status,
            )
            latency = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Log propagation
            self._db.execute(
                """
                INSERT INTO consent_propagation_log
                (event_id, target_system, propagated_at, latency_seconds, status)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (event.event_id, system, datetime.now(timezone.utc), latency, "success"),
            )

            return {"system": system, "status": "success", "latency_seconds": latency}

        except Exception as e:
            logger.error(f"Propagation to {system} failed: {e}")
            return {"system": system, "status": "error", "message": str(e)}

    def _run(self, event_id: str) -> Dict[str, Any]:
        """Propagate a consent event to all downstream systems."""
        # Fetch event from ledger
        row = self._db.execute(
            "SELECT * FROM consent_ledger WHERE event_id = %s", (event_id,)
        ).fetchone()

        if not row:
            return {"status": "error", "message": f"Event {event_id} not found"}

        event = ConsentEvent(**dict(row))

        # Propagate to all systems concurrently
        results = asyncio.run(
            asyncio.gather(*[
                self._propagate_to_system(system, event)
                for system in self._clients.keys()
            ])
        )

        successes = [r for r in results if r["status"] == "success"]
        failures = [r for r in results if r["status"] == "error"]

        return {
            "status": "partial_success" if failures else "success",
            "event_id": event_id,
            "propagated_to": [r["system"] for r in successes],
            "failed_systems": [r["system"] for r in failures],
            "total_latency_seconds": sum(r.get("latency_seconds", 0) for r in successes),
        }


class ConsentDiscrepancyDetectorTool(BaseTool):
    """Tool to detect consent discrepancies across systems."""

    name: str = "consent_discrepancy_detector"
    description: str = """Compares consent state across all systems to detect
    discrepancies where consent status is out of sync."""

    def __init__(self, system_clients: Dict[str, Any], db_conn):
        super().__init__()
        self._clients = system_clients
        self._db = db_conn

    def _run(self, user_id: str) -> Dict[str, Any]:
        """Check consent state across all systems for a user."""
        # Get canonical state from ledger
        ledger_state = self._db.execute(
            """
            SELECT consent_purpose, consent_status
            FROM current_consent
            WHERE user_id = %s
            """,
            (user_id,),
        ).fetchall()

        canonical = {row["consent_purpose"]: row["consent_status"] for row in ledger_state}

        # Check each downstream system
        discrepancies = []
        for system, client in self._clients.items():
            try:
                system_state = client.get_consent_state(user_id)
                for purpose, expected_status in canonical.items():
                    actual_status = system_state.get(purpose)
                    if actual_status != expected_status:
                        discrepancies.append({
                            "system": system,
                            "purpose": purpose,
                            "expected": expected_status,
                            "actual": actual_status,
                        })
            except Exception as e:
                logger.error(f"Failed to check {system}: {e}")
                discrepancies.append({
                    "system": system,
                    "purpose": "all",
                    "expected": "check",
                    "actual": f"error: {e}",
                })

        return {
            "status": "discrepancies_found" if discrepancies else "in_sync",
            "user_id": user_id,
            "canonical_state": canonical,
            "discrepancies": discrepancies,
            "systems_checked": list(self._clients.keys()),
        }`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the DSAR processor and compliance auditor agents with tools for generating legally compliant response packages, validating audit trails, and scoring compliance risks.',
            toolsUsed: ['Document Generator', 'Compliance Rules Engine', 'Risk Scorer'],
            codeSnippets: [
              {
                language: 'python',
                title: 'DSAR Processing and Compliance Audit Tools',
                description:
                  'Tools for automated DSAR response generation and compliance validation.',
                code: `from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DSARType(str, Enum):
    ACCESS = "access"           # Article 15
    RECTIFICATION = "rectification"  # Article 16
    ERASURE = "erasure"         # Article 17
    PORTABILITY = "portability"      # Article 20


class DSARRequest(BaseModel):
    """Schema for DSAR requests."""
    request_id: str
    requester_email: str
    request_type: DSARType
    request_date: datetime
    deadline: datetime
    additional_info: Optional[str] = None


class DSARResponseGeneratorTool(BaseTool):
    """Tool to generate GDPR-compliant DSAR response packages."""

    name: str = "dsar_response_generator"
    description: str = """Generates legally compliant DSAR response packages
    including data inventory, consent history, and formal response letters."""

    def __init__(self, db_conn, data_catalog: Dict[str, Any]):
        super().__init__()
        self._db = db_conn
        self._catalog = data_catalog

    def _run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a complete DSAR response package."""
        dsar = DSARRequest(**request)

        # Gather consent history
        consent_history = self._db.execute(
            """
            SELECT consent_purpose, consent_status, source_system,
                   collection_method, recorded_at
            FROM consent_ledger
            WHERE user_email = %s
            ORDER BY recorded_at
            """,
            (dsar.requester_email,),
        ).fetchall()

        # Gather data holdings from each system
        data_holdings = {}
        for system, config in self._catalog.items():
            try:
                data_holdings[system] = self._query_system_data(
                    system, config, dsar.requester_email
                )
            except Exception as e:
                logger.error(f"Failed to query {system}: {e}")
                data_holdings[system] = {"error": str(e)}

        # Generate response documents
        response_package = {
            "request_id": dsar.request_id,
            "request_type": dsar.request_type.value,
            "requester_email": dsar.requester_email,
            "response_deadline": dsar.deadline.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),

            "acknowledgment_letter": self._generate_acknowledgment(dsar),
            "consent_timeline": [dict(row) for row in consent_history],
            "data_inventory": data_holdings,
            "third_party_disclosures": self._get_third_party_disclosures(dsar.requester_email),
            "response_letter": self._generate_response_letter(dsar, consent_history, data_holdings),
        }

        # Record in DSAR tracking table
        self._db.execute(
            """
            INSERT INTO dsar_requests
            (request_id, requester_email, request_type, request_date,
             deadline, response_generated_at, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                dsar.request_id, dsar.requester_email, dsar.request_type.value,
                dsar.request_date, dsar.deadline, datetime.now(timezone.utc),
                "response_generated",
            ),
        )

        return {"status": "success", "response_package": response_package}

    def _generate_acknowledgment(self, dsar: DSARRequest) -> str:
        """Generate acknowledgment letter text."""
        return f"""
Dear Data Subject,

We acknowledge receipt of your Data Subject Access Request dated {dsar.request_date.strftime('%d %B %Y')}.

Under Article 12(3) of the General Data Protection Regulation (GDPR), we will respond
to your request within one month of receipt. Your response deadline is {dsar.deadline.strftime('%d %B %Y')}.

Request Reference: {dsar.request_id}
Request Type: {dsar.request_type.value.title()} (Article {self._get_article_number(dsar.request_type)})

If you have any questions, please contact our Data Protection Officer at dpo@company.com.

Regards,
Data Protection Team
        """

    def _get_article_number(self, request_type: DSARType) -> int:
        """Map request type to GDPR article number."""
        mapping = {
            DSARType.ACCESS: 15,
            DSARType.RECTIFICATION: 16,
            DSARType.ERASURE: 17,
            DSARType.PORTABILITY: 20,
        }
        return mapping.get(request_type, 15)

    def _query_system_data(
        self, system: str, config: Dict, email: str
    ) -> Dict[str, Any]:
        """Query data holdings from a specific system."""
        # Implementation would call system APIs
        return {"system": system, "data_categories": config.get("data_categories", [])}

    def _get_third_party_disclosures(self, email: str) -> List[Dict]:
        """Get list of third parties data was shared with."""
        return self._db.execute(
            """
            SELECT third_party_name, disclosure_purpose, disclosure_date
            FROM third_party_disclosures
            WHERE user_email = %s
            ORDER BY disclosure_date
            """,
            (email,),
        ).fetchall()

    def _generate_response_letter(
        self,
        dsar: DSARRequest,
        consent_history: List,
        data_holdings: Dict,
    ) -> str:
        """Generate the formal response letter."""
        # Full implementation would format all data appropriately
        return f"DSAR Response for {dsar.requester_email}..."


class ComplianceRiskScorerTool(BaseTool):
    """Tool to score and prioritize compliance risks."""

    name: str = "compliance_risk_scorer"
    description: str = """Scores compliance issues by risk severity based on
    GDPR fine potential and likelihood of regulatory scrutiny."""

    RISK_WEIGHTS = {
        "consent_coverage_gap": 0.3,
        "propagation_sla_breach": 0.2,
        "missing_lawful_basis": 0.4,
        "dsar_deadline_risk": 0.35,
        "third_party_compliance": 0.25,
    }

    def _run(self, audit_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Score and prioritize compliance findings."""
        scored_findings = []

        for finding in audit_findings:
            risk_type = finding.get("risk_type", "unknown")
            base_weight = self.RISK_WEIGHTS.get(risk_type, 0.1)

            # Adjust based on severity factors
            affected_users = finding.get("affected_users", 0)
            duration_hours = finding.get("duration_hours", 0)
            is_eu_data = finding.get("is_eu_data", True)

            # Calculate risk score (0-100)
            risk_score = base_weight * 100
            if affected_users > 1000:
                risk_score *= 1.5
            if duration_hours > 24:
                risk_score *= 1.3
            if is_eu_data:
                risk_score *= 1.2

            risk_score = min(100, risk_score)

            # Estimate fine exposure
            fine_exposure = self._estimate_fine_exposure(risk_type, affected_users)

            scored_findings.append({
                **finding,
                "risk_score": round(risk_score, 1),
                "priority": "critical" if risk_score > 70 else "high" if risk_score > 50 else "medium",
                "estimated_fine_exposure": fine_exposure,
                "recommended_action": self._get_recommended_action(risk_type),
            })

        # Sort by risk score descending
        scored_findings.sort(key=lambda x: x["risk_score"], reverse=True)

        return {
            "status": "success",
            "total_findings": len(scored_findings),
            "critical_count": len([f for f in scored_findings if f["priority"] == "critical"]),
            "findings": scored_findings,
            "aggregate_risk_score": sum(f["risk_score"] for f in scored_findings) / len(scored_findings) if scored_findings else 0,
        }

    def _estimate_fine_exposure(self, risk_type: str, affected_users: int) -> str:
        """Estimate potential GDPR fine exposure."""
        # GDPR fines can be up to 4% of annual turnover or EUR 20M
        base_exposure = {
            "consent_coverage_gap": 50000,
            "propagation_sla_breach": 25000,
            "missing_lawful_basis": 100000,
            "dsar_deadline_risk": 75000,
            "third_party_compliance": 60000,
        }
        exposure = base_exposure.get(risk_type, 10000) * (1 + affected_users / 10000)
        return f"EUR {exposure:,.0f} - {exposure * 2:,.0f}"

    def _get_recommended_action(self, risk_type: str) -> str:
        """Get recommended remediation action."""
        actions = {
            "consent_coverage_gap": "Trigger consent re-sync for affected users",
            "propagation_sla_breach": "Investigate system delays, increase sync frequency",
            "missing_lawful_basis": "Review processing activities, document lawful basis",
            "dsar_deadline_risk": "Escalate to legal, prioritize response completion",
            "third_party_compliance": "Audit third-party DPAs, validate data flows",
        }
        return actions.get(risk_type, "Review and remediate manually")`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement the LangGraph state machine that coordinates GDPR compliance workflows, manages DSAR processing queues, triggers automated remediation, and maintains immutable audit trails.',
            toolsUsed: ['LangGraph', 'PostgreSQL', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph GDPR Compliance Workflow Orchestration',
                description:
                  'State machine implementation for coordinating the multi-agent GDPR compliance system.',
                code: `from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from typing import TypedDict, Annotated, List, Dict, Any, Literal
from datetime import datetime, timezone, timedelta
import operator
import logging
import json

logger = logging.getLogger(__name__)


class GDPRComplianceState(TypedDict):
    """State schema for the GDPR compliance workflow."""
    # Workflow metadata
    run_id: str
    workflow_type: Literal["consent_sync", "dsar_processing", "compliance_audit", "remediation"]
    started_at: str
    status: Literal["running", "completed", "failed", "escalated"]

    # Consent monitoring state
    consent_events_processed: int
    propagation_failures: List[Dict[str, Any]]
    discrepancies_detected: List[Dict[str, Any]]

    # DSAR processing state
    dsar_request: Dict[str, Any]
    response_package: Dict[str, Any]
    dsar_status: Literal["pending", "processing", "review", "sent", "overdue"]

    # Compliance audit state
    audit_findings: List[Dict[str, Any]]
    risk_scores: Dict[str, float]
    compliance_score: float

    # Remediation state
    remediation_actions: List[Dict[str, Any]]
    auto_remediated: int
    escalated_issues: List[Dict[str, Any]]

    # Audit trail
    messages: Annotated[List[str], operator.add]
    errors: List[str]


def create_consent_sync_workflow(
    consent_monitor_agent,
    remediation_agent,
    sla_threshold_seconds: int = 30,
) -> StateGraph:
    """Create workflow for real-time consent synchronization."""

    workflow = StateGraph(GDPRComplianceState)

    def process_consent_events(state: GDPRComplianceState) -> GDPRComplianceState:
        """Process incoming consent events."""
        logger.info("Processing consent events...")
        try:
            result = consent_monitor_agent.execute_task(
                "Process all pending consent events from the webhook queue"
            )
            state["consent_events_processed"] = result.get("events_processed", 0)
            state["propagation_failures"] = result.get("failures", [])
            state["messages"] = [f"Processed {state['consent_events_processed']} consent events"]
        except Exception as e:
            state["errors"] = [f"Consent processing failed: {str(e)}"]
            state["status"] = "failed"
        return state

    def check_propagation_sla(state: GDPRComplianceState) -> GDPRComplianceState:
        """Verify consent changes propagated within SLA."""
        logger.info("Checking propagation SLA...")
        try:
            result = consent_monitor_agent.execute_task(
                f"Check propagation latency for the last {state['consent_events_processed']} events. "
                f"Flag any that exceeded {sla_threshold_seconds} seconds."
            )
            sla_breaches = result.get("sla_breaches", [])
            if sla_breaches:
                state["discrepancies_detected"].extend(sla_breaches)
                state["messages"] = [f"Detected {len(sla_breaches)} SLA breaches"]
            else:
                state["messages"] = ["All propagations within SLA"]
        except Exception as e:
            state["errors"] = [f"SLA check failed: {str(e)}"]
        return state

    def detect_discrepancies(state: GDPRComplianceState) -> GDPRComplianceState:
        """Detect consent state discrepancies across systems."""
        logger.info("Detecting cross-system discrepancies...")
        try:
            result = consent_monitor_agent.execute_task(
                "Compare consent state across all downstream systems. "
                "Report any discrepancies between the consent ledger and system state."
            )
            discrepancies = result.get("discrepancies", [])
            state["discrepancies_detected"].extend(discrepancies)
            state["messages"] = [f"Found {len(discrepancies)} consent discrepancies"]
        except Exception as e:
            state["errors"] = [f"Discrepancy detection failed: {str(e)}"]
        return state

    def auto_remediate(state: GDPRComplianceState) -> GDPRComplianceState:
        """Automatically remediate detected issues."""
        logger.info("Auto-remediating compliance issues...")
        if not state["discrepancies_detected"] and not state["propagation_failures"]:
            state["messages"] = ["No issues to remediate"]
            state["status"] = "completed"
            return state

        try:
            all_issues = state["discrepancies_detected"] + state["propagation_failures"]
            result = remediation_agent.execute_task(
                f"Remediate the following {len(all_issues)} compliance issues: "
                f"{json.dumps(all_issues)}"
            )
            state["auto_remediated"] = result.get("remediated_count", 0)
            state["escalated_issues"] = result.get("escalated", [])

            if state["escalated_issues"]:
                state["status"] = "escalated"
                state["messages"] = [
                    f"Remediated {state['auto_remediated']} issues, "
                    f"escalated {len(state['escalated_issues'])} to legal"
                ]
            else:
                state["status"] = "completed"
                state["messages"] = [f"Successfully remediated all {state['auto_remediated']} issues"]
        except Exception as e:
            state["errors"] = [f"Remediation failed: {str(e)}"]
            state["status"] = "failed"
        return state

    # Add nodes
    workflow.add_node("process_events", process_consent_events)
    workflow.add_node("check_sla", check_propagation_sla)
    workflow.add_node("detect_discrepancies", detect_discrepancies)
    workflow.add_node("remediate", auto_remediate)

    # Add edges
    workflow.set_entry_point("process_events")
    workflow.add_edge("process_events", "check_sla")
    workflow.add_edge("check_sla", "detect_discrepancies")
    workflow.add_edge("detect_discrepancies", "remediate")
    workflow.add_edge("remediate", END)

    return workflow.compile()


def create_dsar_processing_workflow(
    dsar_processor_agent,
    compliance_auditor_agent,
) -> StateGraph:
    """Create workflow for DSAR request processing."""

    workflow = StateGraph(GDPRComplianceState)

    def validate_request(state: GDPRComplianceState) -> GDPRComplianceState:
        """Validate the incoming DSAR request."""
        logger.info(f"Validating DSAR request {state['dsar_request'].get('request_id')}")
        # Validation logic
        state["dsar_status"] = "processing"
        state["messages"] = ["DSAR request validated, beginning processing"]
        return state

    def generate_response(state: GDPRComplianceState) -> GDPRComplianceState:
        """Generate the DSAR response package."""
        logger.info("Generating DSAR response package...")
        try:
            result = dsar_processor_agent.execute_task(
                f"Generate complete DSAR response package for: {json.dumps(state['dsar_request'])}"
            )
            state["response_package"] = result.get("response_package", {})
            state["dsar_status"] = "review"
            state["messages"] = ["DSAR response package generated, pending legal review"]
        except Exception as e:
            state["errors"] = [f"Response generation failed: {str(e)}"]
            state["status"] = "failed"
        return state

    def compliance_review(state: GDPRComplianceState) -> GDPRComplianceState:
        """Audit the response for compliance."""
        logger.info("Running compliance review on DSAR response...")
        try:
            result = compliance_auditor_agent.execute_task(
                f"Review this DSAR response package for GDPR compliance: "
                f"{json.dumps(state['response_package'])}"
            )
            state["audit_findings"] = result.get("findings", [])
            state["compliance_score"] = result.get("compliance_score", 0.0)

            if state["compliance_score"] >= 0.95:
                state["dsar_status"] = "ready_to_send"
                state["status"] = "completed"
                state["messages"] = [f"DSAR response approved (score: {state['compliance_score']:.0%})"]
            else:
                state["status"] = "escalated"
                state["messages"] = [f"DSAR response needs manual review (score: {state['compliance_score']:.0%})"]
        except Exception as e:
            state["errors"] = [f"Compliance review failed: {str(e)}"]
            state["status"] = "failed"
        return state

    # Add nodes
    workflow.add_node("validate", validate_request)
    workflow.add_node("generate", generate_response)
    workflow.add_node("review", compliance_review)

    # Add edges
    workflow.set_entry_point("validate")
    workflow.add_edge("validate", "generate")
    workflow.add_edge("generate", "review")
    workflow.add_edge("review", END)

    return workflow.compile()`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the GDPR compliance agent system with Docker, implement comprehensive audit logging, and set up real-time compliance dashboards for legal and DPO visibility.',
            toolsUsed: ['Docker', 'LangSmith', 'PostgreSQL', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose for GDPR Compliance Agent System',
                description:
                  'Production deployment configuration with immutable audit logging.',
                code: `version: '3.8'

services:
  gdpr-agents:
    build:
      context: .
      dockerfile: Dockerfile.gdpr
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=gdpr-compliance
      - POSTGRES_DSN=postgresql://gdpr:secure_pass@postgres:5432/consent_ledger
      - REDIS_URL=redis://redis:6379/1
      - ONETRUST_WEBHOOK_SECRET=\${ONETRUST_WEBHOOK_SECRET}
      - HUBSPOT_API_KEY=\${HUBSPOT_API_KEY}
      - MAILCHIMP_API_KEY=\${MAILCHIMP_API_KEY}
      - SLACK_LEGAL_CHANNEL=\${SLACK_LEGAL_WEBHOOK}
      - LOG_LEVEL=INFO
      - DSAR_DEADLINE_DAYS=30
      - PROPAGATION_SLA_SECONDS=30
    depends_on:
      - postgres
      - redis
    ports:
      - "8080:8080"
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=gdpr
      - POSTGRES_PASSWORD=secure_pass
      - POSTGRES_DB=consent_ledger
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init.sql
    command: >
      postgres
        -c wal_level=logical
        -c max_wal_senders=10
        -c max_replication_slots=10

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_ADMIN_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    depends_on:
      - postgres

volumes:
  postgres_data:
  redis_data:
  grafana_data:`,
              },
              {
                language: 'python',
                title: 'Immutable Audit Trail Implementation',
                description:
                  'Append-only audit logging for GDPR accountability requirements.',
                code: `import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    CONSENT_RECEIVED = "consent_received"
    CONSENT_PROPAGATED = "consent_propagated"
    DSAR_RECEIVED = "dsar_received"
    DSAR_PROCESSED = "dsar_processed"
    DSAR_SENT = "dsar_sent"
    COMPLIANCE_AUDIT = "compliance_audit"
    REMEDIATION_ACTION = "remediation_action"
    ESCALATION = "escalation"


@dataclass
class AuditEvent:
    """Immutable audit event for GDPR accountability."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    actor: str  # System or user that triggered the event
    subject_id: Optional[str]  # User ID if applicable
    action: str
    details: Dict[str, Any]
    previous_hash: str
    event_hash: str

    def __post_init__(self):
        if not self.event_hash:
            self.event_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash for integrity verification."""
        content = json.dumps({
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "subject_id": self.subject_id,
            "action": self.action,
            "details": self.details,
            "previous_hash": self.previous_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class ImmutableAuditLog:
    """Append-only audit log with hash chain integrity."""

    def __init__(self, db_conn):
        self._db = db_conn
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create audit log table if not exists."""
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS gdpr_audit_log (
                event_id VARCHAR(64) PRIMARY KEY,
                event_type VARCHAR(50) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                actor VARCHAR(255) NOT NULL,
                subject_id VARCHAR(255),
                action TEXT NOT NULL,
                details JSONB NOT NULL,
                previous_hash VARCHAR(64) NOT NULL,
                event_hash VARCHAR(64) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON gdpr_audit_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_subject ON gdpr_audit_log(subject_id);
            CREATE INDEX IF NOT EXISTS idx_audit_type ON gdpr_audit_log(event_type);
        """)

    def _get_last_hash(self) -> str:
        """Get the hash of the most recent audit event."""
        row = self._db.execute(
            "SELECT event_hash FROM gdpr_audit_log ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return row["event_hash"] if row else "genesis"

    def log(
        self,
        event_type: AuditEventType,
        actor: str,
        action: str,
        details: Dict[str, Any],
        subject_id: Optional[str] = None,
    ) -> AuditEvent:
        """Append an event to the immutable audit log."""
        event_id = hashlib.sha256(
            f"{datetime.now(timezone.utc).isoformat()}_{actor}_{action}".encode()
        ).hexdigest()[:32]

        previous_hash = self._get_last_hash()

        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            actor=actor,
            subject_id=subject_id,
            action=action,
            details=details,
            previous_hash=previous_hash,
            event_hash="",  # Will be computed in __post_init__
        )

        self._db.execute(
            """
            INSERT INTO gdpr_audit_log
            (event_id, event_type, timestamp, actor, subject_id, action,
             details, previous_hash, event_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                event.event_id, event.event_type.value, event.timestamp,
                event.actor, event.subject_id, event.action,
                json.dumps(event.details), event.previous_hash, event.event_hash,
            ),
        )

        logger.info(f"Audit event logged: {event.event_id} ({event.event_type.value})")
        return event

    def verify_chain_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the entire audit chain."""
        events = self._db.execute(
            "SELECT * FROM gdpr_audit_log ORDER BY timestamp"
        ).fetchall()

        if not events:
            return {"status": "empty", "events_checked": 0}

        previous_hash = "genesis"
        invalid_events = []

        for row in events:
            event = AuditEvent(**dict(row))
            if event.previous_hash != previous_hash:
                invalid_events.append({
                    "event_id": event.event_id,
                    "expected_previous": previous_hash,
                    "actual_previous": event.previous_hash,
                })

            # Verify event hash
            recomputed = event._compute_hash()
            if recomputed != event.event_hash:
                invalid_events.append({
                    "event_id": event.event_id,
                    "issue": "hash_mismatch",
                    "stored": event.event_hash,
                    "computed": recomputed,
                })

            previous_hash = event.event_hash

        return {
            "status": "valid" if not invalid_events else "corrupted",
            "events_checked": len(events),
            "invalid_events": invalid_events,
        }

    def get_subject_audit_trail(self, subject_id: str) -> List[Dict[str, Any]]:
        """Get complete audit trail for a data subject (for DSAR responses)."""
        events = self._db.execute(
            """
            SELECT event_type, timestamp, action, details
            FROM gdpr_audit_log
            WHERE subject_id = %s
            ORDER BY timestamp
            """,
            (subject_id,),
        ).fetchall()
        return [dict(row) for row in events]`,
              },
            ],
          },
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

      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to consolidate marketing data from multiple platforms into a unified Google Sheets dashboard, automate cross-platform performance analysis, and deliver actionable insights without custom development.',
        estimatedMonthlyCost: '$90 - $150/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Supermetrics ($39/mo)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Google Analytics AI (Free)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Use Supermetrics or native Zapier connectors to pull campaign data from Google Ads, Meta, TikTok, and LinkedIn into a centralized Google Sheets workbook with standardized column naming.',
            toolsUsed: ['Supermetrics', 'Google Sheets', 'Zapier'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Unified Marketing Data Schema for Google Sheets',
                description:
                  'Standardized schema for consolidating cross-platform marketing data.',
                code: `{
  "workbook_name": "Unified_Marketing_Dashboard",
  "sheets": [
    {
      "name": "Daily_Metrics",
      "columns": [
        {"name": "date", "type": "date", "format": "YYYY-MM-DD"},
        {"name": "platform", "type": "string", "values": ["google_ads", "meta", "tiktok", "linkedin"]},
        {"name": "campaign_id", "type": "string", "description": "Platform-specific campaign ID"},
        {"name": "campaign_name", "type": "string", "description": "Original campaign name"},
        {"name": "normalized_name", "type": "string", "description": "Standardized name for grouping"},
        {"name": "channel_group", "type": "string", "values": ["paid_search", "paid_social", "display", "video"]},
        {"name": "impressions", "type": "integer"},
        {"name": "clicks", "type": "integer"},
        {"name": "spend", "type": "currency", "currency": "USD"},
        {"name": "platform_conversions", "type": "integer", "description": "Platform-reported conversions"},
        {"name": "platform_revenue", "type": "currency", "description": "Platform-reported revenue"},
        {"name": "ctr", "type": "percentage", "formula": "=clicks/impressions"},
        {"name": "cpc", "type": "currency", "formula": "=spend/clicks"},
        {"name": "platform_roas", "type": "decimal", "formula": "=platform_revenue/spend"}
      ]
    },
    {
      "name": "Platform_Summary",
      "description": "Aggregated view by platform and date range",
      "columns": [
        {"name": "platform", "type": "string"},
        {"name": "total_spend", "type": "currency"},
        {"name": "total_conversions", "type": "integer"},
        {"name": "blended_cpa", "type": "currency"},
        {"name": "blended_roas", "type": "decimal"},
        {"name": "spend_share_pct", "type": "percentage"}
      ]
    },
    {
      "name": "Conversion_Reconciliation",
      "description": "Compare platform vs server-side conversions",
      "columns": [
        {"name": "date", "type": "date"},
        {"name": "platform", "type": "string"},
        {"name": "platform_conversions", "type": "integer"},
        {"name": "server_conversions", "type": "integer"},
        {"name": "overcount_pct", "type": "percentage", "formula": "=(platform-server)/platform"}
      ]
    }
  ],
  "data_refresh": {
    "frequency": "daily",
    "time": "06:00 UTC",
    "lookback_days": 7
  },
  "sample_row": {
    "date": "2024-01-15",
    "platform": "meta",
    "campaign_id": "23847562341",
    "campaign_name": "US_Prospecting_LAL_Jan24",
    "normalized_name": "us_prospecting_lookalike",
    "channel_group": "paid_social",
    "impressions": 125000,
    "clicks": 3750,
    "spend": 2500.00,
    "platform_conversions": 85,
    "platform_revenue": 6375.00,
    "ctr": 0.03,
    "cpc": 0.67,
    "platform_roas": 2.55
  }
}`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description:
              'Use ChatGPT or Claude to analyze cross-platform performance, identify spend inefficiencies, detect conversion overcounting, and generate budget reallocation recommendations.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Cross-Platform Analysis Prompt Templates',
                description:
                  'Structured prompts for AI to analyze multi-platform marketing data and generate insights.',
                code: `system_prompt: |
  You are a performance marketing analyst specializing in cross-platform
  campaign optimization. You analyze data from Google Ads, Meta, TikTok,
  and LinkedIn to identify inefficiencies and opportunities. You understand
  that each platform reports conversions differently and can detect over-
  counting patterns.

weekly_performance_prompt: |
  ## Cross-Platform Performance Analysis

  ### Reporting Period
  - Date Range: {{start_date}} to {{end_date}}
  - Total Spend: {{total_spend}}
  - Total Platform-Reported Conversions: {{total_platform_conversions}}
  - Total Server-Verified Conversions: {{total_server_conversions}}

  ### Platform Data (CSV)
  {{platform_data_csv}}

  ### Analysis Tasks

  1. **Platform Performance Comparison**
     For each platform, calculate and compare:
     - Spend share (% of total budget)
     - Conversion share (% of verified conversions)
     - Efficiency index (conversion share / spend share)
     - True ROAS using server-verified conversions

  2. **Overcount Analysis**
     Identify platforms where reported conversions significantly exceed
     server-verified conversions:
     - Calculate overcount percentage per platform
     - Flag platforms with >30% overcounting
     - Estimate wasted spend attributed to phantom conversions

  3. **Channel Mix Optimization**
     Based on true performance data:
     - Identify underinvested channels (high efficiency, low spend share)
     - Identify overinvested channels (low efficiency, high spend share)
     - Recommend specific budget shifts with dollar amounts

  4. **Anomaly Detection**
     Flag any unusual patterns:
     - Day-over-day spend spikes >50%
     - CTR drops >25% vs previous period
     - CPA increases >40% on any platform

  5. **Action Items**
     Provide 5-7 specific, actionable recommendations:
     - Budget changes (with exact amounts)
     - Campaign pauses or scaling decisions
     - Creative refresh needs
     - Audience optimization suggestions

  ### Output Format
  Structure your response with clear headers, tables where appropriate,
  and prioritized action items with expected impact.

budget_reallocation_prompt: |
  ## Budget Reallocation Recommendation

  ### Current Budget Allocation
  {{current_allocation_table}}

  ### Performance Data (Last 30 Days)
  {{performance_data_csv}}

  ### Constraints
  - Total monthly budget: {{total_budget}}
  - Minimum per platform: {{min_platform_spend}} (maintain presence)
  - Maximum change per platform: 25% from current allocation
  - Must maintain brand campaign coverage on Google

  ### Generate Reallocation Plan
  1. Calculate optimal allocation based on true ROAS (server-verified)
  2. Apply constraints to create feasible allocation
  3. Project impact on total conversions and blended ROAS
  4. Provide implementation timeline (gradual vs immediate)

  Output as a table with:
  | Platform | Current Spend | Recommended Spend | Change | Expected Impact |`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier to automatically consolidate data daily, run AI-powered analysis weekly, and deliver insights to marketing team via Slack and executive dashboards via email.',
            toolsUsed: ['Zapier Pro', 'OpenAI API', 'Slack', 'Gmail'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Cross-Platform Insights Workflow',
                description:
                  'Automated workflow for daily data consolidation and weekly AI analysis.',
                code: `{
  "workflows": [
    {
      "name": "Daily Platform Data Sync",
      "trigger": {
        "app": "Schedule by Zapier",
        "event": "Every Day",
        "config": {
          "time": "06:00",
          "timezone": "UTC"
        }
      },
      "actions": [
        {
          "step": 1,
          "app": "Supermetrics",
          "action": "Run Query",
          "config": {
            "data_source": "Google Ads",
            "metrics": ["impressions", "clicks", "cost", "conversions", "conversion_value"],
            "dimensions": ["date", "campaign_id", "campaign_name"],
            "date_range": "yesterday"
          }
        },
        {
          "step": 2,
          "app": "Supermetrics",
          "action": "Run Query",
          "config": {
            "data_source": "Facebook Ads",
            "metrics": ["impressions", "clicks", "spend", "purchases", "purchase_value"],
            "dimensions": ["date_start", "campaign_id", "campaign_name"],
            "date_range": "yesterday"
          }
        },
        {
          "step": 3,
          "app": "Supermetrics",
          "action": "Run Query",
          "config": {
            "data_source": "TikTok Ads",
            "metrics": ["impressions", "clicks", "spend", "conversions", "conversion_value"],
            "dimensions": ["date", "campaign_id", "campaign_name"],
            "date_range": "yesterday"
          }
        },
        {
          "step": 4,
          "app": "Supermetrics",
          "action": "Run Query",
          "config": {
            "data_source": "LinkedIn Ads",
            "metrics": ["impressions", "clicks", "cost", "conversions", "conversion_value"],
            "dimensions": ["date", "campaign_id", "campaign_name"],
            "date_range": "yesterday"
          }
        },
        {
          "step": 5,
          "app": "Formatter by Zapier",
          "action": "Utilities",
          "config": {
            "operation": "Merge Arrays",
            "arrays": ["{{step1_rows}}", "{{step2_rows}}", "{{step3_rows}}", "{{step4_rows}}"],
            "add_platform_column": true
          }
        },
        {
          "step": 6,
          "app": "Google Sheets",
          "action": "Create Spreadsheet Row(s)",
          "config": {
            "spreadsheet_id": "{{MARKETING_DASHBOARD_SHEET}}",
            "worksheet": "Daily_Metrics",
            "rows": "{{step5_merged}}"
          }
        }
      ]
    },
    {
      "name": "Weekly Cross-Platform Analysis",
      "trigger": {
        "app": "Schedule by Zapier",
        "event": "Every Week",
        "config": {
          "day_of_week": "Monday",
          "time": "09:00"
        }
      },
      "actions": [
        {
          "step": 1,
          "app": "Google Sheets",
          "action": "Get Many Spreadsheet Rows",
          "config": {
            "spreadsheet_id": "{{MARKETING_DASHBOARD_SHEET}}",
            "worksheet": "Daily_Metrics",
            "filter_formula": "=date>=TODAY()-7"
          }
        },
        {
          "step": 2,
          "app": "Formatter by Zapier",
          "action": "Text Transform",
          "config": {
            "operation": "Convert to CSV",
            "input": "{{step1_rows}}"
          }
        },
        {
          "step": 3,
          "app": "OpenAI (ChatGPT)",
          "action": "Send Prompt",
          "config": {
            "model": "gpt-4-turbo",
            "system_message": "You are a cross-platform marketing analyst...",
            "user_message": "Analyze this week's cross-platform marketing data and provide performance insights:\\n\\n{{step2_csv}}\\n\\nTotal spend: USD {{step1_total_spend}}\\nDate range: {{last_7_days}}",
            "temperature": 0.3,
            "max_tokens": 2500
          }
        },
        {
          "step": 4,
          "app": "Slack",
          "action": "Send Channel Message",
          "config": {
            "channel": "#marketing-performance",
            "message_format": "blocks",
            "blocks": [
              {
                "type": "header",
                "text": "Weekly Cross-Platform Performance Report"
              },
              {
                "type": "section",
                "text": "{{step3_response}}"
              },
              {
                "type": "actions",
                "elements": [
                  {
                    "type": "button",
                    "text": "View Full Dashboard",
                    "url": "{{MARKETING_DASHBOARD_URL}}"
                  }
                ]
              }
            ]
          }
        },
        {
          "step": 5,
          "app": "Gmail",
          "action": "Send Email",
          "config": {
            "to": "marketing-leadership@company.com",
            "subject": "Weekly Marketing Performance - {{current_date}}",
            "body_type": "html",
            "body": "<h1>Cross-Platform Marketing Analysis</h1>{{step3_response_html}}<p><a href='{{MARKETING_DASHBOARD_URL}}'>View Interactive Dashboard</a></p>"
          }
        }
      ],
      "error_handling": {
        "on_error": "notify",
        "notification_channel": "#zapier-alerts"
      }
    },
    {
      "name": "Spend Anomaly Alert",
      "trigger": {
        "app": "Schedule by Zapier",
        "event": "Every Hour",
        "config": {
          "start_time": "08:00",
          "end_time": "20:00"
        }
      },
      "actions": [
        {
          "step": 1,
          "app": "Google Sheets",
          "action": "Lookup Spreadsheet Row",
          "config": {
            "spreadsheet_id": "{{MARKETING_DASHBOARD_SHEET}}",
            "worksheet": "Hourly_Pacing",
            "lookup": "Check if any platform >20% over daily pacing"
          }
        },
        {
          "step": 2,
          "app": "Filter by Zapier",
          "action": "Only Continue If",
          "config": {
            "condition": "{{step1_overpacing_platforms}} is not empty"
          }
        },
        {
          "step": 3,
          "app": "Slack",
          "action": "Send Channel Message",
          "config": {
            "channel": "#marketing-alerts",
            "message": "Spend pacing alert: {{step1_overpacing_platforms}} exceeding daily budget by >20%. Current spend: {{step1_current_spend}}. Review immediately."
          }
        }
      ]
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
          'Deploy a multi-agent system using CrewAI and LangGraph to continuously ingest data from all marketing platforms, unify metrics in real-time, perform automated cross-channel optimization, and execute budget changes autonomously based on performance signals.',
        estimatedMonthlyCost: '$700 - $1,500/month',
        architecture:
          'A Platform Orchestrator agent coordinates four specialist agents: Data Unification Agent normalizes metrics from all platforms, Performance Analyst Agent identifies optimization opportunities, Budget Optimizer Agent generates reallocation recommendations, and Execution Agent implements approved changes via platform APIs.',
        agents: [
          {
            name: 'Data Unification Agent',
            role: 'Cross-Platform Data Integration Specialist',
            goal: 'Continuously ingest and normalize campaign data from Google Ads, Meta, TikTok, and LinkedIn into a unified schema with standardized metrics and deduplication',
            tools: ['Google Ads API', 'Meta Marketing API', 'TikTok Ads API', 'LinkedIn API', 'BigQuery Connector'],
          },
          {
            name: 'Performance Analyst',
            role: 'Cross-Channel Performance Specialist',
            goal: 'Analyze unified marketing data to identify performance trends, detect anomalies, calculate true ROAS with conversion deduplication, and surface optimization opportunities',
            tools: ['Statistical Analysis Engine', 'Anomaly Detector', 'Conversion Deduplicator', 'Trend Analyzer'],
          },
          {
            name: 'Budget Optimizer',
            role: 'Cross-Platform Budget Allocation Specialist',
            goal: 'Generate optimal budget allocations across platforms based on true performance data, respecting business constraints and pacing requirements',
            tools: ['Optimization Solver', 'Constraint Engine', 'Pacing Calculator', 'Scenario Simulator'],
          },
          {
            name: 'Campaign Executor',
            role: 'Multi-Platform Campaign Management Specialist',
            goal: 'Execute approved budget changes and campaign adjustments across all platforms safely, with rollback capability and performance monitoring',
            tools: ['Google Ads API', 'Meta Marketing API', 'TikTok Ads API', 'LinkedIn API', 'Rollback Controller'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with hourly checkpointing and 90-day performance history in BigQuery',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the cross-platform marketing multi-agent system with CrewAI, establishing specialized roles for data unification, performance analysis, budget optimization, and campaign execution.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'CrewAI Cross-Platform Marketing Agent Definitions',
                description:
                  'Production-ready agent definitions for the unified marketing data platform.',
                code: `from crewai import Agent, Crew, Task, Process
from crewai.tools import BaseTool
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class MarketingAgentConfig(BaseModel):
    """Configuration for cross-platform marketing agents."""
    llm_model: str = Field(default="gpt-4-turbo")
    temperature: float = Field(default=0.1)
    max_iterations: int = Field(default=12)
    verbose: bool = Field(default=True)


def create_data_unification_agent(
    config: MarketingAgentConfig,
    tools: List[BaseTool],
) -> Agent:
    """Create the cross-platform data integration agent."""
    return Agent(
        role="Data Unification Agent",
        goal="""Continuously ingest campaign performance data from Google Ads,
        Meta, TikTok, and LinkedIn. Normalize metrics to a unified schema,
        standardize campaign naming, and deduplicate conversions against
        server-side event data.""",
        backstory="""You are a data integration specialist who has built
        pipelines for the largest marketing data warehouses. You understand
        the quirks of each platform's API—Google's micros, Meta's breakdown
        limits, TikTok's timezone handling. You ensure data quality and
        freshness while gracefully handling API rate limits and failures.""",
        tools=tools,
        llm=config.llm_model,
        verbose=config.verbose,
        allow_delegation=False,
        max_iter=config.max_iterations,
    )


def create_performance_analyst_agent(
    config: MarketingAgentConfig,
    tools: List[BaseTool],
) -> Agent:
    """Create the cross-channel performance analysis agent."""
    return Agent(
        role="Performance Analyst",
        goal="""Analyze unified marketing data to identify performance trends,
        detect spend anomalies, calculate true ROAS using deduplicated
        conversions, and surface actionable optimization opportunities.""",
        backstory="""You are a performance marketing expert who has managed
        $100M+ in annual ad spend across channels. You know that platform-
        reported ROAS is inflated by 30-50% due to conversion overcounting.
        You focus on incremental value, not vanity metrics. You detect
        patterns that humans miss and quantify opportunities precisely.""",
        tools=tools,
        llm=config.llm_model,
        verbose=config.verbose,
        allow_delegation=True,
        max_iter=config.max_iterations,
    )


def create_budget_optimizer_agent(
    config: MarketingAgentConfig,
    tools: List[BaseTool],
) -> Agent:
    """Create the cross-platform budget allocation agent."""
    return Agent(
        role="Budget Optimizer",
        goal="""Generate optimal budget allocations across all marketing
        platforms. Maximize blended ROAS while respecting minimum spend
        requirements, pacing constraints, and maximum change thresholds.
        Balance short-term performance with strategic channel presence.""",
        backstory="""You are a marketing operations strategist who combines
        quantitative optimization with practical business sense. You know
        that mathematical optima aren't always implementable—you factor in
        platform learning phases, seasonality, competitive dynamics, and
        organizational change capacity. You recommend changes that are
        both impactful and executable.""",
        tools=tools,
        llm=config.llm_model,
        verbose=config.verbose,
        allow_delegation=True,
        max_iter=config.max_iterations,
    )


def create_campaign_executor_agent(
    config: MarketingAgentConfig,
    tools: List[BaseTool],
) -> Agent:
    """Create the multi-platform campaign management agent."""
    return Agent(
        role="Campaign Executor",
        goal="""Execute approved budget changes and campaign adjustments
        across Google Ads, Meta, TikTok, and LinkedIn. Implement changes
        incrementally, monitor for performance degradation, and rollback
        automatically if metrics breach safety thresholds.""",
        backstory="""You are a campaign operations expert who has executed
        thousands of optimizations across platforms. You know that execution
        is as important as strategy—a poorly timed budget increase can
        waste thousands before the algorithm adjusts. You implement changes
        in controlled increments, validate at each step, and maintain
        detailed audit trails for every action.""",
        tools=tools,
        llm=config.llm_model,
        verbose=config.verbose,
        allow_delegation=False,
        max_iter=config.max_iterations,
    )


def create_marketing_platform_crew(
    config: MarketingAgentConfig,
    unification_tools: List[BaseTool],
    analysis_tools: List[BaseTool],
    optimization_tools: List[BaseTool],
    execution_tools: List[BaseTool],
) -> Crew:
    """Assemble the full cross-platform marketing crew."""
    unifier = create_data_unification_agent(config, unification_tools)
    analyst = create_performance_analyst_agent(config, analysis_tools)
    optimizer = create_budget_optimizer_agent(config, optimization_tools)
    executor = create_campaign_executor_agent(config, execution_tools)

    return Crew(
        agents=[unifier, analyst, optimizer, executor],
        process=Process.hierarchical,
        manager_llm=config.llm_model,
        verbose=config.verbose,
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the data unification agent with tools for connecting to all major ad platform APIs, normalizing metrics to a unified schema, and deduplicating conversions.',
            toolsUsed: ['Google Ads API', 'Meta Marketing API', 'TikTok Ads API', 'LinkedIn API'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Multi-Platform Data Ingestion Tools',
                description:
                  'Custom tools for unified data extraction from all marketing platforms.',
                code: `from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Protocol
from datetime import date, timedelta
from abc import abstractmethod
import hashlib
import re
import logging

logger = logging.getLogger(__name__)


class UnifiedMetric(BaseModel):
    """Standardized metric schema across all platforms."""
    date: date
    platform: str
    campaign_id: str
    campaign_name: str
    normalized_name: str
    channel_group: str
    impressions: int
    clicks: int
    spend: float
    platform_conversions: int
    platform_revenue: float
    verified_conversions: Optional[int] = None
    verified_revenue: Optional[float] = None


class PlatformConnector(Protocol):
    """Protocol for platform-specific connectors."""

    @abstractmethod
    def fetch_metrics(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def normalize_to_unified(self, raw_data: Dict[str, Any]) -> UnifiedMetric:
        ...


class GoogleAdsConnectorTool(BaseTool):
    """Tool to fetch and normalize Google Ads data."""

    name: str = "google_ads_connector"
    description: str = """Fetches campaign metrics from Google Ads API and
    normalizes to the unified marketing schema."""

    def __init__(self, client, customer_id: str):
        super().__init__()
        self._client = client
        self._customer_id = customer_id

    def _run(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch Google Ads metrics for the date range."""
        query = f"""
            SELECT
                segments.date,
                campaign.id,
                campaign.name,
                campaign.advertising_channel_type,
                metrics.impressions,
                metrics.clicks,
                metrics.cost_micros,
                metrics.conversions,
                metrics.conversions_value
            FROM campaign
            WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
                AND campaign.status = 'ENABLED'
        """

        try:
            response = self._client.search(
                customer_id=self._customer_id,
                query=query,
            )

            metrics = []
            for row in response:
                metrics.append(UnifiedMetric(
                    date=row.segments.date,
                    platform="google_ads",
                    campaign_id=str(row.campaign.id),
                    campaign_name=row.campaign.name,
                    normalized_name=self._normalize_name(row.campaign.name),
                    channel_group=self._map_channel(row.campaign.advertising_channel_type),
                    impressions=row.metrics.impressions,
                    clicks=row.metrics.clicks,
                    spend=row.metrics.cost_micros / 1_000_000,
                    platform_conversions=int(row.metrics.conversions),
                    platform_revenue=float(row.metrics.conversions_value),
                ).dict())

            logger.info(f"Google Ads: fetched {len(metrics)} campaign-days")
            return {"status": "success", "platform": "google_ads", "metrics": metrics}

        except Exception as e:
            logger.error(f"Google Ads fetch failed: {e}")
            return {"status": "error", "platform": "google_ads", "message": str(e)}

    def _normalize_name(self, name: str) -> str:
        """Standardize campaign name for cross-platform grouping."""
        normalized = name.lower().strip()
        normalized = re.sub(r'[^a-z0-9_\\-]', '_', normalized)
        normalized = re.sub(r'_+', '_', normalized)
        return normalized.strip('_')

    def _map_channel(self, channel_type: str) -> str:
        """Map Google Ads channel type to unified channel group."""
        mapping = {
            "SEARCH": "paid_search",
            "DISPLAY": "display",
            "VIDEO": "video",
            "SHOPPING": "shopping",
            "PERFORMANCE_MAX": "performance_max",
        }
        return mapping.get(channel_type, "other")


class MetaAdsConnectorTool(BaseTool):
    """Tool to fetch and normalize Meta (Facebook/Instagram) Ads data."""

    name: str = "meta_ads_connector"
    description: str = """Fetches campaign metrics from Meta Marketing API and
    normalizes to the unified marketing schema."""

    def __init__(self, access_token: str, ad_account_id: str):
        super().__init__()
        self._token = access_token
        self._account_id = ad_account_id

    def _run(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Fetch Meta Ads metrics for the date range."""
        import requests

        url = f"https://graph.facebook.com/v18.0/{self._account_id}/insights"
        params = {
            "access_token": self._token,
            "level": "campaign",
            "fields": "campaign_id,campaign_name,impressions,clicks,spend,actions,action_values",
            "time_range": f'{{"since":"{start_date}","until":"{end_date}"}}',
            "time_increment": 1,
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            metrics = []
            for row in data.get("data", []):
                conversions = self._extract_conversions(row.get("actions", []))
                revenue = self._extract_revenue(row.get("action_values", []))

                metrics.append(UnifiedMetric(
                    date=row["date_start"],
                    platform="meta",
                    campaign_id=row["campaign_id"],
                    campaign_name=row["campaign_name"],
                    normalized_name=self._normalize_name(row["campaign_name"]),
                    channel_group="paid_social",
                    impressions=int(row.get("impressions", 0)),
                    clicks=int(row.get("clicks", 0)),
                    spend=float(row.get("spend", 0)),
                    platform_conversions=conversions,
                    platform_revenue=revenue,
                ).dict())

            logger.info(f"Meta Ads: fetched {len(metrics)} campaign-days")
            return {"status": "success", "platform": "meta", "metrics": metrics}

        except Exception as e:
            logger.error(f"Meta Ads fetch failed: {e}")
            return {"status": "error", "platform": "meta", "message": str(e)}

    def _extract_conversions(self, actions: List[Dict]) -> int:
        """Extract purchase conversions from Meta actions array."""
        for action in actions:
            if action.get("action_type") == "purchase":
                return int(action.get("value", 0))
        return 0

    def _extract_revenue(self, action_values: List[Dict]) -> float:
        """Extract purchase revenue from Meta action_values array."""
        for av in action_values:
            if av.get("action_type") == "purchase":
                return float(av.get("value", 0))
        return 0.0

    def _normalize_name(self, name: str) -> str:
        """Standardize campaign name."""
        normalized = name.lower().strip()
        normalized = re.sub(r'[^a-z0-9_\\-]', '_', normalized)
        return re.sub(r'_+', '_', normalized).strip('_')


class ConversionDeduplicatorTool(BaseTool):
    """Tool to deduplicate conversions across platforms."""

    name: str = "conversion_deduplicator"
    description: str = """Deduplicates platform-reported conversions against
    server-side event data to calculate true conversion counts."""

    def __init__(self, db_conn):
        super().__init__()
        self._db = db_conn

    def _run(
        self,
        platform_metrics: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """Deduplicate conversions and enrich with verified counts."""
        # Fetch server-side conversions
        server_conversions = self._db.execute(
            """
            SELECT
                DATE(conversion_time) AS date,
                attributed_platform,
                attributed_campaign_id,
                COUNT(*) AS conversions,
                SUM(revenue) AS revenue
            FROM server_conversions
            WHERE conversion_time BETWEEN %s AND %s
            GROUP BY DATE(conversion_time), attributed_platform, attributed_campaign_id
            """,
            (start_date, end_date),
        ).fetchall()

        # Build lookup dictionary
        server_lookup = {}
        for row in server_conversions:
            key = (str(row["date"]), row["attributed_platform"], row["attributed_campaign_id"])
            server_lookup[key] = {
                "conversions": row["conversions"],
                "revenue": float(row["revenue"]),
            }

        # Enrich platform metrics with verified conversions
        enriched = []
        total_platform_conv = 0
        total_verified_conv = 0

        for metric in platform_metrics:
            key = (metric["date"], metric["platform"], metric["campaign_id"])
            verified = server_lookup.get(key, {"conversions": 0, "revenue": 0.0})

            metric["verified_conversions"] = verified["conversions"]
            metric["verified_revenue"] = verified["revenue"]
            enriched.append(metric)

            total_platform_conv += metric["platform_conversions"]
            total_verified_conv += verified["conversions"]

        overcount_pct = (
            (total_platform_conv - total_verified_conv) / total_platform_conv * 100
            if total_platform_conv > 0 else 0
        )

        return {
            "status": "success",
            "enriched_metrics": enriched,
            "total_platform_conversions": total_platform_conv,
            "total_verified_conversions": total_verified_conv,
            "overcount_percentage": round(overcount_pct, 1),
        }`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the performance analyst and budget optimizer agents with tools for cross-platform comparison, anomaly detection, and constrained optimization.',
            toolsUsed: ['Statistical Analysis Engine', 'Optimization Solver', 'Anomaly Detector'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Cross-Platform Analysis and Optimization Tools',
                description:
                  'Tools for performance analysis and budget optimization across platforms.',
                code: `from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Tuple
from datetime import date, timedelta
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalysisTool(BaseTool):
    """Tool for cross-platform performance analysis."""

    name: str = "performance_analyzer"
    description: str = """Analyzes unified marketing metrics to identify
    performance trends, platform efficiency, and optimization opportunities."""

    def _run(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cross-platform performance."""
        if not metrics:
            return {"status": "error", "message": "No metrics provided"}

        # Aggregate by platform
        platform_stats = {}
        for metric in metrics:
            platform = metric["platform"]
            if platform not in platform_stats:
                platform_stats[platform] = {
                    "spend": 0,
                    "platform_conversions": 0,
                    "verified_conversions": 0,
                    "platform_revenue": 0,
                    "verified_revenue": 0,
                }
            platform_stats[platform]["spend"] += metric["spend"]
            platform_stats[platform]["platform_conversions"] += metric["platform_conversions"]
            platform_stats[platform]["verified_conversions"] += metric.get("verified_conversions", 0)
            platform_stats[platform]["platform_revenue"] += metric["platform_revenue"]
            platform_stats[platform]["verified_revenue"] += metric.get("verified_revenue", 0)

        # Calculate metrics
        total_spend = sum(p["spend"] for p in platform_stats.values())
        total_verified_conv = sum(p["verified_conversions"] for p in platform_stats.values())

        analysis = {}
        for platform, data in platform_stats.items():
            spend_share = data["spend"] / total_spend if total_spend > 0 else 0
            conv_share = data["verified_conversions"] / total_verified_conv if total_verified_conv > 0 else 0
            efficiency_index = conv_share / spend_share if spend_share > 0 else 0

            platform_roas = data["platform_revenue"] / data["spend"] if data["spend"] > 0 else 0
            true_roas = data["verified_revenue"] / data["spend"] if data["spend"] > 0 else 0
            overcount_pct = (
                (data["platform_conversions"] - data["verified_conversions"])
                / data["platform_conversions"] * 100
                if data["platform_conversions"] > 0 else 0
            )

            analysis[platform] = {
                "spend": round(data["spend"], 2),
                "spend_share_pct": round(spend_share * 100, 1),
                "verified_conversions": data["verified_conversions"],
                "conversion_share_pct": round(conv_share * 100, 1),
                "efficiency_index": round(efficiency_index, 2),
                "platform_roas": round(platform_roas, 2),
                "true_roas": round(true_roas, 2),
                "overcount_pct": round(overcount_pct, 1),
                "true_cpa": round(data["spend"] / data["verified_conversions"], 2) if data["verified_conversions"] > 0 else None,
            }

        # Identify opportunities
        opportunities = []
        for platform, data in analysis.items():
            if data["efficiency_index"] > 1.2 and data["spend_share_pct"] < 25:
                opportunities.append({
                    "type": "underinvested",
                    "platform": platform,
                    "efficiency_index": data["efficiency_index"],
                    "recommendation": f"Increase {platform} spend by 15-25%",
                })
            elif data["efficiency_index"] < 0.8 and data["spend_share_pct"] > 20:
                opportunities.append({
                    "type": "overinvested",
                    "platform": platform,
                    "efficiency_index": data["efficiency_index"],
                    "recommendation": f"Reduce {platform} spend by 10-20%",
                })
            if data["overcount_pct"] > 40:
                opportunities.append({
                    "type": "tracking_issue",
                    "platform": platform,
                    "overcount_pct": data["overcount_pct"],
                    "recommendation": f"Review {platform} conversion tracking setup",
                })

        return {
            "status": "success",
            "platform_analysis": analysis,
            "total_spend": round(total_spend, 2),
            "total_verified_conversions": total_verified_conv,
            "blended_true_roas": round(
                sum(p["verified_revenue"] for p in platform_stats.values()) / total_spend, 2
            ) if total_spend > 0 else 0,
            "opportunities": opportunities,
        }


class AnomalyDetectorTool(BaseTool):
    """Tool to detect spend and performance anomalies."""

    name: str = "anomaly_detector"
    description: str = """Detects anomalies in marketing metrics using
    statistical methods. Flags unusual spend patterns, CTR drops, and CPA spikes."""

    def _run(
        self,
        current_metrics: List[Dict[str, Any]],
        historical_metrics: List[Dict[str, Any]],
        threshold_z: float = 2.5,
    ) -> Dict[str, Any]:
        """Detect anomalies comparing current to historical data."""
        anomalies = []

        # Group by platform
        current_by_platform = self._aggregate_by_platform(current_metrics)
        historical_by_platform = self._aggregate_by_platform(historical_metrics)

        for platform, current in current_by_platform.items():
            historical = historical_by_platform.get(platform, {})
            if not historical:
                continue

            # Check spend anomaly
            if "daily_spends" in historical:
                spend_mean = np.mean(historical["daily_spends"])
                spend_std = np.std(historical["daily_spends"])
                if spend_std > 0:
                    z_score = (current["daily_spend"] - spend_mean) / spend_std
                    if abs(z_score) > threshold_z:
                        anomalies.append({
                            "type": "spend_anomaly",
                            "platform": platform,
                            "metric": "daily_spend",
                            "current_value": current["daily_spend"],
                            "expected_value": round(spend_mean, 2),
                            "z_score": round(z_score, 2),
                            "direction": "high" if z_score > 0 else "low",
                        })

            # Check CTR anomaly
            if "ctrs" in historical and len(historical["ctrs"]) > 5:
                ctr_mean = np.mean(historical["ctrs"])
                ctr_std = np.std(historical["ctrs"])
                if ctr_std > 0 and current.get("ctr"):
                    z_score = (current["ctr"] - ctr_mean) / ctr_std
                    if z_score < -threshold_z:  # Only flag drops
                        anomalies.append({
                            "type": "ctr_drop",
                            "platform": platform,
                            "current_ctr": round(current["ctr"] * 100, 2),
                            "expected_ctr": round(ctr_mean * 100, 2),
                            "z_score": round(z_score, 2),
                        })

            # Check CPA anomaly
            if "cpas" in historical and len(historical["cpas"]) > 5:
                cpa_mean = np.mean(historical["cpas"])
                cpa_std = np.std(historical["cpas"])
                if cpa_std > 0 and current.get("cpa"):
                    z_score = (current["cpa"] - cpa_mean) / cpa_std
                    if z_score > threshold_z:  # Only flag increases
                        anomalies.append({
                            "type": "cpa_spike",
                            "platform": platform,
                            "current_cpa": round(current["cpa"], 2),
                            "expected_cpa": round(cpa_mean, 2),
                            "z_score": round(z_score, 2),
                        })

        return {
            "status": "success",
            "anomalies_detected": len(anomalies),
            "anomalies": anomalies,
            "threshold_z_score": threshold_z,
        }

    def _aggregate_by_platform(
        self, metrics: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics by platform for comparison."""
        result = {}
        for metric in metrics:
            platform = metric["platform"]
            if platform not in result:
                result[platform] = {
                    "daily_spends": [],
                    "ctrs": [],
                    "cpas": [],
                    "daily_spend": 0,
                }
            result[platform]["daily_spends"].append(metric["spend"])
            if metric["impressions"] > 0:
                result[platform]["ctrs"].append(metric["clicks"] / metric["impressions"])
            if metric.get("verified_conversions", 0) > 0:
                result[platform]["cpas"].append(metric["spend"] / metric["verified_conversions"])

        # Calculate current period aggregates
        for platform, data in result.items():
            data["daily_spend"] = sum(data["daily_spends"]) / len(data["daily_spends"]) if data["daily_spends"] else 0
            data["ctr"] = np.mean(data["ctrs"]) if data["ctrs"] else None
            data["cpa"] = np.mean(data["cpas"]) if data["cpas"] else None

        return result


class BudgetOptimizationTool(BaseTool):
    """Tool to generate optimal budget allocations."""

    name: str = "budget_optimizer"
    description: str = """Generates optimal budget allocation across platforms
    based on true ROAS, subject to business constraints."""

    def _run(
        self,
        platform_analysis: Dict[str, Dict[str, Any]],
        total_budget: float,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Optimize budget allocation across platforms."""
        constraints = constraints or {}
        min_pct = constraints.get("min_platform_pct", 0.10)
        max_pct = constraints.get("max_platform_pct", 0.45)
        max_change = constraints.get("max_change_pct", 0.25)

        platforms = list(platform_analysis.keys())

        # Calculate efficiency scores (true ROAS normalized)
        roas_values = [platform_analysis[p]["true_roas"] for p in platforms]
        total_roas = sum(roas_values)
        efficiency_weights = {
            p: platform_analysis[p]["true_roas"] / total_roas if total_roas > 0 else 1 / len(platforms)
            for p in platforms
        }

        # Current allocation
        total_current = sum(platform_analysis[p]["spend"] for p in platforms)
        current_allocation = {
            p: platform_analysis[p]["spend"] / total_current if total_current > 0 else 1 / len(platforms)
            for p in platforms
        }

        # Calculate optimal allocation with constraints
        optimal_allocation = {}
        for platform in platforms:
            # Start with efficiency-weighted allocation
            raw_pct = efficiency_weights[platform]

            # Apply min/max constraints
            constrained_pct = max(min_pct, min(max_pct, raw_pct))

            # Apply change rate constraint
            current_pct = current_allocation[platform]
            max_new_pct = current_pct * (1 + max_change)
            min_new_pct = current_pct * (1 - max_change)
            final_pct = max(min_new_pct, min(max_new_pct, constrained_pct))

            optimal_allocation[platform] = final_pct

        # Normalize to sum to 1
        total_pct = sum(optimal_allocation.values())
        optimal_allocation = {p: v / total_pct for p, v in optimal_allocation.items()}

        # Generate recommendations
        recommendations = {}
        for platform in platforms:
            current_spend = platform_analysis[platform]["spend"]
            current_pct = current_allocation[platform]
            new_pct = optimal_allocation[platform]
            new_spend = total_budget * new_pct
            change = new_spend - (total_budget * current_pct)
            change_pct = (new_pct - current_pct) / current_pct * 100 if current_pct > 0 else 0

            recommendations[platform] = {
                "current_spend": round(current_spend, 2),
                "current_pct": round(current_pct * 100, 1),
                "recommended_spend": round(new_spend, 2),
                "recommended_pct": round(new_pct * 100, 1),
                "change_amount": round(change, 2),
                "change_pct": round(change_pct, 1),
                "action": "increase" if change > 0 else "decrease" if change < 0 else "maintain",
            }

        # Project impact
        projected_conversions = sum(
            recommendations[p]["recommended_spend"] / platform_analysis[p]["true_cpa"]
            for p in platforms
            if platform_analysis[p].get("true_cpa")
        )

        return {
            "status": "success",
            "total_budget": total_budget,
            "recommendations": recommendations,
            "projected_conversions": round(projected_conversions, 0),
            "projected_blended_roas": round(
                sum(recommendations[p]["recommended_spend"] * platform_analysis[p]["true_roas"]
                    for p in platforms) / total_budget, 2
            ),
            "constraints_applied": constraints,
        }`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement the LangGraph state machine that coordinates daily data ingestion, continuous analysis, budget optimization cycles, and automated campaign execution with approval workflows.',
            toolsUsed: ['LangGraph', 'Redis', 'BigQuery'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Cross-Platform Marketing Workflow',
                description:
                  'State machine implementation for coordinating the multi-agent marketing platform.',
                code: `from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, List, Dict, Any, Literal
from datetime import datetime, timezone, date, timedelta
import operator
import logging
import json

logger = logging.getLogger(__name__)


class MarketingPlatformState(TypedDict):
    """State schema for the cross-platform marketing workflow."""
    # Workflow metadata
    run_id: str
    workflow_type: Literal["daily_sync", "weekly_optimization", "anomaly_response"]
    started_at: str
    status: Literal["running", "completed", "failed", "awaiting_approval"]

    # Data ingestion state
    platforms_synced: List[str]
    raw_metrics: List[Dict[str, Any]]
    unified_metrics: List[Dict[str, Any]]
    sync_errors: List[Dict[str, Any]]

    # Analysis state
    platform_analysis: Dict[str, Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    opportunities: List[Dict[str, Any]]

    # Optimization state
    current_budget: float
    budget_recommendations: Dict[str, Dict[str, Any]]
    projected_impact: Dict[str, Any]

    # Execution state
    approval_status: Literal["pending", "approved", "rejected"]
    execution_results: Dict[str, Any]
    rollback_checkpoints: List[Dict[str, Any]]

    # Audit trail
    messages: Annotated[List[str], operator.add]
    errors: List[str]


def create_daily_sync_workflow(
    data_unification_agent,
    performance_analyst_agent,
    platforms: List[str] = None,
) -> StateGraph:
    """Create workflow for daily platform data synchronization."""

    platforms = platforms or ["google_ads", "meta", "tiktok", "linkedin"]
    workflow = StateGraph(MarketingPlatformState)

    def ingest_platform_data(state: MarketingPlatformState) -> MarketingPlatformState:
        """Ingest data from all marketing platforms."""
        logger.info("Starting multi-platform data ingestion...")
        state["platforms_synced"] = []
        state["raw_metrics"] = []
        state["sync_errors"] = []

        for platform in platforms:
            try:
                result = data_unification_agent.execute_task(
                    f"Fetch yesterday's campaign metrics from {platform}"
                )
                if result.get("status") == "success":
                    state["platforms_synced"].append(platform)
                    state["raw_metrics"].extend(result.get("metrics", []))
                else:
                    state["sync_errors"].append({
                        "platform": platform,
                        "error": result.get("message", "Unknown error"),
                    })
            except Exception as e:
                state["sync_errors"].append({"platform": platform, "error": str(e)})

        state["messages"] = [
            f"Synced {len(state['platforms_synced'])}/{len(platforms)} platforms, "
            f"{len(state['raw_metrics'])} campaign-days ingested"
        ]
        return state

    def deduplicate_conversions(state: MarketingPlatformState) -> MarketingPlatformState:
        """Deduplicate conversions against server-side data."""
        logger.info("Deduplicating conversions...")
        try:
            result = data_unification_agent.execute_task(
                f"Deduplicate {len(state['raw_metrics'])} platform metrics against server-side conversions"
            )
            state["unified_metrics"] = result.get("enriched_metrics", [])
            overcount = result.get("overcount_percentage", 0)
            state["messages"] = [f"Conversion deduplication complete. Overcount: {overcount}%"]
        except Exception as e:
            state["errors"] = [f"Deduplication failed: {str(e)}"]
            state["unified_metrics"] = state["raw_metrics"]
        return state

    def analyze_performance(state: MarketingPlatformState) -> MarketingPlatformState:
        """Analyze cross-platform performance."""
        logger.info("Analyzing cross-platform performance...")
        try:
            result = performance_analyst_agent.execute_task(
                f"Analyze {len(state['unified_metrics'])} unified metrics for performance insights"
            )
            state["platform_analysis"] = result.get("platform_analysis", {})
            state["opportunities"] = result.get("opportunities", [])
            state["messages"] = [
                f"Performance analysis complete. Found {len(state['opportunities'])} opportunities"
            ]
        except Exception as e:
            state["errors"] = [f"Performance analysis failed: {str(e)}"]
        return state

    def detect_anomalies(state: MarketingPlatformState) -> MarketingPlatformState:
        """Detect performance anomalies."""
        logger.info("Running anomaly detection...")
        try:
            result = performance_analyst_agent.execute_task(
                f"Detect anomalies in current metrics compared to 30-day baseline"
            )
            state["anomalies"] = result.get("anomalies", [])
            if state["anomalies"]:
                state["messages"] = [f"Detected {len(state['anomalies'])} anomalies requiring attention"]
            else:
                state["messages"] = ["No anomalies detected"]
        except Exception as e:
            state["errors"] = [f"Anomaly detection failed: {str(e)}"]
            state["anomalies"] = []
        return state

    def persist_results(state: MarketingPlatformState) -> MarketingPlatformState:
        """Persist unified metrics and analysis to data warehouse."""
        logger.info("Persisting results to BigQuery...")
        try:
            # In production, this would write to BigQuery
            state["messages"] = [
                f"Persisted {len(state['unified_metrics'])} metrics to unified_daily_metrics table"
            ]
            state["status"] = "completed"
        except Exception as e:
            state["errors"] = [f"Persistence failed: {str(e)}"]
            state["status"] = "failed"
        return state

    # Add nodes
    workflow.add_node("ingest", ingest_platform_data)
    workflow.add_node("deduplicate", deduplicate_conversions)
    workflow.add_node("analyze", analyze_performance)
    workflow.add_node("detect_anomalies", detect_anomalies)
    workflow.add_node("persist", persist_results)

    # Add edges
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "deduplicate")
    workflow.add_edge("deduplicate", "analyze")
    workflow.add_edge("analyze", "detect_anomalies")
    workflow.add_edge("detect_anomalies", "persist")
    workflow.add_edge("persist", END)

    return workflow.compile()


def create_optimization_workflow(
    performance_analyst_agent,
    budget_optimizer_agent,
    execution_agent,
    require_approval: bool = True,
) -> StateGraph:
    """Create workflow for budget optimization and execution."""

    workflow = StateGraph(MarketingPlatformState)

    def load_performance_data(state: MarketingPlatformState) -> MarketingPlatformState:
        """Load 30-day performance data for optimization."""
        logger.info("Loading performance data for optimization...")
        try:
            result = performance_analyst_agent.execute_task(
                "Load and analyze the last 30 days of unified marketing metrics"
            )
            state["platform_analysis"] = result.get("platform_analysis", {})
            state["current_budget"] = sum(
                p["spend"] for p in state["platform_analysis"].values()
            )
            state["messages"] = [f"Loaded \${state['current_budget']:,.0f} in spend across platforms"]
        except Exception as e:
            state["errors"] = [f"Data load failed: {str(e)}"]
            state["status"] = "failed"
        return state

    def generate_recommendations(state: MarketingPlatformState) -> MarketingPlatformState:
        """Generate budget optimization recommendations."""
        logger.info("Generating budget recommendations...")
        try:
            result = budget_optimizer_agent.execute_task(
                f"Generate optimal budget allocation for \${state['current_budget']:,.0f} "
                f"based on platform analysis: {json.dumps(state['platform_analysis'])}"
            )
            state["budget_recommendations"] = result.get("recommendations", {})
            state["projected_impact"] = {
                "conversions": result.get("projected_conversions", 0),
                "roas": result.get("projected_blended_roas", 0),
            }
            state["approval_status"] = "pending"
            state["messages"] = [
                f"Generated recommendations with projected ROAS: {state['projected_impact']['roas']}"
            ]
        except Exception as e:
            state["errors"] = [f"Optimization failed: {str(e)}"]
            state["status"] = "failed"
        return state

    def check_approval(state: MarketingPlatformState) -> MarketingPlatformState:
        """Check approval status for budget changes."""
        if state["approval_status"] == "approved":
            state["messages"] = ["Budget changes approved, proceeding to execution"]
        elif state["approval_status"] == "rejected":
            state["messages"] = ["Budget changes rejected"]
            state["status"] = "completed"
        else:
            state["status"] = "awaiting_approval"
            state["messages"] = ["Recommendations pending approval"]
        return state

    def execute_changes(state: MarketingPlatformState) -> MarketingPlatformState:
        """Execute approved budget changes across platforms."""
        logger.info("Executing budget changes...")
        try:
            result = execution_agent.execute_task(
                f"Execute budget changes: {json.dumps(state['budget_recommendations'])}"
            )
            state["execution_results"] = result
            state["status"] = "completed"
            state["messages"] = ["Budget changes executed successfully across all platforms"]
        except Exception as e:
            state["errors"] = [f"Execution failed: {str(e)}"]
            state["status"] = "failed"
        return state

    def route_after_approval(state: MarketingPlatformState) -> str:
        if state["approval_status"] == "approved":
            return "execute"
        elif state["approval_status"] == "rejected":
            return "end"
        return "wait"

    # Add nodes
    workflow.add_node("load_data", load_performance_data)
    workflow.add_node("optimize", generate_recommendations)
    workflow.add_node("check_approval", check_approval)
    workflow.add_node("execute", execute_changes)

    # Add edges
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "optimize")

    if require_approval:
        workflow.add_edge("optimize", "check_approval")
        workflow.add_conditional_edges(
            "check_approval",
            route_after_approval,
            {"execute": "execute", "end": END, "wait": END},
        )
    else:
        workflow.add_edge("optimize", "execute")

    workflow.add_edge("execute", END)

    return workflow.compile()`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the cross-platform marketing agent system with Docker, implement comprehensive monitoring for data freshness and execution accuracy, and create operational dashboards.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose for Cross-Platform Marketing System',
                description:
                  'Production deployment configuration for the multi-agent marketing platform.',
                code: `version: '3.8'

services:
  marketing-agents:
    build:
      context: .
      dockerfile: Dockerfile.marketing
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=marketing-platform
      - REDIS_URL=redis://redis:6379/2
      - BIGQUERY_PROJECT=\${GCP_PROJECT_ID}
      - BIGQUERY_DATASET=marketing_unified
      - GOOGLE_ADS_DEVELOPER_TOKEN=\${GOOGLE_ADS_DEVELOPER_TOKEN}
      - GOOGLE_ADS_CUSTOMER_ID=\${GOOGLE_ADS_CUSTOMER_ID}
      - META_ACCESS_TOKEN=\${META_ACCESS_TOKEN}
      - META_AD_ACCOUNT_ID=\${META_AD_ACCOUNT_ID}
      - TIKTOK_ACCESS_TOKEN=\${TIKTOK_ACCESS_TOKEN}
      - LINKEDIN_ACCESS_TOKEN=\${LINKEDIN_ACCESS_TOKEN}
      - SLACK_MARKETING_WEBHOOK=\${SLACK_MARKETING_WEBHOOK}
      - LOG_LEVEL=INFO
      - REQUIRE_BUDGET_APPROVAL=true
    depends_on:
      - redis
      - postgres
    ports:
      - "8080:8080"
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile.scheduler
    environment:
      - MARKETING_AGENTS_URL=http://marketing-agents:8080
    depends_on:
      - marketing-agents
    command: >
      sh -c "
        echo '0 6 * * * curl -X POST http://marketing-agents:8080/workflows/daily-sync' | crontab - &&
        echo '0 9 * * 1 curl -X POST http://marketing-agents:8080/workflows/weekly-optimization' | crontab - &&
        crond -f
      "

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=marketing
      - POSTGRES_PASSWORD=\${POSTGRES_PASSWORD}
      - POSTGRES_DB=marketing_ops
    volumes:
      - postgres_data:/var/lib/postgresql/data

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
      - postgres

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:`,
              },
              {
                language: 'python',
                title: 'Marketing Platform Observability Implementation',
                description:
                  'Comprehensive monitoring for data freshness, sync accuracy, and execution performance.',
                code: `import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import requests

logger = logging.getLogger(__name__)


# Prometheus metrics
PLATFORM_SYNC_RUNS = Counter(
    'marketing_platform_sync_total',
    'Total platform sync operations',
    ['platform', 'status']
)
PLATFORM_SYNC_DURATION = Histogram(
    'marketing_platform_sync_duration_seconds',
    'Platform sync duration',
    ['platform'],
    buckets=[5, 10, 30, 60, 120, 300, 600]
)
METRICS_INGESTED = Counter(
    'marketing_metrics_ingested_total',
    'Total metrics ingested',
    ['platform']
)
DATA_FRESHNESS_HOURS = Gauge(
    'marketing_data_freshness_hours',
    'Hours since last data update',
    ['platform']
)
CONVERSION_OVERCOUNT_PCT = Gauge(
    'marketing_conversion_overcount_pct',
    'Platform conversion overcounting percentage',
    ['platform']
)
BUDGET_CHANGE_EXECUTED = Counter(
    'marketing_budget_changes_total',
    'Budget changes executed',
    ['platform', 'direction']
)
OPTIMIZATION_PROJECTED_ROAS = Gauge(
    'marketing_optimization_projected_roas',
    'Projected ROAS from optimization'
)


@dataclass
class SyncHealthCheck:
    """Health check result for platform sync."""
    platform: str
    last_sync: datetime
    freshness_hours: float
    metrics_count: int
    status: str
    errors: List[str]


class MarketingPlatformMonitor:
    """Monitor for cross-platform marketing system health."""

    def __init__(
        self,
        db_conn,
        slack_webhook: Optional[str] = None,
        freshness_threshold_hours: float = 26,
        prometheus_port: int = 8000,
    ):
        self._db = db_conn
        self._slack_webhook = slack_webhook
        self._freshness_threshold = freshness_threshold_hours

        start_http_server(prometheus_port)
        logger.info(f"Prometheus metrics server started on port {prometheus_port}")

    def check_platform_health(self) -> List[SyncHealthCheck]:
        """Check sync health for all platforms."""
        platforms = ["google_ads", "meta", "tiktok", "linkedin"]
        results = []

        for platform in platforms:
            row = self._db.execute(
                """
                SELECT
                    MAX(metric_date) AS last_date,
                    COUNT(*) AS metrics_count,
                    MAX(inserted_at) AS last_sync
                FROM unified_daily_metrics
                WHERE platform = %s
                    AND metric_date >= CURRENT_DATE - INTERVAL '7 days'
                """,
                (platform,),
            ).fetchone()

            if row and row["last_sync"]:
                freshness = (datetime.now(timezone.utc) - row["last_sync"]).total_seconds() / 3600
                status = "healthy" if freshness < self._freshness_threshold else "stale"
                errors = [] if status == "healthy" else [f"Data is {freshness:.1f}h old"]
            else:
                freshness = 999
                status = "no_data"
                errors = ["No data found for platform"]

            DATA_FRESHNESS_HOURS.labels(platform=platform).set(freshness)

            results.append(SyncHealthCheck(
                platform=platform,
                last_sync=row["last_sync"] if row else None,
                freshness_hours=freshness,
                metrics_count=row["metrics_count"] if row else 0,
                status=status,
                errors=errors,
            ))

        return results

    def check_conversion_accuracy(self) -> Dict[str, float]:
        """Check conversion overcounting by platform."""
        rows = self._db.execute(
            """
            SELECT
                platform,
                SUM(platform_conversions) AS platform_conv,
                SUM(verified_conversions) AS verified_conv
            FROM unified_daily_metrics
            WHERE metric_date >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY platform
            """
        ).fetchall()

        results = {}
        for row in rows:
            overcount = (
                (row["platform_conv"] - row["verified_conv"]) / row["platform_conv"] * 100
                if row["platform_conv"] > 0 else 0
            )
            results[row["platform"]] = overcount
            CONVERSION_OVERCOUNT_PCT.labels(platform=row["platform"]).set(overcount)

        return results

    def record_sync_completion(
        self,
        platform: str,
        status: str,
        duration_seconds: float,
        metrics_count: int,
    ) -> None:
        """Record platform sync completion metrics."""
        PLATFORM_SYNC_RUNS.labels(platform=platform, status=status).inc()
        PLATFORM_SYNC_DURATION.labels(platform=platform).observe(duration_seconds)
        METRICS_INGESTED.labels(platform=platform).inc(metrics_count)
        logger.info(
            f"{platform} sync {status}: {metrics_count} metrics in {duration_seconds:.1f}s"
        )

    def record_budget_execution(
        self,
        recommendations: Dict[str, Dict[str, Any]],
    ) -> None:
        """Record budget change executions."""
        for platform, rec in recommendations.items():
            direction = rec.get("action", "maintain")
            BUDGET_CHANGE_EXECUTED.labels(platform=platform, direction=direction).inc()

    def alert_on_issues(self, health_checks: List[SyncHealthCheck]) -> None:
        """Send Slack alert for health issues."""
        issues = [h for h in health_checks if h.status != "healthy"]
        if not issues or not self._slack_webhook:
            return

        blocks = [
            f"*{h.platform}*: {h.status} ({h.freshness_hours:.1f}h old)"
            for h in issues
        ]
        payload = {
            "text": f"Marketing Platform Alert - {len(issues)} platform(s) unhealthy",
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": "\\n".join(blocks)}}
            ],
        }

        try:
            requests.post(self._slack_webhook, json=payload, timeout=10)
            logger.warning(f"Sent alert for {len(issues)} platform issues")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return summary."""
        platform_health = self.check_platform_health()
        conversion_accuracy = self.check_conversion_accuracy()

        self.alert_on_issues(platform_health)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform_health": [
                {
                    "platform": h.platform,
                    "status": h.status,
                    "freshness_hours": round(h.freshness_hours, 1),
                    "metrics_count": h.metrics_count,
                }
                for h in platform_health
            ],
            "conversion_accuracy": {
                platform: f"{overcount:.1f}% overcounting"
                for platform, overcount in conversion_accuracy.items()
            },
            "overall_status": (
                "healthy"
                if all(h.status == "healthy" for h in platform_health)
                else "degraded"
            ),
        }`,
              },
            ],
          },
        ],
      },
    },
  ],
};
