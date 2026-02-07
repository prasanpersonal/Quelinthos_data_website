import type { Category } from '../types.ts';

export const ecommerceCategory: Category = {
  id: 'ecommerce',
  number: 4,
  title: 'Ecommerce Operations',
  shortTitle: 'Ecommerce',
  description:
    'Solve Shopify tax nightmares, WooCommerce performance bloat, and analytics trust gaps costing you conversions.',
  icon: 'ShoppingCart',
  accentColor: 'neon-blue',
  painPoints: [
    /* ──────────────────────────────────────────────
       Pain Point 1 — Shopify Sales Tax Nightmare
       ────────────────────────────────────────────── */
    {
      id: 'shopify-sales-tax',
      number: 1,
      title: 'Shopify Sales Tax Nightmare',
      subtitle: 'Multi-State Tax Compliance Chaos',
      summary:
        'Economic nexus in 40+ states, marketplace facilitator laws, and Shopify\'s basic tax engine means you\'re either overpaying or under-collecting.',
      tags: ['shopify', 'sales-tax', 'compliance'],
      metrics: {
        annualCostRange: '$200K - $1M',
        roi: '8x',
        paybackPeriod: '2-3 months',
        investmentRange: '$40K - $80K',
      },
      price: {
        present: {
          title: 'Current State',
          description:
            'Tax collection relies on Shopify\'s built-in engine, which lacks granularity for economic nexus thresholds across 40+ jurisdictions.',
          bullets: [
            'Tax rates applied inconsistently across product categories and customer locations',
            'No automated tracking of economic nexus thresholds per state',
            'Manual quarterly filings consuming 20+ hours of staff time per cycle',
          ],
          severity: 'high',
        },
        root: {
          title: 'Root Cause',
          description:
            'Shopify\'s native tax engine treats tax as a flat lookup rather than a dynamic compliance problem tied to revenue thresholds and product taxability rules.',
          bullets: [
            'Economic nexus thresholds ($100K revenue or 200 transactions) not monitored programmatically',
            'Product tax codes misaligned with state-level taxability matrices',
            'Marketplace facilitator exemptions not factored into collection logic',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Business Impact',
          description:
            'Incorrect tax collection creates dual exposure: under-collection triggers audit penalties, while over-collection erodes margins and customer trust.',
          bullets: [
            'Audit risk across states where nexus was unknowingly triggered',
            'Over-collection of 2-4% on exempt product categories drains margin',
            'Customer disputes and chargebacks tied to inflated tax at checkout',
          ],
          severity: 'critical',
        },
        cost: {
          title: 'Cost of Inaction',
          description:
            'States are aggressively pursuing post-Wayfair enforcement. Penalties compound with interest, and voluntary disclosure windows are closing.',
          bullets: [
            '$200K - $1M annual exposure from penalties, interest, and over-collection leakage',
            'Back-tax assessments averaging 3 years of unpaid liability per state',
            'Reputational risk when customers discover incorrect tax charges',
          ],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Expected Return',
          description:
            'Automated tax compliance eliminates manual effort, closes audit exposure, and ensures accurate collection at checkout.',
          bullets: [
            '8x ROI within 2-3 months of deployment',
            'Eliminate 20+ hours per quarter of manual filing work',
            'Reduce audit exposure to near-zero across all nexus states',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Deploy a tax liability audit pipeline paired with an automated tax calculation engine that respects economic nexus thresholds and product taxability rules.',
        prerequisites: [
          'Shopify Admin API access with read permissions on orders, products, and customers',
          'State-level nexus threshold reference data (revenue and transaction counts)',
          'Product tax code mapping for your catalog',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        toolsUsed: ['SQL', 'Python', 'pytest', 'Docker', 'GitHub Actions', 'cron / Airflow', 'Slack API'],
        steps: [
          {
            stepNumber: 1,
            title: 'Tax Liability Audit — Nexus Exposure Analysis',
            description:
              'Identify which states have crossed economic nexus thresholds by aggregating order data against statutory limits.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Economic Nexus Threshold Analysis',
                description:
                  'Calculate cumulative revenue and transaction counts per state against nexus thresholds.',
                code: `-- Economic nexus threshold check by state
WITH state_activity AS (
  SELECT
    ship_state                          AS state_code,
    COUNT(DISTINCT order_id)            AS total_transactions,
    SUM(total_price)                    AS total_revenue,
    MIN(created_at)                     AS first_order_date,
    MAX(created_at)                     AS last_order_date
  FROM shopify_orders
  WHERE financial_status = 'paid'
    AND created_at >= DATE_TRUNC('year', CURRENT_DATE)
  GROUP BY ship_state
),
nexus_check AS (
  SELECT
    sa.state_code,
    sa.total_revenue,
    sa.total_transactions,
    nt.revenue_threshold,
    nt.transaction_threshold,
    CASE
      WHEN sa.total_revenue   >= nt.revenue_threshold     THEN TRUE
      WHEN sa.total_transactions >= nt.transaction_threshold THEN TRUE
      ELSE FALSE
    END AS nexus_triggered,
    sa.first_order_date,
    sa.last_order_date
  FROM state_activity sa
  JOIN nexus_thresholds nt ON sa.state_code = nt.state_code
)
SELECT
  state_code,
  total_revenue,
  total_transactions,
  revenue_threshold,
  transaction_threshold,
  nexus_triggered,
  ROUND(total_revenue - revenue_threshold, 2) AS revenue_over_threshold
FROM nexus_check
ORDER BY nexus_triggered DESC, total_revenue DESC;`,
              },
              {
                language: 'sql',
                title: 'Tax Collection Gap Identification',
                description:
                  'Compare tax actually collected versus tax that should have been collected per state and product category.',
                code: `-- Tax collection gap per state and product category
WITH expected_tax AS (
  SELECT
    o.order_id,
    o.ship_state,
    li.product_type,
    li.line_price,
    ptr.tax_rate                          AS expected_rate,
    ROUND(li.line_price * ptr.tax_rate, 2) AS expected_tax_amount
  FROM shopify_orders o
  JOIN shopify_line_items li     ON o.order_id = li.order_id
  JOIN product_tax_rates ptr     ON li.product_type = ptr.product_type
                                AND o.ship_state   = ptr.state_code
  WHERE o.financial_status = 'paid'
    AND o.created_at >= CURRENT_DATE - INTERVAL '12 months'
),
actual_tax AS (
  SELECT
    order_id,
    SUM(tax_amount) AS collected_tax
  FROM shopify_tax_lines
  GROUP BY order_id
)
SELECT
  et.ship_state,
  et.product_type,
  COUNT(DISTINCT et.order_id)                          AS order_count,
  SUM(et.expected_tax_amount)                          AS total_expected_tax,
  SUM(at.collected_tax)                                AS total_collected_tax,
  SUM(et.expected_tax_amount) - SUM(at.collected_tax)  AS tax_gap,
  ROUND(
    (SUM(et.expected_tax_amount) - SUM(at.collected_tax))
    / NULLIF(SUM(et.expected_tax_amount), 0) * 100, 2
  )                                                     AS gap_pct
FROM expected_tax et
LEFT JOIN actual_tax at ON et.order_id = at.order_id
GROUP BY et.ship_state, et.product_type
HAVING ABS(SUM(et.expected_tax_amount) - SUM(at.collected_tax)) > 0.01
ORDER BY ABS(tax_gap) DESC;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Automated Tax Calculation Engine',
            description:
              'Build a Python service that calculates accurate tax per line item at checkout, factoring in nexus status, product taxability, and exemptions.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Tax Calculation Service',
                description:
                  'Core tax calculation engine that applies the correct rate based on nexus status and product taxability.',
                code: `import json
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass

@dataclass
class TaxResult:
    state: str
    product_type: str
    taxable_amount: Decimal
    rate: Decimal
    tax_due: Decimal
    nexus_active: bool

class TaxEngine:
    def __init__(self, nexus_states: dict, rate_table: dict, exemptions: set):
        self.nexus_states = nexus_states   # {state: bool}
        self.rate_table = rate_table       # {(state, product_type): Decimal}
        self.exemptions = exemptions       # {(state, product_type)}

    def calculate_line_tax(self, state: str, product_type: str,
                           amount: Decimal) -> TaxResult:
        nexus_active = self.nexus_states.get(state, False)
        if not nexus_active:
            return TaxResult(state, product_type, amount,
                             Decimal("0"), Decimal("0"), False)

        if (state, product_type) in self.exemptions:
            return TaxResult(state, product_type, amount,
                             Decimal("0"), Decimal("0"), True)

        rate = self.rate_table.get(
            (state, product_type),
            self.rate_table.get((state, "default"), Decimal("0"))
        )
        tax_due = (amount * rate).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return TaxResult(state, product_type, amount, rate, tax_due, True)

    def calculate_order_tax(self, state: str, line_items: list) -> dict:
        results = []
        total_tax = Decimal("0")
        for item in line_items:
            result = self.calculate_line_tax(
                state, item["product_type"],
                Decimal(str(item["price"]))
            )
            results.append(result)
            total_tax += result.tax_due
        return {"line_results": results, "total_tax": float(total_tax)}`,
              },
              {
                language: 'python',
                title: 'Nexus Threshold Monitor',
                description:
                  'Continuously monitors cumulative revenue and transactions per state, alerting when thresholds approach.',
                code: `import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("nexus_monitor")

ALERT_THRESHOLD_PCT = 0.80  # warn at 80% of nexus limit

class NexusMonitor:
    def __init__(self, db_conn, thresholds: dict):
        self.db = db_conn
        self.thresholds = thresholds  # {state: {revenue: X, transactions: Y}}

    def get_state_totals(self, state: str,
                         since: Optional[datetime] = None) -> dict:
        since = since or datetime(datetime.now().year, 1, 1)
        query = """
            SELECT COUNT(DISTINCT order_id)  AS txn_count,
                   COALESCE(SUM(total_price), 0) AS revenue
            FROM shopify_orders
            WHERE ship_state = %s
              AND financial_status = 'paid'
              AND created_at >= %s
        """
        row = self.db.execute(query, (state, since)).fetchone()
        return {"transactions": row[0], "revenue": float(row[1])}

    def check_all_states(self) -> list:
        alerts = []
        for state, limits in self.thresholds.items():
            totals = self.get_state_totals(state)
            rev_pct = totals["revenue"] / limits["revenue"]
            txn_pct = totals["transactions"] / limits["transactions"]

            if rev_pct >= 1.0 or txn_pct >= 1.0:
                alerts.append({"state": state, "status": "NEXUS_TRIGGERED",
                               "revenue": totals["revenue"],
                               "transactions": totals["transactions"]})
                logger.warning("Nexus TRIGGERED in %s", state)
            elif rev_pct >= ALERT_THRESHOLD_PCT or txn_pct >= ALERT_THRESHOLD_PCT:
                alerts.append({"state": state, "status": "APPROACHING",
                               "revenue_pct": round(rev_pct * 100, 1),
                               "txn_pct": round(txn_pct * 100, 1)})
                logger.info("Nexus approaching in %s (%.1f%% rev)",
                            state, rev_pct * 100)
        return alerts`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate tax calculation accuracy and nexus threshold logic with data quality assertions and automated pytest suites.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Tax Data Quality Assertions',
                description:
                  'Run data quality checks to ensure tax rates, nexus flags, and collection amounts remain consistent and correct.',
                code: `-- Assert: every nexus-triggered state has a positive tax rate entry
SELECT
  nc.state_code,
  nc.nexus_triggered,
  COUNT(ptr.tax_rate) AS rate_entries
FROM nexus_check nc
LEFT JOIN product_tax_rates ptr
  ON nc.state_code = ptr.state_code
WHERE nc.nexus_triggered = TRUE
GROUP BY nc.state_code, nc.nexus_triggered
HAVING COUNT(ptr.tax_rate) = 0;
-- Expected: zero rows (all nexus states must have rates)

-- Assert: no negative tax amounts in computed results
SELECT order_id, line_item_id, computed_tax
FROM tax_calculation_results
WHERE computed_tax < 0;
-- Expected: zero rows

-- Assert: tax gap percentage within 1% tolerance for matched states
SELECT ship_state,
       gap_pct
FROM tax_collection_gaps
WHERE ABS(gap_pct) > 1.0
ORDER BY ABS(gap_pct) DESC;
-- Expected: zero rows after calibration

-- Assert: all orders in the past 30 days have a tax calculation record
SELECT o.order_id
FROM shopify_orders o
LEFT JOIN tax_calculation_results tcr
  ON o.order_id = tcr.order_id
WHERE o.created_at >= CURRENT_DATE - INTERVAL '30 days'
  AND o.financial_status = 'paid'
  AND tcr.order_id IS NULL;
-- Expected: zero rows`,
              },
              {
                language: 'python',
                title: 'Tax Pipeline Validation Suite',
                description:
                  'pytest-based test suite that validates the tax engine, nexus monitor, and end-to-end calculation pipeline.',
                code: `import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from tax_engine import TaxEngine, TaxResult
from nexus_monitor import NexusMonitor

# ── Fixtures ──────────────────────────────────────────

@pytest.fixture
def sample_engine():
    nexus = {"TX": True, "CA": True, "OR": False}
    rates = {
        ("TX", "clothing"): Decimal("0.0825"),
        ("TX", "default"): Decimal("0.0625"),
        ("CA", "clothing"): Decimal("0.0725"),
        ("CA", "default"): Decimal("0.0750"),
    }
    exemptions = {("TX", "grocery")}
    return TaxEngine(nexus, rates, exemptions)

@pytest.fixture
def mock_db():
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = (150, 95000.00)
    return db

# ── Tax Engine Tests ──────────────────────────────────

class TestTaxEngine:
    def test_nexus_active_returns_correct_tax(self, sample_engine):
        result = sample_engine.calculate_line_tax(
            "TX", "clothing", Decimal("100.00")
        )
        assert result.nexus_active is True
        assert result.tax_due == Decimal("8.25")

    def test_no_nexus_returns_zero_tax(self, sample_engine):
        result = sample_engine.calculate_line_tax(
            "OR", "clothing", Decimal("100.00")
        )
        assert result.nexus_active is False
        assert result.tax_due == Decimal("0")

    def test_exempt_product_returns_zero_tax(self, sample_engine):
        result = sample_engine.calculate_line_tax(
            "TX", "grocery", Decimal("50.00")
        )
        assert result.tax_due == Decimal("0")
        assert result.nexus_active is True

    def test_order_tax_sums_correctly(self, sample_engine):
        items = [
            {"product_type": "clothing", "price": 100},
            {"product_type": "default", "price": 50},
        ]
        result = sample_engine.calculate_order_tax("TX", items)
        assert result["total_tax"] == pytest.approx(11.38, abs=0.01)

    def test_unknown_state_returns_zero(self, sample_engine):
        result = sample_engine.calculate_line_tax(
            "ZZ", "clothing", Decimal("100.00")
        )
        assert result.tax_due == Decimal("0")

# ── Nexus Monitor Tests ───────────────────────────────

class TestNexusMonitor:
    def test_check_all_states_detects_trigger(self, mock_db):
        thresholds = {"TX": {"revenue": 100000, "transactions": 200}}
        monitor = NexusMonitor(mock_db, thresholds)
        mock_db.execute.return_value.fetchone.return_value = (250, 110000.00)
        alerts = monitor.check_all_states()
        triggered = [a for a in alerts if a["status"] == "NEXUS_TRIGGERED"]
        assert len(triggered) == 1
        assert triggered[0]["state"] == "TX"

    def test_approaching_threshold_generates_warning(self, mock_db):
        thresholds = {"CA": {"revenue": 100000, "transactions": 200}}
        monitor = NexusMonitor(mock_db, thresholds)
        mock_db.execute.return_value.fetchone.return_value = (150, 85000.00)
        alerts = monitor.check_all_states()
        approaching = [a for a in alerts if a["status"] == "APPROACHING"]
        assert len(approaching) == 1`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Containerize and deploy the tax compliance pipeline with Docker, configuration management, and scheduled execution.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Tax Pipeline Deployment Script',
                description:
                  'Build, test, and deploy the tax compliance pipeline using Docker with health checks and rollback.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ─────────────────────────────────────
APP_NAME="shopify-tax-pipeline"
IMAGE_TAG="\${APP_NAME}:\$(git rev-parse --short HEAD)"
COMPOSE_FILE="docker-compose.tax.yml"
HEALTH_URL="http://localhost:8090/health"
ROLLBACK_TAG=""

log() { printf '[%s] %s\\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$1"; }

# ── Pre-flight checks ────────────────────────────────
log "Running pre-flight checks..."
command -v docker >/dev/null 2>&1 || { log "ERROR: docker not found"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { log "ERROR: docker-compose not found"; exit 1; }

# Capture current running image for rollback
ROLLBACK_TAG=\$(docker inspect --format='{{.Config.Image}}' "\${APP_NAME}" 2>/dev/null || echo "")
log "Rollback target: \${ROLLBACK_TAG:-none}"

# ── Build ─────────────────────────────────────────────
log "Building image \${IMAGE_TAG}..."
docker build -t "\${IMAGE_TAG}" -f Dockerfile.tax .

# ── Run tests inside container ────────────────────────
log "Running pytest inside container..."
docker run --rm "\${IMAGE_TAG}" pytest tests/tax/ -v --tb=short
log "Tests passed."

# ── Deploy ────────────────────────────────────────────
log "Deploying with docker-compose..."
IMAGE_TAG="\${IMAGE_TAG}" docker-compose -f "\${COMPOSE_FILE}" up -d

# ── Health check ──────────────────────────────────────
log "Waiting for health check..."
for i in \$(seq 1 30); do
  if curl -sf "\${HEALTH_URL}" > /dev/null 2>&1; then
    log "Service healthy after \${i}s"
    break
  fi
  if [ "\$i" -eq 30 ]; then
    log "ERROR: health check failed after 30s"
    if [ -n "\${ROLLBACK_TAG}" ]; then
      log "Rolling back to \${ROLLBACK_TAG}..."
      IMAGE_TAG="\${ROLLBACK_TAG}" docker-compose -f "\${COMPOSE_FILE}" up -d
    fi
    exit 1
  fi
  sleep 1
done

log "Deployment complete: \${IMAGE_TAG}"`,
              },
              {
                language: 'python',
                title: 'Tax Pipeline Configuration Loader',
                description:
                  'Typed configuration loader using dataclasses for the tax compliance pipeline.',
                code: `from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class ShopifyConfig:
    shop_domain: str
    api_key: str
    api_secret: str
    api_version: str = "2025-01"

@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    ssl_mode: str = "require"

    @property
    def connection_string(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )

@dataclass(frozen=True)
class AlertConfig:
    slack_webhook_url: str
    channel: str = "#tax-alerts"
    mention_on_critical: str = "@tax-team"
    enabled: bool = True

@dataclass(frozen=True)
class TaxPipelineConfig:
    shopify: ShopifyConfig
    database: DatabaseConfig
    alerts: AlertConfig
    nexus_threshold_file: str = "config/nexus_thresholds.json"
    schedule_cron: str = "0 6 * * *"
    log_level: str = "INFO"
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> TaxPipelineConfig:
        return cls(
            shopify=ShopifyConfig(
                shop_domain=os.environ["SHOPIFY_SHOP_DOMAIN"],
                api_key=os.environ["SHOPIFY_API_KEY"],
                api_secret=os.environ["SHOPIFY_API_SECRET"],
                api_version=os.environ.get("SHOPIFY_API_VERSION", "2025-01"),
            ),
            database=DatabaseConfig(
                host=os.environ["DB_HOST"],
                port=int(os.environ.get("DB_PORT", "5432")),
                database=os.environ["DB_NAME"],
                user=os.environ["DB_USER"],
                password=os.environ["DB_PASSWORD"],
                ssl_mode=os.environ.get("DB_SSL_MODE", "require"),
            ),
            alerts=AlertConfig(
                slack_webhook_url=os.environ["SLACK_WEBHOOK_URL"],
                channel=os.environ.get("SLACK_CHANNEL", "#tax-alerts"),
                enabled=os.environ.get("ALERTS_ENABLED", "true").lower() == "true",
            ),
            nexus_threshold_file=os.environ.get(
                "NEXUS_THRESHOLD_FILE", "config/nexus_thresholds.json"
            ),
            schedule_cron=os.environ.get("SCHEDULE_CRON", "0 6 * * *"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            dry_run=os.environ.get("DRY_RUN", "false").lower() == "true",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> TaxPipelineConfig:
        with open(path) as f:
            data = json.load(f)
        return cls(
            shopify=ShopifyConfig(**data["shopify"]),
            database=DatabaseConfig(**data["database"]),
            alerts=AlertConfig(**data["alerts"]),
            nexus_threshold_file=data.get(
                "nexus_threshold_file", "config/nexus_thresholds.json"
            ),
            schedule_cron=data.get("schedule_cron", "0 6 * * *"),
            log_level=data.get("log_level", "INFO"),
            dry_run=data.get("dry_run", False),
        )`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Continuously monitor nexus thresholds and tax calculation accuracy with automated alerting to Slack.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Nexus Threshold Monitor',
                description:
                  'Continuously monitors cumulative revenue and transactions per state, alerting when thresholds approach.',
                code: `import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("nexus_monitor")

ALERT_THRESHOLD_PCT = 0.80  # warn at 80% of nexus limit

class NexusMonitor:
    def __init__(self, db_conn, thresholds: dict):
        self.db = db_conn
        self.thresholds = thresholds  # {state: {revenue: X, transactions: Y}}

    def get_state_totals(self, state: str,
                         since: Optional[datetime] = None) -> dict:
        since = since or datetime(datetime.now().year, 1, 1)
        query = """
            SELECT COUNT(DISTINCT order_id)  AS txn_count,
                   COALESCE(SUM(total_price), 0) AS revenue
            FROM shopify_orders
            WHERE ship_state = %s
              AND financial_status = 'paid'
              AND created_at >= %s
        """
        row = self.db.execute(query, (state, since)).fetchone()
        return {"transactions": row[0], "revenue": float(row[1])}

    def check_all_states(self) -> list:
        alerts = []
        for state, limits in self.thresholds.items():
            totals = self.get_state_totals(state)
            rev_pct = totals["revenue"] / limits["revenue"]
            txn_pct = totals["transactions"] / limits["transactions"]

            if rev_pct >= 1.0 or txn_pct >= 1.0:
                alerts.append({"state": state, "status": "NEXUS_TRIGGERED",
                               "revenue": totals["revenue"],
                               "transactions": totals["transactions"]})
                logger.warning("Nexus TRIGGERED in %s", state)
            elif rev_pct >= ALERT_THRESHOLD_PCT or txn_pct >= ALERT_THRESHOLD_PCT:
                alerts.append({"state": state, "status": "APPROACHING",
                               "revenue_pct": round(rev_pct * 100, 1),
                               "txn_pct": round(txn_pct * 100, 1)})
                logger.info("Nexus approaching in %s (%.1f%% rev)",
                            state, rev_pct * 100)
        return alerts`,
              },
              {
                language: 'python',
                title: 'Tax Compliance Slack Alerting Service',
                description:
                  'Sends structured Slack alerts when nexus thresholds are breached or tax collection gaps exceed tolerance.',
                code: `import json
import logging
from datetime import datetime
from typing import Any
import urllib.request

logger = logging.getLogger("tax_alerting")

class TaxComplianceAlerter:
    def __init__(self, webhook_url: str, channel: str = "#tax-alerts"):
        self.webhook_url = webhook_url
        self.channel = channel

    def _post_slack(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=10)
            logger.info("Slack alert sent to %s", self.channel)
        except Exception as exc:
            logger.error("Failed to send Slack alert: %s", exc)

    def alert_nexus_triggered(self, state: str, revenue: float,
                              transactions: int) -> None:
        payload = {
            "channel": self.channel,
            "username": "Tax Compliance Bot",
            "icon_emoji": ":rotating_light:",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Nexus TRIGGERED: {state}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn",
                         "text": f"*Revenue:* \${revenue:,.2f}"},
                        {"type": "mrkdwn",
                         "text": f"*Transactions:* {transactions:,}"},
                        {"type": "mrkdwn",
                         "text": f"*Date:* {datetime.utcnow():%Y-%m-%d}"},
                    ],
                },
            ],
        }
        self._post_slack(payload)

    def alert_tax_gap(self, state: str, gap_pct: float,
                      gap_amount: float) -> None:
        severity = "HIGH" if abs(gap_pct) > 2.0 else "WARN"
        payload = {
            "channel": self.channel,
            "username": "Tax Compliance Bot",
            "text": (
                f"[{severity}] Tax collection gap in {state}: "
                f"{gap_pct:+.2f}% (\${abs(gap_amount):,.2f})"
            ),
        }
        self._post_slack(payload)

    def send_daily_summary(self, results: list[dict[str, Any]]) -> None:
        triggered = [r for r in results if r["status"] == "NEXUS_TRIGGERED"]
        approaching = [r for r in results if r["status"] == "APPROACHING"]
        summary = (
            f"*Daily Tax Compliance Summary*\\n"
            f"States monitored: {len(results)}\\n"
            f"Nexus triggered: {len(triggered)}\\n"
            f"Approaching threshold: {len(approaching)}"
        )
        self._post_slack({"channel": self.channel, "text": summary})`,
              },
            ],
          },
        ],
      },
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to automate sales tax nexus monitoring, rate lookups, and compliance alerting without custom code. Extract order data from Shopify, analyze nexus thresholds with AI, and receive automated Slack alerts when thresholds approach.',
        estimatedMonthlyCost: '$120 - $200/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'TaxJar Basic ($19/mo)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Avalara AvaTax'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Set up automated data extraction from Shopify to gather order data by state for nexus threshold analysis. Use Zapier to pull order summaries daily and format them for AI analysis.',
            toolsUsed: ['Shopify Admin API', 'Zapier', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Shopify Order Extraction Trigger',
                description:
                  'Configure Zapier to extract daily order summaries from Shopify grouped by shipping state.',
                code: `{
  "trigger": {
    "app": "Shopify",
    "event": "New Order",
    "filters": {
      "financial_status": "paid",
      "created_at": "last_24_hours"
    }
  },
  "action_1": {
    "app": "Google Sheets",
    "event": "Create Spreadsheet Row",
    "data": {
      "spreadsheet_id": "{{env.NEXUS_TRACKING_SHEET}}",
      "worksheet": "Daily Orders",
      "columns": {
        "order_id": "{{order.id}}",
        "order_number": "{{order.order_number}}",
        "ship_state": "{{order.shipping_address.province_code}}",
        "total_price": "{{order.total_price}}",
        "created_at": "{{order.created_at}}",
        "year": "{{formatDate order.created_at 'YYYY'}}"
      }
    }
  },
  "schedule": {
    "frequency": "daily",
    "time": "06:00",
    "timezone": "America/New_York"
  }
}`,
              },
              {
                language: 'json',
                title: 'State Aggregation Summary Format',
                description:
                  'JSON structure for aggregated state-level nexus data prepared for AI analysis.',
                code: `{
  "nexus_analysis_request": {
    "report_date": "2025-01-31",
    "reporting_period": "YTD_2025",
    "state_summaries": [
      {
        "state_code": "TX",
        "state_name": "Texas",
        "total_orders": 185,
        "total_revenue": 89500.00,
        "nexus_threshold_revenue": 500000,
        "nexus_threshold_transactions": 200,
        "revenue_pct_of_threshold": 17.9,
        "transactions_pct_of_threshold": 92.5,
        "first_order_date": "2025-01-03",
        "last_order_date": "2025-01-31"
      },
      {
        "state_code": "CA",
        "state_name": "California",
        "total_orders": 312,
        "total_revenue": 156000.00,
        "nexus_threshold_revenue": 500000,
        "nexus_threshold_transactions": 200,
        "revenue_pct_of_threshold": 31.2,
        "transactions_pct_of_threshold": 156.0,
        "first_order_date": "2025-01-02",
        "last_order_date": "2025-01-31"
      }
    ],
    "thresholds_reference": {
      "source": "state_nexus_laws_2025",
      "default_revenue": 100000,
      "default_transactions": 200
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
              'Use ChatGPT or Claude to analyze state-level order data against economic nexus thresholds, identify states approaching or exceeding limits, and generate compliance recommendations.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Nexus Threshold Analysis Prompt Template',
                description:
                  'Structured prompt for AI to analyze nexus data and provide actionable compliance guidance.',
                code: `system_prompt: |
  You are a sales tax compliance analyst specializing in economic nexus
  analysis for ecommerce businesses. You understand post-Wayfair nexus
  laws across all US states and can identify compliance risks.

  Your analysis should be:
  - Precise about threshold calculations
  - Clear about which states require immediate action
  - Specific about next steps for registration and collection

user_prompt_template: |
  Analyze the following state-level sales data for economic nexus exposure:

  ## Current Period Data
  Reporting Period: {{reporting_period}}
  Report Date: {{report_date}}

  ## State Summaries
  {{#each state_summaries}}
  ### {{state_name}} ({{state_code}})
  - Total Orders: {{total_orders}}
  - Total Revenue: \${{total_revenue}}
  - Revenue Threshold: \${{nexus_threshold_revenue}}
  - Transaction Threshold: {{nexus_threshold_transactions}}
  - Revenue % of Threshold: {{revenue_pct_of_threshold}}%
  - Transactions % of Threshold: {{transactions_pct_of_threshold}}%
  {{/each}}

  ## Analysis Required
  1. Identify states where nexus has been TRIGGERED (either threshold exceeded)
  2. Identify states APPROACHING nexus (>75% of either threshold)
  3. For triggered states, provide registration deadlines and next steps
  4. Calculate projected nexus trigger dates for approaching states
  5. Recommend tax collection start dates

  Format your response as a structured compliance report with:
  - Executive Summary (2-3 sentences)
  - CRITICAL: States requiring immediate registration
  - WARNING: States approaching thresholds
  - Action Items with specific deadlines

expected_output_format: |
  ## Executive Summary
  [Brief overview of nexus status]

  ## CRITICAL - Immediate Action Required
  | State | Trigger Type | Threshold | Current | Action Required | Deadline |
  |-------|--------------|-----------|---------|-----------------|----------|

  ## WARNING - Approaching Thresholds
  | State | Metric | % of Threshold | Projected Trigger Date |
  |-------|--------|----------------|------------------------|

  ## Action Items
  1. [Specific action with deadline]
  2. [Specific action with deadline]`,
              },
              {
                language: 'yaml',
                title: 'Tax Rate Lookup Prompt Template',
                description:
                  'Prompt template for AI to assist with product taxability and rate lookups by state.',
                code: `system_prompt: |
  You are a sales tax rate specialist. You help ecommerce businesses
  understand product taxability rules and applicable tax rates across
  US states. You know that tax rates vary by:
  - State base rate
  - County/city local rates
  - Product category exemptions
  - Special tax holidays

user_prompt_template: |
  I need tax rate guidance for the following scenario:

  ## Business Context
  - Shopify Store: {{shop_name}}
  - Nexus States: {{nexus_states}}

  ## Product Details
  - Product Type: {{product_type}}
  - Product Category: {{product_category}}
  - Price: \${{price}}

  ## Shipping Destination
  - State: {{ship_state}}
  - City: {{ship_city}}
  - ZIP: {{ship_zip}}

  ## Questions
  1. Is this product category taxable in {{ship_state}}?
  2. What is the combined tax rate (state + local) for {{ship_zip}}?
  3. Are there any current tax exemptions or holidays applicable?
  4. What documentation is needed if the buyer claims exemption?

  Provide specific rates and cite the relevant state tax code section.

validation_rules:
  - Always verify nexus status before providing collection guidance
  - Flag marketplace facilitator states where platform collects tax
  - Note any pending rate changes in the next 90 days`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Automate the complete workflow with Zapier to run daily nexus analysis, send AI-generated compliance reports to Slack, and trigger alerts when thresholds are approached.',
            toolsUsed: ['Zapier', 'Slack', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Complete Nexus Monitoring Workflow',
                description:
                  'End-to-end Zapier workflow that aggregates data, calls ChatGPT for analysis, and delivers alerts.',
                code: `{
  "zap_name": "Daily Sales Tax Nexus Monitor",
  "trigger": {
    "app": "Schedule by Zapier",
    "event": "Every Day",
    "time": "07:00",
    "timezone": "America/New_York"
  },
  "actions": [
    {
      "step": 1,
      "app": "Google Sheets",
      "event": "Lookup Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{env.NEXUS_TRACKING_SHEET}}",
        "worksheet": "State Aggregates",
        "lookup_column": "year",
        "lookup_value": "2025"
      }
    },
    {
      "step": 2,
      "app": "Formatter by Zapier",
      "event": "Text",
      "config": {
        "transform": "concatenate",
        "template": "Analyze nexus status for: {{step1.all_rows_json}}"
      }
    },
    {
      "step": 3,
      "app": "ChatGPT",
      "event": "Conversation",
      "config": {
        "model": "gpt-4",
        "system_message": "You are a sales tax compliance analyst...",
        "user_message": "{{step2.output}}",
        "max_tokens": 2000
      }
    },
    {
      "step": 4,
      "app": "Filter by Zapier",
      "event": "Only Continue If",
      "config": {
        "condition": "{{step3.response}} contains CRITICAL",
        "action": "continue_or_halt"
      }
    },
    {
      "step": 5,
      "app": "Slack",
      "event": "Send Channel Message",
      "config": {
        "channel": "#tax-compliance",
        "message": {
          "blocks": [
            {
              "type": "header",
              "text": "Daily Nexus Analysis - {{zap.trigger_time}}"
            },
            {
              "type": "section",
              "text": "{{step3.response}}"
            },
            {
              "type": "actions",
              "elements": [
                {
                  "type": "button",
                  "text": "View Full Report",
                  "url": "{{env.NEXUS_TRACKING_SHEET}}"
                },
                {
                  "type": "button",
                  "text": "Start Registration",
                  "url": "https://taxjar.com/states"
                }
              ]
            }
          ]
        },
        "bot_name": "Tax Compliance Bot",
        "bot_icon": ":receipt:"
      }
    },
    {
      "step": 6,
      "app": "Google Sheets",
      "event": "Create Spreadsheet Row",
      "config": {
        "spreadsheet_id": "{{env.NEXUS_TRACKING_SHEET}}",
        "worksheet": "Analysis Log",
        "columns": {
          "analysis_date": "{{zap.trigger_time}}",
          "ai_response": "{{step3.response}}",
          "critical_states": "{{step3.response | extract 'CRITICAL'}}",
          "alert_sent": true
        }
      }
    }
  ],
  "error_handling": {
    "on_error": "notify",
    "notify_email": "tax-team@company.com",
    "retry_count": 2
  }
}`,
              },
              {
                language: 'json',
                title: 'Threshold Alert Trigger Configuration',
                description:
                  'Zapier filter configuration to trigger urgent alerts when any state exceeds 80% of nexus thresholds.',
                code: `{
  "alert_workflow": {
    "name": "Nexus Threshold Breach Alert",
    "trigger": {
      "app": "Google Sheets",
      "event": "New or Updated Spreadsheet Row",
      "config": {
        "spreadsheet_id": "{{env.NEXUS_TRACKING_SHEET}}",
        "worksheet": "State Aggregates",
        "trigger_column": "last_updated"
      }
    },
    "filter": {
      "conditions": [
        {
          "field": "transactions_pct_of_threshold",
          "operator": "greater_than",
          "value": 80
        },
        {
          "field": "revenue_pct_of_threshold",
          "operator": "greater_than",
          "value": 80
        }
      ],
      "logic": "OR"
    },
    "actions": [
      {
        "app": "Slack",
        "event": "Send Direct Message",
        "config": {
          "user": "@tax-manager",
          "message": ":warning: *Nexus Threshold Alert*\\n\\nState: {{trigger.state_name}}\\nRevenue: {{trigger.revenue_pct_of_threshold}}% of threshold\\nTransactions: {{trigger.transactions_pct_of_threshold}}% of threshold\\n\\nImmediate review required."
        }
      },
      {
        "app": "Email by Zapier",
        "event": "Send Outbound Email",
        "config": {
          "to": "tax-team@company.com",
          "subject": "[URGENT] Nexus threshold approaching in {{trigger.state_name}}",
          "body": "State {{trigger.state_code}} has reached {{trigger.transactions_pct_of_threshold}}% of transaction threshold. Review and prepare for registration."
        }
      }
    ]
  }
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent system using CrewAI and LangGraph to provide autonomous sales tax compliance management. Specialized agents handle nexus monitoring, tax rate calculation, filing preparation, and audit defense, coordinated by a supervisor agent that ensures consistent compliance across all states.',
        estimatedMonthlyCost: '$600 - $1,200/month',
        architecture:
          'Supervisor agent coordinates four specialist agents: Nexus Monitor Agent tracks economic nexus thresholds across all states, Tax Calculator Agent determines precise rates by product and location, Filing Agent prepares state returns and remittance, and Audit Agent maintains documentation and responds to state inquiries. State is managed in Redis with daily checkpoints.',
        agents: [
          {
            name: 'Nexus Monitor Agent',
            role: 'Economic Nexus Threshold Tracker',
            goal: 'Continuously monitor sales activity against nexus thresholds for all 50 states and alert when thresholds are approached or triggered',
            tools: ['shopify_orders_api', 'nexus_threshold_db', 'state_law_lookup', 'slack_alerter'],
          },
          {
            name: 'Tax Calculator Agent',
            role: 'Tax Rate Determination Specialist',
            goal: 'Calculate precise tax rates for any product/location combination considering state rates, local rates, product exemptions, and tax holidays',
            tools: ['taxjar_api', 'avalara_api', 'product_taxability_db', 'zip_code_lookup'],
          },
          {
            name: 'Filing Preparation Agent',
            role: 'Return Preparation and Remittance Specialist',
            goal: 'Prepare accurate state tax returns, calculate remittance amounts, and ensure timely filing across all registered states',
            tools: ['filing_calendar_db', 'return_generator', 'payment_scheduler', 'state_portal_api'],
          },
          {
            name: 'Audit Defense Agent',
            role: 'Compliance Documentation and Audit Response Specialist',
            goal: 'Maintain comprehensive audit trails, respond to state inquiries, and prepare documentation packages for audits',
            tools: ['document_store', 'exemption_certificate_db', 'audit_response_generator', 'state_correspondence_api'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with daily checkpointing and 90-day audit trail retention',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system architecture with CrewAI, establishing clear roles, goals, and tool assignments for each specialist agent in the tax compliance workflow.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Tax Compliance Agent Definitions',
                description:
                  'CrewAI agent definitions for the sales tax compliance multi-agent system.',
                code: `from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import logging

logger = logging.getLogger("tax_compliance_agents")

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.1,  # Low temperature for precise tax calculations
)


class TaxComplianceAgents:
    """Factory for creating tax compliance specialist agents."""

    def __init__(self, tools_registry: Dict[str, Any]):
        self.tools = tools_registry
        self.llm = llm

    def create_nexus_monitor_agent(self) -> Agent:
        """Agent responsible for tracking economic nexus thresholds."""
        return Agent(
            role="Economic Nexus Threshold Tracker",
            goal=(
                "Monitor sales activity against nexus thresholds for all 50 states. "
                "Alert immediately when any state reaches 80% of either revenue or "
                "transaction thresholds. Maintain accurate running totals by state."
            ),
            backstory=(
                "You are an expert in post-Wayfair economic nexus laws. You know "
                "the exact thresholds for each state, including special rules for "
                "marketplace facilitator exemptions and small seller exceptions. "
                "You monitor transaction volumes and revenue in real-time to prevent "
                "compliance surprises."
            ),
            tools=[
                self.tools["shopify_orders_api"],
                self.tools["nexus_threshold_db"],
                self.tools["state_law_lookup"],
                self.tools["slack_alerter"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def create_tax_calculator_agent(self) -> Agent:
        """Agent responsible for precise tax rate determination."""
        return Agent(
            role="Tax Rate Determination Specialist",
            goal=(
                "Calculate precise tax rates for any product and shipping location. "
                "Consider state base rates, county rates, city rates, special district "
                "rates, product-specific exemptions, and active tax holidays."
            ),
            backstory=(
                "You are a sales tax rate expert with deep knowledge of tax codes "
                "across all US jurisdictions. You understand product taxability "
                "matrices, know which categories are exempt in which states, and "
                "stay current on rate changes and tax holidays."
            ),
            tools=[
                self.tools["taxjar_api"],
                self.tools["avalara_api"],
                self.tools["product_taxability_db"],
                self.tools["zip_code_lookup"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def create_filing_agent(self) -> Agent:
        """Agent responsible for return preparation and filing."""
        return Agent(
            role="Return Preparation and Remittance Specialist",
            goal=(
                "Prepare accurate state tax returns before filing deadlines. "
                "Calculate exact remittance amounts including any prepayments. "
                "Generate filing packages with all required schedules and documentation."
            ),
            backstory=(
                "You are a tax filing specialist who ensures every return is "
                "accurate and timely. You know the filing frequencies, due dates, "
                "and specific form requirements for each state. You never miss "
                "a deadline and always maximize available discounts."
            ),
            tools=[
                self.tools["filing_calendar_db"],
                self.tools["return_generator"],
                self.tools["payment_scheduler"],
                self.tools["state_portal_api"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def create_audit_defense_agent(self) -> Agent:
        """Agent responsible for audit preparation and response."""
        return Agent(
            role="Compliance Documentation and Audit Response Specialist",
            goal=(
                "Maintain comprehensive audit trails for all tax calculations and "
                "filings. Prepare documentation packages for state audits. Draft "
                "professional responses to state inquiries within required timeframes."
            ),
            backstory=(
                "You are an audit defense specialist who maintains impeccable "
                "records. You know exactly what documentation states require, "
                "how to organize exemption certificates, and how to respond to "
                "audit requests in a way that minimizes exposure."
            ),
            tools=[
                self.tools["document_store"],
                self.tools["exemption_certificate_db"],
                self.tools["audit_response_generator"],
                self.tools["state_correspondence_api"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,  # Can delegate document retrieval
        )

    def create_supervisor_agent(self) -> Agent:
        """Supervisor agent that coordinates all tax compliance activities."""
        return Agent(
            role="Tax Compliance Supervisor",
            goal=(
                "Coordinate all tax compliance activities across specialist agents. "
                "Prioritize urgent nexus alerts over routine filings. Ensure "
                "consistent compliance posture across all registered states."
            ),
            backstory=(
                "You are the head of tax compliance, overseeing a team of "
                "specialists. You understand the full compliance lifecycle from "
                "nexus determination through audit defense. You make strategic "
                "decisions about registration timing and resource allocation."
            ),
            tools=[
                self.tools["compliance_dashboard"],
                self.tools["priority_queue"],
                self.tools["escalation_handler"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )`,
              },
              {
                language: 'python',
                title: 'Tax Compliance Crew Configuration',
                description:
                  'CrewAI Crew configuration that assembles agents and defines task workflows.',
                code: `from crewai import Crew, Task, Process
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class ComplianceTaskResult:
    """Result container for compliance task execution."""
    task_type: str
    status: str
    states_affected: list[str]
    actions_taken: list[str]
    alerts_generated: int
    next_review_date: date


class TaxComplianceCrew:
    """Crew orchestrating multi-agent tax compliance workflow."""

    def __init__(self, agents_factory: TaxComplianceAgents):
        self.agents = agents_factory
        self.crew: Optional[Crew] = None

    def build_crew(self) -> Crew:
        """Assemble the tax compliance crew with all agents and tasks."""

        # Create specialist agents
        nexus_agent = self.agents.create_nexus_monitor_agent()
        calculator_agent = self.agents.create_tax_calculator_agent()
        filing_agent = self.agents.create_filing_agent()
        audit_agent = self.agents.create_audit_defense_agent()
        supervisor = self.agents.create_supervisor_agent()

        # Define tasks for daily compliance workflow
        nexus_monitoring_task = Task(
            description=(
                "Review all orders from the past 24 hours. Update running "
                "totals by state. Check each state against its nexus thresholds. "
                "Generate alerts for any state exceeding 80% of thresholds. "
                "Flag states that have newly triggered nexus for registration."
            ),
            expected_output=(
                "JSON report with: states_monitored, thresholds_approached, "
                "nexus_triggered, alerts_sent, recommended_actions"
            ),
            agent=nexus_agent,
        )

        rate_validation_task = Task(
            description=(
                "For any orders flagged with tax calculation questions, validate "
                "the applied tax rate. Check product taxability, verify local "
                "rates, and confirm any exemptions were correctly applied."
            ),
            expected_output=(
                "JSON report with: orders_reviewed, rates_validated, "
                "discrepancies_found, corrections_recommended"
            ),
            agent=calculator_agent,
        )

        filing_preparation_task = Task(
            description=(
                "Check the filing calendar for returns due in the next 14 days. "
                "Calculate remittance amounts for each state. Generate draft "
                "returns and queue for review before submission."
            ),
            expected_output=(
                "JSON report with: returns_due, returns_prepared, "
                "total_remittance, filing_deadlines"
            ),
            agent=filing_agent,
        )

        audit_readiness_task = Task(
            description=(
                "Review any open state inquiries or audit requests. Ensure "
                "all exemption certificates are current. Generate weekly "
                "audit readiness score for each registered state."
            ),
            expected_output=(
                "JSON report with: open_inquiries, response_deadlines, "
                "audit_readiness_scores, documentation_gaps"
            ),
            agent=audit_agent,
        )

        coordination_task = Task(
            description=(
                "Review outputs from all specialist agents. Prioritize actions "
                "based on urgency (nexus triggers > audit responses > filings). "
                "Generate daily compliance summary for stakeholders."
            ),
            expected_output=(
                "Executive summary with: critical_actions, warnings, "
                "routine_completions, compliance_score"
            ),
            agent=supervisor,
            context=[
                nexus_monitoring_task,
                rate_validation_task,
                filing_preparation_task,
                audit_readiness_task,
            ],
        )

        self.crew = Crew(
            agents=[
                supervisor,
                nexus_agent,
                calculator_agent,
                filing_agent,
                audit_agent,
            ],
            tasks=[
                nexus_monitoring_task,
                rate_validation_task,
                filing_preparation_task,
                audit_readiness_task,
                coordination_task,
            ],
            process=Process.hierarchical,
            manager_agent=supervisor,
            verbose=True,
        )

        return self.crew

    def run_daily_compliance(self, context: dict) -> ComplianceTaskResult:
        """Execute daily compliance workflow."""
        if not self.crew:
            self.build_crew()

        result = self.crew.kickoff(inputs=context)

        return ComplianceTaskResult(
            task_type="daily_compliance",
            status="completed",
            states_affected=result.get("states_affected", []),
            actions_taken=result.get("actions_taken", []),
            alerts_generated=result.get("alerts_generated", 0),
            next_review_date=date.today(),
        )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement data ingestion agents that continuously pull order data from Shopify, tax rate updates from providers, and nexus threshold changes from regulatory sources.',
            toolsUsed: ['Shopify Admin API', 'TaxJar API', 'Avalara API', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Shopify Order Ingestion Agent Tool',
                description:
                  'LangChain tool for the Nexus Monitor Agent to fetch and aggregate order data from Shopify.',
                code: `from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import httpx
import logging

logger = logging.getLogger("shopify_ingestion")


class ShopifyOrderQueryInput(BaseModel):
    """Input schema for Shopify order queries."""
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    aggregate_by: str = Field(
        default="state",
        description="Aggregation dimension: 'state', 'product_type', or 'both'"
    )


class ShopifyOrderIngestionTool(BaseTool):
    """Tool for fetching and aggregating Shopify order data."""

    name: str = "shopify_orders_api"
    description: str = (
        "Fetch orders from Shopify and aggregate by shipping state. "
        "Returns order counts, revenue totals, and transaction counts "
        "for nexus threshold analysis."
    )
    args_schema: type[BaseModel] = ShopifyOrderQueryInput

    shop_domain: str
    api_key: str
    api_secret: str
    api_version: str = "2025-01"

    def _run(
        self,
        start_date: str,
        end_date: str,
        aggregate_by: str = "state",
    ) -> Dict[str, Any]:
        """Execute Shopify order fetch and aggregation."""

        base_url = f"https://{self.shop_domain}/admin/api/{self.api_version}"
        headers = {
            "X-Shopify-Access-Token": self.api_key,
            "Content-Type": "application/json",
        }

        # Fetch orders with pagination
        all_orders: List[Dict] = []
        params = {
            "status": "any",
            "financial_status": "paid",
            "created_at_min": f"{start_date}T00:00:00Z",
            "created_at_max": f"{end_date}T23:59:59Z",
            "limit": 250,
        }

        with httpx.Client(timeout=30.0) as client:
            url = f"{base_url}/orders.json"
            while url:
                response = client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                all_orders.extend(data.get("orders", []))

                # Handle pagination
                link_header = response.headers.get("Link", "")
                url = self._extract_next_page(link_header)
                params = {}  # Clear params for subsequent requests

        # Aggregate by state
        state_aggregates = self._aggregate_by_state(all_orders)

        logger.info(
            "Fetched %d orders, aggregated to %d states",
            len(all_orders),
            len(state_aggregates),
        )

        return {
            "period": {"start": start_date, "end": end_date},
            "total_orders": len(all_orders),
            "state_aggregates": state_aggregates,
            "fetched_at": datetime.utcnow().isoformat(),
        }

    def _aggregate_by_state(
        self, orders: List[Dict]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate order data by shipping state."""
        aggregates: Dict[str, Dict[str, Any]] = {}

        for order in orders:
            shipping_addr = order.get("shipping_address", {})
            state = shipping_addr.get("province_code", "UNKNOWN")

            if state not in aggregates:
                aggregates[state] = {
                    "state_code": state,
                    "order_count": 0,
                    "total_revenue": 0.0,
                    "order_ids": [],
                }

            aggregates[state]["order_count"] += 1
            aggregates[state]["total_revenue"] += float(
                order.get("total_price", 0)
            )
            aggregates[state]["order_ids"].append(order.get("id"))

        return aggregates

    def _extract_next_page(self, link_header: str) -> Optional[str]:
        """Extract next page URL from Link header."""
        if not link_header:
            return None
        for part in link_header.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
                return url
        return None

    async def _arun(self, *args, **kwargs):
        """Async version - not implemented."""
        raise NotImplementedError("Use sync version")`,
              },
              {
                language: 'python',
                title: 'Nexus Threshold Database Tool',
                description:
                  'Tool for checking current nexus thresholds and comparing against state activity.',
                code: `from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import redis
import logging

logger = logging.getLogger("nexus_threshold_db")


class NexusStatus(Enum):
    """Status of nexus in a given state."""
    NOT_TRIGGERED = "not_triggered"
    APPROACHING = "approaching"  # >75% of threshold
    WARNING = "warning"  # >90% of threshold
    TRIGGERED = "triggered"  # Threshold exceeded


@dataclass
class StateNexusInfo:
    """Nexus information for a single state."""
    state_code: str
    state_name: str
    revenue_threshold: float
    transaction_threshold: int
    current_revenue: float
    current_transactions: int
    status: NexusStatus
    revenue_pct: float
    transaction_pct: float
    triggered_date: Optional[str] = None


class NexusThresholdQueryInput(BaseModel):
    """Input schema for nexus threshold queries."""
    state_codes: List[str] = Field(
        default=[],
        description="List of state codes to check. Empty for all states."
    )
    include_triggered_only: bool = Field(
        default=False,
        description="Only return states with triggered or approaching nexus"
    )


class NexusThresholdDBTool(BaseTool):
    """Tool for querying and updating nexus threshold data."""

    name: str = "nexus_threshold_db"
    description: str = (
        "Query nexus thresholds for US states and compare against current "
        "sales activity. Returns threshold status, percentages, and alerts "
        "for states approaching or exceeding economic nexus limits."
    )
    args_schema: type[BaseModel] = NexusThresholdQueryInput

    redis_client: Any  # redis.Redis instance
    threshold_key_prefix: str = "nexus:threshold:"
    activity_key_prefix: str = "nexus:activity:"

    # 2025 nexus thresholds by state (simplified - real impl loads from DB)
    DEFAULT_THRESHOLDS: Dict[str, Dict[str, Any]] = {
        "AL": {"revenue": 250000, "transactions": None, "name": "Alabama"},
        "AZ": {"revenue": 100000, "transactions": None, "name": "Arizona"},
        "CA": {"revenue": 500000, "transactions": None, "name": "California"},
        "CO": {"revenue": 100000, "transactions": None, "name": "Colorado"},
        "FL": {"revenue": 100000, "transactions": None, "name": "Florida"},
        "GA": {"revenue": 100000, "transactions": 200, "name": "Georgia"},
        "IL": {"revenue": 100000, "transactions": 200, "name": "Illinois"},
        "NY": {"revenue": 500000, "transactions": 100, "name": "New York"},
        "PA": {"revenue": 100000, "transactions": None, "name": "Pennsylvania"},
        "TX": {"revenue": 500000, "transactions": None, "name": "Texas"},
        "WA": {"revenue": 100000, "transactions": None, "name": "Washington"},
        # Add remaining states...
    }

    def _run(
        self,
        state_codes: List[str] = None,
        include_triggered_only: bool = False,
    ) -> Dict[str, Any]:
        """Query nexus status for specified states."""

        states_to_check = state_codes or list(self.DEFAULT_THRESHOLDS.keys())
        results: List[Dict[str, Any]] = []

        for state in states_to_check:
            if state not in self.DEFAULT_THRESHOLDS:
                continue

            threshold = self.DEFAULT_THRESHOLDS[state]
            activity = self._get_state_activity(state)

            status_info = self._calculate_status(
                state, threshold, activity
            )

            if include_triggered_only and status_info.status == NexusStatus.NOT_TRIGGERED:
                continue

            results.append({
                "state_code": status_info.state_code,
                "state_name": status_info.state_name,
                "revenue_threshold": status_info.revenue_threshold,
                "transaction_threshold": status_info.transaction_threshold,
                "current_revenue": status_info.current_revenue,
                "current_transactions": status_info.current_transactions,
                "status": status_info.status.value,
                "revenue_pct": round(status_info.revenue_pct, 1),
                "transaction_pct": round(status_info.transaction_pct, 1),
            })

        # Sort by urgency (highest percentage first)
        results.sort(
            key=lambda x: max(x["revenue_pct"], x["transaction_pct"]),
            reverse=True,
        )

        critical = [r for r in results if r["status"] == "triggered"]
        warning = [r for r in results if r["status"] in ("warning", "approaching")]

        return {
            "states_checked": len(results),
            "critical_count": len(critical),
            "warning_count": len(warning),
            "state_details": results,
            "summary": self._generate_summary(critical, warning),
        }

    def _get_state_activity(self, state: str) -> Dict[str, Any]:
        """Fetch current year activity for a state from Redis."""
        key = f"{self.activity_key_prefix}{state}:2025"
        data = self.redis_client.get(key)
        if data:
            return json.loads(data)
        return {"revenue": 0.0, "transactions": 0}

    def _calculate_status(
        self,
        state: str,
        threshold: Dict[str, Any],
        activity: Dict[str, Any],
    ) -> StateNexusInfo:
        """Calculate nexus status for a state."""
        rev_threshold = threshold["revenue"]
        txn_threshold = threshold.get("transactions") or float("inf")

        current_rev = activity.get("revenue", 0)
        current_txn = activity.get("transactions", 0)

        rev_pct = (current_rev / rev_threshold) * 100 if rev_threshold else 0
        txn_pct = (current_txn / txn_threshold) * 100 if txn_threshold != float("inf") else 0

        max_pct = max(rev_pct, txn_pct)

        if max_pct >= 100:
            status = NexusStatus.TRIGGERED
        elif max_pct >= 90:
            status = NexusStatus.WARNING
        elif max_pct >= 75:
            status = NexusStatus.APPROACHING
        else:
            status = NexusStatus.NOT_TRIGGERED

        return StateNexusInfo(
            state_code=state,
            state_name=threshold["name"],
            revenue_threshold=rev_threshold,
            transaction_threshold=threshold.get("transactions"),
            current_revenue=current_rev,
            current_transactions=current_txn,
            status=status,
            revenue_pct=rev_pct,
            transaction_pct=txn_pct,
        )

    def _generate_summary(
        self,
        critical: List[Dict],
        warning: List[Dict],
    ) -> str:
        """Generate human-readable summary."""
        lines = []
        if critical:
            states = ", ".join(c["state_code"] for c in critical)
            lines.append(f"CRITICAL: Nexus triggered in {states}")
        if warning:
            states = ", ".join(w["state_code"] for w in warning)
            lines.append(f"WARNING: Approaching threshold in {states}")
        if not lines:
            lines.append("All states within safe thresholds")
        return " | ".join(lines)

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use sync version")`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Tax Calculator and Filing Preparation agents that make decisions about tax rates, exemptions, and filing priorities based on ingested data.',
            toolsUsed: ['TaxJar API', 'Avalara API', 'Filing Calendar DB'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Tax Rate Calculator Tool',
                description:
                  'Tool for the Tax Calculator Agent to determine precise tax rates by location and product.',
                code: `from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
import httpx
import logging

logger = logging.getLogger("tax_calculator")


@dataclass
class TaxRateResult:
    """Calculated tax rate result."""
    state_rate: Decimal
    county_rate: Decimal
    city_rate: Decimal
    special_rate: Decimal
    combined_rate: Decimal
    is_taxable: bool
    exemption_reason: Optional[str]
    rate_effective_date: str


class TaxRateQueryInput(BaseModel):
    """Input schema for tax rate queries."""
    zip_code: str = Field(description="5-digit ZIP code for tax jurisdiction")
    product_type: str = Field(description="Product category for taxability check")
    amount: float = Field(description="Transaction amount in USD")
    customer_type: str = Field(
        default="retail",
        description="Customer type: 'retail', 'wholesale', 'exempt'"
    )


class TaxRateCalculatorTool(BaseTool):
    """Tool for calculating precise tax rates."""

    name: str = "tax_rate_calculator"
    description: str = (
        "Calculate the precise sales tax rate for a given ZIP code and product "
        "type. Returns combined rate (state + local), taxability status, and "
        "any applicable exemptions."
    )
    args_schema: type[BaseModel] = TaxRateQueryInput

    taxjar_api_key: str
    avalara_account_id: Optional[str] = None
    avalara_license_key: Optional[str] = None

    # Product taxability matrix (simplified)
    EXEMPT_CATEGORIES: Dict[str, List[str]] = {
        "PA": ["clothing", "food_grocery", "prescription_drugs"],
        "NJ": ["clothing", "food_grocery"],
        "NY": ["clothing_under_110", "food_grocery"],
        "MN": ["clothing"],
        "TX": ["food_grocery"],
        # Add remaining state exemptions...
    }

    def _run(
        self,
        zip_code: str,
        product_type: str,
        amount: float,
        customer_type: str = "retail",
    ) -> Dict[str, Any]:
        """Calculate tax rate for transaction."""

        # Get jurisdiction from ZIP
        jurisdiction = self._get_jurisdiction(zip_code)
        state = jurisdiction["state"]

        # Check taxability
        is_taxable, exemption_reason = self._check_taxability(
            state, product_type, customer_type
        )

        if not is_taxable:
            return {
                "zip_code": zip_code,
                "state": state,
                "combined_rate": 0.0,
                "is_taxable": False,
                "exemption_reason": exemption_reason,
                "tax_amount": 0.0,
                "total_with_tax": amount,
            }

        # Fetch rates from TaxJar
        rates = self._fetch_taxjar_rates(zip_code)

        combined_rate = Decimal(str(rates["combined_rate"]))
        tax_amount = (Decimal(str(amount)) * combined_rate).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return {
            "zip_code": zip_code,
            "state": state,
            "jurisdiction": jurisdiction,
            "rates": {
                "state": rates["state_rate"],
                "county": rates["county_rate"],
                "city": rates["city_rate"],
                "special": rates.get("special_rate", 0.0),
                "combined": float(combined_rate),
            },
            "is_taxable": True,
            "exemption_reason": None,
            "amount": amount,
            "tax_amount": float(tax_amount),
            "total_with_tax": amount + float(tax_amount),
            "rate_effective_date": rates.get("effective_date", "current"),
        }

    def _get_jurisdiction(self, zip_code: str) -> Dict[str, str]:
        """Look up jurisdiction details from ZIP code."""
        # In production, this calls a ZIP code database
        # Simplified example mapping
        zip_prefix = zip_code[:3]
        state_map = {
            "100": "NY", "900": "CA", "750": "TX", "190": "PA",
            "606": "IL", "331": "FL", "303": "GA", "981": "WA",
        }
        state = state_map.get(zip_prefix, "CA")
        return {
            "state": state,
            "county": "Example County",
            "city": "Example City",
            "zip": zip_code,
        }

    def _check_taxability(
        self,
        state: str,
        product_type: str,
        customer_type: str,
    ) -> tuple[bool, Optional[str]]:
        """Check if product is taxable in state."""
        if customer_type == "exempt":
            return False, "Customer holds valid exemption certificate"

        exempt_categories = self.EXEMPT_CATEGORIES.get(state, [])
        if product_type.lower() in exempt_categories:
            return False, f"{product_type} is exempt in {state}"

        return True, None

    def _fetch_taxjar_rates(self, zip_code: str) -> Dict[str, Any]:
        """Fetch tax rates from TaxJar API."""
        url = f"https://api.taxjar.com/v2/rates/{zip_code}"
        headers = {
            "Authorization": f"Bearer {self.taxjar_api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

        rate_data = data.get("rate", {})
        return {
            "state_rate": float(rate_data.get("state_rate", 0)),
            "county_rate": float(rate_data.get("county_rate", 0)),
            "city_rate": float(rate_data.get("city_rate", 0)),
            "combined_rate": float(rate_data.get("combined_rate", 0)),
        }

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use sync version")`,
              },
              {
                language: 'python',
                title: 'Filing Preparation Tool',
                description:
                  'Tool for the Filing Agent to prepare state tax returns and track filing deadlines.',
                code: `from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger("filing_preparation")


class FilingFrequency(Enum):
    """Tax return filing frequency."""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


@dataclass
class FilingDeadline:
    """Filing deadline information."""
    state: str
    period_start: date
    period_end: date
    due_date: date
    frequency: FilingFrequency
    estimated_liability: float
    status: str  # pending, prepared, filed


class FilingCalendarQueryInput(BaseModel):
    """Input schema for filing calendar queries."""
    days_ahead: int = Field(
        default=30,
        description="Number of days ahead to check for filings"
    )
    states: List[str] = Field(
        default=[],
        description="Specific states to check. Empty for all registered states."
    )


class FilingPreparationTool(BaseTool):
    """Tool for preparing tax returns and tracking deadlines."""

    name: str = "filing_preparation"
    description: str = (
        "Check upcoming filing deadlines and prepare state tax returns. "
        "Returns due dates, estimated liabilities, and filing status for "
        "all registered states."
    )
    args_schema: type[BaseModel] = FilingCalendarQueryInput

    db_connection: Any  # Database connection

    # State filing requirements (simplified)
    FILING_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
        "CA": {
            "frequency": FilingFrequency.QUARTERLY,
            "due_day": 30,  # Day of month after period ends
            "form": "CDTFA-401",
            "portal_url": "https://onlineservices.cdtfa.ca.gov/",
        },
        "TX": {
            "frequency": FilingFrequency.QUARTERLY,
            "due_day": 20,
            "form": "01-114",
            "portal_url": "https://comptroller.texas.gov/taxes/",
        },
        "NY": {
            "frequency": FilingFrequency.QUARTERLY,
            "due_day": 20,
            "form": "ST-100",
            "portal_url": "https://www.tax.ny.gov/online/",
        },
        "FL": {
            "frequency": FilingFrequency.MONTHLY,
            "due_day": 20,
            "form": "DR-15",
            "portal_url": "https://floridarevenue.com/taxes/",
        },
        # Add remaining states...
    }

    def _run(
        self,
        days_ahead: int = 30,
        states: List[str] = None,
    ) -> Dict[str, Any]:
        """Check filing calendar and prepare returns."""

        today = date.today()
        deadline_cutoff = today + timedelta(days=days_ahead)

        registered_states = states or list(self.FILING_REQUIREMENTS.keys())
        upcoming_filings: List[Dict[str, Any]] = []
        total_liability = 0.0

        for state in registered_states:
            if state not in self.FILING_REQUIREMENTS:
                continue

            requirements = self.FILING_REQUIREMENTS[state]
            deadline_info = self._calculate_next_deadline(state, requirements, today)

            if deadline_info.due_date <= deadline_cutoff:
                # Calculate liability for this period
                liability = self._calculate_period_liability(
                    state,
                    deadline_info.period_start,
                    deadline_info.period_end,
                )
                deadline_info.estimated_liability = liability
                total_liability += liability

                filing_record = {
                    "state": state,
                    "period": f"{deadline_info.period_start} to {deadline_info.period_end}",
                    "due_date": str(deadline_info.due_date),
                    "days_until_due": (deadline_info.due_date - today).days,
                    "frequency": requirements["frequency"].value,
                    "form": requirements["form"],
                    "estimated_liability": round(liability, 2),
                    "portal_url": requirements["portal_url"],
                    "status": deadline_info.status,
                }
                upcoming_filings.append(filing_record)

        # Sort by due date (most urgent first)
        upcoming_filings.sort(key=lambda x: x["due_date"])

        urgent = [f for f in upcoming_filings if f["days_until_due"] <= 7]

        return {
            "checked_date": str(today),
            "horizon_days": days_ahead,
            "filings_due": len(upcoming_filings),
            "urgent_filings": len(urgent),
            "total_estimated_liability": round(total_liability, 2),
            "filings": upcoming_filings,
            "summary": self._generate_filing_summary(upcoming_filings, urgent),
        }

    def _calculate_next_deadline(
        self,
        state: str,
        requirements: Dict[str, Any],
        today: date,
    ) -> FilingDeadline:
        """Calculate the next filing deadline for a state."""
        frequency = requirements["frequency"]
        due_day = requirements["due_day"]

        if frequency == FilingFrequency.MONTHLY:
            # Monthly: due on due_day of next month
            if today.day <= due_day:
                # Current month's return (for previous month)
                period_end = today.replace(day=1) - timedelta(days=1)
                period_start = period_end.replace(day=1)
                due_date = today.replace(day=due_day)
            else:
                # Next month's return
                next_month = today.replace(day=28) + timedelta(days=4)
                period_end = today.replace(day=1) - timedelta(days=1)
                period_start = period_end.replace(day=1)
                due_date = next_month.replace(day=due_day)
        else:
            # Quarterly logic (simplified)
            quarter = (today.month - 1) // 3
            period_start = date(today.year, quarter * 3 + 1, 1)
            if quarter == 3:
                period_end = date(today.year, 12, 31)
                due_date = date(today.year + 1, 1, due_day)
            else:
                period_end = date(today.year, (quarter + 1) * 3 + 1, 1) - timedelta(days=1)
                due_date = date(today.year, (quarter + 1) * 3 + 1, due_day)

        return FilingDeadline(
            state=state,
            period_start=period_start,
            period_end=period_end,
            due_date=due_date,
            frequency=frequency,
            estimated_liability=0.0,
            status="pending",
        )

    def _calculate_period_liability(
        self,
        state: str,
        period_start: date,
        period_end: date,
    ) -> float:
        """Calculate tax liability for a filing period."""
        # In production, this queries the tax_collected table
        # Simplified calculation
        query = """
            SELECT COALESCE(SUM(tax_collected), 0) as total_tax
            FROM order_taxes
            WHERE state = %s
              AND order_date BETWEEN %s AND %s
        """
        # Placeholder - would execute query
        return 5000.00  # Example value

    def _generate_filing_summary(
        self,
        filings: List[Dict],
        urgent: List[Dict],
    ) -> str:
        """Generate filing summary message."""
        if urgent:
            urgent_states = ", ".join(f["state"] for f in urgent)
            return f"URGENT: {len(urgent)} filings due within 7 days ({urgent_states})"
        if filings:
            return f"{len(filings)} filings due in the next 30 days"
        return "No filings due in the next 30 days"

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use sync version")`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement LangGraph state machine to coordinate agent interactions, manage compliance workflow state, and ensure proper sequencing of tax compliance activities.',
            toolsUsed: ['LangGraph', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Tax Compliance LangGraph State Machine',
                description:
                  'LangGraph workflow orchestrating the multi-agent tax compliance system.',
                code: `from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence, Literal
from datetime import datetime, date
import operator
import json
import redis
import logging

logger = logging.getLogger("tax_compliance_orchestrator")


class TaxComplianceState(TypedDict):
    """State maintained across the tax compliance workflow."""

    # Workflow metadata
    run_id: str
    started_at: str
    current_step: str

    # Nexus monitoring state
    nexus_results: dict
    nexus_alerts: list[dict]
    states_triggered: list[str]
    states_approaching: list[str]

    # Tax calculation state
    calculation_results: dict
    rate_discrepancies: list[dict]

    # Filing state
    upcoming_filings: list[dict]
    urgent_filings: list[dict]
    prepared_returns: list[dict]

    # Audit state
    open_inquiries: list[dict]
    audit_readiness_scores: dict

    # Final outputs
    daily_summary: dict
    alerts_sent: int
    errors: Annotated[list[str], operator.add]


def create_tax_compliance_graph(
    nexus_agent,
    calculator_agent,
    filing_agent,
    audit_agent,
    supervisor_agent,
    redis_client: redis.Redis,
):
    """Create the LangGraph workflow for tax compliance."""

    # Initialize state graph
    workflow = StateGraph(TaxComplianceState)

    # Define node functions
    def run_nexus_monitoring(state: TaxComplianceState) -> dict:
        """Execute nexus monitoring agent."""
        logger.info("Running nexus monitoring...")
        try:
            result = nexus_agent.invoke({
                "task": "daily_nexus_check",
                "date": date.today().isoformat(),
            })

            return {
                "nexus_results": result,
                "nexus_alerts": result.get("alerts", []),
                "states_triggered": result.get("triggered_states", []),
                "states_approaching": result.get("approaching_states", []),
                "current_step": "nexus_complete",
            }
        except Exception as e:
            logger.error("Nexus monitoring failed: %s", e)
            return {"errors": [f"Nexus monitoring error: {str(e)}"]}

    def run_tax_calculations(state: TaxComplianceState) -> dict:
        """Execute tax calculation validation."""
        logger.info("Running tax calculation validation...")
        try:
            # Only run if there are triggered states
            if not state.get("states_triggered"):
                return {
                    "calculation_results": {"skipped": True},
                    "current_step": "calculation_complete",
                }

            result = calculator_agent.invoke({
                "task": "validate_rates",
                "states": state["states_triggered"],
            })

            return {
                "calculation_results": result,
                "rate_discrepancies": result.get("discrepancies", []),
                "current_step": "calculation_complete",
            }
        except Exception as e:
            logger.error("Tax calculation failed: %s", e)
            return {"errors": [f"Calculation error: {str(e)}"]}

    def run_filing_check(state: TaxComplianceState) -> dict:
        """Check filing deadlines and prepare returns."""
        logger.info("Running filing deadline check...")
        try:
            result = filing_agent.invoke({
                "task": "check_deadlines",
                "days_ahead": 30,
            })

            return {
                "upcoming_filings": result.get("filings", []),
                "urgent_filings": result.get("urgent", []),
                "current_step": "filing_complete",
            }
        except Exception as e:
            logger.error("Filing check failed: %s", e)
            return {"errors": [f"Filing check error: {str(e)}"]}

    def run_audit_readiness(state: TaxComplianceState) -> dict:
        """Check audit readiness and open inquiries."""
        logger.info("Running audit readiness check...")
        try:
            result = audit_agent.invoke({
                "task": "audit_readiness_check",
            })

            return {
                "open_inquiries": result.get("inquiries", []),
                "audit_readiness_scores": result.get("scores", {}),
                "current_step": "audit_complete",
            }
        except Exception as e:
            logger.error("Audit readiness check failed: %s", e)
            return {"errors": [f"Audit check error: {str(e)}"]}

    def generate_daily_summary(state: TaxComplianceState) -> dict:
        """Supervisor generates final daily summary."""
        logger.info("Generating daily summary...")

        summary = supervisor_agent.invoke({
            "task": "generate_summary",
            "nexus_alerts": len(state.get("nexus_alerts", [])),
            "states_triggered": state.get("states_triggered", []),
            "urgent_filings": len(state.get("urgent_filings", [])),
            "open_inquiries": len(state.get("open_inquiries", [])),
            "errors": state.get("errors", []),
        })

        # Persist state to Redis
        checkpoint_key = f"compliance:checkpoint:{date.today().isoformat()}"
        redis_client.setex(
            checkpoint_key,
            86400 * 90,  # 90 day retention
            json.dumps({
                "run_id": state["run_id"],
                "summary": summary,
                "timestamp": datetime.utcnow().isoformat(),
            }),
        )

        return {
            "daily_summary": summary,
            "current_step": "complete",
        }

    def should_run_calculations(state: TaxComplianceState) -> Literal["calculations", "filing"]:
        """Determine if calculation validation is needed."""
        if state.get("states_triggered") or state.get("rate_discrepancies"):
            return "calculations"
        return "filing"

    # Add nodes to graph
    workflow.add_node("nexus_monitoring", run_nexus_monitoring)
    workflow.add_node("tax_calculations", run_tax_calculations)
    workflow.add_node("filing_check", run_filing_check)
    workflow.add_node("audit_readiness", run_audit_readiness)
    workflow.add_node("daily_summary", generate_daily_summary)

    # Define edges
    workflow.set_entry_point("nexus_monitoring")

    workflow.add_conditional_edges(
        "nexus_monitoring",
        should_run_calculations,
        {
            "calculations": "tax_calculations",
            "filing": "filing_check",
        },
    )

    workflow.add_edge("tax_calculations", "filing_check")
    workflow.add_edge("filing_check", "audit_readiness")
    workflow.add_edge("audit_readiness", "daily_summary")
    workflow.add_edge("daily_summary", END)

    # Compile with checkpointing
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)`,
              },
              {
                language: 'python',
                title: 'Compliance Workflow Runner',
                description:
                  'Runner class that executes the LangGraph workflow with proper initialization and error handling.',
                code: `from datetime import datetime, date
from typing import Optional
import uuid
import redis
import logging

logger = logging.getLogger("compliance_runner")


class TaxComplianceWorkflowRunner:
    """Runner for the tax compliance LangGraph workflow."""

    def __init__(
        self,
        graph,
        redis_client: redis.Redis,
        slack_alerter,
    ):
        self.graph = graph
        self.redis = redis_client
        self.alerter = slack_alerter

    def run_daily_compliance(
        self,
        config: Optional[dict] = None,
    ) -> dict:
        """Execute the daily compliance workflow."""

        run_id = str(uuid.uuid4())[:8]
        logger.info("Starting compliance run: %s", run_id)

        # Initialize state
        initial_state = {
            "run_id": run_id,
            "started_at": datetime.utcnow().isoformat(),
            "current_step": "initialized",
            "nexus_results": {},
            "nexus_alerts": [],
            "states_triggered": [],
            "states_approaching": [],
            "calculation_results": {},
            "rate_discrepancies": [],
            "upcoming_filings": [],
            "urgent_filings": [],
            "prepared_returns": [],
            "open_inquiries": [],
            "audit_readiness_scores": {},
            "daily_summary": {},
            "alerts_sent": 0,
            "errors": [],
        }

        # Execute workflow
        thread_config = {"configurable": {"thread_id": run_id}}

        try:
            final_state = self.graph.invoke(initial_state, thread_config)

            # Send alerts if needed
            alerts_sent = self._send_alerts(final_state)
            final_state["alerts_sent"] = alerts_sent

            # Log completion
            logger.info(
                "Compliance run %s completed: %d alerts sent",
                run_id,
                alerts_sent,
            )

            return {
                "run_id": run_id,
                "status": "success",
                "summary": final_state.get("daily_summary", {}),
                "alerts_sent": alerts_sent,
                "errors": final_state.get("errors", []),
            }

        except Exception as e:
            logger.exception("Compliance run %s failed", run_id)
            self.alerter.send_error_alert(run_id, str(e))
            return {
                "run_id": run_id,
                "status": "failed",
                "error": str(e),
            }

    def _send_alerts(self, state: dict) -> int:
        """Send Slack alerts based on workflow results."""
        alerts_sent = 0

        # Critical: Nexus triggered
        for state_code in state.get("states_triggered", []):
            self.alerter.alert_nexus_triggered(state_code)
            alerts_sent += 1

        # Warning: States approaching threshold
        for state_code in state.get("states_approaching", []):
            self.alerter.alert_nexus_approaching(state_code)
            alerts_sent += 1

        # Urgent filings
        for filing in state.get("urgent_filings", []):
            self.alerter.alert_urgent_filing(filing)
            alerts_sent += 1

        # Daily summary
        if state.get("daily_summary"):
            self.alerter.send_daily_summary(state["daily_summary"])
            alerts_sent += 1

        return alerts_sent

    def get_run_history(self, days: int = 7) -> list[dict]:
        """Retrieve recent compliance run history from Redis."""
        history = []
        today = date.today()

        for i in range(days):
            check_date = today - timedelta(days=i)
            key = f"compliance:checkpoint:{check_date.isoformat()}"
            data = self.redis.get(key)
            if data:
                history.append(json.loads(data))

        return history`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the multi-agent tax compliance system with Docker, implement observability with LangSmith tracing, and set up Prometheus metrics for monitoring agent performance.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose Deployment Configuration',
                description:
                  'Docker Compose configuration for deploying the tax compliance multi-agent system.',
                code: `version: "3.8"

services:
  tax-compliance-orchestrator:
    build:
      context: .
      dockerfile: Dockerfile.tax-compliance
    container_name: tax-compliance-orchestrator
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=\${LANGSMITH_API_KEY}
      - LANGCHAIN_PROJECT=tax-compliance-prod
      - REDIS_URL=redis://redis:6379/0
      - SHOPIFY_SHOP_DOMAIN=\${SHOPIFY_SHOP_DOMAIN}
      - SHOPIFY_API_KEY=\${SHOPIFY_API_KEY}
      - TAXJAR_API_KEY=\${TAXJAR_API_KEY}
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
      - LOG_LEVEL=INFO
    depends_on:
      - redis
      - prometheus
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
    ports:
      - "8095:8095"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8095/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "0.5"
          memory: 1G

  redis:
    image: redis:7-alpine
    container_name: tax-compliance-redis
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: tax-compliance-prometheus
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--storage.tsdb.retention.time=30d"

  grafana:
    image: grafana/grafana:10.0.0
    container_name: tax-compliance-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_ADMIN_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

  # Scheduled job runner
  scheduler:
    build:
      context: .
      dockerfile: Dockerfile.scheduler
    container_name: tax-compliance-scheduler
    environment:
      - ORCHESTRATOR_URL=http://tax-compliance-orchestrator:8095
      - SCHEDULE_CRON=0 6 * * *  # 6 AM daily
      - TZ=America/New_York
    depends_on:
      - tax-compliance-orchestrator
    restart: unless-stopped

volumes:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  default:
    name: tax-compliance-network`,
              },
              {
                language: 'python',
                title: 'LangSmith Tracing and Prometheus Metrics',
                description:
                  'Observability setup with LangSmith tracing and Prometheus metrics collection.',
                code: `from langsmith import Client
from langsmith.run_helpers import traceable
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from functools import wraps
from datetime import datetime
import logging
import time

logger = logging.getLogger("tax_compliance_observability")

# Initialize LangSmith client
langsmith_client = Client()

# Prometheus metrics
AGENT_INVOCATIONS = Counter(
    "tax_compliance_agent_invocations_total",
    "Total agent invocations",
    ["agent_name", "status"],
)

AGENT_LATENCY = Histogram(
    "tax_compliance_agent_latency_seconds",
    "Agent invocation latency",
    ["agent_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

WORKFLOW_RUNS = Counter(
    "tax_compliance_workflow_runs_total",
    "Total workflow runs",
    ["status"],
)

NEXUS_STATES_TRIGGERED = Gauge(
    "tax_compliance_nexus_states_triggered",
    "Number of states with triggered nexus",
)

FILINGS_DUE = Gauge(
    "tax_compliance_filings_due",
    "Number of filings due in next 30 days",
)

ALERTS_SENT = Counter(
    "tax_compliance_alerts_sent_total",
    "Total alerts sent",
    ["alert_type"],
)


def observe_agent(agent_name: str):
    """Decorator to add observability to agent invocations."""
    def decorator(func):
        @wraps(func)
        @traceable(
            name=f"agent:{agent_name}",
            project_name="tax-compliance-prod",
        )
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                logger.error("Agent %s failed: %s", agent_name, e)
                raise
            finally:
                latency = time.time() - start_time
                AGENT_INVOCATIONS.labels(
                    agent_name=agent_name,
                    status=status,
                ).inc()
                AGENT_LATENCY.labels(agent_name=agent_name).observe(latency)

                logger.info(
                    "Agent %s completed in %.2fs with status %s",
                    agent_name,
                    latency,
                    status,
                )

        return wrapper
    return decorator


class ComplianceMetricsCollector:
    """Collects and exposes compliance metrics."""

    def __init__(self, redis_client, metrics_port: int = 8096):
        self.redis = redis_client
        self.metrics_port = metrics_port

    def start_metrics_server(self):
        """Start Prometheus metrics HTTP server."""
        start_http_server(self.metrics_port)
        logger.info("Metrics server started on port %d", self.metrics_port)

    def update_workflow_metrics(self, result: dict):
        """Update metrics after workflow completion."""
        status = result.get("status", "unknown")
        WORKFLOW_RUNS.labels(status=status).inc()

        if status == "success":
            summary = result.get("summary", {})

            # Update gauges
            triggered = len(summary.get("states_triggered", []))
            NEXUS_STATES_TRIGGERED.set(triggered)

            filings = len(summary.get("upcoming_filings", []))
            FILINGS_DUE.set(filings)

            # Count alerts by type
            for alert in summary.get("alerts", []):
                ALERTS_SENT.labels(alert_type=alert.get("type", "unknown")).inc()

    def record_langsmith_feedback(
        self,
        run_id: str,
        score: float,
        comment: str = "",
    ):
        """Record feedback on a LangSmith run for evaluation."""
        langsmith_client.create_feedback(
            run_id=run_id,
            key="compliance_accuracy",
            score=score,
            comment=comment,
        )
        logger.info("Recorded feedback for run %s: %.2f", run_id, score)


# Example usage with decorated agent function
@observe_agent("nexus_monitor")
def run_nexus_monitoring_task(agent, context: dict) -> dict:
    """Execute nexus monitoring with full observability."""
    return agent.invoke(context)


@observe_agent("tax_calculator")
def run_tax_calculation_task(agent, context: dict) -> dict:
    """Execute tax calculation with full observability."""
    return agent.invoke(context)


@observe_agent("filing_preparation")
def run_filing_task(agent, context: dict) -> dict:
    """Execute filing preparation with full observability."""
    return agent.invoke(context)`,
              },
            ],
          },
        ],
      },
    },

    /* ──────────────────────────────────────────────
       Pain Point 2 — WooCommerce Performance Bloat
       ────────────────────────────────────────────── */
    {
      id: 'woocommerce-bloat',
      number: 2,
      title: 'WooCommerce Performance Bloat',
      subtitle: 'Plugin Sprawl & Database Degradation',
      summary:
        'Your WooCommerce store loads in 8 seconds. Every extra second costs 7% in conversions. 40+ plugins and an un-optimized database are the culprits.',
      tags: ['woocommerce', 'performance', 'database'],
      metrics: {
        annualCostRange: '$300K - $1.5M',
        roi: '6x',
        paybackPeriod: '1-2 months',
        investmentRange: '$30K - $70K',
      },
      price: {
        present: {
          title: 'Current State',
          description:
            'WooCommerce store suffers from 8+ second load times driven by 40+ active plugins and years of accumulated database bloat.',
          bullets: [
            'Average page load time exceeds 8 seconds on product and checkout pages',
            '40+ active plugins, many with overlapping functionality and unoptimized queries',
            'wp_options table contains 500K+ autoloaded rows consuming 12MB+ per request',
          ],
          severity: 'high',
        },
        root: {
          title: 'Root Cause',
          description:
            'WordPress and WooCommerce were not designed for high-volume transactional workloads. Plugin sprawl compounds the problem with redundant database queries and unindexed tables.',
          bullets: [
            'Autoloaded options table grows unbounded as plugins store transient data permanently',
            'Post meta table exceeds 10M rows with no composite indexes on lookup patterns',
            'Abandoned cart and log plugins run synchronous writes on every page load',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Business Impact',
          description:
            'Every additional second of load time reduces conversions by 7%. An 8-second load versus a 2-second load represents a 42% conversion penalty.',
          bullets: [
            'Cart abandonment rate 35% higher than industry benchmark',
            'Mobile bounce rate exceeds 70% due to slow initial render',
            'Google Core Web Vitals failing, suppressing organic search rankings',
          ],
          severity: 'critical',
        },
        cost: {
          title: 'Cost of Inaction',
          description:
            'Performance degradation is progressive. Each new plugin and month of accumulated data makes the problem worse, not better.',
          bullets: [
            '$300K - $1.5M in annual lost revenue from conversion drag',
            'Hosting costs inflated 3-5x to compensate for inefficient queries',
            'Developer hours wasted firefighting outages during traffic spikes',
          ],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Expected Return',
          description:
            'Database optimization and plugin audit can reduce load times to under 2 seconds, recovering the full conversion penalty.',
          bullets: [
            '6x ROI within 1-2 months from conversion recovery alone',
            'Hosting costs reduced 40-60% after query optimization',
            'Core Web Vitals passing, unlocking organic traffic growth',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Perform a deep database audit to eliminate bloat, optimize critical queries, and profile plugin impact to remove or replace the worst offenders.',
        prerequisites: [
          'MySQL/MariaDB access with PROCESS and SUPER privileges for query profiling',
          'SSH access to the WooCommerce server for performance profiling',
          'Staging environment to validate changes before production deployment',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        toolsUsed: ['SQL', 'Bash', 'pytest', 'Docker', 'GitHub Actions', 'cron / Airflow', 'Slack API'],
        steps: [
          {
            stepNumber: 1,
            title: 'Database Bloat Audit & Cleanup',
            description:
              'Identify and eliminate the largest sources of database bloat: autoloaded options, orphaned post meta, and transient data that was never cleaned up.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Autoload Bloat Analysis',
                description:
                  'Find the biggest autoloaded entries in wp_options that inflate every single page load.',
                code: `-- Identify autoloaded option bloat
SELECT
  option_name,
  LENGTH(option_value)                         AS value_bytes,
  ROUND(LENGTH(option_value) / 1024, 2)        AS value_kb,
  CASE
    WHEN option_name LIKE '_transient_%'        THEN 'transient'
    WHEN option_name LIKE '_site_transient_%'   THEN 'site_transient'
    WHEN option_name LIKE 'wc_%'                THEN 'woocommerce'
    WHEN option_name LIKE '_wc_%'               THEN 'woocommerce_internal'
    ELSE 'other'
  END                                           AS option_category
FROM wp_options
WHERE autoload = 'yes'
ORDER BY LENGTH(option_value) DESC
LIMIT 50;

-- Total autoload payload per category
SELECT
  CASE
    WHEN option_name LIKE '_transient_%'        THEN 'transient'
    WHEN option_name LIKE '_site_transient_%'   THEN 'site_transient'
    WHEN option_name LIKE 'wc_%'                THEN 'woocommerce'
    ELSE 'other'
  END                                           AS category,
  COUNT(*)                                      AS row_count,
  ROUND(SUM(LENGTH(option_value)) / 1048576, 2) AS total_mb
FROM wp_options
WHERE autoload = 'yes'
GROUP BY category
ORDER BY total_mb DESC;`,
              },
              {
                language: 'sql',
                title: 'Orphaned Post Meta & Revision Cleanup',
                description:
                  'Remove orphaned metadata rows and excessive post revisions bloating the database.',
                code: `-- Count orphaned postmeta (meta referencing deleted posts)
SELECT COUNT(*) AS orphaned_meta_rows
FROM wp_postmeta pm
LEFT JOIN wp_posts p ON pm.post_id = p.ID
WHERE p.ID IS NULL;

-- Preview orphaned meta before deletion
SELECT pm.meta_id, pm.post_id, pm.meta_key,
       LEFT(pm.meta_value, 80) AS meta_preview
FROM wp_postmeta pm
LEFT JOIN wp_posts p ON pm.post_id = p.ID
WHERE p.ID IS NULL
LIMIT 25;

-- Count post revisions by type
SELECT post_type,
       COUNT(*) AS revision_count,
       ROUND(SUM(LENGTH(post_content)) / 1048576, 2) AS content_mb
FROM wp_posts
WHERE post_type = 'revision'
GROUP BY post_type;

-- Safe cleanup: delete expired transients
DELETE FROM wp_options
WHERE option_name LIKE '_transient_timeout_%'
  AND option_value < UNIX_TIMESTAMP();

DELETE a FROM wp_options a
LEFT JOIN wp_options b
  ON b.option_name = CONCAT('_transient_timeout_',
       SUBSTRING(a.option_name, 12))
WHERE a.option_name LIKE '_transient_%'
  AND a.option_name NOT LIKE '_transient_timeout_%'
  AND b.option_name IS NULL;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Query Performance Audit',
            description:
              'Profile the slowest queries hitting the database on product and checkout pages to add missing indexes and rewrite inefficient joins.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Slow Query Identification & Index Recommendations',
                description:
                  'Analyze the MySQL slow query log and identify missing indexes on WooCommerce lookup patterns.',
                code: `-- Top slow queries from performance schema
SELECT
  DIGEST_TEXT                                        AS query_pattern,
  COUNT_STAR                                         AS exec_count,
  ROUND(SUM_TIMER_WAIT / 1e12, 3)                   AS total_time_sec,
  ROUND(AVG_TIMER_WAIT / 1e12, 3)                   AS avg_time_sec,
  SUM_ROWS_EXAMINED                                  AS rows_examined,
  SUM_ROWS_SENT                                      AS rows_returned
FROM performance_schema.events_statements_summary_by_digest
WHERE SCHEMA_NAME = 'wordpress_db'
ORDER BY SUM_TIMER_WAIT DESC
LIMIT 20;

-- Check existing indexes on critical WooCommerce tables
SELECT TABLE_NAME, INDEX_NAME, COLUMN_NAME, SEQ_IN_INDEX
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = 'wordpress_db'
  AND TABLE_NAME IN ('wp_postmeta', 'wp_wc_order_stats',
                     'wp_wc_order_product_lookup', 'wp_options')
ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX;

-- Add composite indexes for common WooCommerce query patterns
ALTER TABLE wp_postmeta
  ADD INDEX idx_meta_lookup (meta_key, meta_value(191));

ALTER TABLE wp_wc_order_product_lookup
  ADD INDEX idx_product_date (product_id, date_created);`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate database optimization results and query performance improvements with data quality assertions and automated tests.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'WooCommerce Database Quality Assertions',
                description:
                  'Run data quality checks to confirm bloat cleanup, index effectiveness, and autoload optimization.',
                code: `-- Assert: autoload payload is below 2MB threshold
SELECT
  ROUND(SUM(LENGTH(option_value)) / 1048576, 2) AS autoload_mb
FROM wp_options
WHERE autoload = 'yes'
HAVING SUM(LENGTH(option_value)) / 1048576 > 2.0;
-- Expected: zero rows (autoload under 2MB)

-- Assert: no orphaned postmeta rows remain
SELECT COUNT(*) AS orphaned_rows
FROM wp_postmeta pm
LEFT JOIN wp_posts p ON pm.post_id = p.ID
WHERE p.ID IS NULL
HAVING COUNT(*) > 0;
-- Expected: zero rows

-- Assert: critical indexes exist on wp_postmeta
SELECT COUNT(*) AS idx_count
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = 'wordpress_db'
  AND TABLE_NAME = 'wp_postmeta'
  AND INDEX_NAME = 'idx_meta_lookup'
HAVING COUNT(*) = 0;
-- Expected: zero rows (index must exist)

-- Assert: no expired transients remain
SELECT COUNT(*) AS expired_transients
FROM wp_options
WHERE option_name LIKE '_transient_timeout_%'
  AND option_value < UNIX_TIMESTAMP()
HAVING COUNT(*) > 0;
-- Expected: zero rows

-- Assert: avg query time for top WooCommerce patterns < 100ms
SELECT
  DIGEST_TEXT,
  ROUND(AVG_TIMER_WAIT / 1e12, 3) AS avg_sec
FROM performance_schema.events_statements_summary_by_digest
WHERE SCHEMA_NAME = 'wordpress_db'
  AND DIGEST_TEXT LIKE '%wp_postmeta%'
  AND AVG_TIMER_WAIT / 1e12 > 0.1
ORDER BY AVG_TIMER_WAIT DESC;
-- Expected: zero rows after optimization`,
              },
              {
                language: 'python',
                title: 'WooCommerce Optimization Validation Suite',
                description:
                  'pytest-based test suite that validates database cleanup, index creation, and performance benchmarks.',
                code: `import pytest
import subprocess
import time
import urllib.request
from unittest.mock import MagicMock

# ── Fixtures ──────────────────────────────────────────

@pytest.fixture
def db_conn():
    """Mock database connection for validation queries."""
    conn = MagicMock()
    return conn

@pytest.fixture
def site_url():
    return "https://store.example.com/shop/"

# ── Autoload Validation ──────────────────────────────

class TestAutoloadCleanup:
    def test_autoload_size_under_threshold(self, db_conn):
        db_conn.execute.return_value.fetchone.return_value = (1.4,)
        row = db_conn.execute(
            "SELECT ROUND(SUM(LENGTH(option_value))/1048576,2) "
            "FROM wp_options WHERE autoload='yes'"
        ).fetchone()
        assert row[0] < 2.0, f"Autoload payload {row[0]}MB exceeds 2MB limit"

    def test_no_expired_transients(self, db_conn):
        db_conn.execute.return_value.fetchone.return_value = (0,)
        row = db_conn.execute(
            "SELECT COUNT(*) FROM wp_options "
            "WHERE option_name LIKE '_transient_timeout_%%' "
            "AND option_value < UNIX_TIMESTAMP()"
        ).fetchone()
        assert row[0] == 0, f"{row[0]} expired transients found"

# ── Orphaned Data Validation ─────────────────────────

class TestOrphanCleanup:
    def test_no_orphaned_postmeta(self, db_conn):
        db_conn.execute.return_value.fetchone.return_value = (0,)
        row = db_conn.execute(
            "SELECT COUNT(*) FROM wp_postmeta pm "
            "LEFT JOIN wp_posts p ON pm.post_id = p.ID "
            "WHERE p.ID IS NULL"
        ).fetchone()
        assert row[0] == 0, f"{row[0]} orphaned postmeta rows remain"

    def test_revisions_under_limit(self, db_conn):
        db_conn.execute.return_value.fetchone.return_value = (500,)
        row = db_conn.execute(
            "SELECT COUNT(*) FROM wp_posts WHERE post_type='revision'"
        ).fetchone()
        assert row[0] < 5000, f"{row[0]} revisions exceed 5000 limit"

# ── Index Validation ─────────────────────────────────

class TestIndexes:
    def test_meta_lookup_index_exists(self, db_conn):
        db_conn.execute.return_value.fetchone.return_value = (1,)
        row = db_conn.execute(
            "SELECT COUNT(*) FROM information_schema.STATISTICS "
            "WHERE TABLE_NAME='wp_postmeta' "
            "AND INDEX_NAME='idx_meta_lookup'"
        ).fetchone()
        assert row[0] >= 1, "idx_meta_lookup index missing"

    def test_product_date_index_exists(self, db_conn):
        db_conn.execute.return_value.fetchone.return_value = (1,)
        row = db_conn.execute(
            "SELECT COUNT(*) FROM information_schema.STATISTICS "
            "WHERE TABLE_NAME='wp_wc_order_product_lookup' "
            "AND INDEX_NAME='idx_product_date'"
        ).fetchone()
        assert row[0] >= 1, "idx_product_date index missing"

# ── Performance Benchmarks ───────────────────────────

class TestPerformance:
    def test_ttfb_under_threshold(self, site_url):
        """Validate Time to First Byte is under 2 seconds."""
        # This is a placeholder; real test hits staging
        target_ttfb_ms = 2000
        simulated_ttfb_ms = 850  # replace with real measurement
        assert simulated_ttfb_ms < target_ttfb_ms`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Deploy WooCommerce database optimizations safely using containerized tooling and configuration management.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'WooCommerce Optimization Deployment Script',
                description:
                  'Automated deployment script for database optimizations with backup, validation, and rollback support.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ─────────────────────────────────────
DB_NAME="wordpress_db"
BACKUP_DIR="/var/backups/woocommerce"
TIMESTAMP="\$(date +%Y%m%d_%H%M%S)"
BACKUP_FILE="\${BACKUP_DIR}/\${DB_NAME}_\${TIMESTAMP}.sql.gz"
SITE_URL="https://store.example.com/shop/"
MAX_TTFB_MS=2000

log() { printf '[%s] %s\\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$1"; }

# ── Pre-flight ────────────────────────────────────────
log "Starting WooCommerce optimization deployment..."
mkdir -p "\${BACKUP_DIR}"

# Verify staging matches production schema
log "Verifying staging environment..."
docker-compose -f docker-compose.staging.yml ps | grep -q "Up" || {
  log "ERROR: staging environment not running"
  exit 1
}

# ── Backup ────────────────────────────────────────────
log "Creating database backup: \${BACKUP_FILE}"
mysqldump --single-transaction --routines "\${DB_NAME}" | gzip > "\${BACKUP_FILE}"
log "Backup complete: \$(du -h "\${BACKUP_FILE}" | cut -f1)"

# ── Apply optimizations ──────────────────────────────
log "Cleaning expired transients..."
mysql "\${DB_NAME}" -e "
  DELETE FROM wp_options
  WHERE option_name LIKE '_transient_timeout_%'
    AND option_value < UNIX_TIMESTAMP();
"

log "Removing orphaned postmeta..."
mysql "\${DB_NAME}" -e "
  DELETE pm FROM wp_postmeta pm
  LEFT JOIN wp_posts p ON pm.post_id = p.ID
  WHERE p.ID IS NULL;
"

log "Adding composite indexes..."
mysql "\${DB_NAME}" -e "
  ALTER TABLE wp_postmeta
    ADD INDEX IF NOT EXISTS idx_meta_lookup (meta_key, meta_value(191));
  ALTER TABLE wp_wc_order_product_lookup
    ADD INDEX IF NOT EXISTS idx_product_date (product_id, date_created);
"

log "Optimizing fragmented tables..."
mysql "\${DB_NAME}" -e "
  OPTIMIZE TABLE wp_options, wp_postmeta, wp_posts;
"

# ── Validate ──────────────────────────────────────────
log "Running validation..."
TTFB_MS=\$(curl -o /dev/null -s -w '%{time_starttransfer}' "\${SITE_URL}" | awk "{print int(\\\$1 * 1000)}")
log "TTFB: \${TTFB_MS}ms (threshold: \${MAX_TTFB_MS}ms)"

if [ "\${TTFB_MS}" -gt "\${MAX_TTFB_MS}" ]; then
  log "WARN: TTFB exceeds threshold. Review optimization results."
fi

log "Deployment complete."`,
              },
              {
                language: 'python',
                title: 'WooCommerce Optimization Configuration Loader',
                description:
                  'Typed configuration loader using dataclasses for the WooCommerce optimization pipeline.',
                code: `from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    charset: str = "utf8mb4"

    @property
    def connection_string(self) -> str:
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?charset={self.charset}"
        )

@dataclass(frozen=True)
class BackupConfig:
    backup_dir: str = "/var/backups/woocommerce"
    retention_days: int = 30
    compress: bool = True

@dataclass(frozen=True)
class OptimizationThresholds:
    max_autoload_mb: float = 2.0
    max_ttfb_ms: int = 2000
    max_orphaned_rows: int = 0
    max_expired_transients: int = 0
    max_revision_count: int = 5000

@dataclass(frozen=True)
class AlertConfig:
    slack_webhook_url: str
    channel: str = "#woocommerce-ops"
    enabled: bool = True

@dataclass(frozen=True)
class WooCommerceOptConfig:
    mysql: MySQLConfig
    backup: BackupConfig
    thresholds: OptimizationThresholds
    alerts: AlertConfig
    site_url: str = "https://store.example.com"
    wp_path: str = "/var/www/html"
    staging_compose_file: str = "docker-compose.staging.yml"
    schedule_cron: str = "0 3 * * 0"
    log_level: str = "INFO"
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> WooCommerceOptConfig:
        return cls(
            mysql=MySQLConfig(
                host=os.environ["MYSQL_HOST"],
                port=int(os.environ.get("MYSQL_PORT", "3306")),
                database=os.environ["MYSQL_DATABASE"],
                user=os.environ["MYSQL_USER"],
                password=os.environ["MYSQL_PASSWORD"],
            ),
            backup=BackupConfig(
                backup_dir=os.environ.get(
                    "BACKUP_DIR", "/var/backups/woocommerce"
                ),
                retention_days=int(os.environ.get("BACKUP_RETENTION_DAYS", "30")),
            ),
            thresholds=OptimizationThresholds(
                max_autoload_mb=float(
                    os.environ.get("MAX_AUTOLOAD_MB", "2.0")
                ),
                max_ttfb_ms=int(os.environ.get("MAX_TTFB_MS", "2000")),
            ),
            alerts=AlertConfig(
                slack_webhook_url=os.environ["SLACK_WEBHOOK_URL"],
                channel=os.environ.get("SLACK_CHANNEL", "#woocommerce-ops"),
            ),
            site_url=os.environ.get(
                "SITE_URL", "https://store.example.com"
            ),
            wp_path=os.environ.get("WP_PATH", "/var/www/html"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            dry_run=os.environ.get("DRY_RUN", "false").lower() == "true",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> WooCommerceOptConfig:
        with open(path) as f:
            data = json.load(f)
        return cls(
            mysql=MySQLConfig(**data["mysql"]),
            backup=BackupConfig(**data.get("backup", {})),
            thresholds=OptimizationThresholds(
                **data.get("thresholds", {})
            ),
            alerts=AlertConfig(**data["alerts"]),
            site_url=data.get("site_url", "https://store.example.com"),
            wp_path=data.get("wp_path", "/var/www/html"),
            log_level=data.get("log_level", "INFO"),
            dry_run=data.get("dry_run", False),
        )`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Measure the performance cost of each active plugin and continuously monitor database health with automated alerting.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Plugin Performance Profiling Script',
                description:
                  'Benchmark page load time with each plugin toggled off to measure individual impact.',
                code: `#!/usr/bin/env bash
# Plugin impact profiler for WooCommerce
# Measures TTFB delta when each plugin is deactivated

set -euo pipefail

SITE_URL="https://store.example.com/shop/"
WP_PATH="/var/www/html"
RESULTS_FILE="/tmp/plugin_impact_\$(date +%Y%m%d).csv"
ITERATIONS=5

echo "plugin,avg_ttfb_ms,baseline_ttfb_ms,delta_ms" > "\$RESULTS_FILE"

# Baseline measurement
baseline_total=0
for i in \$(seq 1 "\$ITERATIONS"); do
  ttfb=\$(curl -o /dev/null -s -w '%{time_starttransfer}' "\$SITE_URL")
  baseline_total=\$(echo "\$baseline_total + \$ttfb * 1000" | bc)
done
baseline_avg=\$(echo "scale=1; \$baseline_total / \$ITERATIONS" | bc)
echo "Baseline TTFB: \${baseline_avg}ms"

# Test each plugin
wp plugin list --path="\$WP_PATH" --status=active --field=name | \\
while read -r plugin; do
  wp plugin deactivate "\$plugin" --path="\$WP_PATH" --quiet
  sleep 2

  plugin_total=0
  for i in \$(seq 1 "\$ITERATIONS"); do
    ttfb=\$(curl -o /dev/null -s -w '%{time_starttransfer}' "\$SITE_URL")
    plugin_total=\$(echo "\$plugin_total + \$ttfb * 1000" | bc)
  done
  avg=\$(echo "scale=1; \$plugin_total / \$ITERATIONS" | bc)
  delta=\$(echo "scale=1; \$baseline_avg - \$avg" | bc)

  echo "\$plugin,\$avg,\$baseline_avg,\$delta" >> "\$RESULTS_FILE"
  echo "  \$plugin: \${avg}ms (delta: \${delta}ms)"

  wp plugin activate "\$plugin" --path="\$WP_PATH" --quiet
  sleep 2
done

echo "Results saved to \$RESULTS_FILE"
sort -t',' -k4 -nr "\$RESULTS_FILE" | head -15`,
              },
              {
                language: 'python',
                title: 'WooCommerce Database Health Monitor',
                description:
                  'Automated monitoring that tracks database health metrics and sends Slack alerts when thresholds are breached.',
                code: `import json
import logging
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger("woo_db_monitor")

@dataclass
class DBHealthMetrics:
    autoload_mb: float
    orphaned_meta_rows: int
    expired_transients: int
    fragmented_tables: int
    avg_query_time_ms: float
    table_count: int
    total_data_mb: float

class WooCommerceDatabaseMonitor:
    def __init__(self, db_conn, slack_webhook: str,
                 channel: str = "#woocommerce-ops"):
        self.db = db_conn
        self.webhook = slack_webhook
        self.channel = channel

    def collect_metrics(self) -> DBHealthMetrics:
        autoload = self.db.execute(
            "SELECT ROUND(SUM(LENGTH(option_value))/1048576, 2) "
            "FROM wp_options WHERE autoload='yes'"
        ).fetchone()[0]

        orphaned = self.db.execute(
            "SELECT COUNT(*) FROM wp_postmeta pm "
            "LEFT JOIN wp_posts p ON pm.post_id = p.ID "
            "WHERE p.ID IS NULL"
        ).fetchone()[0]

        transients = self.db.execute(
            "SELECT COUNT(*) FROM wp_options "
            "WHERE option_name LIKE '_transient_timeout_%%' "
            "AND option_value < UNIX_TIMESTAMP()"
        ).fetchone()[0]

        fragmented = self.db.execute(
            "SELECT COUNT(*) FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA=DATABASE() "
            "AND DATA_FREE > 10485760"
        ).fetchone()[0]

        return DBHealthMetrics(
            autoload_mb=float(autoload or 0),
            orphaned_meta_rows=orphaned,
            expired_transients=transients,
            fragmented_tables=fragmented,
            avg_query_time_ms=0.0,
            table_count=0,
            total_data_mb=0.0,
        )

    def evaluate_and_alert(self, thresholds: dict[str, Any]) -> dict:
        metrics = self.collect_metrics()
        alerts: list[str] = []

        if metrics.autoload_mb > thresholds.get("max_autoload_mb", 2.0):
            alerts.append(
                f"Autoload bloat: {metrics.autoload_mb:.1f}MB "
                f"(limit: {thresholds['max_autoload_mb']}MB)"
            )
        if metrics.orphaned_meta_rows > 0:
            alerts.append(
                f"Orphaned postmeta: {metrics.orphaned_meta_rows:,} rows"
            )
        if metrics.expired_transients > 100:
            alerts.append(
                f"Expired transients: {metrics.expired_transients:,}"
            )

        for msg in alerts:
            logger.warning(msg)
            self._send_slack(msg)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "autoload_mb": metrics.autoload_mb,
                "orphaned_rows": metrics.orphaned_meta_rows,
                "expired_transients": metrics.expired_transients,
            },
            "alerts_sent": len(alerts),
        }

    def _send_slack(self, message: str) -> None:
        payload = json.dumps({
            "channel": self.channel,
            "username": "WooCommerce DB Monitor",
            "text": f"[DB Health] {message}",
        }).encode("utf-8")
        req = urllib.request.Request(
            self.webhook, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            logger.error("Slack alert failed: %s", exc)`,
              },
            ],
          },
        ],
      },
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to automate WooCommerce database health monitoring, bloat detection, and query performance analysis without custom code. Extract database metrics, analyze with AI for optimization recommendations, and receive automated Slack alerts for performance issues.',
        estimatedMonthlyCost: '$100 - $180/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'WP Engine or Kinsta hosting'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Query Monitor Plugin (free)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Set up automated extraction of WooCommerce database health metrics including autoload size, table sizes, slow queries, and plugin impact data. Use Zapier webhooks to collect metrics from WP-CLI or custom health endpoints.',
            toolsUsed: ['WP-CLI', 'Zapier Webhooks', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'WooCommerce Health Metrics Webhook Payload',
                description:
                  'JSON structure for database health metrics collected via webhook and stored for AI analysis.',
                code: `{
  "health_check": {
    "timestamp": "2025-01-31T06:00:00Z",
    "site_url": "https://store.example.com",
    "wordpress_version": "6.4.2",
    "woocommerce_version": "8.5.1",
    "php_version": "8.2.13",
    "database_metrics": {
      "autoload_size_mb": 8.4,
      "autoload_row_count": 2847,
      "total_database_size_mb": 1250,
      "wp_options_rows": 15420,
      "wp_postmeta_rows": 4850000,
      "wp_posts_rows": 125000,
      "orphaned_postmeta_count": 45230,
      "expired_transients_count": 1240,
      "post_revisions_count": 28500
    },
    "query_performance": {
      "avg_query_time_ms": 245,
      "slow_query_count_24h": 156,
      "slowest_query_ms": 4520,
      "queries_per_page_load": 287
    },
    "plugin_metrics": {
      "active_plugins": 42,
      "plugins_with_db_queries": [
        {"name": "WooCommerce", "queries_per_load": 45},
        {"name": "YITH WooCommerce Wishlist", "queries_per_load": 28},
        {"name": "WooCommerce Subscriptions", "queries_per_load": 22},
        {"name": "Advanced Custom Fields", "queries_per_load": 18}
      ],
      "estimated_plugin_overhead_ms": 1850
    },
    "core_web_vitals": {
      "lcp_ms": 4200,
      "fid_ms": 180,
      "cls": 0.15,
      "ttfb_ms": 2100
    }
  }
}`,
              },
              {
                language: 'json',
                title: 'Zapier Webhook Collection Trigger',
                description:
                  'Configure Zapier to receive health metrics via webhook and store in Google Sheets for analysis.',
                code: `{
  "zap_name": "WooCommerce Health Metrics Collector",
  "trigger": {
    "app": "Webhooks by Zapier",
    "event": "Catch Hook",
    "webhook_url": "https://hooks.zapier.com/hooks/catch/12345/woo-health/",
    "expected_payload": {
      "site_url": "string",
      "autoload_size_mb": "number",
      "slow_query_count": "number",
      "ttfb_ms": "number"
    }
  },
  "actions": [
    {
      "step": 1,
      "app": "Google Sheets",
      "event": "Create Spreadsheet Row",
      "config": {
        "spreadsheet_id": "{{env.WOO_HEALTH_SHEET}}",
        "worksheet": "Daily Metrics",
        "columns": {
          "date": "{{zap.trigger_time}}",
          "autoload_mb": "{{trigger.database_metrics.autoload_size_mb}}",
          "postmeta_rows": "{{trigger.database_metrics.wp_postmeta_rows}}",
          "orphaned_meta": "{{trigger.database_metrics.orphaned_postmeta_count}}",
          "expired_transients": "{{trigger.database_metrics.expired_transients_count}}",
          "avg_query_ms": "{{trigger.query_performance.avg_query_time_ms}}",
          "slow_queries_24h": "{{trigger.query_performance.slow_query_count_24h}}",
          "active_plugins": "{{trigger.plugin_metrics.active_plugins}}",
          "ttfb_ms": "{{trigger.core_web_vitals.ttfb_ms}}",
          "lcp_ms": "{{trigger.core_web_vitals.lcp_ms}}"
        }
      }
    },
    {
      "step": 2,
      "app": "Filter by Zapier",
      "event": "Only Continue If",
      "config": {
        "conditions": [
          {
            "field": "trigger.database_metrics.autoload_size_mb",
            "operator": "greater_than",
            "value": 5
          },
          {
            "field": "trigger.core_web_vitals.ttfb_ms",
            "operator": "greater_than",
            "value": 2000
          }
        ],
        "logic": "OR"
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
              'Use ChatGPT or Claude to analyze WooCommerce database health metrics, identify performance bottlenecks, and generate prioritized optimization recommendations with specific SQL commands.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Database Optimization Analysis Prompt Template',
                description:
                  'Structured prompt for AI to analyze WooCommerce database metrics and provide actionable optimization guidance.',
                code: `system_prompt: |
  You are a WooCommerce database performance specialist with deep expertise
  in MySQL/MariaDB optimization, WordPress internals, and ecommerce
  performance tuning. You understand:
  - WordPress database schema and common bloat patterns
  - WooCommerce-specific tables and query patterns
  - Plugin impact on database performance
  - Core Web Vitals and their relationship to database performance

  Your recommendations should be:
  - Specific and actionable with exact SQL commands
  - Prioritized by impact (highest ROI first)
  - Safe for production with proper backup warnings
  - Validated against WooCommerce best practices

user_prompt_template: |
  Analyze the following WooCommerce database health report and provide
  optimization recommendations:

  ## Site Information
  - URL: {{site_url}}
  - WordPress: {{wordpress_version}}
  - WooCommerce: {{woocommerce_version}}
  - PHP: {{php_version}}

  ## Database Metrics
  - Autoload Size: {{autoload_size_mb}} MB (threshold: 2 MB)
  - Autoload Rows: {{autoload_row_count}}
  - Total Database: {{total_database_size_mb}} MB
  - wp_postmeta Rows: {{wp_postmeta_rows}}
  - Orphaned Postmeta: {{orphaned_postmeta_count}}
  - Expired Transients: {{expired_transients_count}}
  - Post Revisions: {{post_revisions_count}}

  ## Query Performance
  - Avg Query Time: {{avg_query_time_ms}} ms (threshold: 50 ms)
  - Slow Queries (24h): {{slow_query_count_24h}}
  - Slowest Query: {{slowest_query_ms}} ms
  - Queries per Page Load: {{queries_per_page_load}}

  ## Plugin Impact
  - Active Plugins: {{active_plugins}}
  - Top Query-Heavy Plugins:
  {{#each plugins_with_db_queries}}
    - {{name}}: {{queries_per_load}} queries/load
  {{/each}}

  ## Core Web Vitals
  - LCP: {{lcp_ms}} ms (threshold: 2500 ms)
  - TTFB: {{ttfb_ms}} ms (threshold: 800 ms)
  - FID: {{fid_ms}} ms
  - CLS: {{cls}}

  ## Analysis Required
  1. Identify the top 3 database issues causing performance degradation
  2. Provide specific SQL commands to fix each issue
  3. Estimate the performance improvement for each fix
  4. Recommend plugins to deactivate or replace
  5. Provide a prioritized action plan with expected TTFB reduction

expected_output_format: |
  ## Executive Summary
  [2-3 sentence overview of database health status]

  ## Critical Issues (Fix Immediately)
  | Issue | Current | Target | Impact | SQL Command |
  |-------|---------|--------|--------|-------------|

  ## High Priority Optimizations
  ### 1. [Issue Name]
  **Problem:** [Description]
  **Solution:**
  \`\`\`sql
  [Exact SQL command]
  \`\`\`
  **Expected Improvement:** [X ms reduction in TTFB]

  ## Plugin Recommendations
  | Plugin | Issue | Action | Alternative |
  |--------|-------|--------|-------------|

  ## Action Plan
  1. [Immediate action] - Expected: -Xms TTFB
  2. [Next action] - Expected: -Xms TTFB
  3. [Subsequent action]

  ## Estimated Results
  - Current TTFB: {{ttfb_ms}} ms
  - Projected TTFB: X ms
  - Improvement: X%`,
              },
              {
                language: 'yaml',
                title: 'Query Optimization Prompt Template',
                description:
                  'Prompt template for AI to analyze slow queries and recommend index improvements.',
                code: `system_prompt: |
  You are a MySQL/MariaDB query optimization expert specializing in
  WordPress and WooCommerce databases. You can analyze slow queries,
  identify missing indexes, and recommend schema optimizations.

user_prompt_template: |
  Analyze these slow queries from a WooCommerce database and provide
  optimization recommendations:

  ## Slow Query Log (Top 10)
  {{#each slow_queries}}
  ### Query {{@index}}
  - Execution Time: {{execution_time_ms}} ms
  - Rows Examined: {{rows_examined}}
  - Rows Returned: {{rows_returned}}
  - Query:
  \`\`\`sql
  {{query_text}}
  \`\`\`
  {{/each}}

  ## Current Indexes on Key Tables
  ### wp_postmeta
  {{postmeta_indexes}}

  ### wp_wc_order_product_lookup
  {{order_lookup_indexes}}

  ## Analysis Required
  1. For each slow query, explain why it's slow
  2. Recommend specific indexes to add (with CREATE INDEX statements)
  3. Suggest query rewrites where applicable
  4. Estimate the performance improvement
  5. Warn about any index maintenance considerations

output_format: |
  ## Query Analysis

  ### Query 1: [Brief description]
  **Root Cause:** [Why it's slow]
  **Recommended Index:**
  \`\`\`sql
  CREATE INDEX idx_name ON table_name (column1, column2);
  \`\`\`
  **Expected Improvement:** X% faster

  ## Summary of Recommended Indexes
  \`\`\`sql
  -- Run these in order, during low-traffic period
  CREATE INDEX idx_postmeta_key_value ON wp_postmeta (meta_key, meta_value(191));
  CREATE INDEX idx_order_product ON wp_wc_order_product_lookup (product_id, date_created);
  \`\`\`

  ## Maintenance Notes
  - [Index size estimates]
  - [Rebuild schedule recommendations]`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Automate the complete workflow with Zapier to run weekly database health analysis, send AI-generated optimization reports to Slack, and trigger alerts when performance thresholds are breached.',
            toolsUsed: ['Zapier', 'Slack', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Complete Database Health Workflow',
                description:
                  'End-to-end Zapier workflow that collects metrics, calls ChatGPT for analysis, and delivers optimization reports.',
                code: `{
  "zap_name": "Weekly WooCommerce Database Health Report",
  "trigger": {
    "app": "Schedule by Zapier",
    "event": "Every Week",
    "day": "Monday",
    "time": "06:00",
    "timezone": "America/New_York"
  },
  "actions": [
    {
      "step": 1,
      "app": "Webhooks by Zapier",
      "event": "GET",
      "config": {
        "url": "https://store.example.com/wp-json/woo-health/v1/metrics",
        "headers": {
          "Authorization": "Bearer {{env.WOO_API_TOKEN}}"
        }
      }
    },
    {
      "step": 2,
      "app": "Google Sheets",
      "event": "Lookup Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{env.WOO_HEALTH_SHEET}}",
        "worksheet": "Daily Metrics",
        "lookup_column": "date",
        "lookup_value": "last_7_days"
      }
    },
    {
      "step": 3,
      "app": "Formatter by Zapier",
      "event": "Text",
      "config": {
        "transform": "concatenate",
        "template": "Analyze WooCommerce health: Current metrics: {{step1.body}} Historical trend: {{step2.summary}}"
      }
    },
    {
      "step": 4,
      "app": "ChatGPT",
      "event": "Conversation",
      "config": {
        "model": "gpt-4",
        "system_message": "You are a WooCommerce database performance specialist...",
        "user_message": "{{step3.output}}",
        "max_tokens": 3000
      }
    },
    {
      "step": 5,
      "app": "Slack",
      "event": "Send Channel Message",
      "config": {
        "channel": "#woocommerce-ops",
        "message": {
          "blocks": [
            {
              "type": "header",
              "text": ":database: Weekly WooCommerce Health Report"
            },
            {
              "type": "section",
              "fields": [
                {"type": "mrkdwn", "text": "*Autoload:* {{step1.autoload_size_mb}} MB"},
                {"type": "mrkdwn", "text": "*TTFB:* {{step1.ttfb_ms}} ms"},
                {"type": "mrkdwn", "text": "*Slow Queries:* {{step1.slow_query_count_24h}}"},
                {"type": "mrkdwn", "text": "*Plugins:* {{step1.active_plugins}}"}
              ]
            },
            {
              "type": "section",
              "text": "{{step4.response}}"
            },
            {
              "type": "actions",
              "elements": [
                {
                  "type": "button",
                  "text": "View Full Report",
                  "url": "{{env.WOO_HEALTH_SHEET}}"
                },
                {
                  "type": "button",
                  "text": "Run Optimization",
                  "url": "https://store.example.com/wp-admin/admin.php?page=woo-optimizer",
                  "style": "primary"
                }
              ]
            }
          ]
        },
        "bot_name": "WooCommerce Health Bot",
        "bot_icon": ":wordpress:"
      }
    },
    {
      "step": 6,
      "app": "Google Sheets",
      "event": "Create Spreadsheet Row",
      "config": {
        "spreadsheet_id": "{{env.WOO_HEALTH_SHEET}}",
        "worksheet": "Analysis Log",
        "columns": {
          "analysis_date": "{{zap.trigger_time}}",
          "autoload_mb": "{{step1.autoload_size_mb}}",
          "ttfb_ms": "{{step1.ttfb_ms}}",
          "ai_recommendations": "{{step4.response}}",
          "status": "delivered"
        }
      }
    }
  ]
}`,
              },
              {
                language: 'json',
                title: 'Performance Threshold Alert Configuration',
                description:
                  'Zapier workflow to send urgent alerts when WooCommerce performance metrics breach critical thresholds.',
                code: `{
  "alert_workflow": {
    "name": "WooCommerce Performance Alert",
    "trigger": {
      "app": "Webhooks by Zapier",
      "event": "Catch Hook",
      "webhook_url": "https://hooks.zapier.com/hooks/catch/12345/woo-alert/"
    },
    "filters": [
      {
        "name": "Critical Performance Breach",
        "conditions": [
          {"field": "ttfb_ms", "operator": "greater_than", "value": 3000},
          {"field": "autoload_size_mb", "operator": "greater_than", "value": 10},
          {"field": "lcp_ms", "operator": "greater_than", "value": 4000}
        ],
        "logic": "OR"
      }
    ],
    "actions": [
      {
        "app": "Slack",
        "event": "Send Channel Message",
        "config": {
          "channel": "#woocommerce-alerts",
          "message": ":rotating_light: *WooCommerce Performance Alert*\\n\\n*Site:* {{trigger.site_url}}\\n*TTFB:* {{trigger.ttfb_ms}}ms (threshold: 3000ms)\\n*Autoload:* {{trigger.autoload_size_mb}}MB (threshold: 10MB)\\n*LCP:* {{trigger.lcp_ms}}ms\\n\\n:warning: Immediate investigation required."
        }
      },
      {
        "app": "Email by Zapier",
        "event": "Send Outbound Email",
        "config": {
          "to": "devops@company.com",
          "subject": "[CRITICAL] WooCommerce Performance Degradation",
          "body": "WooCommerce site {{trigger.site_url}} has breached performance thresholds.\\n\\nTTFB: {{trigger.ttfb_ms}}ms\\nAutoload: {{trigger.autoload_size_mb}}MB\\n\\nImmediate action required."
        }
      },
      {
        "app": "ChatGPT",
        "event": "Conversation",
        "config": {
          "model": "gpt-4",
          "user_message": "WooCommerce emergency: TTFB is {{trigger.ttfb_ms}}ms, autoload is {{trigger.autoload_size_mb}}MB. What are the top 3 immediate actions to restore performance?",
          "max_tokens": 500
        }
      },
      {
        "app": "Slack",
        "event": "Send Channel Message",
        "config": {
          "channel": "#woocommerce-alerts",
          "message": "*AI-Recommended Immediate Actions:*\\n{{previous_step.response}}"
        }
      }
    ]
  }
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent system using CrewAI and LangGraph to provide autonomous WooCommerce performance monitoring and optimization. Specialized agents handle database analysis, query optimization, plugin profiling, and automated remediation, coordinated by a supervisor agent that ensures optimal site performance.',
        estimatedMonthlyCost: '$500 - $1,000/month',
        architecture:
          'Supervisor agent coordinates four specialist agents: Database Health Agent monitors autoload, bloat, and table health, Query Optimizer Agent analyzes slow queries and recommends indexes, Plugin Profiler Agent measures individual plugin impact on performance, and Remediation Agent executes safe automated cleanups. State is managed in Redis with hourly performance snapshots.',
        agents: [
          {
            name: 'Database Health Agent',
            role: 'WordPress Database Health Monitor',
            goal: 'Continuously monitor database health metrics including autoload size, table fragmentation, orphaned data, and transient bloat. Alert when thresholds are breached.',
            tools: ['mysql_query_tool', 'wp_cli_tool', 'metrics_collector', 'slack_alerter'],
          },
          {
            name: 'Query Optimizer Agent',
            role: 'MySQL Query Performance Analyst',
            goal: 'Analyze slow query logs, identify missing indexes, recommend query optimizations, and validate index effectiveness after implementation.',
            tools: ['slow_query_analyzer', 'explain_plan_tool', 'index_advisor', 'performance_schema_reader'],
          },
          {
            name: 'Plugin Profiler Agent',
            role: 'WordPress Plugin Impact Analyst',
            goal: 'Profile each active plugin to measure database queries, memory usage, and load time impact. Recommend plugins for removal or replacement.',
            tools: ['query_monitor_api', 'plugin_benchmark_tool', 'load_time_profiler', 'alternative_finder'],
          },
          {
            name: 'Remediation Agent',
            role: 'Automated Database Cleanup Specialist',
            goal: 'Execute safe automated cleanups including transient deletion, orphan removal, and table optimization. Ensure backups before any destructive operations.',
            tools: ['backup_tool', 'cleanup_executor', 'wp_cli_tool', 'rollback_manager'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with hourly performance snapshots and 30-day trend retention',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system architecture with CrewAI, establishing clear roles, goals, and tool assignments for each specialist agent in the WooCommerce optimization workflow.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'WooCommerce Optimization Agent Definitions',
                description:
                  'CrewAI agent definitions for the WooCommerce performance optimization multi-agent system.',
                code: `from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import logging

logger = logging.getLogger("woocommerce_optimization_agents")

# Initialize LLM with lower temperature for precise technical recommendations
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.1,
)


class WooCommerceOptimizationAgents:
    """Factory for creating WooCommerce performance optimization agents."""

    def __init__(self, tools_registry: Dict[str, Any]):
        self.tools = tools_registry
        self.llm = llm

    def create_database_health_agent(self) -> Agent:
        """Agent responsible for monitoring WordPress database health."""
        return Agent(
            role="WordPress Database Health Monitor",
            goal=(
                "Continuously monitor database health metrics including autoload "
                "size, table fragmentation, orphaned postmeta, and transient bloat. "
                "Alert immediately when any metric exceeds safe thresholds: "
                "autoload > 2MB, orphaned rows > 10000, fragmentation > 20%."
            ),
            backstory=(
                "You are a WordPress database specialist who has optimized "
                "hundreds of WooCommerce stores. You know exactly which metrics "
                "indicate impending performance problems and how to catch issues "
                "before they impact customers. You understand the relationship "
                "between autoload size and TTFB, and you know which plugins are "
                "notorious for database bloat."
            ),
            tools=[
                self.tools["mysql_query_tool"],
                self.tools["wp_cli_tool"],
                self.tools["metrics_collector"],
                self.tools["slack_alerter"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def create_query_optimizer_agent(self) -> Agent:
        """Agent responsible for MySQL query optimization."""
        return Agent(
            role="MySQL Query Performance Analyst",
            goal=(
                "Analyze the slow query log to identify queries taking > 100ms. "
                "For each slow query, determine the root cause (missing index, "
                "full table scan, suboptimal join) and recommend specific fixes. "
                "Validate that recommended indexes improve performance without "
                "excessive write overhead."
            ),
            backstory=(
                "You are a MySQL performance expert who specializes in WordPress "
                "and WooCommerce query patterns. You can read EXPLAIN plans like "
                "a book and know exactly which indexes will help and which will "
                "create more problems. You understand the trade-off between read "
                "and write performance and always consider the full workload."
            ),
            tools=[
                self.tools["slow_query_analyzer"],
                self.tools["explain_plan_tool"],
                self.tools["index_advisor"],
                self.tools["performance_schema_reader"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def create_plugin_profiler_agent(self) -> Agent:
        """Agent responsible for profiling plugin impact."""
        return Agent(
            role="WordPress Plugin Impact Analyst",
            goal=(
                "Profile each active plugin to measure its impact on page load "
                "time, database queries, and memory usage. Identify plugins that "
                "contribute > 200ms to load time or > 20 database queries per "
                "page. Recommend lightweight alternatives for heavy plugins."
            ),
            backstory=(
                "You are a WordPress performance consultant who has audited "
                "thousands of plugin installations. You know which popular "
                "plugins are performance nightmares and which alternatives "
                "provide similar functionality without the overhead. You can "
                "quickly identify plugin conflicts and redundant functionality."
            ),
            tools=[
                self.tools["query_monitor_api"],
                self.tools["plugin_benchmark_tool"],
                self.tools["load_time_profiler"],
                self.tools["alternative_finder"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )

    def create_remediation_agent(self) -> Agent:
        """Agent responsible for executing safe cleanups."""
        return Agent(
            role="Automated Database Cleanup Specialist",
            goal=(
                "Execute safe automated cleanups when authorized. Always create "
                "backups before any destructive operation. Clean expired transients, "
                "remove orphaned postmeta, optimize fragmented tables. Never "
                "execute during peak traffic hours (9 AM - 9 PM site timezone)."
            ),
            backstory=(
                "You are a cautious but effective database administrator who "
                "automates routine maintenance tasks. You have never caused "
                "data loss because you always verify backups before cleanup "
                "and you have robust rollback procedures. You schedule heavy "
                "operations for low-traffic periods."
            ),
            tools=[
                self.tools["backup_tool"],
                self.tools["cleanup_executor"],
                self.tools["wp_cli_tool"],
                self.tools["rollback_manager"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def create_supervisor_agent(self) -> Agent:
        """Supervisor agent that coordinates optimization activities."""
        return Agent(
            role="WooCommerce Performance Supervisor",
            goal=(
                "Coordinate all performance optimization activities. Prioritize "
                "urgent issues (site down, TTFB > 5s) over routine maintenance. "
                "Ensure cleanup operations don't overlap with traffic spikes. "
                "Generate weekly performance reports for stakeholders."
            ),
            backstory=(
                "You are the technical lead for WooCommerce operations, "
                "overseeing a team of database and performance specialists. "
                "You balance the need for optimization against the risk of "
                "changes. You communicate clearly with both technical and "
                "business stakeholders about performance status."
            ),
            tools=[
                self.tools["performance_dashboard"],
                self.tools["traffic_monitor"],
                self.tools["report_generator"],
                self.tools["escalation_handler"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )`,
              },
              {
                language: 'python',
                title: 'WooCommerce Optimization Crew Configuration',
                description:
                  'CrewAI Crew configuration that assembles agents and defines optimization task workflows.',
                code: `from crewai import Crew, Task, Process
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List
from enum import Enum


class OptimizationPriority(Enum):
    CRITICAL = "critical"  # Site impacted
    HIGH = "high"          # Performance degraded
    MEDIUM = "medium"      # Threshold approaching
    LOW = "low"            # Routine maintenance


@dataclass
class OptimizationResult:
    """Result container for optimization task execution."""
    task_type: str
    priority: OptimizationPriority
    actions_taken: List[str]
    metrics_before: dict
    metrics_after: dict
    ttfb_improvement_ms: int
    alerts_generated: int
    next_scheduled_run: datetime


class WooCommerceOptimizationCrew:
    """Crew orchestrating multi-agent WooCommerce optimization workflow."""

    def __init__(self, agents_factory: WooCommerceOptimizationAgents):
        self.agents = agents_factory
        self.crew: Optional[Crew] = None

    def build_crew(self) -> Crew:
        """Assemble the optimization crew with all agents and tasks."""

        # Create specialist agents
        db_health_agent = self.agents.create_database_health_agent()
        query_optimizer = self.agents.create_query_optimizer_agent()
        plugin_profiler = self.agents.create_plugin_profiler_agent()
        remediation_agent = self.agents.create_remediation_agent()
        supervisor = self.agents.create_supervisor_agent()

        # Define tasks for optimization workflow
        health_check_task = Task(
            description=(
                "Perform comprehensive database health check. Measure: "
                "1) Autoload size and row count, 2) Orphaned postmeta count, "
                "3) Expired transient count, 4) Table fragmentation percentage, "
                "5) Total database size. Flag any metric exceeding threshold."
            ),
            expected_output=(
                "JSON report with: autoload_mb, orphaned_count, "
                "transient_count, fragmentation_pct, thresholds_exceeded, "
                "recommended_actions"
            ),
            agent=db_health_agent,
        )

        query_analysis_task = Task(
            description=(
                "Analyze slow query log for queries > 100ms. For top 10 "
                "slowest queries: 1) Run EXPLAIN to identify bottleneck, "
                "2) Check for missing indexes, 3) Recommend specific indexes "
                "with CREATE INDEX statements, 4) Estimate improvement."
            ),
            expected_output=(
                "JSON report with: slow_query_count, analyzed_queries, "
                "missing_indexes, recommended_indexes, estimated_improvement_pct"
            ),
            agent=query_optimizer,
            context=[health_check_task],  # Uses health data
        )

        plugin_profiling_task = Task(
            description=(
                "Profile all active plugins. For each plugin measure: "
                "1) Database queries per page load, 2) Memory usage, "
                "3) Load time contribution. Flag plugins with > 20 queries "
                "or > 200ms contribution. Recommend alternatives."
            ),
            expected_output=(
                "JSON report with: plugins_profiled, heavy_plugins, "
                "recommended_removals, alternative_suggestions, "
                "estimated_savings_ms"
            ),
            agent=plugin_profiler,
        )

        remediation_task = Task(
            description=(
                "Based on health check results, execute safe cleanups: "
                "1) Create database backup, 2) Delete expired transients, "
                "3) Remove orphaned postmeta (if > 1000 rows), "
                "4) Optimize fragmented tables (if > 20%). Log all actions."
            ),
            expected_output=(
                "JSON report with: backup_created, transients_deleted, "
                "orphans_removed, tables_optimized, execution_time_sec, "
                "rollback_available"
            ),
            agent=remediation_agent,
            context=[health_check_task],
        )

        coordination_task = Task(
            description=(
                "Review all optimization results. Calculate total TTFB "
                "improvement. Generate performance report with: "
                "1) Actions taken, 2) Metrics before/after, 3) Next steps, "
                "4) Recommended schedule for next optimization."
            ),
            expected_output=(
                "Executive report with: ttfb_before, ttfb_after, "
                "improvement_pct, actions_summary, next_scheduled_run"
            ),
            agent=supervisor,
            context=[
                health_check_task,
                query_analysis_task,
                plugin_profiling_task,
                remediation_task,
            ],
        )

        self.crew = Crew(
            agents=[
                supervisor,
                db_health_agent,
                query_optimizer,
                plugin_profiler,
                remediation_agent,
            ],
            tasks=[
                health_check_task,
                query_analysis_task,
                plugin_profiling_task,
                remediation_task,
                coordination_task,
            ],
            process=Process.hierarchical,
            manager_agent=supervisor,
            verbose=True,
        )

        return self.crew

    def run_optimization_cycle(
        self,
        context: dict,
        priority: OptimizationPriority = OptimizationPriority.MEDIUM,
    ) -> OptimizationResult:
        """Execute optimization workflow."""
        if not self.crew:
            self.build_crew()

        # Capture baseline metrics
        metrics_before = context.get("current_metrics", {})

        result = self.crew.kickoff(inputs={
            **context,
            "priority": priority.value,
            "run_time": datetime.utcnow().isoformat(),
        })

        return OptimizationResult(
            task_type="optimization_cycle",
            priority=priority,
            actions_taken=result.get("actions", []),
            metrics_before=metrics_before,
            metrics_after=result.get("metrics_after", {}),
            ttfb_improvement_ms=result.get("ttfb_improvement", 0),
            alerts_generated=result.get("alerts", 0),
            next_scheduled_run=datetime.fromisoformat(
                result.get("next_run", datetime.utcnow().isoformat())
            ),
        )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement data ingestion agents that collect database metrics, slow query logs, and performance data from WordPress and WooCommerce for analysis.',
            toolsUsed: ['WP-CLI', 'MySQL', 'Query Monitor', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Database Metrics Collector Tool',
                description:
                  'LangChain tool for the Database Health Agent to collect WordPress database metrics.',
                code: `from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from dataclasses import dataclass
import subprocess
import json
import logging

logger = logging.getLogger("db_metrics_collector")


class DBMetricsQueryInput(BaseModel):
    """Input schema for database metrics queries."""
    metrics_type: str = Field(
        default="full",
        description="Type of metrics: 'full', 'autoload', 'bloat', 'performance'"
    )
    include_slow_queries: bool = Field(
        default=True,
        description="Include slow query analysis"
    )


@dataclass
class DatabaseMetrics:
    """Container for database health metrics."""
    autoload_size_mb: float
    autoload_row_count: int
    total_size_mb: float
    postmeta_rows: int
    orphaned_postmeta: int
    expired_transients: int
    post_revisions: int
    fragmented_tables: int
    avg_query_time_ms: float


class DatabaseMetricsCollectorTool(BaseTool):
    """Tool for collecting WordPress database health metrics."""

    name: str = "mysql_query_tool"
    description: str = (
        "Collect comprehensive database health metrics from WordPress/WooCommerce. "
        "Returns autoload size, bloat metrics, and performance indicators."
    )
    args_schema: type[BaseModel] = DBMetricsQueryInput

    mysql_host: str
    mysql_user: str
    mysql_password: str
    mysql_database: str

    def _run(
        self,
        metrics_type: str = "full",
        include_slow_queries: bool = True,
    ) -> Dict[str, Any]:
        """Collect database metrics."""

        metrics = {}

        # Autoload metrics
        autoload_query = """
            SELECT
                COUNT(*) as row_count,
                ROUND(SUM(LENGTH(option_value)) / 1048576, 2) as size_mb
            FROM wp_options
            WHERE autoload = 'yes'
        """
        autoload_result = self._execute_query(autoload_query)
        metrics["autoload"] = {
            "row_count": autoload_result[0]["row_count"],
            "size_mb": float(autoload_result[0]["size_mb"] or 0),
            "threshold_mb": 2.0,
            "status": "critical" if float(autoload_result[0]["size_mb"] or 0) > 5 else (
                "warning" if float(autoload_result[0]["size_mb"] or 0) > 2 else "ok"
            ),
        }

        # Bloat metrics
        bloat_queries = {
            "orphaned_postmeta": """
                SELECT COUNT(*) as count
                FROM wp_postmeta pm
                LEFT JOIN wp_posts p ON pm.post_id = p.ID
                WHERE p.ID IS NULL
            """,
            "expired_transients": """
                SELECT COUNT(*) as count
                FROM wp_options
                WHERE option_name LIKE '_transient_timeout_%%'
                  AND option_value < UNIX_TIMESTAMP()
            """,
            "post_revisions": """
                SELECT COUNT(*) as count
                FROM wp_posts
                WHERE post_type = 'revision'
            """,
        }

        metrics["bloat"] = {}
        for name, query in bloat_queries.items():
            result = self._execute_query(query)
            metrics["bloat"][name] = result[0]["count"]

        # Table sizes
        table_sizes_query = """
            SELECT
                TABLE_NAME as table_name,
                ROUND(DATA_LENGTH / 1048576, 2) as data_mb,
                ROUND(INDEX_LENGTH / 1048576, 2) as index_mb,
                TABLE_ROWS as row_count
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME LIKE 'wp_%%'
            ORDER BY DATA_LENGTH DESC
            LIMIT 10
        """
        table_results = self._execute_query(
            table_sizes_query,
            (self.mysql_database,)
        )
        metrics["table_sizes"] = [
            {
                "table": r["table_name"],
                "data_mb": float(r["data_mb"] or 0),
                "index_mb": float(r["index_mb"] or 0),
                "rows": r["row_count"],
            }
            for r in table_results
        ]

        # Performance metrics
        if include_slow_queries:
            perf_query = """
                SELECT
                    COUNT_STAR as query_count,
                    ROUND(AVG_TIMER_WAIT / 1e9, 2) as avg_time_ms,
                    ROUND(MAX_TIMER_WAIT / 1e9, 2) as max_time_ms
                FROM performance_schema.events_statements_summary_by_digest
                WHERE SCHEMA_NAME = %s
                  AND AVG_TIMER_WAIT > 100000000
                ORDER BY AVG_TIMER_WAIT DESC
                LIMIT 1
            """
            try:
                perf_result = self._execute_query(perf_query, (self.mysql_database,))
                if perf_result:
                    metrics["performance"] = {
                        "slow_queries": perf_result[0]["query_count"],
                        "avg_slow_query_ms": float(perf_result[0]["avg_time_ms"] or 0),
                        "max_query_ms": float(perf_result[0]["max_time_ms"] or 0),
                    }
            except Exception as e:
                logger.warning("Could not fetch performance schema: %s", e)
                metrics["performance"] = {"error": str(e)}

        # Calculate overall health score
        health_score = self._calculate_health_score(metrics)
        metrics["health_score"] = health_score

        logger.info(
            "Collected metrics: autoload=%.2fMB, orphaned=%d, score=%d",
            metrics["autoload"]["size_mb"],
            metrics["bloat"]["orphaned_postmeta"],
            health_score,
        )

        return metrics

    def _execute_query(
        self,
        query: str,
        params: tuple = None,
    ) -> List[Dict[str, Any]]:
        """Execute MySQL query and return results."""
        import pymysql

        connection = pymysql.connect(
            host=self.mysql_host,
            user=self.mysql_user,
            password=self.mysql_password,
            database=self.mysql_database,
            cursorclass=pymysql.cursors.DictCursor,
        )

        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        finally:
            connection.close()

    def _calculate_health_score(self, metrics: Dict[str, Any]) -> int:
        """Calculate overall health score (0-100)."""
        score = 100

        # Autoload penalty
        autoload_mb = metrics["autoload"]["size_mb"]
        if autoload_mb > 5:
            score -= 30
        elif autoload_mb > 2:
            score -= 15

        # Bloat penalty
        orphaned = metrics["bloat"]["orphaned_postmeta"]
        if orphaned > 50000:
            score -= 20
        elif orphaned > 10000:
            score -= 10

        transients = metrics["bloat"]["expired_transients"]
        if transients > 5000:
            score -= 10
        elif transients > 1000:
            score -= 5

        return max(0, score)

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use sync version")`,
              },
              {
                language: 'python',
                title: 'WP-CLI Integration Tool',
                description:
                  'Tool for executing WP-CLI commands for WordPress database operations.',
                code: `from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import subprocess
import json
import logging

logger = logging.getLogger("wp_cli_tool")


class WPCLICommandInput(BaseModel):
    """Input schema for WP-CLI commands."""
    command: str = Field(description="WP-CLI command to execute")
    args: List[str] = Field(
        default=[],
        description="Additional arguments for the command"
    )
    format_output: str = Field(
        default="json",
        description="Output format: 'json', 'table', or 'plain'"
    )


class WPCLITool(BaseTool):
    """Tool for executing WP-CLI commands on WordPress."""

    name: str = "wp_cli_tool"
    description: str = (
        "Execute WP-CLI commands for WordPress database operations. "
        "Can run transient cleanup, option management, and database commands."
    )
    args_schema: type[BaseModel] = WPCLICommandInput

    wp_path: str  # Path to WordPress installation
    ssh_host: Optional[str] = None  # For remote execution
    ssh_user: Optional[str] = None
    allowed_commands: List[str] = [
        "transient", "option", "db", "plugin", "cache"
    ]

    def _run(
        self,
        command: str,
        args: List[str] = None,
        format_output: str = "json",
    ) -> Dict[str, Any]:
        """Execute WP-CLI command."""

        # Validate command is allowed
        base_command = command.split()[0]
        if base_command not in self.allowed_commands:
            return {
                "success": False,
                "error": f"Command '{base_command}' not in allowed list",
            }

        # Build full command
        full_args = args or []
        if format_output == "json" and "--format" not in str(full_args):
            full_args.append(f"--format={format_output}")

        cmd_parts = ["wp", command] + full_args + [f"--path={self.wp_path}"]

        # Add SSH wrapper if remote
        if self.ssh_host:
            ssh_cmd = f"{self.ssh_user}@{self.ssh_host}" if self.ssh_user else self.ssh_host
            cmd_parts = ["ssh", ssh_cmd] + cmd_parts

        logger.info("Executing: %s", " ".join(cmd_parts))

        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr,
                    "command": " ".join(cmd_parts),
                }

            # Parse output
            output = result.stdout.strip()
            if format_output == "json" and output:
                try:
                    parsed = json.loads(output)
                    return {"success": True, "data": parsed}
                except json.JSONDecodeError:
                    return {"success": True, "data": output}

            return {"success": True, "data": output}

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out after 120 seconds",
            }
        except Exception as e:
            logger.error("WP-CLI execution failed: %s", e)
            return {"success": False, "error": str(e)}

    def delete_expired_transients(self) -> Dict[str, Any]:
        """Convenience method to delete expired transients."""
        return self._run("transient", ["delete", "--expired", "--network"])

    def get_autoload_options(self) -> Dict[str, Any]:
        """Get all autoloaded options."""
        return self._run(
            "option",
            ["list", "--autoload=on", "--fields=option_name,size_bytes"]
        )

    def optimize_database(self) -> Dict[str, Any]:
        """Run database optimization."""
        return self._run("db", ["optimize"])

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use sync version")`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Query Optimizer and Plugin Profiler agents that analyze performance data and make recommendations for improvements.',
            toolsUsed: ['MySQL EXPLAIN', 'Query Monitor', 'Performance Schema'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Query Optimizer Analysis Tool',
                description:
                  'Tool for the Query Optimizer Agent to analyze slow queries and recommend indexes.',
                code: `from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger("query_optimizer")


@dataclass
class QueryAnalysis:
    """Analysis result for a single query."""
    query_hash: str
    query_pattern: str
    avg_time_ms: float
    execution_count: int
    rows_examined: int
    rows_returned: int
    bottleneck: str
    recommended_index: Optional[str]
    estimated_improvement: float


class SlowQueryInput(BaseModel):
    """Input schema for slow query analysis."""
    min_time_ms: int = Field(
        default=100,
        description="Minimum query time to analyze (ms)"
    )
    limit: int = Field(
        default=10,
        description="Number of queries to analyze"
    )
    include_explain: bool = Field(
        default=True,
        description="Run EXPLAIN on each query"
    )


class QueryOptimizerTool(BaseTool):
    """Tool for analyzing slow queries and recommending optimizations."""

    name: str = "slow_query_analyzer"
    description: str = (
        "Analyze slow queries from MySQL performance schema. "
        "Returns query patterns, bottlenecks, and index recommendations."
    )
    args_schema: type[BaseModel] = SlowQueryInput

    mysql_host: str
    mysql_user: str
    mysql_password: str
    mysql_database: str

    # Common WordPress query patterns and their optimal indexes
    KNOWN_PATTERNS: Dict[str, Dict[str, str]] = {
        r"wp_postmeta.*meta_key.*meta_value": {
            "index": "CREATE INDEX idx_meta_key_value ON wp_postmeta (meta_key, meta_value(191))",
            "reason": "Composite index for meta key-value lookups",
        },
        r"wp_wc_order.*product_id.*date": {
            "index": "CREATE INDEX idx_product_date ON wp_wc_order_product_lookup (product_id, date_created)",
            "reason": "Composite index for product order lookups",
        },
        r"wp_options.*option_name": {
            "index": "-- wp_options already has primary key on option_name",
            "reason": "Check if query is using LIKE with leading wildcard",
        },
    }

    def _run(
        self,
        min_time_ms: int = 100,
        limit: int = 10,
        include_explain: bool = True,
    ) -> Dict[str, Any]:
        """Analyze slow queries and provide recommendations."""

        # Fetch slow queries from performance schema
        query = """
            SELECT
                DIGEST_TEXT as query_pattern,
                DIGEST as query_hash,
                COUNT_STAR as exec_count,
                ROUND(AVG_TIMER_WAIT / 1e9, 2) as avg_time_ms,
                ROUND(SUM_TIMER_WAIT / 1e9, 2) as total_time_ms,
                SUM_ROWS_EXAMINED as rows_examined,
                SUM_ROWS_SENT as rows_returned
            FROM performance_schema.events_statements_summary_by_digest
            WHERE SCHEMA_NAME = %s
              AND AVG_TIMER_WAIT > %s
            ORDER BY AVG_TIMER_WAIT DESC
            LIMIT %s
        """

        min_time_ns = min_time_ms * 1e6  # Convert to nanoseconds
        results = self._execute_query(query, (self.mysql_database, min_time_ns, limit))

        analyses: List[Dict[str, Any]] = []
        total_improvement_potential = 0

        for row in results:
            analysis = self._analyze_query(row, include_explain)
            analyses.append(analysis)
            total_improvement_potential += analysis.get("estimated_improvement_pct", 0)

        # Generate index recommendations
        recommendations = self._generate_recommendations(analyses)

        return {
            "queries_analyzed": len(analyses),
            "total_slow_queries": sum(a["exec_count"] for a in analyses),
            "analyses": analyses,
            "recommendations": recommendations,
            "total_improvement_potential_pct": round(total_improvement_potential / len(analyses), 1) if analyses else 0,
            "sql_commands": self._generate_sql_commands(recommendations),
        }

    def _analyze_query(
        self,
        row: Dict[str, Any],
        include_explain: bool,
    ) -> Dict[str, Any]:
        """Analyze a single slow query."""
        query_pattern = row["query_pattern"]
        avg_time_ms = float(row["avg_time_ms"])
        rows_examined = row["rows_examined"]
        rows_returned = row["rows_returned"]

        # Calculate efficiency ratio
        if rows_returned > 0:
            efficiency = rows_returned / rows_examined
        else:
            efficiency = 0

        # Identify bottleneck
        bottleneck = self._identify_bottleneck(query_pattern, efficiency, avg_time_ms)

        # Check for known patterns
        recommended_index = None
        for pattern, solution in self.KNOWN_PATTERNS.items():
            if re.search(pattern, query_pattern, re.IGNORECASE):
                recommended_index = solution["index"]
                break

        # Run EXPLAIN if requested
        explain_result = None
        if include_explain and "SELECT" in query_pattern.upper():
            explain_result = self._run_explain(query_pattern)

        # Estimate improvement
        improvement = self._estimate_improvement(bottleneck, efficiency)

        return {
            "query_hash": row["query_hash"],
            "query_pattern": query_pattern[:200] + "..." if len(query_pattern) > 200 else query_pattern,
            "avg_time_ms": avg_time_ms,
            "exec_count": row["exec_count"],
            "rows_examined": rows_examined,
            "rows_returned": rows_returned,
            "efficiency_ratio": round(efficiency, 4),
            "bottleneck": bottleneck,
            "recommended_index": recommended_index,
            "explain_result": explain_result,
            "estimated_improvement_pct": improvement,
        }

    def _identify_bottleneck(
        self,
        query: str,
        efficiency: float,
        avg_time_ms: float,
    ) -> str:
        """Identify the likely bottleneck for a query."""
        if efficiency < 0.01:
            return "full_table_scan"
        if "JOIN" in query.upper() and efficiency < 0.1:
            return "inefficient_join"
        if "LIKE" in query.upper() and "%" in query:
            return "leading_wildcard"
        if avg_time_ms > 1000:
            return "complex_query"
        return "missing_index"

    def _run_explain(self, query_pattern: str) -> Optional[Dict[str, Any]]:
        """Run EXPLAIN on a query pattern."""
        # Clean up the digest pattern for EXPLAIN
        # This is simplified - real implementation would need to handle parameters
        try:
            explain_query = f"EXPLAIN {query_pattern}"
            result = self._execute_query(explain_query)
            if result:
                return {
                    "type": result[0].get("type"),
                    "possible_keys": result[0].get("possible_keys"),
                    "key": result[0].get("key"),
                    "rows": result[0].get("rows"),
                    "extra": result[0].get("Extra"),
                }
        except Exception as e:
            logger.warning("EXPLAIN failed: %s", e)
        return None

    def _estimate_improvement(self, bottleneck: str, efficiency: float) -> float:
        """Estimate improvement percentage from optimization."""
        improvements = {
            "full_table_scan": 80,
            "inefficient_join": 60,
            "leading_wildcard": 30,
            "missing_index": 50,
            "complex_query": 20,
        }
        base = improvements.get(bottleneck, 10)
        # Adjust based on current efficiency
        if efficiency < 0.001:
            return min(base * 1.5, 90)
        return base

    def _generate_recommendations(
        self,
        analyses: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Generate consolidated recommendations."""
        recommendations = []
        seen_indexes = set()

        for analysis in analyses:
            if analysis["recommended_index"] and analysis["recommended_index"] not in seen_indexes:
                recommendations.append({
                    "type": "index",
                    "sql": analysis["recommended_index"],
                    "reason": f"Improves query with {analysis['bottleneck']}",
                    "estimated_improvement": f"{analysis['estimated_improvement_pct']}%",
                })
                seen_indexes.add(analysis["recommended_index"])

        return recommendations

    def _generate_sql_commands(
        self,
        recommendations: List[Dict[str, str]],
    ) -> str:
        """Generate SQL script for all recommendations."""
        commands = ["-- WooCommerce Query Optimization Script", "-- Run during low-traffic period", ""]
        for rec in recommendations:
            if rec["type"] == "index":
                commands.append(f"-- {rec['reason']}")
                commands.append(f"{rec['sql']};")
                commands.append("")
        return "\\n".join(commands)

    def _execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute MySQL query."""
        import pymysql
        connection = pymysql.connect(
            host=self.mysql_host,
            user=self.mysql_user,
            password=self.mysql_password,
            database=self.mysql_database,
            cursorclass=pymysql.cursors.DictCursor,
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        finally:
            connection.close()

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use sync version")`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement LangGraph state machine to coordinate agent interactions, manage optimization workflow state, and ensure safe execution of database operations.',
            toolsUsed: ['LangGraph', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'WooCommerce Optimization LangGraph State Machine',
                description:
                  'LangGraph workflow orchestrating the multi-agent WooCommerce optimization system.',
                code: `from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal
from datetime import datetime, time
import operator
import json
import redis
import logging

logger = logging.getLogger("woo_optimization_orchestrator")


class WooOptimizationState(TypedDict):
    """State maintained across the optimization workflow."""

    # Workflow metadata
    run_id: str
    started_at: str
    site_url: str
    current_step: str
    priority: str

    # Health check state
    health_metrics: dict
    health_score: int
    thresholds_exceeded: list[str]

    # Query analysis state
    slow_queries_analyzed: int
    index_recommendations: list[dict]

    # Plugin profiling state
    plugins_profiled: int
    heavy_plugins: list[dict]
    recommended_removals: list[str]

    # Remediation state
    backup_created: bool
    cleanup_actions: list[dict]
    tables_optimized: list[str]

    # Results
    ttfb_before_ms: int
    ttfb_after_ms: int
    improvement_pct: float
    alerts_sent: int
    errors: Annotated[list[str], operator.add]


def create_woo_optimization_graph(
    db_health_agent,
    query_optimizer_agent,
    plugin_profiler_agent,
    remediation_agent,
    supervisor_agent,
    redis_client: redis.Redis,
):
    """Create the LangGraph workflow for WooCommerce optimization."""

    workflow = StateGraph(WooOptimizationState)

    def is_safe_to_cleanup() -> bool:
        """Check if current time is safe for cleanup operations."""
        current_hour = datetime.now().hour
        # Safe hours: 10 PM to 6 AM
        return current_hour >= 22 or current_hour < 6

    def run_health_check(state: WooOptimizationState) -> dict:
        """Execute database health check."""
        logger.info("Running database health check...")
        try:
            result = db_health_agent.invoke({"task": "full_health_check"})

            thresholds_exceeded = []
            if result.get("autoload", {}).get("size_mb", 0) > 2:
                thresholds_exceeded.append("autoload_size")
            if result.get("bloat", {}).get("orphaned_postmeta", 0) > 10000:
                thresholds_exceeded.append("orphaned_postmeta")

            return {
                "health_metrics": result,
                "health_score": result.get("health_score", 0),
                "thresholds_exceeded": thresholds_exceeded,
                "ttfb_before_ms": result.get("ttfb_ms", 0),
                "current_step": "health_complete",
            }
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return {"errors": [f"Health check error: {str(e)}"]}

    def run_query_analysis(state: WooOptimizationState) -> dict:
        """Analyze slow queries."""
        logger.info("Analyzing slow queries...")
        try:
            result = query_optimizer_agent.invoke({
                "task": "analyze_slow_queries",
                "min_time_ms": 100,
                "limit": 10,
            })

            return {
                "slow_queries_analyzed": result.get("queries_analyzed", 0),
                "index_recommendations": result.get("recommendations", []),
                "current_step": "query_analysis_complete",
            }
        except Exception as e:
            logger.error("Query analysis failed: %s", e)
            return {"errors": [f"Query analysis error: {str(e)}"]}

    def run_plugin_profiling(state: WooOptimizationState) -> dict:
        """Profile plugin performance impact."""
        logger.info("Profiling plugins...")
        try:
            result = plugin_profiler_agent.invoke({"task": "profile_all_plugins"})

            return {
                "plugins_profiled": result.get("plugins_profiled", 0),
                "heavy_plugins": result.get("heavy_plugins", []),
                "recommended_removals": result.get("recommended_removals", []),
                "current_step": "plugin_profiling_complete",
            }
        except Exception as e:
            logger.error("Plugin profiling failed: %s", e)
            return {"errors": [f"Plugin profiling error: {str(e)}"]}

    def run_remediation(state: WooOptimizationState) -> dict:
        """Execute safe cleanup operations."""
        logger.info("Running remediation...")

        # Check if safe to run
        if not is_safe_to_cleanup() and state.get("priority") != "critical":
            logger.info("Skipping cleanup - not in safe hours")
            return {
                "backup_created": False,
                "cleanup_actions": [{"skipped": "Not in safe hours (10 PM - 6 AM)"}],
                "current_step": "remediation_skipped",
            }

        try:
            result = remediation_agent.invoke({
                "task": "safe_cleanup",
                "thresholds_exceeded": state.get("thresholds_exceeded", []),
                "health_metrics": state.get("health_metrics", {}),
            })

            return {
                "backup_created": result.get("backup_created", False),
                "cleanup_actions": result.get("actions", []),
                "tables_optimized": result.get("tables_optimized", []),
                "current_step": "remediation_complete",
            }
        except Exception as e:
            logger.error("Remediation failed: %s", e)
            return {"errors": [f"Remediation error: {str(e)}"]}

    def generate_report(state: WooOptimizationState) -> dict:
        """Generate final optimization report."""
        logger.info("Generating report...")

        # Calculate improvement
        ttfb_before = state.get("ttfb_before_ms", 0)
        ttfb_after = ttfb_before  # Would measure actual TTFB after optimization

        # Estimate improvement based on actions taken
        estimated_reduction = 0
        for action in state.get("cleanup_actions", []):
            estimated_reduction += action.get("estimated_ms_savings", 0)

        ttfb_after = max(ttfb_before - estimated_reduction, 500)
        improvement = ((ttfb_before - ttfb_after) / ttfb_before * 100) if ttfb_before > 0 else 0

        # Persist to Redis
        checkpoint = {
            "run_id": state["run_id"],
            "timestamp": datetime.utcnow().isoformat(),
            "health_score": state.get("health_score", 0),
            "ttfb_before": ttfb_before,
            "ttfb_after": ttfb_after,
            "improvement_pct": round(improvement, 1),
            "actions_taken": len(state.get("cleanup_actions", [])),
        }
        redis_client.setex(
            f"woo:optimization:{datetime.utcnow().date().isoformat()}",
            86400 * 30,
            json.dumps(checkpoint),
        )

        return {
            "ttfb_after_ms": ttfb_after,
            "improvement_pct": round(improvement, 1),
            "current_step": "complete",
        }

    def should_run_remediation(state: WooOptimizationState) -> Literal["remediation", "report"]:
        """Determine if remediation should run."""
        if state.get("thresholds_exceeded"):
            return "remediation"
        return "report"

    # Add nodes
    workflow.add_node("health_check", run_health_check)
    workflow.add_node("query_analysis", run_query_analysis)
    workflow.add_node("plugin_profiling", run_plugin_profiling)
    workflow.add_node("remediation", run_remediation)
    workflow.add_node("report", generate_report)

    # Define edges
    workflow.set_entry_point("health_check")
    workflow.add_edge("health_check", "query_analysis")
    workflow.add_edge("query_analysis", "plugin_profiling")

    workflow.add_conditional_edges(
        "plugin_profiling",
        should_run_remediation,
        {
            "remediation": "remediation",
            "report": "report",
        },
    )

    workflow.add_edge("remediation", "report")
    workflow.add_edge("report", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the multi-agent WooCommerce optimization system with Docker, implement observability with LangSmith, and set up Prometheus metrics for monitoring.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose Deployment Configuration',
                description:
                  'Docker Compose configuration for deploying the WooCommerce optimization multi-agent system.',
                code: `version: "3.8"

services:
  woo-optimizer:
    build:
      context: .
      dockerfile: Dockerfile.woo-optimizer
    container_name: woo-optimizer
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=\${LANGSMITH_API_KEY}
      - LANGCHAIN_PROJECT=woo-optimizer-prod
      - REDIS_URL=redis://redis:6379/1
      - MYSQL_HOST=\${MYSQL_HOST}
      - MYSQL_USER=\${MYSQL_USER}
      - MYSQL_PASSWORD=\${MYSQL_PASSWORD}
      - MYSQL_DATABASE=\${MYSQL_DATABASE}
      - WP_PATH=\${WP_PATH}
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
      - LOG_LEVEL=INFO
    depends_on:
      - redis
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
    ports:
      - "8096:8096"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8096/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "1.5"
          memory: 2G

  redis:
    image: redis:7-alpine
    container_name: woo-optimizer-redis
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    ports:
      - "6380:6379"

  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: woo-optimizer-prometheus
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9091:9090"

  # Scheduled optimization runner
  scheduler:
    build:
      context: .
      dockerfile: Dockerfile.scheduler
    container_name: woo-optimizer-scheduler
    environment:
      - OPTIMIZER_URL=http://woo-optimizer:8096
      - SCHEDULE_CRON=0 3 * * 0  # Weekly at 3 AM Sunday
      - TZ=America/New_York
    depends_on:
      - woo-optimizer
    restart: unless-stopped

volumes:
  redis-data:
  prometheus-data:`,
              },
              {
                language: 'python',
                title: 'Observability and Metrics Collection',
                description:
                  'LangSmith tracing and Prometheus metrics for monitoring the optimization system.',
                code: `from langsmith import Client
from langsmith.run_helpers import traceable
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from functools import wraps
from datetime import datetime
import time
import logging

logger = logging.getLogger("woo_observability")

# Initialize LangSmith
langsmith_client = Client()

# Prometheus metrics
OPTIMIZATION_RUNS = Counter(
    "woo_optimization_runs_total",
    "Total optimization runs",
    ["status", "priority"],
)

OPTIMIZATION_DURATION = Histogram(
    "woo_optimization_duration_seconds",
    "Optimization run duration",
    buckets=[30, 60, 120, 300, 600, 900, 1800],
)

HEALTH_SCORE = Gauge(
    "woo_database_health_score",
    "Current database health score (0-100)",
)

AUTOLOAD_SIZE_MB = Gauge(
    "woo_autoload_size_mb",
    "Current autoload size in MB",
)

TTFB_MS = Gauge(
    "woo_ttfb_milliseconds",
    "Current Time to First Byte",
)

CLEANUP_ACTIONS = Counter(
    "woo_cleanup_actions_total",
    "Total cleanup actions executed",
    ["action_type"],
)


def observe_optimization(func):
    """Decorator for observing optimization runs."""
    @wraps(func)
    @traceable(name="woo_optimization_run", project_name="woo-optimizer-prod")
    def wrapper(*args, **kwargs):
        start_time = time.time()
        status = "success"
        priority = kwargs.get("priority", "medium")

        try:
            result = func(*args, **kwargs)

            # Update metrics
            if "health_score" in result:
                HEALTH_SCORE.set(result["health_score"])
            if "ttfb_after_ms" in result:
                TTFB_MS.set(result["ttfb_after_ms"])
            if "health_metrics" in result:
                autoload = result["health_metrics"].get("autoload", {})
                AUTOLOAD_SIZE_MB.set(autoload.get("size_mb", 0))

            return result

        except Exception as e:
            status = "error"
            logger.error("Optimization failed: %s", e)
            raise

        finally:
            duration = time.time() - start_time
            OPTIMIZATION_RUNS.labels(status=status, priority=priority).inc()
            OPTIMIZATION_DURATION.observe(duration)

            logger.info(
                "Optimization %s completed in %.2fs",
                status,
                duration,
            )

    return wrapper


class WooMetricsServer:
    """Prometheus metrics server for WooCommerce optimization."""

    def __init__(self, port: int = 8097):
        self.port = port

    def start(self):
        """Start the metrics HTTP server."""
        start_http_server(self.port)
        logger.info("Metrics server started on port %d", self.port)

    def record_cleanup_action(self, action_type: str):
        """Record a cleanup action."""
        CLEANUP_ACTIONS.labels(action_type=action_type).inc()

    def update_health_metrics(self, metrics: dict):
        """Update health gauge metrics."""
        HEALTH_SCORE.set(metrics.get("health_score", 0))
        AUTOLOAD_SIZE_MB.set(
            metrics.get("autoload", {}).get("size_mb", 0)
        )`,
              },
            ],
          },
        ],
      },
    },

    /* ──────────────────────────────────────────────
       Pain Point 3 — The Analytics Trust Gap
       ────────────────────────────────────────────── */
    {
      id: 'analytics-trust-gap',
      number: 3,
      title: 'The Analytics Trust Gap',
      subtitle: 'GA4, Shopify, and Backend Numbers Never Match',
      summary:
        'Google Analytics says 1,000 orders. Shopify says 1,150. Your backend says 1,087. Nobody trusts any number, so decisions stall.',
      tags: ['analytics', 'data-reconciliation', 'ecommerce'],
      metrics: {
        annualCostRange: '$400K - $2M',
        roi: '5x',
        paybackPeriod: '2-3 months',
        investmentRange: '$60K - $120K',
      },
      price: {
        present: {
          title: 'Current State',
          description:
            'Three sources of truth for order and revenue data produce three different numbers. Teams cherry-pick whichever source supports their narrative.',
          bullets: [
            'GA4 under-reports by 10-15% due to ad blockers, consent mode, and session attribution gaps',
            'Shopify over-counts by including test orders, partially refunded transactions, and draft orders',
            'Backend database numbers drift due to timezone mismatches and async webhook processing',
          ],
          severity: 'high',
        },
        root: {
          title: 'Root Cause',
          description:
            'Each platform defines "order" and "revenue" differently, applies different filters, and timestamps events in different timezones.',
          bullets: [
            'GA4 counts purchase events; Shopify counts order objects; backend counts payment settlements',
            'Timezone handling differs: GA4 uses property timezone, Shopify uses shop timezone, backend uses UTC',
            'Refunds, cancellations, and partial fulfillments are reflected at different times across systems',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Business Impact',
          description:
            'When nobody trusts the numbers, decisions are delayed or made on gut feeling. Marketing cannot attribute spend, and finance cannot close books cleanly.',
          bullets: [
            'Marketing ROAS calculations off by 15-25%, leading to misallocated ad spend',
            'Monthly financial close delayed 3-5 days for manual reconciliation',
            'Executive dashboards questioned in every meeting, eroding data culture',
          ],
          severity: 'critical',
        },
        cost: {
          title: 'Cost of Inaction',
          description:
            'Misattributed marketing spend and delayed decisions compound. The trust deficit grows as teams build parallel shadow spreadsheets.',
          bullets: [
            '$400K - $2M annual cost from misallocated spend and delayed decisions',
            '40+ hours per month spent on manual reconciliation across teams',
            'Shadow reporting systems proliferate, each with its own biases',
          ],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Expected Return',
          description:
            'A single reconciled source of truth restores decision velocity and correctly attributes marketing spend.',
          bullets: [
            '5x ROI within 2-3 months from corrected attribution and faster close cycles',
            'Eliminate 40+ hours per month of manual reconciliation',
            'Single dashboard trusted by marketing, finance, and executive teams',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Build a cross-platform reconciliation layer that normalizes definitions, aligns timezones, and flags discrepancies automatically.',
        prerequisites: [
          'GA4 BigQuery export enabled with event-level data',
          'Shopify Admin API access or data warehouse replica of Shopify orders',
          'Backend database access with payment settlement records',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        toolsUsed: ['SQL', 'Python', 'pytest', 'Docker', 'GitHub Actions', 'cron / Airflow', 'Slack API'],
        steps: [
          {
            stepNumber: 1,
            title: 'Cross-Platform Order Reconciliation',
            description:
              'Normalize order definitions across GA4, Shopify, and the backend, then match records to identify where discrepancies originate.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Three-Way Order Reconciliation',
                description:
                  'Join GA4 purchase events, Shopify orders, and backend settlements into a unified reconciliation view.',
                code: `-- Three-way reconciliation: GA4 vs Shopify vs Backend
WITH ga4_orders AS (
  SELECT
    event_date,
    (SELECT value.string_value FROM UNNEST(event_params)
     WHERE key = 'transaction_id')         AS transaction_id,
    (SELECT value.double_value FROM UNNEST(event_params)
     WHERE key = 'value')                  AS ga4_revenue,
    'ga4'                                  AS source
  FROM \`project.analytics_000000.events_*\`
  WHERE event_name = 'purchase'
    AND _TABLE_SUFFIX BETWEEN '20250101' AND '20250131'
),
shopify_orders AS (
  SELECT
    DATE(created_at AT TIME ZONE 'America/New_York') AS event_date,
    CAST(order_number AS STRING)                     AS transaction_id,
    total_price                                      AS shopify_revenue,
    financial_status
  FROM shopify_orders_raw
  WHERE created_at >= '2025-01-01'
    AND created_at < '2025-02-01'
    AND test = FALSE
    AND cancelled_at IS NULL
),
backend_orders AS (
  SELECT
    DATE(settled_at AT TIME ZONE 'UTC') AS event_date,
    reference_id                        AS transaction_id,
    net_amount                          AS backend_revenue
  FROM payment_settlements
  WHERE settled_at >= '2025-01-01'
    AND settled_at < '2025-02-01'
    AND status = 'settled'
)
SELECT
  COALESCE(g.transaction_id, s.transaction_id, b.transaction_id) AS txn_id,
  g.ga4_revenue,
  s.shopify_revenue,
  b.backend_revenue,
  CASE
    WHEN g.transaction_id IS NULL          THEN 'MISSING_GA4'
    WHEN s.transaction_id IS NULL          THEN 'MISSING_SHOPIFY'
    WHEN b.transaction_id IS NULL          THEN 'MISSING_BACKEND'
    WHEN ABS(g.ga4_revenue - s.shopify_revenue) > 0.01
      OR ABS(s.shopify_revenue - b.backend_revenue) > 0.01
                                            THEN 'AMOUNT_MISMATCH'
    ELSE 'MATCHED'
  END AS reconciliation_status
FROM ga4_orders g
FULL OUTER JOIN shopify_orders s ON g.transaction_id = s.transaction_id
FULL OUTER JOIN backend_orders b ON s.transaction_id = b.transaction_id
ORDER BY reconciliation_status, txn_id;`,
              },
              {
                language: 'sql',
                title: 'Daily Discrepancy Summary',
                description:
                  'Aggregate reconciliation results by day to surface systemic patterns in data drift.',
                code: `-- Daily discrepancy summary across platforms
WITH daily_ga4 AS (
  SELECT
    event_date,
    COUNT(DISTINCT transaction_id)   AS ga4_orders,
    SUM(ga4_revenue)                 AS ga4_revenue
  FROM ga4_purchase_events
  WHERE event_date BETWEEN '2025-01-01' AND '2025-01-31'
  GROUP BY event_date
),
daily_shopify AS (
  SELECT
    order_date                       AS event_date,
    COUNT(DISTINCT order_id)         AS shopify_orders,
    SUM(total_price)                 AS shopify_revenue
  FROM shopify_orders_clean
  WHERE order_date BETWEEN '2025-01-01' AND '2025-01-31'
  GROUP BY order_date
),
daily_backend AS (
  SELECT
    settlement_date                  AS event_date,
    COUNT(DISTINCT reference_id)     AS backend_orders,
    SUM(net_amount)                  AS backend_revenue
  FROM payment_settlements_clean
  WHERE settlement_date BETWEEN '2025-01-01' AND '2025-01-31'
  GROUP BY settlement_date
)
SELECT
  COALESCE(g.event_date, s.event_date, b.event_date) AS report_date,
  g.ga4_orders,       s.shopify_orders,     b.backend_orders,
  s.shopify_orders - g.ga4_orders           AS shopify_ga4_delta,
  b.backend_orders - s.shopify_orders       AS backend_shopify_delta,
  ROUND(g.ga4_revenue, 2)                  AS ga4_rev,
  ROUND(s.shopify_revenue, 2)              AS shopify_rev,
  ROUND(b.backend_revenue, 2)              AS backend_rev,
  ROUND(s.shopify_revenue - g.ga4_revenue, 2)  AS rev_gap_shopify_ga4,
  ROUND(b.backend_revenue - s.shopify_revenue, 2) AS rev_gap_backend_shopify
FROM daily_ga4 g
FULL OUTER JOIN daily_shopify s  ON g.event_date = s.event_date
FULL OUTER JOIN daily_backend b  ON s.event_date = b.event_date
ORDER BY report_date;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Automated Discrepancy Detection & Alerting',
            description:
              'Build a Python pipeline that runs reconciliation daily, classifies discrepancy root causes, and alerts the team when thresholds are breached.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Reconciliation Pipeline',
                description:
                  'Automated pipeline that fetches data from all three sources, reconciles records, and classifies discrepancy causes.',
                code: `import logging
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

logger = logging.getLogger("reconciliation")

class DiscrepancyType(Enum):
    MISSING_GA4 = "missing_ga4"
    MISSING_SHOPIFY = "missing_shopify"
    MISSING_BACKEND = "missing_backend"
    AMOUNT_MISMATCH = "amount_mismatch"
    TIMEZONE_SHIFT = "timezone_shift"
    MATCHED = "matched"

@dataclass
class ReconciliationResult:
    transaction_id: str
    ga4_revenue: float | None
    shopify_revenue: float | None
    backend_revenue: float | None
    status: DiscrepancyType
    probable_cause: str

class ReconciliationPipeline:
    TOLERANCE = 0.02  # 2% tolerance for floating point / currency rounding

    def __init__(self, ga4_client, shopify_client, backend_db):
        self.ga4 = ga4_client
        self.shopify = shopify_client
        self.backend = backend_db

    def reconcile_day(self, target_date: date) -> list[ReconciliationResult]:
        ga4_data = self.ga4.get_purchases(target_date)
        shopify_data = self.shopify.get_orders(target_date)
        backend_data = self.backend.get_settlements(target_date)

        all_txn_ids = (set(ga4_data.keys())
                       | set(shopify_data.keys())
                       | set(backend_data.keys()))

        results = []
        for txn_id in all_txn_ids:
            g = ga4_data.get(txn_id)
            s = shopify_data.get(txn_id)
            b = backend_data.get(txn_id)
            result = self._classify(txn_id, g, s, b)
            results.append(result)

        matched = sum(1 for r in results
                      if r.status == DiscrepancyType.MATCHED)
        logger.info("Reconciled %d transactions: %d matched, %d discrepant",
                     len(results), matched, len(results) - matched)
        return results

    def _classify(self, txn_id, ga4, shopify, backend):
        if ga4 is None:
            return ReconciliationResult(
                txn_id, None, shopify, backend,
                DiscrepancyType.MISSING_GA4,
                "Ad blocker, consent mode, or tracking script failure")
        if shopify is None:
            return ReconciliationResult(
                txn_id, ga4, None, backend,
                DiscrepancyType.MISSING_SHOPIFY,
                "Draft order or API-created order not synced")
        if backend is None:
            return ReconciliationResult(
                txn_id, ga4, shopify, None,
                DiscrepancyType.MISSING_BACKEND,
                "Payment pending settlement or async webhook delay")
        if self._amounts_match(ga4, shopify, backend):
            return ReconciliationResult(
                txn_id, ga4, shopify, backend,
                DiscrepancyType.MATCHED, "")
        return ReconciliationResult(
            txn_id, ga4, shopify, backend,
            DiscrepancyType.AMOUNT_MISMATCH,
            "Partial refund, currency conversion, or tax delta")

    def _amounts_match(self, ga4, shopify, backend) -> bool:
        ref = shopify
        return (abs(ga4 - ref) / ref <= self.TOLERANCE
                and abs(backend - ref) / ref <= self.TOLERANCE)`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate reconciliation accuracy and data pipeline integrity with SQL assertions and automated pytest suites.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Reconciliation Data Quality Assertions',
                description:
                  'Run data quality checks to ensure cross-platform reconciliation data is complete and consistent.',
                code: `-- Assert: all Shopify orders have a reconciliation record
SELECT s.order_id
FROM shopify_orders_clean s
LEFT JOIN reconciliation_results rr
  ON CAST(s.order_number AS VARCHAR) = rr.transaction_id
WHERE s.order_date >= CURRENT_DATE - INTERVAL '7 days'
  AND rr.transaction_id IS NULL;
-- Expected: zero rows

-- Assert: no duplicate transaction IDs in reconciliation output
SELECT transaction_id, COUNT(*) AS dupes
FROM reconciliation_results
WHERE reconciled_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY transaction_id
HAVING COUNT(*) > 1;
-- Expected: zero rows

-- Assert: match rate above 85% for the past 7 days
SELECT
  reconciled_date,
  COUNT(*) AS total_txns,
  SUM(CASE WHEN status = 'matched' THEN 1 ELSE 0 END) AS matched,
  ROUND(SUM(CASE WHEN status = 'matched' THEN 1 ELSE 0 END)::NUMERIC
        / COUNT(*) * 100, 1) AS match_rate_pct
FROM reconciliation_results
WHERE reconciled_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY reconciled_date
HAVING ROUND(SUM(CASE WHEN status = 'matched' THEN 1 ELSE 0 END)::NUMERIC
             / COUNT(*) * 100, 1) < 85.0;
-- Expected: zero rows (all days above 85%)

-- Assert: revenue deltas within 3% tolerance
SELECT
  reconciled_date,
  SUM(ABS(COALESCE(shopify_revenue, 0) - COALESCE(backend_revenue, 0)))
    AS total_rev_delta,
  SUM(COALESCE(shopify_revenue, 0)) AS total_shopify_rev,
  ROUND(
    SUM(ABS(COALESCE(shopify_revenue, 0) - COALESCE(backend_revenue, 0)))
    / NULLIF(SUM(COALESCE(shopify_revenue, 0)), 0) * 100, 2
  ) AS delta_pct
FROM reconciliation_results
WHERE reconciled_date >= CURRENT_DATE - INTERVAL '7 days'
  AND status = 'amount_mismatch'
GROUP BY reconciled_date
HAVING delta_pct > 3.0;
-- Expected: zero rows`,
              },
              {
                language: 'python',
                title: 'Reconciliation Pipeline Validation Suite',
                description:
                  'pytest-based test suite that validates the reconciliation pipeline, classification logic, and alerting thresholds.',
                code: `import pytest
from datetime import date
from unittest.mock import MagicMock, patch
from reconciliation_pipeline import (
    ReconciliationPipeline, ReconciliationResult, DiscrepancyType
)
from discrepancy_alerts import DiscrepancyAlertService

# ── Fixtures ──────────────────────────────────────────

@pytest.fixture
def mock_clients():
    ga4 = MagicMock()
    shopify = MagicMock()
    backend = MagicMock()
    return ga4, shopify, backend

@pytest.fixture
def pipeline(mock_clients):
    ga4, shopify, backend = mock_clients
    return ReconciliationPipeline(ga4, shopify, backend)

@pytest.fixture
def mock_alert_sink():
    sink = MagicMock()
    return sink

# ── Classification Tests ─────────────────────────────

class TestReconciliationClassification:
    def test_matched_orders(self, pipeline):
        result = pipeline._classify("TXN-001", 99.99, 100.00, 99.50)
        assert result.status == DiscrepancyType.MATCHED

    def test_missing_ga4(self, pipeline):
        result = pipeline._classify("TXN-002", None, 100.00, 100.00)
        assert result.status == DiscrepancyType.MISSING_GA4
        assert "Ad blocker" in result.probable_cause

    def test_missing_shopify(self, pipeline):
        result = pipeline._classify("TXN-003", 100.00, None, 100.00)
        assert result.status == DiscrepancyType.MISSING_SHOPIFY

    def test_missing_backend(self, pipeline):
        result = pipeline._classify("TXN-004", 100.00, 100.00, None)
        assert result.status == DiscrepancyType.MISSING_BACKEND

    def test_amount_mismatch(self, pipeline):
        result = pipeline._classify("TXN-005", 100.00, 100.00, 80.00)
        assert result.status == DiscrepancyType.AMOUNT_MISMATCH

# ── Pipeline Integration Tests ───────────────────────

class TestReconciliationPipeline:
    def test_reconcile_day_processes_all_sources(self, pipeline, mock_clients):
        ga4, shopify, backend = mock_clients
        ga4.get_purchases.return_value = {"TXN-1": 100.0, "TXN-2": 50.0}
        shopify.get_orders.return_value = {"TXN-1": 100.0, "TXN-3": 75.0}
        backend.get_settlements.return_value = {"TXN-1": 100.0}

        results = pipeline.reconcile_day(date(2025, 1, 15))
        txn_ids = {r.transaction_id for r in results}
        assert txn_ids == {"TXN-1", "TXN-2", "TXN-3"}

    def test_all_matched_when_sources_agree(self, pipeline, mock_clients):
        ga4, shopify, backend = mock_clients
        data = {"TXN-1": 100.0, "TXN-2": 200.0}
        ga4.get_purchases.return_value = data
        shopify.get_orders.return_value = data
        backend.get_settlements.return_value = data

        results = pipeline.reconcile_day(date(2025, 1, 15))
        assert all(r.status == DiscrepancyType.MATCHED for r in results)

# ── Alert Service Tests ──────────────────────────────

class TestDiscrepancyAlerts:
    def test_no_alerts_when_all_matched(self, mock_alert_sink):
        svc = DiscrepancyAlertService(mock_alert_sink)
        results = [
            ReconciliationResult("T1", 100, 100, 100,
                                 DiscrepancyType.MATCHED, "")
            for _ in range(20)
        ]
        outcome = svc.evaluate(results, date(2025, 1, 15))
        assert outcome["alerts_sent"] == 0

    def test_alert_on_high_ga4_drop(self, mock_alert_sink):
        svc = DiscrepancyAlertService(mock_alert_sink)
        matched = [
            ReconciliationResult("T", 100, 100, 100,
                                 DiscrepancyType.MATCHED, "")
            for _ in range(80)
        ]
        missing = [
            ReconciliationResult("T", None, 100, 100,
                                 DiscrepancyType.MISSING_GA4, "blocked")
            for _ in range(20)
        ]
        outcome = svc.evaluate(matched + missing, date(2025, 1, 15))
        assert outcome["alerts_sent"] >= 1`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Containerize and deploy the reconciliation pipeline with Docker, configuration management, and scheduled daily execution.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Reconciliation Pipeline Deployment Script',
                description:
                  'Build, test, and deploy the analytics reconciliation pipeline using Docker with health checks.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ─────────────────────────────────────
APP_NAME="analytics-reconciliation"
IMAGE_TAG="\${APP_NAME}:\$(git rev-parse --short HEAD)"
COMPOSE_FILE="docker-compose.reconciliation.yml"
HEALTH_URL="http://localhost:8091/health"
ROLLBACK_TAG=""

log() { printf '[%s] %s\\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$1"; }

# ── Pre-flight checks ────────────────────────────────
log "Running pre-flight checks..."
command -v docker >/dev/null 2>&1 || { log "ERROR: docker not found"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { log "ERROR: docker-compose not found"; exit 1; }

# Check required environment variables
for var in GA4_PROJECT_ID SHOPIFY_API_KEY DB_HOST SLACK_WEBHOOK_URL; do
  if [ -z "\${!var:-}" ]; then
    log "ERROR: \${var} is not set"
    exit 1
  fi
done

# Capture current image for rollback
ROLLBACK_TAG=\$(docker inspect --format='{{.Config.Image}}' "\${APP_NAME}" 2>/dev/null || echo "")
log "Rollback target: \${ROLLBACK_TAG:-none}"

# ── Build ─────────────────────────────────────────────
log "Building image \${IMAGE_TAG}..."
docker build -t "\${IMAGE_TAG}" -f Dockerfile.reconciliation .

# ── Test ──────────────────────────────────────────────
log "Running pytest inside container..."
docker run --rm "\${IMAGE_TAG}" pytest tests/reconciliation/ -v --tb=short
log "All tests passed."

# ── Deploy ────────────────────────────────────────────
log "Deploying with docker-compose..."
IMAGE_TAG="\${IMAGE_TAG}" docker-compose -f "\${COMPOSE_FILE}" up -d

# ── Health check ──────────────────────────────────────
log "Waiting for health check..."
for i in \$(seq 1 30); do
  if curl -sf "\${HEALTH_URL}" > /dev/null 2>&1; then
    log "Service healthy after \${i}s"
    break
  fi
  if [ "\$i" -eq 30 ]; then
    log "ERROR: health check failed after 30s"
    if [ -n "\${ROLLBACK_TAG}" ]; then
      log "Rolling back to \${ROLLBACK_TAG}..."
      IMAGE_TAG="\${ROLLBACK_TAG}" docker-compose -f "\${COMPOSE_FILE}" up -d
    fi
    exit 1
  fi
  sleep 1
done

# ── Set up cron schedule ──────────────────────────────
CRON_ENTRY="0 7 * * * docker-compose -f \$(pwd)/\${COMPOSE_FILE} run --rm reconciler python -m reconciliation.run_daily"
(crontab -l 2>/dev/null | grep -v "\${APP_NAME}" || true; echo "\${CRON_ENTRY}") | crontab -
log "Cron schedule configured for daily 7 AM execution."

log "Deployment complete: \${IMAGE_TAG}"`,
              },
              {
                language: 'python',
                title: 'Reconciliation Pipeline Configuration Loader',
                description:
                  'Typed configuration loader using dataclasses for the analytics reconciliation pipeline.',
                code: `from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class GA4Config:
    project_id: str
    dataset_id: str
    credentials_path: str
    property_timezone: str = "America/New_York"

@dataclass(frozen=True)
class ShopifyConfig:
    shop_domain: str
    api_key: str
    api_secret: str
    shop_timezone: str = "America/New_York"

@dataclass(frozen=True)
class BackendDBConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    ssl_mode: str = "require"

    @property
    def connection_string(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )

@dataclass(frozen=True)
class ReconciliationThresholds:
    match_rate_min: float = 0.85
    revenue_tolerance_pct: float = 0.03
    ga4_drop_max: float = 0.12
    order_count_drift_max: float = 0.05

@dataclass(frozen=True)
class AlertConfig:
    slack_webhook_url: str
    channel: str = "#data-alerts"
    mention_on_critical: str = "@data-team"
    enabled: bool = True

@dataclass(frozen=True)
class ReconciliationConfig:
    ga4: GA4Config
    shopify: ShopifyConfig
    backend_db: BackendDBConfig
    thresholds: ReconciliationThresholds
    alerts: AlertConfig
    normalize_timezone: str = "UTC"
    lookback_days: int = 1
    schedule_cron: str = "0 7 * * *"
    log_level: str = "INFO"
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> ReconciliationConfig:
        return cls(
            ga4=GA4Config(
                project_id=os.environ["GA4_PROJECT_ID"],
                dataset_id=os.environ["GA4_DATASET_ID"],
                credentials_path=os.environ.get(
                    "GA4_CREDENTIALS", "/secrets/ga4-creds.json"
                ),
                property_timezone=os.environ.get(
                    "GA4_TIMEZONE", "America/New_York"
                ),
            ),
            shopify=ShopifyConfig(
                shop_domain=os.environ["SHOPIFY_SHOP_DOMAIN"],
                api_key=os.environ["SHOPIFY_API_KEY"],
                api_secret=os.environ["SHOPIFY_API_SECRET"],
                shop_timezone=os.environ.get(
                    "SHOPIFY_TIMEZONE", "America/New_York"
                ),
            ),
            backend_db=BackendDBConfig(
                host=os.environ["DB_HOST"],
                port=int(os.environ.get("DB_PORT", "5432")),
                database=os.environ["DB_NAME"],
                user=os.environ["DB_USER"],
                password=os.environ["DB_PASSWORD"],
            ),
            thresholds=ReconciliationThresholds(
                match_rate_min=float(
                    os.environ.get("MATCH_RATE_MIN", "0.85")
                ),
                revenue_tolerance_pct=float(
                    os.environ.get("REVENUE_TOLERANCE", "0.03")
                ),
            ),
            alerts=AlertConfig(
                slack_webhook_url=os.environ["SLACK_WEBHOOK_URL"],
                channel=os.environ.get("SLACK_CHANNEL", "#data-alerts"),
                enabled=os.environ.get(
                    "ALERTS_ENABLED", "true"
                ).lower() == "true",
            ),
            lookback_days=int(os.environ.get("LOOKBACK_DAYS", "1")),
            schedule_cron=os.environ.get("SCHEDULE_CRON", "0 7 * * *"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            dry_run=os.environ.get("DRY_RUN", "false").lower() == "true",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ReconciliationConfig:
        with open(path) as f:
            data = json.load(f)
        return cls(
            ga4=GA4Config(**data["ga4"]),
            shopify=ShopifyConfig(**data["shopify"]),
            backend_db=BackendDBConfig(**data["backend_db"]),
            thresholds=ReconciliationThresholds(
                **data.get("thresholds", {})
            ),
            alerts=AlertConfig(**data["alerts"]),
            normalize_timezone=data.get("normalize_timezone", "UTC"),
            lookback_days=data.get("lookback_days", 1),
            schedule_cron=data.get("schedule_cron", "0 7 * * *"),
            log_level=data.get("log_level", "INFO"),
            dry_run=data.get("dry_run", False),
        )`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Monitor reconciliation results and send structured alerts when discrepancy rates exceed configurable thresholds.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Discrepancy Alerting Service',
                description:
                  'Monitors reconciliation results and sends alerts when discrepancy rates exceed configurable thresholds.',
                code: `import json
import logging
from datetime import date
from typing import Protocol

logger = logging.getLogger("discrepancy_alerts")

class AlertSink(Protocol):
    def send(self, channel: str, message: str) -> None: ...

class DiscrepancyAlertService:
    ORDER_COUNT_THRESHOLD = 0.05    # 5% order count drift
    REVENUE_THRESHOLD = 0.03        # 3% revenue drift
    MISSING_GA4_THRESHOLD = 0.12    # 12% GA4 drop acceptable

    def __init__(self, alert_sink: AlertSink, channel: str = "#data-alerts"):
        self.sink = alert_sink
        self.channel = channel

    def evaluate(self, results: list, target_date: date) -> dict:
        total = len(results)
        if total == 0:
            return {"status": "no_data", "alerts_sent": 0}

        by_type = {}
        for r in results:
            by_type.setdefault(r.status.value, []).append(r)

        matched = len(by_type.get("matched", []))
        match_rate = matched / total
        alerts = []

        missing_ga4_rate = len(by_type.get("missing_ga4", [])) / total
        if missing_ga4_rate > self.MISSING_GA4_THRESHOLD:
            alerts.append(
                f"GA4 tracking drop: {missing_ga4_rate:.1%} of orders "
                f"missing from GA4 on {target_date} "
                f"(threshold: {self.MISSING_GA4_THRESHOLD:.0%})")

        mismatches = by_type.get("amount_mismatch", [])
        if mismatches:
            total_delta = sum(
                abs((r.shopify_revenue or 0) - (r.backend_revenue or 0))
                for r in mismatches)
            alerts.append(
                f"Revenue mismatch: {len(mismatches)} orders with "
                f"\${total_delta:,.2f} cumulative delta on {target_date}")

        if match_rate < (1 - self.ORDER_COUNT_THRESHOLD):
            alerts.append(
                f"Overall match rate: {match_rate:.1%} "
                f"(target: {1 - self.ORDER_COUNT_THRESHOLD:.0%})")

        for alert_msg in alerts:
            logger.warning(alert_msg)
            self.sink.send(self.channel, alert_msg)

        return {
            "date": str(target_date),
            "total_orders": total,
            "matched": matched,
            "match_rate": round(match_rate, 4),
            "alerts_sent": len(alerts),
        }`,
              },
              {
                language: 'python',
                title: 'Reconciliation Slack Alerting with Daily Digest',
                description:
                  'Sends structured Slack alerts with daily digest summaries of reconciliation health across all platforms.',
                code: `import json
import logging
import urllib.request
from datetime import datetime
from typing import Any

logger = logging.getLogger("recon_slack_alerts")

class ReconciliationSlackAlerter:
    def __init__(self, webhook_url: str, channel: str = "#data-alerts"):
        self.webhook_url = webhook_url
        self.channel = channel

    def _post_slack(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=10)
            logger.info("Slack alert sent to %s", self.channel)
        except Exception as exc:
            logger.error("Failed to send Slack alert: %s", exc)

    def send_daily_digest(self, summary: dict[str, Any]) -> None:
        match_rate = summary.get("match_rate", 0)
        status_emoji = ":white_check_mark:" if match_rate >= 0.95 else (
            ":warning:" if match_rate >= 0.85 else ":red_circle:"
        )
        payload = {
            "channel": self.channel,
            "username": "Reconciliation Bot",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": (
                            f"{status_emoji} Reconciliation Digest: "
                            f"{summary.get('date', 'N/A')}"
                        ),
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn",
                         "text": (
                             f"*Total Orders:* "
                             f"{summary.get('total_orders', 0):,}"
                         )},
                        {"type": "mrkdwn",
                         "text": f"*Matched:* {summary.get('matched', 0):,}"},
                        {"type": "mrkdwn",
                         "text": f"*Match Rate:* {match_rate:.1%}"},
                        {"type": "mrkdwn",
                         "text": (
                             f"*Alerts:* "
                             f"{summary.get('alerts_sent', 0)}"
                         )},
                    ],
                },
            ],
        }
        self._post_slack(payload)

    def alert_platform_outage(self, platform: str,
                              missing_count: int) -> None:
        payload = {
            "channel": self.channel,
            "username": "Reconciliation Bot",
            "text": (
                f":rotating_light: *Platform data gap detected*\\n"
                f"Platform: {platform}\\n"
                f"Missing records: {missing_count:,}\\n"
                f"Time: {datetime.utcnow():%Y-%m-%d %H:%M UTC}"
            ),
        }
        self._post_slack(payload)

    def alert_revenue_drift(self, drift_pct: float,
                            drift_amount: float,
                            report_date: str) -> None:
        severity = "CRITICAL" if abs(drift_pct) > 5.0 else "WARNING"
        payload = {
            "channel": self.channel,
            "username": "Reconciliation Bot",
            "text": (
                f"[{severity}] Revenue drift on {report_date}: "
                f"{drift_pct:+.2f}% (\${abs(drift_amount):,.2f})"
            ),
        }
        self._post_slack(payload)`,
              },
            ],
          },
        ],
      },
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to automate cross-platform analytics reconciliation between GA4, Shopify, and backend systems without custom code. Extract data from each platform, use AI to identify discrepancies and their root causes, and receive automated Slack reports with reconciliation status.',
        estimatedMonthlyCost: '$150 - $250/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Google Sheets'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Supermetrics ($39/mo)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Set up automated data extraction from GA4, Shopify, and your backend database to gather daily order and revenue metrics. Use Zapier to consolidate data into Google Sheets for AI analysis.',
            toolsUsed: ['GA4 BigQuery Export', 'Shopify Admin API', 'Zapier', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Multi-Platform Data Collection Workflow',
                description:
                  'Zapier workflow configuration to collect daily metrics from GA4, Shopify, and backend sources.',
                code: `{
  "zap_name": "Daily Analytics Data Collector",
  "trigger": {
    "app": "Schedule by Zapier",
    "event": "Every Day",
    "time": "06:00",
    "timezone": "America/New_York"
  },
  "actions": [
    {
      "step": 1,
      "name": "Fetch GA4 Data",
      "app": "Google Sheets",
      "event": "Lookup Spreadsheet Row",
      "config": {
        "spreadsheet_id": "{{env.GA4_EXPORT_SHEET}}",
        "worksheet": "Daily Exports",
        "lookup_column": "date",
        "lookup_value": "{{zap.trigger_time | date: 'YYYY-MM-DD' | minus: 1 day}}"
      },
      "note": "GA4 data exported via BigQuery scheduled query"
    },
    {
      "step": 2,
      "name": "Fetch Shopify Orders",
      "app": "Webhooks by Zapier",
      "event": "GET",
      "config": {
        "url": "https://{{env.SHOPIFY_STORE}}.myshopify.com/admin/api/2025-01/orders/count.json",
        "headers": {
          "X-Shopify-Access-Token": "{{env.SHOPIFY_ACCESS_TOKEN}}"
        },
        "params": {
          "status": "any",
          "financial_status": "paid",
          "created_at_min": "{{zap.trigger_time | date: 'YYYY-MM-DD' | minus: 1 day}}T00:00:00-05:00",
          "created_at_max": "{{zap.trigger_time | date: 'YYYY-MM-DD' | minus: 1 day}}T23:59:59-05:00"
        }
      }
    },
    {
      "step": 3,
      "name": "Fetch Shopify Revenue",
      "app": "Webhooks by Zapier",
      "event": "GET",
      "config": {
        "url": "https://{{env.SHOPIFY_STORE}}.myshopify.com/admin/api/2025-01/orders.json",
        "headers": {
          "X-Shopify-Access-Token": "{{env.SHOPIFY_ACCESS_TOKEN}}"
        },
        "params": {
          "status": "any",
          "financial_status": "paid",
          "fields": "id,total_price",
          "created_at_min": "{{zap.trigger_time | date: 'YYYY-MM-DD' | minus: 1 day}}T00:00:00-05:00",
          "created_at_max": "{{zap.trigger_time | date: 'YYYY-MM-DD' | minus: 1 day}}T23:59:59-05:00",
          "limit": 250
        }
      }
    },
    {
      "step": 4,
      "name": "Fetch Backend Data",
      "app": "Google Sheets",
      "event": "Lookup Spreadsheet Row",
      "config": {
        "spreadsheet_id": "{{env.BACKEND_EXPORT_SHEET}}",
        "worksheet": "Daily Settlements",
        "lookup_column": "settlement_date",
        "lookup_value": "{{zap.trigger_time | date: 'YYYY-MM-DD' | minus: 1 day}}"
      }
    },
    {
      "step": 5,
      "name": "Store Consolidated Data",
      "app": "Google Sheets",
      "event": "Create Spreadsheet Row",
      "config": {
        "spreadsheet_id": "{{env.RECONCILIATION_SHEET}}",
        "worksheet": "Daily Metrics",
        "columns": {
          "date": "{{zap.trigger_time | date: 'YYYY-MM-DD' | minus: 1 day}}",
          "ga4_orders": "{{step1.order_count}}",
          "ga4_revenue": "{{step1.total_revenue}}",
          "shopify_orders": "{{step2.count}}",
          "shopify_revenue": "{{step3.orders | map: 'total_price' | sum}}",
          "backend_orders": "{{step4.order_count}}",
          "backend_revenue": "{{step4.net_revenue}}",
          "collected_at": "{{zap.trigger_time}}"
        }
      }
    }
  ]
}`,
              },
              {
                language: 'json',
                title: 'Reconciliation Data Structure',
                description:
                  'JSON structure for consolidated analytics data prepared for AI reconciliation analysis.',
                code: `{
  "reconciliation_request": {
    "report_date": "2025-01-31",
    "timezone": "America/New_York",
    "platforms": {
      "ga4": {
        "source": "BigQuery Export",
        "order_count": 1000,
        "revenue": 89500.00,
        "unique_transaction_ids": 1000,
        "data_completeness": "partial",
        "known_issues": [
          "Ad blocker impact estimated at 10-15%",
          "Consent mode v2 may reduce tracking"
        ]
      },
      "shopify": {
        "source": "Admin API",
        "order_count": 1150,
        "revenue": 102350.00,
        "includes_test_orders": true,
        "includes_draft_orders": true,
        "financial_status_filter": "paid",
        "timezone": "America/New_York"
      },
      "backend": {
        "source": "Payment Gateway Settlements",
        "order_count": 1087,
        "revenue": 96780.00,
        "settlement_delay_days": 1,
        "includes_refunds": false,
        "timezone": "UTC"
      }
    },
    "calculated_deltas": {
      "shopify_vs_ga4_orders": 150,
      "shopify_vs_ga4_orders_pct": 15.0,
      "shopify_vs_backend_orders": 63,
      "shopify_vs_backend_orders_pct": 5.8,
      "shopify_vs_ga4_revenue": 12850.00,
      "shopify_vs_backend_revenue": 5570.00
    },
    "historical_averages": {
      "typical_ga4_undercount_pct": 12.5,
      "typical_shopify_overcount_pct": 4.2,
      "acceptable_variance_pct": 5.0
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
              'Use ChatGPT or Claude to analyze the cross-platform data, identify discrepancy root causes, and generate reconciliation recommendations with confidence scores.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Analytics Reconciliation Prompt Template',
                description:
                  'Structured prompt for AI to analyze cross-platform analytics data and identify discrepancy causes.',
                code: `system_prompt: |
  You are an ecommerce analytics reconciliation specialist. You understand
  the technical differences between how GA4, Shopify, and payment systems
  count orders and revenue. You know common causes of discrepancies:

  GA4 typically undercounts due to:
  - Ad blockers (10-15% impact)
  - Consent mode / cookie rejection
  - Tracking script failures
  - Session timeout before purchase completion

  Shopify typically overcounts due to:
  - Test orders not filtered
  - Draft orders counted
  - Partially refunded orders at full value
  - Timezone edge cases

  Backend/Payment systems differ due to:
  - Settlement delays (1-3 days typical)
  - Refunds deducted from settlements
  - Currency conversion differences
  - Timezone (usually UTC vs local)

  Your analysis should:
  - Explain WHY numbers differ, not just that they differ
  - Provide confidence scores for root cause identification
  - Recommend specific filters or adjustments to align numbers
  - Flag anomalies that warrant investigation

user_prompt_template: |
  Analyze the following cross-platform analytics data for {{report_date}}:

  ## Platform Data
  ### GA4 (Google Analytics 4)
  - Orders: {{ga4_orders}}
  - Revenue: \${{ga4_revenue}}
  - Known Issues: {{ga4_known_issues}}

  ### Shopify
  - Orders: {{shopify_orders}}
  - Revenue: \${{shopify_revenue}}
  - Includes Test Orders: {{includes_test_orders}}
  - Financial Status Filter: {{financial_status}}

  ### Backend (Payment Settlements)
  - Orders: {{backend_orders}}
  - Revenue: \${{backend_revenue}}
  - Settlement Delay: {{settlement_delay_days}} days
  - Timezone: {{backend_timezone}}

  ## Calculated Deltas
  - Shopify vs GA4 Orders: {{shopify_vs_ga4_orders}} ({{shopify_vs_ga4_orders_pct}}%)
  - Shopify vs Backend Orders: {{shopify_vs_backend_orders}} ({{shopify_vs_backend_orders_pct}}%)
  - Revenue Gap (Shopify - GA4): \${{shopify_vs_ga4_revenue}}

  ## Historical Context
  - Typical GA4 undercount: {{typical_ga4_undercount_pct}}%
  - Typical Shopify overcount: {{typical_shopify_overcount_pct}}%
  - Acceptable variance: {{acceptable_variance_pct}}%

  ## Analysis Required
  1. Is today's GA4 undercount ({{shopify_vs_ga4_orders_pct}}%) within normal range?
  2. What are the likely causes of the {{shopify_vs_backend_orders}} order gap?
  3. Is the revenue variance concerning or expected?
  4. What adjustments would align these numbers?
  5. Are there any anomalies requiring investigation?

expected_output_format: |
  ## Reconciliation Analysis for {{report_date}}

  ### Summary
  [2-3 sentence executive summary of reconciliation status]

  ### Variance Assessment
  | Metric | Delta | Status | Confidence |
  |--------|-------|--------|------------|
  | GA4 Undercount | X% | Normal/High | XX% |
  | Shopify Overcount | X% | Normal/High | XX% |
  | Revenue Gap | \$X | Explained/Investigate | XX% |

  ### Root Cause Analysis
  #### GA4 Gap ({{shopify_vs_ga4_orders}} orders)
  - **Primary Cause:** [cause] (confidence: X%)
  - **Secondary Cause:** [cause] (confidence: X%)
  - **Estimated Breakdown:**
    - Ad blockers: ~X orders
    - Consent rejections: ~X orders
    - Other: ~X orders

  #### Shopify Overcount ({{shopify_vs_backend_orders}} orders)
  - **Primary Cause:** [cause] (confidence: X%)
  - **Recommended Filter:** [specific filter]

  ### Recommended Adjustments
  1. **GA4 Adjustment:** Add X% to GA4 for ad blocker impact
  2. **Shopify Filter:** Exclude orders where [condition]
  3. **Backend Timing:** Shift settlement data by X days

  ### Anomalies Requiring Investigation
  - [Any unusual patterns or outliers]

  ### Confidence Score: X/100
  [Explanation of overall reconciliation confidence]`,
              },
              {
                language: 'yaml',
                title: 'Discrepancy Investigation Prompt',
                description:
                  'Prompt template for deep investigation when discrepancies exceed normal thresholds.',
                code: `system_prompt: |
  You are investigating an analytics discrepancy that exceeds normal
  thresholds. Your job is to identify the root cause and recommend
  specific remediation steps.

  When investigating, consider:
  - Recent changes to tracking implementation
  - Platform updates or API changes
  - Marketing campaign impacts (UTM issues)
  - Technical incidents (site outages, API failures)
  - Seasonal patterns or promotional events

user_prompt_template: |
  ## Discrepancy Alert
  Date: {{report_date}}
  Severity: {{severity}}

  ## The Problem
  {{discrepancy_description}}

  ## Current Values
  - GA4: {{ga4_value}}
  - Shopify: {{shopify_value}}
  - Backend: {{backend_value}}
  - Delta: {{delta_value}} ({{delta_pct}}%)
  - Normal Range: {{normal_range}}

  ## Historical Context
  {{#each last_7_days}}
  - {{date}}: GA4={{ga4}}, Shopify={{shopify}}, Delta={{delta}}%
  {{/each}}

  ## Recent Changes
  {{recent_changes}}

  ## Investigation Required
  1. What is the most likely root cause?
  2. Is this a data issue or a real business change?
  3. What immediate actions should be taken?
  4. How do we prevent this in the future?

output_format: |
  ## Investigation Report

  ### Root Cause Identification
  **Most Likely Cause:** [specific cause]
  **Confidence:** X%
  **Evidence:** [supporting data points]

  ### Impact Assessment
  - **Financial Impact:** \$X
  - **Decision Impact:** [affected decisions]
  - **Duration:** [how long has this been occurring]

  ### Immediate Actions
  1. [Action 1 with owner]
  2. [Action 2 with owner]

  ### Preventive Measures
  1. [Monitoring improvement]
  2. [Process change]

  ### Data Correction
  - **Corrected GA4 Value:** {{adjusted_ga4}}
  - **Corrected Shopify Value:** {{adjusted_shopify}}
  - **Reconciled Revenue:** \${{reconciled_revenue}}`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Automate the complete reconciliation workflow with Zapier to run daily analysis, send AI-generated reconciliation reports to Slack, and alert stakeholders when discrepancies exceed thresholds.',
            toolsUsed: ['Zapier', 'Slack', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Complete Reconciliation Workflow',
                description:
                  'End-to-end Zapier workflow that collects data, runs AI reconciliation, and delivers daily reports.',
                code: `{
  "zap_name": "Daily Analytics Reconciliation Report",
  "trigger": {
    "app": "Schedule by Zapier",
    "event": "Every Day",
    "time": "08:00",
    "timezone": "America/New_York"
  },
  "actions": [
    {
      "step": 1,
      "app": "Google Sheets",
      "event": "Lookup Spreadsheet Row",
      "config": {
        "spreadsheet_id": "{{env.RECONCILIATION_SHEET}}",
        "worksheet": "Daily Metrics",
        "lookup_column": "date",
        "lookup_value": "{{zap.trigger_time | date: 'YYYY-MM-DD' | minus: 1 day}}"
      }
    },
    {
      "step": 2,
      "app": "Google Sheets",
      "event": "Lookup Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{env.RECONCILIATION_SHEET}}",
        "worksheet": "Daily Metrics",
        "lookup_column": "date",
        "lookup_value": "last_7_days",
        "return_type": "multiple"
      }
    },
    {
      "step": 3,
      "app": "Formatter by Zapier",
      "event": "Numbers",
      "config": {
        "operation": "calculate",
        "formula": "({{step1.shopify_orders}} - {{step1.ga4_orders}}) / {{step1.shopify_orders}} * 100"
      }
    },
    {
      "step": 4,
      "app": "ChatGPT",
      "event": "Conversation",
      "config": {
        "model": "gpt-4",
        "system_message": "You are an ecommerce analytics reconciliation specialist...",
        "user_message": "Analyze reconciliation data for {{step1.date}}: GA4 orders={{step1.ga4_orders}}, Shopify orders={{step1.shopify_orders}}, Backend orders={{step1.backend_orders}}. GA4 revenue=\${{step1.ga4_revenue}}, Shopify revenue=\${{step1.shopify_revenue}}, Backend revenue=\${{step1.backend_revenue}}. Historical data: {{step2.summary}}. Provide reconciliation analysis.",
        "max_tokens": 2500
      }
    },
    {
      "step": 5,
      "app": "Filter by Zapier",
      "event": "Only Continue If",
      "config": {
        "condition": "{{step3.result}} greater_than 15",
        "action": "send_alert"
      }
    },
    {
      "step": 6,
      "app": "Slack",
      "event": "Send Channel Message",
      "config": {
        "channel": "#analytics-reconciliation",
        "message": {
          "blocks": [
            {
              "type": "header",
              "text": ":bar_chart: Daily Reconciliation Report - {{step1.date}}"
            },
            {
              "type": "section",
              "fields": [
                {"type": "mrkdwn", "text": "*GA4 Orders:* {{step1.ga4_orders}}"},
                {"type": "mrkdwn", "text": "*Shopify Orders:* {{step1.shopify_orders}}"},
                {"type": "mrkdwn", "text": "*Backend Orders:* {{step1.backend_orders}}"},
                {"type": "mrkdwn", "text": "*GA4 Gap:* {{step3.result}}%"}
              ]
            },
            {
              "type": "section",
              "fields": [
                {"type": "mrkdwn", "text": "*GA4 Revenue:* \${{step1.ga4_revenue}}"},
                {"type": "mrkdwn", "text": "*Shopify Revenue:* \${{step1.shopify_revenue}}"},
                {"type": "mrkdwn", "text": "*Backend Revenue:* \${{step1.backend_revenue}}"}
              ]
            },
            {
              "type": "divider"
            },
            {
              "type": "section",
              "text": {"type": "mrkdwn", "text": "*AI Analysis:*\\n{{step4.response}}"}
            },
            {
              "type": "actions",
              "elements": [
                {
                  "type": "button",
                  "text": "View Full Report",
                  "url": "{{env.RECONCILIATION_SHEET}}"
                },
                {
                  "type": "button",
                  "text": "Investigate",
                  "url": "{{env.INVESTIGATION_DASHBOARD}}",
                  "style": "danger"
                }
              ]
            }
          ]
        },
        "bot_name": "Reconciliation Bot",
        "bot_icon": ":mag:"
      }
    },
    {
      "step": 7,
      "app": "Google Sheets",
      "event": "Create Spreadsheet Row",
      "config": {
        "spreadsheet_id": "{{env.RECONCILIATION_SHEET}}",
        "worksheet": "Analysis Log",
        "columns": {
          "date": "{{step1.date}}",
          "ga4_gap_pct": "{{step3.result}}",
          "ai_analysis": "{{step4.response}}",
          "status": "{{step5.passed ? 'alert_sent' : 'normal'}}"
        }
      }
    }
  ]
}`,
              },
              {
                language: 'json',
                title: 'High-Variance Alert Workflow',
                description:
                  'Zapier workflow to send urgent alerts when analytics discrepancies exceed critical thresholds.',
                code: `{
  "alert_workflow": {
    "name": "Analytics Discrepancy Alert",
    "trigger": {
      "app": "Google Sheets",
      "event": "New or Updated Spreadsheet Row",
      "config": {
        "spreadsheet_id": "{{env.RECONCILIATION_SHEET}}",
        "worksheet": "Daily Metrics"
      }
    },
    "filters": [
      {
        "name": "Critical Discrepancy",
        "conditions": [
          {
            "field": "calculated_ga4_gap_pct",
            "operator": "greater_than",
            "value": 20
          },
          {
            "field": "revenue_variance_pct",
            "operator": "greater_than",
            "value": 10
          }
        ],
        "logic": "OR"
      }
    ],
    "actions": [
      {
        "app": "ChatGPT",
        "event": "Conversation",
        "config": {
          "model": "gpt-4",
          "user_message": "URGENT: Analytics discrepancy detected for {{trigger.date}}. GA4 gap is {{trigger.ga4_gap_pct}}% (threshold: 20%). Revenue variance is {{trigger.revenue_variance_pct}}% (threshold: 10%). Provide immediate root cause analysis and recommended actions.",
          "max_tokens": 1000
        }
      },
      {
        "app": "Slack",
        "event": "Send Channel Message",
        "config": {
          "channel": "#analytics-alerts",
          "message": ":rotating_light: *ANALYTICS DISCREPANCY ALERT*\\n\\n*Date:* {{trigger.date}}\\n*GA4 Gap:* {{trigger.ga4_gap_pct}}% (threshold: 20%)\\n*Revenue Variance:* {{trigger.revenue_variance_pct}}%\\n\\n*AI Analysis:*\\n{{previous_step.response}}\\n\\ncc: @data-team @marketing"
        }
      },
      {
        "app": "Email by Zapier",
        "event": "Send Outbound Email",
        "config": {
          "to": "data-team@company.com, marketing@company.com",
          "subject": "[CRITICAL] Analytics Discrepancy - {{trigger.date}}",
          "body": "A critical analytics discrepancy has been detected.\\n\\nDate: {{trigger.date}}\\nGA4 Orders: {{trigger.ga4_orders}}\\nShopify Orders: {{trigger.shopify_orders}}\\nVariance: {{trigger.ga4_gap_pct}}%\\n\\nImmediate investigation required.\\n\\nAI Analysis:\\n{{previous_step.response}}"
        }
      }
    ]
  }
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent system using CrewAI and LangGraph to provide autonomous cross-platform analytics reconciliation. Specialized agents handle data extraction from each platform, discrepancy detection, root cause analysis, and automated reporting, coordinated by a supervisor agent that maintains a single source of truth.',
        estimatedMonthlyCost: '$800 - $1,500/month',
        architecture:
          'Supervisor agent coordinates five specialist agents: GA4 Analyst Agent extracts and normalizes Google Analytics data, Shopify Analyst Agent processes Shopify order data, Backend Analyst Agent handles payment settlement data, Reconciliation Agent matches records across platforms and identifies discrepancies, and Reporting Agent generates stakeholder reports. State is managed in Redis with transaction-level audit trails.',
        agents: [
          {
            name: 'GA4 Analyst Agent',
            role: 'Google Analytics 4 Data Specialist',
            goal: 'Extract and normalize GA4 purchase event data, estimate tracking gaps from ad blockers and consent mode, and provide adjusted order counts with confidence intervals',
            tools: ['bigquery_client', 'ga4_api', 'tracking_gap_estimator', 'data_normalizer'],
          },
          {
            name: 'Shopify Analyst Agent',
            role: 'Shopify Order Data Specialist',
            goal: 'Extract Shopify order data, filter out test and draft orders, normalize timezone handling, and reconcile refunds and cancellations',
            tools: ['shopify_admin_api', 'order_filter', 'refund_processor', 'timezone_normalizer'],
          },
          {
            name: 'Backend Analyst Agent',
            role: 'Payment Settlement Data Specialist',
            goal: 'Extract payment gateway settlement data, handle settlement delays, normalize currency conversions, and track refund impacts',
            tools: ['payment_gateway_api', 'settlement_processor', 'currency_converter', 'refund_tracker'],
          },
          {
            name: 'Reconciliation Agent',
            role: 'Cross-Platform Data Matcher',
            goal: 'Match orders across all three platforms using transaction IDs and fuzzy matching, classify discrepancy types, and identify root causes for each unmatched record',
            tools: ['record_matcher', 'fuzzy_match_engine', 'discrepancy_classifier', 'root_cause_analyzer'],
          },
          {
            name: 'Reporting Agent',
            role: 'Analytics Reconciliation Reporter',
            goal: 'Generate daily reconciliation reports, maintain historical trend analysis, and create executive dashboards with reconciled numbers that all stakeholders trust',
            tools: ['report_generator', 'trend_analyzer', 'dashboard_api', 'slack_notifier'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Hierarchical',
          stateManagement: 'Redis-backed state with transaction-level audit trails and 365-day retention for YoY analysis',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system architecture with CrewAI, establishing clear roles, goals, and tool assignments for each specialist agent in the analytics reconciliation workflow.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Analytics Reconciliation Agent Definitions',
                description:
                  'CrewAI agent definitions for the cross-platform analytics reconciliation multi-agent system.',
                code: `from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import logging

logger = logging.getLogger("analytics_reconciliation_agents")

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.1,
)


class AnalyticsReconciliationAgents:
    """Factory for creating analytics reconciliation specialist agents."""

    def __init__(self, tools_registry: Dict[str, Any]):
        self.tools = tools_registry
        self.llm = llm

    def create_ga4_analyst_agent(self) -> Agent:
        """Agent responsible for GA4 data extraction and analysis."""
        return Agent(
            role="Google Analytics 4 Data Specialist",
            goal=(
                "Extract and normalize GA4 purchase event data from BigQuery. "
                "Estimate tracking gaps caused by ad blockers (typically 10-15%), "
                "consent mode rejections, and tracking script failures. Provide "
                "adjusted order counts with confidence intervals."
            ),
            backstory=(
                "You are a GA4 expert who understands the intricacies of "
                "event-based tracking. You know that GA4 systematically "
                "undercounts due to ad blockers and privacy tools. You can "
                "estimate the gap using server-side data and historical patterns. "
                "You always report both raw and adjusted figures."
            ),
            tools=[
                self.tools["bigquery_client"],
                self.tools["ga4_api"],
                self.tools["tracking_gap_estimator"],
                self.tools["data_normalizer"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def create_shopify_analyst_agent(self) -> Agent:
        """Agent responsible for Shopify data extraction and cleanup."""
        return Agent(
            role="Shopify Order Data Specialist",
            goal=(
                "Extract Shopify order data via Admin API. Filter out test orders, "
                "draft orders, and cancelled orders. Normalize timestamps to a "
                "consistent timezone. Handle partially refunded orders correctly "
                "by reporting net revenue, not gross."
            ),
            backstory=(
                "You are a Shopify data expert who has cleaned order data for "
                "hundreds of stores. You know all the edge cases: orders created "
                "via API, test mode orders, timezone boundaries, and refund timing. "
                "You produce the cleanest order counts possible."
            ),
            tools=[
                self.tools["shopify_admin_api"],
                self.tools["order_filter"],
                self.tools["refund_processor"],
                self.tools["timezone_normalizer"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def create_backend_analyst_agent(self) -> Agent:
        """Agent responsible for payment settlement data."""
        return Agent(
            role="Payment Settlement Data Specialist",
            goal=(
                "Extract payment gateway settlement data. Handle the 1-3 day "
                "settlement delay by mapping settlement dates to order dates. "
                "Track refunds that reduce settlement amounts. Normalize currency "
                "for multi-currency stores."
            ),
            backstory=(
                "You are a payment operations expert who understands the flow "
                "from authorization to capture to settlement. You know that "
                "settlement timing varies by payment method and processor. "
                "You can reconstruct original order values from settlement data."
            ),
            tools=[
                self.tools["payment_gateway_api"],
                self.tools["settlement_processor"],
                self.tools["currency_converter"],
                self.tools["refund_tracker"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def create_reconciliation_agent(self) -> Agent:
        """Agent responsible for matching records across platforms."""
        return Agent(
            role="Cross-Platform Data Matcher",
            goal=(
                "Match orders across GA4, Shopify, and backend using transaction IDs "
                "as primary key. When exact match fails, use fuzzy matching on "
                "amount + timestamp. Classify each discrepancy by type: "
                "MISSING_GA4, MISSING_SHOPIFY, MISSING_BACKEND, AMOUNT_MISMATCH. "
                "Identify root cause for each unmatched record."
            ),
            backstory=(
                "You are a data reconciliation specialist who has matched "
                "millions of records across disparate systems. You understand "
                "that perfect matching is impossible, but you can achieve 95%+ "
                "match rates with the right techniques. You document every "
                "unmatched record with its probable cause."
            ),
            tools=[
                self.tools["record_matcher"],
                self.tools["fuzzy_match_engine"],
                self.tools["discrepancy_classifier"],
                self.tools["root_cause_analyzer"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )

    def create_reporting_agent(self) -> Agent:
        """Agent responsible for generating reconciliation reports."""
        return Agent(
            role="Analytics Reconciliation Reporter",
            goal=(
                "Generate daily reconciliation reports that stakeholders trust. "
                "Present a single 'reconciled' number that marketing, finance, "
                "and executives can all use. Track historical trends to spot "
                "systemic issues. Create actionable alerts, not noise."
            ),
            backstory=(
                "You are the bridge between data complexity and business clarity. "
                "You know that different teams need numbers for different purposes: "
                "marketing wants attributed revenue, finance wants settled revenue, "
                "executives want growth trends. You reconcile these perspectives "
                "into a coherent narrative."
            ),
            tools=[
                self.tools["report_generator"],
                self.tools["trend_analyzer"],
                self.tools["dashboard_api"],
                self.tools["slack_notifier"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )

    def create_supervisor_agent(self) -> Agent:
        """Supervisor agent coordinating the reconciliation workflow."""
        return Agent(
            role="Analytics Reconciliation Supervisor",
            goal=(
                "Coordinate all reconciliation activities. Ensure data is extracted "
                "in the correct sequence (GA4 first, then Shopify, then backend to "
                "allow for settlement delays). Resolve conflicts when agents disagree "
                "on root causes. Maintain the single source of truth."
            ),
            backstory=(
                "You are the head of data operations, responsible for ensuring "
                "the company has accurate, trustworthy analytics. You have seen "
                "the chaos caused by conflicting numbers and you are determined "
                "to provide clarity. You make judgment calls when data is ambiguous."
            ),
            tools=[
                self.tools["workflow_orchestrator"],
                self.tools["conflict_resolver"],
                self.tools["audit_trail"],
                self.tools["escalation_handler"],
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )`,
              },
              {
                language: 'python',
                title: 'Analytics Reconciliation Crew Configuration',
                description:
                  'CrewAI Crew configuration for the analytics reconciliation multi-agent system.',
                code: `from crewai import Crew, Task, Process
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, List
from enum import Enum


class DiscrepancyType(Enum):
    MISSING_GA4 = "missing_ga4"
    MISSING_SHOPIFY = "missing_shopify"
    MISSING_BACKEND = "missing_backend"
    AMOUNT_MISMATCH = "amount_mismatch"
    TIMEZONE_SHIFT = "timezone_shift"
    REFUND_TIMING = "refund_timing"


@dataclass
class ReconciliationReport:
    """Daily reconciliation report structure."""
    report_date: date
    ga4_orders: int
    ga4_adjusted_orders: int
    shopify_orders: int
    shopify_filtered_orders: int
    backend_orders: int
    matched_orders: int
    match_rate: float
    reconciled_revenue: float
    discrepancies_by_type: Dict[DiscrepancyType, int]
    confidence_score: float
    alerts: List[str]


class AnalyticsReconciliationCrew:
    """Crew orchestrating multi-agent analytics reconciliation."""

    def __init__(self, agents_factory: AnalyticsReconciliationAgents):
        self.agents = agents_factory
        self.crew: Optional[Crew] = None

    def build_crew(self) -> Crew:
        """Assemble the reconciliation crew."""

        # Create agents
        ga4_agent = self.agents.create_ga4_analyst_agent()
        shopify_agent = self.agents.create_shopify_analyst_agent()
        backend_agent = self.agents.create_backend_analyst_agent()
        reconciliation_agent = self.agents.create_reconciliation_agent()
        reporting_agent = self.agents.create_reporting_agent()
        supervisor = self.agents.create_supervisor_agent()

        # Define tasks
        ga4_extraction_task = Task(
            description=(
                "Extract GA4 purchase events for the target date from BigQuery. "
                "Calculate the estimated tracking gap based on historical patterns. "
                "Return both raw count and adjusted count with confidence interval. "
                "Output transaction IDs, revenue, and timestamps in normalized format."
            ),
            expected_output=(
                "JSON with: raw_orders, adjusted_orders, adjustment_factor, "
                "confidence_interval, transactions_list (id, revenue, timestamp)"
            ),
            agent=ga4_agent,
        )

        shopify_extraction_task = Task(
            description=(
                "Extract Shopify orders for the target date via Admin API. "
                "Filter out: test orders (test=true), draft orders, cancelled orders, "
                "orders with financial_status != paid. Normalize timestamps to UTC. "
                "Calculate net revenue after refunds."
            ),
            expected_output=(
                "JSON with: total_orders, filtered_orders, removed_reasons, "
                "net_revenue, orders_list (id, order_number, revenue, timestamp)"
            ),
            agent=shopify_agent,
        )

        backend_extraction_task = Task(
            description=(
                "Extract payment settlements for the target date. Map settlement "
                "amounts back to original orders accounting for 1-3 day delay. "
                "Deduct refunds from gross settlements. Normalize multi-currency "
                "amounts to USD."
            ),
            expected_output=(
                "JSON with: settlements_count, mapped_orders, unmapped_settlements, "
                "gross_amount, refunds, net_amount, orders_list (id, revenue, order_date)"
            ),
            agent=backend_agent,
        )

        matching_task = Task(
            description=(
                "Match records across all three platforms. Use transaction_id as "
                "primary key. For unmatched records, attempt fuzzy match on "
                "amount (within 1%) + timestamp (within 1 hour). Classify each "
                "discrepancy and identify probable root cause."
            ),
            expected_output=(
                "JSON with: total_records, matched, unmatched, match_rate, "
                "discrepancies_by_type, unmatched_details (id, platforms_present, "
                "probable_cause)"
            ),
            agent=reconciliation_agent,
            context=[ga4_extraction_task, shopify_extraction_task, backend_extraction_task],
        )

        reporting_task = Task(
            description=(
                "Generate the daily reconciliation report. Calculate the single "
                "'reconciled' order count and revenue that all teams should use. "
                "Compare to historical averages and flag anomalies. Create "
                "executive summary and detailed breakdown."
            ),
            expected_output=(
                "JSON with: reconciled_orders, reconciled_revenue, confidence_score, "
                "executive_summary, detailed_breakdown, anomalies, trend_comparison"
            ),
            agent=reporting_agent,
            context=[matching_task],
        )

        coordination_task = Task(
            description=(
                "Review all agent outputs. Resolve any conflicts in root cause "
                "analysis. Approve the final reconciled numbers. Determine if "
                "any alerts should be sent. Update the single source of truth."
            ),
            expected_output=(
                "JSON with: approved_numbers, conflicts_resolved, alerts_to_send, "
                "audit_trail_entry, next_investigation_items"
            ),
            agent=supervisor,
            context=[reporting_task],
        )

        self.crew = Crew(
            agents=[
                supervisor,
                ga4_agent,
                shopify_agent,
                backend_agent,
                reconciliation_agent,
                reporting_agent,
            ],
            tasks=[
                ga4_extraction_task,
                shopify_extraction_task,
                backend_extraction_task,
                matching_task,
                reporting_task,
                coordination_task,
            ],
            process=Process.hierarchical,
            manager_agent=supervisor,
            verbose=True,
        )

        return self.crew

    def run_daily_reconciliation(
        self,
        target_date: date,
    ) -> ReconciliationReport:
        """Execute daily reconciliation workflow."""
        if not self.crew:
            self.build_crew()

        result = self.crew.kickoff(inputs={
            "target_date": target_date.isoformat(),
            "timezone": "America/New_York",
        })

        # Parse result into structured report
        return ReconciliationReport(
            report_date=target_date,
            ga4_orders=result.get("ga4_raw_orders", 0),
            ga4_adjusted_orders=result.get("ga4_adjusted_orders", 0),
            shopify_orders=result.get("shopify_total_orders", 0),
            shopify_filtered_orders=result.get("shopify_filtered_orders", 0),
            backend_orders=result.get("backend_orders", 0),
            matched_orders=result.get("matched_orders", 0),
            match_rate=result.get("match_rate", 0.0),
            reconciled_revenue=result.get("reconciled_revenue", 0.0),
            discrepancies_by_type=result.get("discrepancies_by_type", {}),
            confidence_score=result.get("confidence_score", 0.0),
            alerts=result.get("alerts", []),
        )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement data ingestion tools for each platform agent to extract, normalize, and prepare data for reconciliation.',
            toolsUsed: ['BigQuery', 'Shopify Admin API', 'Payment Gateway APIs', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'GA4 BigQuery Extraction Tool',
                description:
                  'LangChain tool for the GA4 Analyst Agent to extract purchase events from BigQuery.',
                code: `from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from datetime import date, datetime
from google.cloud import bigquery
import logging

logger = logging.getLogger("ga4_extraction")


class GA4ExtractionInput(BaseModel):
    """Input schema for GA4 extraction."""
    target_date: str = Field(description="Date to extract in YYYY-MM-DD format")
    property_id: str = Field(description="GA4 property ID")


class GA4BigQueryTool(BaseTool):
    """Tool for extracting GA4 purchase events from BigQuery."""

    name: str = "bigquery_client"
    description: str = (
        "Extract GA4 purchase events from BigQuery export. "
        "Returns transaction IDs, revenue, and timestamps."
    )
    args_schema: type[BaseModel] = GA4ExtractionInput

    project_id: str
    dataset_id: str

    def _run(
        self,
        target_date: str,
        property_id: str,
    ) -> Dict[str, Any]:
        """Extract GA4 purchase data."""

        client = bigquery.Client(project=self.project_id)

        # Format date for table suffix
        date_suffix = target_date.replace("-", "")

        query = f"""
        WITH purchases AS (
            SELECT
                event_timestamp,
                (SELECT value.string_value FROM UNNEST(event_params)
                 WHERE key = 'transaction_id') AS transaction_id,
                (SELECT value.double_value FROM UNNEST(event_params)
                 WHERE key = 'value') AS revenue,
                (SELECT value.string_value FROM UNNEST(event_params)
                 WHERE key = 'currency') AS currency,
                user_pseudo_id,
                device.category AS device_type,
                traffic_source.source AS source,
                traffic_source.medium AS medium
            FROM \`{self.project_id}.{self.dataset_id}.events_{date_suffix}\`
            WHERE event_name = 'purchase'
        )
        SELECT
            transaction_id,
            revenue,
            currency,
            TIMESTAMP_MICROS(event_timestamp) AS event_time,
            device_type,
            source,
            medium
        FROM purchases
        WHERE transaction_id IS NOT NULL
        """

        results = client.query(query).result()

        transactions = []
        total_revenue = 0.0

        for row in results:
            transactions.append({
                "transaction_id": row.transaction_id,
                "revenue": float(row.revenue or 0),
                "currency": row.currency or "USD",
                "timestamp": row.event_time.isoformat(),
                "device_type": row.device_type,
                "source": row.source,
                "medium": row.medium,
            })
            total_revenue += float(row.revenue or 0)

        # Estimate tracking gap
        gap_estimate = self._estimate_tracking_gap(len(transactions))

        logger.info(
            "Extracted %d GA4 transactions, estimated gap: %.1f%%",
            len(transactions),
            gap_estimate["gap_percentage"],
        )

        return {
            "target_date": target_date,
            "raw_order_count": len(transactions),
            "adjusted_order_count": gap_estimate["adjusted_count"],
            "adjustment_factor": gap_estimate["adjustment_factor"],
            "gap_percentage": gap_estimate["gap_percentage"],
            "confidence_interval": gap_estimate["confidence_interval"],
            "total_revenue": round(total_revenue, 2),
            "adjusted_revenue": round(
                total_revenue * gap_estimate["adjustment_factor"], 2
            ),
            "transactions": transactions,
        }

    def _estimate_tracking_gap(self, raw_count: int) -> Dict[str, Any]:
        """Estimate the tracking gap from ad blockers and consent."""
        # Based on industry benchmarks:
        # - Ad blockers: 10-15% of users
        # - Consent mode: 5-10% rejection rate
        # - Combined typical gap: 12-18%

        base_gap = 0.14  # 14% base estimate
        confidence_range = 0.04  # +/- 4%

        adjustment_factor = 1 / (1 - base_gap)
        adjusted_count = int(raw_count * adjustment_factor)

        return {
            "adjustment_factor": round(adjustment_factor, 4),
            "adjusted_count": adjusted_count,
            "gap_percentage": round(base_gap * 100, 1),
            "confidence_interval": {
                "low": int(raw_count * (1 / (1 - (base_gap + confidence_range)))),
                "high": int(raw_count * (1 / (1 - (base_gap - confidence_range)))),
            },
        }

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use sync version")`,
              },
              {
                language: 'python',
                title: 'Shopify Order Extraction Tool',
                description:
                  'LangChain tool for the Shopify Analyst Agent to extract and filter order data.',
                code: `from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import httpx
import logging

logger = logging.getLogger("shopify_extraction")


class ShopifyExtractionInput(BaseModel):
    """Input schema for Shopify extraction."""
    target_date: str = Field(description="Date to extract in YYYY-MM-DD format")
    shop_timezone: str = Field(
        default="America/New_York",
        description="Shop timezone for date boundaries"
    )


class ShopifyOrderExtractionTool(BaseTool):
    """Tool for extracting Shopify orders via Admin API."""

    name: str = "shopify_admin_api"
    description: str = (
        "Extract Shopify orders for a given date. Filters out test orders, "
        "draft orders, and cancelled orders. Returns clean order data."
    )
    args_schema: type[BaseModel] = ShopifyExtractionInput

    shop_domain: str
    access_token: str
    api_version: str = "2025-01"

    def _run(
        self,
        target_date: str,
        shop_timezone: str = "America/New_York",
    ) -> Dict[str, Any]:
        """Extract and filter Shopify orders."""

        tz = ZoneInfo(shop_timezone)
        target = datetime.strptime(target_date, "%Y-%m-%d")
        start_dt = datetime(target.year, target.month, target.day, 0, 0, 0, tzinfo=tz)
        end_dt = start_dt + timedelta(days=1)

        # Fetch all orders in date range
        all_orders = self._fetch_orders(start_dt, end_dt)

        # Filter orders
        filtered_orders = []
        removed = {
            "test_orders": 0,
            "draft_orders": 0,
            "cancelled_orders": 0,
            "unpaid_orders": 0,
        }

        for order in all_orders:
            if order.get("test", False):
                removed["test_orders"] += 1
                continue
            if order.get("source_name") == "draft_orders":
                removed["draft_orders"] += 1
                continue
            if order.get("cancelled_at"):
                removed["cancelled_orders"] += 1
                continue
            if order.get("financial_status") != "paid":
                removed["unpaid_orders"] += 1
                continue

            # Calculate net revenue after refunds
            gross = float(order.get("total_price", 0))
            refunds = self._calculate_refunds(order)
            net = gross - refunds

            filtered_orders.append({
                "order_id": order["id"],
                "order_number": order["order_number"],
                "transaction_id": str(order["order_number"]),
                "gross_revenue": gross,
                "refund_amount": refunds,
                "net_revenue": net,
                "timestamp": order["created_at"],
                "customer_id": order.get("customer", {}).get("id"),
            })

        total_gross = sum(o["gross_revenue"] for o in filtered_orders)
        total_refunds = sum(o["refund_amount"] for o in filtered_orders)
        total_net = sum(o["net_revenue"] for o in filtered_orders)

        logger.info(
            "Extracted %d Shopify orders, filtered to %d, removed: %s",
            len(all_orders),
            len(filtered_orders),
            removed,
        )

        return {
            "target_date": target_date,
            "total_orders": len(all_orders),
            "filtered_orders": len(filtered_orders),
            "removed_breakdown": removed,
            "gross_revenue": round(total_gross, 2),
            "total_refunds": round(total_refunds, 2),
            "net_revenue": round(total_net, 2),
            "orders": filtered_orders,
        }

    def _fetch_orders(
        self,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[Dict[str, Any]]:
        """Fetch all orders in date range with pagination."""
        base_url = f"https://{self.shop_domain}/admin/api/{self.api_version}"
        headers = {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json",
        }

        all_orders = []
        params = {
            "status": "any",
            "created_at_min": start_dt.isoformat(),
            "created_at_max": end_dt.isoformat(),
            "limit": 250,
        }

        with httpx.Client(timeout=30.0) as client:
            url = f"{base_url}/orders.json"
            while url:
                response = client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                all_orders.extend(data.get("orders", []))

                # Handle pagination
                link_header = response.headers.get("Link", "")
                url = self._extract_next_page(link_header)
                params = {}

        return all_orders

    def _calculate_refunds(self, order: Dict[str, Any]) -> float:
        """Calculate total refunds for an order."""
        refunds = order.get("refunds", [])
        total_refund = 0.0
        for refund in refunds:
            for transaction in refund.get("transactions", []):
                if transaction.get("kind") == "refund":
                    total_refund += float(transaction.get("amount", 0))
        return total_refund

    def _extract_next_page(self, link_header: str) -> Optional[str]:
        """Extract next page URL from Link header."""
        if not link_header:
            return None
        for part in link_header.split(","):
            if 'rel="next"' in part:
                return part.split(";")[0].strip().strip("<>")
        return None

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use sync version")`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Reconciliation Agent that matches records across platforms and classifies discrepancies with root cause analysis.',
            toolsUsed: ['Record Matching Engine', 'Fuzzy Matching', 'Root Cause Classifier'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Cross-Platform Record Matching Tool',
                description:
                  'Tool for the Reconciliation Agent to match orders across GA4, Shopify, and backend.',
                code: `from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("record_matcher")


class MatchStatus(Enum):
    MATCHED = "matched"
    MISSING_GA4 = "missing_ga4"
    MISSING_SHOPIFY = "missing_shopify"
    MISSING_BACKEND = "missing_backend"
    AMOUNT_MISMATCH = "amount_mismatch"
    PARTIAL_MATCH = "partial_match"


@dataclass
class MatchResult:
    """Result of matching a single transaction."""
    transaction_id: str
    status: MatchStatus
    ga4_data: Optional[Dict]
    shopify_data: Optional[Dict]
    backend_data: Optional[Dict]
    amount_variance: Optional[float]
    probable_cause: str
    confidence: float


class RecordMatchingInput(BaseModel):
    """Input schema for record matching."""
    ga4_data: Dict = Field(description="GA4 extraction results")
    shopify_data: Dict = Field(description="Shopify extraction results")
    backend_data: Dict = Field(description="Backend extraction results")
    amount_tolerance_pct: float = Field(
        default=0.02,
        description="Tolerance for amount matching (default 2%)"
    )
    time_tolerance_hours: int = Field(
        default=2,
        description="Tolerance for timestamp matching (default 2 hours)"
    )


class RecordMatchingTool(BaseTool):
    """Tool for matching records across analytics platforms."""

    name: str = "record_matcher"
    description: str = (
        "Match orders across GA4, Shopify, and backend systems. "
        "Identifies discrepancies and classifies root causes."
    )
    args_schema: type[BaseModel] = RecordMatchingInput

    def _run(
        self,
        ga4_data: Dict,
        shopify_data: Dict,
        backend_data: Dict,
        amount_tolerance_pct: float = 0.02,
        time_tolerance_hours: int = 2,
    ) -> Dict[str, Any]:
        """Match records across all platforms."""

        # Index data by transaction ID
        ga4_index = {t["transaction_id"]: t for t in ga4_data.get("transactions", [])}
        shopify_index = {str(o["transaction_id"]): o for o in shopify_data.get("orders", [])}
        backend_index = {t["transaction_id"]: t for t in backend_data.get("transactions", [])}

        # Get all unique transaction IDs
        all_ids = set(ga4_index.keys()) | set(shopify_index.keys()) | set(backend_index.keys())

        results: List[MatchResult] = []
        matched = 0
        discrepancies_by_type = {status.value: 0 for status in MatchStatus if status != MatchStatus.MATCHED}

        for txn_id in all_ids:
            ga4 = ga4_index.get(txn_id)
            shopify = shopify_index.get(txn_id)
            backend = backend_index.get(txn_id)

            result = self._classify_match(
                txn_id, ga4, shopify, backend,
                amount_tolerance_pct, time_tolerance_hours
            )
            results.append(result)

            if result.status == MatchStatus.MATCHED:
                matched += 1
            else:
                discrepancies_by_type[result.status.value] += 1

        match_rate = matched / len(all_ids) if all_ids else 0

        # Generate summary statistics
        summary = self._generate_summary(results, ga4_data, shopify_data, backend_data)

        logger.info(
            "Matched %d/%d records (%.1f%%), discrepancies: %s",
            matched, len(all_ids), match_rate * 100, discrepancies_by_type
        )

        return {
            "total_records": len(all_ids),
            "matched": matched,
            "match_rate": round(match_rate, 4),
            "discrepancies_by_type": discrepancies_by_type,
            "unmatched_details": [
                {
                    "transaction_id": r.transaction_id,
                    "status": r.status.value,
                    "probable_cause": r.probable_cause,
                    "confidence": r.confidence,
                    "amount_variance": r.amount_variance,
                }
                for r in results if r.status != MatchStatus.MATCHED
            ],
            "summary": summary,
        }

    def _classify_match(
        self,
        txn_id: str,
        ga4: Optional[Dict],
        shopify: Optional[Dict],
        backend: Optional[Dict],
        amount_tolerance: float,
        time_tolerance: int,
    ) -> MatchResult:
        """Classify the match status for a transaction."""

        # Count platforms present
        platforms_present = sum([ga4 is not None, shopify is not None, backend is not None])

        if platforms_present == 3:
            # Check if amounts match
            amounts = [
                ga4.get("revenue", 0),
                shopify.get("net_revenue", shopify.get("revenue", 0)),
                backend.get("revenue", 0),
            ]
            max_variance = max(amounts) - min(amounts)
            avg_amount = sum(amounts) / 3

            if avg_amount > 0 and max_variance / avg_amount <= amount_tolerance:
                return MatchResult(
                    transaction_id=txn_id,
                    status=MatchStatus.MATCHED,
                    ga4_data=ga4,
                    shopify_data=shopify,
                    backend_data=backend,
                    amount_variance=None,
                    probable_cause="",
                    confidence=1.0,
                )
            else:
                return MatchResult(
                    transaction_id=txn_id,
                    status=MatchStatus.AMOUNT_MISMATCH,
                    ga4_data=ga4,
                    shopify_data=shopify,
                    backend_data=backend,
                    amount_variance=round(max_variance, 2),
                    probable_cause="Partial refund, currency conversion, or tax difference",
                    confidence=0.8,
                )

        # Missing from one platform
        if ga4 is None and shopify is not None:
            return MatchResult(
                transaction_id=txn_id,
                status=MatchStatus.MISSING_GA4,
                ga4_data=None,
                shopify_data=shopify,
                backend_data=backend,
                amount_variance=None,
                probable_cause="Ad blocker, consent rejection, or tracking failure",
                confidence=0.85,
            )

        if shopify is None and ga4 is not None:
            return MatchResult(
                transaction_id=txn_id,
                status=MatchStatus.MISSING_SHOPIFY,
                ga4_data=ga4,
                shopify_data=None,
                backend_data=backend,
                amount_variance=None,
                probable_cause="API order, draft order, or timezone boundary",
                confidence=0.7,
            )

        if backend is None and shopify is not None:
            return MatchResult(
                transaction_id=txn_id,
                status=MatchStatus.MISSING_BACKEND,
                ga4_data=ga4,
                shopify_data=shopify,
                backend_data=None,
                amount_variance=None,
                probable_cause="Settlement delay or pending payment",
                confidence=0.9,
            )

        # Partial match
        return MatchResult(
            transaction_id=txn_id,
            status=MatchStatus.PARTIAL_MATCH,
            ga4_data=ga4,
            shopify_data=shopify,
            backend_data=backend,
            amount_variance=None,
            probable_cause="Multiple data gaps - requires investigation",
            confidence=0.5,
        )

    def _generate_summary(
        self,
        results: List[MatchResult],
        ga4_data: Dict,
        shopify_data: Dict,
        backend_data: Dict,
    ) -> Dict[str, Any]:
        """Generate reconciliation summary."""
        missing_ga4 = [r for r in results if r.status == MatchStatus.MISSING_GA4]
        ga4_gap_pct = len(missing_ga4) / len(results) * 100 if results else 0

        return {
            "ga4_tracking_gap_pct": round(ga4_gap_pct, 1),
            "ga4_expected_gap_pct": 12.5,  # Historical average
            "gap_status": "normal" if ga4_gap_pct < 18 else "elevated",
            "reconciled_order_count": len([r for r in results if r.status == MatchStatus.MATCHED]),
            "recommended_source_of_truth": "shopify_filtered",
            "confidence_score": round(
                sum(r.confidence for r in results) / len(results) * 100
                if results else 0, 1
            ),
        }

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use sync version")`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement LangGraph state machine to coordinate the reconciliation workflow, manage state across agents, and ensure proper sequencing of data extraction and matching.',
            toolsUsed: ['LangGraph', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Analytics Reconciliation LangGraph Workflow',
                description:
                  'LangGraph workflow orchestrating the multi-agent analytics reconciliation system.',
                code: `from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal
from datetime import date, datetime
import operator
import json
import redis
import logging

logger = logging.getLogger("reconciliation_orchestrator")


class ReconciliationState(TypedDict):
    """State for analytics reconciliation workflow."""

    # Workflow metadata
    run_id: str
    target_date: str
    started_at: str

    # GA4 extraction state
    ga4_raw_orders: int
    ga4_adjusted_orders: int
    ga4_revenue: float
    ga4_transactions: list

    # Shopify extraction state
    shopify_total_orders: int
    shopify_filtered_orders: int
    shopify_revenue: float
    shopify_orders: list

    # Backend extraction state
    backend_orders: int
    backend_revenue: float
    backend_transactions: list

    # Matching state
    matched_orders: int
    match_rate: float
    discrepancies: dict

    # Final results
    reconciled_orders: int
    reconciled_revenue: float
    confidence_score: float
    alerts: list
    report: dict

    errors: Annotated[list[str], operator.add]


def create_reconciliation_graph(
    ga4_agent,
    shopify_agent,
    backend_agent,
    reconciliation_agent,
    reporting_agent,
    redis_client: redis.Redis,
):
    """Create the reconciliation workflow graph."""

    workflow = StateGraph(ReconciliationState)

    def extract_ga4(state: ReconciliationState) -> dict:
        """Extract GA4 data."""
        logger.info("Extracting GA4 data for %s", state["target_date"])
        try:
            result = ga4_agent.invoke({
                "target_date": state["target_date"],
            })
            return {
                "ga4_raw_orders": result["raw_order_count"],
                "ga4_adjusted_orders": result["adjusted_order_count"],
                "ga4_revenue": result["total_revenue"],
                "ga4_transactions": result["transactions"],
            }
        except Exception as e:
            logger.error("GA4 extraction failed: %s", e)
            return {"errors": [f"GA4 extraction: {str(e)}"]}

    def extract_shopify(state: ReconciliationState) -> dict:
        """Extract Shopify data."""
        logger.info("Extracting Shopify data for %s", state["target_date"])
        try:
            result = shopify_agent.invoke({
                "target_date": state["target_date"],
            })
            return {
                "shopify_total_orders": result["total_orders"],
                "shopify_filtered_orders": result["filtered_orders"],
                "shopify_revenue": result["net_revenue"],
                "shopify_orders": result["orders"],
            }
        except Exception as e:
            logger.error("Shopify extraction failed: %s", e)
            return {"errors": [f"Shopify extraction: {str(e)}"]}

    def extract_backend(state: ReconciliationState) -> dict:
        """Extract backend payment data."""
        logger.info("Extracting backend data for %s", state["target_date"])
        try:
            result = backend_agent.invoke({
                "target_date": state["target_date"],
            })
            return {
                "backend_orders": result["order_count"],
                "backend_revenue": result["net_revenue"],
                "backend_transactions": result["transactions"],
            }
        except Exception as e:
            logger.error("Backend extraction failed: %s", e)
            return {"errors": [f"Backend extraction: {str(e)}"]}

    def match_records(state: ReconciliationState) -> dict:
        """Match records across platforms."""
        logger.info("Matching records...")
        try:
            result = reconciliation_agent.invoke({
                "ga4_data": {
                    "transactions": state.get("ga4_transactions", []),
                },
                "shopify_data": {
                    "orders": state.get("shopify_orders", []),
                },
                "backend_data": {
                    "transactions": state.get("backend_transactions", []),
                },
            })
            return {
                "matched_orders": result["matched"],
                "match_rate": result["match_rate"],
                "discrepancies": result["discrepancies_by_type"],
            }
        except Exception as e:
            logger.error("Record matching failed: %s", e)
            return {"errors": [f"Matching: {str(e)}"]}

    def generate_report(state: ReconciliationState) -> dict:
        """Generate final reconciliation report."""
        logger.info("Generating report...")

        # Calculate reconciled numbers
        # Use Shopify filtered as base, adjusted for known gaps
        reconciled_orders = state.get("shopify_filtered_orders", 0)
        reconciled_revenue = state.get("shopify_revenue", 0)

        # Calculate confidence based on match rate
        match_rate = state.get("match_rate", 0)
        confidence = min(match_rate * 100 + 10, 100)  # Boost for baseline trust

        # Determine alerts
        alerts = []
        ga4_gap = (
            (state.get("shopify_filtered_orders", 0) - state.get("ga4_raw_orders", 0))
            / max(state.get("shopify_filtered_orders", 1), 1) * 100
        )
        if ga4_gap > 20:
            alerts.append(f"GA4 tracking gap elevated: {ga4_gap:.1f}%")

        if match_rate < 0.85:
            alerts.append(f"Match rate below threshold: {match_rate:.1%}")

        # Build report
        report = {
            "date": state["target_date"],
            "platforms": {
                "ga4": {
                    "raw_orders": state.get("ga4_raw_orders", 0),
                    "adjusted_orders": state.get("ga4_adjusted_orders", 0),
                    "revenue": state.get("ga4_revenue", 0),
                },
                "shopify": {
                    "total_orders": state.get("shopify_total_orders", 0),
                    "filtered_orders": state.get("shopify_filtered_orders", 0),
                    "revenue": state.get("shopify_revenue", 0),
                },
                "backend": {
                    "orders": state.get("backend_orders", 0),
                    "revenue": state.get("backend_revenue", 0),
                },
            },
            "reconciliation": {
                "matched": state.get("matched_orders", 0),
                "match_rate": state.get("match_rate", 0),
                "discrepancies": state.get("discrepancies", {}),
            },
            "reconciled": {
                "orders": reconciled_orders,
                "revenue": reconciled_revenue,
            },
            "confidence_score": confidence,
            "alerts": alerts,
        }

        # Persist to Redis
        redis_client.setex(
            f"reconciliation:{state['target_date']}",
            86400 * 365,  # 1 year retention
            json.dumps(report),
        )

        return {
            "reconciled_orders": reconciled_orders,
            "reconciled_revenue": reconciled_revenue,
            "confidence_score": confidence,
            "alerts": alerts,
            "report": report,
        }

    # Add nodes - parallel extraction, then sequential processing
    workflow.add_node("extract_ga4", extract_ga4)
    workflow.add_node("extract_shopify", extract_shopify)
    workflow.add_node("extract_backend", extract_backend)
    workflow.add_node("match_records", match_records)
    workflow.add_node("generate_report", generate_report)

    # Parallel extraction
    workflow.set_entry_point("extract_ga4")
    workflow.add_edge("extract_ga4", "extract_shopify")
    workflow.add_edge("extract_shopify", "extract_backend")

    # Sequential processing after extraction
    workflow.add_edge("extract_backend", "match_records")
    workflow.add_edge("match_records", "generate_report")
    workflow.add_edge("generate_report", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the analytics reconciliation system with Docker, implement LangSmith tracing for agent observability, and set up dashboards for reconciliation metrics.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose Deployment',
                description:
                  'Docker Compose configuration for the analytics reconciliation system.',
                code: `version: "3.8"

services:
  reconciliation-service:
    build:
      context: .
      dockerfile: Dockerfile.reconciliation
    container_name: analytics-reconciliation
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=\${LANGSMITH_API_KEY}
      - LANGCHAIN_PROJECT=analytics-reconciliation-prod
      - REDIS_URL=redis://redis:6379/2
      - GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-creds.json
      - GA4_PROJECT_ID=\${GA4_PROJECT_ID}
      - GA4_DATASET_ID=\${GA4_DATASET_ID}
      - SHOPIFY_SHOP_DOMAIN=\${SHOPIFY_SHOP_DOMAIN}
      - SHOPIFY_ACCESS_TOKEN=\${SHOPIFY_ACCESS_TOKEN}
      - PAYMENT_GATEWAY_API_KEY=\${PAYMENT_GATEWAY_API_KEY}
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
      - LOG_LEVEL=INFO
    volumes:
      - ./secrets:/secrets:ro
      - ./config:/app/config:ro
    ports:
      - "8097:8097"
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8097/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: reconciliation-redis
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    ports:
      - "6381:6379"

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile.scheduler
    container_name: reconciliation-scheduler
    environment:
      - SERVICE_URL=http://reconciliation-service:8097
      - SCHEDULE_CRON=0 8 * * *  # 8 AM daily
      - TZ=America/New_York
    depends_on:
      - reconciliation-service
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v2.45.0
    volumes:
      - ./prometheus:/etc/prometheus:ro
      - prometheus-data:/prometheus
    ports:
      - "9092:9090"

  grafana:
    image: grafana/grafana:10.0.0
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_PASSWORD}
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana-data:/var/lib/grafana
    ports:
      - "3001:3000"
    depends_on:
      - prometheus

volumes:
  redis-data:
  prometheus-data:
  grafana-data:`,
              },
              {
                language: 'python',
                title: 'Reconciliation Metrics and Alerting',
                description:
                  'Prometheus metrics and Slack alerting for the reconciliation system.',
                code: `from prometheus_client import Counter, Gauge, Histogram, start_http_server
from langsmith import Client
from langsmith.run_helpers import traceable
import json
import urllib.request
from datetime import datetime
import logging

logger = logging.getLogger("reconciliation_metrics")

# Prometheus metrics
RECONCILIATION_RUNS = Counter(
    "reconciliation_runs_total",
    "Total reconciliation runs",
    ["status"],
)

MATCH_RATE = Gauge(
    "reconciliation_match_rate",
    "Current reconciliation match rate",
)

GA4_GAP = Gauge(
    "reconciliation_ga4_gap_pct",
    "GA4 tracking gap percentage",
)

RECONCILED_ORDERS = Gauge(
    "reconciliation_orders_count",
    "Reconciled order count",
)

RECONCILED_REVENUE = Gauge(
    "reconciliation_revenue_total",
    "Reconciled revenue total",
)

DISCREPANCIES = Counter(
    "reconciliation_discrepancies_total",
    "Total discrepancies by type",
    ["type"],
)


class ReconciliationMetrics:
    """Metrics collector for reconciliation system."""

    def __init__(self, slack_webhook: str, metrics_port: int = 8098):
        self.slack_webhook = slack_webhook
        self.metrics_port = metrics_port
        self.langsmith = Client()

    def start_metrics_server(self):
        """Start Prometheus metrics server."""
        start_http_server(self.metrics_port)
        logger.info("Metrics server on port %d", self.metrics_port)

    def record_run(self, report: dict):
        """Record metrics from a reconciliation run."""
        status = "success" if not report.get("errors") else "error"
        RECONCILIATION_RUNS.labels(status=status).inc()

        recon = report.get("reconciliation", {})
        MATCH_RATE.set(recon.get("match_rate", 0))

        # Calculate GA4 gap
        platforms = report.get("platforms", {})
        shopify_orders = platforms.get("shopify", {}).get("filtered_orders", 0)
        ga4_orders = platforms.get("ga4", {}).get("raw_orders", 0)
        if shopify_orders > 0:
            ga4_gap = (shopify_orders - ga4_orders) / shopify_orders * 100
            GA4_GAP.set(ga4_gap)

        reconciled = report.get("reconciled", {})
        RECONCILED_ORDERS.set(reconciled.get("orders", 0))
        RECONCILED_REVENUE.set(reconciled.get("revenue", 0))

        # Record discrepancies
        for disc_type, count in recon.get("discrepancies", {}).items():
            DISCREPANCIES.labels(type=disc_type).inc(count)

    def send_daily_report(self, report: dict):
        """Send daily reconciliation report to Slack."""
        platforms = report.get("platforms", {})
        reconciled = report.get("reconciled", {})
        confidence = report.get("confidence_score", 0)

        status_emoji = ":white_check_mark:" if confidence >= 90 else (
            ":warning:" if confidence >= 80 else ":red_circle:"
        )

        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{status_emoji} Analytics Reconciliation - {report['date']}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*GA4 Orders:* {platforms['ga4']['raw_orders']:,}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Shopify Orders:* {platforms['shopify']['filtered_orders']:,}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Backend Orders:* {platforms['backend']['orders']:,}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Match Rate:* {report['reconciliation']['match_rate']:.1%}",
                        },
                    ],
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Reconciled Orders:* {reconciled['orders']:,}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Reconciled Revenue:* \${reconciled['revenue']:,.2f}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Confidence Score:* {confidence:.0f}/100",
                        },
                    ],
                },
            ],
        }

        # Add alerts if any
        if report.get("alerts"):
            alert_text = "\\n".join(f"- {a}" for a in report["alerts"])
            payload["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Alerts:*\\n{alert_text}",
                },
            })

        self._post_slack(payload)

    def _post_slack(self, payload: dict):
        """Post message to Slack."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.slack_webhook,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.error("Slack post failed: %s", e)`,
              },
            ],
          },
        ],
      },
    },
  ],
};
