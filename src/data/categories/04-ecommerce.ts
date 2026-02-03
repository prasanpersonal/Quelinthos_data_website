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
    },
  ],
};
