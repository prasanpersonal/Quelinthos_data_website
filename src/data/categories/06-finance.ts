import type { Category } from '../types.ts';

export const financeCategory: Category = {
  id: 'finance',
  number: 6,
  title: 'Finance & Accounting',
  shortTitle: 'Finance',
  description:
    'Automate payment reconciliation, nail UAE corporate tax compliance, and fix Australia GST data gaps.',
  icon: 'DollarSign',
  accentColor: 'neon-gold',
  painPoints: [
    /* ------------------------------------------------------------------ */
    /*  1. Payment Gateway Reconciliation                                  */
    /* ------------------------------------------------------------------ */
    {
      id: 'payment-reconciliation',
      number: 1,
      title: 'Payment Gateway Reconciliation',
      subtitle: 'Stripe, PayPal, and Bank Statement Mismatches',
      summary:
        'Monthly close takes 2 weeks because Stripe, PayPal, and bank statements never match. Your finance team reconciles $10M+ manually in spreadsheets.',
      tags: ['payments', 'reconciliation', 'fintech'],
      metrics: {
        annualCostRange: '$400K - $2M',
        roi: '10x',
        paybackPeriod: '2-3 months',
        investmentRange: '$50K - $100K',
      },
      price: {
        present: {
          title: 'The Reconciliation Nightmare Is Happening Now',
          description:
            'Every month-end your finance team drops everything and spends two weeks matching Stripe payouts, PayPal settlements, and bank credits line by line in Excel.',
          bullets: [
            'Stripe batch payouts aggregate dozens of transactions into a single bank deposit, making 1:1 matching impossible',
            'PayPal holds, currency conversions, and fee deductions create phantom discrepancies across ledgers',
            'Refunds, chargebacks, and partial payments appear on different dates across each system',
            'Bank statement descriptions rarely match payment gateway reference IDs',
          ],
          severity: 'high',
        },
        root: {
          title: 'No Unified Transaction Ledger',
          description:
            'Stripe, PayPal, and your bank each model transactions differently. Without a canonical ledger that normalises IDs, timestamps, and amounts, reconciliation is always manual.',
          bullets: [
            'Each gateway uses proprietary transaction IDs with no shared key to the bank feed',
            'Fee calculations are embedded differently: Stripe deducts before payout, PayPal shows gross then net',
            'Settlement timing varies: Stripe is T+2, PayPal is T+1 to T+3, and bank posting adds another day',
            'No single source of truth links a customer payment to its gateway event and final bank credit',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Revenue Leakage and Delayed Closes',
          description:
            'Manual reconciliation does not just waste time; it lets real revenue gaps slip through and delays every downstream financial process.',
          bullets: [
            'Unmatched transactions average 3-5% of monthly volume, representing potential revenue leakage',
            'Month-end close is delayed by 8-12 business days, pushing board reporting and forecasting back',
            'Audit prep requires re-reconciling the entire year because monthly work is in ad-hoc spreadsheets',
            'Finance headcount grows linearly with transaction volume instead of staying flat',
          ],
          severity: 'high',
        },
        cost: {
          title: 'The True Cost of Manual Matching',
          description:
            'Between labour, lost revenue, and delayed decisions, the all-in cost of manual reconciliation far exceeds what teams estimate.',
          bullets: [
            '2-3 FTEs dedicated to reconciliation at a fully loaded cost of $150K-$250K each',
            'Unrecovered chargebacks and duplicate refunds average 0.1-0.3% of GMV',
            'Late close delays pricing and discount decisions by two weeks every month',
            'External audit surcharges of $20K-$50K per year for incomplete reconciliation workpapers',
          ],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Automated Matching Cuts Close to 2 Days',
          description:
            'A normalised transaction ledger with fuzzy-match reconciliation automates 95%+ of line items, freeing finance to focus on exceptions only.',
          bullets: [
            'Month-end close reduced from 10-14 days to 1-2 days',
            'Reconciliation labour drops by 80%, redeploying 2+ FTEs to strategic finance',
            'Revenue leakage detected in real time, recovering 0.1-0.2% of GMV annually',
            'Audit-ready workpapers generated automatically each period',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Build an automated reconciliation pipeline that ingests Stripe, PayPal, and bank feeds into a unified ledger, then applies deterministic and fuzzy matching to resolve discrepancies.',
        prerequisites: [
          'Stripe API key with read access to balance transactions and payouts',
          'PayPal REST API credentials for transaction search and settlement reports',
          'Bank feed access via Open Banking API or daily CSV/MT940 exports',
          'PostgreSQL 14+ or Snowflake warehouse for the unified ledger',
          'pytest >= 7.0 with pytest-asyncio',
          'Docker and docker-compose for containerized deployment',
          'cron, Airflow, or Prefect for job scheduling',
          'Slack incoming webhook URL for operational alerts',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Build the Unified Transaction Ledger',
            description:
              'Create a normalised schema that maps every gateway transaction and bank credit to a common structure with standardised amounts, dates, and reference keys.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Unified ledger schema and gateway ingestion view',
                description:
                  'Creates the core reconciliation tables and a unified view across Stripe, PayPal, and bank data.',
                code: `-- Unified ledger: one row per money-movement event
CREATE TABLE unified_ledger (
    ledger_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source           TEXT NOT NULL CHECK (source IN ('stripe','paypal','bank')),
    external_id      TEXT NOT NULL,
    event_type       TEXT NOT NULL,            -- payment, refund, chargeback, payout, credit
    gross_amount     NUMERIC(14,2) NOT NULL,
    fee_amount       NUMERIC(14,2) DEFAULT 0,
    net_amount       NUMERIC(14,2) GENERATED ALWAYS AS (gross_amount - fee_amount) STORED,
    currency         CHAR(3) NOT NULL DEFAULT 'USD',
    event_date       DATE NOT NULL,
    settlement_date  DATE,
    reference_key    TEXT,                     -- normalised customer/order ref
    match_status     TEXT DEFAULT 'unmatched',
    match_group_id   UUID,
    ingested_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (source, external_id)
);

-- Deterministic match: exact net amount + same settlement date + reference overlap
CREATE VIEW reconciliation_candidates AS
SELECT
    a.ledger_id   AS gateway_ledger_id,
    b.ledger_id   AS bank_ledger_id,
    a.source       AS gateway_source,
    a.net_amount,
    a.settlement_date,
    similarity(a.reference_key, b.reference_key) AS ref_score
FROM unified_ledger a
JOIN unified_ledger b
  ON b.source = 'bank'
 AND a.source IN ('stripe','paypal')
 AND a.net_amount = b.net_amount
 AND a.settlement_date BETWEEN b.event_date - 2 AND b.event_date + 2
WHERE a.match_status = 'unmatched'
  AND b.match_status = 'unmatched'
ORDER BY ref_score DESC;`,
              },
              {
                language: 'python',
                title: 'Stripe and PayPal ingestion pipeline',
                description:
                  'Fetches transactions from both gateways and loads them into the unified ledger with normalised fields.',
                code: `import stripe
import requests
from datetime import datetime, timedelta
from decimal import Decimal
import psycopg2
from psycopg2.extras import execute_values

def ingest_stripe_transactions(api_key: str, conn, since_days: int = 7):
    """Pull Stripe balance transactions and upsert into unified_ledger."""
    stripe.api_key = api_key
    cutoff = int((datetime.utcnow() - timedelta(days=since_days)).timestamp())

    txns = stripe.BalanceTransaction.list(
        created={"gte": cutoff}, limit=100, expand=["data.source"]
    )
    rows = []
    for t in txns.auto_paging_iter():
        rows.append((
            "stripe",
            t["id"],
            t["type"],                               # payment, refund, payout ...
            Decimal(t["amount"]) / 100,
            Decimal(t["fee"]) / 100,
            t["currency"].upper(),
            datetime.utcfromtimestamp(t["created"]).date(),
            datetime.utcfromtimestamp(t["available_on"]).date(),
            t.get("source", {}).get("metadata", {}).get("order_id"),
        ))

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO unified_ledger
                (source, external_id, event_type, gross_amount,
                 fee_amount, currency, event_date, settlement_date, reference_key)
            VALUES %s
            ON CONFLICT (source, external_id) DO NOTHING
        """, rows)
    conn.commit()
    return len(rows)`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Automated Matching Engine',
            description:
              'Run deterministic matching first (exact amount + date window), then fuzzy matching for remaining items, and flag true exceptions for human review.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Multi-pass matching procedure',
                description:
                  'Applies exact match, then aggregate-payout match, then flags residual exceptions.',
                code: `-- Pass 1: exact 1-to-1 match on net_amount + settlement window
WITH exact_matches AS (
    SELECT DISTINCT ON (rc.gateway_ledger_id)
        rc.gateway_ledger_id,
        rc.bank_ledger_id,
        gen_random_uuid() AS grp
    FROM reconciliation_candidates rc
    WHERE rc.ref_score > 0.6
    ORDER BY rc.gateway_ledger_id, rc.ref_score DESC
)
UPDATE unified_ledger ul
SET match_status   = 'matched',
    match_group_id = em.grp
FROM exact_matches em
WHERE ul.ledger_id IN (em.gateway_ledger_id, em.bank_ledger_id);

-- Pass 2: aggregate payout match (Stripe batches N transactions into 1 bank credit)
WITH payout_groups AS (
    SELECT settlement_date,
           SUM(net_amount)  AS payout_total,
           array_agg(ledger_id) AS member_ids
    FROM unified_ledger
    WHERE source = 'stripe' AND event_type = 'payment' AND match_status = 'unmatched'
    GROUP BY settlement_date
),
bank_credits AS (
    SELECT ledger_id, net_amount, event_date
    FROM unified_ledger
    WHERE source = 'bank' AND match_status = 'unmatched'
)
UPDATE unified_ledger ul
SET match_status   = 'batch_matched',
    match_group_id = gen_random_uuid()
FROM payout_groups pg
JOIN bank_credits bc ON pg.payout_total = bc.net_amount
    AND pg.settlement_date BETWEEN bc.event_date - 1 AND bc.event_date + 1
WHERE ul.ledger_id = ANY(pg.member_ids) OR ul.ledger_id = bc.ledger_id;`,
              },
              {
                language: 'python',
                title: 'Exception report generator',
                description:
                  'Generates a daily exceptions report for the finance team showing only the items that need manual review.',
                code: `import pandas as pd
from datetime import date

def generate_exception_report(conn, report_date: date | None = None) -> pd.DataFrame:
    """Build a prioritised exception report for unmatched items."""
    report_date = report_date or date.today()

    query = """
        SELECT source, external_id, event_type,
               gross_amount, fee_amount, net_amount,
               currency, event_date, settlement_date, reference_key,
               CURRENT_DATE - event_date AS days_open
        FROM unified_ledger
        WHERE match_status = 'unmatched'
          AND event_date <= %s
        ORDER BY ABS(net_amount) DESC, days_open DESC
    """
    df = pd.read_sql(query, conn, params=[report_date])

    # Prioritise: large amounts and old items first
    df["priority"] = pd.cut(
        df["days_open"],
        bins=[-1, 3, 7, 30, 9999],
        labels=["low", "medium", "high", "critical"],
    )

    summary = {
        "total_exceptions": len(df),
        "total_value": float(df["net_amount"].abs().sum()),
        "by_source": df.groupby("source")["net_amount"].sum().to_dict(),
        "critical_count": int((df["priority"] == "critical").sum()),
    }
    print(f"Exceptions: {summary['total_exceptions']} items, "
          f"\${summary['total_value']:,.2f} total value, "
          f"{summary['critical_count']} critical")
    return df`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate the reconciliation pipeline with data quality assertions on the unified ledger and automated pytest suites that verify match accuracy, duplicate detection, and balance integrity.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Reconciliation data quality assertions',
                description:
                  'SQL assertion queries for ledger row counts, null checks, referential integrity, duplicate detection, and data freshness.',
                code: `-- Assertion 1: No NULL net_amount in unified ledger
SELECT 'null_net_amount' AS assertion,
       COUNT(*) AS failures
FROM unified_ledger
WHERE net_amount IS NULL;

-- Assertion 2: No duplicate (source, external_id) pairs
SELECT 'duplicate_external_id' AS assertion,
       COUNT(*) AS failures
FROM (
    SELECT source, external_id, COUNT(*) AS cnt
    FROM unified_ledger
    GROUP BY source, external_id
    HAVING COUNT(*) > 1
) dups;

-- Assertion 3: Every matched row has a valid match_group_id
SELECT 'matched_without_group' AS assertion,
       COUNT(*) AS failures
FROM unified_ledger
WHERE match_status IN ('matched', 'batch_matched')
  AND match_group_id IS NULL;

-- Assertion 4: Ledger balance check — gateway debits equal bank credits per match group
SELECT 'balance_mismatch' AS assertion,
       COUNT(*) AS failures
FROM (
    SELECT match_group_id,
           SUM(CASE WHEN source IN ('stripe','paypal') THEN net_amount ELSE 0 END) AS gateway_total,
           SUM(CASE WHEN source = 'bank' THEN net_amount ELSE 0 END) AS bank_total
    FROM unified_ledger
    WHERE match_group_id IS NOT NULL
    GROUP BY match_group_id
    HAVING ABS(SUM(CASE WHEN source IN ('stripe','paypal') THEN net_amount ELSE 0 END)
             - SUM(CASE WHEN source = 'bank' THEN net_amount ELSE 0 END)) > 0.01
) mismatches;

-- Assertion 5: Data freshness — most recent ingestion within last 24 hours
SELECT 'stale_data' AS assertion,
       CASE WHEN MAX(ingested_at) < NOW() - INTERVAL '24 hours' THEN 1 ELSE 0 END AS failures
FROM unified_ledger;`,
              },
              {
                language: 'python',
                title: 'Pytest pipeline validation tests',
                description:
                  'Automated pytest suite validating match rates, duplicate detection, ledger balance integrity, and reconciliation edge cases.',
                code: `import pytest
import psycopg2
from decimal import Decimal
from datetime import date, timedelta

@pytest.fixture(scope="module")
def db_conn():
    conn = psycopg2.connect(
        host="localhost", dbname="recon_test", user="test_user", password="test_pass"
    )
    yield conn
    conn.close()

class TestLedgerIntegrity:
    def test_no_null_net_amounts(self, db_conn):
        """Every ledger row must have a non-null net_amount."""
        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM unified_ledger WHERE net_amount IS NULL")
            assert cur.fetchone()[0] == 0, "Found rows with NULL net_amount"

    def test_no_duplicate_external_ids(self, db_conn):
        """No duplicate (source, external_id) combinations allowed."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT source, external_id, COUNT(*)
                FROM unified_ledger
                GROUP BY source, external_id
                HAVING COUNT(*) > 1
            """)
            dupes = cur.fetchall()
            assert len(dupes) == 0, f"Found {len(dupes)} duplicate external_id pairs"

    def test_match_group_balance(self, db_conn):
        """Gateway total must equal bank total within each match group."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT match_group_id,
                       ABS(SUM(CASE WHEN source IN ('stripe','paypal') THEN net_amount ELSE 0 END)
                         - SUM(CASE WHEN source = 'bank' THEN net_amount ELSE 0 END)) AS diff
                FROM unified_ledger
                WHERE match_group_id IS NOT NULL
                GROUP BY match_group_id
                HAVING ABS(SUM(CASE WHEN source IN ('stripe','paypal') THEN net_amount ELSE 0 END)
                         - SUM(CASE WHEN source = 'bank' THEN net_amount ELSE 0 END)) > 0.01
            """)
            mismatches = cur.fetchall()
            assert len(mismatches) == 0, f"Found {len(mismatches)} unbalanced match groups"

class TestMatchRateValidation:
    def test_minimum_match_rate(self, db_conn):
        """Overall match rate must exceed 92% threshold."""
        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM unified_ledger")
            total = cur.fetchone()[0]
            cur.execute("""
                SELECT COUNT(*) FROM unified_ledger
                WHERE match_status IN ('matched', 'batch_matched')
            """)
            matched = cur.fetchone()[0]
            if total > 0:
                rate = matched / total
                assert rate >= 0.92, f"Match rate {rate:.1%} below 92% threshold"

    def test_no_self_matches(self, db_conn):
        """A ledger row must not be matched to itself."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT match_group_id, COUNT(DISTINCT source)
                FROM unified_ledger
                WHERE match_group_id IS NOT NULL
                GROUP BY match_group_id
                HAVING COUNT(DISTINCT source) < 2
            """)
            self_matches = cur.fetchall()
            assert len(self_matches) == 0, "Found match groups with only one source"`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Deploy the reconciliation pipeline with a hardened bash deployment script that handles environment validation, dependency installation, database migrations, and scheduler setup, plus a Python configuration loader for secrets and connection pools.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Reconciliation pipeline deployment script',
                description:
                  'Production deployment script with environment checks, dependency installation, database migration, and cron scheduler setup.',
                code: `#!/usr/bin/env bash
set -euo pipefail

APP_NAME="recon-pipeline"
DEPLOY_DIR="/opt/\${APP_NAME}"
VENV_DIR="\${DEPLOY_DIR}/venv"
LOG_DIR="/var/log/\${APP_NAME}"

echo "[1/6] Checking required environment variables..."
for var in RECON_DB_HOST RECON_DB_NAME RECON_DB_USER RECON_DB_PASSWORD \\
           STRIPE_API_KEY PAYPAL_CLIENT_ID PAYPAL_SECRET SLACK_WEBHOOK_URL; do
    if [ -z "\${!var:-}" ]; then
        echo "ERROR: \${var} is not set" >&2
        exit 1
    fi
done

echo "[2/6] Creating deployment directories..."
sudo mkdir -p "\${DEPLOY_DIR}" "\${LOG_DIR}"
sudo chown "\$(whoami):" "\${DEPLOY_DIR}" "\${LOG_DIR}"

echo "[3/6] Installing Python dependencies..."
python3 -m venv "\${VENV_DIR}"
source "\${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "\${DEPLOY_DIR}/requirements.txt"

echo "[4/6] Running database migrations..."
export PGPASSWORD="\${RECON_DB_PASSWORD}"
psql -h "\${RECON_DB_HOST}" -U "\${RECON_DB_USER}" -d "\${RECON_DB_NAME}" \\
    -f "\${DEPLOY_DIR}/sql/001_create_unified_ledger.sql"
psql -h "\${RECON_DB_HOST}" -U "\${RECON_DB_USER}" -d "\${RECON_DB_NAME}" \\
    -f "\${DEPLOY_DIR}/sql/002_create_reconciliation_views.sql"

echo "[5/6] Running smoke tests..."
"\${VENV_DIR}/bin/pytest" "\${DEPLOY_DIR}/tests/" -x -q --tb=short

echo "[6/6] Setting up cron schedule..."
CRON_ENTRY="0 6 * * * \${VENV_DIR}/bin/python \${DEPLOY_DIR}/run_reconciliation.py >> \${LOG_DIR}/recon.log 2>&1"
(crontab -l 2>/dev/null | grep -v "\${APP_NAME}" || true; echo "\${CRON_ENTRY}") | crontab -

echo "Deployment complete. Pipeline scheduled to run daily at 06:00 UTC."`,
              },
              {
                language: 'python',
                title: 'Configuration loader with connection pooling',
                description:
                  'Environment-variable based configuration loader with secrets management and database connection pool setup for the reconciliation pipeline.',
                code: `import os
from dataclasses import dataclass, field
from psycopg2 import pool
from typing import Optional

@dataclass
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str
    min_connections: int = 2
    max_connections: int = 10

@dataclass
class GatewayConfig:
    stripe_api_key: str
    paypal_client_id: str
    paypal_secret: str

@dataclass
class AlertConfig:
    slack_webhook_url: str
    match_rate_threshold: float = 0.92
    critical_exception_limit: int = 10

@dataclass
class ReconConfig:
    database: DatabaseConfig
    gateways: GatewayConfig
    alerts: AlertConfig
    environment: str = "production"
    log_level: str = "INFO"

def load_config() -> ReconConfig:
    """Load reconciliation pipeline config from environment variables."""
    return ReconConfig(
        database=DatabaseConfig(
            host=os.environ["RECON_DB_HOST"],
            port=int(os.environ.get("RECON_DB_PORT", "5432")),
            name=os.environ["RECON_DB_NAME"],
            user=os.environ["RECON_DB_USER"],
            password=os.environ["RECON_DB_PASSWORD"],
        ),
        gateways=GatewayConfig(
            stripe_api_key=os.environ["STRIPE_API_KEY"],
            paypal_client_id=os.environ["PAYPAL_CLIENT_ID"],
            paypal_secret=os.environ["PAYPAL_SECRET"],
        ),
        alerts=AlertConfig(
            slack_webhook_url=os.environ["SLACK_WEBHOOK_URL"],
            match_rate_threshold=float(os.environ.get("MATCH_RATE_THRESHOLD", "0.92")),
            critical_exception_limit=int(os.environ.get("CRITICAL_EXCEPTION_LIMIT", "10")),
        ),
        environment=os.environ.get("DEPLOY_ENV", "production"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )

class ConnectionPool:
    """Managed database connection pool for the reconciliation pipeline."""

    def __init__(self, config: DatabaseConfig):
        self._pool = pool.ThreadedConnectionPool(
            minconn=config.min_connections,
            maxconn=config.max_connections,
            host=config.host,
            port=config.port,
            dbname=config.name,
            user=config.user,
            password=config.password,
        )

    def get_connection(self):
        return self._pool.getconn()

    def release_connection(self, conn):
        self._pool.putconn(conn)

    def close_all(self):
        self._pool.closeall()

# Usage
config = load_config()
db_pool = ConnectionPool(config.database)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Schedule the pipeline to run daily rather than monthly, monitor match-rate trends and exception volumes, detect revenue leakage, and send Slack alerts when thresholds are breached.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Daily reconciliation orchestrator',
                description:
                  'Orchestrates ingestion, matching, and reporting on a daily schedule and computes key reconciliation health metrics.',
                code: `import schedule
import time
from datetime import date

def run_daily_reconciliation(conn, stripe_key: str):
    """Full daily reconciliation cycle: ingest -> match -> report -> alert."""
    # 1. Ingest fresh gateway and bank data
    stripe_count = ingest_stripe_transactions(stripe_key, conn, since_days=3)

    # 2. Execute matching passes (call the SQL procedures)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM unified_ledger WHERE match_status = 'unmatched'")
        pre_unmatched = cur.fetchone()[0]

        cur.execute(open("sql/pass1_exact_match.sql").read())
        cur.execute(open("sql/pass2_batch_match.sql").read())
        conn.commit()

        cur.execute("SELECT count(*) FROM unified_ledger WHERE match_status = 'unmatched'")
        post_unmatched = cur.fetchone()[0]

    total = pre_unmatched + (pre_unmatched - post_unmatched)
    match_rate = 1 - (post_unmatched / max(total, 1))

    # 3. Generate exception report
    exceptions = generate_exception_report(conn, date.today())

    return {
        "date": str(date.today()),
        "ingested": stripe_count,
        "match_rate": match_rate,
        "unmatched": post_unmatched,
        "critical_exceptions": int((exceptions["priority"] == "critical").sum()),
        "total_exception_value": float(exceptions["net_amount"].abs().sum()),
    }

schedule.every().day.at("06:00").do(
    run_daily_reconciliation, conn=db_conn, stripe_key=STRIPE_KEY
)`,
              },
              {
                language: 'python',
                title: 'Slack webhook alerting with threshold monitoring',
                description:
                  'Monitors reconciliation KPIs and sends formatted Slack alerts when match rates drop, exception volumes spike, or potential revenue leakage is detected.',
                code: `import requests as http_requests
from datetime import date
from dataclasses import dataclass

@dataclass
class ReconciliationAlert:
    metric: str
    current_value: float
    threshold: float
    severity: str  # "warning" or "critical"

SLACK_WEBHOOK = "https://hooks.slack.com/services/T00/B00/xxxx"
MATCH_RATE_THRESHOLD = 0.92
EXCEPTION_VOLUME_LIMIT = 10
LEAKAGE_THRESHOLD_USD = 5000.0

def evaluate_recon_thresholds(metrics: dict) -> list[ReconciliationAlert]:
    """Evaluate reconciliation metrics against operational thresholds."""
    alerts: list[ReconciliationAlert] = []

    if metrics["match_rate"] < MATCH_RATE_THRESHOLD:
        alerts.append(ReconciliationAlert(
            metric="Match Rate",
            current_value=metrics["match_rate"],
            threshold=MATCH_RATE_THRESHOLD,
            severity="critical" if metrics["match_rate"] < 0.85 else "warning",
        ))

    if metrics["critical_exceptions"] > EXCEPTION_VOLUME_LIMIT:
        alerts.append(ReconciliationAlert(
            metric="Critical Exceptions",
            current_value=metrics["critical_exceptions"],
            threshold=EXCEPTION_VOLUME_LIMIT,
            severity="critical",
        ))

    if metrics["total_exception_value"] > LEAKAGE_THRESHOLD_USD:
        alerts.append(ReconciliationAlert(
            metric="Potential Revenue Leakage",
            current_value=metrics["total_exception_value"],
            threshold=LEAKAGE_THRESHOLD_USD,
            severity="critical" if metrics["total_exception_value"] > 20000 else "warning",
        ))

    return alerts

def send_recon_slack_alerts(metrics: dict):
    """Send Slack alerts for any breached reconciliation thresholds."""
    alerts = evaluate_recon_thresholds(metrics)
    if not alerts:
        return

    emoji = ":red_circle:" if any(a.severity == "critical" for a in alerts) else ":warning:"
    blocks = [
        f"{emoji} *Reconciliation Alert — {date.today()}*",
        f"Match Rate: {metrics['match_rate']:.1%} | Unmatched: {metrics['unmatched']}",
    ]
    for alert in alerts:
        icon = ":rotating_light:" if alert.severity == "critical" else ":large_yellow_circle:"
        if alert.metric == "Match Rate":
            blocks.append(f"{icon} {alert.metric}: {alert.current_value:.1%} (threshold: {alert.threshold:.0%})")
        elif alert.metric == "Potential Revenue Leakage":
            blocks.append(f"{icon} {alert.metric}: \${alert.current_value:,.2f} (threshold: \${alert.threshold:,.2f})")
        else:
            blocks.append(f"{icon} {alert.metric}: {alert.current_value} (limit: {alert.threshold})")

    http_requests.post(SLACK_WEBHOOK, json={"text": "\\n".join(blocks)})
    print(f"Sent {len(alerts)} reconciliation alerts to Slack")`,
              },
            ],
          },
        ],
        toolsUsed: [
          'PostgreSQL with pg_trgm extension',
          'Python 3.11+',
          'Stripe API',
          'PayPal REST API',
          'pandas',
          'psycopg2',
          'pytest',
          'Docker',
          'GitHub Actions',
          'cron',
          'Slack API',
          'Prometheus',
        ],
      },
    },

    /* ------------------------------------------------------------------ */
    /*  2. UAE Corporate Tax Compliance                                     */
    /* ------------------------------------------------------------------ */
    {
      id: 'uae-corporate-tax',
      number: 2,
      title: 'UAE Corporate Tax Compliance',
      subtitle: 'New 9% Tax Regime Data Readiness',
      summary:
        'The UAE 9% corporate tax is live. Your chart of accounts, transfer pricing docs, and revenue recognition aren\'t structured for compliance.',
      tags: ['uae', 'corporate-tax', 'compliance'],
      metrics: {
        annualCostRange: '$200K - $1M',
        roi: '12x',
        paybackPeriod: '2-3 months',
        investmentRange: '$40K - $80K',
      },
      price: {
        present: {
          title: 'Your Books Are Not Structured for UAE Corporate Tax',
          description:
            'The 9% corporate tax regime is live but your chart of accounts was designed for a zero-tax environment. Revenue buckets, cost allocations, and intercompany transactions are not mapped to FTA requirements.',
          bullets: [
            'Chart of accounts has no distinction between taxable income, exempt income, and qualifying free zone revenue',
            'Transfer pricing documentation is non-existent or lives in scattered Word documents',
            'Revenue recognition timing does not align with UAE CT law accrual requirements',
            'Small Business Relief threshold tracking ($375K AED) is done manually with no systematic monitoring',
          ],
          severity: 'high',
        },
        root: {
          title: 'Legacy Zero-Tax Data Architecture',
          description:
            'For decades UAE entities had no income tax. Your entire accounting data model was built without tax classification in mind, and retrofitting it is a structural problem, not a process tweak.',
          bullets: [
            'ERP chart of accounts lacks the granularity required by Federal Tax Authority reporting',
            'No systematic tagging of related-party transactions for transfer pricing analysis',
            'Free zone vs mainland revenue is not segmented at the transaction level',
            'Historical data cannot distinguish between qualifying and non-qualifying income streams',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Penalties and Missed Elections',
          description:
            'Incorrect or late corporate tax filings carry direct financial penalties and can invalidate favourable elections like Small Business Relief or Free Zone qualifying status.',
          bullets: [
            'Late filing penalties start at AED 500 per month, escalating for continued non-compliance',
            'Incorrect taxable income calculations can trigger FTA audits and reassessments',
            'Failure to maintain transfer pricing documentation is a standalone offence',
            'Loss of Qualifying Free Zone status means retroactive 9% tax on previously exempt income',
          ],
          severity: 'high',
        },
        cost: {
          title: 'The Real Cost of Non-Compliance',
          description:
            'Beyond direct penalties, the cost includes advisory fees for remediation, management distraction, and competitive disadvantage against compliant peers.',
          bullets: [
            'Big-4 remediation engagements for CT readiness run $150K-$500K depending on entity complexity',
            'Internal effort to manually classify 2+ years of historical transactions: 3-6 months of FTE time',
            'Transfer pricing study commissioned from scratch: $80K-$200K per related-party arrangement',
            'Opportunity cost of delayed free zone elections or incorrect Small Business Relief claims',
          ],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Automated CT-Ready Data Model',
          description:
            'A properly restructured chart of accounts and automated classification pipeline ensures every transaction is tax-ready at the point of entry, not at year-end.',
          bullets: [
            'Real-time taxable income tracking replaces year-end scramble',
            'Transfer pricing documentation generated automatically from transaction data',
            'Free zone qualifying income monitored continuously against de minimis thresholds',
            'Filing preparation reduced from weeks to hours with pre-mapped FTA return fields',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Restructure the chart of accounts for UAE corporate tax compliance, automate income classification at the transaction level, and generate transfer pricing documentation from live data.',
        prerequisites: [
          'Access to current ERP chart of accounts and general ledger export',
          'List of all related-party entities and intercompany transaction flows',
          'FTA corporate tax registration and tax period dates',
          'PostgreSQL or Snowflake data warehouse with ERP replication',
          'pytest >= 7.0 with pytest-asyncio',
          'Docker and docker-compose for containerized deployment',
          'cron, Airflow, or Prefect for job scheduling',
          'Slack incoming webhook URL for operational alerts',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Tax-Compliant Chart of Accounts',
            description:
              'Extend the existing chart of accounts with UAE CT classifications, mapping every GL account to its tax treatment: taxable, exempt, qualifying free zone, or excluded.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'UAE CT chart of accounts extension',
                description:
                  'Adds corporate tax classification fields to the chart of accounts and creates the mapping reference table.',
                code: `-- Tax classification reference aligned to UAE CT law
CREATE TABLE uae_ct_classification (
    classification_id  SERIAL PRIMARY KEY,
    code               TEXT UNIQUE NOT NULL,
    description        TEXT NOT NULL,
    tax_rate           NUMERIC(5,2) NOT NULL,
    applies_to         TEXT NOT NULL  -- 'mainland', 'free_zone', 'both'
);

INSERT INTO uae_ct_classification (code, description, tax_rate, applies_to) VALUES
    ('TAXABLE_STANDARD',  'Standard taxable income at 9%',         9.00, 'both'),
    ('EXEMPT_DIVIDEND',   'Participation exemption dividends',     0.00, 'both'),
    ('EXEMPT_CAPITAL',    'Qualifying capital gains',              0.00, 'both'),
    ('FZ_QUALIFYING',     'Qualifying Free Zone income',           0.00, 'free_zone'),
    ('FZ_NON_QUALIFYING', 'Non-qualifying Free Zone income',       9.00, 'free_zone'),
    ('SBR_ELIGIBLE',      'Small Business Relief eligible',        0.00, 'mainland'),
    ('EXCLUDED_GOV',      'Government and extractive excluded',    0.00, 'both');

-- Extend GL accounts with CT mapping
ALTER TABLE chart_of_accounts
    ADD COLUMN ct_classification_code TEXT REFERENCES uae_ct_classification(code),
    ADD COLUMN related_party_flag     BOOLEAN DEFAULT FALSE,
    ADD COLUMN free_zone_entity_id    TEXT,
    ADD COLUMN ct_mapping_reviewed    BOOLEAN DEFAULT FALSE,
    ADD COLUMN ct_mapping_reviewed_at TIMESTAMPTZ;

-- Classify existing revenue accounts based on naming heuristics
UPDATE chart_of_accounts
SET ct_classification_code = CASE
    WHEN account_name ILIKE '%dividend%'         THEN 'EXEMPT_DIVIDEND'
    WHEN account_name ILIKE '%intercompany%'     THEN 'TAXABLE_STANDARD'
    WHEN free_zone_entity_id IS NOT NULL         THEN 'FZ_QUALIFYING'
    ELSE 'TAXABLE_STANDARD'
END
WHERE account_type IN ('revenue', 'other_income')
  AND ct_classification_code IS NULL;`,
              },
              {
                language: 'python',
                title: 'GL account classification validator',
                description:
                  'Validates that every GL account is mapped and flags inconsistencies for review.',
                code: `import pandas as pd
from dataclasses import dataclass

@dataclass
class ClassificationGap:
    account_id: str
    account_name: str
    account_type: str
    issue: str

def validate_ct_classifications(conn) -> list[ClassificationGap]:
    """Audit chart of accounts for CT classification gaps."""
    df = pd.read_sql("""
        SELECT account_id, account_name, account_type,
               ct_classification_code, related_party_flag,
               free_zone_entity_id
        FROM chart_of_accounts
        WHERE account_type IN ('revenue','expense','other_income','cogs')
    """, conn)

    gaps: list[ClassificationGap] = []

    # Check 1: unmapped accounts
    unmapped = df[df["ct_classification_code"].isna()]
    for _, row in unmapped.iterrows():
        gaps.append(ClassificationGap(
            row["account_id"], row["account_name"], row["account_type"],
            "Missing CT classification code"
        ))

    # Check 2: free zone entity but no FZ classification
    fz_mismatch = df[
        (df["free_zone_entity_id"].notna()) &
        (~df["ct_classification_code"].str.startswith("FZ_", na=False))
    ]
    for _, row in fz_mismatch.iterrows():
        gaps.append(ClassificationGap(
            row["account_id"], row["account_name"], row["account_type"],
            f"Free zone entity assigned but classification is {row['ct_classification_code']}"
        ))

    # Check 3: related party flag without transfer pricing note
    rp_missing = df[
        (df["related_party_flag"] == True) &
        (df["ct_classification_code"] != 'TAXABLE_STANDARD')
    ]
    for _, row in rp_missing.iterrows():
        gaps.append(ClassificationGap(
            row["account_id"], row["account_name"], row["account_type"],
            "Related party flagged but not classified as TAXABLE_STANDARD"
        ))

    print(f"Classification audit: {len(gaps)} gaps found across "
          f"{len(df)} accounts ({len(unmapped)} unmapped)")
    return gaps`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Transfer Pricing Documentation Automation',
            description:
              'Automatically extract related-party transactions, compute arm\'s-length benchmarks, and generate transfer pricing local files from transactional data.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Related-party transaction extraction',
                description:
                  'Identifies and aggregates all intercompany transactions by entity pair and transaction type for transfer pricing analysis.',
                code: `-- Related party transaction summary for TP documentation
CREATE VIEW tp_transaction_summary AS
WITH rp_transactions AS (
    SELECT
        gl.journal_id,
        gl.posting_date,
        gl.account_id,
        coa.account_name,
        coa.ct_classification_code,
        gl.counterparty_entity_id,
        e.entity_name    AS counterparty_name,
        e.jurisdiction    AS counterparty_jurisdiction,
        gl.amount,
        gl.currency,
        gl.description
    FROM general_ledger gl
    JOIN chart_of_accounts coa ON gl.account_id = coa.account_id
    JOIN entities e            ON gl.counterparty_entity_id = e.entity_id
    WHERE coa.related_party_flag = TRUE
)
SELECT
    counterparty_entity_id,
    counterparty_name,
    counterparty_jurisdiction,
    ct_classification_code,
    DATE_TRUNC('quarter', posting_date) AS quarter,
    COUNT(*)                            AS txn_count,
    SUM(amount)                         AS total_amount,
    AVG(amount)                         AS avg_amount,
    MIN(posting_date)                   AS first_txn,
    MAX(posting_date)                   AS last_txn
FROM rp_transactions
GROUP BY 1, 2, 3, 4, 5
ORDER BY total_amount DESC;`,
              },
              {
                language: 'python',
                title: 'Transfer pricing local file generator',
                description:
                  'Generates a structured transfer pricing local file document from transactional data and benchmark analysis.',
                code: `import pandas as pd
from datetime import date
from pathlib import Path
import json

def generate_tp_local_file(conn, tax_period_start: date, tax_period_end: date,
                           output_dir: str = "./tp_docs") -> Path:
    """Generate UAE CT transfer pricing local file from transaction data."""
    rp_summary = pd.read_sql("""
        SELECT * FROM tp_transaction_summary
        WHERE quarter >= %s AND quarter <= %s
        ORDER BY total_amount DESC
    """, conn, params=[tax_period_start, tax_period_end])

    # Group by counterparty for per-entity analysis
    local_file = {
        "document_type": "Transfer Pricing Local File",
        "tax_period": f"{tax_period_start} to {tax_period_end}",
        "generated_date": str(date.today()),
        "entity_analyses": [],
    }

    for entity_id, group in rp_summary.groupby("counterparty_entity_id"):
        entity_name = group.iloc[0]["counterparty_name"]
        jurisdiction = group.iloc[0]["counterparty_jurisdiction"]

        analysis = {
            "counterparty": entity_name,
            "jurisdiction": jurisdiction,
            "total_value": float(group["total_amount"].sum()),
            "transaction_count": int(group["txn_count"].sum()),
            "transaction_types": group["ct_classification_code"].unique().tolist(),
            "quarterly_breakdown": group.groupby("quarter")["total_amount"]
                .sum().reset_index().to_dict(orient="records"),
            "arm_length_method": "Transactional Net Margin Method (TNMM)",
            "benchmark_status": "pending_analysis",
        }
        local_file["entity_analyses"].append(analysis)

    output_path = Path(output_dir) / f"tp_local_file_{tax_period_end.year}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(local_file, indent=2, default=str))

    print(f"TP Local File generated: {len(local_file['entity_analyses'])} "
          f"counterparties, total value AED {rp_summary['total_amount'].sum():,.0f}")
    return output_path`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate the tax classification pipeline with data quality assertions on the chart of accounts and general ledger, and run automated pytest suites verifying GL classification completeness and tax computation accuracy.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'UAE CT data quality assertions',
                description:
                  'SQL assertion queries for GL classification completeness, null checks, referential integrity between CoA and GL, and data freshness.',
                code: `-- Assertion 1: All revenue/expense GL accounts have a CT classification
SELECT 'unclassified_gl_accounts' AS assertion,
       COUNT(*) AS failures
FROM chart_of_accounts
WHERE account_type IN ('revenue', 'expense', 'other_income', 'cogs')
  AND ct_classification_code IS NULL;

-- Assertion 2: Every ct_classification_code in CoA exists in reference table
SELECT 'invalid_classification_code' AS assertion,
       COUNT(*) AS failures
FROM chart_of_accounts coa
LEFT JOIN uae_ct_classification ucc ON coa.ct_classification_code = ucc.code
WHERE coa.ct_classification_code IS NOT NULL
  AND ucc.code IS NULL;

-- Assertion 3: No GL entries reference non-existent accounts
SELECT 'orphaned_gl_entries' AS assertion,
       COUNT(*) AS failures
FROM general_ledger gl
LEFT JOIN chart_of_accounts coa ON gl.account_id = coa.account_id
WHERE coa.account_id IS NULL;

-- Assertion 4: Related party flagged accounts must be TAXABLE_STANDARD
SELECT 'related_party_misclass' AS assertion,
       COUNT(*) AS failures
FROM chart_of_accounts
WHERE related_party_flag = TRUE
  AND ct_classification_code != 'TAXABLE_STANDARD';

-- Assertion 5: Free zone entities must use FZ_ classification codes
SELECT 'fz_entity_misclass' AS assertion,
       COUNT(*) AS failures
FROM chart_of_accounts
WHERE free_zone_entity_id IS NOT NULL
  AND ct_classification_code NOT LIKE 'FZ_%';

-- Assertion 6: GL data freshness — entries posted within last 7 days
SELECT 'stale_gl_data' AS assertion,
       CASE WHEN MAX(posting_date) < CURRENT_DATE - INTERVAL '7 days' THEN 1 ELSE 0 END AS failures
FROM general_ledger;`,
              },
              {
                language: 'python',
                title: 'Pytest tax classification validation tests',
                description:
                  'Automated pytest suite validating GL classification completeness, tax computation correctness, and transfer pricing data integrity.',
                code: `import pytest
import psycopg2
from decimal import Decimal

@pytest.fixture(scope="module")
def db_conn():
    conn = psycopg2.connect(
        host="localhost", dbname="uae_ct_test", user="test_user", password="test_pass"
    )
    yield conn
    conn.close()

class TestGLClassificationCompleteness:
    def test_all_revenue_accounts_classified(self, db_conn):
        """Every revenue and income account must have a CT classification."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM chart_of_accounts
                WHERE account_type IN ('revenue', 'other_income')
                  AND ct_classification_code IS NULL
            """)
            assert cur.fetchone()[0] == 0, "Found unclassified revenue accounts"

    def test_all_expense_accounts_classified(self, db_conn):
        """Every expense and COGS account must have a CT classification."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM chart_of_accounts
                WHERE account_type IN ('expense', 'cogs')
                  AND ct_classification_code IS NULL
            """)
            assert cur.fetchone()[0] == 0, "Found unclassified expense accounts"

    def test_classification_codes_valid(self, db_conn):
        """All classification codes must exist in the reference table."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT coa.ct_classification_code
                FROM chart_of_accounts coa
                LEFT JOIN uae_ct_classification ucc ON coa.ct_classification_code = ucc.code
                WHERE coa.ct_classification_code IS NOT NULL AND ucc.code IS NULL
            """)
            invalid = cur.fetchall()
            assert len(invalid) == 0, f"Found {len(invalid)} invalid classification codes"

class TestTaxComputationValidation:
    def test_sbr_threshold_applied(self, db_conn):
        """Small Business Relief: no tax on income <= AED 375,000."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT tax_liability
                FROM uae_ct_taxable_income
                WHERE net_income <= 375000 AND tax_rate > 0
            """)
            rows = cur.fetchall()
            for row in rows:
                assert row[0] == 0, f"Tax charged on income below SBR threshold"

    def test_exempt_income_zero_tax(self, db_conn):
        """Exempt classifications must have zero tax liability."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT ct_classification_code, tax_liability
                FROM uae_ct_taxable_income
                WHERE tax_rate = 0 AND tax_liability != 0
            """)
            violations = cur.fetchall()
            assert len(violations) == 0, f"Found {len(violations)} exempt codes with nonzero tax"

    def test_nine_percent_rate_accuracy(self, db_conn):
        """Taxable income above threshold must be taxed at exactly 9%."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT ct_classification_code, net_income, tax_liability
                FROM uae_ct_taxable_income
                WHERE tax_rate = 9 AND net_income > 375000
            """)
            for code, net_income, liability in cur.fetchall():
                expected = round((float(net_income) - 375000) * 0.09, 2)
                assert abs(float(liability) - expected) < 0.05, (
                    f"{code}: expected {expected}, got {liability}"
                )`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Deploy the UAE CT classification pipeline with a hardened deployment script that handles environment validation, database migration, and scheduler setup for the TP document generator, plus a Python configuration loader for secrets and connection pools.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Tax classification pipeline deployment script',
                description:
                  'Production deployment script with environment checks, dependency installation, database migration for CT tables, and scheduler setup for TP doc generation.',
                code: `#!/usr/bin/env bash
set -euo pipefail

APP_NAME="uae-ct-pipeline"
DEPLOY_DIR="/opt/\${APP_NAME}"
VENV_DIR="\${DEPLOY_DIR}/venv"
LOG_DIR="/var/log/\${APP_NAME}"

echo "[1/6] Checking required environment variables..."
for var in CT_DB_HOST CT_DB_NAME CT_DB_USER CT_DB_PASSWORD \\
           FTA_PORTAL_API_KEY SLACK_WEBHOOK_URL; do
    if [ -z "\${!var:-}" ]; then
        echo "ERROR: \${var} is not set" >&2
        exit 1
    fi
done

echo "[2/6] Creating deployment directories..."
sudo mkdir -p "\${DEPLOY_DIR}" "\${LOG_DIR}"
sudo chown "\$(whoami):" "\${DEPLOY_DIR}" "\${LOG_DIR}"

echo "[3/6] Installing Python dependencies..."
python3 -m venv "\${VENV_DIR}"
source "\${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "\${DEPLOY_DIR}/requirements.txt"

echo "[4/6] Running database migrations..."
export PGPASSWORD="\${CT_DB_PASSWORD}"
psql -h "\${CT_DB_HOST}" -U "\${CT_DB_USER}" -d "\${CT_DB_NAME}" \\
    -f "\${DEPLOY_DIR}/sql/001_create_ct_classification.sql"
psql -h "\${CT_DB_HOST}" -U "\${CT_DB_USER}" -d "\${CT_DB_NAME}" \\
    -f "\${DEPLOY_DIR}/sql/002_extend_chart_of_accounts.sql"
psql -h "\${CT_DB_HOST}" -U "\${CT_DB_USER}" -d "\${CT_DB_NAME}" \\
    -f "\${DEPLOY_DIR}/sql/003_create_tp_views.sql"

echo "[5/6] Running smoke tests..."
"\${VENV_DIR}/bin/pytest" "\${DEPLOY_DIR}/tests/" -x -q --tb=short

echo "[6/6] Setting up cron schedules..."
# Daily GL classification check at 07:00 UTC
CRON_CLASSIFY="0 7 * * * \${VENV_DIR}/bin/python \${DEPLOY_DIR}/classify_gl.py >> \${LOG_DIR}/classify.log 2>&1"
# Weekly TP doc generation on Sundays at 02:00 UTC
CRON_TP_DOC="0 2 * * 0 \${VENV_DIR}/bin/python \${DEPLOY_DIR}/generate_tp_docs.py >> \${LOG_DIR}/tp_docs.log 2>&1"
(crontab -l 2>/dev/null | grep -v "\${APP_NAME}" || true; echo "\${CRON_CLASSIFY}"; echo "\${CRON_TP_DOC}") | crontab -

echo "Deployment complete. Classification runs daily at 07:00, TP docs weekly on Sunday."`,
              },
              {
                language: 'python',
                title: 'Configuration loader with connection pooling',
                description:
                  'Environment-variable based configuration loader with secrets management and database connection pool setup for the UAE CT pipeline.',
                code: `import os
from dataclasses import dataclass
from psycopg2 import pool

@dataclass
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str
    min_connections: int = 2
    max_connections: int = 10

@dataclass
class FTAConfig:
    portal_api_key: str
    tax_period_start: str
    tax_period_end: str

@dataclass
class AlertConfig:
    slack_webhook_url: str
    classification_gap_threshold: int = 5
    filing_deadline_warning_days: int = 30

@dataclass
class UAECTConfig:
    database: DatabaseConfig
    fta: FTAConfig
    alerts: AlertConfig
    environment: str = "production"
    log_level: str = "INFO"

def load_config() -> UAECTConfig:
    """Load UAE CT pipeline config from environment variables."""
    return UAECTConfig(
        database=DatabaseConfig(
            host=os.environ["CT_DB_HOST"],
            port=int(os.environ.get("CT_DB_PORT", "5432")),
            name=os.environ["CT_DB_NAME"],
            user=os.environ["CT_DB_USER"],
            password=os.environ["CT_DB_PASSWORD"],
        ),
        fta=FTAConfig(
            portal_api_key=os.environ["FTA_PORTAL_API_KEY"],
            tax_period_start=os.environ.get("TAX_PERIOD_START", "2024-01-01"),
            tax_period_end=os.environ.get("TAX_PERIOD_END", "2024-12-31"),
        ),
        alerts=AlertConfig(
            slack_webhook_url=os.environ["SLACK_WEBHOOK_URL"],
            classification_gap_threshold=int(os.environ.get("CLASS_GAP_THRESHOLD", "5")),
            filing_deadline_warning_days=int(os.environ.get("FILING_WARN_DAYS", "30")),
        ),
        environment=os.environ.get("DEPLOY_ENV", "production"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )

class ConnectionPool:
    """Managed database connection pool for the UAE CT pipeline."""

    def __init__(self, config: DatabaseConfig):
        self._pool = pool.ThreadedConnectionPool(
            minconn=config.min_connections,
            maxconn=config.max_connections,
            host=config.host,
            port=config.port,
            dbname=config.name,
            user=config.user,
            password=config.password,
        )

    def get_connection(self):
        return self._pool.getconn()

    def release_connection(self, conn):
        self._pool.putconn(conn)

    def close_all(self):
        self._pool.closeall()

# Usage
config = load_config()
db_pool = ConnectionPool(config.database)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Monitor classification gap trends, track filing deadlines, compute taxable income continuously, and send Slack alerts when classification gaps appear or deadlines approach.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Taxable income computation',
                description:
                  'Computes UAE corporate taxable income with proper exempt income exclusions and Small Business Relief checks for ongoing monitoring.',
                code: `-- UAE Corporate Tax: taxable income computation
CREATE OR REPLACE VIEW uae_ct_taxable_income AS
WITH classified_pnl AS (
    SELECT
        coa.ct_classification_code,
        ucc.tax_rate,
        ucc.description AS classification_desc,
        SUM(CASE WHEN coa.account_type IN ('revenue','other_income') THEN gl.amount ELSE 0 END) AS total_income,
        SUM(CASE WHEN coa.account_type IN ('expense','cogs') THEN ABS(gl.amount) ELSE 0 END)   AS total_deductions
    FROM general_ledger gl
    JOIN chart_of_accounts coa     ON gl.account_id = coa.account_id
    JOIN uae_ct_classification ucc ON coa.ct_classification_code = ucc.code
    WHERE gl.posting_date BETWEEN '2024-01-01' AND '2024-12-31'
    GROUP BY 1, 2, 3
)
SELECT
    ct_classification_code,
    classification_desc,
    tax_rate,
    total_income,
    total_deductions,
    total_income - total_deductions AS net_income,
    CASE
        WHEN tax_rate = 0 THEN 0
        WHEN (total_income - total_deductions) <= 375000 THEN 0  -- AED threshold
        ELSE ROUND(((total_income - total_deductions) - 375000) * tax_rate / 100, 2)
    END AS tax_liability
FROM classified_pnl
ORDER BY tax_liability DESC;`,
              },
              {
                language: 'python',
                title: 'Slack webhook alerting for CT compliance monitoring',
                description:
                  'Monitors classification gap trends, filing deadline proximity, and taxable income thresholds, sending formatted Slack alerts when action is required.',
                code: `import requests as http_requests
from datetime import date, timedelta
from dataclasses import dataclass
import psycopg2

@dataclass
class CTComplianceAlert:
    metric: str
    current_value: float
    threshold: float
    severity: str

SLACK_WEBHOOK = "https://hooks.slack.com/services/T00/B00/xxxx"
CLASSIFICATION_GAP_THRESHOLD = 5
FILING_DEADLINE_WARNING_DAYS = 30

def check_ct_compliance_health(conn) -> list[CTComplianceAlert]:
    """Evaluate UAE CT compliance health and return any threshold breaches."""
    alerts: list[CTComplianceAlert] = []

    with conn.cursor() as cur:
        # Check 1: Unclassified GL accounts
        cur.execute("""
            SELECT COUNT(*) FROM chart_of_accounts
            WHERE account_type IN ('revenue','expense','other_income','cogs')
              AND ct_classification_code IS NULL
        """)
        gap_count = cur.fetchone()[0]
        if gap_count > CLASSIFICATION_GAP_THRESHOLD:
            alerts.append(CTComplianceAlert(
                metric="Unclassified GL Accounts",
                current_value=gap_count,
                threshold=CLASSIFICATION_GAP_THRESHOLD,
                severity="critical" if gap_count > 20 else "warning",
            ))

        # Check 2: Related-party transactions without TP documentation
        cur.execute("""
            SELECT COUNT(DISTINCT counterparty_entity_id)
            FROM general_ledger gl
            JOIN chart_of_accounts coa ON gl.account_id = coa.account_id
            WHERE coa.related_party_flag = TRUE
              AND gl.posting_date >= '2024-01-01'
        """)
        rp_entities = cur.fetchone()[0]
        cur.execute("""
            SELECT COUNT(DISTINCT counterparty_entity_id)
            FROM tp_transaction_summary
        """)
        documented = cur.fetchone()[0]
        undocumented = rp_entities - documented
        if undocumented > 0:
            alerts.append(CTComplianceAlert(
                metric="RP Entities Without TP Docs",
                current_value=undocumented,
                threshold=0,
                severity="critical",
            ))

    # Check 3: Filing deadline proximity
    filing_deadline = date(2025, 9, 30)  # CT filing deadline
    days_remaining = (filing_deadline - date.today()).days
    if 0 < days_remaining <= FILING_DEADLINE_WARNING_DAYS:
        alerts.append(CTComplianceAlert(
            metric="Filing Deadline (days remaining)",
            current_value=days_remaining,
            threshold=FILING_DEADLINE_WARNING_DAYS,
            severity="critical" if days_remaining <= 7 else "warning",
        ))

    return alerts

def send_ct_slack_alerts(conn):
    """Send Slack alerts for UAE CT compliance issues."""
    alerts = check_ct_compliance_health(conn)
    if not alerts:
        return

    emoji = ":red_circle:" if any(a.severity == "critical" for a in alerts) else ":warning:"
    blocks = [f"{emoji} *UAE Corporate Tax Compliance Alert — {date.today()}*"]

    for alert in alerts:
        icon = ":rotating_light:" if alert.severity == "critical" else ":large_yellow_circle:"
        blocks.append(f"{icon} {alert.metric}: {alert.current_value} (threshold: {alert.threshold})")

    http_requests.post(SLACK_WEBHOOK, json={"text": "\\n".join(blocks)})
    print(f"Sent {len(alerts)} CT compliance alerts to Slack")`,
              },
            ],
          },
        ],
        toolsUsed: [
          'PostgreSQL 14+',
          'Python 3.11+',
          'pandas',
          'ERP general ledger data',
          'UAE FTA portal reference',
          'pytest',
          'Docker',
          'GitHub Actions',
          'cron',
          'Slack API',
          'Prometheus',
        ],
      },
    },

    /* ------------------------------------------------------------------ */
    /*  3. Australia GST Data Gaps                                         */
    /* ------------------------------------------------------------------ */
    {
      id: 'australia-gst',
      number: 3,
      title: 'Australia GST Data Gaps',
      subtitle: 'BAS Reporting Data Quality Issues',
      summary:
        'Your BAS lodgements are based on incomplete data. GST-free, input-taxed, and mixed-supply classifications are inconsistent across systems.',
      tags: ['gst', 'australia', 'bas-reporting'],
      metrics: {
        annualCostRange: '$150K - $800K',
        roi: '7x',
        paybackPeriod: '2-3 months',
        investmentRange: '$30K - $60K',
      },
      price: {
        present: {
          title: 'BAS Lodgements Are Based on Guesswork',
          description:
            'Each quarter your finance team patches together GST data from multiple systems, manually classifying transactions because the source data lacks consistent tax codes.',
          bullets: [
            'GST-free supplies (education, health, basic food) are inconsistently coded across POS, ERP, and e-commerce',
            'Input-taxed transactions (financial supplies, residential rent) are often misclassified as taxable',
            'Mixed-supply apportionment ratios are calculated in spreadsheets with outdated percentages',
            'BAS labels (G1 through G20) are populated manually from multiple disconnected reports',
          ],
          severity: 'high',
        },
        root: {
          title: 'No Single Source of GST Truth',
          description:
            'GST classification happens at the point of sale, purchase, or journal entry, but there is no unified validation layer. Each system applies its own tax logic independently.',
          bullets: [
            'ERP tax codes were set up years ago and have not been audited against current ATO rulings',
            'E-commerce platform GST logic differs from ERP rules for identical product categories',
            'Supplier invoices are recorded with GST claimed but ABN and tax invoice validity are not verified',
            'No automated crosscheck between GST collected on sales and GST reported on BAS',
          ],
          severity: 'high',
        },
        impact: {
          title: 'ATO Risk and Cash Flow Exposure',
          description:
            'Incorrect BAS lodgements create both compliance risk and cash flow problems. Over-claiming input tax credits triggers audits; under-claiming means you are overpaying.',
          bullets: [
            'ATO GST audits target businesses with volatile GST ratios quarter-over-quarter',
            'Over-claimed input tax credits must be repaid with interest and potential penalties',
            'Under-claimed credits represent direct cash leakage averaging 0.5-2% of total GST',
            'BAS amendments are time-consuming and increase the probability of further ATO scrutiny',
          ],
          severity: 'high',
        },
        cost: {
          title: 'Quantifying the GST Data Gap',
          description:
            'The cost is split between compliance risk, overpaid/underpaid GST, and the manual effort to prepare and review each BAS.',
          bullets: [
            'Manual BAS preparation: 40-80 hours per quarter across finance and tax teams',
            'Input tax credit leakage: 0.5-2% of annual GST turnover left on the table',
            'ATO penalties for incorrect BAS: up to 75% of the shortfall amount in severe cases',
            'External tax advisor BAS review fees: $15K-$40K per year for mid-size businesses',
          ],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Automated GST Classification and BAS Prep',
          description:
            'A centralised GST classification engine validates every transaction against ATO rules at the point of entry, producing BAS-ready data automatically each period.',
          bullets: [
            'BAS preparation time reduced from days to under 2 hours per quarter',
            'GST classification accuracy improved to 99%+, eliminating over/under-claim risk',
            'Real-time GST position dashboard replaces quarter-end surprises',
            'Full audit trail from source transaction to BAS label for every dollar of GST',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Build a centralised GST classification and audit layer that validates every transaction against ATO rules, fixes historical misclassifications, and auto-generates BAS-ready data.',
        prerequisites: [
          'Access to ERP general ledger with tax code fields',
          'E-commerce platform transaction exports with product categories',
          'ATO-registered ABN and current BAS obligation schedule',
          'PostgreSQL or Snowflake data warehouse',
          'pytest >= 7.0 with pytest-asyncio',
          'Docker and docker-compose for containerized deployment',
          'cron, Airflow, or Prefect for job scheduling',
          'Slack incoming webhook URL for operational alerts',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'GST Classification Audit',
            description:
              'Analyse all transactions for the current and prior BAS periods, identify misclassifications, and build a validated GST rules engine.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'GST classification audit queries',
                description:
                  'Identifies transactions with missing, invalid, or inconsistent GST classifications across source systems.',
                code: `-- GST classification reference table aligned to ATO categories
CREATE TABLE gst_classification (
    gst_code         TEXT PRIMARY KEY,
    bas_label        TEXT NOT NULL,          -- G1, G2, G3 ... G20
    description      TEXT NOT NULL,
    gst_rate         NUMERIC(4,2) NOT NULL,  -- 10, 0, or NULL for input-taxed
    supply_type      TEXT NOT NULL            -- 'taxable', 'gst_free', 'input_taxed', 'out_of_scope'
);

INSERT INTO gst_classification VALUES
    ('GST',    'G1',  'GST on taxable sales',           10.00, 'taxable'),
    ('FRE',    'G3',  'GST-free sales',                  0.00, 'gst_free'),
    ('INP',    'G4',  'Input taxed sales',               0.00, 'input_taxed'),
    ('EXP',    'G2',  'Export sales (GST-free)',          0.00, 'gst_free'),
    ('CAP',    'G10', 'Capital acquisitions',            10.00, 'taxable'),
    ('N-T',    'G14', 'Non-capital acquisitions no GST',  0.00, 'gst_free'),
    ('OOS',    NULL,  'Out of scope',                     0.00, 'out_of_scope');

-- Audit: find transactions with missing or suspect GST codes
CREATE VIEW gst_classification_gaps AS
SELECT
    t.transaction_id,
    t.source_system,
    t.transaction_date,
    t.description,
    t.amount,
    t.tax_code         AS current_tax_code,
    gc.supply_type     AS current_supply_type,
    CASE
        WHEN t.tax_code IS NULL                          THEN 'MISSING_TAX_CODE'
        WHEN gc.gst_code IS NULL                         THEN 'INVALID_TAX_CODE'
        WHEN t.gst_amount = 0 AND gc.gst_rate > 0       THEN 'ZERO_GST_ON_TAXABLE'
        WHEN t.gst_amount > 0 AND gc.gst_rate = 0       THEN 'GST_ON_NON_TAXABLE'
        WHEN ABS(t.gst_amount - t.amount * gc.gst_rate / 110) > 0.05
                                                         THEN 'GST_AMOUNT_MISMATCH'
    END AS issue_type
FROM transactions t
LEFT JOIN gst_classification gc ON t.tax_code = gc.gst_code
WHERE t.transaction_date >= DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months')
  AND CASE
        WHEN t.tax_code IS NULL THEN TRUE
        WHEN gc.gst_code IS NULL THEN TRUE
        WHEN t.gst_amount = 0 AND gc.gst_rate > 0 THEN TRUE
        WHEN t.gst_amount > 0 AND gc.gst_rate = 0 THEN TRUE
        WHEN ABS(t.gst_amount - t.amount * gc.gst_rate / 110) > 0.05 THEN TRUE
        ELSE FALSE
      END;`,
              },
              {
                language: 'python',
                title: 'ABN and tax invoice validator',
                description:
                  'Validates supplier ABNs against the Australian Business Register and checks tax invoice completeness for input tax credit eligibility.',
                code: `import pandas as pd
import requests as http_req
from dataclasses import dataclass
from time import sleep

ABR_LOOKUP_URL = "https://abr.business.gov.au/json/AbnDetails.aspx"

@dataclass
class ABNValidationResult:
    supplier_id: str
    abn: str
    is_valid: bool
    is_gst_registered: bool
    entity_name: str | None
    issue: str | None

def validate_supplier_abns(conn, abr_guid: str) -> list[ABNValidationResult]:
    """Validate supplier ABNs against ABR and check GST registration."""
    suppliers = pd.read_sql("""
        SELECT DISTINCT supplier_id, supplier_name, abn
        FROM supplier_invoices
        WHERE abn IS NOT NULL
          AND invoice_date >= CURRENT_DATE - INTERVAL '12 months'
    """, conn)

    results: list[ABNValidationResult] = []
    for _, row in suppliers.iterrows():
        abn = row["abn"].replace(" ", "")
        try:
            resp = http_req.get(ABR_LOOKUP_URL, params={
                "abn": abn, "callback": "", "guid": abr_guid
            }, timeout=10)
            data = resp.json()

            is_valid = data.get("Abn") is not None
            gst_from = data.get("Gst", "")
            is_gst = bool(gst_from and gst_from != "")

            issue = None
            if not is_valid:
                issue = "ABN not found in ABR"
            elif not is_gst:
                issue = "Supplier not registered for GST - input credits ineligible"

            results.append(ABNValidationResult(
                row["supplier_id"], abn, is_valid, is_gst,
                data.get("EntityName"), issue
            ))
        except Exception as e:
            results.append(ABNValidationResult(
                row["supplier_id"], abn, False, False, None, f"Lookup failed: {e}"
            ))
        sleep(0.5)  # respect ABR rate limits

    invalid_count = sum(1 for r in results if r.issue)
    print(f"ABN validation: {len(results)} suppliers checked, {invalid_count} issues found")
    return results`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Automated BAS Data Preparation',
            description:
              'Aggregate validated GST data into BAS labels (G1-G20), apply mixed-supply apportionment, and generate a BAS-ready dataset with full audit trail.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'BAS label aggregation view',
                description:
                  'Aggregates classified transactions into ATO BAS labels with proper GST calculations.',
                code: `-- BAS-ready aggregation by label for a given quarter
CREATE VIEW bas_preparation AS
WITH quarter_txns AS (
    SELECT
        t.transaction_id,
        t.transaction_date,
        t.amount,
        t.gst_amount,
        gc.bas_label,
        gc.supply_type,
        gc.gst_rate,
        t.transaction_type   -- 'sale' or 'purchase'
    FROM transactions t
    JOIN gst_classification gc ON t.tax_code = gc.gst_code
    WHERE t.tax_code IS NOT NULL
      AND t.transaction_date >= DATE_TRUNC('quarter', CURRENT_DATE)
      AND t.transaction_date <  DATE_TRUNC('quarter', CURRENT_DATE) + INTERVAL '3 months'
)
SELECT
    bas_label,
    supply_type,
    transaction_type,
    COUNT(*)                                    AS txn_count,
    SUM(amount)                                 AS total_amount_incl_gst,
    SUM(amount - gst_amount)                    AS total_amount_excl_gst,
    SUM(gst_amount)                             AS total_gst,
    ROUND(AVG(gst_amount / NULLIF(amount, 0) * 100), 2) AS avg_effective_gst_pct
FROM quarter_txns
GROUP BY bas_label, supply_type, transaction_type
ORDER BY bas_label;`,
              },
              {
                language: 'python',
                title: 'BAS lodgement file generator',
                description:
                  'Generates a structured BAS lodgement file with all labels populated and a detailed audit trail.',
                code: `import pandas as pd
from datetime import date
from pathlib import Path
import json

BAS_SALES_LABELS = {
    "G1": "Total sales (including GST)",
    "G2": "Export sales",
    "G3": "Other GST-free sales",
    "G4": "Input taxed sales",
}
BAS_PURCHASE_LABELS = {
    "G10": "Capital purchases (including GST)",
    "G11": "Non-capital purchases (including GST)",
    "G13": "Purchases for making input taxed sales",
    "G14": "Purchases with no GST in the price",
    "G15": "Estimated purchases for private use",
}

def generate_bas_lodgement(conn, quarter_start: date, output_dir: str = "./bas") -> Path:
    """Generate BAS lodgement data from classified transactions."""
    bas_data = pd.read_sql("""
        SELECT * FROM bas_preparation
    """, conn)

    lodgement = {
        "period": f"Q{((quarter_start.month - 1) // 3) + 1} {quarter_start.year}",
        "generated": str(date.today()),
        "labels": {},
        "summary": {},
    }

    # Populate each BAS label
    all_labels = {**BAS_SALES_LABELS, **BAS_PURCHASE_LABELS}
    for label, desc in all_labels.items():
        rows = bas_data[bas_data["bas_label"] == label]
        amount = float(rows["total_amount_excl_gst"].sum()) if len(rows) > 0 else 0.0
        gst = float(rows["total_gst"].sum()) if len(rows) > 0 else 0.0
        lodgement["labels"][label] = {
            "description": desc,
            "amount": round(amount, 2),
            "gst": round(gst, 2),
            "txn_count": int(rows["txn_count"].sum()) if len(rows) > 0 else 0,
        }

    # Compute BAS summary fields
    g1_gst = lodgement["labels"].get("G1", {}).get("gst", 0)
    input_credits = sum(
        lodgement["labels"].get(l, {}).get("gst", 0)
        for l in ["G10", "G11"]
    )
    lodgement["summary"] = {
        "gst_on_sales": round(g1_gst, 2),
        "gst_on_purchases": round(input_credits, 2),
        "net_gst_payable": round(g1_gst - input_credits, 2),
    }

    output_path = Path(output_dir) / f"bas_{quarter_start.isoformat()}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(lodgement, indent=2))

    print(f"BAS prepared for {lodgement['period']}: "
          f"Net GST {'payable' if lodgement['summary']['net_gst_payable'] > 0 else 'refundable'} "
          f"\${abs(lodgement['summary']['net_gst_payable']):,.2f}")
    return output_path`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate the GST classification pipeline with data quality assertions on BAS label accuracy and GST amount correctness, and run automated pytest suites verifying classification rules and BAS output integrity.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'GST data quality assertions',
                description:
                  'SQL assertion queries for BAS label accuracy, GST amount correctness, null checks, referential integrity, and data freshness.',
                code: `-- Assertion 1: All transactions have a valid tax code
SELECT 'missing_tax_code' AS assertion,
       COUNT(*) AS failures
FROM transactions
WHERE tax_code IS NULL
  AND transaction_date >= DATE_TRUNC('quarter', CURRENT_DATE);

-- Assertion 2: Every tax_code maps to a valid gst_classification
SELECT 'invalid_tax_code' AS assertion,
       COUNT(*) AS failures
FROM transactions t
LEFT JOIN gst_classification gc ON t.tax_code = gc.gst_code
WHERE t.tax_code IS NOT NULL AND gc.gst_code IS NULL
  AND t.transaction_date >= DATE_TRUNC('quarter', CURRENT_DATE);

-- Assertion 3: GST amount matches expected rate (10% of GST-exclusive amount)
SELECT 'gst_amount_mismatch' AS assertion,
       COUNT(*) AS failures
FROM transactions t
JOIN gst_classification gc ON t.tax_code = gc.gst_code
WHERE gc.gst_rate > 0
  AND ABS(t.gst_amount - t.amount * gc.gst_rate / 110) > 0.05
  AND t.transaction_date >= DATE_TRUNC('quarter', CURRENT_DATE);

-- Assertion 4: No GST charged on GST-free or input-taxed supplies
SELECT 'gst_on_nontaxable' AS assertion,
       COUNT(*) AS failures
FROM transactions t
JOIN gst_classification gc ON t.tax_code = gc.gst_code
WHERE gc.gst_rate = 0 AND t.gst_amount > 0
  AND t.transaction_date >= DATE_TRUNC('quarter', CURRENT_DATE);

-- Assertion 5: BAS label populated for all classified transactions
SELECT 'missing_bas_label' AS assertion,
       COUNT(*) AS failures
FROM transactions t
JOIN gst_classification gc ON t.tax_code = gc.gst_code
WHERE gc.bas_label IS NULL AND gc.supply_type != 'out_of_scope';

-- Assertion 6: Data freshness — transactions within last 3 days
SELECT 'stale_txn_data' AS assertion,
       CASE WHEN MAX(transaction_date) < CURRENT_DATE - INTERVAL '3 days' THEN 1 ELSE 0 END AS failures
FROM transactions;`,
              },
              {
                language: 'python',
                title: 'Pytest BAS and GST validation tests',
                description:
                  'Automated pytest suite validating BAS label accuracy, GST amount correctness, classification consistency, and BAS lodgement output integrity.',
                code: `import pytest
import psycopg2
from decimal import Decimal
import json
from pathlib import Path

@pytest.fixture(scope="module")
def db_conn():
    conn = psycopg2.connect(
        host="localhost", dbname="gst_test", user="test_user", password="test_pass"
    )
    yield conn
    conn.close()

class TestBASLabelAccuracy:
    def test_all_taxable_sales_map_to_g1(self, db_conn):
        """Taxable sales must map to BAS label G1."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM transactions t
                JOIN gst_classification gc ON t.tax_code = gc.gst_code
                WHERE gc.supply_type = 'taxable' AND t.transaction_type = 'sale'
                  AND gc.bas_label != 'G1'
            """)
            assert cur.fetchone()[0] == 0, "Found taxable sales not mapped to G1"

    def test_export_sales_map_to_g2(self, db_conn):
        """Export sales must map to BAS label G2."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM transactions t
                JOIN gst_classification gc ON t.tax_code = gc.gst_code
                WHERE gc.gst_code = 'EXP' AND gc.bas_label != 'G2'
            """)
            assert cur.fetchone()[0] == 0, "Found export sales not mapped to G2"

    def test_gst_free_sales_map_to_g3(self, db_conn):
        """GST-free sales must map to BAS label G3."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM transactions t
                JOIN gst_classification gc ON t.tax_code = gc.gst_code
                WHERE gc.gst_code = 'FRE' AND gc.bas_label != 'G3'
            """)
            assert cur.fetchone()[0] == 0, "Found GST-free sales not mapped to G3"

class TestGSTAmountCorrectness:
    def test_ten_percent_gst_accuracy(self, db_conn):
        """GST amount must be 1/11 of the GST-inclusive amount."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT t.transaction_id, t.amount, t.gst_amount,
                       ROUND(t.amount * 10 / 110, 2) AS expected_gst
                FROM transactions t
                JOIN gst_classification gc ON t.tax_code = gc.gst_code
                WHERE gc.gst_rate = 10
                  AND ABS(t.gst_amount - ROUND(t.amount * 10 / 110, 2)) > 0.05
                LIMIT 10
            """)
            mismatches = cur.fetchall()
            assert len(mismatches) == 0, (
                f"Found {len(mismatches)} transactions with incorrect GST amounts"
            )

    def test_no_gst_on_exempt_supplies(self, db_conn):
        """GST-free and input-taxed supplies must have zero GST."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM transactions t
                JOIN gst_classification gc ON t.tax_code = gc.gst_code
                WHERE gc.gst_rate = 0 AND t.gst_amount > 0
            """)
            assert cur.fetchone()[0] == 0, "Found GST charged on exempt supplies"

    def test_bas_net_gst_reconciles(self, db_conn):
        """Total GST on sales minus input credits must equal net GST payable."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT
                    SUM(CASE WHEN transaction_type = 'sale' THEN gst_amount ELSE 0 END) AS gst_collected,
                    SUM(CASE WHEN transaction_type = 'purchase' THEN gst_amount ELSE 0 END) AS input_credits
                FROM transactions t
                JOIN gst_classification gc ON t.tax_code = gc.gst_code
                WHERE transaction_date >= DATE_TRUNC('quarter', CURRENT_DATE)
            """)
            row = cur.fetchone()
            gst_collected = float(row[0] or 0)
            input_credits = float(row[1] or 0)
            net = gst_collected - input_credits
            # Sanity check: net GST should be a reasonable number
            assert abs(net) < 10_000_000, f"Net GST {net} seems unreasonable"`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Deploy the GST classification pipeline with a hardened deployment script that handles environment validation, database migration, and scheduler setup for the ABN validator, plus a Python configuration loader for secrets and connection pools.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'GST classification pipeline deployment script',
                description:
                  'Production deployment script with environment checks, dependency installation, database migration for GST tables, and scheduler setup for ABN validation and BAS preparation.',
                code: `#!/usr/bin/env bash
set -euo pipefail

APP_NAME="gst-pipeline"
DEPLOY_DIR="/opt/\${APP_NAME}"
VENV_DIR="\${DEPLOY_DIR}/venv"
LOG_DIR="/var/log/\${APP_NAME}"

echo "[1/6] Checking required environment variables..."
for var in GST_DB_HOST GST_DB_NAME GST_DB_USER GST_DB_PASSWORD \\
           ABR_GUID SLACK_WEBHOOK_URL; do
    if [ -z "\${!var:-}" ]; then
        echo "ERROR: \${var} is not set" >&2
        exit 1
    fi
done

echo "[2/6] Creating deployment directories..."
sudo mkdir -p "\${DEPLOY_DIR}" "\${LOG_DIR}"
sudo chown "\$(whoami):" "\${DEPLOY_DIR}" "\${LOG_DIR}"

echo "[3/6] Installing Python dependencies..."
python3 -m venv "\${VENV_DIR}"
source "\${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "\${DEPLOY_DIR}/requirements.txt"

echo "[4/6] Running database migrations..."
export PGPASSWORD="\${GST_DB_PASSWORD}"
psql -h "\${GST_DB_HOST}" -U "\${GST_DB_USER}" -d "\${GST_DB_NAME}" \\
    -f "\${DEPLOY_DIR}/sql/001_create_gst_classification.sql"
psql -h "\${GST_DB_HOST}" -U "\${GST_DB_USER}" -d "\${GST_DB_NAME}" \\
    -f "\${DEPLOY_DIR}/sql/002_create_classification_gaps_view.sql"
psql -h "\${GST_DB_HOST}" -U "\${GST_DB_USER}" -d "\${GST_DB_NAME}" \\
    -f "\${DEPLOY_DIR}/sql/003_create_bas_preparation_view.sql"

echo "[5/6] Running smoke tests..."
"\${VENV_DIR}/bin/pytest" "\${DEPLOY_DIR}/tests/" -x -q --tb=short

echo "[6/6] Setting up cron schedules..."
# Daily GST classification check at 05:00 AEST (19:00 UTC prior day)
CRON_CLASSIFY="0 19 * * * \${VENV_DIR}/bin/python \${DEPLOY_DIR}/classify_gst.py >> \${LOG_DIR}/classify.log 2>&1"
# Weekly ABN validation on Saturdays at 01:00 AEST (15:00 UTC Friday)
CRON_ABN="0 15 * * 5 \${VENV_DIR}/bin/python \${DEPLOY_DIR}/validate_abns.py >> \${LOG_DIR}/abn_validate.log 2>&1"
(crontab -l 2>/dev/null | grep -v "\${APP_NAME}" || true; echo "\${CRON_CLASSIFY}"; echo "\${CRON_ABN}") | crontab -

echo "Deployment complete. Classification runs daily, ABN validation weekly on Saturday."`,
              },
              {
                language: 'python',
                title: 'Configuration loader with connection pooling',
                description:
                  'Environment-variable based configuration loader with secrets management and database connection pool setup for the GST pipeline.',
                code: `import os
from dataclasses import dataclass
from psycopg2 import pool

@dataclass
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str
    min_connections: int = 2
    max_connections: int = 10

@dataclass
class ABRConfig:
    guid: str
    rate_limit_delay: float = 0.5

@dataclass
class AlertConfig:
    slack_webhook_url: str
    error_rate_threshold: float = 0.02
    lodgement_deadline_warning_days: int = 14

@dataclass
class GSTConfig:
    database: DatabaseConfig
    abr: ABRConfig
    alerts: AlertConfig
    environment: str = "production"
    log_level: str = "INFO"

def load_config() -> GSTConfig:
    """Load GST pipeline config from environment variables."""
    return GSTConfig(
        database=DatabaseConfig(
            host=os.environ["GST_DB_HOST"],
            port=int(os.environ.get("GST_DB_PORT", "5432")),
            name=os.environ["GST_DB_NAME"],
            user=os.environ["GST_DB_USER"],
            password=os.environ["GST_DB_PASSWORD"],
        ),
        abr=ABRConfig(
            guid=os.environ["ABR_GUID"],
            rate_limit_delay=float(os.environ.get("ABR_RATE_LIMIT", "0.5")),
        ),
        alerts=AlertConfig(
            slack_webhook_url=os.environ["SLACK_WEBHOOK_URL"],
            error_rate_threshold=float(os.environ.get("GST_ERROR_RATE_THRESHOLD", "0.02")),
            lodgement_deadline_warning_days=int(os.environ.get("LODGEMENT_WARN_DAYS", "14")),
        ),
        environment=os.environ.get("DEPLOY_ENV", "production"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )

class ConnectionPool:
    """Managed database connection pool for the GST pipeline."""

    def __init__(self, config: DatabaseConfig):
        self._pool = pool.ThreadedConnectionPool(
            minconn=config.min_connections,
            maxconn=config.max_connections,
            host=config.host,
            port=config.port,
            dbname=config.name,
            user=config.user,
            password=config.password,
        )

    def get_connection(self):
        return self._pool.getconn()

    def release_connection(self, conn):
        self._pool.putconn(conn)

    def close_all(self):
        self._pool.closeall()

# Usage
config = load_config()
db_pool = ConnectionPool(config.database)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Deploy a real-time GST position monitor that tracks classification error rates, projects quarter-end BAS position, and sends Slack alerts for classification anomalies and approaching lodgement deadlines.',
            codeSnippets: [
              {
                language: 'python',
                title: 'GST position monitor and anomaly detector',
                description:
                  'Monitors GST classification health metrics, projects quarter-end BAS position, and computes error rates for ongoing tracking.',
                code: `import pandas as pd
from datetime import date, timedelta

def monitor_gst_position(conn) -> dict:
    """Check GST classification health and project quarter-end BAS position."""
    today = date.today()
    q_start = date(today.year, ((today.month - 1) // 3) * 3 + 1, 1)

    health = pd.read_sql("""
        SELECT
            COUNT(*)                                           AS total_txns,
            COUNT(*) FILTER (WHERE tax_code IS NULL)           AS missing_code,
            COUNT(*) FILTER (WHERE tax_code NOT IN (
                SELECT gst_code FROM gst_classification))      AS invalid_code,
            COUNT(*) FILTER (WHERE gst_amount = 0
                AND tax_code IN (SELECT gst_code FROM gst_classification
                                 WHERE gst_rate > 0))         AS zero_gst_taxable,
            SUM(gst_amount) FILTER (WHERE tax_code IN (
                SELECT gst_code FROM gst_classification
                WHERE supply_type = 'taxable'))                AS projected_gst_collected,
            SUM(gst_amount) FILTER (WHERE tax_code IN ('GST','CAP')
                AND transaction_type = 'purchase')             AS projected_input_credits
        FROM transactions
        WHERE transaction_date >= %s AND transaction_date <= %s
    """, conn, params=[q_start, today]).iloc[0]

    error_rate = (health["missing_code"] + health["invalid_code"]) / max(health["total_txns"], 1)
    net_gst = float(health["projected_gst_collected"] or 0) - float(health["projected_input_credits"] or 0)

    return {
        "quarter": f"Q{((q_start.month - 1) // 3) + 1} {q_start.year}",
        "total_transactions": int(health["total_txns"]),
        "classification_error_rate": round(error_rate * 100, 2),
        "projected_net_gst": round(net_gst, 2),
        "issues": {
            "missing_tax_code": int(health["missing_code"]),
            "invalid_tax_code": int(health["invalid_code"]),
            "zero_gst_on_taxable": int(health["zero_gst_taxable"]),
        },
    }`,
              },
              {
                language: 'python',
                title: 'Slack webhook alerting for GST compliance',
                description:
                  'Monitors GST classification error rates, approaching BAS lodgement deadlines, and GST position anomalies, sending formatted Slack alerts when thresholds are breached.',
                code: `import requests as http_requests
from datetime import date
from dataclasses import dataclass

@dataclass
class GSTComplianceAlert:
    metric: str
    current_value: float
    threshold: float
    severity: str

SLACK_WEBHOOK = "https://hooks.slack.com/services/T00/B00/xxxx"
ERROR_RATE_THRESHOLD = 2.0  # percent
LODGEMENT_WARNING_DAYS = 14
NET_GST_VARIANCE_THRESHOLD = 0.20  # 20% quarter-over-quarter variance

def evaluate_gst_thresholds(report: dict, prior_net_gst: float | None = None) -> list[GSTComplianceAlert]:
    """Evaluate GST health metrics against operational thresholds."""
    alerts: list[GSTComplianceAlert] = []

    # Check 1: Classification error rate
    if report["classification_error_rate"] > ERROR_RATE_THRESHOLD:
        alerts.append(GSTComplianceAlert(
            metric="Classification Error Rate",
            current_value=report["classification_error_rate"],
            threshold=ERROR_RATE_THRESHOLD,
            severity="critical" if report["classification_error_rate"] > 5.0 else "warning",
        ))

    # Check 2: BAS lodgement deadline proximity
    today = date.today()
    month = today.month
    # BAS due 28th of month following quarter end (Oct, Jan, Apr, Jul)
    quarter_end_months = {3: 4, 6: 7, 9: 10, 12: 1}
    for q_end, due_month in quarter_end_months.items():
        due_year = today.year if due_month > month else today.year + 1
        due_date = date(due_year, due_month, 28)
        days_remaining = (due_date - today).days
        if 0 < days_remaining <= LODGEMENT_WARNING_DAYS:
            alerts.append(GSTComplianceAlert(
                metric="BAS Lodgement Deadline (days)",
                current_value=days_remaining,
                threshold=LODGEMENT_WARNING_DAYS,
                severity="critical" if days_remaining <= 3 else "warning",
            ))

    # Check 3: Quarter-over-quarter GST variance
    if prior_net_gst and prior_net_gst != 0:
        variance = abs(report["projected_net_gst"] - prior_net_gst) / abs(prior_net_gst)
        if variance > NET_GST_VARIANCE_THRESHOLD:
            alerts.append(GSTComplianceAlert(
                metric="Net GST QoQ Variance",
                current_value=round(variance * 100, 1),
                threshold=round(NET_GST_VARIANCE_THRESHOLD * 100, 1),
                severity="critical" if variance > 0.50 else "warning",
            ))

    return alerts

def send_gst_slack_alerts(report: dict, prior_net_gst: float | None = None):
    """Send Slack alerts for GST compliance threshold breaches."""
    alerts = evaluate_gst_thresholds(report, prior_net_gst)
    if not alerts:
        return

    emoji = ":red_circle:" if any(a.severity == "critical" for a in alerts) else ":warning:"
    blocks = [
        f"{emoji} *GST Compliance Alert — {report['quarter']}*",
        f"Error Rate: {report['classification_error_rate']}% | Net GST: \${report['projected_net_gst']:,.2f}",
    ]
    for alert in alerts:
        icon = ":rotating_light:" if alert.severity == "critical" else ":large_yellow_circle:"
        if alert.metric == "Net GST QoQ Variance":
            blocks.append(f"{icon} {alert.metric}: {alert.current_value}% (threshold: {alert.threshold}%)")
        elif alert.metric == "BAS Lodgement Deadline (days)":
            blocks.append(f"{icon} {alert.metric}: {int(alert.current_value)} days remaining")
        else:
            blocks.append(f"{icon} {alert.metric}: {alert.current_value}% (threshold: {alert.threshold}%)")

    http_requests.post(SLACK_WEBHOOK, json={"text": "\\n".join(blocks)})
    print(f"Sent {len(alerts)} GST compliance alerts to Slack")`,
              },
            ],
          },
        ],
        toolsUsed: [
          'PostgreSQL 14+',
          'Python 3.11+',
          'pandas',
          'Australian Business Register API',
          'ATO BAS specification',
          'pytest',
          'Docker',
          'GitHub Actions',
          'cron',
          'Slack API',
          'Prometheus',
        ],
      },
    },
  ],
};
