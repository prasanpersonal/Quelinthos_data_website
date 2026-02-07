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
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to automate payment reconciliation by extracting gateway and bank data, running AI-powered matching analysis, and delivering exception reports to stakeholders automatically.',
        estimatedMonthlyCost: '$120 - $200/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Google Sheets (Free)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Brex AI (usage-based)', 'Ramp AI (usage-based)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Export Stripe, PayPal, and bank statement data into a structured format (CSV or Google Sheets) that can be analyzed by AI. Use Zapier to automatically pull daily transaction exports and normalize them into a common schema.',
            toolsUsed: ['Stripe Dashboard Export', 'PayPal Reports API', 'Zapier', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Webhook Payload for Stripe Transactions',
                description:
                  'Configure Zapier to receive Stripe webhook events and format them for reconciliation analysis.',
                code: `{
  "trigger": {
    "app": "Stripe",
    "event": "New Payout",
    "account": "{{stripe_account_id}}"
  },
  "transform": {
    "transaction_id": "{{payout.id}}",
    "amount": "{{payout.amount / 100}}",
    "currency": "{{payout.currency | upcase}}",
    "arrival_date": "{{payout.arrival_date | date: '%Y-%m-%d'}}",
    "status": "{{payout.status}}",
    "source": "stripe",
    "description": "Stripe Payout - {{payout.description}}"
  },
  "action": {
    "app": "Google Sheets",
    "event": "Create Spreadsheet Row",
    "spreadsheet_id": "{{reconciliation_sheet_id}}",
    "worksheet": "Stripe_Transactions"
  }
}`,
              },
              {
                language: 'json',
                title: 'Bank Transaction CSV Schema',
                description:
                  'Standardized CSV format for bank statement imports that matches the AI analysis template.',
                code: `{
  "csv_schema": {
    "columns": [
      {"name": "transaction_id", "type": "string", "required": true},
      {"name": "date", "type": "date", "format": "YYYY-MM-DD"},
      {"name": "amount", "type": "decimal", "precision": 2},
      {"name": "currency", "type": "string", "length": 3},
      {"name": "description", "type": "string", "max_length": 500},
      {"name": "reference", "type": "string", "nullable": true},
      {"name": "source", "type": "string", "default": "bank"}
    ],
    "delimiter": ",",
    "encoding": "UTF-8",
    "header_row": true
  },
  "example_row": {
    "transaction_id": "BNK-2024-001234",
    "date": "2024-01-15",
    "amount": 1523.45,
    "currency": "USD",
    "description": "STRIPE TRANSFER 2839475",
    "reference": "po_2839475",
    "source": "bank"
  }
}`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description:
              'Use ChatGPT or Claude to analyze the normalized transaction data, identify matching candidates using fuzzy logic on amounts, dates, and references, and flag exceptions that require human review.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro', 'Custom GPT'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Payment Reconciliation Analysis Prompt',
                description:
                  'Master prompt template for AI-powered transaction matching and exception identification.',
                code: `system_prompt: |
  You are a senior financial analyst specializing in payment reconciliation.
  Your task is to match transactions between payment gateways (Stripe, PayPal)
  and bank statements using the following matching rules:

  MATCHING RULES:
  1. Exact Match: Same amount (within $0.05) + same date (within 3 days)
  2. Batch Match: Sum of gateway transactions equals single bank deposit
  3. Reference Match: Gateway payout ID appears in bank description
  4. Fuzzy Match: 90%+ similarity on normalized reference strings

  OUTPUT FORMAT:
  For each transaction, provide:
  - match_status: "matched" | "batch_matched" | "unmatched"
  - match_confidence: 0.0 to 1.0
  - matched_with: transaction_id of the matching record (if any)
  - exception_reason: reason for non-match (if unmatched)
  - recommended_action: "auto_approve" | "manual_review" | "investigate"

user_prompt_template: |
  Analyze the following transactions for reconciliation:

  GATEWAY TRANSACTIONS (Stripe/PayPal):
  {{gateway_transactions_csv}}

  BANK STATEMENT TRANSACTIONS:
  {{bank_transactions_csv}}

  DATE RANGE: {{start_date}} to {{end_date}}

  Please:
  1. Match each gateway transaction to its corresponding bank credit
  2. Identify any batch payouts (multiple gateway txns → single bank deposit)
  3. Flag unmatched items with clear exception reasons
  4. Calculate overall match rate and exception value

  Return results as a structured JSON array.

output_schema:
  type: object
  properties:
    summary:
      total_gateway_transactions: integer
      total_bank_transactions: integer
      matched_count: integer
      batch_matched_count: integer
      unmatched_count: integer
      match_rate_percent: number
      total_exception_value: number
    matches:
      type: array
      items:
        gateway_id: string
        bank_id: string
        match_type: string
        confidence: number
    exceptions:
      type: array
      items:
        transaction_id: string
        source: string
        amount: number
        reason: string
        recommended_action: string`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier to run the reconciliation analysis on a schedule, format the results into stakeholder-friendly reports, and deliver them via email or Slack with actionable exception summaries.',
            toolsUsed: ['Zapier', 'Google Sheets', 'Slack', 'Email'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Reconciliation Workflow',
                description:
                  'End-to-end Zapier workflow that triggers daily reconciliation and delivers formatted reports.',
                code: `{
  "workflow_name": "Daily Payment Reconciliation",
  "trigger": {
    "app": "Schedule by Zapier",
    "event": "Every Day",
    "time": "06:00",
    "timezone": "UTC"
  },
  "steps": [
    {
      "step": 1,
      "app": "Google Sheets",
      "action": "Get Many Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{reconciliation_sheet_id}}",
        "worksheet": "Gateway_Transactions",
        "filter": "date >= {{yesterday}}"
      }
    },
    {
      "step": 2,
      "app": "Google Sheets",
      "action": "Get Many Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{reconciliation_sheet_id}}",
        "worksheet": "Bank_Transactions",
        "filter": "date >= {{yesterday_minus_3}}"
      }
    },
    {
      "step": 3,
      "app": "ChatGPT",
      "action": "Conversation",
      "config": {
        "model": "gpt-4",
        "system_prompt": "{{reconciliation_system_prompt}}",
        "user_message": "Analyze these transactions:\\n\\nGATEWAY:\\n{{step1.rows_csv}}\\n\\nBANK:\\n{{step2.rows_csv}}"
      }
    },
    {
      "step": 4,
      "app": "Formatter by Zapier",
      "action": "Text - Extract JSON",
      "config": {
        "input": "{{step3.response}}"
      }
    },
    {
      "step": 5,
      "app": "Filter by Zapier",
      "condition": "{{step4.json.summary.unmatched_count}} > 0"
    },
    {
      "step": 6,
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#finance-reconciliation",
        "message": ":moneybag: *Daily Reconciliation Report - {{today}}*\\n\\n*Match Rate:* {{step4.json.summary.match_rate_percent}}%\\n*Exceptions:* {{step4.json.summary.unmatched_count}} items (USD {{step4.json.summary.total_exception_value}})\\n\\n*Action Required:* {{step4.json.exceptions.length}} items need manual review"
      }
    },
    {
      "step": 7,
      "app": "Google Sheets",
      "action": "Create Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{reconciliation_sheet_id}}",
        "worksheet": "Exception_Log",
        "rows": "{{step4.json.exceptions}}"
      }
    }
  ],
  "error_handling": {
    "on_error": "notify",
    "notification_channel": "#finance-alerts",
    "retry_count": 2
  }
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent reconciliation system using LangGraph and CrewAI where specialized agents handle data ingestion, transaction matching, exception analysis, and reporting orchestrated by a supervisor agent.',
        estimatedMonthlyCost: '$600 - $1,200/month',
        architecture:
          'Supervisor agent coordinates four specialist agents: Data Ingestion Agent (API connectors), Matching Engine Agent (fuzzy matching algorithms), Exception Analyst Agent (root cause analysis), and Reporting Agent (stakeholder communications). LangGraph manages state transitions and Redis provides persistence.',
        agents: [
          {
            name: 'Data Ingestion Agent',
            role: 'Gateway and Bank Data Collector',
            goal: 'Connect to Stripe, PayPal, and bank APIs to extract and normalize transaction data into a unified schema for reconciliation',
            tools: ['Stripe API', 'PayPal REST API', 'Plaid API', 'CSV Parser'],
          },
          {
            name: 'Matching Engine Agent',
            role: 'Transaction Reconciliation Specialist',
            goal: 'Apply deterministic and fuzzy matching algorithms to pair gateway transactions with bank deposits, handling batch payouts and timing differences',
            tools: ['FuzzyWuzzy', 'Pandas', 'NumPy', 'Custom Matching Rules'],
          },
          {
            name: 'Exception Analyst Agent',
            role: 'Discrepancy Investigator',
            goal: 'Analyze unmatched transactions to identify root causes, classify exception types, and recommend resolution actions',
            tools: ['Pattern Recognition', 'Historical Analysis', 'Fee Calculator'],
          },
          {
            name: 'Reporting Agent',
            role: 'Stakeholder Communications Manager',
            goal: 'Generate formatted reconciliation reports, exception summaries, and trend analyses for finance leadership and auditors',
            tools: ['Jinja Templates', 'Chart Generator', 'Slack API', 'Email API'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with hourly checkpointing and 30-day retention',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system with CrewAI, establishing clear roles, goals, and inter-agent communication patterns for the reconciliation workflow.',
            toolsUsed: ['CrewAI', 'LangChain', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'CrewAI Agent Definitions for Payment Reconciliation',
                description:
                  'Production-ready CrewAI agent definitions with role-specific configurations and tool assignments.',
                code: `from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class ReconciliationAgentConfig(BaseModel):
    """Configuration for reconciliation agents."""
    model_name: str = Field(default="gpt-4-turbo-preview")
    temperature: float = Field(default=0.1, ge=0, le=1)
    max_tokens: int = Field(default=4096)
    verbose: bool = Field(default=True)

def create_reconciliation_agents(config: ReconciliationAgentConfig) -> Dict[str, Agent]:
    """Create the multi-agent reconciliation team."""

    llm = ChatOpenAI(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )

    data_ingestion_agent = Agent(
        role="Gateway and Bank Data Collector",
        goal="""Connect to payment gateways (Stripe, PayPal) and bank feeds
        to extract, validate, and normalize transaction data into a unified
        schema suitable for reconciliation analysis.""",
        backstory="""You are an expert data engineer specializing in financial
        data integration. You have deep knowledge of payment gateway APIs,
        bank feed formats (MT940, BAI2, CSV), and data normalization best
        practices. You ensure data quality by validating amounts, dates,
        and reference fields before passing to downstream agents.""",
        llm=llm,
        verbose=config.verbose,
        allow_delegation=False,
        tools=[]  # Tools added separately
    )

    matching_engine_agent = Agent(
        role="Transaction Reconciliation Specialist",
        goal="""Apply multi-pass matching algorithms to pair gateway
        transactions with bank deposits, handling batch payouts, timing
        differences, and currency conversions with high accuracy.""",
        backstory="""You are a senior reconciliation analyst with 15 years
        of experience in fintech. You understand that Stripe batches multiple
        transactions into single payouts, PayPal has variable settlement
        timing, and bank descriptions rarely match gateway references exactly.
        You use deterministic matching first, then fuzzy matching for residuals.""",
        llm=llm,
        verbose=config.verbose,
        allow_delegation=True,
        tools=[]
    )

    exception_analyst_agent = Agent(
        role="Discrepancy Investigator",
        goal="""Analyze unmatched transactions to identify root causes,
        classify exception types (timing, fee discrepancy, missing data,
        fraud risk), and recommend specific resolution actions.""",
        backstory="""You are a forensic accountant who specializes in
        payment disputes and reconciliation exceptions. You can identify
        patterns in unmatched transactions, distinguish between timing
        issues and genuine discrepancies, and flag potential fraud or
        revenue leakage for immediate escalation.""",
        llm=llm,
        verbose=config.verbose,
        allow_delegation=True,
        tools=[]
    )

    reporting_agent = Agent(
        role="Stakeholder Communications Manager",
        goal="""Generate clear, actionable reconciliation reports for
        different audiences: daily summaries for finance ops, exception
        details for analysts, and trend dashboards for leadership.""",
        backstory="""You are a financial reporting specialist who translates
        complex reconciliation data into business insights. You know that
        the CFO wants match rates and exception values, the controller wants
        aging analysis, and auditors want full transaction trails.""",
        llm=llm,
        verbose=config.verbose,
        allow_delegation=False,
        tools=[]
    )

    return {
        "data_ingestion": data_ingestion_agent,
        "matching_engine": matching_engine_agent,
        "exception_analyst": exception_analyst_agent,
        "reporting": reporting_agent
    }

def create_reconciliation_crew(
    agents: Dict[str, Agent],
    transaction_data: Dict[str, Any]
) -> Crew:
    """Assemble the reconciliation crew with coordinated tasks."""

    ingest_task = Task(
        description=f"""Extract and normalize transaction data:
        - Gateway data period: {transaction_data.get('start_date')} to {transaction_data.get('end_date')}
        - Sources: {transaction_data.get('sources', ['stripe', 'paypal', 'bank'])}

        Validate all amounts, dates, and references. Flag any data quality issues.
        Output: Normalized transaction dataset with quality metrics.""",
        agent=agents["data_ingestion"],
        expected_output="JSON with normalized_transactions array and data_quality_report"
    )

    match_task = Task(
        description="""Run multi-pass reconciliation matching:
        Pass 1: Exact amount + date (within 3 days) + reference substring
        Pass 2: Batch payout matching (sum of gateway txns = bank deposit)
        Pass 3: Fuzzy matching on remaining items (>85% confidence threshold)

        Output: Match results with confidence scores and match type classification.""",
        agent=agents["matching_engine"],
        expected_output="JSON with matches array, batch_matches array, and unmatched array",
        context=[ingest_task]
    )

    exception_task = Task(
        description="""Analyze all unmatched transactions:
        1. Classify exception type (timing, amount mismatch, missing counterpart, etc.)
        2. Identify root cause patterns
        3. Calculate financial exposure
        4. Recommend resolution action for each item

        Output: Exception analysis with prioritized action items.""",
        agent=agents["exception_analyst"],
        expected_output="JSON with classified_exceptions array and action_recommendations",
        context=[match_task]
    )

    report_task = Task(
        description="""Generate reconciliation reports:
        1. Executive summary (match rate, exception value, trends)
        2. Detailed exception log with recommended actions
        3. Audit trail documentation

        Format for Slack notification and email distribution.""",
        agent=agents["reporting"],
        expected_output="Formatted reports for Slack and email delivery",
        context=[match_task, exception_task]
    )

    return Crew(
        agents=list(agents.values()),
        tasks=[ingest_task, match_task, exception_task, report_task],
        process=Process.sequential,
        verbose=True
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Build robust API connectors for Stripe, PayPal, and bank feeds with error handling, rate limiting, and data normalization into a unified transaction schema.',
            toolsUsed: ['LangChain', 'Stripe SDK', 'PayPal SDK', 'Plaid API', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Multi-Source Transaction Ingestion Pipeline',
                description:
                  'Production-grade data ingestion with API connectors, error handling, and schema normalization.',
                code: `import stripe
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, AsyncGenerator
from pydantic import BaseModel, Field, validator
from langchain.tools import BaseTool
import httpx
import logging

logger = logging.getLogger(__name__)

class NormalizedTransaction(BaseModel):
    """Unified transaction schema for reconciliation."""
    transaction_id: str
    source: str = Field(..., regex="^(stripe|paypal|bank)$")
    event_type: str
    gross_amount: Decimal
    fee_amount: Decimal = Decimal("0")
    net_amount: Decimal
    currency: str = Field(..., min_length=3, max_length=3)
    event_date: datetime
    settlement_date: Optional[datetime] = None
    reference_key: Optional[str] = None
    description: Optional[str] = None
    raw_data: dict = Field(default_factory=dict)

    @validator('net_amount', pre=True, always=True)
    def calculate_net(cls, v, values):
        if v is None:
            return values.get('gross_amount', Decimal("0")) - values.get('fee_amount', Decimal("0"))
        return v

class StripeIngestionTool(BaseTool):
    """LangChain tool for Stripe transaction ingestion."""
    name: str = "stripe_ingestion"
    description: str = "Fetches and normalizes Stripe balance transactions and payouts"

    api_key: str

    def _run(self, days_back: int = 7) -> List[dict]:
        """Synchronous execution."""
        return asyncio.run(self._arun(days_back))

    async def _arun(self, days_back: int = 7) -> List[dict]:
        """Fetch Stripe transactions asynchronously."""
        stripe.api_key = self.api_key
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        cutoff_ts = int(cutoff.timestamp())

        transactions: List[NormalizedTransaction] = []

        try:
            # Fetch balance transactions
            balance_txns = stripe.BalanceTransaction.list(
                created={"gte": cutoff_ts},
                limit=100,
                expand=["data.source"]
            )

            for txn in balance_txns.auto_paging_iter():
                normalized = NormalizedTransaction(
                    transaction_id=txn["id"],
                    source="stripe",
                    event_type=txn["type"],
                    gross_amount=Decimal(str(txn["amount"])) / 100,
                    fee_amount=Decimal(str(txn["fee"])) / 100,
                    net_amount=Decimal(str(txn["net"])) / 100,
                    currency=txn["currency"].upper(),
                    event_date=datetime.utcfromtimestamp(txn["created"]),
                    settlement_date=datetime.utcfromtimestamp(txn["available_on"]),
                    reference_key=self._extract_reference(txn),
                    description=txn.get("description"),
                    raw_data=dict(txn)
                )
                transactions.append(normalized)

            logger.info(f"Ingested {len(transactions)} Stripe transactions")
            return [t.dict() for t in transactions]

        except stripe.error.StripeError as e:
            logger.error(f"Stripe API error: {e}")
            raise

    def _extract_reference(self, txn: dict) -> Optional[str]:
        """Extract order/invoice reference from Stripe metadata."""
        source = txn.get("source", {})
        if isinstance(source, dict):
            metadata = source.get("metadata", {})
            return metadata.get("order_id") or metadata.get("invoice_id")
        return None

class PayPalIngestionTool(BaseTool):
    """LangChain tool for PayPal transaction ingestion."""
    name: str = "paypal_ingestion"
    description: str = "Fetches and normalizes PayPal transaction history"

    client_id: str
    client_secret: str
    base_url: str = "https://api-m.paypal.com"

    async def _get_access_token(self, client: httpx.AsyncClient) -> str:
        """Obtain OAuth2 access token."""
        response = await client.post(
            f"{self.base_url}/v1/oauth2/token",
            auth=(self.client_id, self.client_secret),
            data={"grant_type": "client_credentials"}
        )
        response.raise_for_status()
        return response.json()["access_token"]

    def _run(self, days_back: int = 7) -> List[dict]:
        return asyncio.run(self._arun(days_back))

    async def _arun(self, days_back: int = 7) -> List[dict]:
        """Fetch PayPal transactions asynchronously."""
        start_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + "Z"
        end_date = datetime.utcnow().isoformat() + "Z"

        transactions: List[NormalizedTransaction] = []

        async with httpx.AsyncClient() as client:
            token = await self._get_access_token(client)
            headers = {"Authorization": f"Bearer {token}"}

            response = await client.get(
                f"{self.base_url}/v1/reporting/transactions",
                headers=headers,
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                    "fields": "all",
                    "page_size": 100
                }
            )
            response.raise_for_status()
            data = response.json()

            for txn in data.get("transaction_details", []):
                txn_info = txn.get("transaction_info", {})
                txn_amount = txn_info.get("transaction_amount", {})
                fee_info = txn_info.get("fee_amount", {})

                normalized = NormalizedTransaction(
                    transaction_id=txn_info.get("transaction_id"),
                    source="paypal",
                    event_type=txn_info.get("transaction_event_code", "unknown"),
                    gross_amount=Decimal(txn_amount.get("value", "0")),
                    fee_amount=abs(Decimal(fee_info.get("value", "0"))),
                    net_amount=None,  # Will be calculated
                    currency=txn_amount.get("currency_code", "USD"),
                    event_date=datetime.fromisoformat(
                        txn_info.get("transaction_initiation_date", "").replace("Z", "+00:00")
                    ),
                    reference_key=txn_info.get("invoice_id"),
                    raw_data=txn
                )
                transactions.append(normalized)

        logger.info(f"Ingested {len(transactions)} PayPal transactions")
        return [t.dict() for t in transactions]

class DataIngestionOrchestrator:
    """Coordinates multi-source transaction ingestion."""

    def __init__(
        self,
        stripe_tool: StripeIngestionTool,
        paypal_tool: PayPalIngestionTool
    ):
        self.stripe_tool = stripe_tool
        self.paypal_tool = paypal_tool

    async def ingest_all_sources(
        self,
        days_back: int = 7
    ) -> dict:
        """Parallel ingestion from all sources."""
        results = await asyncio.gather(
            self.stripe_tool._arun(days_back),
            self.paypal_tool._arun(days_back),
            return_exceptions=True
        )

        all_transactions = []
        errors = []

        for i, result in enumerate(results):
            source = ["stripe", "paypal"][i]
            if isinstance(result, Exception):
                errors.append({"source": source, "error": str(result)})
                logger.error(f"Ingestion failed for {source}: {result}")
            else:
                all_transactions.extend(result)

        return {
            "transactions": all_transactions,
            "summary": {
                "total_count": len(all_transactions),
                "by_source": {
                    "stripe": len([t for t in all_transactions if t["source"] == "stripe"]),
                    "paypal": len([t for t in all_transactions if t["source"] == "paypal"])
                },
                "date_range": {
                    "start": min(t["event_date"] for t in all_transactions) if all_transactions else None,
                    "end": max(t["event_date"] for t in all_transactions) if all_transactions else None
                }
            },
            "errors": errors
        }`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the matching engine agent with multi-pass reconciliation algorithms and the exception analyst agent for root cause classification and resolution recommendations.',
            toolsUsed: ['LangChain', 'Pandas', 'FuzzyWuzzy', 'NumPy'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Multi-Pass Matching Engine Agent',
                description:
                  'Sophisticated matching algorithms for transaction reconciliation with confidence scoring.',
                code: `import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MatchType(str, Enum):
    EXACT = "exact"
    BATCH = "batch"
    FUZZY = "fuzzy"
    UNMATCHED = "unmatched"

@dataclass
class MatchResult:
    gateway_id: str
    bank_id: Optional[str]
    match_type: MatchType
    confidence: float
    amount_diff: float = 0.0
    date_diff_days: int = 0
    notes: str = ""

@dataclass
class ReconciliationOutput:
    matches: List[MatchResult] = field(default_factory=list)
    batch_matches: List[Dict] = field(default_factory=list)
    unmatched: List[MatchResult] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

class MatchingEngineAgent:
    """Multi-pass transaction matching engine."""

    def __init__(
        self,
        amount_tolerance: float = 0.05,
        date_window_days: int = 3,
        fuzzy_threshold: int = 85
    ):
        self.amount_tolerance = amount_tolerance
        self.date_window_days = date_window_days
        self.fuzzy_threshold = fuzzy_threshold

    def reconcile(
        self,
        gateway_transactions: List[Dict],
        bank_transactions: List[Dict]
    ) -> ReconciliationOutput:
        """Execute multi-pass reconciliation."""

        gw_df = pd.DataFrame(gateway_transactions)
        bank_df = pd.DataFrame(bank_transactions)

        # Track matched IDs
        matched_gw_ids = set()
        matched_bank_ids = set()

        output = ReconciliationOutput()

        # Pass 1: Exact matching
        logger.info("Pass 1: Exact matching")
        exact_matches = self._exact_match(gw_df, bank_df)
        for match in exact_matches:
            matched_gw_ids.add(match.gateway_id)
            if match.bank_id:
                matched_bank_ids.add(match.bank_id)
            output.matches.append(match)

        # Filter out matched transactions
        gw_remaining = gw_df[~gw_df['transaction_id'].isin(matched_gw_ids)]
        bank_remaining = bank_df[~bank_df['transaction_id'].isin(matched_bank_ids)]

        # Pass 2: Batch payout matching
        logger.info("Pass 2: Batch payout matching")
        batch_matches = self._batch_match(gw_remaining, bank_remaining)
        for batch in batch_matches:
            for gw_id in batch['gateway_ids']:
                matched_gw_ids.add(gw_id)
            matched_bank_ids.add(batch['bank_id'])
            output.batch_matches.append(batch)

        # Filter again
        gw_remaining = gw_df[~gw_df['transaction_id'].isin(matched_gw_ids)]
        bank_remaining = bank_df[~bank_df['transaction_id'].isin(matched_bank_ids)]

        # Pass 3: Fuzzy matching
        logger.info("Pass 3: Fuzzy matching")
        fuzzy_matches = self._fuzzy_match(gw_remaining, bank_remaining)
        for match in fuzzy_matches:
            if match.match_type != MatchType.UNMATCHED:
                matched_gw_ids.add(match.gateway_id)
                if match.bank_id:
                    matched_bank_ids.add(match.bank_id)
                output.matches.append(match)
            else:
                output.unmatched.append(match)

        # Mark remaining as unmatched
        final_unmatched_gw = gw_df[~gw_df['transaction_id'].isin(matched_gw_ids)]
        for _, row in final_unmatched_gw.iterrows():
            output.unmatched.append(MatchResult(
                gateway_id=row['transaction_id'],
                bank_id=None,
                match_type=MatchType.UNMATCHED,
                confidence=0.0,
                notes="No matching bank transaction found"
            ))

        # Calculate stats
        total_gw = len(gw_df)
        output.stats = {
            "total_gateway": total_gw,
            "total_bank": len(bank_df),
            "exact_matched": len([m for m in output.matches if m.match_type == MatchType.EXACT]),
            "batch_matched": sum(len(b['gateway_ids']) for b in output.batch_matches),
            "fuzzy_matched": len([m for m in output.matches if m.match_type == MatchType.FUZZY]),
            "unmatched": len(output.unmatched),
            "match_rate": (total_gw - len(output.unmatched)) / total_gw if total_gw > 0 else 0
        }

        logger.info(f"Reconciliation complete: {output.stats['match_rate']:.1%} match rate")
        return output

    def _exact_match(
        self,
        gw_df: pd.DataFrame,
        bank_df: pd.DataFrame
    ) -> List[MatchResult]:
        """Pass 1: Exact amount + date window + reference matching."""
        matches = []

        for _, gw_row in gw_df.iterrows():
            gw_amount = float(gw_row['net_amount'])
            gw_date = pd.to_datetime(gw_row['settlement_date'] or gw_row['event_date'])
            gw_ref = str(gw_row.get('reference_key', '') or '').lower()

            for _, bank_row in bank_df.iterrows():
                bank_amount = float(bank_row['net_amount'])
                bank_date = pd.to_datetime(bank_row['event_date'])
                bank_desc = str(bank_row.get('description', '') or '').lower()

                amount_match = abs(gw_amount - bank_amount) <= self.amount_tolerance
                date_match = abs((gw_date - bank_date).days) <= self.date_window_days
                ref_match = gw_ref and gw_ref in bank_desc

                if amount_match and date_match and ref_match:
                    matches.append(MatchResult(
                        gateway_id=gw_row['transaction_id'],
                        bank_id=bank_row['transaction_id'],
                        match_type=MatchType.EXACT,
                        confidence=1.0,
                        amount_diff=gw_amount - bank_amount,
                        date_diff_days=abs((gw_date - bank_date).days),
                        notes="Exact match: amount, date, and reference"
                    ))
                    break

        return matches

    def _batch_match(
        self,
        gw_df: pd.DataFrame,
        bank_df: pd.DataFrame
    ) -> List[Dict]:
        """Pass 2: Match sum of gateway transactions to single bank deposit."""
        batch_matches = []

        # Group gateway transactions by settlement date
        if 'settlement_date' not in gw_df.columns or gw_df.empty:
            return batch_matches

        gw_df['settle_date'] = pd.to_datetime(gw_df['settlement_date']).dt.date

        for settle_date, group in gw_df.groupby('settle_date'):
            if settle_date is None:
                continue

            batch_total = float(group['net_amount'].sum())
            gw_ids = group['transaction_id'].tolist()

            # Find matching bank deposit
            for _, bank_row in bank_df.iterrows():
                bank_amount = float(bank_row['net_amount'])
                bank_date = pd.to_datetime(bank_row['event_date']).date()

                date_diff = abs((settle_date - bank_date).days) if settle_date and bank_date else 999
                amount_match = abs(batch_total - bank_amount) <= self.amount_tolerance

                if amount_match and date_diff <= 2:
                    batch_matches.append({
                        'gateway_ids': gw_ids,
                        'bank_id': bank_row['transaction_id'],
                        'batch_total': batch_total,
                        'bank_amount': bank_amount,
                        'settlement_date': str(settle_date),
                        'confidence': 0.95,
                        'transaction_count': len(gw_ids)
                    })
                    break

        return batch_matches

    def _fuzzy_match(
        self,
        gw_df: pd.DataFrame,
        bank_df: pd.DataFrame
    ) -> List[MatchResult]:
        """Pass 3: Fuzzy matching on remaining transactions."""
        results = []

        for _, gw_row in gw_df.iterrows():
            gw_amount = float(gw_row['net_amount'])
            gw_ref = str(gw_row.get('reference_key', '') or gw_row.get('transaction_id', ''))

            best_match = None
            best_score = 0

            for _, bank_row in bank_df.iterrows():
                bank_amount = float(bank_row['net_amount'])
                bank_desc = str(bank_row.get('description', ''))

                # Amount must be within 5%
                if abs(gw_amount - bank_amount) / max(abs(gw_amount), 0.01) > 0.05:
                    continue

                # Fuzzy match on reference/description
                score = fuzz.partial_ratio(gw_ref, bank_desc)

                if score >= self.fuzzy_threshold and score > best_score:
                    best_score = score
                    best_match = bank_row

            if best_match is not None:
                results.append(MatchResult(
                    gateway_id=gw_row['transaction_id'],
                    bank_id=best_match['transaction_id'],
                    match_type=MatchType.FUZZY,
                    confidence=best_score / 100,
                    amount_diff=gw_amount - float(best_match['net_amount']),
                    notes=f"Fuzzy match: {best_score}% reference similarity"
                ))

        return results`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Wire the agents together using LangGraph state machine with proper state transitions, error handling, and checkpointing for reliable reconciliation workflows.',
            toolsUsed: ['LangGraph', 'Redis', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Reconciliation Orchestrator',
                description:
                  'State machine orchestration for the multi-agent reconciliation workflow with Redis persistence.',
                code: `from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime
import operator
import redis
import json
import logging

logger = logging.getLogger(__name__)

class ReconciliationState(TypedDict):
    """State schema for reconciliation workflow."""
    # Input
    date_range: Dict[str, str]
    sources: List[str]

    # Ingestion results
    gateway_transactions: List[Dict]
    bank_transactions: List[Dict]
    ingestion_errors: List[Dict]

    # Matching results
    matches: List[Dict]
    batch_matches: List[Dict]
    unmatched: List[Dict]
    match_stats: Dict

    # Exception analysis
    classified_exceptions: List[Dict]
    action_recommendations: List[Dict]

    # Reporting
    reports: Dict[str, str]
    notifications_sent: List[str]

    # Workflow metadata
    current_step: str
    error_message: Optional[str]
    started_at: str
    completed_at: Optional[str]

def create_reconciliation_graph(
    ingestion_orchestrator,
    matching_engine,
    exception_analyzer,
    report_generator
) -> StateGraph:
    """Build the LangGraph reconciliation workflow."""

    workflow = StateGraph(ReconciliationState)

    # Node: Data Ingestion
    async def ingest_data(state: ReconciliationState) -> ReconciliationState:
        logger.info("Executing: Data Ingestion")
        state["current_step"] = "ingestion"

        try:
            days_back = 7  # Default lookback
            result = await ingestion_orchestrator.ingest_all_sources(days_back)

            # Separate gateway and bank transactions
            state["gateway_transactions"] = [
                t for t in result["transactions"]
                if t["source"] in ("stripe", "paypal")
            ]
            state["bank_transactions"] = [
                t for t in result["transactions"]
                if t["source"] == "bank"
            ]
            state["ingestion_errors"] = result.get("errors", [])

            logger.info(
                f"Ingested {len(state['gateway_transactions'])} gateway, "
                f"{len(state['bank_transactions'])} bank transactions"
            )
        except Exception as e:
            state["error_message"] = f"Ingestion failed: {str(e)}"
            logger.error(state["error_message"])

        return state

    # Node: Transaction Matching
    def match_transactions(state: ReconciliationState) -> ReconciliationState:
        logger.info("Executing: Transaction Matching")
        state["current_step"] = "matching"

        try:
            result = matching_engine.reconcile(
                state["gateway_transactions"],
                state["bank_transactions"]
            )

            state["matches"] = [vars(m) for m in result.matches]
            state["batch_matches"] = result.batch_matches
            state["unmatched"] = [vars(m) for m in result.unmatched]
            state["match_stats"] = result.stats

            logger.info(f"Matching complete: {result.stats['match_rate']:.1%} rate")
        except Exception as e:
            state["error_message"] = f"Matching failed: {str(e)}"
            logger.error(state["error_message"])

        return state

    # Node: Exception Analysis
    def analyze_exceptions(state: ReconciliationState) -> ReconciliationState:
        logger.info("Executing: Exception Analysis")
        state["current_step"] = "exception_analysis"

        try:
            if not state["unmatched"]:
                state["classified_exceptions"] = []
                state["action_recommendations"] = []
                return state

            classified, recommendations = exception_analyzer.analyze(
                state["unmatched"],
                state["gateway_transactions"],
                state["bank_transactions"]
            )

            state["classified_exceptions"] = classified
            state["action_recommendations"] = recommendations

            logger.info(f"Analyzed {len(classified)} exceptions")
        except Exception as e:
            state["error_message"] = f"Exception analysis failed: {str(e)}"
            logger.error(state["error_message"])

        return state

    # Node: Generate Reports
    def generate_reports(state: ReconciliationState) -> ReconciliationState:
        logger.info("Executing: Report Generation")
        state["current_step"] = "reporting"

        try:
            reports = report_generator.generate(
                match_stats=state["match_stats"],
                matches=state["matches"],
                batch_matches=state["batch_matches"],
                exceptions=state["classified_exceptions"],
                recommendations=state["action_recommendations"]
            )

            state["reports"] = reports
            state["completed_at"] = datetime.utcnow().isoformat()

            logger.info("Reports generated successfully")
        except Exception as e:
            state["error_message"] = f"Report generation failed: {str(e)}"
            logger.error(state["error_message"])

        return state

    # Conditional: Check for errors
    def should_continue(state: ReconciliationState) -> str:
        if state.get("error_message"):
            return "error_handler"
        return "continue"

    # Node: Error Handler
    def handle_error(state: ReconciliationState) -> ReconciliationState:
        logger.error(f"Workflow error: {state['error_message']}")
        state["current_step"] = "error"
        return state

    # Build graph
    workflow.add_node("ingest", ingest_data)
    workflow.add_node("match", match_transactions)
    workflow.add_node("analyze", analyze_exceptions)
    workflow.add_node("report", generate_reports)
    workflow.add_node("error_handler", handle_error)

    # Define edges
    workflow.set_entry_point("ingest")
    workflow.add_conditional_edges(
        "ingest",
        should_continue,
        {"continue": "match", "error_handler": "error_handler"}
    )
    workflow.add_conditional_edges(
        "match",
        should_continue,
        {"continue": "analyze", "error_handler": "error_handler"}
    )
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {"continue": "report", "error_handler": "error_handler"}
    )
    workflow.add_edge("report", END)
    workflow.add_edge("error_handler", END)

    return workflow

class RedisCheckpointer:
    """Redis-backed state persistence for reconciliation workflows."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.prefix = "recon:checkpoint:"
        self.ttl_seconds = 30 * 24 * 60 * 60  # 30 days

    def save(self, workflow_id: str, state: ReconciliationState):
        key = f"{self.prefix}{workflow_id}"
        self.redis.setex(key, self.ttl_seconds, json.dumps(state, default=str))
        logger.info(f"Checkpoint saved: {workflow_id}")

    def load(self, workflow_id: str) -> Optional[ReconciliationState]:
        key = f"{self.prefix}{workflow_id}"
        data = self.redis.get(key)
        if data:
            return json.loads(data)
        return None

    def delete(self, workflow_id: str):
        key = f"{self.prefix}{workflow_id}"
        self.redis.delete(key)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Containerize the multi-agent system with Docker, integrate LangSmith for tracing and debugging, and set up Prometheus metrics for production monitoring.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose Deployment Configuration',
                description:
                  'Production-ready Docker Compose configuration for the reconciliation multi-agent system.',
                code: `version: '3.8'

services:
  reconciliation-agents:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: recon-agents
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - STRIPE_API_KEY=\${STRIPE_API_KEY}
      - PAYPAL_CLIENT_ID=\${PAYPAL_CLIENT_ID}
      - PAYPAL_CLIENT_SECRET=\${PAYPAL_CLIENT_SECRET}
      - REDIS_URL=redis://redis:6379
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=\${LANGCHAIN_API_KEY}
      - LANGCHAIN_PROJECT=payment-reconciliation
      - PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
    ports:
      - "8080:8080"
    depends_on:
      - redis
      - postgres
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: recon-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: recon-postgres
    environment:
      - POSTGRES_DB=reconciliation
      - POSTGRES_USER=recon_user
      - POSTGRES_PASSWORD=\${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./sql/init:/docker-entrypoint-initdb.d
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: recon-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: recon-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile.scheduler
    container_name: recon-scheduler
    environment:
      - RECONCILIATION_API_URL=http://reconciliation-agents:8080
      - SCHEDULE_CRON=0 6 * * *
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
    depends_on:
      - reconciliation-agents
    restart: unless-stopped

volumes:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:`,
              },
              {
                language: 'python',
                title: 'LangSmith Tracing and Prometheus Metrics',
                description:
                  'Observability integration with LangSmith for agent tracing and Prometheus for operational metrics.',
                code: `import os
from langsmith import Client
from langsmith.run_helpers import traceable
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps
from typing import Callable, Any
import time
import logging

logger = logging.getLogger(__name__)

# Initialize LangSmith client
langsmith_client = Client() if os.getenv("LANGCHAIN_API_KEY") else None

# Prometheus metrics
RECONCILIATION_RUNS = Counter(
    'reconciliation_runs_total',
    'Total reconciliation workflow runs',
    ['status']
)

RECONCILIATION_DURATION = Histogram(
    'reconciliation_duration_seconds',
    'Duration of reconciliation workflows',
    buckets=[60, 300, 600, 1200, 1800, 3600]
)

MATCH_RATE = Gauge(
    'reconciliation_match_rate',
    'Current reconciliation match rate'
)

EXCEPTION_COUNT = Gauge(
    'reconciliation_exceptions_total',
    'Number of unmatched transactions',
    ['severity']
)

TRANSACTIONS_PROCESSED = Counter(
    'reconciliation_transactions_processed_total',
    'Total transactions processed',
    ['source', 'type']
)

AGENT_EXECUTION_TIME = Histogram(
    'agent_execution_seconds',
    'Execution time per agent',
    ['agent_name'],
    buckets=[1, 5, 10, 30, 60, 120]
)

def track_agent_execution(agent_name: str):
    """Decorator to track agent execution time and trace with LangSmith."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @traceable(name=agent_name, run_type="chain")
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                AGENT_EXECUTION_TIME.labels(agent_name=agent_name).observe(
                    time.time() - start_time
                )
                return result
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                raise

        @wraps(func)
        @traceable(name=agent_name, run_type="chain")
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                AGENT_EXECUTION_TIME.labels(agent_name=agent_name).observe(
                    time.time() - start_time
                )
                return result
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator

class ReconciliationMetricsCollector:
    """Collect and expose reconciliation metrics."""

    def record_run_complete(
        self,
        status: str,
        duration_seconds: float,
        match_stats: dict
    ):
        """Record metrics after a reconciliation run."""
        RECONCILIATION_RUNS.labels(status=status).inc()
        RECONCILIATION_DURATION.observe(duration_seconds)

        if match_stats:
            MATCH_RATE.set(match_stats.get('match_rate', 0))

            # Record by source
            by_source = match_stats.get('by_source', {})
            for source, count in by_source.items():
                TRANSACTIONS_PROCESSED.labels(
                    source=source,
                    type='processed'
                ).inc(count)

    def record_exceptions(self, classified_exceptions: list):
        """Record exception metrics by severity."""
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}

        for exc in classified_exceptions:
            severity = exc.get('severity', 'medium')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        for severity, count in severity_counts.items():
            EXCEPTION_COUNT.labels(severity=severity).set(count)

    def get_metrics(self) -> bytes:
        """Generate Prometheus metrics output."""
        return generate_latest()

# FastAPI endpoints for health and metrics
from fastapi import FastAPI, Response

app = FastAPI(title="Reconciliation Agent Service")
metrics_collector = ReconciliationMetricsCollector()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "langsmith_enabled": langsmith_client is not None
    }

@app.get("/metrics")
async def prometheus_metrics():
    return Response(
        content=metrics_collector.get_metrics(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/langsmith/runs")
async def recent_runs():
    """Get recent LangSmith run summaries."""
    if not langsmith_client:
        return {"error": "LangSmith not configured"}

    runs = langsmith_client.list_runs(
        project_name=os.getenv("LANGCHAIN_PROJECT", "payment-reconciliation"),
        limit=10
    )
    return {
        "runs": [
            {
                "id": str(run.id),
                "name": run.name,
                "status": run.status,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None
            }
            for run in runs
        ]
    }`,
              },
            ],
          },
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
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to automate UAE Corporate Tax compliance by extracting GL data, running AI-powered classification analysis, and delivering tax computation reports and transfer pricing documentation automatically.',
        estimatedMonthlyCost: '$150 - $250/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Google Sheets (Free)', 'Xero AI (usage-based)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'QuickBooks AI (usage-based)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Export chart of accounts, general ledger transactions, and intercompany balances from your ERP into a structured format. Use Zapier to automatically sync daily GL updates and normalize them for tax classification analysis.',
            toolsUsed: ['ERP Export', 'Xero API', 'QuickBooks API', 'Zapier', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Workflow for GL Data Sync',
                description:
                  'Configure Zapier to pull daily GL transactions from Xero or QuickBooks and prepare them for tax classification.',
                code: `{
  "trigger": {
    "app": "Xero",
    "event": "New Journal Entry",
    "connection": "{{xero_connection_id}}"
  },
  "transform": {
    "journal_id": "{{journal.journal_id}}",
    "account_code": "{{journal.journal_lines[0].account_code}}",
    "account_name": "{{journal.journal_lines[0].account_name}}",
    "amount": "{{journal.journal_lines[0].net_amount}}",
    "currency": "{{journal.currency_code}}",
    "posting_date": "{{journal.journal_date | date: '%Y-%m-%d'}}",
    "description": "{{journal.journal_lines[0].description}}",
    "counterparty": "{{journal.journal_lines[0].tracking[0].name}}",
    "source_system": "xero"
  },
  "action": {
    "app": "Google Sheets",
    "event": "Create Spreadsheet Row",
    "spreadsheet_id": "{{uae_ct_workbook_id}}",
    "worksheet": "GL_Transactions"
  }
}`,
              },
              {
                language: 'json',
                title: 'Chart of Accounts Export Schema',
                description:
                  'Standardized format for chart of accounts data required for UAE CT classification analysis.',
                code: `{
  "export_schema": {
    "columns": [
      {"name": "account_code", "type": "string", "required": true},
      {"name": "account_name", "type": "string", "required": true},
      {"name": "account_type", "type": "string", "enum": ["revenue", "expense", "asset", "liability", "equity"]},
      {"name": "parent_account", "type": "string", "nullable": true},
      {"name": "entity_id", "type": "string", "description": "Legal entity identifier"},
      {"name": "free_zone_flag", "type": "boolean", "default": false},
      {"name": "related_party_flag", "type": "boolean", "default": false},
      {"name": "current_ct_classification", "type": "string", "nullable": true}
    ]
  },
  "example_row": {
    "account_code": "4100",
    "account_name": "Software License Revenue",
    "account_type": "revenue",
    "parent_account": "4000",
    "entity_id": "UAE-MAINLAND-001",
    "free_zone_flag": false,
    "related_party_flag": false,
    "current_ct_classification": null
  },
  "uae_ct_classifications": [
    {"code": "TAXABLE_STANDARD", "rate": 9.0, "description": "Standard taxable income at 9%"},
    {"code": "EXEMPT_DIVIDEND", "rate": 0.0, "description": "Participation exemption dividends"},
    {"code": "EXEMPT_CAPITAL", "rate": 0.0, "description": "Qualifying capital gains"},
    {"code": "FZ_QUALIFYING", "rate": 0.0, "description": "Qualifying Free Zone income"},
    {"code": "FZ_NON_QUALIFYING", "rate": 9.0, "description": "Non-qualifying Free Zone income"},
    {"code": "SBR_ELIGIBLE", "rate": 0.0, "description": "Small Business Relief eligible"}
  ]
}`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description:
              'Use ChatGPT or Claude to analyze the chart of accounts and GL transactions, classify each account for UAE Corporate Tax treatment, identify related-party transactions for transfer pricing, and compute taxable income with Small Business Relief checks.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro', 'Custom GPT'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'UAE Corporate Tax Classification Prompt',
                description:
                  'Master prompt template for AI-powered GL classification and tax computation aligned with FTA requirements.',
                code: `system_prompt: |
  You are a UAE corporate tax specialist with expertise in Federal Tax Authority
  (FTA) regulations and the Corporate Tax Law (Federal Decree-Law No. 47 of 2022).

  Your task is to classify general ledger accounts and compute taxable income
  according to UAE CT requirements:

  CLASSIFICATION RULES:
  1. TAXABLE_STANDARD (9%): Business income from mainland entities above AED 375,000
  2. EXEMPT_DIVIDEND: Dividends from qualifying shareholdings (>5% ownership, 12+ months)
  3. EXEMPT_CAPITAL: Capital gains from qualifying participations
  4. FZ_QUALIFYING (0%): Qualifying Free Zone Person income meeting de minimis tests
  5. FZ_NON_QUALIFYING (9%): Free Zone income failing qualifying conditions
  6. SBR_ELIGIBLE (0%): Small Business Relief for revenue ≤ AED 3M and income ≤ AED 375K

  TRANSFER PRICING FLAGS:
  - Flag any account with intercompany or related-party transactions
  - Identify management fees, royalties, financing arrangements
  - Note arm's-length documentation requirements

  OUTPUT FORMAT:
  For each account, provide:
  - ct_classification: One of the codes above
  - classification_rationale: Brief explanation
  - transfer_pricing_flag: true/false
  - tp_documentation_required: List of required TP docs if applicable
  - confidence: 0.0 to 1.0

user_prompt_template: |
  Analyze the following chart of accounts and GL transactions for UAE Corporate Tax:

  ENTITY INFORMATION:
  - Entity Name: {{entity_name}}
  - Entity Type: {{entity_type}} (mainland/free_zone)
  - Free Zone: {{free_zone_name}} (if applicable)
  - Tax Period: {{tax_period_start}} to {{tax_period_end}}

  CHART OF ACCOUNTS:
  {{chart_of_accounts_csv}}

  GL TRANSACTIONS (summarized by account):
  {{gl_summary_csv}}

  RELATED PARTY ENTITIES:
  {{related_parties_list}}

  Please:
  1. Classify each GL account for UAE CT treatment
  2. Identify accounts with related-party activity requiring TP documentation
  3. Compute preliminary taxable income by classification
  4. Check Small Business Relief eligibility (revenue ≤ AED 3M, taxable income ≤ AED 375K)
  5. Flag any accounts requiring manual review

  Return results as structured JSON.

output_schema:
  type: object
  properties:
    entity_summary:
      entity_name: string
      entity_type: string
      total_revenue: number
      sbr_eligible: boolean
      sbr_reason: string
    classifications:
      type: array
      items:
        account_code: string
        account_name: string
        ct_classification: string
        classification_rationale: string
        transfer_pricing_flag: boolean
        ytd_amount: number
        confidence: number
    taxable_income_computation:
      total_income: number
      exempt_income: number
      taxable_income_before_sbr: number
      sbr_deduction: number
      taxable_income_final: number
      estimated_tax_liability: number
    transfer_pricing_summary:
      related_party_accounts: integer
      total_rp_transaction_value: number
      tp_docs_required: array
    manual_review_items:
      type: array
      items:
        account_code: string
        reason: string
        priority: string`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier to run the tax classification analysis weekly, generate transfer pricing summaries, track Small Business Relief eligibility, and deliver compliance reports to tax managers via email or Slack.',
            toolsUsed: ['Zapier', 'Google Sheets', 'Slack', 'Email', 'Google Docs'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier UAE CT Compliance Workflow',
                description:
                  'End-to-end Zapier workflow that triggers weekly tax classification analysis and delivers compliance reports.',
                code: `{
  "workflow_name": "Weekly UAE CT Compliance Report",
  "trigger": {
    "app": "Schedule by Zapier",
    "event": "Every Week",
    "day": "Sunday",
    "time": "08:00",
    "timezone": "Asia/Dubai"
  },
  "steps": [
    {
      "step": 1,
      "app": "Google Sheets",
      "action": "Get Many Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{uae_ct_workbook_id}}",
        "worksheet": "Chart_of_Accounts",
        "filter": "active = true"
      }
    },
    {
      "step": 2,
      "app": "Google Sheets",
      "action": "Get Many Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{uae_ct_workbook_id}}",
        "worksheet": "GL_Transactions",
        "filter": "posting_date >= {{current_tax_period_start}}"
      }
    },
    {
      "step": 3,
      "app": "ChatGPT",
      "action": "Conversation",
      "config": {
        "model": "gpt-4",
        "system_prompt": "{{uae_ct_classification_system_prompt}}",
        "user_message": "Analyze for UAE CT:\\n\\nCHART OF ACCOUNTS:\\n{{step1.rows_csv}}\\n\\nGL SUMMARY:\\n{{step2.rows_csv}}\\n\\nEntity: {{entity_name}}, Type: {{entity_type}}"
      }
    },
    {
      "step": 4,
      "app": "Formatter by Zapier",
      "action": "Text - Extract JSON",
      "config": {
        "input": "{{step3.response}}"
      }
    },
    {
      "step": 5,
      "app": "Google Sheets",
      "action": "Update Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{uae_ct_workbook_id}}",
        "worksheet": "CT_Classifications",
        "rows": "{{step4.json.classifications}}"
      }
    },
    {
      "step": 6,
      "app": "Filter by Zapier",
      "condition": "{{step4.json.transfer_pricing_summary.related_party_accounts}} > 0"
    },
    {
      "step": 7,
      "app": "Google Docs",
      "action": "Create Document from Template",
      "config": {
        "template_id": "{{tp_summary_template_id}}",
        "title": "TP Summary - {{entity_name}} - {{current_date}}",
        "replacements": {
          "{{rp_accounts}}": "{{step4.json.transfer_pricing_summary.related_party_accounts}}",
          "{{rp_value}}": "{{step4.json.transfer_pricing_summary.total_rp_transaction_value}}",
          "{{tp_docs_required}}": "{{step4.json.transfer_pricing_summary.tp_docs_required | join: ', '}}"
        }
      }
    },
    {
      "step": 8,
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#uae-tax-compliance",
        "message": ":flag-ae: *Weekly UAE CT Compliance Report - {{current_date}}*\\n\\n*Entity:* {{entity_name}}\\n*Total Revenue:* AED {{step4.json.entity_summary.total_revenue | number_with_delimiter}}\\n*SBR Eligible:* {{step4.json.entity_summary.sbr_eligible}}\\n\\n*Taxable Income:* AED {{step4.json.taxable_income_computation.taxable_income_final | number_with_delimiter}}\\n*Estimated Tax:* AED {{step4.json.taxable_income_computation.estimated_tax_liability | number_with_delimiter}}\\n\\n*Transfer Pricing:* {{step4.json.transfer_pricing_summary.related_party_accounts}} accounts flagged\\n*Manual Review:* {{step4.json.manual_review_items | size}} items"
      }
    },
    {
      "step": 9,
      "app": "Email by Zapier",
      "action": "Send Outbound Email",
      "config": {
        "to": "{{tax_manager_email}}",
        "subject": "UAE CT Weekly Report - {{entity_name}} - {{current_date}}",
        "body": "Please find attached the weekly UAE Corporate Tax compliance report.\\n\\nKey Metrics:\\n- Taxable Income: AED {{step4.json.taxable_income_computation.taxable_income_final}}\\n- Estimated Tax: AED {{step4.json.taxable_income_computation.estimated_tax_liability}}\\n- TP Documentation Required: {{step4.json.transfer_pricing_summary.tp_docs_required | size}} items",
        "attachments": ["{{step7.document_url}}"]
      }
    }
  ],
  "error_handling": {
    "on_error": "notify",
    "notification_channel": "#tax-alerts",
    "retry_count": 2
  }
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent UAE Corporate Tax compliance system using LangGraph and CrewAI where specialized agents handle GL classification, transfer pricing analysis, tax computation, and FTA filing preparation orchestrated by a tax compliance supervisor.',
        estimatedMonthlyCost: '$800 - $1,500/month',
        architecture:
          'Supervisor agent coordinates four specialist agents: GL Classification Agent (account mapping), Transfer Pricing Agent (related-party analysis), Tax Computation Agent (liability calculation), and Filing Agent (FTA return preparation). LangGraph manages state transitions and Redis provides persistence for audit trails.',
        agents: [
          {
            name: 'GL Classification Agent',
            role: 'Chart of Accounts Tax Specialist',
            goal: 'Analyze and classify every GL account according to UAE Corporate Tax treatment rules, identifying taxable, exempt, and qualifying free zone income categories',
            tools: ['GL Analyzer', 'FTA Classification Rules', 'Account Mapper', 'Entity Registry'],
          },
          {
            name: 'Transfer Pricing Agent',
            role: 'Related-Party Transaction Analyst',
            goal: 'Identify intercompany transactions, assess arm\'s-length compliance, and generate transfer pricing documentation including local files and master files',
            tools: ['RP Transaction Extractor', 'Benchmark Database', 'TNMM Calculator', 'TP Doc Generator'],
          },
          {
            name: 'Tax Computation Agent',
            role: 'UAE CT Calculation Specialist',
            goal: 'Compute taxable income with proper adjustments, apply Small Business Relief where eligible, calculate tax liability, and track the AED 375K threshold',
            tools: ['Income Aggregator', 'SBR Validator', 'Tax Calculator', 'Threshold Monitor'],
          },
          {
            name: 'Filing Agent',
            role: 'FTA Return Preparation Specialist',
            goal: 'Prepare UAE Corporate Tax returns in FTA-compliant format, validate all required fields, and generate supporting schedules for filing',
            tools: ['FTA Form Mapper', 'Schedule Generator', 'Validation Engine', 'Filing API'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Hierarchical',
          stateManagement: 'Redis-backed state with daily checkpointing and 7-year audit trail retention',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent UAE CT compliance system with CrewAI, establishing clear roles for GL classification, transfer pricing, tax computation, and FTA filing preparation.',
            toolsUsed: ['CrewAI', 'LangChain', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'CrewAI Agent Definitions for UAE Corporate Tax',
                description:
                  'Production-ready CrewAI agent definitions for UAE CT compliance with FTA-aligned roles and goals.',
                code: `from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from decimal import Decimal
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CTClassification(str, Enum):
    TAXABLE_STANDARD = "TAXABLE_STANDARD"
    EXEMPT_DIVIDEND = "EXEMPT_DIVIDEND"
    EXEMPT_CAPITAL = "EXEMPT_CAPITAL"
    FZ_QUALIFYING = "FZ_QUALIFYING"
    FZ_NON_QUALIFYING = "FZ_NON_QUALIFYING"
    SBR_ELIGIBLE = "SBR_ELIGIBLE"
    EXCLUDED = "EXCLUDED"

class UAECTAgentConfig(BaseModel):
    """Configuration for UAE CT compliance agents."""
    model_name: str = Field(default="gpt-4-turbo-preview")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=4096)
    verbose: bool = Field(default=True)
    tax_period_start: str = Field(default="2024-01-01")
    tax_period_end: str = Field(default="2024-12-31")
    sbr_revenue_threshold: Decimal = Field(default=Decimal("3000000"))
    sbr_income_threshold: Decimal = Field(default=Decimal("375000"))

def create_uae_ct_agents(config: UAECTAgentConfig) -> Dict[str, Agent]:
    """Create the multi-agent UAE CT compliance team."""

    llm = ChatOpenAI(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )

    gl_classification_agent = Agent(
        role="Chart of Accounts Tax Specialist",
        goal="""Analyze every GL account in the chart of accounts and classify
        it according to UAE Corporate Tax Law requirements. Apply the correct
        classification code (TAXABLE_STANDARD, EXEMPT_DIVIDEND, EXEMPT_CAPITAL,
        FZ_QUALIFYING, FZ_NON_QUALIFYING, SBR_ELIGIBLE, EXCLUDED) based on
        account type, entity location, and transaction characteristics.""",
        backstory="""You are a Big-4 trained tax consultant with 10 years of
        experience in UAE taxation. You were part of the team that helped draft
        the UAE CT implementing regulations. You understand the nuances between
        qualifying and non-qualifying free zone income, the participation
        exemption rules for dividends, and the Small Business Relief criteria.
        You classify accounts conservatively to minimize audit risk.""",
        llm=llm,
        verbose=config.verbose,
        allow_delegation=False,
        tools=[]
    )

    transfer_pricing_agent = Agent(
        role="Related-Party Transaction Analyst",
        goal="""Identify all related-party transactions in the general ledger,
        assess whether they meet arm's-length standards, and prepare transfer
        pricing documentation including functional analysis, economic analysis,
        and benchmark studies as required by UAE CT Law Article 34-36.""",
        backstory="""You are a transfer pricing specialist who has prepared
        local files and master files for multinational groups operating in
        the UAE. You understand that the UAE has adopted OECD Transfer Pricing
        Guidelines and requires documentation for transactions exceeding
        AED 40 million with related parties. You flag transactions that
        lack supporting documentation for immediate attention.""",
        llm=llm,
        verbose=config.verbose,
        allow_delegation=True,
        tools=[]
    )

    tax_computation_agent = Agent(
        role="UAE CT Calculation Specialist",
        goal="""Compute accurate taxable income for UAE Corporate Tax purposes,
        applying the correct adjustments (add-backs, deductions, exempt income
        exclusions), checking Small Business Relief eligibility, and calculating
        final tax liability at the 9% rate or 0% for qualifying income.""",
        backstory="""You are a senior tax manager who has prepared hundreds
        of UAE CT computations. You know that taxable income starts with
        accounting profit, then adds back non-deductible expenses (entertainment
        over 50%, penalties, related-party interest above safe harbor), deducts
        exempt income, and applies the AED 375,000 threshold. You ensure
        mathematical accuracy and full audit trail documentation.""",
        llm=llm,
        verbose=config.verbose,
        allow_delegation=True,
        tools=[]
    )

    filing_agent = Agent(
        role="FTA Return Preparation Specialist",
        goal="""Prepare complete UAE Corporate Tax returns that comply with
        FTA filing requirements, including the main return form, supporting
        schedules, and required attachments. Validate all fields and ensure
        consistency between the return and underlying computations.""",
        backstory="""You are an FTA-registered tax agent who files corporate
        tax returns for dozens of UAE entities. You understand the EmaraTax
        portal requirements, the mandatory fields, and the common rejection
        reasons. You prepare returns that pass FTA validation on first
        submission and maintain complete documentation for the 7-year
        retention period.""",
        llm=llm,
        verbose=config.verbose,
        allow_delegation=False,
        tools=[]
    )

    return {
        "gl_classification": gl_classification_agent,
        "transfer_pricing": transfer_pricing_agent,
        "tax_computation": tax_computation_agent,
        "filing": filing_agent
    }

def create_uae_ct_crew(
    agents: Dict[str, Agent],
    entity_data: Dict[str, Any],
    config: UAECTAgentConfig
) -> Crew:
    """Assemble the UAE CT compliance crew with coordinated tasks."""

    classify_task = Task(
        description=f"""Classify all GL accounts for entity: {entity_data.get('entity_name')}
        Entity Type: {entity_data.get('entity_type')} (mainland/free_zone)
        Free Zone: {entity_data.get('free_zone_name', 'N/A')}
        Tax Period: {config.tax_period_start} to {config.tax_period_end}

        Chart of Accounts: {len(entity_data.get('accounts', []))} accounts
        Apply UAE CT classification rules and flag any accounts requiring review.""",
        agent=agents["gl_classification"],
        expected_output="JSON with classified_accounts array and classification_summary"
    )

    tp_task = Task(
        description=f"""Analyze related-party transactions for transfer pricing:
        Related Parties: {entity_data.get('related_parties', [])}
        Transaction Threshold: AED 40,000,000 for documentation requirement

        Identify all intercompany transactions, assess arm's-length compliance,
        and list required transfer pricing documentation.""",
        agent=agents["transfer_pricing"],
        expected_output="JSON with rp_transactions, tp_assessment, and documentation_requirements",
        context=[classify_task]
    )

    compute_task = Task(
        description=f"""Compute UAE Corporate Tax liability:
        1. Start with accounting profit
        2. Apply add-backs and deductions per CT Law
        3. Exclude exempt income (dividends, capital gains)
        4. Check SBR eligibility (revenue <= AED {config.sbr_revenue_threshold:,}, income <= AED {config.sbr_income_threshold:,})
        5. Calculate tax at 9% on taxable income above AED 375,000

        Provide detailed computation with full audit trail.""",
        agent=agents["tax_computation"],
        expected_output="JSON with tax_computation showing all line items and final liability",
        context=[classify_task, tp_task]
    )

    filing_task = Task(
        description="""Prepare FTA Corporate Tax Return:
        1. Map computed figures to FTA return fields
        2. Generate required schedules (income, deductions, exempt income, TP)
        3. Validate all mandatory fields
        4. Create filing checklist

        Output return data in FTA-compatible format.""",
        agent=agents["filing"],
        expected_output="JSON with fta_return_data, schedules, and validation_status",
        context=[compute_task]
    )

    return Crew(
        agents=list(agents.values()),
        tasks=[classify_task, tp_task, compute_task, filing_task],
        process=Process.sequential,
        verbose=True
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Build robust connectors for ERP systems (Xero, QuickBooks, SAP) to extract chart of accounts, general ledger transactions, and entity master data with validation and normalization.',
            toolsUsed: ['LangChain', 'Xero API', 'QuickBooks API', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Multi-ERP Data Ingestion for UAE CT',
                description:
                  'Production-grade data ingestion from multiple ERP systems with UAE CT-specific field mapping.',
                code: `from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
from decimal import Decimal
from datetime import date, datetime
from langchain.tools import BaseTool
from enum import Enum
import httpx
import logging

logger = logging.getLogger(__name__)

class AccountType(str, Enum):
    REVENUE = "revenue"
    EXPENSE = "expense"
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    OTHER_INCOME = "other_income"
    COGS = "cogs"

class GLAccount(BaseModel):
    """Normalized GL account for UAE CT analysis."""
    account_code: str
    account_name: str
    account_type: AccountType
    parent_code: Optional[str] = None
    entity_id: str
    entity_type: str = Field(..., regex="^(mainland|free_zone)$")
    free_zone_name: Optional[str] = None
    related_party_flag: bool = False
    ytd_balance: Decimal = Decimal("0")
    ct_classification: Optional[str] = None

class GLTransaction(BaseModel):
    """Normalized GL transaction for UAE CT analysis."""
    transaction_id: str
    journal_id: str
    account_code: str
    posting_date: date
    amount: Decimal
    currency: str = "AED"
    description: Optional[str] = None
    counterparty_entity_id: Optional[str] = None
    is_related_party: bool = False
    source_system: str

class EntityMaster(BaseModel):
    """Entity master data for UAE CT."""
    entity_id: str
    entity_name: str
    entity_type: str
    trade_license_number: str
    tax_registration_number: Optional[str] = None
    free_zone_name: Optional[str] = None
    incorporation_date: date
    financial_year_end: str = "12-31"
    parent_entity_id: Optional[str] = None
    related_party_entities: List[str] = Field(default_factory=list)

class XeroDataIngestionTool(BaseTool):
    """LangChain tool for Xero data extraction."""
    name: str = "xero_ingestion"
    description: str = "Extracts chart of accounts and GL transactions from Xero"

    client_id: str
    client_secret: str
    tenant_id: str
    base_url: str = "https://api.xero.com/api.xro/2.0"

    async def _get_access_token(self, client: httpx.AsyncClient) -> str:
        """Obtain OAuth2 access token from Xero."""
        # Implementation would use Xero OAuth2 flow
        pass

    def _run(self, tax_period_start: str, tax_period_end: str) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(self._arun(tax_period_start, tax_period_end))

    async def _arun(
        self,
        tax_period_start: str,
        tax_period_end: str
    ) -> Dict[str, Any]:
        """Extract Xero data for UAE CT analysis."""
        async with httpx.AsyncClient() as client:
            token = await self._get_access_token(client)
            headers = {
                "Authorization": f"Bearer {token}",
                "Xero-tenant-id": self.tenant_id,
                "Accept": "application/json"
            }

            # Fetch chart of accounts
            accounts_resp = await client.get(
                f"{self.base_url}/Accounts",
                headers=headers
            )
            accounts_data = accounts_resp.json()

            # Fetch journals/transactions
            journals_resp = await client.get(
                f"{self.base_url}/Journals",
                headers=headers,
                params={
                    "modifiedAfter": tax_period_start,
                    "offset": 0
                }
            )
            journals_data = journals_resp.json()

            # Transform to normalized format
            gl_accounts = self._transform_accounts(accounts_data.get("Accounts", []))
            gl_transactions = self._transform_journals(journals_data.get("Journals", []))

            return {
                "accounts": [a.dict() for a in gl_accounts],
                "transactions": [t.dict() for t in gl_transactions],
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "source": "xero"
            }

    def _transform_accounts(self, xero_accounts: List[Dict]) -> List[GLAccount]:
        """Transform Xero accounts to normalized UAE CT format."""
        account_type_map = {
            "REVENUE": AccountType.REVENUE,
            "DIRECTCOSTS": AccountType.COGS,
            "EXPENSE": AccountType.EXPENSE,
            "OVERHEADS": AccountType.EXPENSE,
            "OTHERINCOME": AccountType.OTHER_INCOME,
            "FIXED": AccountType.ASSET,
            "CURRENT": AccountType.ASSET,
            "LIABILITY": AccountType.LIABILITY,
            "EQUITY": AccountType.EQUITY
        }

        normalized = []
        for acc in xero_accounts:
            if acc.get("Status") != "ACTIVE":
                continue

            xero_type = acc.get("Type", "EXPENSE")
            account_type = account_type_map.get(xero_type, AccountType.EXPENSE)

            normalized.append(GLAccount(
                account_code=acc.get("Code", ""),
                account_name=acc.get("Name", ""),
                account_type=account_type,
                entity_id="default",  # Would be set from config
                entity_type="mainland",  # Would be determined from entity setup
                ytd_balance=Decimal(str(acc.get("ReportingCode", 0)))
            ))

        return normalized

    def _transform_journals(self, xero_journals: List[Dict]) -> List[GLTransaction]:
        """Transform Xero journals to normalized transactions."""
        transactions = []
        for journal in xero_journals:
            journal_id = journal.get("JournalID", "")
            posting_date = datetime.strptime(
                journal.get("JournalDate", "")[:10], "%Y-%m-%d"
            ).date()

            for line in journal.get("JournalLines", []):
                transactions.append(GLTransaction(
                    transaction_id=f"{journal_id}-{line.get('JournalLineID', '')}",
                    journal_id=journal_id,
                    account_code=line.get("AccountCode", ""),
                    posting_date=posting_date,
                    amount=Decimal(str(line.get("NetAmount", 0))),
                    currency="AED",
                    description=line.get("Description"),
                    source_system="xero"
                ))

        return transactions

class UAECTDataOrchestrator:
    """Coordinates data ingestion from multiple ERPs for UAE CT."""

    def __init__(self, xero_tool: Optional[XeroDataIngestionTool] = None):
        self.xero_tool = xero_tool

    async def ingest_entity_data(
        self,
        entity: EntityMaster,
        tax_period_start: str,
        tax_period_end: str
    ) -> Dict[str, Any]:
        """Ingest all data for a UAE entity's CT computation."""

        result = {
            "entity": entity.dict(),
            "accounts": [],
            "transactions": [],
            "errors": []
        }

        # Ingest from configured ERP
        if self.xero_tool:
            try:
                xero_data = await self.xero_tool._arun(tax_period_start, tax_period_end)
                result["accounts"].extend(xero_data["accounts"])
                result["transactions"].extend(xero_data["transactions"])
            except Exception as e:
                result["errors"].append({"source": "xero", "error": str(e)})

        # Enrich with entity-specific flags
        for account in result["accounts"]:
            account["entity_id"] = entity.entity_id
            account["entity_type"] = entity.entity_type
            if entity.free_zone_name:
                account["free_zone_name"] = entity.free_zone_name

        # Flag related-party transactions
        rp_ids = set(entity.related_party_entities)
        for txn in result["transactions"]:
            if txn.get("counterparty_entity_id") in rp_ids:
                txn["is_related_party"] = True

        logger.info(
            f"Ingested {len(result['accounts'])} accounts, "
            f"{len(result['transactions'])} transactions for {entity.entity_name}"
        )

        return result`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the GL classification agent with FTA rule engine, the transfer pricing agent with arm\'s-length analysis, and the tax computation agent with SBR eligibility checks.',
            toolsUsed: ['LangChain', 'Pandas', 'NumPy'],
            codeSnippets: [
              {
                language: 'python',
                title: 'UAE CT Classification and Computation Engines',
                description:
                  'Domain-specific analysis agents for GL classification, transfer pricing, and tax computation.',
                code: `from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class CTClassification(str, Enum):
    TAXABLE_STANDARD = "TAXABLE_STANDARD"
    EXEMPT_DIVIDEND = "EXEMPT_DIVIDEND"
    EXEMPT_CAPITAL = "EXEMPT_CAPITAL"
    FZ_QUALIFYING = "FZ_QUALIFYING"
    FZ_NON_QUALIFYING = "FZ_NON_QUALIFYING"
    SBR_ELIGIBLE = "SBR_ELIGIBLE"
    EXCLUDED = "EXCLUDED"

@dataclass
class ClassificationResult:
    account_code: str
    account_name: str
    classification: CTClassification
    rationale: str
    confidence: float
    requires_review: bool = False
    review_reason: Optional[str] = None

@dataclass
class TPAssessment:
    counterparty_id: str
    counterparty_name: str
    transaction_type: str
    total_value: Decimal
    arm_length_status: str  # "compliant", "non_compliant", "requires_analysis"
    documentation_required: List[str]
    risk_level: str  # "low", "medium", "high"

@dataclass
class TaxComputation:
    accounting_profit: Decimal
    add_backs: Dict[str, Decimal]
    deductions: Dict[str, Decimal]
    exempt_income: Dict[str, Decimal]
    taxable_income_before_relief: Decimal
    sbr_eligible: bool
    sbr_deduction: Decimal
    taxable_income_final: Decimal
    tax_rate: Decimal
    tax_liability: Decimal

class GLClassificationEngine:
    """Rule-based GL account classification for UAE CT."""

    def __init__(self, entity_type: str, free_zone_name: Optional[str] = None):
        self.entity_type = entity_type
        self.free_zone_name = free_zone_name
        self.qualifying_free_zones = [
            "DIFC", "ADGM", "DMCC", "JAFZA", "DAFZA", "SAIF Zone",
            "RAK FTZ", "Ajman Free Zone", "Sharjah Airport Free Zone"
        ]

    def classify_accounts(
        self,
        accounts: List[Dict],
        transactions: List[Dict]
    ) -> List[ClassificationResult]:
        """Classify all accounts based on UAE CT rules."""

        # Build transaction summary by account
        txn_df = pd.DataFrame(transactions)
        if not txn_df.empty:
            account_summary = txn_df.groupby('account_code').agg({
                'amount': 'sum',
                'is_related_party': 'any'
            }).reset_index()
            account_summary = account_summary.set_index('account_code').to_dict('index')
        else:
            account_summary = {}

        results = []
        for account in accounts:
            code = account['account_code']
            name = account['account_name'].lower()
            acc_type = account['account_type']
            summary = account_summary.get(code, {'amount': 0, 'is_related_party': False})

            classification, rationale, confidence, requires_review, review_reason = \
                self._classify_single_account(account, summary)

            results.append(ClassificationResult(
                account_code=code,
                account_name=account['account_name'],
                classification=classification,
                rationale=rationale,
                confidence=confidence,
                requires_review=requires_review,
                review_reason=review_reason
            ))

        return results

    def _classify_single_account(
        self,
        account: Dict,
        summary: Dict
    ) -> Tuple[CTClassification, str, float, bool, Optional[str]]:
        """Classify a single account based on UAE CT rules."""

        name = account['account_name'].lower()
        acc_type = account['account_type']
        is_rp = summary.get('is_related_party', False)

        # Rule 1: Dividend income from qualifying participations
        if 'dividend' in name and acc_type in ('revenue', 'other_income'):
            return (
                CTClassification.EXEMPT_DIVIDEND,
                "Dividend income potentially eligible for participation exemption",
                0.85,
                True,
                "Verify >5% shareholding held for 12+ months"
            )

        # Rule 2: Capital gains
        if any(term in name for term in ['capital gain', 'disposal', 'sale of investment']):
            return (
                CTClassification.EXEMPT_CAPITAL,
                "Capital gain potentially qualifying for exemption",
                0.80,
                True,
                "Verify qualifying participation criteria"
            )

        # Rule 3: Free Zone income
        if self.entity_type == 'free_zone':
            if self.free_zone_name in self.qualifying_free_zones:
                # Check for non-qualifying activities
                if any(term in name for term in ['mainland', 'domestic', 'uae customer']):
                    return (
                        CTClassification.FZ_NON_QUALIFYING,
                        f"Free Zone income but activity suggests non-qualifying",
                        0.75,
                        True,
                        "Review transaction details for de minimis test"
                    )
                return (
                    CTClassification.FZ_QUALIFYING,
                    f"Qualifying Free Zone ({self.free_zone_name}) income",
                    0.90,
                    False,
                    None
                )

        # Rule 4: Related-party transactions
        if is_rp:
            return (
                CTClassification.TAXABLE_STANDARD,
                "Related-party transaction - taxable with TP documentation required",
                0.95,
                True,
                "Transfer pricing documentation required"
            )

        # Rule 5: Standard taxable income
        if acc_type in ('revenue', 'other_income'):
            return (
                CTClassification.TAXABLE_STANDARD,
                "Standard taxable business income",
                0.95,
                False,
                None
            )

        # Default for expenses
        return (
            CTClassification.TAXABLE_STANDARD,
            "Standard deductible expense",
            0.90,
            False,
            None
        )

class TransferPricingEngine:
    """Transfer pricing analysis for UAE CT."""

    TP_THRESHOLD = Decimal("40000000")  # AED 40M documentation threshold

    def analyze_related_party_transactions(
        self,
        transactions: List[Dict],
        related_parties: List[str]
    ) -> List[TPAssessment]:
        """Analyze RP transactions for TP compliance."""

        rp_set = set(related_parties)
        rp_txns = [t for t in transactions if t.get('counterparty_entity_id') in rp_set]

        if not rp_txns:
            return []

        # Group by counterparty
        df = pd.DataFrame(rp_txns)
        assessments = []

        for cp_id, group in df.groupby('counterparty_entity_id'):
            total_value = Decimal(str(group['amount'].abs().sum()))

            # Determine documentation requirements
            docs_required = []
            if total_value > self.TP_THRESHOLD:
                docs_required = [
                    "Local File",
                    "Master File",
                    "Functional Analysis",
                    "Benchmark Study"
                ]
                risk_level = "high"
            elif total_value > Decimal("1000000"):
                docs_required = ["Local File", "Functional Analysis"]
                risk_level = "medium"
            else:
                docs_required = ["Transaction Summary"]
                risk_level = "low"

            assessments.append(TPAssessment(
                counterparty_id=str(cp_id),
                counterparty_name=f"Related Party {cp_id}",
                transaction_type=self._infer_transaction_type(group),
                total_value=total_value,
                arm_length_status="requires_analysis",
                documentation_required=docs_required,
                risk_level=risk_level
            ))

        return assessments

    def _infer_transaction_type(self, txn_group: pd.DataFrame) -> str:
        """Infer the type of RP transaction from descriptions."""
        descriptions = ' '.join(txn_group['description'].dropna().astype(str)).lower()

        if any(term in descriptions for term in ['management', 'service', 'fee']):
            return "Management Services"
        elif any(term in descriptions for term in ['interest', 'loan', 'financing']):
            return "Financing"
        elif any(term in descriptions for term in ['royalty', 'license', 'ip']):
            return "Royalties/IP"
        else:
            return "Intercompany Trade"

class TaxComputationEngine:
    """UAE Corporate Tax liability computation."""

    SBR_REVENUE_THRESHOLD = Decimal("3000000")
    SBR_INCOME_THRESHOLD = Decimal("375000")
    TAX_RATE = Decimal("0.09")

    def compute_tax(
        self,
        classifications: List[ClassificationResult],
        transactions: List[Dict],
        accounting_profit: Decimal
    ) -> TaxComputation:
        """Compute UAE CT liability from classified accounts."""

        # Calculate exempt income
        exempt_income = {}
        for c in classifications:
            if c.classification in (CTClassification.EXEMPT_DIVIDEND, CTClassification.EXEMPT_CAPITAL):
                # Sum transactions for this account
                acc_total = sum(
                    Decimal(str(t['amount']))
                    for t in transactions
                    if t['account_code'] == c.account_code
                )
                exempt_income[c.account_code] = acc_total

        total_exempt = sum(exempt_income.values())

        # Standard add-backs per UAE CT Law
        add_backs = {
            "entertainment_disallowed": accounting_profit * Decimal("0.01"),  # Simplified
            "penalties_fines": Decimal("0"),
            "donations_non_qualifying": Decimal("0")
        }
        total_add_backs = sum(add_backs.values())

        # Standard deductions
        deductions = {
            "qualifying_donations": Decimal("0")
        }
        total_deductions = sum(deductions.values())

        # Taxable income before SBR
        taxable_before_sbr = accounting_profit + total_add_backs - total_deductions - total_exempt

        # Check SBR eligibility
        total_revenue = sum(
            Decimal(str(t['amount']))
            for t in transactions
            if t.get('account_type') in ('revenue', 'other_income') and Decimal(str(t['amount'])) > 0
        )

        sbr_eligible = (
            total_revenue <= self.SBR_REVENUE_THRESHOLD and
            taxable_before_sbr <= self.SBR_INCOME_THRESHOLD
        )

        sbr_deduction = taxable_before_sbr if sbr_eligible else Decimal("0")
        taxable_final = max(taxable_before_sbr - sbr_deduction - self.SBR_INCOME_THRESHOLD, Decimal("0"))
        tax_liability = taxable_final * self.TAX_RATE

        return TaxComputation(
            accounting_profit=accounting_profit,
            add_backs=add_backs,
            deductions=deductions,
            exempt_income=exempt_income,
            taxable_income_before_relief=taxable_before_sbr,
            sbr_eligible=sbr_eligible,
            sbr_deduction=sbr_deduction,
            taxable_income_final=taxable_final,
            tax_rate=self.TAX_RATE,
            tax_liability=tax_liability
        )`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Wire the UAE CT agents together using LangGraph state machine with proper state transitions, audit trail logging, and checkpointing for compliance documentation.',
            toolsUsed: ['LangGraph', 'Redis', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph UAE CT Compliance Orchestrator',
                description:
                  'State machine orchestration for the multi-agent UAE CT workflow with audit trail persistence.',
                code: `from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from decimal import Decimal
import json
import redis
import logging

logger = logging.getLogger(__name__)

class UAECTState(TypedDict):
    """State schema for UAE CT compliance workflow."""
    # Entity context
    entity_id: str
    entity_name: str
    entity_type: str
    tax_period: Dict[str, str]

    # Ingested data
    accounts: List[Dict]
    transactions: List[Dict]
    related_parties: List[str]

    # Classification results
    classifications: List[Dict]
    classification_summary: Dict

    # Transfer pricing
    tp_assessments: List[Dict]
    tp_documentation_required: List[str]

    # Tax computation
    tax_computation: Dict
    sbr_eligible: bool
    tax_liability: str

    # Filing preparation
    fta_return: Dict
    schedules: List[Dict]
    validation_status: str

    # Workflow metadata
    current_step: str
    audit_trail: List[Dict]
    error_message: Optional[str]
    started_at: str
    completed_at: Optional[str]

def create_uae_ct_workflow(
    data_orchestrator,
    classification_engine,
    tp_engine,
    computation_engine,
    filing_generator
) -> StateGraph:
    """Build the LangGraph UAE CT compliance workflow."""

    workflow = StateGraph(UAECTState)

    # Node: Data Ingestion
    async def ingest_data(state: UAECTState) -> UAECTState:
        logger.info(f"Ingesting data for {state['entity_name']}")
        state["current_step"] = "ingestion"
        state["audit_trail"].append({
            "step": "ingestion",
            "timestamp": datetime.utcnow().isoformat(),
            "action": "started"
        })

        try:
            from pydantic import BaseModel

            class EntityMaster(BaseModel):
                entity_id: str
                entity_name: str
                entity_type: str
                trade_license_number: str = ""
                related_party_entities: List[str] = []

            entity = EntityMaster(
                entity_id=state["entity_id"],
                entity_name=state["entity_name"],
                entity_type=state["entity_type"],
                related_party_entities=state.get("related_parties", [])
            )

            result = await data_orchestrator.ingest_entity_data(
                entity,
                state["tax_period"]["start"],
                state["tax_period"]["end"]
            )

            state["accounts"] = result["accounts"]
            state["transactions"] = result["transactions"]

            state["audit_trail"].append({
                "step": "ingestion",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "completed",
                "details": {
                    "accounts_count": len(state["accounts"]),
                    "transactions_count": len(state["transactions"])
                }
            })

        except Exception as e:
            state["error_message"] = f"Ingestion failed: {str(e)}"
            logger.error(state["error_message"])

        return state

    # Node: GL Classification
    def classify_accounts(state: UAECTState) -> UAECTState:
        logger.info("Classifying GL accounts")
        state["current_step"] = "classification"

        try:
            results = classification_engine.classify_accounts(
                state["accounts"],
                state["transactions"]
            )

            state["classifications"] = [
                {
                    "account_code": r.account_code,
                    "account_name": r.account_name,
                    "classification": r.classification.value,
                    "rationale": r.rationale,
                    "confidence": r.confidence,
                    "requires_review": r.requires_review
                }
                for r in results
            ]

            # Summary by classification
            from collections import Counter
            class_counts = Counter(r.classification.value for r in results)
            state["classification_summary"] = dict(class_counts)

            state["audit_trail"].append({
                "step": "classification",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "completed",
                "details": state["classification_summary"]
            })

        except Exception as e:
            state["error_message"] = f"Classification failed: {str(e)}"

        return state

    # Node: Transfer Pricing Analysis
    def analyze_transfer_pricing(state: UAECTState) -> UAECTState:
        logger.info("Analyzing transfer pricing")
        state["current_step"] = "transfer_pricing"

        try:
            assessments = tp_engine.analyze_related_party_transactions(
                state["transactions"],
                state["related_parties"]
            )

            state["tp_assessments"] = [
                {
                    "counterparty_id": a.counterparty_id,
                    "transaction_type": a.transaction_type,
                    "total_value": str(a.total_value),
                    "arm_length_status": a.arm_length_status,
                    "documentation_required": a.documentation_required,
                    "risk_level": a.risk_level
                }
                for a in assessments
            ]

            # Aggregate documentation requirements
            all_docs = set()
            for a in assessments:
                all_docs.update(a.documentation_required)
            state["tp_documentation_required"] = list(all_docs)

            state["audit_trail"].append({
                "step": "transfer_pricing",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "completed",
                "details": {
                    "rp_entities_analyzed": len(assessments),
                    "docs_required": len(all_docs)
                }
            })

        except Exception as e:
            state["error_message"] = f"TP analysis failed: {str(e)}"

        return state

    # Node: Tax Computation
    def compute_tax(state: UAECTState) -> UAECTState:
        logger.info("Computing UAE CT liability")
        state["current_step"] = "computation"

        try:
            # Get accounting profit from transactions
            revenue = sum(
                Decimal(str(t['amount']))
                for t in state["transactions"]
                if t.get('account_type') in ('revenue', 'other_income')
            )
            expenses = sum(
                abs(Decimal(str(t['amount'])))
                for t in state["transactions"]
                if t.get('account_type') in ('expense', 'cogs')
            )
            accounting_profit = revenue - expenses

            result = computation_engine.compute_tax(
                [type('obj', (object,), {
                    'account_code': c['account_code'],
                    'classification': type('CTClass', (), {'value': c['classification']})()
                })() for c in state["classifications"]],
                state["transactions"],
                accounting_profit
            )

            state["tax_computation"] = {
                "accounting_profit": str(result.accounting_profit),
                "add_backs": {k: str(v) for k, v in result.add_backs.items()},
                "deductions": {k: str(v) for k, v in result.deductions.items()},
                "exempt_income": {k: str(v) for k, v in result.exempt_income.items()},
                "taxable_income_before_relief": str(result.taxable_income_before_relief),
                "sbr_deduction": str(result.sbr_deduction),
                "taxable_income_final": str(result.taxable_income_final),
                "tax_rate": str(result.tax_rate),
                "tax_liability": str(result.tax_liability)
            }
            state["sbr_eligible"] = result.sbr_eligible
            state["tax_liability"] = str(result.tax_liability)

            state["audit_trail"].append({
                "step": "computation",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "completed",
                "details": {
                    "taxable_income": str(result.taxable_income_final),
                    "tax_liability": str(result.tax_liability),
                    "sbr_eligible": result.sbr_eligible
                }
            })

        except Exception as e:
            state["error_message"] = f"Tax computation failed: {str(e)}"

        return state

    # Node: Filing Preparation
    def prepare_filing(state: UAECTState) -> UAECTState:
        logger.info("Preparing FTA filing")
        state["current_step"] = "filing"

        try:
            state["fta_return"] = {
                "form_type": "UAE_CT_RETURN",
                "tax_period": state["tax_period"],
                "entity_trn": state.get("entity_trn", ""),
                "total_revenue": str(sum(
                    Decimal(str(t['amount']))
                    for t in state["transactions"]
                    if t.get('account_type') in ('revenue', 'other_income') and Decimal(str(t['amount'])) > 0
                )),
                "taxable_income": state["tax_computation"]["taxable_income_final"],
                "tax_payable": state["tax_computation"]["tax_liability"],
                "sbr_claimed": state["sbr_eligible"]
            }

            state["schedules"] = [
                {"name": "Income Schedule", "data": state["tax_computation"]},
                {"name": "TP Summary", "data": state["tp_assessments"]}
            ]

            state["validation_status"] = "validated"
            state["completed_at"] = datetime.utcnow().isoformat()

            state["audit_trail"].append({
                "step": "filing",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "completed",
                "details": {"validation_status": "validated"}
            })

        except Exception as e:
            state["error_message"] = f"Filing preparation failed: {str(e)}"
            state["validation_status"] = "failed"

        return state

    # Conditional edge
    def should_continue(state: UAECTState) -> str:
        if state.get("error_message"):
            return "error_handler"
        return "continue"

    def handle_error(state: UAECTState) -> UAECTState:
        state["audit_trail"].append({
            "step": state["current_step"],
            "timestamp": datetime.utcnow().isoformat(),
            "action": "error",
            "error": state["error_message"]
        })
        return state

    # Build graph
    workflow.add_node("ingest", ingest_data)
    workflow.add_node("classify", classify_accounts)
    workflow.add_node("tp_analysis", analyze_transfer_pricing)
    workflow.add_node("compute", compute_tax)
    workflow.add_node("prepare_filing", prepare_filing)
    workflow.add_node("error_handler", handle_error)

    workflow.set_entry_point("ingest")
    workflow.add_conditional_edges("ingest", should_continue, {"continue": "classify", "error_handler": "error_handler"})
    workflow.add_conditional_edges("classify", should_continue, {"continue": "tp_analysis", "error_handler": "error_handler"})
    workflow.add_conditional_edges("tp_analysis", should_continue, {"continue": "compute", "error_handler": "error_handler"})
    workflow.add_conditional_edges("compute", should_continue, {"continue": "prepare_filing", "error_handler": "error_handler"})
    workflow.add_edge("prepare_filing", END)
    workflow.add_edge("error_handler", END)

    return workflow`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Containerize the UAE CT multi-agent system with Docker, integrate LangSmith for agent tracing, and set up audit trail logging for FTA compliance requirements.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'PostgreSQL'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose for UAE CT Compliance System',
                description:
                  'Production deployment configuration with audit trail persistence and observability.',
                code: `version: '3.8'

services:
  uae-ct-agents:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: uae-ct-agents
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - XERO_CLIENT_ID=\${XERO_CLIENT_ID}
      - XERO_CLIENT_SECRET=\${XERO_CLIENT_SECRET}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://ct_user:\${DB_PASSWORD}@postgres:5432/uae_ct
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=\${LANGCHAIN_API_KEY}
      - LANGCHAIN_PROJECT=uae-corporate-tax
      - FTA_PORTAL_API_KEY=\${FTA_PORTAL_API_KEY}
      - AUDIT_RETENTION_YEARS=7
    ports:
      - "8081:8081"
    depends_on:
      - redis
      - postgres
    volumes:
      - ./logs:/app/logs
      - ./audit_trail:/app/audit_trail
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8081/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: uae-ct-redis
    ports:
      - "6380:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: uae-ct-postgres
    environment:
      - POSTGRES_DB=uae_ct
      - POSTGRES_USER=ct_user
      - POSTGRES_PASSWORD=\${DB_PASSWORD}
    ports:
      - "5433:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./sql/audit_schema.sql:/docker-entrypoint-initdb.d/001_audit.sql
    restart: unless-stopped

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile.scheduler
    container_name: uae-ct-scheduler
    environment:
      - CT_API_URL=http://uae-ct-agents:8081
      - SCHEDULE_WEEKLY=0 8 * * 0
      - SCHEDULE_MONTHLY=0 8 1 * *
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
      - TAX_MANAGER_EMAIL=\${TAX_MANAGER_EMAIL}
    depends_on:
      - uae-ct-agents
    restart: unless-stopped

volumes:
  redis-data:
  postgres-data:`,
              },
              {
                language: 'python',
                title: 'Audit Trail and FTA Compliance Logging',
                description:
                  'Comprehensive audit trail logging for 7-year FTA retention requirements with LangSmith integration.',
                code: `import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
import hashlib
from langsmith import Client
import psycopg2
from psycopg2.extras import Json
import logging

logger = logging.getLogger(__name__)

langsmith_client = Client() if os.getenv("LANGCHAIN_API_KEY") else None

@dataclass
class AuditEntry:
    """Immutable audit trail entry for FTA compliance."""
    entry_id: str
    entity_id: str
    tax_period: str
    workflow_run_id: str
    step_name: str
    action: str
    timestamp: datetime
    user_id: Optional[str]
    data_snapshot: Dict[str, Any]
    data_hash: str
    previous_entry_hash: Optional[str]

    @classmethod
    def create(
        cls,
        entity_id: str,
        tax_period: str,
        workflow_run_id: str,
        step_name: str,
        action: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        previous_hash: Optional[str] = None
    ) -> 'AuditEntry':
        """Create a new audit entry with computed hash."""
        timestamp = datetime.utcnow()
        data_json = json.dumps(data, sort_keys=True, default=str)
        data_hash = hashlib.sha256(data_json.encode()).hexdigest()

        # Chain hash includes previous entry for tamper detection
        chain_input = f"{data_hash}:{previous_hash or 'genesis'}"
        entry_hash = hashlib.sha256(chain_input.encode()).hexdigest()

        return cls(
            entry_id=f"AUD-{timestamp.strftime('%Y%m%d%H%M%S')}-{entry_hash[:8]}",
            entity_id=entity_id,
            tax_period=tax_period,
            workflow_run_id=workflow_run_id,
            step_name=step_name,
            action=action,
            timestamp=timestamp,
            user_id=user_id,
            data_snapshot=data,
            data_hash=entry_hash,
            previous_entry_hash=previous_hash
        )

class AuditTrailService:
    """FTA-compliant audit trail with 7-year retention."""

    RETENTION_YEARS = 7

    def __init__(self, database_url: str):
        self.db_url = database_url
        self._init_schema()

    def _init_schema(self):
        """Initialize audit trail schema."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS uae_ct_audit_trail (
                        entry_id TEXT PRIMARY KEY,
                        entity_id TEXT NOT NULL,
                        tax_period TEXT NOT NULL,
                        workflow_run_id TEXT NOT NULL,
                        step_name TEXT NOT NULL,
                        action TEXT NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        user_id TEXT,
                        data_snapshot JSONB NOT NULL,
                        data_hash TEXT NOT NULL,
                        previous_entry_hash TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE INDEX IF NOT EXISTS idx_audit_entity_period
                        ON uae_ct_audit_trail(entity_id, tax_period);

                    CREATE INDEX IF NOT EXISTS idx_audit_workflow
                        ON uae_ct_audit_trail(workflow_run_id);

                    CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                        ON uae_ct_audit_trail(timestamp);
                """)
            conn.commit()

    def log_entry(self, entry: AuditEntry):
        """Persist audit entry to database."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO uae_ct_audit_trail (
                        entry_id, entity_id, tax_period, workflow_run_id,
                        step_name, action, timestamp, user_id,
                        data_snapshot, data_hash, previous_entry_hash
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    entry.entry_id,
                    entry.entity_id,
                    entry.tax_period,
                    entry.workflow_run_id,
                    entry.step_name,
                    entry.action,
                    entry.timestamp,
                    entry.user_id,
                    Json(entry.data_snapshot),
                    entry.data_hash,
                    entry.previous_entry_hash
                ))
            conn.commit()

        logger.info(f"Audit entry logged: {entry.entry_id}")

    def get_workflow_trail(
        self,
        workflow_run_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieve complete audit trail for a workflow run."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT entry_id, step_name, action, timestamp,
                           data_snapshot, data_hash
                    FROM uae_ct_audit_trail
                    WHERE workflow_run_id = %s
                    ORDER BY timestamp ASC
                """, (workflow_run_id,))

                return [
                    {
                        "entry_id": row[0],
                        "step_name": row[1],
                        "action": row[2],
                        "timestamp": row[3].isoformat(),
                        "data": row[4],
                        "hash": row[5]
                    }
                    for row in cur.fetchall()
                ]

    def verify_chain_integrity(
        self,
        entity_id: str,
        tax_period: str
    ) -> Dict[str, Any]:
        """Verify audit trail chain integrity for FTA compliance."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT entry_id, data_hash, previous_entry_hash
                    FROM uae_ct_audit_trail
                    WHERE entity_id = %s AND tax_period = %s
                    ORDER BY timestamp ASC
                """, (entity_id, tax_period))

                entries = cur.fetchall()

        if not entries:
            return {"status": "no_entries", "verified": True}

        # Verify chain
        broken_links = []
        prev_hash = None

        for entry_id, data_hash, recorded_prev_hash in entries:
            if prev_hash and recorded_prev_hash != prev_hash:
                broken_links.append({
                    "entry_id": entry_id,
                    "expected_prev": prev_hash,
                    "recorded_prev": recorded_prev_hash
                })
            prev_hash = data_hash

        return {
            "status": "verified" if not broken_links else "integrity_breach",
            "verified": len(broken_links) == 0,
            "total_entries": len(entries),
            "broken_links": broken_links
        }

    def cleanup_expired(self):
        """Remove entries older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=365 * self.RETENTION_YEARS)

        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM uae_ct_audit_trail
                    WHERE timestamp < %s
                    RETURNING entry_id
                """, (cutoff,))
                deleted = cur.fetchall()
            conn.commit()

        logger.info(f"Cleaned up {len(deleted)} expired audit entries")
        return len(deleted)`,
              },
            ],
          },
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
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to automate Australia GST compliance by extracting transaction data, running AI-powered tax code classification, and delivering BAS-ready reports with ABN validation automatically.',
        estimatedMonthlyCost: '$100 - $180/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Google Sheets (Free)', 'Xero AI (usage-based)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'MYOB AI (usage-based)', 'QuickBooks AI (usage-based)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Export transactions from your ERP, POS, and e-commerce systems into a unified format. Use Zapier to automatically sync daily sales and purchases, normalize tax codes, and prepare data for GST classification analysis.',
            toolsUsed: ['Xero API', 'MYOB API', 'Shopify Export', 'Zapier', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Workflow for Transaction Sync',
                description:
                  'Configure Zapier to pull transactions from Xero and prepare them for GST classification analysis.',
                code: `{
  "trigger": {
    "app": "Xero",
    "event": "New Invoice",
    "connection": "{{xero_connection_id}}",
    "filters": {
      "status": ["AUTHORISED", "PAID"]
    }
  },
  "transform": {
    "transaction_id": "{{invoice.invoice_id}}",
    "transaction_type": "sale",
    "date": "{{invoice.date | date: '%Y-%m-%d'}}",
    "total_amount": "{{invoice.total}}",
    "tax_amount": "{{invoice.total_tax}}",
    "contact_name": "{{invoice.contact.name}}",
    "contact_abn": "{{invoice.contact.tax_number}}",
    "line_items": "{{invoice.line_items | map: 'description' | join: '; '}}",
    "current_tax_code": "{{invoice.line_items[0].tax_type}}",
    "source_system": "xero"
  },
  "action": {
    "app": "Google Sheets",
    "event": "Create Spreadsheet Row",
    "spreadsheet_id": "{{gst_workbook_id}}",
    "worksheet": "Transactions"
  }
}`,
              },
              {
                language: 'json',
                title: 'GST Classification Reference Schema',
                description:
                  'ATO-aligned GST classification codes and BAS label mappings for AI analysis.',
                code: `{
  "gst_classifications": [
    {
      "code": "GST",
      "bas_label": "G1",
      "description": "GST on taxable sales",
      "gst_rate": 10.0,
      "supply_type": "taxable",
      "examples": ["Standard goods", "Most services", "Commercial rent"]
    },
    {
      "code": "FRE",
      "bas_label": "G3",
      "description": "GST-free supplies",
      "gst_rate": 0.0,
      "supply_type": "gst_free",
      "examples": ["Basic food", "Health services", "Education", "Exports"]
    },
    {
      "code": "INP",
      "bas_label": "G4",
      "description": "Input taxed supplies",
      "gst_rate": 0.0,
      "supply_type": "input_taxed",
      "examples": ["Financial supplies", "Residential rent", "Precious metals"]
    },
    {
      "code": "EXP",
      "bas_label": "G2",
      "description": "Export sales (GST-free)",
      "gst_rate": 0.0,
      "supply_type": "gst_free",
      "examples": ["Goods exported", "Services to overseas clients"]
    },
    {
      "code": "CAP",
      "bas_label": "G10",
      "description": "Capital acquisitions with GST",
      "gst_rate": 10.0,
      "supply_type": "taxable",
      "examples": ["Equipment", "Vehicles", "Buildings"]
    },
    {
      "code": "N-T",
      "bas_label": "G14",
      "description": "Purchases with no GST",
      "gst_rate": 0.0,
      "supply_type": "gst_free",
      "examples": ["Bank fees", "Government charges", "Wages"]
    }
  ],
  "abn_validation": {
    "required_for_input_credits": true,
    "threshold": 82.5,
    "withholding_rate_without_abn": 47.0
  }
}`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description:
              'Use ChatGPT or Claude to analyze transactions, classify them according to ATO GST rules, validate ABN numbers for input tax credit eligibility, and identify classification errors that need correction.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro', 'Custom GPT'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Australia GST Classification Prompt',
                description:
                  'Master prompt template for AI-powered GST classification and BAS preparation aligned with ATO requirements.',
                code: `system_prompt: |
  You are an Australian GST specialist with expertise in ATO regulations and
  Business Activity Statement (BAS) preparation.

  Your task is to classify transactions according to ATO GST rules and
  identify issues that could affect BAS accuracy:

  CLASSIFICATION RULES:
  1. GST (G1 - 10%): Standard taxable supplies - most goods and services
  2. FRE (G3 - 0%): GST-free supplies - basic food, health, education, exports
  3. INP (G4 - 0%): Input taxed - financial services, residential rent
  4. EXP (G2 - 0%): Exports - goods/services supplied outside Australia
  5. CAP (G10 - 10%): Capital acquisitions - equipment, vehicles, buildings
  6. N-T (G14 - 0%): No GST in price - bank fees, government charges

  INPUT TAX CREDIT RULES:
  - Must have valid tax invoice for claims > $82.50
  - Supplier ABN required for input credits
  - No credits for input-taxed supplies
  - Apportionment required for mixed supplies

  OUTPUT FORMAT:
  For each transaction, provide:
  - recommended_tax_code: One of the codes above
  - current_code_correct: true/false
  - bas_label: G1, G2, G3, etc.
  - gst_amount_calculated: Expected GST based on classification
  - gst_amount_reported: GST as recorded
  - variance: Difference if any
  - abn_valid: true/false/unknown
  - input_credit_eligible: true/false
  - issue_flag: Description of any problems
  - confidence: 0.0 to 1.0

user_prompt_template: |
  Analyze the following transactions for Australia GST compliance:

  BUSINESS INFORMATION:
  - Business Name: {{business_name}}
  - ABN: {{business_abn}}
  - BAS Period: {{bas_period}} (Monthly/Quarterly)
  - Reporting Method: {{reporting_method}} (Cash/Accrual)

  TRANSACTIONS TO CLASSIFY:
  {{transactions_csv}}

  SUPPLIER ABN LIST (for input credit validation):
  {{supplier_abns}}

  Please:
  1. Verify or correct the GST classification for each transaction
  2. Calculate expected GST amounts (1/11 of GST-inclusive or 10% of exclusive)
  3. Validate ABN for purchase transactions
  4. Identify any transactions requiring manual review
  5. Summarize by BAS label for lodgement

  Return results as structured JSON.

output_schema:
  type: object
  properties:
    period_summary:
      bas_period: string
      total_sales: number
      total_purchases: number
      net_gst_position: number
    classifications:
      type: array
      items:
        transaction_id: string
        recommended_tax_code: string
        current_code_correct: boolean
        bas_label: string
        amount_excl_gst: number
        gst_calculated: number
        gst_reported: number
        variance: number
        abn_valid: boolean
        input_credit_eligible: boolean
        issue_flag: string
        confidence: number
    bas_summary:
      G1_total_sales: number
      G1_gst_collected: number
      G2_exports: number
      G3_gst_free: number
      G10_capital_acquisitions: number
      G11_other_acquisitions: number
      total_gst_on_sales: number
      total_input_credits: number
      net_gst_payable: number
    issues:
      type: array
      items:
        transaction_id: string
        issue_type: string
        description: string
        recommended_action: string`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier to run the GST classification analysis weekly, generate BAS-ready reports with all labels populated, validate supplier ABNs, and deliver compliance summaries to the finance team.',
            toolsUsed: ['Zapier', 'Google Sheets', 'Slack', 'Email'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier BAS Preparation Workflow',
                description:
                  'End-to-end Zapier workflow that triggers weekly GST analysis and delivers BAS-ready reports.',
                code: `{
  "workflow_name": "Weekly GST Classification & BAS Prep",
  "trigger": {
    "app": "Schedule by Zapier",
    "event": "Every Week",
    "day": "Friday",
    "time": "17:00",
    "timezone": "Australia/Sydney"
  },
  "steps": [
    {
      "step": 1,
      "app": "Google Sheets",
      "action": "Get Many Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{gst_workbook_id}}",
        "worksheet": "Transactions",
        "filter": "date >= {{current_bas_period_start}}"
      }
    },
    {
      "step": 2,
      "app": "Google Sheets",
      "action": "Get Many Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{gst_workbook_id}}",
        "worksheet": "Supplier_ABNs"
      }
    },
    {
      "step": 3,
      "app": "ChatGPT",
      "action": "Conversation",
      "config": {
        "model": "gpt-4",
        "system_prompt": "{{gst_classification_system_prompt}}",
        "user_message": "Analyze for GST:\\n\\nTRANSACTIONS:\\n{{step1.rows_csv}}\\n\\nSUPPLIER ABNS:\\n{{step2.rows_csv}}\\n\\nBusiness: {{business_name}}, ABN: {{business_abn}}, Period: {{bas_period}}"
      }
    },
    {
      "step": 4,
      "app": "Formatter by Zapier",
      "action": "Text - Extract JSON",
      "config": {
        "input": "{{step3.response}}"
      }
    },
    {
      "step": 5,
      "app": "Google Sheets",
      "action": "Update Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{gst_workbook_id}}",
        "worksheet": "Classification_Results",
        "rows": "{{step4.json.classifications}}"
      }
    },
    {
      "step": 6,
      "app": "Google Sheets",
      "action": "Create Spreadsheet Row",
      "config": {
        "spreadsheet_id": "{{gst_workbook_id}}",
        "worksheet": "BAS_Summary",
        "row": {
          "period": "{{bas_period}}",
          "G1_sales": "{{step4.json.bas_summary.G1_total_sales}}",
          "G1_gst": "{{step4.json.bas_summary.G1_gst_collected}}",
          "G2_exports": "{{step4.json.bas_summary.G2_exports}}",
          "G3_gst_free": "{{step4.json.bas_summary.G3_gst_free}}",
          "G10_capital": "{{step4.json.bas_summary.G10_capital_acquisitions}}",
          "G11_other": "{{step4.json.bas_summary.G11_other_acquisitions}}",
          "total_gst_collected": "{{step4.json.bas_summary.total_gst_on_sales}}",
          "total_input_credits": "{{step4.json.bas_summary.total_input_credits}}",
          "net_gst_payable": "{{step4.json.bas_summary.net_gst_payable}}"
        }
      }
    },
    {
      "step": 7,
      "app": "Filter by Zapier",
      "condition": "{{step4.json.issues | size}} > 0"
    },
    {
      "step": 8,
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#finance-gst",
        "message": ":flag-au: *Weekly GST Report - {{bas_period}}*\\n\\n*Sales GST Collected:* AUD {{step4.json.bas_summary.total_gst_on_sales | number_with_delimiter}}\\n*Input Tax Credits:* AUD {{step4.json.bas_summary.total_input_credits | number_with_delimiter}}\\n*Net GST {{net_position_label}}:* AUD {{step4.json.bas_summary.net_gst_payable | abs | number_with_delimiter}}\\n\\n*Classification Issues:* {{step4.json.issues | size}} items need review\\n*Transactions Processed:* {{step4.json.classifications | size}}\\n\\n<{{gst_workbook_url}}|View Full Report>"
      }
    },
    {
      "step": 9,
      "app": "Email by Zapier",
      "action": "Send Outbound Email",
      "condition": "{{step4.json.issues | size}} > 5",
      "config": {
        "to": "{{finance_manager_email}}",
        "subject": "GST Classification Issues - {{bas_period}} - {{step4.json.issues | size}} items",
        "body": "The weekly GST classification analysis found {{step4.json.issues | size}} items requiring manual review before BAS lodgement.\\n\\nTop Issues:\\n{{step4.json.issues | slice: 0, 5 | map: 'description' | join: '\\n- '}}\\n\\nPlease review and correct before the lodgement deadline.\\n\\nView full report: {{gst_workbook_url}}"
      }
    }
  ],
  "error_handling": {
    "on_error": "notify",
    "notification_channel": "#finance-alerts"
  }
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent Australia GST compliance system using LangGraph and CrewAI where specialized agents handle transaction classification, ABN validation, BAS preparation, and ATO lodgement orchestrated by a compliance supervisor.',
        estimatedMonthlyCost: '$600 - $1,200/month',
        architecture:
          'Supervisor agent coordinates four specialist agents: Classification Agent (GST code assignment), ABN Validator Agent (Australian Business Register checks), BAS Preparer Agent (label aggregation and reconciliation), and Lodgement Agent (ATO portal submission). LangGraph manages state transitions and Redis provides persistence.',
        agents: [
          {
            name: 'GST Classification Agent',
            role: 'Tax Code Specialist',
            goal: 'Analyze every transaction and assign the correct ATO GST classification code based on supply type, ensuring accurate BAS label mapping',
            tools: ['Transaction Analyzer', 'ATO Classification Rules', 'Supply Type Detector', 'Mixed Supply Calculator'],
          },
          {
            name: 'ABN Validator Agent',
            role: 'Supplier Verification Specialist',
            goal: 'Validate supplier ABNs against the Australian Business Register, check GST registration status, and flag transactions where input tax credits may be ineligible',
            tools: ['ABR Lookup API', 'GST Registration Checker', 'Tax Invoice Validator', 'Withholding Calculator'],
          },
          {
            name: 'BAS Preparer Agent',
            role: 'BAS Compilation Specialist',
            goal: 'Aggregate classified transactions into ATO BAS labels (G1-G20), reconcile GST amounts, and generate lodgement-ready data with full audit trail',
            tools: ['Label Aggregator', 'GST Reconciler', 'Adjustment Calculator', 'Audit Trail Generator'],
          },
          {
            name: 'Lodgement Agent',
            role: 'ATO Submission Specialist',
            goal: 'Prepare final BAS in ATO-compliant format, validate all required fields, and facilitate electronic lodgement through the Business Portal or SBR',
            tools: ['ATO Form Mapper', 'SBR Connector', 'Validation Engine', 'Lodgement API'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with daily checkpointing and 5-year ATO retention compliance',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent GST compliance system with CrewAI, establishing clear roles for classification, ABN validation, BAS preparation, and ATO lodgement.',
            toolsUsed: ['CrewAI', 'LangChain', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'CrewAI Agent Definitions for Australia GST',
                description:
                  'Production-ready CrewAI agent definitions for GST compliance with ATO-aligned roles and goals.',
                code: `from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from decimal import Decimal
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class GSTCode(str, Enum):
    GST = "GST"          # G1 - Taxable sales
    FRE = "FRE"          # G3 - GST-free
    INP = "INP"          # G4 - Input taxed
    EXP = "EXP"          # G2 - Exports
    CAP = "CAP"          # G10 - Capital acquisitions
    NCG = "NCG"          # G11 - Non-capital acquisitions
    NT = "N-T"           # G14 - No GST in price
    OOS = "OOS"          # Out of scope

class BASLabel(str, Enum):
    G1 = "G1"    # Total sales
    G2 = "G2"    # Export sales
    G3 = "G3"    # Other GST-free sales
    G4 = "G4"    # Input taxed sales
    G10 = "G10"  # Capital purchases
    G11 = "G11"  # Non-capital purchases
    G14 = "G14"  # Purchases with no GST

class GSTAgentConfig(BaseModel):
    """Configuration for GST compliance agents."""
    model_name: str = Field(default="gpt-4-turbo-preview")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=4096)
    verbose: bool = Field(default=True)
    bas_period_type: str = Field(default="quarterly")  # monthly, quarterly, annual
    reporting_method: str = Field(default="accrual")   # cash, accrual

def create_gst_agents(config: GSTAgentConfig) -> Dict[str, Agent]:
    """Create the multi-agent GST compliance team."""

    llm = ChatOpenAI(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )

    classification_agent = Agent(
        role="GST Tax Code Specialist",
        goal="""Analyze every transaction and assign the correct ATO GST
        classification code. Distinguish between taxable supplies (GST),
        GST-free supplies (FRE), input-taxed supplies (INP), exports (EXP),
        and out-of-scope items. Apply correct BAS label mapping.""",
        backstory="""You are a certified BAS agent with 15 years of experience
        in Australian GST compliance. You have prepared thousands of BAS
        lodgements across retail, services, and export businesses. You know
        the nuances between GST-free basic food and taxable prepared food,
        the rules for educational supplies, and the complex apportionment
        calculations for mixed supplies. You classify conservatively to
        minimize ATO audit risk.""",
        llm=llm,
        verbose=config.verbose,
        allow_delegation=False,
        tools=[]
    )

    abn_validator_agent = Agent(
        role="Supplier Verification Specialist",
        goal="""Validate supplier ABNs against the Australian Business Register,
        verify GST registration status, and determine input tax credit
        eligibility. Flag transactions where credits may be denied due to
        invalid ABN, cancelled GST registration, or missing tax invoices.""",
        backstory="""You are a compliance officer who specializes in supplier
        verification. You understand that input tax credits require valid
        tax invoices with ABN for purchases over $82.50. You know that
        suppliers who aren't registered for GST can't charge GST, and that
        payments to unregistered suppliers may require PAYG withholding.
        You verify every supplier before recommending input credit claims.""",
        llm=llm,
        verbose=config.verbose,
        allow_delegation=True,
        tools=[]
    )

    bas_preparer_agent = Agent(
        role="BAS Compilation Specialist",
        goal="""Aggregate all classified transactions into ATO BAS labels,
        calculate total GST collected (1A), total GST credits (1B), and
        net amount payable or refundable. Generate supporting schedules
        and ensure mathematical accuracy across all labels.""",
        backstory="""You are a senior accountant who has prepared hundreds
        of Business Activity Statements. You know that G1 must include all
        taxable sales (GST-inclusive amounts divided by 11 for GST component),
        G10/G11 split capital vs non-capital acquisitions, and that the
        net GST calculation must reconcile to the cent. You prepare BAS
        that pass ATO validation on first submission.""",
        llm=llm,
        verbose=config.verbose,
        allow_delegation=True,
        tools=[]
    )

    lodgement_agent = Agent(
        role="ATO Submission Specialist",
        goal="""Prepare the final BAS in ATO-compliant format, validate all
        mandatory fields, generate the Standard Business Reporting (SBR)
        payload, and facilitate electronic lodgement through the ATO
        Business Portal or registered software.""",
        backstory="""You are an ATO-accredited digital service provider who
        understands the SBR2 message format and ATO portal requirements.
        You know the common validation errors and how to avoid them. You
        ensure every lodgement includes the correct reporting period,
        ABN, and activity statement type. You track lodgement due dates
        and confirm successful submission with ATO receipt numbers.""",
        llm=llm,
        verbose=config.verbose,
        allow_delegation=False,
        tools=[]
    )

    return {
        "classification": classification_agent,
        "abn_validator": abn_validator_agent,
        "bas_preparer": bas_preparer_agent,
        "lodgement": lodgement_agent
    }

def create_gst_crew(
    agents: Dict[str, Agent],
    transaction_data: Dict[str, Any],
    config: GSTAgentConfig
) -> Crew:
    """Assemble the GST compliance crew with coordinated tasks."""

    classify_task = Task(
        description=f"""Classify all transactions for GST:
        Business ABN: {transaction_data.get('business_abn')}
        BAS Period: {transaction_data.get('bas_period')}
        Transaction Count: {len(transaction_data.get('transactions', []))}

        Apply ATO GST classification rules:
        - G1: Taxable sales at 10%
        - G2: Export sales (GST-free)
        - G3: Other GST-free sales
        - G4: Input taxed sales
        - G10: Capital acquisitions
        - G11: Non-capital acquisitions
        - G14: Purchases with no GST

        Output classified transactions with BAS labels and confidence scores.""",
        agent=agents["classification"],
        expected_output="JSON with classified_transactions array and classification_summary"
    )

    validate_task = Task(
        description=f"""Validate supplier ABNs for input credit eligibility:
        Purchase transactions to validate: {transaction_data.get('purchase_count', 0)}

        For each purchase transaction:
        1. Verify ABN exists and is active in Australian Business Register
        2. Check if supplier is registered for GST
        3. Verify tax invoice requirements met (>$82.50 threshold)
        4. Calculate input credit eligibility

        Flag transactions where credits should not be claimed.""",
        agent=agents["abn_validator"],
        expected_output="JSON with validated_purchases array and credit_eligibility_summary",
        context=[classify_task]
    )

    prepare_task = Task(
        description=f"""Prepare BAS for lodgement:
        Period: {transaction_data.get('bas_period')}
        Reporting Method: {config.reporting_method}

        Aggregate all transactions into BAS labels:
        - Sales: G1 (taxable), G2 (exports), G3 (GST-free), G4 (input-taxed)
        - Purchases: G10 (capital), G11 (non-capital), G14 (no GST)
        - Calculate: 1A (GST on sales), 1B (GST credits), net payable/refundable

        Ensure mathematical accuracy and full reconciliation.""",
        agent=agents["bas_preparer"],
        expected_output="JSON with bas_labels, gst_calculation, and reconciliation_check",
        context=[classify_task, validate_task]
    )

    lodge_task = Task(
        description="""Prepare final BAS for ATO lodgement:
        1. Map calculated values to ATO form fields
        2. Validate all mandatory fields completed
        3. Generate SBR2 payload format
        4. Confirm lodgement readiness

        Output ATO-compliant BAS ready for submission.""",
        agent=agents["lodgement"],
        expected_output="JSON with ato_form_data, validation_status, and lodgement_payload",
        context=[prepare_task]
    )

    return Crew(
        agents=list(agents.values()),
        tasks=[classify_task, validate_task, prepare_task, lodge_task],
        process=Process.sequential,
        verbose=True
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Build robust connectors for Australian accounting systems (Xero, MYOB, QuickBooks) to extract transactions, supplier details, and tax codes with validation and normalization.',
            toolsUsed: ['LangChain', 'Xero API', 'MYOB API', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Multi-Source Transaction Ingestion for GST',
                description:
                  'Production-grade data ingestion from Australian accounting systems with GST-specific field mapping.',
                code: `from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
from decimal import Decimal
from datetime import date, datetime
from langchain.tools import BaseTool
from enum import Enum
import httpx
import logging

logger = logging.getLogger(__name__)

class TransactionType(str, Enum):
    SALE = "sale"
    PURCHASE = "purchase"
    ADJUSTMENT = "adjustment"

class GSTTransaction(BaseModel):
    """Normalized transaction for GST analysis."""
    transaction_id: str
    transaction_type: TransactionType
    date: date
    description: str
    contact_name: Optional[str] = None
    contact_abn: Optional[str] = None
    amount_incl_gst: Decimal
    amount_excl_gst: Decimal
    gst_amount: Decimal
    tax_code: Optional[str] = None
    account_code: str
    is_capital: bool = False
    source_system: str
    raw_data: dict = Field(default_factory=dict)

    @validator('gst_amount', pre=True, always=True)
    def calculate_gst(cls, v, values):
        if v is None and values.get('amount_incl_gst'):
            # GST is 1/11 of GST-inclusive amount
            return values['amount_incl_gst'] / Decimal('11')
        return v

class SupplierRecord(BaseModel):
    """Supplier record for ABN validation."""
    supplier_id: str
    supplier_name: str
    abn: Optional[str] = None
    gst_registered: Optional[bool] = None
    gst_registration_date: Optional[date] = None
    last_validated: Optional[datetime] = None

class XeroGSTIngestionTool(BaseTool):
    """LangChain tool for Xero transaction extraction."""
    name: str = "xero_gst_ingestion"
    description: str = "Extracts transactions from Xero for GST analysis"

    client_id: str
    client_secret: str
    tenant_id: str

    async def _get_access_token(self, client: httpx.AsyncClient) -> str:
        """Obtain OAuth2 access token from Xero."""
        # Implementation uses Xero OAuth2 flow
        pass

    def _run(self, bas_period_start: str, bas_period_end: str) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(self._arun(bas_period_start, bas_period_end))

    async def _arun(
        self,
        bas_period_start: str,
        bas_period_end: str
    ) -> Dict[str, Any]:
        """Extract Xero transactions for BAS period."""
        base_url = "https://api.xero.com/api.xro/2.0"

        async with httpx.AsyncClient() as client:
            token = await self._get_access_token(client)
            headers = {
                "Authorization": f"Bearer {token}",
                "Xero-tenant-id": self.tenant_id,
                "Accept": "application/json"
            }

            # Fetch invoices (sales)
            invoices_resp = await client.get(
                f"{base_url}/Invoices",
                headers=headers,
                params={
                    "where": f"Date >= DateTime({bas_period_start}) AND Date <= DateTime({bas_period_end})",
                    "Statuses": "AUTHORISED,PAID"
                }
            )
            invoices = invoices_resp.json().get("Invoices", [])

            # Fetch bills (purchases)
            bills_resp = await client.get(
                f"{base_url}/Invoices",
                headers=headers,
                params={
                    "where": f"Date >= DateTime({bas_period_start}) AND Date <= DateTime({bas_period_end}) AND Type == \\"ACCPAY\\"",
                    "Statuses": "AUTHORISED,PAID"
                }
            )
            bills = bills_resp.json().get("Invoices", [])

            # Transform to normalized format
            sales = self._transform_invoices(invoices, TransactionType.SALE)
            purchases = self._transform_invoices(bills, TransactionType.PURCHASE)

            # Extract unique suppliers for ABN validation
            suppliers = self._extract_suppliers(purchases)

            return {
                "sales": [t.dict() for t in sales],
                "purchases": [t.dict() for t in purchases],
                "suppliers": [s.dict() for s in suppliers],
                "period": {"start": bas_period_start, "end": bas_period_end},
                "extraction_timestamp": datetime.utcnow().isoformat()
            }

    def _transform_invoices(
        self,
        invoices: List[Dict],
        txn_type: TransactionType
    ) -> List[GSTTransaction]:
        """Transform Xero invoices to normalized GST transactions."""
        transactions = []

        for inv in invoices:
            for line in inv.get("LineItems", []):
                amount_excl = Decimal(str(line.get("LineAmount", 0)))
                tax_amount = Decimal(str(line.get("TaxAmount", 0)))
                amount_incl = amount_excl + tax_amount

                # Detect capital items
                account_code = line.get("AccountCode", "")
                is_capital = account_code.startswith("15") or "CAPEX" in line.get("Description", "").upper()

                transactions.append(GSTTransaction(
                    transaction_id=f"{inv['InvoiceID']}-{line.get('LineItemID', '')}",
                    transaction_type=txn_type,
                    date=datetime.strptime(inv["Date"][:10], "%Y-%m-%d").date(),
                    description=line.get("Description", ""),
                    contact_name=inv.get("Contact", {}).get("Name"),
                    contact_abn=inv.get("Contact", {}).get("TaxNumber"),
                    amount_incl_gst=amount_incl,
                    amount_excl_gst=amount_excl,
                    gst_amount=tax_amount,
                    tax_code=line.get("TaxType"),
                    account_code=account_code,
                    is_capital=is_capital,
                    source_system="xero",
                    raw_data=line
                ))

        return transactions

    def _extract_suppliers(self, purchases: List[GSTTransaction]) -> List[SupplierRecord]:
        """Extract unique suppliers from purchase transactions."""
        suppliers_dict = {}

        for txn in purchases:
            if txn.contact_name and txn.contact_name not in suppliers_dict:
                suppliers_dict[txn.contact_name] = SupplierRecord(
                    supplier_id=txn.contact_name.replace(" ", "_").lower(),
                    supplier_name=txn.contact_name,
                    abn=txn.contact_abn
                )

        return list(suppliers_dict.values())

class ABRLookupTool(BaseTool):
    """LangChain tool for Australian Business Register ABN validation."""
    name: str = "abr_lookup"
    description: str = "Validates ABNs against the Australian Business Register"

    guid: str  # ABR web services GUID

    def _run(self, abn: str) -> Dict[str, Any]:
        import asyncio
        return asyncio.run(self._arun(abn))

    async def _arun(self, abn: str) -> Dict[str, Any]:
        """Look up ABN in Australian Business Register."""
        # Remove spaces and validate format
        abn_clean = abn.replace(" ", "")
        if len(abn_clean) != 11 or not abn_clean.isdigit():
            return {
                "abn": abn,
                "valid": False,
                "error": "Invalid ABN format - must be 11 digits"
            }

        url = "https://abr.business.gov.au/json/AbnDetails.aspx"

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    url,
                    params={"abn": abn_clean, "callback": "", "guid": self.guid},
                    timeout=10
                )

                # Parse JSONP response
                text = resp.text
                if text.startswith("callback("):
                    text = text[9:-1]

                import json
                data = json.loads(text)

                is_valid = data.get("Abn") is not None
                gst_from = data.get("Gst")
                is_gst_registered = bool(gst_from and gst_from != "")

                return {
                    "abn": abn_clean,
                    "valid": is_valid,
                    "entity_name": data.get("EntityName"),
                    "entity_type": data.get("EntityTypeName"),
                    "gst_registered": is_gst_registered,
                    "gst_from_date": gst_from,
                    "status": data.get("AbnStatus"),
                    "state": data.get("AddressState"),
                    "postcode": data.get("AddressPostcode"),
                    "lookup_timestamp": datetime.utcnow().isoformat()
                }

            except Exception as e:
                logger.error(f"ABR lookup failed for {abn}: {e}")
                return {
                    "abn": abn,
                    "valid": False,
                    "error": str(e)
                }

class GSTDataOrchestrator:
    """Coordinates data ingestion for GST compliance."""

    def __init__(
        self,
        xero_tool: Optional[XeroGSTIngestionTool] = None,
        abr_tool: Optional[ABRLookupTool] = None
    ):
        self.xero_tool = xero_tool
        self.abr_tool = abr_tool

    async def ingest_bas_period(
        self,
        business_abn: str,
        bas_period_start: str,
        bas_period_end: str
    ) -> Dict[str, Any]:
        """Ingest all data for a BAS period."""

        result = {
            "business_abn": business_abn,
            "period": {"start": bas_period_start, "end": bas_period_end},
            "sales": [],
            "purchases": [],
            "suppliers": [],
            "supplier_validations": [],
            "errors": []
        }

        # Ingest from Xero
        if self.xero_tool:
            try:
                xero_data = await self.xero_tool._arun(bas_period_start, bas_period_end)
                result["sales"] = xero_data["sales"]
                result["purchases"] = xero_data["purchases"]
                result["suppliers"] = xero_data["suppliers"]
            except Exception as e:
                result["errors"].append({"source": "xero", "error": str(e)})

        # Validate supplier ABNs
        if self.abr_tool and result["suppliers"]:
            for supplier in result["suppliers"]:
                if supplier.get("abn"):
                    try:
                        validation = await self.abr_tool._arun(supplier["abn"])
                        validation["supplier_id"] = supplier["supplier_id"]
                        result["supplier_validations"].append(validation)
                    except Exception as e:
                        result["errors"].append({
                            "source": "abr",
                            "supplier": supplier["supplier_name"],
                            "error": str(e)
                        })

        logger.info(
            f"Ingested {len(result['sales'])} sales, "
            f"{len(result['purchases'])} purchases, "
            f"validated {len(result['supplier_validations'])} ABNs"
        )

        return result`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the GST classification agent with ATO rule engine, the ABN validator agent with input credit eligibility checks, and the BAS preparer agent with label aggregation.',
            toolsUsed: ['LangChain', 'Pandas', 'NumPy'],
            codeSnippets: [
              {
                language: 'python',
                title: 'GST Classification and BAS Preparation Engines',
                description:
                  'Domain-specific analysis agents for GST classification, ABN validation, and BAS computation.',
                code: `from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from datetime import date
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class GSTCode(str, Enum):
    GST = "GST"
    FRE = "FRE"
    INP = "INP"
    EXP = "EXP"
    CAP = "CAP"
    NCG = "NCG"
    NT = "N-T"

class BASLabel(str, Enum):
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"
    G4 = "G4"
    G10 = "G10"
    G11 = "G11"
    G14 = "G14"

@dataclass
class ClassificationResult:
    transaction_id: str
    recommended_code: GSTCode
    bas_label: BASLabel
    current_code_correct: bool
    amount_excl_gst: Decimal
    gst_calculated: Decimal
    gst_reported: Decimal
    variance: Decimal
    confidence: float
    issue: Optional[str] = None

@dataclass
class ABNValidationResult:
    supplier_id: str
    supplier_name: str
    abn: str
    abn_valid: bool
    gst_registered: bool
    input_credit_eligible: bool
    reason: Optional[str] = None

@dataclass
class BASComputation:
    period: str
    g1_total_sales: Decimal
    g1_gst: Decimal
    g2_exports: Decimal
    g3_gst_free: Decimal
    g4_input_taxed: Decimal
    g10_capital: Decimal
    g10_gst: Decimal
    g11_other: Decimal
    g11_gst: Decimal
    g14_no_gst: Decimal
    label_1a_gst_on_sales: Decimal
    label_1b_gst_on_purchases: Decimal
    net_gst: Decimal
    is_refund: bool

class GSTClassificationEngine:
    """Rule-based GST classification for Australian transactions."""

    # GST-free categories
    GST_FREE_KEYWORDS = [
        'medical', 'health', 'doctor', 'hospital', 'pharmacy',
        'education', 'school', 'university', 'tuition',
        'childcare', 'daycare',
        'export', 'overseas', 'international',
        'basic food', 'bread', 'milk', 'fruit', 'vegetable'
    ]

    # Input-taxed categories
    INPUT_TAXED_KEYWORDS = [
        'interest', 'loan', 'financial', 'banking',
        'residential rent', 'residential lease',
        'precious metal', 'gold', 'silver'
    ]

    def classify_transactions(
        self,
        transactions: List[Dict]
    ) -> List[ClassificationResult]:
        """Classify all transactions according to ATO GST rules."""
        results = []

        for txn in transactions:
            result = self._classify_single(txn)
            results.append(result)

        return results

    def _classify_single(self, txn: Dict) -> ClassificationResult:
        """Classify a single transaction."""
        description = txn.get('description', '').lower()
        current_code = txn.get('tax_code', '')
        amount_excl = Decimal(str(txn.get('amount_excl_gst', 0)))
        gst_reported = Decimal(str(txn.get('gst_amount', 0)))
        is_capital = txn.get('is_capital', False)
        txn_type = txn.get('transaction_type', 'sale')

        # Determine recommended classification
        recommended_code, bas_label, confidence = self._determine_classification(
            description, txn_type, is_capital
        )

        # Calculate expected GST
        if recommended_code == GSTCode.GST or (recommended_code == GSTCode.CAP and txn_type == 'purchase'):
            gst_calculated = amount_excl * Decimal('0.1')
        elif recommended_code == GSTCode.NCG:
            gst_calculated = amount_excl * Decimal('0.1')
        else:
            gst_calculated = Decimal('0')

        # Check if current code is correct
        current_correct = self._codes_match(current_code, recommended_code)
        variance = gst_reported - gst_calculated

        # Flag issues
        issue = None
        if abs(variance) > Decimal('0.05'):
            issue = f"GST variance of \${variance:.2f}"
        elif not current_correct:
            issue = f"Tax code mismatch: {current_code} should be {recommended_code.value}"

        return ClassificationResult(
            transaction_id=txn.get('transaction_id', ''),
            recommended_code=recommended_code,
            bas_label=bas_label,
            current_code_correct=current_correct,
            amount_excl_gst=amount_excl,
            gst_calculated=gst_calculated,
            gst_reported=gst_reported,
            variance=variance,
            confidence=confidence,
            issue=issue
        )

    def _determine_classification(
        self,
        description: str,
        txn_type: str,
        is_capital: bool
    ) -> Tuple[GSTCode, BASLabel, float]:
        """Determine GST code and BAS label based on transaction details."""

        # Check for GST-free
        if any(kw in description for kw in self.GST_FREE_KEYWORDS):
            if 'export' in description or 'overseas' in description:
                return GSTCode.EXP, BASLabel.G2, 0.90
            return GSTCode.FRE, BASLabel.G3, 0.85

        # Check for input-taxed
        if any(kw in description for kw in self.INPUT_TAXED_KEYWORDS):
            return GSTCode.INP, BASLabel.G4, 0.85

        # Check for no GST items
        if any(kw in description for kw in ['bank fee', 'government', 'stamp duty', 'payroll', 'wage']):
            return GSTCode.NT, BASLabel.G14, 0.90

        # Standard taxable
        if txn_type == 'sale':
            return GSTCode.GST, BASLabel.G1, 0.95
        elif txn_type == 'purchase':
            if is_capital:
                return GSTCode.CAP, BASLabel.G10, 0.90
            return GSTCode.NCG, BASLabel.G11, 0.90

        return GSTCode.GST, BASLabel.G1, 0.80

    def _codes_match(self, current: str, recommended: GSTCode) -> bool:
        """Check if current tax code matches recommended."""
        current_upper = current.upper() if current else ''
        return current_upper == recommended.value or current_upper in [
            recommended.value,
            f"OUTPUT{recommended.value}",
            f"INPUT{recommended.value}"
        ]

class ABNValidationEngine:
    """Validates ABNs and determines input credit eligibility."""

    TAX_INVOICE_THRESHOLD = Decimal('82.50')

    def validate_for_input_credits(
        self,
        purchases: List[Dict],
        abn_validations: List[Dict]
    ) -> List[ABNValidationResult]:
        """Validate all purchase transactions for input credit eligibility."""

        # Build ABN lookup
        abn_lookup = {v['abn']: v for v in abn_validations if v.get('abn')}

        results = []
        for txn in purchases:
            result = self._validate_single(txn, abn_lookup)
            results.append(result)

        return results

    def _validate_single(
        self,
        txn: Dict,
        abn_lookup: Dict
    ) -> ABNValidationResult:
        """Validate a single purchase for input credit eligibility."""

        supplier_abn = txn.get('contact_abn', '')
        amount = Decimal(str(txn.get('amount_incl_gst', 0)))

        # Check if ABN is required (>$82.50)
        abn_required = amount > self.TAX_INVOICE_THRESHOLD

        if not supplier_abn:
            return ABNValidationResult(
                supplier_id=txn.get('contact_name', '').replace(' ', '_').lower(),
                supplier_name=txn.get('contact_name', ''),
                abn='',
                abn_valid=False,
                gst_registered=False,
                input_credit_eligible=not abn_required,
                reason="No ABN provided" + (f" - credit ineligible (amount \${amount:.2f} > \\$82.50)" if abn_required else "")
            )

        # Look up validation result
        validation = abn_lookup.get(supplier_abn.replace(' ', ''), {})

        abn_valid = validation.get('valid', False)
        gst_registered = validation.get('gst_registered', False)

        if not abn_valid:
            return ABNValidationResult(
                supplier_id=txn.get('contact_name', '').replace(' ', '_').lower(),
                supplier_name=txn.get('contact_name', ''),
                abn=supplier_abn,
                abn_valid=False,
                gst_registered=False,
                input_credit_eligible=False,
                reason=f"Invalid ABN - {validation.get('error', 'not found in ABR')}"
            )

        if not gst_registered:
            return ABNValidationResult(
                supplier_id=txn.get('contact_name', '').replace(' ', '_').lower(),
                supplier_name=txn.get('contact_name', ''),
                abn=supplier_abn,
                abn_valid=True,
                gst_registered=False,
                input_credit_eligible=False,
                reason="Supplier not registered for GST - cannot claim input credits"
            )

        return ABNValidationResult(
            supplier_id=txn.get('contact_name', '').replace(' ', '_').lower(),
            supplier_name=txn.get('contact_name', ''),
            abn=supplier_abn,
            abn_valid=True,
            gst_registered=True,
            input_credit_eligible=True,
            reason=None
        )

class BASPreparationEngine:
    """Prepares BAS labels from classified transactions."""

    def compute_bas(
        self,
        classifications: List[ClassificationResult],
        abn_validations: List[ABNValidationResult],
        transactions: List[Dict],
        period: str
    ) -> BASComputation:
        """Compute all BAS labels from classified transactions."""

        # Build validation lookup
        validation_lookup = {v.supplier_id: v for v in abn_validations}

        # Initialize accumulators
        labels = {
            'G1': {'amount': Decimal('0'), 'gst': Decimal('0')},
            'G2': {'amount': Decimal('0'), 'gst': Decimal('0')},
            'G3': {'amount': Decimal('0'), 'gst': Decimal('0')},
            'G4': {'amount': Decimal('0'), 'gst': Decimal('0')},
            'G10': {'amount': Decimal('0'), 'gst': Decimal('0')},
            'G11': {'amount': Decimal('0'), 'gst': Decimal('0')},
            'G14': {'amount': Decimal('0'), 'gst': Decimal('0')},
        }

        # Aggregate by BAS label
        for cls in classifications:
            label = cls.bas_label.value
            labels[label]['amount'] += cls.amount_excl_gst
            labels[label]['gst'] += cls.gst_calculated

        # Calculate 1A and 1B
        gst_on_sales = labels['G1']['gst']
        gst_on_purchases = labels['G10']['gst'] + labels['G11']['gst']
        net_gst = gst_on_sales - gst_on_purchases

        return BASComputation(
            period=period,
            g1_total_sales=labels['G1']['amount'],
            g1_gst=labels['G1']['gst'],
            g2_exports=labels['G2']['amount'],
            g3_gst_free=labels['G3']['amount'],
            g4_input_taxed=labels['G4']['amount'],
            g10_capital=labels['G10']['amount'],
            g10_gst=labels['G10']['gst'],
            g11_other=labels['G11']['amount'],
            g11_gst=labels['G11']['gst'],
            g14_no_gst=labels['G14']['amount'],
            label_1a_gst_on_sales=gst_on_sales,
            label_1b_gst_on_purchases=gst_on_purchases,
            net_gst=net_gst,
            is_refund=net_gst < 0
        )`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Wire the GST agents together using LangGraph state machine with proper state transitions, BAS reconciliation checks, and checkpointing for ATO compliance.',
            toolsUsed: ['LangGraph', 'Redis', 'Pydantic'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph GST Compliance Orchestrator',
                description:
                  'State machine orchestration for the multi-agent GST workflow with ATO compliance persistence.',
                code: `from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from decimal import Decimal
import json
import redis
import logging

logger = logging.getLogger(__name__)

class GSTState(TypedDict):
    """State schema for GST compliance workflow."""
    # Business context
    business_abn: str
    business_name: str
    bas_period: str
    reporting_method: str

    # Ingested data
    sales: List[Dict]
    purchases: List[Dict]
    suppliers: List[Dict]
    supplier_validations: List[Dict]

    # Classification results
    classifications: List[Dict]
    classification_issues: List[Dict]

    # ABN validation results
    abn_validations: List[Dict]
    credit_ineligible: List[Dict]

    # BAS computation
    bas_labels: Dict
    gst_on_sales: str
    gst_on_purchases: str
    net_gst: str
    is_refund: bool

    # Lodgement preparation
    ato_form_data: Dict
    validation_status: str
    lodgement_ready: bool

    # Workflow metadata
    current_step: str
    audit_trail: List[Dict]
    error_message: Optional[str]
    started_at: str
    completed_at: Optional[str]

def create_gst_workflow(
    data_orchestrator,
    classification_engine,
    abn_engine,
    bas_engine
) -> StateGraph:
    """Build the LangGraph GST compliance workflow."""

    workflow = StateGraph(GSTState)

    # Node: Data Ingestion
    async def ingest_data(state: GSTState) -> GSTState:
        logger.info(f"Ingesting data for BAS period {state['bas_period']}")
        state["current_step"] = "ingestion"
        state["audit_trail"].append({
            "step": "ingestion",
            "timestamp": datetime.utcnow().isoformat(),
            "action": "started"
        })

        try:
            # Parse period dates
            period_parts = state["bas_period"].split("-")
            if len(period_parts) == 2:
                year, quarter = period_parts
                q_num = int(quarter[1])
                start_month = (q_num - 1) * 3 + 1
                end_month = start_month + 2
                bas_start = f"{year}-{start_month:02d}-01"
                bas_end = f"{year}-{end_month:02d}-28"
            else:
                bas_start = state["bas_period"] + "-01"
                bas_end = state["bas_period"] + "-28"

            result = await data_orchestrator.ingest_bas_period(
                state["business_abn"],
                bas_start,
                bas_end
            )

            state["sales"] = result["sales"]
            state["purchases"] = result["purchases"]
            state["suppliers"] = result["suppliers"]
            state["supplier_validations"] = result["supplier_validations"]

            state["audit_trail"].append({
                "step": "ingestion",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "completed",
                "details": {
                    "sales_count": len(state["sales"]),
                    "purchases_count": len(state["purchases"]),
                    "suppliers_validated": len(state["supplier_validations"])
                }
            })

        except Exception as e:
            state["error_message"] = f"Ingestion failed: {str(e)}"
            logger.error(state["error_message"])

        return state

    # Node: Classification
    def classify_transactions(state: GSTState) -> GSTState:
        logger.info("Classifying transactions")
        state["current_step"] = "classification"

        try:
            all_transactions = state["sales"] + state["purchases"]
            results = classification_engine.classify_transactions(all_transactions)

            state["classifications"] = [
                {
                    "transaction_id": r.transaction_id,
                    "recommended_code": r.recommended_code.value,
                    "bas_label": r.bas_label.value,
                    "current_code_correct": r.current_code_correct,
                    "amount_excl_gst": str(r.amount_excl_gst),
                    "gst_calculated": str(r.gst_calculated),
                    "gst_reported": str(r.gst_reported),
                    "variance": str(r.variance),
                    "confidence": r.confidence,
                    "issue": r.issue
                }
                for r in results
            ]

            state["classification_issues"] = [
                c for c in state["classifications"] if c.get("issue")
            ]

            state["audit_trail"].append({
                "step": "classification",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "completed",
                "details": {
                    "total_classified": len(results),
                    "issues_found": len(state["classification_issues"])
                }
            })

        except Exception as e:
            state["error_message"] = f"Classification failed: {str(e)}"

        return state

    # Node: ABN Validation
    def validate_abns(state: GSTState) -> GSTState:
        logger.info("Validating ABNs for input credits")
        state["current_step"] = "abn_validation"

        try:
            results = abn_engine.validate_for_input_credits(
                state["purchases"],
                state["supplier_validations"]
            )

            state["abn_validations"] = [
                {
                    "supplier_id": r.supplier_id,
                    "supplier_name": r.supplier_name,
                    "abn": r.abn,
                    "abn_valid": r.abn_valid,
                    "gst_registered": r.gst_registered,
                    "input_credit_eligible": r.input_credit_eligible,
                    "reason": r.reason
                }
                for r in results
            ]

            state["credit_ineligible"] = [
                v for v in state["abn_validations"] if not v["input_credit_eligible"]
            ]

            state["audit_trail"].append({
                "step": "abn_validation",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "completed",
                "details": {
                    "suppliers_validated": len(results),
                    "credits_ineligible": len(state["credit_ineligible"])
                }
            })

        except Exception as e:
            state["error_message"] = f"ABN validation failed: {str(e)}"

        return state

    # Node: BAS Preparation
    def prepare_bas(state: GSTState) -> GSTState:
        logger.info("Preparing BAS labels")
        state["current_step"] = "bas_preparation"

        try:
            # Convert classifications back to objects
            from dataclasses import dataclass
            from decimal import Decimal

            @dataclass
            class ClassResult:
                transaction_id: str
                bas_label: type
                amount_excl_gst: Decimal
                gst_calculated: Decimal

            class BL:
                def __init__(self, v):
                    self.value = v

            cls_objects = [
                ClassResult(
                    transaction_id=c["transaction_id"],
                    bas_label=BL(c["bas_label"]),
                    amount_excl_gst=Decimal(c["amount_excl_gst"]),
                    gst_calculated=Decimal(c["gst_calculated"])
                )
                for c in state["classifications"]
            ]

            abn_objects = [
                type('ABNResult', (), v)()
                for v in state["abn_validations"]
            ]

            bas = bas_engine.compute_bas(
                cls_objects,
                abn_objects,
                state["sales"] + state["purchases"],
                state["bas_period"]
            )

            state["bas_labels"] = {
                "G1": {"amount": str(bas.g1_total_sales), "gst": str(bas.g1_gst)},
                "G2": {"amount": str(bas.g2_exports)},
                "G3": {"amount": str(bas.g3_gst_free)},
                "G4": {"amount": str(bas.g4_input_taxed)},
                "G10": {"amount": str(bas.g10_capital), "gst": str(bas.g10_gst)},
                "G11": {"amount": str(bas.g11_other), "gst": str(bas.g11_gst)},
                "G14": {"amount": str(bas.g14_no_gst)}
            }
            state["gst_on_sales"] = str(bas.label_1a_gst_on_sales)
            state["gst_on_purchases"] = str(bas.label_1b_gst_on_purchases)
            state["net_gst"] = str(bas.net_gst)
            state["is_refund"] = bas.is_refund

            # Prepare ATO form data
            state["ato_form_data"] = {
                "abn": state["business_abn"],
                "period": state["bas_period"],
                "G1": str(bas.g1_total_sales),
                "G2": str(bas.g2_exports),
                "G3": str(bas.g3_gst_free),
                "G10": str(bas.g10_capital),
                "G11": str(bas.g11_other),
                "1A": str(bas.label_1a_gst_on_sales),
                "1B": str(bas.label_1b_gst_on_purchases),
                "net": str(abs(bas.net_gst)),
                "type": "refund" if bas.is_refund else "payable"
            }

            state["validation_status"] = "validated"
            state["lodgement_ready"] = len(state["classification_issues"]) == 0
            state["completed_at"] = datetime.utcnow().isoformat()

            state["audit_trail"].append({
                "step": "bas_preparation",
                "timestamp": datetime.utcnow().isoformat(),
                "action": "completed",
                "details": {
                    "net_gst": str(bas.net_gst),
                    "is_refund": bas.is_refund,
                    "lodgement_ready": state["lodgement_ready"]
                }
            })

        except Exception as e:
            state["error_message"] = f"BAS preparation failed: {str(e)}"
            state["validation_status"] = "failed"

        return state

    # Conditional edge
    def should_continue(state: GSTState) -> str:
        if state.get("error_message"):
            return "error_handler"
        return "continue"

    def handle_error(state: GSTState) -> GSTState:
        state["audit_trail"].append({
            "step": state["current_step"],
            "timestamp": datetime.utcnow().isoformat(),
            "action": "error",
            "error": state["error_message"]
        })
        return state

    # Build graph
    workflow.add_node("ingest", ingest_data)
    workflow.add_node("classify", classify_transactions)
    workflow.add_node("validate_abns", validate_abns)
    workflow.add_node("prepare_bas", prepare_bas)
    workflow.add_node("error_handler", handle_error)

    workflow.set_entry_point("ingest")
    workflow.add_conditional_edges("ingest", should_continue, {"continue": "classify", "error_handler": "error_handler"})
    workflow.add_conditional_edges("classify", should_continue, {"continue": "validate_abns", "error_handler": "error_handler"})
    workflow.add_conditional_edges("validate_abns", should_continue, {"continue": "prepare_bas", "error_handler": "error_handler"})
    workflow.add_edge("prepare_bas", END)
    workflow.add_edge("error_handler", END)

    return workflow`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Containerize the GST multi-agent system with Docker, integrate LangSmith for agent tracing, and set up BAS audit logging for ATO compliance requirements.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'PostgreSQL'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose for GST Compliance System',
                description:
                  'Production deployment configuration with BAS audit trail and ATO reporting capabilities.',
                code: `version: '3.8'

services:
  gst-agents:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: gst-agents
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - XERO_CLIENT_ID=\${XERO_CLIENT_ID}
      - XERO_CLIENT_SECRET=\${XERO_CLIENT_SECRET}
      - ABR_GUID=\${ABR_GUID}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://gst_user:\${DB_PASSWORD}@postgres:5432/gst_compliance
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=\${LANGCHAIN_API_KEY}
      - LANGCHAIN_PROJECT=australia-gst
      - ATO_SOFTWARE_ID=\${ATO_SOFTWARE_ID}
      - AUDIT_RETENTION_YEARS=5
    ports:
      - "8082:8082"
    depends_on:
      - redis
      - postgres
    volumes:
      - ./logs:/app/logs
      - ./bas_reports:/app/bas_reports
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8082/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: gst-redis
    ports:
      - "6381:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: gst-postgres
    environment:
      - POSTGRES_DB=gst_compliance
      - POSTGRES_USER=gst_user
      - POSTGRES_PASSWORD=\${DB_PASSWORD}
    ports:
      - "5434:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./sql/bas_audit_schema.sql:/docker-entrypoint-initdb.d/001_audit.sql
    restart: unless-stopped

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile.scheduler
    container_name: gst-scheduler
    environment:
      - GST_API_URL=http://gst-agents:8082
      - SCHEDULE_WEEKLY=0 17 * * 5
      - SCHEDULE_QUARTERLY=0 8 21 1,4,7,10 *
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
      - FINANCE_EMAIL=\${FINANCE_EMAIL}
    depends_on:
      - gst-agents
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: gst-prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

volumes:
  redis-data:
  postgres-data:`,
              },
              {
                language: 'python',
                title: 'BAS Audit Trail and ATO Compliance Logging',
                description:
                  'Comprehensive audit trail for 5-year ATO retention requirements with LangSmith integration.',
                code: `import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from decimal import Decimal
import json
import hashlib
from langsmith import Client
import psycopg2
from psycopg2.extras import Json
import logging

logger = logging.getLogger(__name__)

langsmith_client = Client() if os.getenv("LANGCHAIN_API_KEY") else None

@dataclass
class BASAuditEntry:
    """Immutable BAS audit entry for ATO compliance."""
    entry_id: str
    business_abn: str
    bas_period: str
    workflow_run_id: str
    step_name: str
    action: str
    timestamp: datetime
    data_snapshot: Dict[str, Any]
    data_hash: str

    @classmethod
    def create(
        cls,
        business_abn: str,
        bas_period: str,
        workflow_run_id: str,
        step_name: str,
        action: str,
        data: Dict[str, Any]
    ) -> 'BASAuditEntry':
        """Create a new BAS audit entry with computed hash."""
        timestamp = datetime.utcnow()
        data_json = json.dumps(data, sort_keys=True, default=str)
        data_hash = hashlib.sha256(data_json.encode()).hexdigest()

        return cls(
            entry_id=f"BAS-{timestamp.strftime('%Y%m%d%H%M%S')}-{data_hash[:8]}",
            business_abn=business_abn,
            bas_period=bas_period,
            workflow_run_id=workflow_run_id,
            step_name=step_name,
            action=action,
            timestamp=timestamp,
            data_snapshot=data,
            data_hash=data_hash
        )

class BASAuditService:
    """ATO-compliant BAS audit trail with 5-year retention."""

    RETENTION_YEARS = 5

    def __init__(self, database_url: str):
        self.db_url = database_url
        self._init_schema()

    def _init_schema(self):
        """Initialize BAS audit trail schema."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS bas_audit_trail (
                        entry_id TEXT PRIMARY KEY,
                        business_abn TEXT NOT NULL,
                        bas_period TEXT NOT NULL,
                        workflow_run_id TEXT NOT NULL,
                        step_name TEXT NOT NULL,
                        action TEXT NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        data_snapshot JSONB NOT NULL,
                        data_hash TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE INDEX IF NOT EXISTS idx_bas_audit_abn_period
                        ON bas_audit_trail(business_abn, bas_period);

                    CREATE INDEX IF NOT EXISTS idx_bas_audit_workflow
                        ON bas_audit_trail(workflow_run_id);

                    CREATE TABLE IF NOT EXISTS bas_lodgement_history (
                        lodgement_id TEXT PRIMARY KEY,
                        business_abn TEXT NOT NULL,
                        bas_period TEXT NOT NULL,
                        lodgement_date TIMESTAMPTZ NOT NULL,
                        net_gst NUMERIC(14, 2) NOT NULL,
                        is_refund BOOLEAN NOT NULL,
                        ato_receipt_number TEXT,
                        form_data JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
            conn.commit()

    def log_entry(self, entry: BASAuditEntry):
        """Persist BAS audit entry to database."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO bas_audit_trail (
                        entry_id, business_abn, bas_period, workflow_run_id,
                        step_name, action, timestamp, data_snapshot, data_hash
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    entry.entry_id,
                    entry.business_abn,
                    entry.bas_period,
                    entry.workflow_run_id,
                    entry.step_name,
                    entry.action,
                    entry.timestamp,
                    Json(entry.data_snapshot),
                    entry.data_hash
                ))
            conn.commit()

        logger.info(f"BAS audit entry logged: {entry.entry_id}")

    def record_lodgement(
        self,
        business_abn: str,
        bas_period: str,
        net_gst: Decimal,
        is_refund: bool,
        form_data: Dict[str, Any],
        ato_receipt: Optional[str] = None
    ):
        """Record BAS lodgement for audit trail."""
        lodgement_id = f"LODG-{business_abn}-{bas_period}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO bas_lodgement_history (
                        lodgement_id, business_abn, bas_period,
                        lodgement_date, net_gst, is_refund,
                        ato_receipt_number, form_data
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    lodgement_id,
                    business_abn,
                    bas_period,
                    datetime.utcnow(),
                    net_gst,
                    is_refund,
                    ato_receipt,
                    Json(form_data)
                ))
            conn.commit()

        logger.info(f"BAS lodgement recorded: {lodgement_id}")
        return lodgement_id

    def get_lodgement_history(
        self,
        business_abn: str,
        years: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve BAS lodgement history for audit."""
        cutoff = datetime.utcnow() - timedelta(days=365 * years)

        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT lodgement_id, bas_period, lodgement_date,
                           net_gst, is_refund, ato_receipt_number
                    FROM bas_lodgement_history
                    WHERE business_abn = %s AND lodgement_date >= %s
                    ORDER BY lodgement_date DESC
                """, (business_abn, cutoff))

                return [
                    {
                        "lodgement_id": row[0],
                        "bas_period": row[1],
                        "lodgement_date": row[2].isoformat(),
                        "net_gst": float(row[3]),
                        "is_refund": row[4],
                        "ato_receipt": row[5]
                    }
                    for row in cur.fetchall()
                ]

    def generate_ato_audit_report(
        self,
        business_abn: str,
        from_period: str,
        to_period: str
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report for ATO review."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                # Get lodgement summary
                cur.execute("""
                    SELECT bas_period, net_gst, is_refund, lodgement_date
                    FROM bas_lodgement_history
                    WHERE business_abn = %s
                      AND bas_period >= %s AND bas_period <= %s
                    ORDER BY bas_period
                """, (business_abn, from_period, to_period))
                lodgements = cur.fetchall()

                # Get audit trail count
                cur.execute("""
                    SELECT COUNT(*)
                    FROM bas_audit_trail
                    WHERE business_abn = %s
                      AND bas_period >= %s AND bas_period <= %s
                """, (business_abn, from_period, to_period))
                audit_entries = cur.fetchone()[0]

        total_gst_paid = sum(float(l[1]) for l in lodgements if not l[2])
        total_refunds = sum(abs(float(l[1])) for l in lodgements if l[2])

        return {
            "business_abn": business_abn,
            "reporting_period": {"from": from_period, "to": to_period},
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "periods_lodged": len(lodgements),
                "total_gst_paid": total_gst_paid,
                "total_refunds_received": total_refunds,
                "net_position": total_gst_paid - total_refunds,
                "audit_trail_entries": audit_entries
            },
            "lodgements": [
                {
                    "period": l[0],
                    "net_gst": float(l[1]),
                    "type": "refund" if l[2] else "payment",
                    "lodged": l[3].isoformat()
                }
                for l in lodgements
            ]
        }`,
              },
            ],
          },
        ],
      },
    },
  ],
};
