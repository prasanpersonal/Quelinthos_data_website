import type { Category } from '../types.ts';

export const supplyChainCategory: Category = {
  id: 'supply-chain',
  number: 5,
  title: 'Supply Chain & Logistics',
  shortTitle: 'Supply Chain',
  description:
    'Fix inventory blind spots, eliminate spreadsheet shadow IT, and automate import customs friction.',
  icon: 'Truck',
  accentColor: 'neon-purple',
  painPoints: [
    /* ── Pain Point 1: Inventory Data Latency ─────────────────────────── */
    {
      id: 'inventory-latency',
      number: 1,
      title: 'Inventory Data Latency',
      subtitle: 'Real-Time Visibility Gap',
      summary:
        'Your inventory counts are 4-24 hours stale. Overselling, stockouts, and emergency shipments are eating your margins.',
      tags: ['inventory', 'real-time', 'supply-chain'],
      metrics: {
        annualCostRange: '$1M - $5M',
        roi: '8x',
        paybackPeriod: '3-4 months',
        investmentRange: '$120K - $250K',
      },
      price: {
        present: {
          title: 'Current Inventory Visibility Gap',
          description:
            'Warehouse management systems batch-sync inventory every 4-24 hours, leaving downstream teams with stale counts.',
          bullets: [
            'Inventory snapshots are 4-24 hours old by the time planners see them',
            'Order promising relies on morning counts that drift throughout the day',
            'Multi-warehouse allocation uses yesterday\'s data for today\'s decisions',
          ],
          severity: 'critical',
        },
        root: {
          title: 'Batch ETL Between WMS and ERP',
          description:
            'Legacy batch integrations between warehouse management and ERP systems create a persistent data gap.',
          bullets: [
            'WMS exports flat files on a nightly schedule instead of streaming changes',
            'ERP inventory tables updated via scheduled stored procedures with 6-hour lag',
            'No change-data-capture layer to propagate movements in real time',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Overselling, Stockouts & Emergency Freight',
          description:
            'Stale inventory data creates a cascade of costly operational failures across the entire fulfillment chain.',
          bullets: [
            'Overselling when available-to-promise exceeds actual stock — triggering cancellations and chargebacks',
            'Stockouts on fast-moving SKUs because replenishment signals arrive hours too late',
            'Emergency air freight at 5-8x standard cost to cover gaps that real-time data would have prevented',
          ],
          severity: 'critical',
        },
        cost: {
          title: 'Financial Impact of Stale Inventory',
          description:
            'Inventory latency compounds into seven figures annually through lost sales, excess freight, and carrying cost miscalculation.',
          bullets: [
            '$400K-$1.2M/yr in emergency expedited shipping to cover stockouts',
            '$300K-$800K/yr in lost revenue from oversold and cancelled orders',
            '$200K-$600K/yr in excess safety stock held as a buffer against data lag',
          ],
          severity: 'critical',
        },
        expectedReturn: {
          title: 'Real-Time Inventory Sync ROI',
          description:
            'Event-driven inventory pipelines eliminate the batch gap and unlock immediate margin recovery.',
          bullets: [
            'Reduce expedited freight spend by 60-75% within the first quarter',
            'Cut oversell cancellation rate from 4% to under 0.3%',
            'Lower safety-stock buffers by 30%, freeing $500K+ in working capital',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Build an event-driven inventory synchronization layer that streams warehouse movements into a real-time available-to-promise view, replacing batch ETL with sub-second updates.',
        prerequisites: [
          'Access to WMS transaction logs or message queue (Kafka / RabbitMQ)',
          'PostgreSQL 14+ or Snowflake for the real-time inventory store',
          'Python 3.10+ with asyncio support',
          'pytest >= 7.0 with pytest-asyncio',
          'Docker and docker-compose for containerized deployment',
          'cron, Airflow, or Prefect for job scheduling',
          'Slack incoming webhook URL for operational alerts',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Create the Real-Time Inventory Materialized View',
            description:
              'Build a SQL layer that maintains a live available-to-promise quantity by combining on-hand stock, pending receipts, and allocated orders.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Real-Time Available-to-Promise View',
                description:
                  'Materialized view that computes live ATP by warehouse and SKU, refreshed on every inventory event.',
                code: `-- Real-time available-to-promise inventory view
CREATE MATERIALIZED VIEW mv_inventory_atp AS
SELECT
    w.warehouse_id,
    w.warehouse_name,
    s.sku,
    s.product_name,
    COALESCE(oh.on_hand_qty, 0)            AS on_hand,
    COALESCE(alloc.allocated_qty, 0)       AS allocated,
    COALESCE(inbound.pending_receipt, 0)   AS in_transit,
    COALESCE(oh.on_hand_qty, 0)
      - COALESCE(alloc.allocated_qty, 0)
      + COALESCE(inbound.pending_receipt, 0) AS atp_quantity,
    NOW()                                  AS snapshot_ts
FROM warehouses w
CROSS JOIN skus s
LEFT JOIN inventory_on_hand oh
    ON oh.warehouse_id = w.warehouse_id AND oh.sku = s.sku
LEFT JOIN (
    SELECT warehouse_id, sku, SUM(qty) AS allocated_qty
    FROM order_allocations
    WHERE status IN ('reserved', 'picking')
    GROUP BY warehouse_id, sku
) alloc ON alloc.warehouse_id = w.warehouse_id AND alloc.sku = s.sku
LEFT JOIN (
    SELECT dest_warehouse_id AS warehouse_id, sku,
           SUM(expected_qty) AS pending_receipt
    FROM inbound_shipments
    WHERE status = 'in_transit'
    GROUP BY dest_warehouse_id, sku
) inbound ON inbound.warehouse_id = w.warehouse_id AND inbound.sku = s.sku;

CREATE UNIQUE INDEX idx_atp_wh_sku
    ON mv_inventory_atp (warehouse_id, sku);`,
              },
              {
                language: 'sql',
                title: 'Inventory Event Log Table',
                description:
                  'Append-only event log that captures every inventory movement for auditability and replay.',
                code: `-- Append-only inventory event stream
CREATE TABLE inventory_events (
    event_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_type    VARCHAR(30) NOT NULL,  -- 'receipt', 'pick', 'adjustment', 'transfer'
    warehouse_id  INT NOT NULL REFERENCES warehouses(warehouse_id),
    sku           VARCHAR(40) NOT NULL,
    qty_delta     INT NOT NULL,
    reference_id  VARCHAR(80),
    recorded_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_system VARCHAR(30) NOT NULL   -- 'wms', 'erp', 'manual'
);

CREATE INDEX idx_events_sku_ts
    ON inventory_events (sku, recorded_at DESC);

CREATE INDEX idx_events_warehouse_ts
    ON inventory_events (warehouse_id, recorded_at DESC);

-- Trigger function to refresh the ATP view on each event
CREATE OR REPLACE FUNCTION fn_refresh_atp()
RETURNS TRIGGER AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_inventory_atp;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_inventory_event_refresh
    AFTER INSERT ON inventory_events
    FOR EACH STATEMENT
    EXECUTE FUNCTION fn_refresh_atp();`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Build the Event-Driven Inventory Update Pipeline',
            description:
              'Python service that consumes WMS movement messages from Kafka and writes inventory events in real time, keeping the ATP view perpetually fresh.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Kafka Inventory Event Consumer',
                description:
                  'Async consumer that reads WMS inventory movements from Kafka and persists them as inventory events.',
                code: `import asyncio
import json
from datetime import datetime, timezone
from aiokafka import AIOKafkaConsumer
import asyncpg

DB_DSN = "postgresql://inv_svc:secret@db-host:5432/inventory"
KAFKA_BROKERS = "kafka-1:9092,kafka-2:9092"
TOPIC = "wms.inventory.movements"

async def process_movement(pool: asyncpg.Pool, msg: dict) -> None:
    """Persist a single WMS movement as an inventory event."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO inventory_events
                (event_type, warehouse_id, sku, qty_delta, reference_id, source_system)
            VALUES ($1, $2, $3, $4, $5, 'wms')
            """,
            msg["movement_type"],
            int(msg["warehouse_id"]),
            msg["sku"],
            int(msg["qty_delta"]),
            msg.get("reference_id"),
        )

async def main() -> None:
    pool = await asyncpg.create_pool(DB_DSN, min_size=4, max_size=12)
    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BROKERS,
        group_id="inventory-atp-writer",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    await consumer.start()
    try:
        async for record in consumer:
            await process_movement(pool, record.value)
            await consumer.commit()
    finally:
        await consumer.stop()
        await pool.close()

if __name__ == "__main__":
    asyncio.run(main())`,
              },
              {
                language: 'python',
                title: 'ATP Snapshot Health Monitor',
                description:
                  'Monitoring script that alerts when ATP staleness exceeds the target threshold.',
                code: `import asyncio
import asyncpg
from datetime import datetime, timezone, timedelta

DB_DSN = "postgresql://inv_svc:secret@db-host:5432/inventory"
STALENESS_THRESHOLD = timedelta(seconds=30)

async def check_atp_freshness() -> dict:
    """Return freshness metrics for the ATP materialized view."""
    conn = await asyncpg.connect(DB_DSN)
    try:
        row = await conn.fetchrow(
            """
            SELECT
                MAX(snapshot_ts)                          AS latest_snapshot,
                NOW() - MAX(snapshot_ts)                  AS staleness,
                COUNT(DISTINCT sku)                       AS tracked_skus,
                COUNT(*) FILTER (WHERE atp_quantity < 0)  AS negative_atp_rows
            FROM mv_inventory_atp
            """
        )
        staleness: timedelta = row["staleness"]
        report = {
            "latest_snapshot": row["latest_snapshot"].isoformat(),
            "staleness_seconds": staleness.total_seconds(),
            "is_fresh": staleness <= STALENESS_THRESHOLD,
            "tracked_skus": row["tracked_skus"],
            "negative_atp_rows": row["negative_atp_rows"],
        }
        if not report["is_fresh"]:
            print(f"ALERT: ATP view is {staleness.total_seconds():.1f}s stale")
        if report["negative_atp_rows"] > 0:
            print(f"WARN: {report['negative_atp_rows']} SKUs with negative ATP")
        return report
    finally:
        await conn.close()

if __name__ == "__main__":
    result = asyncio.run(check_atp_freshness())
    print(result)`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Automated data quality assertions and pipeline validation tests that verify ATP accuracy, event completeness, and guard against negative-stock anomalies before changes reach production.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Inventory Data Quality Assertions',
                description:
                  'SQL assertion queries that validate row counts, null checks, referential integrity, and freshness of the inventory pipeline.',
                code: `-- ============================================================
-- Inventory Pipeline Data Quality Assertions
-- Run as a post-refresh validation gate
-- ============================================================

-- 1. Row count sanity: ATP view must cover all active SKUs
SELECT
    'atp_coverage' AS assertion,
    CASE
        WHEN COUNT(*) >= (SELECT COUNT(*) FROM skus WHERE is_active)
        THEN 'PASS'
        ELSE 'FAIL: ATP covers ' || COUNT(*) || ' rows but '
             || (SELECT COUNT(*) FROM skus WHERE is_active) || ' active SKUs exist'
    END AS result
FROM mv_inventory_atp;

-- 2. No NULL warehouse or SKU in the ATP view
SELECT
    'no_null_keys' AS assertion,
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL: ' || COUNT(*) || ' rows with NULL warehouse_id or sku'
    END AS result
FROM mv_inventory_atp
WHERE warehouse_id IS NULL OR sku IS NULL;

-- 3. Referential integrity: every event references a valid warehouse
SELECT
    'event_warehouse_fk' AS assertion,
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL: ' || COUNT(*) || ' events reference missing warehouses'
    END AS result
FROM inventory_events e
LEFT JOIN warehouses w ON w.warehouse_id = e.warehouse_id
WHERE w.warehouse_id IS NULL;

-- 4. Freshness: ATP snapshot must be less than 60 seconds old
SELECT
    'atp_freshness' AS assertion,
    CASE
        WHEN EXTRACT(EPOCH FROM NOW() - MAX(snapshot_ts)) < 60
        THEN 'PASS'
        ELSE 'FAIL: ATP is ' || ROUND(EXTRACT(EPOCH FROM NOW() - MAX(snapshot_ts)))
             || 's stale'
    END AS result
FROM mv_inventory_atp;

-- 5. No negative on-hand (physical impossibility)
SELECT
    'no_negative_on_hand' AS assertion,
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL: ' || COUNT(*) || ' SKUs with negative on_hand'
    END AS result
FROM mv_inventory_atp
WHERE on_hand < 0;`,
              },
              {
                language: 'python',
                title: 'Pytest Pipeline Validation Suite',
                description:
                  'pytest-based tests that validate the inventory event pipeline end-to-end: ATP accuracy, event completeness, and negative-stock detection.',
                code: `import asyncio
import pytest
import asyncpg
from datetime import datetime, timezone, timedelta

DB_DSN = "postgresql://inv_svc:secret@db-host:5432/inventory"


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def db_pool():
    pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=4)
    yield pool
    await pool.close()


@pytest.mark.asyncio
async def test_atp_row_count_matches_active_skus(db_pool):
    """ATP view must have at least one row per active SKU."""
    async with db_pool.acquire() as conn:
        atp_count = await conn.fetchval(
            "SELECT COUNT(*) FROM mv_inventory_atp"
        )
        sku_count = await conn.fetchval(
            "SELECT COUNT(*) FROM skus WHERE is_active"
        )
    assert atp_count >= sku_count, (
        f"ATP has {atp_count} rows but {sku_count} active SKUs exist"
    )


@pytest.mark.asyncio
async def test_no_negative_on_hand(db_pool):
    """Physical on-hand stock must never be negative."""
    async with db_pool.acquire() as conn:
        negatives = await conn.fetchval(
            "SELECT COUNT(*) FROM mv_inventory_atp WHERE on_hand < 0"
        )
    assert negatives == 0, f"{negatives} SKUs have negative on-hand"


@pytest.mark.asyncio
async def test_atp_freshness_under_60s(db_pool):
    """ATP materialized view must be refreshed within 60 seconds."""
    async with db_pool.acquire() as conn:
        staleness = await conn.fetchval(
            "SELECT EXTRACT(EPOCH FROM NOW() - MAX(snapshot_ts)) FROM mv_inventory_atp"
        )
    assert staleness is not None, "No snapshot_ts found in ATP view"
    assert staleness < 60, f"ATP view is {staleness:.1f}s stale (threshold: 60s)"


@pytest.mark.asyncio
async def test_event_completeness_no_orphan_warehouses(db_pool):
    """Every inventory event must reference a valid warehouse."""
    async with db_pool.acquire() as conn:
        orphans = await conn.fetchval("""
            SELECT COUNT(*)
            FROM inventory_events e
            LEFT JOIN warehouses w ON w.warehouse_id = e.warehouse_id
            WHERE w.warehouse_id IS NULL
        """)
    assert orphans == 0, f"{orphans} events reference non-existent warehouses"


@pytest.mark.asyncio
async def test_no_duplicate_events_within_window(db_pool):
    """No duplicate event_id should exist in the last hour."""
    async with db_pool.acquire() as conn:
        dupes = await conn.fetchval("""
            SELECT COUNT(*) FROM (
                SELECT event_id, COUNT(*)
                FROM inventory_events
                WHERE recorded_at >= NOW() - INTERVAL '1 hour'
                GROUP BY event_id HAVING COUNT(*) > 1
            ) d
        """)
    assert dupes == 0, f"{dupes} duplicate event_ids in the last hour"`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Production deployment automation for the Kafka inventory consumer: container build, database migration, health checks, and scheduler configuration.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Inventory Consumer Deployment Script',
                description:
                  'Deployment script that validates the environment, installs dependencies, runs DB migrations, and sets up the Kafka consumer as a managed container.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Inventory ATP Consumer — Deployment Script
# ============================================================

APP_NAME="inventory-atp-consumer"
DEPLOY_ENV="\${DEPLOY_ENV:?ERROR: DEPLOY_ENV must be set (staging|production)}"
DB_HOST="\${DB_HOST:?ERROR: DB_HOST must be set}"
KAFKA_BROKERS="\${KAFKA_BROKERS:?ERROR: KAFKA_BROKERS must be set}"
IMAGE_TAG="\${IMAGE_TAG:-latest}"

echo "==> Deploying \${APP_NAME} to \${DEPLOY_ENV} (image tag: \${IMAGE_TAG})"

# --- Pre-flight checks ---
echo "==> Running pre-flight checks..."
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "ERROR: docker-compose not found"; exit 1; }
command -v psql >/dev/null 2>&1 || { echo "ERROR: psql not found"; exit 1; }

# Verify database connectivity
psql "host=\${DB_HOST} dbname=inventory user=inv_svc" -c "SELECT 1;" >/dev/null 2>&1 \\
    || { echo "ERROR: Cannot connect to database at \${DB_HOST}"; exit 1; }
echo "    Database connectivity: OK"

# --- Install / update Python dependencies ---
echo "==> Installing Python dependencies..."
pip install --quiet --upgrade -r requirements.txt

# --- Run database migrations ---
echo "==> Running database migrations..."
psql "host=\${DB_HOST} dbname=inventory user=inv_svc" -f migrations/001_create_inventory_events.sql
psql "host=\${DB_HOST} dbname=inventory user=inv_svc" -f migrations/002_create_atp_view.sql
psql "host=\${DB_HOST} dbname=inventory user=inv_svc" -f migrations/003_create_refresh_trigger.sql
echo "    Migrations applied successfully"

# --- Build and deploy container ---
echo "==> Building Docker image..."
docker build -t "\${APP_NAME}:\${IMAGE_TAG}" .

echo "==> Stopping existing container (if any)..."
docker-compose -f "docker-compose.\${DEPLOY_ENV}.yml" down --remove-orphans || true

echo "==> Starting \${APP_NAME} container..."
docker-compose -f "docker-compose.\${DEPLOY_ENV}.yml" up -d "\${APP_NAME}"

# --- Health check ---
echo "==> Waiting for health check..."
for i in {1..30}; do
    if docker inspect --format='{{.State.Health.Status}}' "\${APP_NAME}" 2>/dev/null | grep -q "healthy"; then
        echo "    Health check passed on attempt \${i}"
        break
    fi
    if [ "\${i}" -eq 30 ]; then
        echo "ERROR: Health check failed after 30 attempts"
        docker logs "\${APP_NAME}" --tail 50
        exit 1
    fi
    sleep 2
done

echo "==> Deployment of \${APP_NAME} to \${DEPLOY_ENV} complete."`,
              },
              {
                language: 'python',
                title: 'Inventory Pipeline Configuration Loader',
                description:
                  'Environment-based configuration loader with secrets management and connection pool setup for the inventory pipeline.',
                code: `import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    min_pool_size: int = 4
    max_pool_size: int = 12

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class KafkaConfig:
    brokers: str
    topic: str
    group_id: str
    auto_offset_reset: str = "earliest"


@dataclass(frozen=True)
class AlertConfig:
    slack_webhook_url: str
    staleness_threshold_seconds: int = 30
    negative_atp_alert: bool = True
    alert_channel: str = "#inventory-alerts"


@dataclass(frozen=True)
class AppConfig:
    env: str
    db: DatabaseConfig
    kafka: KafkaConfig
    alerts: AlertConfig
    log_level: str = "INFO"


def _require_env(key: str) -> str:
    """Fetch a required environment variable or raise."""
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(f"Required env var {key} is not set")
    return value


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    return AppConfig(
        env=_require_env("DEPLOY_ENV"),
        db=DatabaseConfig(
            host=_require_env("DB_HOST"),
            port=int(os.environ.get("DB_PORT", "5432")),
            database=os.environ.get("DB_NAME", "inventory"),
            user=_require_env("DB_USER"),
            password=_require_env("DB_PASSWORD"),
            min_pool_size=int(os.environ.get("DB_POOL_MIN", "4")),
            max_pool_size=int(os.environ.get("DB_POOL_MAX", "12")),
        ),
        kafka=KafkaConfig(
            brokers=_require_env("KAFKA_BROKERS"),
            topic=os.environ.get("KAFKA_TOPIC", "wms.inventory.movements"),
            group_id=os.environ.get("KAFKA_GROUP_ID", "inventory-atp-writer"),
        ),
        alerts=AlertConfig(
            slack_webhook_url=_require_env("SLACK_WEBHOOK_URL"),
            staleness_threshold_seconds=int(
                os.environ.get("ATP_STALENESS_THRESHOLD_SEC", "30")
            ),
            negative_atp_alert=os.environ.get(
                "NEGATIVE_ATP_ALERT", "true"
            ).lower() == "true",
        ),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )


if __name__ == "__main__":
    config = load_config()
    print(f"Environment: {config.env}")
    print(f"Database:    {config.db.host}:{config.db.port}/{config.db.database}")
    print(f"Kafka:       {config.kafka.brokers} -> {config.kafka.topic}")
    print(f"Alerts:      {config.alerts.alert_channel}")`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Production monitoring that reconciles event-driven ATP against the WMS nightly snapshot, tracks inventory freshness, and sends Slack alerts when drift or negative ATP anomalies are detected.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'ATP Drift Reconciliation Report',
                description:
                  'Compares ATP computed from events against the latest WMS snapshot to detect synchronization drift.',
                code: `-- Reconcile event-driven ATP against WMS nightly snapshot
WITH event_atp AS (
    SELECT
        warehouse_id,
        sku,
        SUM(qty_delta) AS net_qty_from_events
    FROM inventory_events
    WHERE recorded_at >= CURRENT_DATE
    GROUP BY warehouse_id, sku
),
drift AS (
    SELECT
        w.warehouse_id,
        w.sku,
        w.wms_on_hand,
        COALESCE(e.net_qty_from_events, 0)  AS event_net,
        mv.atp_quantity                      AS current_atp,
        ABS(w.wms_on_hand + COALESCE(e.net_qty_from_events, 0)
            - mv.on_hand)                    AS drift_qty
    FROM wms_nightly_snapshot w
    LEFT JOIN event_atp e
        ON e.warehouse_id = w.warehouse_id AND e.sku = w.sku
    LEFT JOIN mv_inventory_atp mv
        ON mv.warehouse_id = w.warehouse_id AND mv.sku = w.sku
)
SELECT *
FROM drift
WHERE drift_qty > 0
ORDER BY drift_qty DESC
LIMIT 50;`,
              },
              {
                language: 'python',
                title: 'Inventory Slack Alerting with Threshold Monitoring',
                description:
                  'Monitors ATP freshness, drift, and negative-stock anomalies and sends Slack webhook alerts when thresholds are breached.',
                code: `import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Any

import asyncpg
import httpx

SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/T00/B00/xxxx"
DB_DSN = "postgresql://inv_svc:secret@db-host:5432/inventory"

STALENESS_THRESHOLD = timedelta(seconds=30)
DRIFT_THRESHOLD_QTY = 5
NEGATIVE_ATP_THRESHOLD = 0


async def send_slack_alert(title: str, details: list[dict[str, Any]],
                           severity: str = "warning") -> None:
    """Post a structured alert to Slack via incoming webhook."""
    color_map = {"info": "#36a64f", "warning": "#ff9900", "critical": "#ff0000"}
    fields = [
        {"title": d["label"], "value": str(d["value"]), "short": True}
        for d in details
    ]
    payload = {
        "attachments": [{
            "color": color_map.get(severity, "#ff9900"),
            "title": f":rotating_light: {title}",
            "fields": fields,
            "footer": "Inventory ATP Monitor",
            "ts": int(datetime.now(timezone.utc).timestamp()),
        }]
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(SLACK_WEBHOOK_URL, json=payload)
        resp.raise_for_status()


async def check_and_alert() -> dict[str, Any]:
    """Run all inventory health checks and fire Slack alerts on breaches."""
    pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=4)
    alerts_fired = []

    try:
        async with pool.acquire() as conn:
            # --- Freshness check ---
            staleness_row = await conn.fetchrow("""
                SELECT
                    MAX(snapshot_ts)                    AS latest_snapshot,
                    NOW() - MAX(snapshot_ts)            AS staleness
                FROM mv_inventory_atp
            """)
            staleness: timedelta = staleness_row["staleness"]
            if staleness > STALENESS_THRESHOLD:
                await send_slack_alert(
                    "ATP View Stale",
                    [
                        {"label": "Staleness", "value": f"{staleness.total_seconds():.0f}s"},
                        {"label": "Threshold", "value": f"{STALENESS_THRESHOLD.total_seconds():.0f}s"},
                        {"label": "Last Refresh", "value": str(staleness_row["latest_snapshot"])},
                    ],
                    severity="critical",
                )
                alerts_fired.append("staleness")

            # --- Negative ATP check ---
            neg_count = await conn.fetchval("""
                SELECT COUNT(*) FROM mv_inventory_atp WHERE atp_quantity < 0
            """)
            if neg_count > NEGATIVE_ATP_THRESHOLD:
                top_negatives = await conn.fetch("""
                    SELECT warehouse_id, sku, atp_quantity
                    FROM mv_inventory_atp
                    WHERE atp_quantity < 0
                    ORDER BY atp_quantity ASC
                    LIMIT 5
                """)
                details = [{"label": "Negative ATP SKUs", "value": neg_count}]
                for row in top_negatives:
                    details.append({
                        "label": f"{row['sku']} @ WH {row['warehouse_id']}",
                        "value": row["atp_quantity"],
                    })
                await send_slack_alert("Negative ATP Detected", details, severity="warning")
                alerts_fired.append("negative_atp")

            # --- Drift check ---
            drift_count = await conn.fetchval("""
                WITH event_atp AS (
                    SELECT warehouse_id, sku, SUM(qty_delta) AS net_qty
                    FROM inventory_events
                    WHERE recorded_at >= CURRENT_DATE
                    GROUP BY warehouse_id, sku
                )
                SELECT COUNT(*)
                FROM wms_nightly_snapshot w
                LEFT JOIN event_atp e
                    ON e.warehouse_id = w.warehouse_id AND e.sku = w.sku
                LEFT JOIN mv_inventory_atp mv
                    ON mv.warehouse_id = w.warehouse_id AND mv.sku = w.sku
                WHERE ABS(w.wms_on_hand + COALESCE(e.net_qty, 0)
                          - COALESCE(mv.on_hand, 0)) > $1
            """, DRIFT_THRESHOLD_QTY)
            if drift_count > 0:
                await send_slack_alert(
                    "Inventory Drift Detected",
                    [
                        {"label": "SKUs with drift", "value": drift_count},
                        {"label": "Drift threshold", "value": f"> {DRIFT_THRESHOLD_QTY} units"},
                    ],
                    severity="warning",
                )
                alerts_fired.append("drift")

    finally:
        await pool.close()

    return {"alerts_fired": alerts_fired, "checked_at": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    result = asyncio.run(check_and_alert())
    print(json.dumps(result, indent=2))`,
              },
            ],
          },
        ],
        toolsUsed: ['PostgreSQL', 'Python', 'Kafka', 'asyncpg', 'aiokafka', 'pytest', 'Docker', 'GitHub Actions', 'cron', 'Slack API', 'Prometheus'],
      },
    },

    /* ── Pain Point 2: Spreadsheet Shadow IT ──────────────────────────── */
    {
      id: 'spreadsheet-shadow-it',
      number: 2,
      title: 'Spreadsheet Shadow IT',
      subtitle: 'Critical Supply Chain Data in Excel Files',
      summary:
        'Your supply chain runs on 200+ Excel files emailed between teams. Version control is nonexistent and one wrong VLOOKUP costs $500K.',
      tags: ['spreadsheets', 'shadow-it', 'data-governance'],
      metrics: {
        annualCostRange: '$500K - $3M',
        roi: '6x',
        paybackPeriod: '4-6 months',
        investmentRange: '$100K - $200K',
      },
      price: {
        present: {
          title: 'Spreadsheet-Driven Supply Chain Operations',
          description:
            'Critical supply chain decisions depend on hundreds of Excel files with no versioning, no validation, and no audit trail.',
          bullets: [
            '200+ Excel files circulate weekly across procurement, logistics, and planning teams',
            'Key supplier scorecards and demand forecasts live only in personal spreadsheets',
            'Multiple conflicting versions of the same data emailed across departments',
          ],
          severity: 'critical',
        },
        root: {
          title: 'No Centralized Data Platform for Operational Teams',
          description:
            'Operational teams adopted spreadsheets because ERP reports are too rigid and IT backlogs prevent custom solutions.',
          bullets: [
            'ERP lacks self-service reporting so planners build their own in Excel',
            'IT ticket queues run 6-12 weeks for basic data requests',
            'No lightweight database layer exists between ERP and end-user analytics',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Data Errors Cascade Into Costly Decisions',
          description:
            'Formula errors and stale data in uncontrolled spreadsheets drive bad procurement, routing, and inventory decisions.',
          bullets: [
            'A single broken VLOOKUP in a procurement file caused a $500K duplicate order',
            'Demand forecast spreadsheets are 2 weeks stale by the time planning reviews them',
            'No audit trail — impossible to determine who changed a critical freight rate table',
          ],
          severity: 'critical',
        },
        cost: {
          title: 'Annual Cost of Spreadsheet Risk',
          description:
            'Spreadsheet errors and the labor to maintain them impose a significant hidden tax on supply chain operations.',
          bullets: [
            '$200K-$1M/yr in direct losses from formula errors and stale data decisions',
            '$150K-$400K/yr in FTE time spent manually maintaining and reconciling spreadsheets',
            '$100K-$500K/yr in compliance exposure from undocumented data lineage',
          ],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Centralized Data Model ROI',
          description:
            'Migrating spreadsheet data into a governed, centralized model eliminates error classes entirely and accelerates decision cycles.',
          bullets: [
            'Eliminate formula-error losses — saving $200K-$1M in the first year',
            'Reclaim 15-20 hours per week of analyst time spent on spreadsheet wrangling',
            'Full audit trail satisfies SOX and customs compliance with zero manual effort',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Discover, catalog, and migrate critical supply chain spreadsheets into a centralized PostgreSQL data model with automated ingestion, validation, and self-service reporting.',
        prerequisites: [
          'Python 3.10+ with openpyxl and pandas',
          'PostgreSQL 14+ or equivalent relational database',
          'Access to shared drives / email attachments containing the spreadsheets',
          'pytest >= 7.0 with pytest-asyncio',
          'Docker and docker-compose for containerized deployment',
          'cron, Airflow, or Prefect for job scheduling',
          'Slack incoming webhook URL for operational alerts',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Discover and Catalog Spreadsheet Assets',
            description:
              'Scan shared drives and mailboxes to build a complete inventory of supply chain spreadsheets, their owners, refresh frequency, and downstream consumers.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Spreadsheet Discovery Scanner',
                description:
                  'Crawls shared drive paths and catalogs every Excel file with metadata for triage and migration prioritization.',
                code: `import os
import hashlib
from pathlib import Path
from datetime import datetime
from openpyxl import load_workbook
import pandas as pd

SCAN_ROOTS = [
    Path("/mnt/shared/supply-chain"),
    Path("/mnt/shared/procurement"),
    Path("/mnt/shared/logistics"),
]

def scan_spreadsheets(roots: list[Path]) -> pd.DataFrame:
    """Discover all Excel files and extract structural metadata."""
    records = []
    for root in roots:
        for xlsx_path in root.rglob("*.xlsx"):
            stat = xlsx_path.stat()
            wb = load_workbook(xlsx_path, read_only=True, data_only=True)
            file_hash = hashlib.sha256(xlsx_path.read_bytes()).hexdigest()[:16]
            records.append({
                "file_path": str(xlsx_path),
                "file_name": xlsx_path.name,
                "department": xlsx_path.parts[3] if len(xlsx_path.parts) > 3 else "unknown",
                "size_kb": round(stat.st_size / 1024, 1),
                "last_modified": datetime.fromtimestamp(stat.st_mtime),
                "sheet_count": len(wb.sheetnames),
                "sheets": ", ".join(wb.sheetnames),
                "content_hash": file_hash,
            })
            wb.close()

    catalog = pd.DataFrame(records)
    catalog.sort_values("last_modified", ascending=False, inplace=True)
    catalog.to_csv("spreadsheet_catalog.csv", index=False)
    print(f"Cataloged {len(catalog)} spreadsheets across {len(roots)} roots")
    return catalog

if __name__ == "__main__":
    df = scan_spreadsheets(SCAN_ROOTS)
    print(df[["file_name", "department", "size_kb", "sheet_count"]].head(20))`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Migrate Spreadsheets to Centralized Data Model',
            description:
              'Parse the highest-priority spreadsheets, validate their contents, and load them into normalized PostgreSQL tables with full lineage tracking.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Spreadsheet Migration Pipeline',
                description:
                  'Reads Excel files, validates schemas and data quality, and upserts rows into PostgreSQL with lineage metadata.',
                code: `import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from datetime import datetime

DB_CONN = "host=db-host dbname=supplychain user=etl_svc password=secret"

SCHEMA_RULES = {
    "supplier_scorecard": {
        "required_cols": ["supplier_id", "supplier_name", "lead_time_days",
                          "quality_score", "on_time_pct"],
        "types": {"lead_time_days": "int", "quality_score": "float",
                  "on_time_pct": "float"},
    },
}

def validate_and_load(file_path: Path, table: str) -> dict:
    """Validate a spreadsheet against schema rules and load into Postgres."""
    rules = SCHEMA_RULES[table]
    df = pd.read_excel(file_path, engine="openpyxl")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = set(rules["required_cols"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col, dtype in rules["types"].items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    pre_count = len(df)
    df.dropna(subset=rules["required_cols"], inplace=True)

    df["_source_file"] = str(file_path)
    df["_loaded_at"] = datetime.utcnow()

    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor()
    cols = list(df.columns)
    sql = f"""
        INSERT INTO sc.{table} ({', '.join(cols)})
        VALUES %s
        ON CONFLICT (supplier_id) DO UPDATE SET
            {', '.join(f'{c}=EXCLUDED.{c}' for c in cols if c != 'supplier_id')}
    """
    execute_values(cur, sql, df.values.tolist())
    conn.commit()
    cur.close()
    conn.close()
    return {"rows_loaded": len(df), "rows_dropped": pre_count - len(df)}

if __name__ == "__main__":
    result = validate_and_load(
        Path("/mnt/shared/procurement/supplier_scorecard_2025.xlsx"),
        "supplier_scorecard",
    )
    print(result)`,
              },
              {
                language: 'sql',
                title: 'Centralized Supplier Scorecard Table',
                description:
                  'Normalized table that replaces the supplier scorecard spreadsheet with proper constraints and lineage.',
                code: `-- Centralized supplier scorecard — replaces Excel files
CREATE SCHEMA IF NOT EXISTS sc;

CREATE TABLE sc.supplier_scorecard (
    supplier_id     VARCHAR(20) PRIMARY KEY,
    supplier_name   VARCHAR(200) NOT NULL,
    lead_time_days  INT NOT NULL CHECK (lead_time_days >= 0),
    quality_score   NUMERIC(5,2) CHECK (quality_score BETWEEN 0 AND 100),
    on_time_pct     NUMERIC(5,2) CHECK (on_time_pct BETWEEN 0 AND 100),
    risk_tier       VARCHAR(10) GENERATED ALWAYS AS (
        CASE
            WHEN quality_score >= 95 AND on_time_pct >= 98 THEN 'low'
            WHEN quality_score >= 85 AND on_time_pct >= 90 THEN 'medium'
            ELSE 'high'
        END
    ) STORED,
    _source_file    TEXT,
    _loaded_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scorecard_risk ON sc.supplier_scorecard (risk_tier);

-- Audit trigger for change tracking
CREATE TABLE sc.supplier_scorecard_audit (
    audit_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    supplier_id VARCHAR(20),
    changed_at  TIMESTAMPTZ DEFAULT NOW(),
    old_row     JSONB,
    new_row     JSONB
);

CREATE OR REPLACE FUNCTION sc.fn_scorecard_audit()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO sc.supplier_scorecard_audit (supplier_id, old_row, new_row)
    VALUES (NEW.supplier_id, to_jsonb(OLD), to_jsonb(NEW));
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_scorecard_audit
    BEFORE UPDATE ON sc.supplier_scorecard
    FOR EACH ROW EXECUTE FUNCTION sc.fn_scorecard_audit();`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Automated data quality assertions and pytest-based validation that verify migration completeness, row count reconciliation, and detect schema drift between spreadsheets and the centralized database.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Spreadsheet Migration Data Quality Assertions',
                description:
                  'SQL assertion queries that validate row counts, null checks, referential integrity, and freshness after migrating spreadsheets into the centralized data model.',
                code: `-- ============================================================
-- Spreadsheet Migration Data Quality Assertions
-- Run after each ingestion cycle
-- ============================================================

-- 1. Row count reconciliation: loaded rows must match source
SELECT
    'row_count_reconciliation' AS assertion,
    CASE
        WHEN ABS(db.cnt - src.expected_rows) <= 1
        THEN 'PASS'
        ELSE 'FAIL: DB has ' || db.cnt || ' rows vs ' || src.expected_rows || ' expected'
    END AS result
FROM (SELECT COUNT(*) AS cnt FROM sc.supplier_scorecard) db
CROSS JOIN (
    SELECT value::int AS expected_rows
    FROM sc.ingestion_metadata
    WHERE table_name = 'supplier_scorecard'
      AND key = 'last_source_row_count'
    ORDER BY recorded_at DESC LIMIT 1
) src;

-- 2. No NULL required fields after migration
SELECT
    'no_null_required_fields' AS assertion,
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL: ' || COUNT(*) || ' rows with NULL in required columns'
    END AS result
FROM sc.supplier_scorecard
WHERE supplier_id IS NULL
   OR supplier_name IS NULL
   OR lead_time_days IS NULL;

-- 3. Referential integrity: all supplier_ids referenced in orders exist
SELECT
    'supplier_fk_integrity' AS assertion,
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL: ' || COUNT(*) || ' orders reference non-existent suppliers'
    END AS result
FROM sc.purchase_orders po
LEFT JOIN sc.supplier_scorecard ss ON ss.supplier_id = po.supplier_id
WHERE ss.supplier_id IS NULL;

-- 4. Schema drift: check that expected columns still exist
SELECT
    'schema_columns_present' AS assertion,
    CASE
        WHEN COUNT(*) = 5 THEN 'PASS'
        ELSE 'FAIL: expected 5 core columns, found ' || COUNT(*)
    END AS result
FROM information_schema.columns
WHERE table_schema = 'sc'
  AND table_name = 'supplier_scorecard'
  AND column_name IN (
      'supplier_id', 'supplier_name', 'lead_time_days',
      'quality_score', 'on_time_pct'
  );

-- 5. Data freshness: last load must be within 24 hours
SELECT
    'load_freshness' AS assertion,
    CASE
        WHEN MAX(_loaded_at) >= NOW() - INTERVAL '24 hours'
        THEN 'PASS'
        ELSE 'FAIL: last load was ' ||
             ROUND(EXTRACT(EPOCH FROM NOW() - MAX(_loaded_at)) / 3600)
             || ' hours ago'
    END AS result
FROM sc.supplier_scorecard;`,
              },
              {
                language: 'python',
                title: 'Pytest Migration Validation Suite',
                description:
                  'pytest-based tests that validate migration completeness, row count reconciliation, schema drift, and data quality across migrated spreadsheet tables.',
                code: `import pytest
import psycopg2
import psycopg2.extras
from pathlib import Path

DB_CONN = "host=db-host dbname=supplychain user=etl_svc password=secret"
EXPECTED_CORE_COLUMNS = {
    "supplier_id", "supplier_name", "lead_time_days",
    "quality_score", "on_time_pct",
}


@pytest.fixture(scope="module")
def db_cursor():
    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    yield cur
    cur.close()
    conn.close()


def test_row_count_matches_source(db_cursor):
    """Migrated row count must match the source spreadsheet row count."""
    db_cursor.execute("SELECT COUNT(*) AS cnt FROM sc.supplier_scorecard")
    db_count = db_cursor.fetchone()["cnt"]

    db_cursor.execute("""
        SELECT value::int AS expected
        FROM sc.ingestion_metadata
        WHERE table_name = 'supplier_scorecard'
          AND key = 'last_source_row_count'
        ORDER BY recorded_at DESC LIMIT 1
    """)
    row = db_cursor.fetchone()
    assert row is not None, "No source row count recorded in ingestion_metadata"
    assert abs(db_count - row["expected"]) <= 1, (
        f"DB has {db_count} rows vs {row['expected']} expected from source"
    )


def test_no_null_required_columns(db_cursor):
    """Required columns must not contain NULLs after migration."""
    for col in ["supplier_id", "supplier_name", "lead_time_days"]:
        db_cursor.execute(
            f"SELECT COUNT(*) AS cnt FROM sc.supplier_scorecard WHERE {col} IS NULL"
        )
        nulls = db_cursor.fetchone()["cnt"]
        assert nulls == 0, f"{nulls} NULL values found in column {col}"


def test_schema_drift_detection(db_cursor):
    """All expected core columns must exist in the target table."""
    db_cursor.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'sc' AND table_name = 'supplier_scorecard'
    """)
    actual_columns = {row["column_name"] for row in db_cursor.fetchall()}
    missing = EXPECTED_CORE_COLUMNS - actual_columns
    assert not missing, f"Schema drift detected — missing columns: {missing}"


def test_quality_score_bounds(db_cursor):
    """Quality scores must be between 0 and 100."""
    db_cursor.execute("""
        SELECT COUNT(*) AS cnt FROM sc.supplier_scorecard
        WHERE quality_score < 0 OR quality_score > 100
    """)
    out_of_bounds = db_cursor.fetchone()["cnt"]
    assert out_of_bounds == 0, (
        f"{out_of_bounds} rows have quality_score outside [0, 100]"
    )


def test_no_duplicate_supplier_ids(db_cursor):
    """Supplier IDs must be unique (no duplicate rows from re-ingestion)."""
    db_cursor.execute("""
        SELECT supplier_id, COUNT(*) AS cnt
        FROM sc.supplier_scorecard
        GROUP BY supplier_id HAVING COUNT(*) > 1
    """)
    dupes = db_cursor.fetchall()
    assert len(dupes) == 0, (
        f"Found {len(dupes)} duplicate supplier_ids: "
        f"{[d['supplier_id'] for d in dupes[:5]]}"
    )`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Production deployment automation for the spreadsheet ingestion pipeline: environment validation, dependency setup, database migration, file-watcher cron scheduling, and health monitoring.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Spreadsheet Ingestion Deployment Script',
                description:
                  'Deployment script that validates the environment, installs dependencies, runs DB migrations, and configures the file-watcher cron job for automated ingestion.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Spreadsheet Ingestion Pipeline — Deployment Script
# ============================================================

APP_NAME="spreadsheet-ingestion"
DEPLOY_ENV="\${DEPLOY_ENV:?ERROR: DEPLOY_ENV must be set (staging|production)}"
DB_HOST="\${DB_HOST:?ERROR: DB_HOST must be set}"
WATCH_DIR="\${WATCH_DIR:?ERROR: WATCH_DIR must be set}"
SLACK_WEBHOOK="\${SLACK_WEBHOOK:?ERROR: SLACK_WEBHOOK must be set}"

echo "==> Deploying \${APP_NAME} to \${DEPLOY_ENV}"

# --- Pre-flight checks ---
echo "==> Running pre-flight checks..."
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
command -v psql >/dev/null 2>&1 || { echo "ERROR: psql not found"; exit 1; }

# Verify watch directory exists
[ -d "\${WATCH_DIR}" ] || { echo "ERROR: Watch directory \${WATCH_DIR} does not exist"; exit 1; }
echo "    Watch directory: OK (\${WATCH_DIR})"

# Verify database connectivity
psql "host=\${DB_HOST} dbname=supplychain user=etl_svc" -c "SELECT 1;" >/dev/null 2>&1 \\
    || { echo "ERROR: Cannot connect to database at \${DB_HOST}"; exit 1; }
echo "    Database connectivity: OK"

# --- Install Python dependencies ---
echo "==> Installing Python dependencies..."
pip install --quiet --upgrade -r requirements.txt

# --- Run database migrations ---
echo "==> Running database migrations..."
psql "host=\${DB_HOST} dbname=supplychain user=etl_svc" -f migrations/001_create_sc_schema.sql
psql "host=\${DB_HOST} dbname=supplychain user=etl_svc" -f migrations/002_create_supplier_scorecard.sql
psql "host=\${DB_HOST} dbname=supplychain user=etl_svc" -f migrations/003_create_audit_triggers.sql
psql "host=\${DB_HOST} dbname=supplychain user=etl_svc" -f migrations/004_create_ingestion_metadata.sql
echo "    Migrations applied successfully"

# --- Build and deploy container ---
echo "==> Building Docker image..."
docker build -t "\${APP_NAME}:latest" .

echo "==> Deploying container..."
docker-compose -f "docker-compose.\${DEPLOY_ENV}.yml" down --remove-orphans || true
docker-compose -f "docker-compose.\${DEPLOY_ENV}.yml" up -d "\${APP_NAME}"

# --- Set up file-watcher cron ---
echo "==> Configuring cron schedule..."
CRON_ENTRY="*/15 * * * * cd /opt/\${APP_NAME} && python3 ingest_runner.py >> /var/log/\${APP_NAME}.log 2>&1"
(crontab -l 2>/dev/null | grep -v "\${APP_NAME}"; echo "\${CRON_ENTRY}") | crontab -
echo "    Cron job installed (every 15 minutes)"

# --- Verify deployment ---
echo "==> Verifying deployment..."
sleep 3
if docker ps --filter "name=\${APP_NAME}" --format '{{.Status}}' | grep -q "Up"; then
    echo "    Container is running"
else
    echo "ERROR: Container failed to start"
    docker logs "\${APP_NAME}" --tail 30
    exit 1
fi

echo "==> Deployment of \${APP_NAME} to \${DEPLOY_ENV} complete."`,
              },
              {
                language: 'python',
                title: 'Spreadsheet Pipeline Configuration Loader',
                description:
                  'Environment-based configuration loader with secrets management and connection setup for the spreadsheet ingestion pipeline.',
                code: `import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str

    @property
    def conn_string(self) -> str:
        return (
            f"host={self.host} port={self.port} "
            f"dbname={self.database} user={self.user} password={self.password}"
        )


@dataclass(frozen=True)
class IngestionConfig:
    watch_dir: Path
    archive_dir: Path
    scan_interval_minutes: int = 15
    max_file_size_mb: int = 50
    allowed_extensions: tuple = (".xlsx", ".xls", ".csv")


@dataclass(frozen=True)
class AlertConfig:
    slack_webhook_url: str
    alert_channel: str = "#data-ingestion"
    alert_on_schema_drift: bool = True
    alert_on_load_failure: bool = True


@dataclass(frozen=True)
class AppConfig:
    env: str
    db: DatabaseConfig
    ingestion: IngestionConfig
    alerts: AlertConfig
    log_level: str = "INFO"


def _require_env(key: str) -> str:
    """Fetch a required environment variable or raise."""
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(f"Required env var {key} is not set")
    return value


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    watch = Path(_require_env("WATCH_DIR"))
    archive = watch.parent / "archive"

    return AppConfig(
        env=_require_env("DEPLOY_ENV"),
        db=DatabaseConfig(
            host=_require_env("DB_HOST"),
            port=int(os.environ.get("DB_PORT", "5432")),
            database=os.environ.get("DB_NAME", "supplychain"),
            user=_require_env("DB_USER"),
            password=_require_env("DB_PASSWORD"),
        ),
        ingestion=IngestionConfig(
            watch_dir=watch,
            archive_dir=Path(os.environ.get("ARCHIVE_DIR", str(archive))),
            scan_interval_minutes=int(
                os.environ.get("SCAN_INTERVAL_MINUTES", "15")
            ),
            max_file_size_mb=int(os.environ.get("MAX_FILE_SIZE_MB", "50")),
        ),
        alerts=AlertConfig(
            slack_webhook_url=_require_env("SLACK_WEBHOOK"),
            alert_channel=os.environ.get(
                "ALERT_CHANNEL", "#data-ingestion"
            ),
        ),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )


if __name__ == "__main__":
    config = load_config()
    print(f"Environment:  {config.env}")
    print(f"Database:     {config.db.host}:{config.db.port}/{config.db.database}")
    print(f"Watch dir:    {config.ingestion.watch_dir}")
    print(f"Archive dir:  {config.ingestion.archive_dir}")
    print(f"Scan every:   {config.ingestion.scan_interval_minutes} minutes")`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Production monitoring that tracks file processing health, data quality trends, ingestion success rates, and sends Slack alerts when sync failures or schema drift are detected.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Ingestion Health & Data Quality Dashboard',
                description:
                  'Queries that track file processing success rates, row count trends, and data quality metrics across all ingested spreadsheet tables.',
                code: `-- ============================================================
-- Spreadsheet Ingestion Health Dashboard Queries
-- ============================================================

-- 1. Ingestion success/failure rates over the last 7 days
SELECT
    DATE(recorded_at) AS ingestion_date,
    COUNT(*) FILTER (WHERE status = 'success')  AS successful,
    COUNT(*) FILTER (WHERE status = 'failed')   AS failed,
    COUNT(*) FILTER (WHERE status = 'skipped')  AS skipped,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE status = 'success') / NULLIF(COUNT(*), 0), 1
    ) AS success_rate_pct
FROM sc.ingestion_log
WHERE recorded_at >= NOW() - INTERVAL '7 days'
GROUP BY DATE(recorded_at)
ORDER BY ingestion_date DESC;

-- 2. Row count trend per table — detect unexpected drops
SELECT
    table_name,
    DATE(recorded_at) AS load_date,
    value::int AS row_count,
    LAG(value::int) OVER (
        PARTITION BY table_name ORDER BY recorded_at
    ) AS prev_row_count,
    value::int - LAG(value::int) OVER (
        PARTITION BY table_name ORDER BY recorded_at
    ) AS row_delta
FROM sc.ingestion_metadata
WHERE key = 'last_source_row_count'
ORDER BY table_name, recorded_at DESC;

-- 3. Files pending in watch directory longer than 1 hour
SELECT
    file_name,
    detected_at,
    NOW() - detected_at AS age,
    status,
    error_message
FROM sc.file_tracking
WHERE status IN ('pending', 'failed')
  AND detected_at < NOW() - INTERVAL '1 hour'
ORDER BY detected_at ASC;`,
              },
              {
                language: 'python',
                title: 'Spreadsheet Ingestion Slack Alerting',
                description:
                  'Monitors ingestion health, detects sync failures and schema drift, and sends Slack webhook alerts when thresholds are breached.',
                code: `import json
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras
import requests

SLACK_WEBHOOK = "https://hooks.slack.com/services/T00/B00/xxxx"
DB_CONN = "host=db-host dbname=supplychain user=etl_svc password=secret"

FAILURE_RATE_THRESHOLD = 0.10  # Alert if > 10% failure rate
ROW_DROP_THRESHOLD = 0.20      # Alert if row count drops > 20%


def send_slack_alert(title: str, details: list[dict[str, Any]],
                     severity: str = "warning") -> None:
    """Post a structured alert to Slack via incoming webhook."""
    color_map = {"info": "#36a64f", "warning": "#ff9900", "critical": "#ff0000"}
    fields = [
        {"title": d["label"], "value": str(d["value"]), "short": True}
        for d in details
    ]
    payload = {
        "attachments": [{
            "color": color_map.get(severity, "#ff9900"),
            "title": f":rotating_light: {title}",
            "fields": fields,
            "footer": "Spreadsheet Ingestion Monitor",
            "ts": int(datetime.now(timezone.utc).timestamp()),
        }]
    }
    requests.post(SLACK_WEBHOOK, json=payload)


def check_and_alert() -> dict[str, Any]:
    """Run ingestion health checks and fire Slack alerts on breaches."""
    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    alerts_fired = []

    # --- Failure rate check ---
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE status = 'failed') AS failures,
            COUNT(*) AS total
        FROM sc.ingestion_log
        WHERE recorded_at >= NOW() - INTERVAL '24 hours'
    """)
    row = cur.fetchone()
    if row["total"] > 0:
        failure_rate = row["failures"] / row["total"]
        if failure_rate > FAILURE_RATE_THRESHOLD:
            send_slack_alert(
                "High Ingestion Failure Rate",
                [
                    {"label": "Failure rate", "value": f"{failure_rate:.1%}"},
                    {"label": "Failed files", "value": row["failures"]},
                    {"label": "Total processed", "value": row["total"]},
                ],
                severity="critical",
            )
            alerts_fired.append("high_failure_rate")

    # --- Row count drop check ---
    cur.execute("""
        WITH recent AS (
            SELECT table_name, value::int AS row_count,
                   LAG(value::int) OVER (
                       PARTITION BY table_name ORDER BY recorded_at
                   ) AS prev_count
            FROM sc.ingestion_metadata
            WHERE key = 'last_source_row_count'
        )
        SELECT table_name, row_count, prev_count
        FROM recent
        WHERE prev_count IS NOT NULL
          AND prev_count > 0
          AND (prev_count - row_count)::float / prev_count > %s
    """, (ROW_DROP_THRESHOLD,))
    drops = cur.fetchall()
    for drop in drops:
        send_slack_alert(
            f"Row Count Drop: {drop['table_name']}",
            [
                {"label": "Current rows", "value": drop["row_count"]},
                {"label": "Previous rows", "value": drop["prev_count"]},
                {"label": "Drop %", "value": f"{(drop['prev_count'] - drop['row_count']) / drop['prev_count']:.1%}"},
            ],
            severity="warning",
        )
        alerts_fired.append(f"row_drop_{drop['table_name']}")

    # --- Stuck files check ---
    cur.execute("""
        SELECT COUNT(*) AS cnt
        FROM sc.file_tracking
        WHERE status IN ('pending', 'failed')
          AND detected_at < NOW() - INTERVAL '1 hour'
    """)
    stuck = cur.fetchone()["cnt"]
    if stuck > 0:
        send_slack_alert(
            "Stuck Files in Watch Directory",
            [{"label": "Files stuck > 1hr", "value": stuck}],
            severity="warning",
        )
        alerts_fired.append("stuck_files")

    cur.close()
    conn.close()
    return {"alerts_fired": alerts_fired, "checked_at": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    result = check_and_alert()
    print(json.dumps(result, indent=2))`,
              },
            ],
          },
        ],
        toolsUsed: ['PostgreSQL', 'Python', 'pandas', 'openpyxl', 'psycopg2', 'pytest', 'Docker', 'GitHub Actions', 'cron', 'Slack API', 'Prometheus'],
      },
    },

    /* ── Pain Point 3: Import Customs Friction ────────────────────────── */
    {
      id: 'import-customs-friction',
      number: 3,
      title: 'Import Customs Friction',
      subtitle: 'Manual HS Code Classification & Customs Documentation',
      summary:
        'Manual customs documentation causes 3-5 day delays per shipment. Misclassified HS codes trigger audits and penalties.',
      tags: ['customs', 'import', 'automation'],
      metrics: {
        annualCostRange: '$300K - $1.5M',
        roi: '5x',
        paybackPeriod: '3-4 months',
        investmentRange: '$70K - $130K',
      },
      price: {
        present: {
          title: 'Manual Customs Classification & Documentation',
          description:
            'Customs brokers manually classify HS codes and prepare documentation for every shipment, creating bottlenecks and error-prone paperwork.',
          bullets: [
            'Each import shipment requires 2-4 hours of manual HS code lookup and form preparation',
            'Customs brokers rely on tribal knowledge — no systematized classification logic',
            'Documentation errors are caught at the border, adding 3-5 days of clearance delays',
          ],
          severity: 'high',
        },
        root: {
          title: 'No Automated Classification or Document Generation',
          description:
            'Product catalogs lack standardized HS code mappings and there is no system to auto-generate compliant customs forms.',
          bullets: [
            'Product master data does not include HS codes — brokers classify from scratch each time',
            'Customs form templates are filled manually in Word/PDF with copy-paste errors',
            'No feedback loop from customs rejections back into classification accuracy',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Shipment Delays, Penalties & Audit Exposure',
          description:
            'Misclassification and paperwork errors create a recurring cycle of delays, fines, and regulatory scrutiny.',
          bullets: [
            '15-20% of shipments are delayed 3-5 business days due to documentation issues',
            'HS code misclassification triggers duty underpayment penalties averaging $25K per audit',
            'Customs audit frequency increases with each misclassification — compounding future risk',
          ],
          severity: 'critical',
        },
        cost: {
          title: 'Financial Impact of Customs Friction',
          description:
            'The combined cost of delays, penalties, manual labor, and demurrage charges adds up rapidly for high-volume importers.',
          bullets: [
            '$100K-$500K/yr in customs penalties and duty adjustments from misclassification',
            '$80K-$300K/yr in demurrage and detention charges from clearance delays',
            '$60K-$200K/yr in broker labor that could be automated',
          ],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Automated Customs Automation ROI',
          description:
            'ML-driven HS code classification and automated document generation slash clearance times and eliminate recurring penalty exposure.',
          bullets: [
            'Reduce average clearance time from 5 days to 1.5 days',
            'Cut misclassification rate from 12% to under 2% with ML-assisted coding',
            'Automate 80% of customs document preparation — freeing broker capacity for exceptions',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Deploy an ML-powered HS code classification service backed by a product-to-tariff mapping database, paired with automated customs document generation that pre-fills and validates forms before submission.',
        prerequisites: [
          'Python 3.10+ with scikit-learn and pandas',
          'PostgreSQL 14+ for product and tariff data',
          'Historical shipment and HS code data for model training',
          'pytest >= 7.0 with pytest-asyncio',
          'Docker and docker-compose for containerized deployment',
          'cron, Airflow, or Prefect for job scheduling',
          'Slack incoming webhook URL for operational alerts',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Build the HS Code Classification Model',
            description:
              'Train a text classifier on historical product descriptions and their correct HS codes to automate the classification step.',
            codeSnippets: [
              {
                language: 'python',
                title: 'HS Code ML Classifier',
                description:
                  'TF-IDF + gradient boosting classifier that maps product descriptions to 6-digit HS codes with confidence scores.',
                code: `import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

def train_hs_classifier(training_csv: str) -> Pipeline:
    """Train an HS code classifier from historical shipment data."""
    df = pd.read_csv(training_csv)
    df["text"] = (
        df["product_description"].str.lower() + " " +
        df["material"].fillna("").str.lower() + " " +
        df["country_of_origin"].fillna("").str.lower()
    )
    df = df.dropna(subset=["hs_code_6"])

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            sublinear_tf=True,
        )),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )),
    ])

    scores = cross_val_score(pipeline, df["text"], df["hs_code_6"], cv=5, scoring="accuracy")
    print(f"Cross-val accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    pipeline.fit(df["text"], df["hs_code_6"])
    joblib.dump(pipeline, "hs_classifier_v1.joblib")
    print("Model saved to hs_classifier_v1.joblib")
    return pipeline

def classify_product(pipeline: Pipeline, description: str,
                     material: str = "", origin: str = "") -> dict:
    """Classify a single product and return top-3 predictions with confidence."""
    text = f"{description.lower()} {material.lower()} {origin.lower()}"
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    top_3_idx = proba.argsort()[-3:][::-1]
    return {
        "predictions": [
            {"hs_code": classes[i], "confidence": round(float(proba[i]), 4)}
            for i in top_3_idx
        ],
        "needs_review": proba[top_3_idx[0]] < 0.85,
    }

if __name__ == "__main__":
    model = train_hs_classifier("historical_shipments.csv")
    result = classify_product(model, "Stainless steel hex bolts M10", "steel", "DE")
    print(result)`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Create the Customs Document Automation Layer',
            description:
              'SQL-driven document assembly that joins shipment, product, and tariff data to auto-generate compliant customs entry forms.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Customs Entry Data Assembly',
                description:
                  'Query that assembles all data required for a customs entry form by joining shipment, product, and tariff tables.',
                code: `-- Assemble customs entry data for a given shipment
CREATE OR REPLACE VIEW v_customs_entry_data AS
SELECT
    s.shipment_id,
    s.bill_of_lading,
    s.vessel_name,
    s.port_of_entry,
    s.estimated_arrival,
    li.line_number,
    p.product_description,
    li.quantity,
    li.unit_of_measure,
    li.declared_value_usd,
    hc.hs_code_6,
    hc.hs_code_10,
    hc.classification_confidence,
    t.duty_rate_pct,
    ROUND(li.declared_value_usd * t.duty_rate_pct / 100, 2)  AS estimated_duty,
    t.requires_license,
    t.anti_dumping_flag,
    sup.supplier_name,
    sup.country_of_origin,
    CASE
        WHEN hc.classification_confidence < 0.85 THEN 'MANUAL_REVIEW'
        WHEN t.anti_dumping_flag THEN 'AD_CVD_REVIEW'
        ELSE 'AUTO_CLEAR'
    END AS routing_status
FROM shipments s
JOIN shipment_line_items li ON li.shipment_id = s.shipment_id
JOIN products p             ON p.product_id = li.product_id
JOIN hs_classifications hc  ON hc.product_id = p.product_id
JOIN tariff_schedule t      ON t.hs_code_10 = hc.hs_code_10
JOIN suppliers sup          ON sup.supplier_id = li.supplier_id
WHERE s.status = 'pre_clearance';`,
              },
              {
                language: 'sql',
                title: 'Classification Feedback Loop',
                description:
                  'Table and queries that capture customs rejection feedback to continuously improve the ML classifier.',
                code: `-- Track customs classification outcomes for model retraining
CREATE TABLE customs_classification_feedback (
    feedback_id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    shipment_id       VARCHAR(40) NOT NULL,
    product_id        VARCHAR(40) NOT NULL,
    predicted_hs_code VARCHAR(10) NOT NULL,
    actual_hs_code    VARCHAR(10),
    was_accepted      BOOLEAN NOT NULL DEFAULT TRUE,
    rejection_reason  TEXT,
    customs_officer   VARCHAR(100),
    recorded_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feedback_product
    ON customs_classification_feedback (product_id, recorded_at DESC);

-- Misclassification report for model retraining
SELECT
    predicted_hs_code,
    actual_hs_code,
    COUNT(*)                                         AS occurrences,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_errors
FROM customs_classification_feedback
WHERE NOT was_accepted
  AND actual_hs_code IS NOT NULL
GROUP BY predicted_hs_code, actual_hs_code
ORDER BY occurrences DESC
LIMIT 20;`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Automated data quality assertions and pytest-based validation for HS code classification accuracy, confidence threshold enforcement, and customs document completeness.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Customs Classification Data Quality Assertions',
                description:
                  'SQL assertion queries that validate classification coverage, confidence thresholds, referential integrity against the tariff schedule, and feedback loop freshness.',
                code: `-- ============================================================
-- Customs Classification Data Quality Assertions
-- Run after each classification batch
-- ============================================================

-- 1. Classification coverage: all active products must have an HS code
SELECT
    'classification_coverage' AS assertion,
    CASE
        WHEN unclassified = 0 THEN 'PASS'
        ELSE 'FAIL: ' || unclassified || ' active products lack HS classification'
    END AS result
FROM (
    SELECT COUNT(*) AS unclassified
    FROM products p
    LEFT JOIN hs_classifications hc ON hc.product_id = p.product_id
    WHERE p.is_active AND hc.product_id IS NULL
) sub;

-- 2. Confidence threshold: no auto-cleared shipment should have confidence < 0.85
SELECT
    'confidence_threshold' AS assertion,
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL: ' || COUNT(*) || ' auto-cleared items below 0.85 confidence'
    END AS result
FROM v_customs_entry_data
WHERE routing_status = 'AUTO_CLEAR'
  AND classification_confidence < 0.85;

-- 3. Tariff schedule integrity: every classified HS code must exist in schedule
SELECT
    'tariff_fk_integrity' AS assertion,
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL: ' || COUNT(*) || ' classifications reference missing tariff entries'
    END AS result
FROM hs_classifications hc
LEFT JOIN tariff_schedule t ON t.hs_code_10 = hc.hs_code_10
WHERE t.hs_code_10 IS NULL;

-- 4. Feedback loop freshness: recent rejections must be captured
SELECT
    'feedback_freshness' AS assertion,
    CASE
        WHEN MAX(recorded_at) >= NOW() - INTERVAL '7 days'
        THEN 'PASS'
        ELSE 'FAIL: no classification feedback recorded in 7 days'
    END AS result
FROM customs_classification_feedback;

-- 5. Misclassification rate must be under 5%
SELECT
    'misclassification_rate' AS assertion,
    CASE
        WHEN rejection_rate <= 0.05 THEN 'PASS'
        ELSE 'FAIL: misclassification rate is ' ||
             ROUND(rejection_rate * 100, 1) || '%'
    END AS result
FROM (
    SELECT
        COUNT(*) FILTER (WHERE NOT was_accepted)::float
        / NULLIF(COUNT(*), 0) AS rejection_rate
    FROM customs_classification_feedback
    WHERE recorded_at >= NOW() - INTERVAL '30 days'
) sub;`,
              },
              {
                language: 'python',
                title: 'Pytest Classification Validation Suite',
                description:
                  'pytest-based tests that validate HS code classification accuracy, confidence thresholds, and customs document completeness.',
                code: `import pytest
import joblib
import pandas as pd
import psycopg2
import psycopg2.extras

DB_CONN = "host=db-host dbname=supplychain user=customs_svc password=secret"
MODEL_PATH = "hs_classifier_v1.joblib"
CONFIDENCE_THRESHOLD = 0.85
ACCURACY_THRESHOLD = 0.90


@pytest.fixture(scope="module")
def db_cursor():
    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    yield cur
    cur.close()
    conn.close()


@pytest.fixture(scope="module")
def classifier():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def validation_data():
    return pd.read_csv("validation_shipments.csv")


def test_model_accuracy_above_threshold(classifier, validation_data):
    """Model accuracy on validation set must exceed the threshold."""
    validation_data["text"] = (
        validation_data["product_description"].str.lower() + " " +
        validation_data["material"].fillna("").str.lower() + " " +
        validation_data["country_of_origin"].fillna("").str.lower()
    )
    predictions = classifier.predict(validation_data["text"])
    accuracy = (predictions == validation_data["hs_code_6"]).mean()
    assert accuracy >= ACCURACY_THRESHOLD, (
        f"Model accuracy {accuracy:.3f} is below threshold {ACCURACY_THRESHOLD}"
    )


def test_confidence_scores_above_threshold(classifier, validation_data):
    """Top-1 confidence for known-good products must exceed threshold."""
    validation_data["text"] = (
        validation_data["product_description"].str.lower() + " " +
        validation_data["material"].fillna("").str.lower() + " " +
        validation_data["country_of_origin"].fillna("").str.lower()
    )
    probas = classifier.predict_proba(validation_data["text"])
    top_confidences = probas.max(axis=1)
    low_conf_count = (top_confidences < CONFIDENCE_THRESHOLD).sum()
    low_conf_pct = low_conf_count / len(top_confidences)
    assert low_conf_pct < 0.15, (
        f"{low_conf_pct:.1%} of predictions below {CONFIDENCE_THRESHOLD} confidence"
    )


def test_all_active_products_classified(db_cursor):
    """Every active product must have an HS classification."""
    db_cursor.execute("""
        SELECT COUNT(*) AS cnt
        FROM products p
        LEFT JOIN hs_classifications hc ON hc.product_id = p.product_id
        WHERE p.is_active AND hc.product_id IS NULL
    """)
    unclassified = db_cursor.fetchone()["cnt"]
    assert unclassified == 0, (
        f"{unclassified} active products lack HS classification"
    )


def test_tariff_schedule_referential_integrity(db_cursor):
    """Every HS code in classifications must exist in the tariff schedule."""
    db_cursor.execute("""
        SELECT COUNT(*) AS cnt
        FROM hs_classifications hc
        LEFT JOIN tariff_schedule t ON t.hs_code_10 = hc.hs_code_10
        WHERE t.hs_code_10 IS NULL
    """)
    orphans = db_cursor.fetchone()["cnt"]
    assert orphans == 0, (
        f"{orphans} classifications reference missing tariff entries"
    )


def test_no_auto_clear_below_confidence(db_cursor):
    """No auto-cleared shipment line should have confidence below threshold."""
    db_cursor.execute("""
        SELECT COUNT(*) AS cnt
        FROM v_customs_entry_data
        WHERE routing_status = 'AUTO_CLEAR'
          AND classification_confidence < %s
    """, (CONFIDENCE_THRESHOLD,))
    violations = db_cursor.fetchone()["cnt"]
    assert violations == 0, (
        f"{violations} auto-cleared items have confidence below {CONFIDENCE_THRESHOLD}"
    )`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Production deployment automation for the HS code classifier API and customs document generator: container builds, model artifact deployment, database migrations, and API health checks.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Customs Pipeline Deployment Script',
                description:
                  'Deployment script that validates the environment, installs dependencies, deploys the classifier API and document generator containers, and verifies health.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Customs Classification & Doc Gen — Deployment Script
# ============================================================

APP_CLASSIFIER="customs-classifier-api"
APP_DOCGEN="customs-doc-generator"
DEPLOY_ENV="\${DEPLOY_ENV:?ERROR: DEPLOY_ENV must be set (staging|production)}"
DB_HOST="\${DB_HOST:?ERROR: DB_HOST must be set}"
MODEL_PATH="\${MODEL_PATH:?ERROR: MODEL_PATH must be set (path to .joblib model)}"
SLACK_WEBHOOK="\${SLACK_WEBHOOK:?ERROR: SLACK_WEBHOOK must be set}"
IMAGE_TAG="\${IMAGE_TAG:-latest}"

echo "==> Deploying customs pipeline to \${DEPLOY_ENV} (tag: \${IMAGE_TAG})"

# --- Pre-flight checks ---
echo "==> Running pre-flight checks..."
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "ERROR: docker-compose not found"; exit 1; }
command -v psql >/dev/null 2>&1 || { echo "ERROR: psql not found"; exit 1; }

# Verify model artifact exists
[ -f "\${MODEL_PATH}" ] || { echo "ERROR: Model file not found at \${MODEL_PATH}"; exit 1; }
echo "    Model artifact: OK (\${MODEL_PATH})"

# Verify database connectivity
psql "host=\${DB_HOST} dbname=supplychain user=customs_svc" -c "SELECT 1;" >/dev/null 2>&1 \\
    || { echo "ERROR: Cannot connect to database at \${DB_HOST}"; exit 1; }
echo "    Database connectivity: OK"

# --- Install Python dependencies ---
echo "==> Installing Python dependencies..."
pip install --quiet --upgrade -r requirements.txt

# --- Run database migrations ---
echo "==> Running database migrations..."
psql "host=\${DB_HOST} dbname=supplychain user=customs_svc" -f migrations/001_create_hs_classifications.sql
psql "host=\${DB_HOST} dbname=supplychain user=customs_svc" -f migrations/002_create_tariff_schedule.sql
psql "host=\${DB_HOST} dbname=supplychain user=customs_svc" -f migrations/003_create_feedback_table.sql
psql "host=\${DB_HOST} dbname=supplychain user=customs_svc" -f migrations/004_create_entry_view.sql
echo "    Migrations applied successfully"

# --- Build containers ---
echo "==> Building Docker images..."
docker build -t "\${APP_CLASSIFIER}:\${IMAGE_TAG}" -f Dockerfile.classifier .
docker build -t "\${APP_DOCGEN}:\${IMAGE_TAG}" -f Dockerfile.docgen .

# --- Deploy ---
echo "==> Stopping existing containers..."
docker-compose -f "docker-compose.\${DEPLOY_ENV}.yml" down --remove-orphans || true

echo "==> Starting classifier API..."
docker-compose -f "docker-compose.\${DEPLOY_ENV}.yml" up -d "\${APP_CLASSIFIER}"

echo "==> Starting document generator..."
docker-compose -f "docker-compose.\${DEPLOY_ENV}.yml" up -d "\${APP_DOCGEN}"

# --- Health checks ---
echo "==> Running health checks..."
for app in "\${APP_CLASSIFIER}" "\${APP_DOCGEN}"; do
    for i in {1..30}; do
        if docker inspect --format='{{.State.Health.Status}}' "\${app}" 2>/dev/null | grep -q "healthy"; then
            echo "    \${app}: healthy (attempt \${i})"
            break
        fi
        if [ "\${i}" -eq 30 ]; then
            echo "ERROR: \${app} health check failed"
            docker logs "\${app}" --tail 30
            exit 1
        fi
        sleep 2
    done
done

# --- Notify ---
curl -s -X POST "\${SLACK_WEBHOOK}" \\
    -H 'Content-Type: application/json' \\
    -d "{\"text\": \":white_check_mark: Customs pipeline deployed to \${DEPLOY_ENV} (tag: \${IMAGE_TAG})\"}"

echo "==> Customs pipeline deployment to \${DEPLOY_ENV} complete."`,
              },
              {
                language: 'python',
                title: 'Customs Pipeline Configuration Loader',
                description:
                  'Environment-based configuration loader with secrets management and connection setup for the customs classification and document generation pipeline.',
                code: `import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str

    @property
    def conn_string(self) -> str:
        return (
            f"host={self.host} port={self.port} "
            f"dbname={self.database} user={self.user} password={self.password}"
        )


@dataclass(frozen=True)
class ClassifierConfig:
    model_path: Path
    confidence_threshold: float = 0.85
    top_k_predictions: int = 3
    api_port: int = 8080


@dataclass(frozen=True)
class DocGenConfig:
    output_dir: Path
    template_dir: Path
    api_port: int = 8081


@dataclass(frozen=True)
class AlertConfig:
    slack_webhook_url: str
    alert_channel: str = "#customs-alerts"
    misclassification_rate_threshold: float = 0.05
    confidence_drop_alert: bool = True


@dataclass(frozen=True)
class AppConfig:
    env: str
    db: DatabaseConfig
    classifier: ClassifierConfig
    docgen: DocGenConfig
    alerts: AlertConfig
    log_level: str = "INFO"


def _require_env(key: str) -> str:
    """Fetch a required environment variable or raise."""
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(f"Required env var {key} is not set")
    return value


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    return AppConfig(
        env=_require_env("DEPLOY_ENV"),
        db=DatabaseConfig(
            host=_require_env("DB_HOST"),
            port=int(os.environ.get("DB_PORT", "5432")),
            database=os.environ.get("DB_NAME", "supplychain"),
            user=_require_env("DB_USER"),
            password=_require_env("DB_PASSWORD"),
        ),
        classifier=ClassifierConfig(
            model_path=Path(_require_env("MODEL_PATH")),
            confidence_threshold=float(
                os.environ.get("CONFIDENCE_THRESHOLD", "0.85")
            ),
            top_k_predictions=int(os.environ.get("TOP_K_PREDICTIONS", "3")),
            api_port=int(os.environ.get("CLASSIFIER_API_PORT", "8080")),
        ),
        docgen=DocGenConfig(
            output_dir=Path(
                os.environ.get("DOCGEN_OUTPUT_DIR", "/opt/customs/generated_docs")
            ),
            template_dir=Path(
                os.environ.get("DOCGEN_TEMPLATE_DIR", "/opt/customs/templates")
            ),
            api_port=int(os.environ.get("DOCGEN_API_PORT", "8081")),
        ),
        alerts=AlertConfig(
            slack_webhook_url=_require_env("SLACK_WEBHOOK"),
            alert_channel=os.environ.get(
                "ALERT_CHANNEL", "#customs-alerts"
            ),
            misclassification_rate_threshold=float(
                os.environ.get("MISCLASS_RATE_THRESHOLD", "0.05")
            ),
        ),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )


if __name__ == "__main__":
    config = load_config()
    print(f"Environment:   {config.env}")
    print(f"Database:      {config.db.host}:{config.db.port}/{config.db.database}")
    print(f"Classifier:    {config.classifier.model_path} (port {config.classifier.api_port})")
    print(f"Doc Generator: {config.docgen.output_dir} (port {config.docgen.api_port})")
    print(f"Alerts:        {config.alerts.alert_channel}")`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Production monitoring that tracks classification confidence trends, customs rejection rates, document generation throughput, and sends Slack alerts when misclassification or rejection thresholds are breached.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Classification Confidence & Rejection Dashboard',
                description:
                  'Queries that track classification confidence distribution, rejection trends, and misclassification patterns for the customs pipeline.',
                code: `-- ============================================================
-- Customs Classification Monitoring Dashboard Queries
-- ============================================================

-- 1. Classification confidence distribution (last 30 days)
SELECT
    CASE
        WHEN classification_confidence >= 0.95 THEN '0.95-1.00 (high)'
        WHEN classification_confidence >= 0.85 THEN '0.85-0.95 (auto-clear)'
        WHEN classification_confidence >= 0.70 THEN '0.70-0.85 (review)'
        ELSE '< 0.70 (manual)'
    END AS confidence_band,
    COUNT(*) AS classifications,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct
FROM hs_classifications
WHERE classified_at >= NOW() - INTERVAL '30 days'
GROUP BY 1
ORDER BY MIN(classification_confidence) DESC;

-- 2. Customs rejection rate trend (weekly)
SELECT
    DATE_TRUNC('week', recorded_at) AS week_start,
    COUNT(*) AS total_feedback,
    COUNT(*) FILTER (WHERE NOT was_accepted) AS rejections,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE NOT was_accepted)
        / NULLIF(COUNT(*), 0), 2
    ) AS rejection_rate_pct
FROM customs_classification_feedback
WHERE recorded_at >= NOW() - INTERVAL '90 days'
GROUP BY DATE_TRUNC('week', recorded_at)
ORDER BY week_start DESC;

-- 3. Top misclassification pairs for model retraining
SELECT
    predicted_hs_code,
    actual_hs_code,
    COUNT(*) AS occurrences,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_errors
FROM customs_classification_feedback
WHERE NOT was_accepted
  AND actual_hs_code IS NOT NULL
  AND recorded_at >= NOW() - INTERVAL '30 days'
GROUP BY predicted_hs_code, actual_hs_code
ORDER BY occurrences DESC
LIMIT 20;`,
              },
              {
                language: 'python',
                title: 'Customs Classification Slack Alerting',
                description:
                  'Monitors classification confidence trends, rejection rates, and document generation health, sending Slack webhook alerts when thresholds are breached.',
                code: `import json
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras
import requests

SLACK_WEBHOOK = "https://hooks.slack.com/services/T00/B00/xxxx"
DB_CONN = "host=db-host dbname=supplychain user=customs_svc password=secret"

REJECTION_RATE_THRESHOLD = 0.05     # Alert if > 5% rejection rate
LOW_CONFIDENCE_THRESHOLD = 0.20     # Alert if > 20% of classifications are low-conf
CONFIDENCE_CUTOFF = 0.85


def send_slack_alert(title: str, details: list[dict[str, Any]],
                     severity: str = "warning") -> None:
    """Post a structured alert to Slack via incoming webhook."""
    color_map = {"info": "#36a64f", "warning": "#ff9900", "critical": "#ff0000"}
    fields = [
        {"title": d["label"], "value": str(d["value"]), "short": True}
        for d in details
    ]
    payload = {
        "attachments": [{
            "color": color_map.get(severity, "#ff9900"),
            "title": f":rotating_light: {title}",
            "fields": fields,
            "footer": "Customs Classification Monitor",
            "ts": int(datetime.now(timezone.utc).timestamp()),
        }]
    }
    requests.post(SLACK_WEBHOOK, json=payload)


def check_and_alert() -> dict[str, Any]:
    """Run customs classification health checks and fire alerts."""
    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    alerts_fired = []

    # --- Rejection rate check ---
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE NOT was_accepted) AS rejections,
            COUNT(*) AS total
        FROM customs_classification_feedback
        WHERE recorded_at >= NOW() - INTERVAL '7 days'
    """)
    row = cur.fetchone()
    if row["total"] > 0:
        rejection_rate = row["rejections"] / row["total"]
        if rejection_rate > REJECTION_RATE_THRESHOLD:
            send_slack_alert(
                "High Customs Rejection Rate",
                [
                    {"label": "Rejection rate", "value": f"{rejection_rate:.1%}"},
                    {"label": "Rejections (7d)", "value": row["rejections"]},
                    {"label": "Total (7d)", "value": row["total"]},
                    {"label": "Threshold", "value": f"{REJECTION_RATE_THRESHOLD:.0%}"},
                ],
                severity="critical",
            )
            alerts_fired.append("high_rejection_rate")

    # --- Low confidence rate check ---
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE classification_confidence < %s) AS low_conf,
            COUNT(*) AS total
        FROM hs_classifications
        WHERE classified_at >= NOW() - INTERVAL '7 days'
    """, (CONFIDENCE_CUTOFF,))
    row = cur.fetchone()
    if row["total"] > 0:
        low_conf_rate = row["low_conf"] / row["total"]
        if low_conf_rate > LOW_CONFIDENCE_THRESHOLD:
            send_slack_alert(
                "Classification Confidence Degradation",
                [
                    {"label": "Low-confidence rate", "value": f"{low_conf_rate:.1%}"},
                    {"label": "Low-conf count", "value": row["low_conf"]},
                    {"label": "Confidence cutoff", "value": str(CONFIDENCE_CUTOFF)},
                ],
                severity="warning",
            )
            alerts_fired.append("low_confidence")

    # --- Top misclassification patterns ---
    cur.execute("""
        SELECT predicted_hs_code, actual_hs_code, COUNT(*) AS cnt
        FROM customs_classification_feedback
        WHERE NOT was_accepted AND actual_hs_code IS NOT NULL
          AND recorded_at >= NOW() - INTERVAL '7 days'
        GROUP BY predicted_hs_code, actual_hs_code
        ORDER BY cnt DESC
        LIMIT 3
    """)
    top_errors = cur.fetchall()
    if top_errors and top_errors[0]["cnt"] >= 5:
        details = [{"label": "Top error pattern", "value": "See below"}]
        for err in top_errors:
            details.append({
                "label": f"{err['predicted_hs_code']} -> {err['actual_hs_code']}",
                "value": f"{err['cnt']} occurrences",
            })
        send_slack_alert(
            "Recurring Misclassification Patterns",
            details,
            severity="warning",
        )
        alerts_fired.append("misclassification_patterns")

    cur.close()
    conn.close()
    return {"alerts_fired": alerts_fired, "checked_at": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    result = check_and_alert()
    print(json.dumps(result, indent=2))`,
              },
            ],
          },
        ],
        toolsUsed: ['PostgreSQL', 'Python', 'scikit-learn', 'reportlab', 'psycopg2', 'pytest', 'Docker', 'GitHub Actions', 'cron', 'Slack API', 'Prometheus'],
      },
    },
  ],
};
