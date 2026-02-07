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
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to create an automated ATP monitoring dashboard that alerts on inventory discrepancies, analyzes demand patterns from exported WMS data, and generates daily inventory health reports without custom code.',
        estimatedMonthlyCost: '$100 - $200/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Google Sheets (Free)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Flexport API (Usage-based)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Export WMS inventory snapshots and order allocation data to Google Sheets on a scheduled basis, formatting the data for AI analysis of ATP accuracy and demand patterns.',
            toolsUsed: ['Google Sheets', 'Zapier', 'WMS Export API'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier WMS Data Extract Trigger',
                description:
                  'Zapier configuration to extract WMS inventory data daily and append to Google Sheets for analysis.',
                code: `{
  "trigger": {
    "app": "Schedule by Zapier",
    "event": "Every Day",
    "time": "06:00 AM",
    "timezone": "America/New_York"
  },
  "action_1": {
    "app": "Webhooks by Zapier",
    "event": "GET",
    "url": "https://wms.company.com/api/v1/inventory/snapshot",
    "headers": {
      "Authorization": "Bearer {{wms_api_token}}",
      "Content-Type": "application/json"
    },
    "querystring": {
      "warehouse_ids": "WH001,WH002,WH003",
      "include_allocated": "true",
      "include_in_transit": "true"
    }
  },
  "action_2": {
    "app": "Google Sheets",
    "event": "Create Spreadsheet Row(s)",
    "spreadsheet_id": "{{inventory_tracking_sheet_id}}",
    "worksheet": "Daily_Snapshots",
    "row_data": {
      "snapshot_date": "{{current_date}}",
      "warehouse_id": "{{action_1.warehouse_id}}",
      "sku": "{{action_1.sku}}",
      "on_hand_qty": "{{action_1.on_hand}}",
      "allocated_qty": "{{action_1.allocated}}",
      "in_transit_qty": "{{action_1.in_transit}}",
      "atp_calculated": "{{action_1.on_hand - action_1.allocated + action_1.in_transit}}"
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
              'Use ChatGPT to analyze inventory patterns, identify ATP discrepancies between WMS and ERP, detect stockout risks, and recommend safety stock adjustments based on demand velocity.',
            toolsUsed: ['ChatGPT Plus', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'ATP Discrepancy Analysis Prompt Template',
                description:
                  'Structured prompt for ChatGPT to analyze inventory data and identify ATP calculation issues.',
                code: `system_prompt: |
  You are a supply chain inventory analyst specializing in ATP (Available-to-Promise)
  accuracy and demand forecasting. Analyze the provided inventory data to identify
  discrepancies, stockout risks, and optimization opportunities.

user_prompt_template: |
  ## Inventory Analysis Request

  **Date Range:** {{start_date}} to {{end_date}}
  **Warehouses:** {{warehouse_list}}

  ### Current Inventory Snapshot
  \`\`\`csv
  {{inventory_data_csv}}
  \`\`\`

  ### Recent Order Allocations (Last 7 Days)
  \`\`\`csv
  {{allocation_data_csv}}
  \`\`\`

  ### Analysis Tasks:
  1. **ATP Accuracy Check**: Compare calculated ATP against reported ATP. Flag any
     discrepancies greater than 5% or 10 units.

  2. **Stockout Risk Assessment**: Identify SKUs where:
     - Current ATP < 3 days of average daily demand
     - In-transit inventory won't arrive before projected stockout
     - Safety stock has been breached

  3. **Demand Velocity Analysis**: Calculate daily sell-through rate for top 20 SKUs
     and flag any showing >20% increase vs. prior period.

  4. **Recommendations**: Provide specific actions:
     - SKUs requiring immediate replenishment
     - Suggested safety stock adjustments
     - Warehouse transfer opportunities to balance inventory

  ### Output Format:
  Provide a structured JSON response with:
  - discrepancies: [{sku, warehouse, calculated_atp, reported_atp, variance_pct}]
  - stockout_risks: [{sku, warehouse, current_atp, daily_demand, days_of_supply, risk_level}]
  - velocity_alerts: [{sku, current_velocity, prior_velocity, change_pct}]
  - recommendations: [{priority, action, sku, details, estimated_impact}]

expected_output_format: json
temperature: 0.3
max_tokens: 4000`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Automate the analysis workflow with Zapier to run daily, parse ChatGPT responses, and deliver actionable alerts via Slack and email to inventory planners.',
            toolsUsed: ['Zapier', 'Slack', 'Gmail'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier ATP Alert Automation Workflow',
                description:
                  'Complete Zapier workflow that triggers AI analysis and delivers formatted alerts to the operations team.',
                code: `{
  "workflow_name": "Daily ATP Analysis & Alerts",
  "trigger": {
    "app": "Schedule by Zapier",
    "event": "Every Day",
    "time": "07:00 AM",
    "timezone": "America/New_York"
  },
  "action_1": {
    "app": "Google Sheets",
    "event": "Get Many Spreadsheet Rows",
    "spreadsheet_id": "{{inventory_tracking_sheet_id}}",
    "worksheet": "Daily_Snapshots",
    "filter": {
      "snapshot_date": "{{yesterday_date}}"
    }
  },
  "action_2": {
    "app": "ChatGPT",
    "event": "Conversation",
    "model": "gpt-4",
    "system_message": "You are a supply chain inventory analyst...",
    "user_message": "Analyze this inventory data: {{action_1.rows_json}}",
    "temperature": 0.3
  },
  "action_3": {
    "app": "Formatter by Zapier",
    "event": "Text - Extract Pattern",
    "input": "{{action_2.response}}",
    "pattern": "stockout_risks.*?\\[([^\\]]+)\\]"
  },
  "action_4": {
    "app": "Filter by Zapier",
    "condition": "{{action_3.extracted_count}} > 0"
  },
  "action_5": {
    "app": "Slack",
    "event": "Send Channel Message",
    "channel": "#inventory-alerts",
    "message_blocks": [
      {
        "type": "header",
        "text": ":warning: Daily ATP Analysis - {{current_date}}"
      },
      {
        "type": "section",
        "text": "*Stockout Risks Identified:* {{action_3.stockout_count}}"
      },
      {
        "type": "section",
        "text": "*High Priority Items:*\\n{{action_2.high_priority_summary}}"
      },
      {
        "type": "actions",
        "elements": [
          {
            "type": "button",
            "text": "View Full Report",
            "url": "{{google_sheets_report_url}}"
          }
        ]
      }
    ]
  },
  "action_6": {
    "app": "Gmail",
    "event": "Send Email",
    "to": "inventory-team@company.com",
    "subject": "Daily ATP Analysis Report - {{current_date}}",
    "body_html": "<h2>Inventory Health Summary</h2>{{action_2.email_formatted_response}}"
  }
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent system using CrewAI and LangGraph that continuously monitors inventory movements, predicts stockouts using ML demand sensing, automatically calculates optimal ATP across warehouses, and orchestrates replenishment recommendations in real-time.',
        estimatedMonthlyCost: '$600 - $1,500/month',
        architecture:
          'A Supervisor agent coordinates four specialist agents: Inventory Monitor Agent streams WMS events and detects anomalies, Demand Sensing Agent uses ML to forecast short-term demand, ATP Calculator Agent computes real-time available-to-promise across all locations, and Replenishment Agent generates and prioritizes stock transfer and reorder recommendations.',
        agents: [
          {
            name: 'Inventory Monitor Agent',
            role: 'Real-Time Inventory Event Processor',
            goal: 'Stream and process WMS inventory movements in real-time, detect quantity anomalies, and maintain a live inventory position across all warehouses.',
            tools: ['Kafka Consumer', 'Redis Cache', 'Anomaly Detection Model', 'PostgreSQL'],
          },
          {
            name: 'Demand Sensing Agent',
            role: 'Short-Term Demand Forecaster',
            goal: 'Analyze recent sales velocity, promotional calendars, and external signals to generate 7-day demand forecasts at SKU-location level.',
            tools: ['Prophet Model', 'Sales API', 'Promotion Calendar API', 'Feature Store'],
          },
          {
            name: 'ATP Calculator Agent',
            role: 'Available-to-Promise Engine',
            goal: 'Calculate real-time ATP by combining current inventory positions with demand forecasts, committed orders, and in-transit shipments.',
            tools: ['Inventory State Store', 'Order Management API', 'Shipment Tracker', 'ATP Rules Engine'],
          },
          {
            name: 'Replenishment Agent',
            role: 'Stock Optimization Recommender',
            goal: 'Generate prioritized replenishment recommendations including inter-warehouse transfers, purchase orders, and safety stock adjustments.',
            tools: ['Optimization Solver', 'Supplier Lead Time DB', 'Cost Calculator', 'Slack Notifier'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with hourly checkpointing and event sourcing for full auditability',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system with CrewAI, establishing each agent\'s role, goals, tools, and inter-agent communication protocols for real-time inventory orchestration.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'CrewAI Inventory Agent Definitions',
                description:
                  'Complete CrewAI agent setup for the four-agent inventory management system with detailed role definitions and tool assignments.',
                code: `from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InventoryAgentConfig:
    """Configuration for inventory management agents."""

    def __init__(self, openai_api_key: str, model: str = "gpt-4-turbo"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0.1,
        )

    def create_inventory_monitor_agent(self, tools: List[Any]) -> Agent:
        """Create the real-time inventory monitoring agent."""
        return Agent(
            role="Real-Time Inventory Monitor",
            goal="""Stream and process WMS inventory movements in real-time,
            detect quantity anomalies (negative stock, sudden drops >20%,
            variance from expected), and maintain a live inventory position
            across all warehouses with sub-minute latency.""",
            backstory="""You are an expert inventory analyst with 15 years
            of experience in warehouse operations. You've developed an intuition
            for detecting inventory discrepancies before they cause stockouts.
            You monitor inventory events 24/7 and immediately flag anomalies
            that could indicate theft, system errors, or demand spikes.""",
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=10,
        )

    def create_demand_sensing_agent(self, tools: List[Any]) -> Agent:
        """Create the demand forecasting agent."""
        return Agent(
            role="Demand Sensing Specialist",
            goal="""Generate accurate 7-day demand forecasts at SKU-location
            level by analyzing sales velocity, promotional impacts, seasonal
            patterns, and external signals like weather and economic indicators.""",
            backstory="""You are a demand planning expert who has built
            forecasting models for Fortune 500 retailers. You understand that
            traditional forecasts fail during promotions and anomalies. You
            specialize in 'demand sensing' - adjusting forecasts in real-time
            based on the latest signals rather than relying on stale history.""",
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            max_iter=15,
        )

    def create_atp_calculator_agent(self, tools: List[Any]) -> Agent:
        """Create the ATP calculation agent."""
        return Agent(
            role="ATP Calculation Engine",
            goal="""Calculate real-time Available-to-Promise quantities by
            combining current inventory positions with demand forecasts,
            committed customer orders, pending shipments, and configurable
            safety stock rules. Ensure ATP never goes negative and allocation
            priorities are respected.""",
            backstory="""You are an order promising specialist who has
            designed ATP engines for major e-commerce platforms. You understand
            the delicate balance between customer satisfaction (high ATP = more
            promises) and operational risk (low ATP = stockouts). You apply
            sophisticated allocation rules based on channel priority, customer
            tier, and margin contribution.""",
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=8,
        )

    def create_replenishment_agent(self, tools: List[Any]) -> Agent:
        """Create the replenishment recommendation agent."""
        return Agent(
            role="Replenishment Optimizer",
            goal="""Generate prioritized replenishment recommendations including
            inter-warehouse transfers, purchase orders to suppliers, and safety
            stock adjustments. Optimize for service level targets while
            minimizing total inventory investment and logistics costs.""",
            backstory="""You are a supply chain optimization expert who has
            saved companies millions through intelligent inventory positioning.
            You balance the trade-offs between holding costs, stockout costs,
            and expedited freight. You know when to recommend a warehouse
            transfer vs. a new PO, and you factor in supplier lead times,
            MOQs, and freight consolidation opportunities.""",
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            max_iter=12,
        )


def create_inventory_crew(config: InventoryAgentConfig, tools: Dict[str, List[Any]]) -> Crew:
    """Create the complete inventory management crew."""

    monitor_agent = config.create_inventory_monitor_agent(tools["monitor"])
    demand_agent = config.create_demand_sensing_agent(tools["demand"])
    atp_agent = config.create_atp_calculator_agent(tools["atp"])
    replenishment_agent = config.create_replenishment_agent(tools["replenishment"])

    # Define tasks for each agent
    monitor_task = Task(
        description="""Process the latest batch of inventory events from WMS.
        For each event:
        1. Update the real-time inventory position in Redis
        2. Check for anomalies (negative qty, >20% sudden change, variance)
        3. Flag any SKU-locations that need immediate attention

        Current event batch: {event_batch}""",
        expected_output="JSON with updated positions and flagged anomalies",
        agent=monitor_agent,
    )

    demand_task = Task(
        description="""Generate 7-day demand forecast for flagged SKU-locations.
        Consider:
        1. Last 30 days sales velocity
        2. Active promotions from calendar
        3. Day-of-week patterns
        4. Any external signals (weather, events)

        SKUs to forecast: {flagged_skus}""",
        expected_output="JSON with daily demand forecasts per SKU-location",
        agent=demand_agent,
        context=[monitor_task],
    )

    atp_task = Task(
        description="""Calculate ATP for each SKU-location using:
        1. Current inventory position from monitor
        2. Demand forecast from sensing agent
        3. Open customer orders and allocations
        4. In-transit shipments with ETA
        5. Safety stock rules by SKU class

        Inventory positions: {inventory_positions}
        Demand forecasts: {demand_forecasts}""",
        expected_output="JSON with ATP by SKU-location and days of supply",
        agent=atp_agent,
        context=[monitor_task, demand_task],
    )

    replenishment_task = Task(
        description="""Generate replenishment recommendations based on ATP analysis.
        For each SKU-location with <5 days of supply:
        1. Check if transfer from another warehouse is feasible
        2. If not, generate PO recommendation with optimal qty
        3. Calculate priority score based on stockout risk and margin
        4. Estimate cost of recommendation vs. stockout cost

        ATP analysis: {atp_analysis}""",
        expected_output="JSON with prioritized replenishment actions",
        agent=replenishment_agent,
        context=[atp_task],
    )

    return Crew(
        agents=[monitor_agent, demand_agent, atp_agent, replenishment_agent],
        tasks=[monitor_task, demand_task, atp_task, replenishment_task],
        process=Process.sequential,
        verbose=True,
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the Inventory Monitor Agent with real-time Kafka event streaming, Redis state management, and anomaly detection for continuous inventory position tracking.',
            toolsUsed: ['Kafka', 'Redis', 'asyncio', 'scikit-learn'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Real-Time Inventory Monitor Agent Implementation',
                description:
                  'Production-ready implementation of the Inventory Monitor Agent with Kafka streaming, Redis state, and anomaly detection.',
                code: `import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from aiokafka import AIOKafkaConsumer
import redis.asyncio as redis
import numpy as np
from langchain.tools import tool

logger = logging.getLogger(__name__)


@dataclass
class InventoryPosition:
    """Represents current inventory position for a SKU at a location."""
    sku: str
    warehouse_id: str
    on_hand: int
    allocated: int
    in_transit: int
    atp: int
    last_updated: str
    event_count_24h: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class InventoryAnomaly:
    """Represents a detected inventory anomaly."""
    sku: str
    warehouse_id: str
    anomaly_type: str  # 'negative_stock', 'sudden_drop', 'variance', 'velocity_spike'
    severity: str  # 'low', 'medium', 'high', 'critical'
    current_value: float
    expected_value: float
    deviation_pct: float
    detected_at: str
    details: str


class InventoryStateStore:
    """Redis-backed inventory state management."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.position_prefix = "inv:position:"
        self.history_prefix = "inv:history:"
        self.anomaly_prefix = "inv:anomaly:"

    async def connect(self) -> None:
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)
        logger.info("Connected to Redis state store")

    async def close(self) -> None:
        if self.redis:
            await self.redis.close()

    async def update_position(self, position: InventoryPosition) -> None:
        """Update inventory position in Redis."""
        key = f"{self.position_prefix}{position.warehouse_id}:{position.sku}"
        await self.redis.hset(key, mapping=position.to_dict())
        await self.redis.expire(key, 86400 * 7)  # 7 day TTL

        # Also append to time-series history
        history_key = f"{self.history_prefix}{position.warehouse_id}:{position.sku}"
        await self.redis.zadd(
            history_key,
            {json.dumps({"ts": position.last_updated, "atp": position.atp}):
             datetime.fromisoformat(position.last_updated).timestamp()}
        )
        # Keep only last 7 days of history
        cutoff = datetime.now(timezone.utc).timestamp() - (86400 * 7)
        await self.redis.zremrangebyscore(history_key, "-inf", cutoff)

    async def get_position(self, warehouse_id: str, sku: str) -> Optional[InventoryPosition]:
        """Retrieve current inventory position."""
        key = f"{self.position_prefix}{warehouse_id}:{sku}"
        data = await self.redis.hgetall(key)
        if not data:
            return None
        return InventoryPosition(
            sku=data["sku"],
            warehouse_id=data["warehouse_id"],
            on_hand=int(data["on_hand"]),
            allocated=int(data["allocated"]),
            in_transit=int(data["in_transit"]),
            atp=int(data["atp"]),
            last_updated=data["last_updated"],
            event_count_24h=int(data.get("event_count_24h", 0)),
        )

    async def get_position_history(
        self, warehouse_id: str, sku: str, hours: int = 24
    ) -> List[Tuple[datetime, int]]:
        """Get historical ATP values for anomaly detection."""
        history_key = f"{self.history_prefix}{warehouse_id}:{sku}"
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        entries = await self.redis.zrangebyscore(history_key, cutoff, "+inf")
        result = []
        for entry in entries:
            data = json.loads(entry)
            result.append((datetime.fromisoformat(data["ts"]), data["atp"]))
        return result

    async def record_anomaly(self, anomaly: InventoryAnomaly) -> None:
        """Record detected anomaly for alerting."""
        key = f"{self.anomaly_prefix}{anomaly.warehouse_id}:{anomaly.sku}:{anomaly.detected_at}"
        await self.redis.hset(key, mapping=asdict(anomaly))
        await self.redis.expire(key, 86400 * 30)  # 30 day retention

        # Add to anomaly stream for real-time alerting
        await self.redis.xadd(
            "inv:anomaly_stream",
            asdict(anomaly),
            maxlen=10000,
        )


class InventoryAnomalyDetector:
    """Detects inventory anomalies using statistical methods."""

    def __init__(self, state_store: InventoryStateStore):
        self.state_store = state_store
        self.sudden_drop_threshold = 0.20  # 20% drop triggers alert
        self.velocity_spike_threshold = 2.0  # 2x normal velocity

    async def detect_anomalies(
        self, position: InventoryPosition, event: Dict
    ) -> List[InventoryAnomaly]:
        """Detect anomalies in the new inventory position."""
        anomalies = []
        now = datetime.now(timezone.utc).isoformat()

        # Check for negative stock
        if position.on_hand < 0:
            anomalies.append(InventoryAnomaly(
                sku=position.sku,
                warehouse_id=position.warehouse_id,
                anomaly_type="negative_stock",
                severity="critical",
                current_value=position.on_hand,
                expected_value=0,
                deviation_pct=100.0,
                detected_at=now,
                details=f"Negative on-hand quantity: {position.on_hand}",
            ))

        # Check for sudden ATP drop
        history = await self.state_store.get_position_history(
            position.warehouse_id, position.sku, hours=24
        )
        if len(history) >= 2:
            prev_atp = history[-2][1] if len(history) > 1 else history[-1][1]
            if prev_atp > 0:
                drop_pct = (prev_atp - position.atp) / prev_atp
                if drop_pct >= self.sudden_drop_threshold:
                    anomalies.append(InventoryAnomaly(
                        sku=position.sku,
                        warehouse_id=position.warehouse_id,
                        anomaly_type="sudden_drop",
                        severity="high" if drop_pct >= 0.5 else "medium",
                        current_value=position.atp,
                        expected_value=prev_atp,
                        deviation_pct=drop_pct * 100,
                        detected_at=now,
                        details=f"ATP dropped {drop_pct:.1%} from {prev_atp} to {position.atp}",
                    ))

        # Check for velocity spike (using event count as proxy)
        if len(history) >= 24:
            atp_values = [h[1] for h in history]
            avg_change = np.mean(np.abs(np.diff(atp_values)))
            current_change = abs(position.atp - (history[-1][1] if history else position.atp))
            if avg_change > 0 and current_change > avg_change * self.velocity_spike_threshold:
                anomalies.append(InventoryAnomaly(
                    sku=position.sku,
                    warehouse_id=position.warehouse_id,
                    anomaly_type="velocity_spike",
                    severity="medium",
                    current_value=current_change,
                    expected_value=avg_change,
                    deviation_pct=((current_change / avg_change) - 1) * 100,
                    detected_at=now,
                    details=f"Movement velocity {current_change/avg_change:.1f}x normal",
                ))

        return anomalies


class InventoryMonitorAgent:
    """Main agent class for real-time inventory monitoring."""

    def __init__(
        self,
        kafka_brokers: str,
        kafka_topic: str,
        redis_url: str,
    ):
        self.kafka_brokers = kafka_brokers
        self.kafka_topic = kafka_topic
        self.state_store = InventoryStateStore(redis_url)
        self.anomaly_detector = InventoryAnomalyDetector(self.state_store)
        self.consumer: Optional[AIOKafkaConsumer] = None

    async def start(self) -> None:
        """Initialize connections and start processing."""
        await self.state_store.connect()

        self.consumer = AIOKafkaConsumer(
            self.kafka_topic,
            bootstrap_servers=self.kafka_brokers,
            group_id="inventory-monitor-agent",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        await self.consumer.start()
        logger.info(f"Started consuming from {self.kafka_topic}")

    async def stop(self) -> None:
        """Gracefully shutdown."""
        if self.consumer:
            await self.consumer.stop()
        await self.state_store.close()

    async def process_event(self, event: Dict) -> Tuple[InventoryPosition, List[InventoryAnomaly]]:
        """Process a single inventory event."""
        # Calculate new position
        position = InventoryPosition(
            sku=event["sku"],
            warehouse_id=event["warehouse_id"],
            on_hand=event.get("on_hand", 0),
            allocated=event.get("allocated", 0),
            in_transit=event.get("in_transit", 0),
            atp=event.get("on_hand", 0) - event.get("allocated", 0) + event.get("in_transit", 0),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        # Detect anomalies
        anomalies = await self.anomaly_detector.detect_anomalies(position, event)

        # Persist state
        await self.state_store.update_position(position)
        for anomaly in anomalies:
            await self.state_store.record_anomaly(anomaly)
            logger.warning(f"Anomaly detected: {anomaly.anomaly_type} for {anomaly.sku}@{anomaly.warehouse_id}")

        return position, anomalies

    async def run(self) -> None:
        """Main event processing loop."""
        await self.start()
        try:
            async for record in self.consumer:
                try:
                    position, anomalies = await self.process_event(record.value)
                    if anomalies:
                        logger.info(f"Processed event with {len(anomalies)} anomalies: {position.sku}")
                except Exception as e:
                    logger.error(f"Error processing event: {e}", exc_info=True)
        finally:
            await self.stop()


# LangChain tool for CrewAI integration
@tool
def get_inventory_positions(warehouse_ids: str, skus: str) -> str:
    """
    Get current inventory positions for specified warehouses and SKUs.

    Args:
        warehouse_ids: Comma-separated warehouse IDs
        skus: Comma-separated SKU codes

    Returns:
        JSON string with inventory positions and any recent anomalies
    """
    # This would be implemented to query the Redis state store
    pass


@tool
def get_recent_anomalies(hours: int = 24) -> str:
    """
    Get inventory anomalies detected in the last N hours.

    Args:
        hours: Number of hours to look back (default 24)

    Returns:
        JSON string with anomaly details sorted by severity
    """
    # This would be implemented to query the Redis anomaly stream
    pass`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Demand Sensing and ATP Calculator agents with ML-based forecasting, real-time ATP computation, and intelligent allocation rules.',
            toolsUsed: ['Prophet', 'NumPy', 'pandas', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Demand Sensing & ATP Calculator Agents',
                description:
                  'Implementation of the Demand Sensing Agent with Prophet forecasting and the ATP Calculator Agent with configurable allocation rules.',
                code: `import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from prophet import Prophet
from langchain.tools import tool
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class DemandForecast:
    """7-day demand forecast for a SKU-location."""
    sku: str
    warehouse_id: str
    forecast_date: str
    daily_forecasts: List[Dict[str, Any]]  # [{date, forecast, lower, upper}]
    avg_daily_demand: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    confidence: float
    model_metrics: Dict[str, float]


@dataclass
class ATPResult:
    """ATP calculation result for a SKU-location."""
    sku: str
    warehouse_id: str
    current_atp: int
    projected_atp_7d: List[Dict[str, int]]  # [{date, atp}]
    days_of_supply: float
    stockout_date: Optional[str]
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    allocation_priority: int
    safety_stock: int
    safety_stock_breach: bool


class DemandSensingAgent:
    """Agent for short-term demand forecasting using Prophet."""

    def __init__(self, redis_url: str):
        self.redis = None
        self.redis_url = redis_url
        self.sales_history_prefix = "sales:history:"
        self.promo_calendar_key = "promo:calendar"

    async def connect(self) -> None:
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)

    async def close(self) -> None:
        if self.redis:
            await self.redis.close()

    async def get_sales_history(
        self, warehouse_id: str, sku: str, days: int = 90
    ) -> pd.DataFrame:
        """Fetch sales history from Redis."""
        key = f"{self.sales_history_prefix}{warehouse_id}:{sku}"
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        entries = await self.redis.zrangebyscore(key, cutoff, "+inf")

        records = []
        for entry in entries:
            data = json.loads(entry)
            records.append({
                "ds": pd.to_datetime(data["date"]),
                "y": data["units_sold"],
            })

        return pd.DataFrame(records) if records else pd.DataFrame(columns=["ds", "y"])

    async def get_promotions(
        self, sku: str, start_date: datetime, end_date: datetime
    ) -> List[Dict]:
        """Fetch promotion calendar for demand adjustment."""
        promos = await self.redis.hget(self.promo_calendar_key, sku)
        if not promos:
            return []

        promo_list = json.loads(promos)
        return [
            p for p in promo_list
            if start_date <= datetime.fromisoformat(p["start_date"]) <= end_date
        ]

    def fit_prophet_model(
        self, history: pd.DataFrame, promotions: List[Dict]
    ) -> Prophet:
        """Fit Prophet model with optional promotion regressors."""
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.80,
        )

        # Add promotion regressor if we have promo data
        if promotions:
            history["promo"] = 0
            for promo in promotions:
                promo_start = pd.to_datetime(promo["start_date"])
                promo_end = pd.to_datetime(promo["end_date"])
                mask = (history["ds"] >= promo_start) & (history["ds"] <= promo_end)
                history.loc[mask, "promo"] = promo.get("lift_factor", 1.5)
            model.add_regressor("promo")

        # Suppress Prophet logging
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)

        model.fit(history)
        return model

    async def generate_forecast(
        self, warehouse_id: str, sku: str
    ) -> DemandForecast:
        """Generate 7-day demand forecast for a SKU-location."""
        history = await self.get_sales_history(warehouse_id, sku, days=90)

        if len(history) < 14:
            # Insufficient history - use simple average
            avg_demand = history["y"].mean() if len(history) > 0 else 0
            return DemandForecast(
                sku=sku,
                warehouse_id=warehouse_id,
                forecast_date=datetime.now(timezone.utc).isoformat(),
                daily_forecasts=[
                    {
                        "date": (datetime.now(timezone.utc) + timedelta(days=i)).strftime("%Y-%m-%d"),
                        "forecast": avg_demand,
                        "lower": avg_demand * 0.7,
                        "upper": avg_demand * 1.3,
                    }
                    for i in range(1, 8)
                ],
                avg_daily_demand=avg_demand,
                trend="stable",
                confidence=0.5,
                model_metrics={"method": "simple_average", "data_points": len(history)},
            )

        # Get promotions for forecast period
        start_date = datetime.now(timezone.utc)
        end_date = start_date + timedelta(days=7)
        promotions = await self.get_promotions(sku, start_date, end_date)

        # Fit Prophet model
        model = self.fit_prophet_model(history.copy(), promotions)

        # Generate forecast
        future = model.make_future_dataframe(periods=7)
        if promotions:
            future["promo"] = 0
            for promo in promotions:
                promo_start = pd.to_datetime(promo["start_date"])
                promo_end = pd.to_datetime(promo["end_date"])
                mask = (future["ds"] >= promo_start) & (future["ds"] <= promo_end)
                future.loc[mask, "promo"] = promo.get("lift_factor", 1.5)

        forecast = model.predict(future)
        forecast_7d = forecast.tail(7)

        # Calculate trend
        first_half_avg = forecast_7d.head(3)["yhat"].mean()
        second_half_avg = forecast_7d.tail(3)["yhat"].mean()
        if second_half_avg > first_half_avg * 1.1:
            trend = "increasing"
        elif second_half_avg < first_half_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"

        return DemandForecast(
            sku=sku,
            warehouse_id=warehouse_id,
            forecast_date=datetime.now(timezone.utc).isoformat(),
            daily_forecasts=[
                {
                    "date": row["ds"].strftime("%Y-%m-%d"),
                    "forecast": max(0, round(row["yhat"], 1)),
                    "lower": max(0, round(row["yhat_lower"], 1)),
                    "upper": max(0, round(row["yhat_upper"], 1)),
                }
                for _, row in forecast_7d.iterrows()
            ],
            avg_daily_demand=round(forecast_7d["yhat"].mean(), 1),
            trend=trend,
            confidence=0.8 if len(history) >= 30 else 0.6,
            model_metrics={
                "method": "prophet",
                "data_points": len(history),
                "has_promotions": len(promotions) > 0,
            },
        )


class ATPCalculatorAgent:
    """Agent for real-time ATP calculation with allocation rules."""

    def __init__(self, redis_url: str):
        self.redis = None
        self.redis_url = redis_url
        self.safety_stock_rules = {
            "A": 7,   # A-class SKUs: 7 days safety stock
            "B": 5,   # B-class SKUs: 5 days
            "C": 3,   # C-class SKUs: 3 days
        }
        self.allocation_priorities = {
            "retail": 1,      # Highest priority
            "wholesale": 2,
            "marketplace": 3,
            "internal": 4,    # Lowest priority
        }

    async def connect(self) -> None:
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)

    async def close(self) -> None:
        if self.redis:
            await self.redis.close()

    async def get_open_orders(self, warehouse_id: str, sku: str) -> List[Dict]:
        """Get open orders with allocation status."""
        key = f"orders:open:{warehouse_id}:{sku}"
        orders_json = await self.redis.get(key)
        return json.loads(orders_json) if orders_json else []

    async def get_inbound_shipments(self, warehouse_id: str, sku: str) -> List[Dict]:
        """Get expected inbound shipments with ETAs."""
        key = f"shipments:inbound:{warehouse_id}:{sku}"
        shipments_json = await self.redis.get(key)
        return json.loads(shipments_json) if shipments_json else []

    async def get_sku_class(self, sku: str) -> str:
        """Get ABC classification for a SKU."""
        sku_class = await self.redis.hget("sku:classification", sku)
        return sku_class if sku_class in ["A", "B", "C"] else "C"

    async def calculate_atp(
        self,
        warehouse_id: str,
        sku: str,
        current_on_hand: int,
        current_allocated: int,
        demand_forecast: DemandForecast,
    ) -> ATPResult:
        """Calculate ATP with 7-day projection."""
        sku_class = await self.get_sku_class(sku)
        safety_stock_days = self.safety_stock_rules[sku_class]
        safety_stock = int(demand_forecast.avg_daily_demand * safety_stock_days)

        inbound = await self.get_inbound_shipments(warehouse_id, sku)
        open_orders = await self.get_open_orders(warehouse_id, sku)

        # Build day-by-day ATP projection
        current_atp = current_on_hand - current_allocated
        projected_atp = []
        running_atp = current_atp
        stockout_date = None

        for day_forecast in demand_forecast.daily_forecasts:
            forecast_date = day_forecast["date"]
            daily_demand = day_forecast["forecast"]

            # Add any inbound arriving this day
            inbound_qty = sum(
                s["quantity"] for s in inbound
                if s.get("eta_date") == forecast_date
            )

            # Subtract forecasted demand
            running_atp = running_atp + inbound_qty - daily_demand

            projected_atp.append({
                "date": forecast_date,
                "atp": max(0, int(running_atp)),
                "inbound": inbound_qty,
                "demand": daily_demand,
            })

            if running_atp <= 0 and stockout_date is None:
                stockout_date = forecast_date

        # Calculate days of supply
        if demand_forecast.avg_daily_demand > 0:
            days_of_supply = current_atp / demand_forecast.avg_daily_demand
        else:
            days_of_supply = 999  # Effectively infinite if no demand

        # Determine risk level
        if days_of_supply < 3:
            risk_level = "critical"
        elif days_of_supply < 5:
            risk_level = "high"
        elif days_of_supply < 7:
            risk_level = "medium"
        else:
            risk_level = "low"

        return ATPResult(
            sku=sku,
            warehouse_id=warehouse_id,
            current_atp=current_atp,
            projected_atp_7d=projected_atp,
            days_of_supply=round(days_of_supply, 1),
            stockout_date=stockout_date,
            risk_level=risk_level,
            allocation_priority=1,  # Would be calculated based on channel mix
            safety_stock=safety_stock,
            safety_stock_breach=current_atp < safety_stock,
        )


# LangChain tools for CrewAI integration
@tool
def forecast_demand(warehouse_id: str, sku: str) -> str:
    """
    Generate a 7-day demand forecast for a SKU at a warehouse.

    Args:
        warehouse_id: Warehouse identifier
        sku: SKU code

    Returns:
        JSON with daily forecasts, trend, and confidence
    """
    # Implementation would instantiate DemandSensingAgent and call generate_forecast
    pass


@tool
def calculate_atp_projection(
    warehouse_id: str, sku: str, on_hand: int, allocated: int
) -> str:
    """
    Calculate ATP with 7-day projection based on demand forecast.

    Args:
        warehouse_id: Warehouse identifier
        sku: SKU code
        on_hand: Current on-hand quantity
        allocated: Current allocated quantity

    Returns:
        JSON with current ATP, projections, risk level, and stockout date
    """
    # Implementation would call ATPCalculatorAgent
    pass`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement the LangGraph state machine that orchestrates all agents, manages conversation state, handles errors, and routes decisions through the supervisor pattern.',
            toolsUsed: ['LangGraph', 'Redis', 'asyncio'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Supervisor Orchestration',
                description:
                  'Complete LangGraph implementation for orchestrating the multi-agent inventory system with state management and error handling.',
                code: `import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Dict, List, Literal, Optional, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class InventoryState(TypedDict):
    """State schema for the inventory orchestration workflow."""
    messages: Annotated[Sequence[BaseMessage], "Conversation messages"]
    current_step: str
    warehouse_ids: List[str]
    skus_to_process: List[str]
    inventory_positions: Dict[str, Dict]  # {warehouse:sku -> position}
    demand_forecasts: Dict[str, Dict]     # {warehouse:sku -> forecast}
    atp_results: Dict[str, Dict]          # {warehouse:sku -> atp}
    anomalies: List[Dict]
    replenishment_actions: List[Dict]
    errors: List[str]
    iteration_count: int
    last_checkpoint: str


class InventoryOrchestrator:
    """LangGraph-based orchestrator for the inventory agent system."""

    def __init__(
        self,
        openai_api_key: str,
        redis_url: str,
        model: str = "gpt-4-turbo",
    ):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0,
        )
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    async def connect(self) -> None:
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)

    async def close(self) -> None:
        if self.redis:
            await self.redis.close()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(InventoryState)

        # Add nodes for each agent's processing step
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("monitor_inventory", self._monitor_inventory_node)
        workflow.add_node("sense_demand", self._sense_demand_node)
        workflow.add_node("calculate_atp", self._calculate_atp_node)
        workflow.add_node("generate_replenishment", self._replenishment_node)
        workflow.add_node("checkpoint", self._checkpoint_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Add conditional edges from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "monitor": "monitor_inventory",
                "demand": "sense_demand",
                "atp": "calculate_atp",
                "replenish": "generate_replenishment",
                "checkpoint": "checkpoint",
                "end": END,
                "error": "error_handler",
            },
        )

        # Add edges back to supervisor after each node completes
        workflow.add_edge("monitor_inventory", "supervisor")
        workflow.add_edge("sense_demand", "supervisor")
        workflow.add_edge("calculate_atp", "supervisor")
        workflow.add_edge("generate_replenishment", "supervisor")
        workflow.add_edge("checkpoint", "supervisor")
        workflow.add_edge("error_handler", "supervisor")

        return workflow.compile(checkpointer=self.checkpointer)

    async def _supervisor_node(self, state: InventoryState) -> Dict:
        """Supervisor agent that decides the next step."""
        messages = state.get("messages", [])
        current_step = state.get("current_step", "init")
        iteration = state.get("iteration_count", 0)

        # Create supervisor prompt based on current state
        supervisor_prompt = f"""You are the supervisor agent coordinating an inventory management system.

Current state:
- Step: {current_step}
- Iteration: {iteration}
- Warehouses: {state.get('warehouse_ids', [])}
- SKUs pending: {len(state.get('skus_to_process', []))}
- Positions collected: {len(state.get('inventory_positions', {}))}
- Forecasts generated: {len(state.get('demand_forecasts', {}))}
- ATP calculated: {len(state.get('atp_results', {}))}
- Anomalies found: {len(state.get('anomalies', []))}
- Actions generated: {len(state.get('replenishment_actions', []))}
- Errors: {state.get('errors', [])}

Workflow sequence:
1. monitor_inventory - Collect current positions and detect anomalies
2. sense_demand - Generate demand forecasts for flagged SKUs
3. calculate_atp - Calculate ATP using positions and forecasts
4. generate_replenishment - Create replenishment recommendations
5. checkpoint - Save state (every 10 iterations or when complete)

Based on the current state, decide the next action. Respond with ONLY one of:
- "monitor" - if positions not yet collected
- "demand" - if positions collected but forecasts needed
- "atp" - if forecasts ready but ATP not calculated
- "replenish" - if ATP ready but actions not generated
- "checkpoint" - if iteration % 10 == 0 or all steps complete
- "end" - if all processing complete and checkpointed
- "error" - if unrecoverable errors exist

Your decision:"""

        response = await self.llm.ainvoke([HumanMessage(content=supervisor_prompt)])
        decision = response.content.strip().lower()

        # Update state with supervisor's decision
        return {
            "messages": messages + [AIMessage(content=f"Supervisor decided: {decision}")],
            "current_step": decision,
            "iteration_count": iteration + 1,
        }

    def _route_from_supervisor(self, state: InventoryState) -> str:
        """Route to the next node based on supervisor decision."""
        step = state.get("current_step", "init")

        valid_routes = ["monitor", "demand", "atp", "replenish", "checkpoint", "end", "error"]
        if step in valid_routes:
            return step
        return "error"

    async def _monitor_inventory_node(self, state: InventoryState) -> Dict:
        """Execute the inventory monitoring agent."""
        logger.info("Executing inventory monitor agent")

        try:
            # This would call the actual InventoryMonitorAgent
            # For now, simulate the result
            positions = {}
            anomalies = []

            for wh in state.get("warehouse_ids", []):
                for sku in state.get("skus_to_process", []):
                    key = f"{wh}:{sku}"
                    # Fetch from Redis state store
                    position_data = await self.redis.hgetall(f"inv:position:{key}")
                    if position_data:
                        positions[key] = position_data

                    # Check for recent anomalies
                    anomaly_keys = await self.redis.keys(f"inv:anomaly:{key}:*")
                    for ak in anomaly_keys[-5:]:  # Last 5 anomalies
                        anomaly_data = await self.redis.hgetall(ak)
                        if anomaly_data:
                            anomalies.append(anomaly_data)

            return {
                "inventory_positions": positions,
                "anomalies": anomalies,
                "current_step": "monitor_complete",
            }
        except Exception as e:
            logger.error(f"Monitor agent error: {e}")
            return {
                "errors": state.get("errors", []) + [f"Monitor error: {str(e)}"],
            }

    async def _sense_demand_node(self, state: InventoryState) -> Dict:
        """Execute the demand sensing agent."""
        logger.info("Executing demand sensing agent")

        try:
            forecasts = {}
            positions = state.get("inventory_positions", {})

            for key, position in positions.items():
                warehouse_id, sku = key.split(":")
                # This would call DemandSensingAgent.generate_forecast
                # Simulate forecast result
                forecast_key = f"demand:forecast:{key}"
                forecast_data = await self.redis.hgetall(forecast_key)
                if forecast_data:
                    forecasts[key] = forecast_data

            return {
                "demand_forecasts": forecasts,
                "current_step": "demand_complete",
            }
        except Exception as e:
            logger.error(f"Demand sensing error: {e}")
            return {
                "errors": state.get("errors", []) + [f"Demand error: {str(e)}"],
            }

    async def _calculate_atp_node(self, state: InventoryState) -> Dict:
        """Execute the ATP calculator agent."""
        logger.info("Executing ATP calculator agent")

        try:
            atp_results = {}
            positions = state.get("inventory_positions", {})
            forecasts = state.get("demand_forecasts", {})

            for key in positions.keys():
                if key in forecasts:
                    # This would call ATPCalculatorAgent.calculate_atp
                    atp_results[key] = {
                        "current_atp": positions[key].get("atp", 0),
                        "risk_level": "medium",  # Calculated
                    }

            return {
                "atp_results": atp_results,
                "current_step": "atp_complete",
            }
        except Exception as e:
            logger.error(f"ATP calculation error: {e}")
            return {
                "errors": state.get("errors", []) + [f"ATP error: {str(e)}"],
            }

    async def _replenishment_node(self, state: InventoryState) -> Dict:
        """Execute the replenishment recommendation agent."""
        logger.info("Executing replenishment agent")

        try:
            actions = []
            atp_results = state.get("atp_results", {})

            for key, atp in atp_results.items():
                if atp.get("risk_level") in ["high", "critical"]:
                    warehouse_id, sku = key.split(":")
                    actions.append({
                        "action_type": "replenish",
                        "warehouse_id": warehouse_id,
                        "sku": sku,
                        "priority": "high" if atp["risk_level"] == "critical" else "medium",
                        "recommended_qty": 100,  # Would be calculated
                    })

            return {
                "replenishment_actions": actions,
                "current_step": "replenish_complete",
            }
        except Exception as e:
            logger.error(f"Replenishment error: {e}")
            return {
                "errors": state.get("errors", []) + [f"Replenishment error: {str(e)}"],
            }

    async def _checkpoint_node(self, state: InventoryState) -> Dict:
        """Save state checkpoint to Redis."""
        logger.info("Saving state checkpoint")

        checkpoint_key = f"orchestrator:checkpoint:{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        checkpoint_data = {
            "warehouse_ids": json.dumps(state.get("warehouse_ids", [])),
            "atp_results": json.dumps(state.get("atp_results", {})),
            "replenishment_actions": json.dumps(state.get("replenishment_actions", [])),
            "anomalies": json.dumps(state.get("anomalies", [])),
            "iteration_count": state.get("iteration_count", 0),
        }

        await self.redis.hset(checkpoint_key, mapping=checkpoint_data)
        await self.redis.expire(checkpoint_key, 86400 * 7)  # 7 day retention

        return {
            "last_checkpoint": checkpoint_key,
            "current_step": "checkpoint_complete",
        }

    async def _error_handler_node(self, state: InventoryState) -> Dict:
        """Handle errors and attempt recovery."""
        errors = state.get("errors", [])
        logger.error(f"Error handler invoked with {len(errors)} errors: {errors}")

        # Log errors for alerting
        for error in errors:
            await self.redis.xadd(
                "orchestrator:errors",
                {"error": error, "timestamp": datetime.now(timezone.utc).isoformat()},
                maxlen=1000,
            )

        return {
            "errors": [],  # Clear after logging
            "current_step": "error_handled",
        }

    async def run(
        self, warehouse_ids: List[str], skus: List[str], thread_id: str = "default"
    ) -> Dict:
        """Execute the orchestration workflow."""
        await self.connect()

        initial_state = {
            "messages": [HumanMessage(content="Starting inventory orchestration")],
            "current_step": "init",
            "warehouse_ids": warehouse_ids,
            "skus_to_process": skus,
            "inventory_positions": {},
            "demand_forecasts": {},
            "atp_results": {},
            "anomalies": [],
            "replenishment_actions": [],
            "errors": [],
            "iteration_count": 0,
            "last_checkpoint": "",
        }

        config = {"configurable": {"thread_id": thread_id}}

        try:
            final_state = await self.graph.ainvoke(initial_state, config)
            return {
                "status": "complete",
                "atp_results": final_state.get("atp_results", {}),
                "replenishment_actions": final_state.get("replenishment_actions", []),
                "anomalies": final_state.get("anomalies", []),
                "iterations": final_state.get("iteration_count", 0),
            }
        finally:
            await self.close()`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the multi-agent system with Docker, implement LangSmith tracing for agent observability, and set up Prometheus metrics for production monitoring.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose for Multi-Agent Deployment',
                description:
                  'Complete Docker Compose configuration for deploying the inventory agent system with Redis, Kafka, and observability stack.',
                code: `version: '3.8'

services:
  # Core Infrastructure
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:29093
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:29093
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_LOG_DIRS: /var/lib/kafka/data
      CLUSTER_ID: inventory-agents-cluster
    volumes:
      - kafka_data:/var/lib/kafka/data
    healthcheck:
      test: ["CMD", "kafka-topics", "--bootstrap-server", "localhost:9092", "--list"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Agent Services
  inventory-monitor-agent:
    build:
      context: .
      dockerfile: Dockerfile.agents
    command: python -m agents.inventory_monitor
    environment:
      - KAFKA_BROKERS=kafka:9092
      - KAFKA_TOPIC=wms.inventory.movements
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=inventory-agents
      - LOG_LEVEL=INFO
    depends_on:
      redis:
        condition: service_healthy
      kafka:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G

  orchestrator:
    build:
      context: .
      dockerfile: Dockerfile.agents
    command: python -m agents.orchestrator
    ports:
      - "8080:8080"
    environment:
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=inventory-agents
      - LOG_LEVEL=INFO
    depends_on:
      redis:
        condition: service_healthy
      inventory-monitor-agent:
        condition: service_started
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G

  # Observability Stack
  prometheus:
    image: prom/prometheus:v2.47.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=15d'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.1.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  kafka_data:
  prometheus_data:
  grafana_data:`,
              },
              {
                language: 'python',
                title: 'LangSmith Tracing & Prometheus Metrics',
                description:
                  'Observability integration with LangSmith for agent tracing and Prometheus metrics for production monitoring.',
                code: `import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from langsmith import Client
from langsmith.run_helpers import traceable

logger = logging.getLogger(__name__)

# Prometheus Metrics
AGENT_INVOCATIONS = Counter(
    "inventory_agent_invocations_total",
    "Total agent invocations",
    ["agent_name", "status"],
)

AGENT_LATENCY = Histogram(
    "inventory_agent_latency_seconds",
    "Agent execution latency",
    ["agent_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

INVENTORY_POSITIONS = Gauge(
    "inventory_positions_total",
    "Number of tracked inventory positions",
    ["warehouse_id"],
)

ATP_RISK_LEVEL = Gauge(
    "inventory_atp_risk_level",
    "ATP risk level (0=low, 1=medium, 2=high, 3=critical)",
    ["warehouse_id", "sku"],
)

ANOMALIES_DETECTED = Counter(
    "inventory_anomalies_detected_total",
    "Total anomalies detected",
    ["anomaly_type", "severity"],
)

REPLENISHMENT_ACTIONS = Counter(
    "inventory_replenishment_actions_total",
    "Total replenishment actions generated",
    ["action_type", "priority"],
)

ORCHESTRATOR_ITERATIONS = Counter(
    "inventory_orchestrator_iterations_total",
    "Total orchestrator iterations",
    ["status"],
)


@dataclass
class AgentMetrics:
    """Metrics container for a single agent execution."""
    agent_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def complete(self, status: str = "success") -> None:
        self.end_time = time.time()
        self.status = status

    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class ObservabilityManager:
    """Centralized observability for the agent system."""

    def __init__(
        self,
        langsmith_api_key: Optional[str] = None,
        langsmith_project: str = "inventory-agents",
        prometheus_port: int = 8000,
    ):
        self.langsmith_client = None
        if langsmith_api_key:
            self.langsmith_client = Client(api_key=langsmith_api_key)
        self.langsmith_project = langsmith_project
        self.prometheus_port = prometheus_port
        self._started = False

    def start_metrics_server(self) -> None:
        """Start Prometheus metrics HTTP server."""
        if not self._started:
            start_http_server(self.prometheus_port)
            self._started = True
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")

    @asynccontextmanager
    async def trace_agent(
        self,
        agent_name: str,
        inputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing agent execution."""
        metrics = AgentMetrics(
            agent_name=agent_name,
            start_time=time.time(),
            metadata=metadata or {},
        )

        try:
            yield metrics
            metrics.complete("success")
            AGENT_INVOCATIONS.labels(agent_name=agent_name, status="success").inc()
        except Exception as e:
            metrics.complete("error")
            metrics.error = str(e)
            AGENT_INVOCATIONS.labels(agent_name=agent_name, status="error").inc()
            raise
        finally:
            AGENT_LATENCY.labels(agent_name=agent_name).observe(metrics.duration)

            # Log to LangSmith if configured
            if self.langsmith_client:
                try:
                    self.langsmith_client.create_run(
                        name=agent_name,
                        run_type="chain",
                        inputs=inputs,
                        outputs={"status": metrics.status, "duration": metrics.duration},
                        error=metrics.error,
                        extra=metrics.metadata,
                        project_name=self.langsmith_project,
                    )
                except Exception as e:
                    logger.warning(f"Failed to log to LangSmith: {e}")

    def record_inventory_position(self, warehouse_id: str, position_count: int) -> None:
        """Record current inventory position count."""
        INVENTORY_POSITIONS.labels(warehouse_id=warehouse_id).set(position_count)

    def record_atp_risk(self, warehouse_id: str, sku: str, risk_level: str) -> None:
        """Record ATP risk level for a SKU."""
        risk_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        ATP_RISK_LEVEL.labels(
            warehouse_id=warehouse_id, sku=sku
        ).set(risk_map.get(risk_level, 0))

    def record_anomaly(self, anomaly_type: str, severity: str) -> None:
        """Record detected anomaly."""
        ANOMALIES_DETECTED.labels(
            anomaly_type=anomaly_type, severity=severity
        ).inc()

    def record_replenishment_action(self, action_type: str, priority: str) -> None:
        """Record generated replenishment action."""
        REPLENISHMENT_ACTIONS.labels(
            action_type=action_type, priority=priority
        ).inc()

    def record_orchestrator_iteration(self, status: str) -> None:
        """Record orchestrator iteration."""
        ORCHESTRATOR_ITERATIONS.labels(status=status).inc()


def traced_agent(agent_name: str):
    """Decorator for tracing agent methods with LangSmith."""
    def decorator(func: Callable):
        @wraps(func)
        @traceable(name=agent_name, run_type="chain")
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Usage example
async def main():
    observability = ObservabilityManager(
        langsmith_api_key="your-api-key",
        langsmith_project="inventory-agents",
        prometheus_port=8000,
    )
    observability.start_metrics_server()

    # Example: Trace an agent execution
    async with observability.trace_agent(
        agent_name="inventory_monitor",
        inputs={"warehouse_ids": ["WH001", "WH002"]},
        metadata={"batch_size": 100},
    ) as metrics:
        # Agent execution would happen here
        await asyncio.sleep(0.1)  # Simulated work

    # Record metrics
    observability.record_inventory_position("WH001", 1500)
    observability.record_atp_risk("WH001", "SKU-001", "high")
    observability.record_anomaly("sudden_drop", "high")

    logger.info("Observability demo complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())`,
              },
            ],
          },
        ],
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
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to automatically ingest spreadsheets from shared drives and email attachments, infer schemas, validate data quality, and generate consolidated reports without building a custom ETL pipeline.',
        estimatedMonthlyCost: '$120 - $200/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Google Sheets (Free)', 'Airtable ($20/mo)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Project44 API (Usage-based)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Configure Zapier to monitor shared drive folders and email attachments for new spreadsheets, extract their contents, and stage them in Google Sheets for AI-powered analysis.',
            toolsUsed: ['Zapier', 'Google Drive', 'Gmail', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Spreadsheet Ingestion Trigger',
                description:
                  'Zapier configuration to detect new Excel files in shared drives and email attachments, then extract to Google Sheets.',
                code: `{
  "workflow_name": "Spreadsheet Auto-Ingestion",
  "triggers": [
    {
      "trigger_id": "drive_trigger",
      "app": "Google Drive",
      "event": "New File in Folder",
      "folder_id": "{{shared_drive_supply_chain_folder}}",
      "file_types": [".xlsx", ".xls", ".csv"],
      "include_subfolders": true
    },
    {
      "trigger_id": "email_trigger",
      "app": "Gmail",
      "event": "New Attachment",
      "search_query": "from:(*@suppliers.com OR *@logistics.com) has:attachment filename:(xlsx OR csv)",
      "label": "supply-chain-data"
    }
  ],
  "action_1": {
    "app": "Filter by Zapier",
    "condition": {
      "field": "{{trigger.file_extension}}",
      "operator": "is_one_of",
      "values": [".xlsx", ".xls", ".csv"]
    }
  },
  "action_2": {
    "app": "Google Sheets",
    "event": "Create Spreadsheet from File",
    "file_url": "{{trigger.file_url}}",
    "destination_folder": "{{staging_folder_id}}",
    "spreadsheet_name": "STAGED_{{trigger.file_name}}_{{timestamp}}"
  },
  "action_3": {
    "app": "Google Sheets",
    "event": "Get All Rows",
    "spreadsheet_id": "{{action_2.spreadsheet_id}}",
    "worksheet": "Sheet1",
    "limit": 500
  },
  "action_4": {
    "app": "Formatter by Zapier",
    "event": "Utilities - Line Item to Text",
    "input": "{{action_3.rows}}",
    "separator": "\\n"
  },
  "action_5": {
    "app": "Google Sheets",
    "event": "Create Spreadsheet Row",
    "spreadsheet_id": "{{ingestion_log_sheet}}",
    "worksheet": "Ingestion_Log",
    "row_data": {
      "ingestion_id": "{{uuid}}",
      "source_type": "{{trigger.type}}",
      "original_filename": "{{trigger.file_name}}",
      "staged_sheet_id": "{{action_2.spreadsheet_id}}",
      "row_count": "{{action_3.row_count}}",
      "column_count": "{{action_3.column_count}}",
      "ingested_at": "{{current_timestamp}}",
      "status": "staged"
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
              'Use ChatGPT to automatically infer spreadsheet schemas, detect data quality issues, identify duplicate or conflicting data across files, and suggest standardization rules.',
            toolsUsed: ['ChatGPT Plus', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Schema Inference & Data Quality Prompt Template',
                description:
                  'Structured prompt for ChatGPT to analyze spreadsheet structure, infer schemas, and detect data quality issues.',
                code: `system_prompt: |
  You are a data engineer specializing in spreadsheet analysis and schema inference.
  Your job is to analyze raw spreadsheet data, infer proper schemas, detect quality
  issues, and recommend standardization rules for enterprise data governance.

user_prompt_template: |
  ## Spreadsheet Analysis Request

  **File Name:** {{file_name}}
  **Source:** {{source_type}} ({{source_path}})
  **Ingestion Time:** {{ingestion_timestamp}}

  ### Raw Data Sample (First 50 Rows)
  \`\`\`csv
  {{spreadsheet_data_csv}}
  \`\`\`

  ### Analysis Tasks:

  1. **Schema Inference**
     - Identify column names and infer data types (string, integer, float, date, boolean)
     - Detect which columns appear to be primary keys or foreign keys
     - Identify columns that should be required vs nullable
     - Suggest appropriate column name standardization (snake_case, no spaces)

  2. **Data Quality Assessment**
     - Count and percentage of null/empty values per column
     - Identify columns with inconsistent data types (mixed strings/numbers)
     - Detect duplicate rows based on likely key columns
     - Flag columns with suspicious values (outliers, invalid formats)
     - Check date columns for impossible values (future dates, dates before 1900)

  3. **Standardization Recommendations**
     - Suggest data type conversions needed
     - Recommend value mappings for inconsistent categorical values
     - Identify columns that need normalization (e.g., phone numbers, addresses)
     - Suggest validation rules for each column

  4. **Relationship Detection**
     - Compare column names against known supply chain entities:
       - supplier_id, supplier_name, vendor_code
       - sku, product_id, item_number
       - warehouse_id, location_code
       - po_number, order_id
     - Flag potential join keys for data consolidation

  ### Output Format
  Provide a structured JSON response with:
  \`\`\`json
  {
    "inferred_schema": {
      "columns": [
        {
          "original_name": "string",
          "standardized_name": "string",
          "inferred_type": "string|integer|float|date|boolean",
          "nullable": boolean,
          "is_key": boolean,
          "validation_rule": "string"
        }
      ]
    },
    "quality_issues": [
      {
        "column": "string",
        "issue_type": "null_values|type_mismatch|duplicates|outliers|invalid_format",
        "severity": "low|medium|high|critical",
        "affected_rows": number,
        "sample_values": ["string"],
        "recommendation": "string"
      }
    ],
    "standardization_rules": [
      {
        "column": "string",
        "rule_type": "type_conversion|value_mapping|normalization",
        "details": "string"
      }
    ],
    "detected_entity_type": "supplier_data|inventory|orders|shipments|unknown",
    "join_key_candidates": ["string"]
  }
  \`\`\`

expected_output_format: json
temperature: 0.2
max_tokens: 4000`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Automate the complete ingestion workflow with Zapier to process new spreadsheets, apply AI-inferred schemas, validate data quality, and deliver consolidated reports to stakeholders.',
            toolsUsed: ['Zapier', 'Airtable', 'Slack', 'Gmail'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Schema Validation & Consolidation Workflow',
                description:
                  'Complete Zapier workflow that applies AI schema inference, validates data quality, and consolidates approved data into Airtable.',
                code: `{
  "workflow_name": "Spreadsheet Validation & Consolidation",
  "trigger": {
    "app": "Google Sheets",
    "event": "New Row in Spreadsheet",
    "spreadsheet_id": "{{ingestion_log_sheet}}",
    "worksheet": "Ingestion_Log",
    "filter": {
      "status": "staged"
    }
  },
  "action_1": {
    "app": "Google Sheets",
    "event": "Get All Rows",
    "spreadsheet_id": "{{trigger.staged_sheet_id}}",
    "worksheet": "Sheet1",
    "limit": 100
  },
  "action_2": {
    "app": "ChatGPT",
    "event": "Conversation",
    "model": "gpt-4",
    "system_message": "You are a data engineer specializing in spreadsheet analysis...",
    "user_message": "Analyze this spreadsheet data:\\n{{action_1.rows_csv}}",
    "temperature": 0.2
  },
  "action_3": {
    "app": "Formatter by Zapier",
    "event": "Text - Extract JSON",
    "input": "{{action_2.response}}",
    "json_path": "$.quality_issues"
  },
  "action_4": {
    "app": "Filter by Zapier",
    "condition": {
      "field": "{{action_3.critical_issues_count}}",
      "operator": "equals",
      "value": 0
    },
    "continue_if_true": true
  },
  "action_5_approved": {
    "app": "Airtable",
    "event": "Create Records",
    "base_id": "{{supply_chain_data_base}}",
    "table": "{{action_2.detected_entity_type}}",
    "records": "{{action_1.rows}}",
    "field_mapping": "{{action_2.inferred_schema.columns}}"
  },
  "action_5_rejected": {
    "app": "Slack",
    "event": "Send Channel Message",
    "channel": "#data-quality-alerts",
    "message_blocks": [
      {
        "type": "header",
        "text": ":warning: Spreadsheet Requires Review"
      },
      {
        "type": "section",
        "fields": [
          {"type": "mrkdwn", "text": "*File:* {{trigger.original_filename}}"},
          {"type": "mrkdwn", "text": "*Source:* {{trigger.source_type}}"},
          {"type": "mrkdwn", "text": "*Critical Issues:* {{action_3.critical_issues_count}}"}
        ]
      },
      {
        "type": "section",
        "text": "*Issues Found:*\\n{{action_3.quality_issues_summary}}"
      },
      {
        "type": "actions",
        "elements": [
          {
            "type": "button",
            "text": "Review in Sheets",
            "url": "https://docs.google.com/spreadsheets/d/{{trigger.staged_sheet_id}}"
          },
          {
            "type": "button",
            "text": "Override & Approve",
            "action_id": "approve_{{trigger.ingestion_id}}"
          }
        ]
      }
    ]
  },
  "action_6": {
    "app": "Google Sheets",
    "event": "Update Spreadsheet Row",
    "spreadsheet_id": "{{ingestion_log_sheet}}",
    "worksheet": "Ingestion_Log",
    "row": "{{trigger.row_number}}",
    "updates": {
      "status": "{{action_4.passed ? 'approved' : 'review_required'}}",
      "schema_json": "{{action_2.inferred_schema}}",
      "quality_score": "{{action_3.quality_score}}",
      "processed_at": "{{current_timestamp}}"
    }
  },
  "action_7": {
    "app": "Gmail",
    "event": "Send Email",
    "to": "{{file_owner_email}}",
    "subject": "Spreadsheet Processing Complete: {{trigger.original_filename}}",
    "body_html": "<h2>Processing Summary</h2><p>Status: {{action_4.passed ? 'Approved' : 'Review Required'}}</p><p>{{action_2.summary}}</p>"
  }
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent system using CrewAI and LangGraph that autonomously discovers spreadsheets across the organization, infers and validates schemas using ML, detects duplicates and conflicts across files, and migrates validated data into a governed data platform with full lineage tracking.',
        estimatedMonthlyCost: '$700 - $1,400/month',
        architecture:
          'A Supervisor agent coordinates four specialist agents: Discovery Agent scans file systems and email for spreadsheets, Schema Inference Agent uses ML to detect data types and relationships, Validation Agent checks data quality and identifies conflicts across files, and Migration Agent loads validated data into the target platform with lineage metadata.',
        agents: [
          {
            name: 'Discovery Agent',
            role: 'Spreadsheet Asset Discovery',
            goal: 'Continuously scan shared drives, email attachments, and collaboration platforms to discover and catalog all spreadsheet assets with ownership and usage metadata.',
            tools: ['Google Drive API', 'SharePoint API', 'Gmail API', 'File System Scanner', 'Redis Cache'],
          },
          {
            name: 'Schema Inference Agent',
            role: 'Intelligent Schema Detection',
            goal: 'Analyze spreadsheet structure and content to infer accurate schemas, detect entity types, identify key columns, and suggest standardized naming conventions.',
            tools: ['pandas', 'dataprep', 'Schema Registry', 'Entity Resolution Model'],
          },
          {
            name: 'Validation Agent',
            role: 'Data Quality Guardian',
            goal: 'Validate data quality against inferred schemas, detect duplicates and conflicts across multiple source files, and flag records requiring human review.',
            tools: ['Great Expectations', 'Dedupe.io', 'Conflict Resolution Rules', 'PostgreSQL'],
          },
          {
            name: 'Migration Agent',
            role: 'Data Platform Loader',
            goal: 'Transform validated spreadsheet data into the target schema and load into the governed data platform with complete lineage tracking and audit trail.',
            tools: ['dbt', 'PostgreSQL', 'Lineage Tracker', 'Slack Notifier'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Sequential',
          stateManagement: 'PostgreSQL-backed state with file-level checkpointing and full audit logging',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system with CrewAI, establishing each agent\'s role, goals, tools, and the sequential workflow for spreadsheet discovery, validation, and migration.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'CrewAI Spreadsheet Processing Agent Definitions',
                description:
                  'Complete CrewAI agent setup for the four-agent spreadsheet governance system with detailed role definitions.',
                code: `from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpreadsheetAgentConfig:
    """Configuration for spreadsheet processing agents."""

    def __init__(self, openai_api_key: str, model: str = "gpt-4-turbo"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0.1,
        )

    def create_discovery_agent(self, tools: List[Any]) -> Agent:
        """Create the spreadsheet discovery agent."""
        return Agent(
            role="Spreadsheet Discovery Specialist",
            goal="""Continuously scan shared drives, email attachments, SharePoint,
            and collaboration platforms to discover ALL spreadsheet assets used
            for supply chain operations. Catalog each file with ownership, last
            modified date, apparent purpose, and downstream dependencies.""",
            backstory="""You are a data governance expert who has audited data
            practices at Fortune 500 companies. You know that shadow IT spreadsheets
            hide in the most unexpected places - personal drives, email threads,
            and forgotten SharePoint sites. You leave no stone unturned in your
            quest to catalog every spreadsheet that touches critical business data.""",
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=20,
        )

    def create_schema_inference_agent(self, tools: List[Any]) -> Agent:
        """Create the schema inference agent."""
        return Agent(
            role="Schema Inference Engineer",
            goal="""Analyze spreadsheet structure and content to infer accurate
            schemas. Detect data types, identify primary/foreign keys, recognize
            entity types (supplier, product, order, shipment), and suggest
            standardized column names that align with the enterprise data model.""",
            backstory="""You are a data modeling expert with deep experience in
            supply chain master data management. You've seen every possible way
            that business users name columns - from cryptic abbreviations to
            verbose descriptions. You can identify a supplier ID whether it's
            called 'VendorNo', 'SUPP_CODE', or 'Supplier Number'. You understand
            the relationships between supply chain entities and can detect when
            a spreadsheet is really a join of multiple tables.""",
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            max_iter=15,
        )

    def create_validation_agent(self, tools: List[Any]) -> Agent:
        """Create the data validation agent."""
        return Agent(
            role="Data Quality Validator",
            goal="""Validate data quality against inferred schemas and business
            rules. Detect duplicates within and across spreadsheets, identify
            conflicting values for the same entity, flag data quality issues,
            and determine which records can be auto-migrated vs. need human review.""",
            backstory="""You are a data quality expert who has cleaned millions
            of records in enterprise migrations. You know that spreadsheet data
            is notoriously messy - inconsistent formats, typos, stale records,
            and outright conflicts between different versions. You apply statistical
            techniques to detect anomalies and use fuzzy matching to identify
            near-duplicates. You're conservative about auto-approving data - when
            in doubt, flag for human review.""",
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            max_iter=15,
        )

    def create_migration_agent(self, tools: List[Any]) -> Agent:
        """Create the data migration agent."""
        return Agent(
            role="Data Platform Migration Specialist",
            goal="""Transform validated spreadsheet data into the target schema
            and load into the governed data platform. Apply standardization rules,
            resolve foreign key references, maintain complete lineage tracking,
            and ensure audit trail compliance for SOX and regulatory requirements.""",
            backstory="""You are a data engineering expert who has led enterprise
            data platform migrations. You understand that migration isn't just
            about moving data - it's about establishing trust. Every record needs
            provenance: where it came from, when it was loaded, what transformations
            were applied, and who approved it. You build pipelines that auditors
            love because the lineage is crystal clear.""",
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=12,
        )


def create_spreadsheet_crew(config: SpreadsheetAgentConfig, tools: Dict[str, List[Any]]) -> Crew:
    """Create the complete spreadsheet processing crew."""

    discovery_agent = config.create_discovery_agent(tools["discovery"])
    schema_agent = config.create_schema_inference_agent(tools["schema"])
    validation_agent = config.create_validation_agent(tools["validation"])
    migration_agent = config.create_migration_agent(tools["migration"])

    # Define sequential tasks
    discovery_task = Task(
        description="""Scan the configured source locations for spreadsheet files:
        1. Search Google Drive folders: {drive_folders}
        2. Search SharePoint sites: {sharepoint_sites}
        3. Search recent email attachments matching patterns
        4. For each file, extract: path, name, owner, last modified, size, sheet names

        Prioritize files by:
        - Last modified (more recent = higher priority)
        - Size (larger = likely more important)
        - Location (shared folders > personal drives)

        Output a ranked list of files to process.""",
        expected_output="JSON list of discovered files with metadata and priority scores",
        agent=discovery_agent,
    )

    schema_task = Task(
        description="""For each discovered spreadsheet, analyze structure and infer schema:
        1. Read first 100 rows to understand data patterns
        2. Detect column data types (string, int, float, date, boolean)
        3. Identify potential key columns (unique values, naming patterns)
        4. Recognize entity type (supplier, product, inventory, order, shipment)
        5. Suggest standardized column names per enterprise naming convention
        6. Detect if sheet appears to be a pivot table or summary (skip these)

        Discovered files: {discovered_files}""",
        expected_output="JSON with inferred schema per file including column mappings",
        agent=schema_agent,
        context=[discovery_task],
    )

    validation_task = Task(
        description="""Validate data quality for files with inferred schemas:
        1. Check for null/empty values in required columns
        2. Validate data types match inferred schema
        3. Run cross-file duplicate detection on key columns
        4. Identify conflicting values (same key, different data)
        5. Flag records that fail validation rules
        6. Calculate quality score (0-100) per file

        Files with schemas: {schema_results}
        Quality threshold for auto-approval: 90""",
        expected_output="JSON with validation results, quality scores, and conflict details",
        agent=validation_agent,
        context=[schema_task],
    )

    migration_task = Task(
        description="""Migrate validated data to the target platform:
        1. For files with quality score >= 90: auto-migrate
        2. Apply column transformations per inferred schema
        3. Resolve foreign key references against master data
        4. Insert into target tables with lineage metadata
        5. Generate migration report with row counts and lineage links
        6. Archive source files to processed folder

        Validation results: {validation_results}
        Target database: {target_database}""",
        expected_output="JSON migration report with success/failure counts and lineage IDs",
        agent=migration_agent,
        context=[validation_task],
    )

    return Crew(
        agents=[discovery_agent, schema_agent, validation_agent, migration_agent],
        tasks=[discovery_task, schema_task, validation_task, migration_task],
        process=Process.sequential,
        verbose=True,
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the Discovery Agent with multi-platform file scanning, metadata extraction, and intelligent prioritization for spreadsheet cataloging.',
            toolsUsed: ['Google Drive API', 'SharePoint API', 'asyncio', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Spreadsheet Discovery Agent Implementation',
                description:
                  'Production-ready implementation of the Discovery Agent with multi-platform scanning and prioritization.',
                code: `import asyncio
import hashlib
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import aiofiles
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class SpreadsheetAsset:
    """Represents a discovered spreadsheet asset."""
    asset_id: str
    source_type: str  # 'google_drive', 'sharepoint', 'local', 'email'
    file_path: str
    file_name: str
    file_extension: str
    owner_email: Optional[str]
    last_modified: str
    file_size_bytes: int
    sheet_names: List[str]
    content_hash: str
    priority_score: float
    discovered_at: str
    status: str = "discovered"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DiscoveryConfig:
    """Configuration for spreadsheet discovery."""
    google_drive_folders: List[str]
    sharepoint_sites: List[str]
    local_paths: List[str]
    email_search_days: int = 30
    supported_extensions: tuple = (".xlsx", ".xls", ".csv")
    max_file_size_mb: int = 100
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "recency_days": 0.4,
        "size_mb": 0.2,
        "shared_location": 0.3,
        "modification_frequency": 0.1,
    })


class SpreadsheetDiscoveryAgent:
    """Agent for discovering spreadsheet assets across platforms."""

    def __init__(
        self,
        config: DiscoveryConfig,
        google_credentials: Optional[Credentials] = None,
        redis_url: str = "redis://localhost:6379",
    ):
        self.config = config
        self.google_creds = google_credentials
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.asset_prefix = "spreadsheet:asset:"
        self.scan_log_prefix = "spreadsheet:scan:"

    async def connect(self) -> None:
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)
        logger.info("Discovery agent connected to Redis")

    async def close(self) -> None:
        if self.redis:
            await self.redis.close()

    def _calculate_priority_score(
        self,
        last_modified: datetime,
        file_size_bytes: int,
        is_shared: bool,
        modification_count: int = 1,
    ) -> float:
        """Calculate priority score for processing order."""
        weights = self.config.priority_weights
        now = datetime.now(timezone.utc)

        # Recency score (0-1): more recent = higher
        days_old = (now - last_modified).days
        recency_score = max(0, 1 - (days_old / 365))

        # Size score (0-1): larger files often more important
        size_mb = file_size_bytes / (1024 * 1024)
        size_score = min(1, size_mb / 10)  # Cap at 10MB

        # Shared location score
        shared_score = 1.0 if is_shared else 0.3

        # Modification frequency score
        freq_score = min(1, modification_count / 10)

        priority = (
            weights["recency_days"] * recency_score +
            weights["size_mb"] * size_score +
            weights["shared_location"] * shared_score +
            weights["modification_frequency"] * freq_score
        )

        return round(priority, 3)

    def _compute_content_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of file content."""
        return hashlib.sha256(content).hexdigest()[:16]

    async def scan_google_drive(self, folder_id: str) -> List[SpreadsheetAsset]:
        """Scan a Google Drive folder for spreadsheets."""
        if not self.google_creds:
            logger.warning("Google credentials not configured, skipping Drive scan")
            return []

        assets = []
        service = build("drive", "v3", credentials=self.google_creds)

        try:
            # Build query for spreadsheet files
            extensions = " or ".join(
                f"name contains '{ext}'" for ext in self.config.supported_extensions
            )
            query = f"'{folder_id}' in parents and ({extensions}) and trashed = false"

            results = service.files().list(
                q=query,
                fields="files(id, name, mimeType, owners, modifiedTime, size, parents)",
                pageSize=100,
            ).execute()

            for file in results.get("files", []):
                last_modified = datetime.fromisoformat(
                    file["modifiedTime"].replace("Z", "+00:00")
                )
                file_size = int(file.get("size", 0))

                if file_size > self.config.max_file_size_mb * 1024 * 1024:
                    logger.info(f"Skipping large file: {file['name']} ({file_size} bytes)")
                    continue

                asset = SpreadsheetAsset(
                    asset_id=f"gdrive_{file['id']}",
                    source_type="google_drive",
                    file_path=f"https://drive.google.com/file/d/{file['id']}",
                    file_name=file["name"],
                    file_extension=Path(file["name"]).suffix.lower(),
                    owner_email=file["owners"][0]["emailAddress"] if file.get("owners") else None,
                    last_modified=last_modified.isoformat(),
                    file_size_bytes=file_size,
                    sheet_names=[],  # Would be populated after download
                    content_hash="",  # Would be computed after download
                    priority_score=self._calculate_priority_score(
                        last_modified, file_size, is_shared=True
                    ),
                    discovered_at=datetime.now(timezone.utc).isoformat(),
                    metadata={"drive_id": file["id"], "parents": file.get("parents", [])},
                )
                assets.append(asset)

        except Exception as e:
            logger.error(f"Error scanning Google Drive folder {folder_id}: {e}")

        return assets

    async def scan_local_path(self, scan_path: str) -> List[SpreadsheetAsset]:
        """Scan a local filesystem path for spreadsheets."""
        assets = []
        root = Path(scan_path)

        if not root.exists():
            logger.warning(f"Local path does not exist: {scan_path}")
            return assets

        try:
            for file_path in root.rglob("*"):
                if file_path.suffix.lower() not in self.config.supported_extensions:
                    continue

                if not file_path.is_file():
                    continue

                stat = file_path.stat()
                file_size = stat.st_size

                if file_size > self.config.max_file_size_mb * 1024 * 1024:
                    logger.info(f"Skipping large file: {file_path} ({file_size} bytes)")
                    continue

                last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

                # Compute content hash
                async with aiofiles.open(file_path, "rb") as f:
                    content = await f.read(1024 * 1024)  # First 1MB for hash
                content_hash = self._compute_content_hash(content)

                # Check if shared location
                is_shared = any(
                    shared in str(file_path).lower()
                    for shared in ["shared", "team", "common", "public"]
                )

                asset = SpreadsheetAsset(
                    asset_id=f"local_{content_hash}",
                    source_type="local",
                    file_path=str(file_path),
                    file_name=file_path.name,
                    file_extension=file_path.suffix.lower(),
                    owner_email=None,
                    last_modified=last_modified.isoformat(),
                    file_size_bytes=file_size,
                    sheet_names=[],
                    content_hash=content_hash,
                    priority_score=self._calculate_priority_score(
                        last_modified, file_size, is_shared=is_shared
                    ),
                    discovered_at=datetime.now(timezone.utc).isoformat(),
                    metadata={"local_path": str(file_path)},
                )
                assets.append(asset)

        except Exception as e:
            logger.error(f"Error scanning local path {scan_path}: {e}")

        return assets

    async def run_discovery(self) -> List[SpreadsheetAsset]:
        """Run full discovery across all configured sources."""
        all_assets = []

        # Scan Google Drive folders
        for folder_id in self.config.google_drive_folders:
            logger.info(f"Scanning Google Drive folder: {folder_id}")
            assets = await self.scan_google_drive(folder_id)
            all_assets.extend(assets)

        # Scan local paths
        for local_path in self.config.local_paths:
            logger.info(f"Scanning local path: {local_path}")
            assets = await self.scan_local_path(local_path)
            all_assets.extend(assets)

        # Sort by priority score (highest first)
        all_assets.sort(key=lambda a: a.priority_score, reverse=True)

        # Persist to Redis
        for asset in all_assets:
            key = f"{self.asset_prefix}{asset.asset_id}"
            await self.redis.hset(key, mapping=asset.to_dict())
            await self.redis.expire(key, 86400 * 30)  # 30 day retention

        # Log scan summary
        scan_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        await self.redis.hset(
            f"{self.scan_log_prefix}{scan_id}",
            mapping={
                "scan_id": scan_id,
                "total_assets": len(all_assets),
                "google_drive_count": len([a for a in all_assets if a.source_type == "google_drive"]),
                "local_count": len([a for a in all_assets if a.source_type == "local"]),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        logger.info(f"Discovery complete: {len(all_assets)} assets found")
        return all_assets`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Schema Inference and Validation agents with ML-based type detection, entity recognition, and cross-file conflict detection.',
            toolsUsed: ['pandas', 'dataprep', 'dedupe', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Schema Inference & Validation Agents',
                description:
                  'Implementation of the Schema Inference Agent with ML type detection and the Validation Agent with duplicate/conflict detection.',
                code: `import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from langchain.tools import tool
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class ColumnSchema:
    """Inferred schema for a single column."""
    original_name: str
    standardized_name: str
    inferred_type: str  # string, integer, float, date, boolean
    nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    unique_ratio: float
    null_ratio: float
    sample_values: List[str]
    validation_pattern: Optional[str] = None


@dataclass
class InferredSchema:
    """Complete inferred schema for a spreadsheet."""
    asset_id: str
    entity_type: str  # supplier, product, inventory, order, shipment, unknown
    columns: List[ColumnSchema]
    row_count: int
    confidence_score: float
    join_key_candidates: List[str]
    detected_relationships: List[Dict[str, str]]
    inferred_at: str


@dataclass
class ValidationResult:
    """Validation result for a spreadsheet."""
    asset_id: str
    quality_score: float  # 0-100
    total_rows: int
    valid_rows: int
    issues: List[Dict[str, Any]]
    duplicates_found: int
    conflicts_found: int
    auto_approve: bool
    validated_at: str


class SchemaInferenceAgent:
    """Agent for inferring spreadsheet schemas using ML techniques."""

    # Patterns for common supply chain entity columns
    ENTITY_PATTERNS = {
        "supplier": ["supplier", "vendor", "supp", "vend"],
        "product": ["sku", "product", "item", "part", "upc", "gtin"],
        "order": ["order", "po", "purchase", "sales_order"],
        "shipment": ["shipment", "tracking", "container", "bol", "awb"],
        "warehouse": ["warehouse", "location", "wh", "dc", "facility"],
    }

    TYPE_PATTERNS = {
        "email": r"^[\\w.-]+@[\\w.-]+\\.\\w+$",
        "phone": r"^[\\+]?[\\d\\s\\-\\(\\)]{7,}$",
        "date_iso": r"^\\d{4}-\\d{2}-\\d{2}",
        "currency": r"^\\$?[\\d,]+\\.?\\d*$",
        "percentage": r"^\\d+\\.?\\d*%$",
    }

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None

    async def connect(self) -> None:
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)

    async def close(self) -> None:
        if self.redis:
            await self.redis.close()

    def _standardize_column_name(self, name: str) -> str:
        """Convert column name to snake_case standard."""
        # Remove special characters, convert to lowercase
        clean = re.sub(r"[^\\w\\s]", "", str(name).lower())
        # Replace spaces and multiple underscores with single underscore
        clean = re.sub(r"[\\s_]+", "_", clean).strip("_")
        return clean

    def _infer_column_type(self, series: pd.Series) -> str:
        """Infer the data type of a column."""
        # Drop nulls for type inference
        non_null = series.dropna()
        if len(non_null) == 0:
            return "string"

        # Check if already numeric
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                return "integer"
            return "float"

        # Check if boolean
        unique_vals = set(non_null.astype(str).str.lower())
        if unique_vals.issubset({"true", "false", "yes", "no", "1", "0", "y", "n"}):
            return "boolean"

        # Try to parse as date
        try:
            pd.to_datetime(non_null.head(100), errors="raise")
            return "date"
        except (ValueError, TypeError):
            pass

        # Try to parse as numeric
        try:
            pd.to_numeric(non_null.head(100).str.replace(",", "").str.replace("$", ""))
            return "float"
        except (ValueError, TypeError):
            pass

        return "string"

    def _detect_entity_type(self, columns: List[str]) -> Tuple[str, float]:
        """Detect the likely entity type based on column names."""
        column_set = set(col.lower() for col in columns)
        scores = {}

        for entity, patterns in self.ENTITY_PATTERNS.items():
            matches = sum(
                1 for col in column_set
                for pattern in patterns
                if pattern in col
            )
            if matches > 0:
                scores[entity] = matches / len(patterns)

        if not scores:
            return "unknown", 0.0

        best_entity = max(scores, key=scores.get)
        return best_entity, scores[best_entity]

    def _identify_key_columns(
        self, df: pd.DataFrame, columns: List[ColumnSchema]
    ) -> Tuple[List[str], List[str]]:
        """Identify potential primary and foreign key columns."""
        primary_candidates = []
        foreign_candidates = []

        for col_schema in columns:
            col_name = col_schema.original_name

            # High uniqueness suggests primary key
            if col_schema.unique_ratio > 0.95 and col_schema.null_ratio < 0.01:
                # Check for common key naming patterns
                std_name = col_schema.standardized_name
                if any(pattern in std_name for pattern in ["_id", "_code", "_no", "_number"]):
                    primary_candidates.append(std_name)

            # Check for foreign key patterns
            for entity, patterns in self.ENTITY_PATTERNS.items():
                if any(pattern in col_schema.standardized_name for pattern in patterns):
                    if col_schema.unique_ratio < 0.5:  # Many-to-one relationship
                        foreign_candidates.append(col_schema.standardized_name)
                        break

        return primary_candidates, foreign_candidates

    async def infer_schema(
        self, asset_id: str, df: pd.DataFrame
    ) -> InferredSchema:
        """Infer complete schema for a spreadsheet."""
        columns = []

        for col_name in df.columns:
            series = df[col_name]
            inferred_type = self._infer_column_type(series)
            non_null_count = series.notna().sum()
            unique_count = series.nunique()

            # Get sample values
            sample_values = (
                series.dropna()
                .astype(str)
                .head(5)
                .tolist()
            )

            col_schema = ColumnSchema(
                original_name=str(col_name),
                standardized_name=self._standardize_column_name(col_name),
                inferred_type=inferred_type,
                nullable=series.isna().any(),
                is_primary_key=False,  # Updated below
                is_foreign_key=False,  # Updated below
                unique_ratio=unique_count / len(df) if len(df) > 0 else 0,
                null_ratio=1 - (non_null_count / len(df)) if len(df) > 0 else 0,
                sample_values=sample_values,
            )
            columns.append(col_schema)

        # Detect entity type
        entity_type, entity_confidence = self._detect_entity_type(
            [c.standardized_name for c in columns]
        )

        # Identify keys
        primary_keys, foreign_keys = self._identify_key_columns(df, columns)

        # Update key flags
        for col in columns:
            if col.standardized_name in primary_keys:
                col.is_primary_key = True
            if col.standardized_name in foreign_keys:
                col.is_foreign_key = True

        schema = InferredSchema(
            asset_id=asset_id,
            entity_type=entity_type,
            columns=columns,
            row_count=len(df),
            confidence_score=entity_confidence,
            join_key_candidates=primary_keys + foreign_keys,
            detected_relationships=[],  # Would be populated by cross-file analysis
            inferred_at=datetime.now(timezone.utc).isoformat(),
        )

        return schema


class ValidationAgent:
    """Agent for validating data quality and detecting conflicts."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.quality_weights = {
            "null_ratio": 0.3,
            "type_consistency": 0.3,
            "duplicate_ratio": 0.2,
            "format_compliance": 0.2,
        }

    async def connect(self) -> None:
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)

    async def close(self) -> None:
        if self.redis:
            await self.redis.close()

    def _check_type_consistency(self, series: pd.Series, expected_type: str) -> float:
        """Check what percentage of values match the expected type."""
        if expected_type == "string":
            return 1.0  # Everything can be a string

        non_null = series.dropna()
        if len(non_null) == 0:
            return 1.0

        try:
            if expected_type == "integer":
                pd.to_numeric(non_null, errors="raise").astype(int)
            elif expected_type == "float":
                pd.to_numeric(non_null, errors="raise")
            elif expected_type == "date":
                pd.to_datetime(non_null, errors="raise")
            elif expected_type == "boolean":
                valid_bools = {"true", "false", "yes", "no", "1", "0", "y", "n"}
                matches = non_null.astype(str).str.lower().isin(valid_bools)
                return matches.sum() / len(non_null)
            return 1.0
        except (ValueError, TypeError):
            return 0.5  # Partial compliance

    def _find_duplicates(
        self, df: pd.DataFrame, key_columns: List[str]
    ) -> Tuple[int, List[Dict]]:
        """Find duplicate rows based on key columns."""
        if not key_columns:
            return 0, []

        valid_keys = [k for k in key_columns if k in df.columns]
        if not valid_keys:
            return 0, []

        duplicated = df.duplicated(subset=valid_keys, keep=False)
        dup_count = duplicated.sum()

        # Get sample duplicate groups
        dup_samples = []
        if dup_count > 0:
            dup_groups = df[duplicated].groupby(valid_keys).size()
            for keys, count in dup_groups.head(5).items():
                dup_samples.append({
                    "key_values": dict(zip(valid_keys, keys if isinstance(keys, tuple) else [keys])),
                    "duplicate_count": int(count),
                })

        return int(dup_count), dup_samples

    async def validate(
        self,
        asset_id: str,
        df: pd.DataFrame,
        schema: InferredSchema,
    ) -> ValidationResult:
        """Validate data quality against inferred schema."""
        issues = []
        total_rows = len(df)
        valid_rows = total_rows

        # Check each column
        null_scores = []
        type_scores = []

        for col_schema in schema.columns:
            col_name = col_schema.original_name
            if col_name not in df.columns:
                continue

            series = df[col_name]

            # Null check
            null_count = series.isna().sum()
            null_ratio = null_count / total_rows if total_rows > 0 else 0
            null_scores.append(1 - null_ratio)

            if null_ratio > 0.5 and not col_schema.nullable:
                issues.append({
                    "column": col_name,
                    "issue_type": "high_null_ratio",
                    "severity": "high" if null_ratio > 0.8 else "medium",
                    "details": f"{null_ratio:.1%} null values",
                    "affected_rows": int(null_count),
                })

            # Type consistency check
            type_score = self._check_type_consistency(series, col_schema.inferred_type)
            type_scores.append(type_score)

            if type_score < 0.9:
                issues.append({
                    "column": col_name,
                    "issue_type": "type_mismatch",
                    "severity": "medium",
                    "details": f"Expected {col_schema.inferred_type}, {type_score:.1%} compliant",
                    "affected_rows": int((1 - type_score) * total_rows),
                })

        # Duplicate check
        key_cols = [c.standardized_name for c in schema.columns if c.is_primary_key]
        dup_count, dup_samples = self._find_duplicates(df, key_cols)

        if dup_count > 0:
            issues.append({
                "column": ",".join(key_cols),
                "issue_type": "duplicates",
                "severity": "high",
                "details": f"{dup_count} duplicate rows on key columns",
                "affected_rows": dup_count,
                "samples": dup_samples,
            })

        # Calculate quality score
        avg_null_score = np.mean(null_scores) if null_scores else 1.0
        avg_type_score = np.mean(type_scores) if type_scores else 1.0
        dup_score = 1 - (dup_count / total_rows) if total_rows > 0 else 1.0

        quality_score = (
            self.quality_weights["null_ratio"] * avg_null_score +
            self.quality_weights["type_consistency"] * avg_type_score +
            self.quality_weights["duplicate_ratio"] * dup_score +
            self.quality_weights["format_compliance"] * 1.0  # Placeholder
        ) * 100

        # Count valid rows (no critical issues)
        critical_issues = [i for i in issues if i["severity"] == "critical"]
        if critical_issues:
            valid_rows = total_rows - sum(i["affected_rows"] for i in critical_issues)

        return ValidationResult(
            asset_id=asset_id,
            quality_score=round(quality_score, 1),
            total_rows=total_rows,
            valid_rows=max(0, valid_rows),
            issues=issues,
            duplicates_found=dup_count,
            conflicts_found=0,  # Would be from cross-file analysis
            auto_approve=quality_score >= 90 and len(critical_issues) == 0,
            validated_at=datetime.now(timezone.utc).isoformat(),
        )`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement the LangGraph state machine that orchestrates the sequential workflow from discovery through migration with checkpoint and error handling.',
            toolsUsed: ['LangGraph', 'PostgreSQL', 'asyncio'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Sequential Orchestration',
                description:
                  'Complete LangGraph implementation for orchestrating the spreadsheet processing pipeline with PostgreSQL-backed state.',
                code: `import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Dict, List, Literal, Optional, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import asyncpg

logger = logging.getLogger(__name__)


class SpreadsheetState(TypedDict):
    """State schema for the spreadsheet processing workflow."""
    messages: Annotated[Sequence[BaseMessage], "Conversation messages"]
    current_phase: str  # discovery, schema, validation, migration, complete
    discovered_assets: List[Dict]
    schemas_inferred: Dict[str, Dict]
    validation_results: Dict[str, Dict]
    migration_results: Dict[str, Dict]
    assets_pending: List[str]
    assets_approved: List[str]
    assets_rejected: List[str]
    errors: List[Dict]
    run_id: str
    started_at: str
    last_checkpoint: str


class SpreadsheetOrchestrator:
    """LangGraph-based orchestrator for spreadsheet processing pipeline."""

    def __init__(
        self,
        openai_api_key: str,
        postgres_dsn: str,
        model: str = "gpt-4-turbo",
    ):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0,
        )
        self.postgres_dsn = postgres_dsn
        self.pool: Optional[asyncpg.Pool] = None
        self.checkpointer: Optional[AsyncPostgresSaver] = None
        self.graph = self._build_graph()

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self.postgres_dsn, min_size=2, max_size=10)
        self.checkpointer = AsyncPostgresSaver(self.pool)
        await self.checkpointer.setup()
        logger.info("Orchestrator connected to PostgreSQL")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for sequential processing."""
        workflow = StateGraph(SpreadsheetState)

        # Add nodes for each processing phase
        workflow.add_node("discover", self._discover_node)
        workflow.add_node("infer_schemas", self._schema_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("migrate", self._migrate_node)
        workflow.add_node("checkpoint", self._checkpoint_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Set entry point
        workflow.set_entry_point("discover")

        # Define sequential flow with conditional routing
        workflow.add_conditional_edges(
            "discover",
            self._route_after_discover,
            {
                "schema": "infer_schemas",
                "error": "error_handler",
            },
        )

        workflow.add_conditional_edges(
            "infer_schemas",
            self._route_after_schema,
            {
                "validate": "validate",
                "checkpoint": "checkpoint",
                "error": "error_handler",
            },
        )

        workflow.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {
                "migrate": "migrate",
                "checkpoint": "checkpoint",
                "error": "error_handler",
            },
        )

        workflow.add_conditional_edges(
            "migrate",
            self._route_after_migrate,
            {
                "complete": END,
                "checkpoint": "checkpoint",
                "error": "error_handler",
            },
        )

        # Checkpoint returns to the appropriate next phase
        workflow.add_conditional_edges(
            "checkpoint",
            self._route_after_checkpoint,
            {
                "schema": "infer_schemas",
                "validate": "validate",
                "migrate": "migrate",
                "complete": END,
            },
        )

        workflow.add_edge("error_handler", "checkpoint")

        return workflow.compile(checkpointer=self.checkpointer)

    async def _discover_node(self, state: SpreadsheetState) -> Dict:
        """Execute discovery phase."""
        logger.info("Starting discovery phase")

        try:
            # This would instantiate and run DiscoveryAgent
            # Simulated result
            discovered = [
                {"asset_id": "asset_001", "file_name": "supplier_data.xlsx", "priority": 0.9},
                {"asset_id": "asset_002", "file_name": "inventory_report.xlsx", "priority": 0.8},
            ]

            return {
                "discovered_assets": discovered,
                "assets_pending": [a["asset_id"] for a in discovered],
                "current_phase": "discovery_complete",
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Discovered {len(discovered)} spreadsheet assets")
                ],
            }
        except Exception as e:
            logger.error(f"Discovery error: {e}")
            return {
                "errors": state.get("errors", []) + [{
                    "phase": "discovery",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            }

    async def _schema_node(self, state: SpreadsheetState) -> Dict:
        """Execute schema inference phase."""
        logger.info("Starting schema inference phase")

        try:
            pending = state.get("assets_pending", [])
            schemas = state.get("schemas_inferred", {})

            # Process up to 10 assets per iteration
            batch = pending[:10]

            for asset_id in batch:
                # This would instantiate SchemaInferenceAgent
                schemas[asset_id] = {
                    "entity_type": "supplier",
                    "columns": ["supplier_id", "name", "contact"],
                    "confidence": 0.85,
                }

            remaining = pending[10:]

            return {
                "schemas_inferred": schemas,
                "assets_pending": remaining,
                "current_phase": "schema_complete" if not remaining else "schema_in_progress",
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Inferred schemas for {len(batch)} assets, {len(remaining)} remaining")
                ],
            }
        except Exception as e:
            logger.error(f"Schema inference error: {e}")
            return {
                "errors": state.get("errors", []) + [{
                    "phase": "schema",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            }

    async def _validate_node(self, state: SpreadsheetState) -> Dict:
        """Execute validation phase."""
        logger.info("Starting validation phase")

        try:
            schemas = state.get("schemas_inferred", {})
            validations = state.get("validation_results", {})
            approved = state.get("assets_approved", [])
            rejected = state.get("assets_rejected", [])

            for asset_id, schema in schemas.items():
                if asset_id in validations:
                    continue

                # This would instantiate ValidationAgent
                result = {
                    "quality_score": 92.5,
                    "auto_approve": True,
                    "issues": [],
                }
                validations[asset_id] = result

                if result["auto_approve"]:
                    approved.append(asset_id)
                else:
                    rejected.append(asset_id)

            return {
                "validation_results": validations,
                "assets_approved": approved,
                "assets_rejected": rejected,
                "current_phase": "validation_complete",
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Validated {len(validations)} assets: {len(approved)} approved, {len(rejected)} need review")
                ],
            }
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "errors": state.get("errors", []) + [{
                    "phase": "validation",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            }

    async def _migrate_node(self, state: SpreadsheetState) -> Dict:
        """Execute migration phase."""
        logger.info("Starting migration phase")

        try:
            approved = state.get("assets_approved", [])
            migrations = state.get("migration_results", {})

            for asset_id in approved:
                if asset_id in migrations:
                    continue

                # This would execute actual migration
                migrations[asset_id] = {
                    "status": "success",
                    "rows_migrated": 150,
                    "lineage_id": f"lineage_{asset_id}",
                }

            return {
                "migration_results": migrations,
                "current_phase": "migration_complete",
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Migrated {len(migrations)} assets to target platform")
                ],
            }
        except Exception as e:
            logger.error(f"Migration error: {e}")
            return {
                "errors": state.get("errors", []) + [{
                    "phase": "migration",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            }

    async def _checkpoint_node(self, state: SpreadsheetState) -> Dict:
        """Save checkpoint to PostgreSQL."""
        logger.info("Saving checkpoint")

        checkpoint_id = f"checkpoint_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # State is automatically persisted by LangGraph checkpointer
        # This node just updates the checkpoint timestamp

        return {
            "last_checkpoint": checkpoint_id,
        }

    async def _error_handler_node(self, state: SpreadsheetState) -> Dict:
        """Handle errors with retry logic."""
        errors = state.get("errors", [])
        logger.error(f"Error handler invoked with {len(errors)} errors")

        # Log to PostgreSQL for alerting
        if self.pool:
            async with self.pool.acquire() as conn:
                for error in errors[-5:]:  # Last 5 errors
                    await conn.execute(
                        """
                        INSERT INTO spreadsheet_errors (run_id, phase, error, timestamp)
                        VALUES ($1, $2, $3, $4)
                        """,
                        state.get("run_id"),
                        error.get("phase"),
                        error.get("error"),
                        error.get("timestamp"),
                    )

        return {
            "errors": [],  # Clear after logging
        }

    def _route_after_discover(self, state: SpreadsheetState) -> str:
        if state.get("errors"):
            return "error"
        return "schema"

    def _route_after_schema(self, state: SpreadsheetState) -> str:
        if state.get("errors"):
            return "error"
        if state.get("assets_pending"):
            return "checkpoint"  # More to process
        return "validate"

    def _route_after_validate(self, state: SpreadsheetState) -> str:
        if state.get("errors"):
            return "error"
        return "migrate"

    def _route_after_migrate(self, state: SpreadsheetState) -> str:
        if state.get("errors"):
            return "error"
        return "complete"

    def _route_after_checkpoint(self, state: SpreadsheetState) -> str:
        phase = state.get("current_phase", "")
        if "schema" in phase and state.get("assets_pending"):
            return "schema"
        if "schema" in phase:
            return "validate"
        if "validation" in phase:
            return "migrate"
        return "complete"

    async def run(
        self,
        drive_folders: List[str],
        local_paths: List[str],
        run_id: Optional[str] = None,
    ) -> Dict:
        """Execute the complete spreadsheet processing pipeline."""
        await self.connect()

        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        initial_state = {
            "messages": [HumanMessage(content="Starting spreadsheet processing pipeline")],
            "current_phase": "init",
            "discovered_assets": [],
            "schemas_inferred": {},
            "validation_results": {},
            "migration_results": {},
            "assets_pending": [],
            "assets_approved": [],
            "assets_rejected": [],
            "errors": [],
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_checkpoint": "",
        }

        config = {"configurable": {"thread_id": run_id}}

        try:
            final_state = await self.graph.ainvoke(initial_state, config)
            return {
                "status": "complete",
                "run_id": run_id,
                "discovered": len(final_state.get("discovered_assets", [])),
                "approved": len(final_state.get("assets_approved", [])),
                "rejected": len(final_state.get("assets_rejected", [])),
                "migrated": len(final_state.get("migration_results", {})),
            }
        finally:
            await self.close()`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the spreadsheet processing pipeline with Docker, implement lineage tracking, and set up comprehensive monitoring for data governance compliance.',
            toolsUsed: ['Docker', 'dbt', 'LangSmith', 'Prometheus'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose for Spreadsheet Pipeline',
                description:
                  'Complete Docker Compose configuration for deploying the spreadsheet processing agents with PostgreSQL and observability.',
                code: `version: '3.8'

services:
  # Database for state and lineage
  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: spreadsheet_pipeline
      POSTGRES_USER: pipeline_svc
      POSTGRES_PASSWORD: \${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U pipeline_svc -d spreadsheet_pipeline"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Agent Services
  discovery-agent:
    build:
      context: .
      dockerfile: Dockerfile.agents
    command: python -m agents.discovery
    environment:
      - REDIS_URL=redis://redis:6379
      - POSTGRES_DSN=postgresql://pipeline_svc:\${POSTGRES_PASSWORD}@postgres:5432/spreadsheet_pipeline
      - GOOGLE_APPLICATION_CREDENTIALS=/secrets/google-creds.json
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=spreadsheet-pipeline
      - SCAN_INTERVAL_MINUTES=60
    volumes:
      - ./secrets:/secrets:ro
      - /mnt/shared:/mnt/shared:ro
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  orchestrator:
    build:
      context: .
      dockerfile: Dockerfile.agents
    command: python -m agents.orchestrator
    ports:
      - "8080:8080"
    environment:
      - REDIS_URL=redis://redis:6379
      - POSTGRES_DSN=postgresql://pipeline_svc:\${POSTGRES_PASSWORD}@postgres:5432/spreadsheet_pipeline
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=spreadsheet-pipeline
      - AUTO_APPROVE_THRESHOLD=90
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      discovery-agent:
        condition: service_started
    restart: unless-stopped

  # dbt for data transformations
  dbt:
    build:
      context: ./dbt
      dockerfile: Dockerfile.dbt
    environment:
      - DBT_PROFILES_DIR=/dbt
      - POSTGRES_HOST=postgres
      - POSTGRES_PASSWORD=\${POSTGRES_PASSWORD}
    volumes:
      - ./dbt:/dbt
    depends_on:
      postgres:
        condition: service_healthy
    profiles:
      - tools

  # Observability
  prometheus:
    image: prom/prometheus:v2.47.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.1.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:`,
              },
              {
                language: 'python',
                title: 'Data Lineage Tracking & Metrics',
                description:
                  'Lineage tracking implementation for SOX compliance and Prometheus metrics for pipeline monitoring.',
                code: `import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import asyncpg
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

# Prometheus Metrics
ASSETS_DISCOVERED = Counter(
    "spreadsheet_assets_discovered_total",
    "Total spreadsheet assets discovered",
    ["source_type"],
)

ASSETS_PROCESSED = Counter(
    "spreadsheet_assets_processed_total",
    "Total spreadsheet assets processed",
    ["status"],  # approved, rejected, error
)

QUALITY_SCORE = Histogram(
    "spreadsheet_quality_score",
    "Distribution of quality scores",
    buckets=[50, 60, 70, 80, 85, 90, 95, 100],
)

ROWS_MIGRATED = Counter(
    "spreadsheet_rows_migrated_total",
    "Total rows migrated to target platform",
    ["entity_type"],
)

PIPELINE_DURATION = Histogram(
    "spreadsheet_pipeline_duration_seconds",
    "Pipeline execution duration",
    ["phase"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

ACTIVE_RUNS = Gauge(
    "spreadsheet_pipeline_active_runs",
    "Number of active pipeline runs",
)


@dataclass
class LineageRecord:
    """Data lineage record for audit trail."""
    lineage_id: str
    asset_id: str
    source_type: str
    source_path: str
    original_filename: str
    schema_version: str
    entity_type: str
    row_count: int
    target_table: str
    transformation_applied: List[str]
    quality_score: float
    approved_by: str  # 'auto' or user email
    approved_at: str
    migrated_at: str
    run_id: str


class LineageTracker:
    """Tracks data lineage for compliance and auditing."""

    def __init__(self, postgres_dsn: str):
        self.postgres_dsn = postgres_dsn
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self.postgres_dsn, min_size=2, max_size=10)

        # Ensure lineage table exists
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS data_lineage (
                    lineage_id VARCHAR(100) PRIMARY KEY,
                    asset_id VARCHAR(100) NOT NULL,
                    source_type VARCHAR(50) NOT NULL,
                    source_path TEXT NOT NULL,
                    original_filename VARCHAR(255) NOT NULL,
                    schema_version VARCHAR(50),
                    entity_type VARCHAR(50),
                    row_count INT,
                    target_table VARCHAR(100),
                    transformations JSONB,
                    quality_score NUMERIC(5,2),
                    approved_by VARCHAR(100),
                    approved_at TIMESTAMPTZ,
                    migrated_at TIMESTAMPTZ,
                    run_id VARCHAR(100),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(asset_id, run_id)
                );

                CREATE INDEX IF NOT EXISTS idx_lineage_asset ON data_lineage(asset_id);
                CREATE INDEX IF NOT EXISTS idx_lineage_run ON data_lineage(run_id);
                CREATE INDEX IF NOT EXISTS idx_lineage_target ON data_lineage(target_table);
            """)

        logger.info("Lineage tracker connected to PostgreSQL")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def record_lineage(self, record: LineageRecord) -> None:
        """Record a lineage entry for a migrated asset."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO data_lineage (
                    lineage_id, asset_id, source_type, source_path,
                    original_filename, schema_version, entity_type, row_count,
                    target_table, transformations, quality_score,
                    approved_by, approved_at, migrated_at, run_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (asset_id, run_id) DO UPDATE SET
                    migrated_at = EXCLUDED.migrated_at,
                    row_count = EXCLUDED.row_count
                """,
                record.lineage_id,
                record.asset_id,
                record.source_type,
                record.source_path,
                record.original_filename,
                record.schema_version,
                record.entity_type,
                record.row_count,
                record.target_table,
                json.dumps(record.transformation_applied),
                record.quality_score,
                record.approved_by,
                record.approved_at,
                record.migrated_at,
                record.run_id,
            )

        # Update metrics
        ROWS_MIGRATED.labels(entity_type=record.entity_type).inc(record.row_count)
        QUALITY_SCORE.observe(record.quality_score)

        logger.info(f"Recorded lineage: {record.lineage_id} -> {record.target_table}")

    async def get_lineage_for_table(self, target_table: str) -> List[Dict]:
        """Get all source lineage for a target table."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM data_lineage
                WHERE target_table = $1
                ORDER BY migrated_at DESC
                """,
                target_table,
            )
            return [dict(row) for row in rows]

    async def get_lineage_for_asset(self, asset_id: str) -> List[Dict]:
        """Get lineage history for a specific source asset."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM data_lineage
                WHERE asset_id = $1
                ORDER BY migrated_at DESC
                """,
                asset_id,
            )
            return [dict(row) for row in rows]

    async def generate_audit_report(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Generate audit report for compliance review."""
        async with self.pool.acquire() as conn:
            # Summary stats
            summary = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_migrations,
                    COUNT(DISTINCT asset_id) as unique_assets,
                    SUM(row_count) as total_rows,
                    AVG(quality_score) as avg_quality_score,
                    COUNT(*) FILTER (WHERE approved_by = 'auto') as auto_approved,
                    COUNT(*) FILTER (WHERE approved_by != 'auto') as manual_approved
                FROM data_lineage
                WHERE migrated_at BETWEEN $1 AND $2
                """,
                start_date,
                end_date,
            )

            # By entity type
            by_entity = await conn.fetch(
                """
                SELECT
                    entity_type,
                    COUNT(*) as count,
                    SUM(row_count) as rows,
                    AVG(quality_score) as avg_quality
                FROM data_lineage
                WHERE migrated_at BETWEEN $1 AND $2
                GROUP BY entity_type
                ORDER BY rows DESC
                """,
                start_date,
                end_date,
            )

            # Quality issues (scores below 90)
            quality_issues = await conn.fetch(
                """
                SELECT
                    lineage_id, asset_id, original_filename,
                    quality_score, approved_by, migrated_at
                FROM data_lineage
                WHERE migrated_at BETWEEN $1 AND $2
                  AND quality_score < 90
                ORDER BY quality_score ASC
                LIMIT 20
                """,
                start_date,
                end_date,
            )

            return {
                "report_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "summary": dict(summary) if summary else {},
                "by_entity_type": [dict(row) for row in by_entity],
                "quality_concerns": [dict(row) for row in quality_issues],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }


class PipelineMetrics:
    """Manages Prometheus metrics for the pipeline."""

    def __init__(self, port: int = 8000):
        self.port = port
        self._started = False

    def start_server(self) -> None:
        if not self._started:
            start_http_server(self.port)
            self._started = True
            logger.info(f"Metrics server started on port {self.port}")

    def record_discovery(self, source_type: str, count: int = 1) -> None:
        ASSETS_DISCOVERED.labels(source_type=source_type).inc(count)

    def record_processed(self, status: str) -> None:
        ASSETS_PROCESSED.labels(status=status).inc()

    def record_quality_score(self, score: float) -> None:
        QUALITY_SCORE.observe(score)

    def record_rows_migrated(self, entity_type: str, count: int) -> None:
        ROWS_MIGRATED.labels(entity_type=entity_type).inc(count)

    def record_phase_duration(self, phase: str, duration_seconds: float) -> None:
        PIPELINE_DURATION.labels(phase=phase).observe(duration_seconds)

    def set_active_runs(self, count: int) -> None:
        ACTIVE_RUNS.set(count)`,
              },
            ],
          },
        ],
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
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude with Zapier to automate HS code lookup, validate customs documentation, and generate pre-filled entry forms by analyzing product descriptions against tariff schedules without building a custom ML classifier.',
        estimatedMonthlyCost: '$150 - $200/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Google Sheets (Free)', 'Flexport API (Usage-based)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'FourKites AI (Usage-based)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Extract product master data and shipment details into a structured format suitable for AI-powered HS code classification and customs document generation.',
            toolsUsed: ['Zapier', 'Google Sheets', 'ERP API'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Shipment Data Extraction Workflow',
                description:
                  'Zapier configuration to extract shipment and product data from ERP and prepare for customs classification.',
                code: `{
  "workflow_name": "Customs Data Extraction",
  "trigger": {
    "app": "Webhooks by Zapier",
    "event": "Catch Hook",
    "webhook_url": "https://hooks.zapier.com/hooks/catch/123/customs/",
    "description": "Triggered when a new import shipment is created in ERP"
  },
  "action_1": {
    "app": "Webhooks by Zapier",
    "event": "GET",
    "url": "https://erp.company.com/api/v1/shipments/{{trigger.shipment_id}}",
    "headers": {
      "Authorization": "Bearer {{erp_api_token}}",
      "Content-Type": "application/json"
    }
  },
  "action_2": {
    "app": "Webhooks by Zapier",
    "event": "GET",
    "url": "https://erp.company.com/api/v1/shipments/{{trigger.shipment_id}}/line-items",
    "headers": {
      "Authorization": "Bearer {{erp_api_token}}"
    }
  },
  "action_3": {
    "app": "Google Sheets",
    "event": "Create Spreadsheet Row",
    "spreadsheet_id": "{{customs_processing_sheet}}",
    "worksheet": "Pending_Classification",
    "row_data": {
      "shipment_id": "{{trigger.shipment_id}}",
      "bill_of_lading": "{{action_1.bill_of_lading}}",
      "port_of_entry": "{{action_1.port_of_entry}}",
      "estimated_arrival": "{{action_1.eta}}",
      "supplier_name": "{{action_1.supplier.name}}",
      "country_of_origin": "{{action_1.supplier.country}}",
      "total_declared_value": "{{action_1.total_value_usd}}",
      "line_items_json": "{{action_2.line_items}}",
      "status": "pending_classification",
      "created_at": "{{current_timestamp}}"
    }
  },
  "action_4": {
    "app": "Looping by Zapier",
    "event": "Create Loop from Line Items",
    "items": "{{action_2.line_items}}",
    "actions": [
      {
        "app": "Google Sheets",
        "event": "Create Spreadsheet Row",
        "spreadsheet_id": "{{customs_processing_sheet}}",
        "worksheet": "Line_Items",
        "row_data": {
          "shipment_id": "{{trigger.shipment_id}}",
          "line_number": "{{loop.index}}",
          "product_id": "{{loop.item.product_id}}",
          "product_description": "{{loop.item.description}}",
          "material_composition": "{{loop.item.material}}",
          "quantity": "{{loop.item.quantity}}",
          "unit_of_measure": "{{loop.item.uom}}",
          "declared_value_usd": "{{loop.item.value_usd}}",
          "country_of_origin": "{{loop.item.origin_country}}",
          "hs_code_suggested": "",
          "classification_confidence": "",
          "status": "pending"
        }
      }
    ]
  }
}`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description:
              'Use ChatGPT to classify products into HS codes by analyzing descriptions against tariff schedule rules, identify potential compliance issues, and suggest duty optimization strategies.',
            toolsUsed: ['ChatGPT Plus', 'US Harmonized Tariff Schedule'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'HS Code Classification Prompt Template',
                description:
                  'Structured prompt for ChatGPT to classify products into correct HS codes with confidence scores and compliance flags.',
                code: `system_prompt: |
  You are an expert customs broker and tariff classification specialist with
  20 years of experience classifying goods under the Harmonized System (HS).
  You have memorized the US Harmonized Tariff Schedule and understand the
  General Rules of Interpretation (GRI) that govern classification.

  Key classification principles you apply:
  1. GRI 1: Classification is determined by the terms of the headings
  2. GRI 2(a): Incomplete or unfinished articles are classified as complete
  3. GRI 3: When goods are classifiable under multiple headings, use the
     most specific description
  4. GRI 6: Classification within a heading follows the same principles

user_prompt_template: |
  ## HS Code Classification Request

  **Shipment ID:** {{shipment_id}}
  **Port of Entry:** {{port_of_entry}}
  **Country of Origin:** {{country_of_origin}}

  ### Products to Classify

  {{#each line_items}}
  #### Line {{line_number}}
  - **Description:** {{product_description}}
  - **Material Composition:** {{material_composition}}
  - **Country of Origin:** {{country_of_origin}}
  - **Declared Value:** \${{declared_value_usd}}
  - **Quantity:** {{quantity}} {{unit_of_measure}}
  {{/each}}

  ### Classification Tasks

  For each product, provide:

  1. **HS Code (6-digit):** The international harmonized code
  2. **HS Code (10-digit):** The US-specific tariff code (HTS)
  3. **Confidence Score (0-100):** How confident you are in this classification
  4. **Classification Rationale:** Brief explanation citing relevant GRI rules
  5. **Duty Rate:** The applicable duty rate from the tariff schedule
  6. **Compliance Flags:**
     - Anti-dumping/CVD orders that may apply
     - Licenses or permits required
     - Restricted/prohibited items
     - FTA eligibility (USMCA, etc.)

  ### Special Considerations

  - If the description is ambiguous, provide top 2-3 possible classifications
  - Flag any items that likely require binding ruling requests
  - Identify items where material composition significantly impacts classification
  - Note any recent tariff changes or Section 301 tariffs that apply

  ### Output Format
  \`\`\`json
  {
    "shipment_id": "string",
    "classifications": [
      {
        "line_number": number,
        "product_description": "string",
        "hs_code_6": "string",
        "hs_code_10": "string",
        "confidence_score": number,
        "rationale": "string",
        "gri_rules_applied": ["string"],
        "duty_rate_pct": number,
        "estimated_duty_usd": number,
        "compliance_flags": {
          "anti_dumping": boolean,
          "cvd_order": boolean,
          "license_required": boolean,
          "section_301": boolean,
          "fta_eligible": "USMCA|none|unknown"
        },
        "alternative_classifications": [
          {
            "hs_code_10": "string",
            "confidence": number,
            "rationale": "string"
          }
        ],
        "requires_binding_ruling": boolean,
        "notes": "string"
      }
    ],
    "shipment_summary": {
      "total_estimated_duty": number,
      "high_risk_items": number,
      "items_needing_review": number,
      "fta_savings_potential": number
    }
  }
  \`\`\`

expected_output_format: json
temperature: 0.2
max_tokens: 4000`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Automate the complete customs workflow with Zapier to classify products, generate pre-filled CBP forms, alert on compliance issues, and deliver documentation to customs brokers.',
            toolsUsed: ['Zapier', 'Google Docs', 'Slack', 'Gmail'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Customs Documentation Automation',
                description:
                  'Complete Zapier workflow that classifies products, generates customs entry documents, and routes for approval.',
                code: `{
  "workflow_name": "Customs Classification & Documentation",
  "trigger": {
    "app": "Google Sheets",
    "event": "New Row in Spreadsheet",
    "spreadsheet_id": "{{customs_processing_sheet}}",
    "worksheet": "Pending_Classification",
    "filter": {
      "status": "pending_classification"
    }
  },
  "action_1": {
    "app": "Google Sheets",
    "event": "Get Many Spreadsheet Rows",
    "spreadsheet_id": "{{customs_processing_sheet}}",
    "worksheet": "Line_Items",
    "filter": {
      "shipment_id": "{{trigger.shipment_id}}",
      "status": "pending"
    }
  },
  "action_2": {
    "app": "ChatGPT",
    "event": "Conversation",
    "model": "gpt-4",
    "system_message": "You are an expert customs broker and tariff classification specialist...",
    "user_message": "Classify these products for import:\\n{{action_1.rows_formatted}}\\nCountry of Origin: {{trigger.country_of_origin}}",
    "temperature": 0.2
  },
  "action_3": {
    "app": "Formatter by Zapier",
    "event": "Text - Extract JSON",
    "input": "{{action_2.response}}",
    "extract_all": true
  },
  "action_4": {
    "app": "Looping by Zapier",
    "event": "Loop Through Classifications",
    "items": "{{action_3.classifications}}",
    "actions": [
      {
        "app": "Google Sheets",
        "event": "Update Spreadsheet Row",
        "spreadsheet_id": "{{customs_processing_sheet}}",
        "worksheet": "Line_Items",
        "lookup_column": "line_number",
        "lookup_value": "{{loop.item.line_number}}",
        "updates": {
          "hs_code_suggested": "{{loop.item.hs_code_10}}",
          "classification_confidence": "{{loop.item.confidence_score}}",
          "estimated_duty": "{{loop.item.estimated_duty_usd}}",
          "compliance_flags": "{{loop.item.compliance_flags}}",
          "classification_rationale": "{{loop.item.rationale}}",
          "status": "classified"
        }
      }
    ]
  },
  "action_5": {
    "app": "Filter by Zapier",
    "condition": {
      "field": "{{action_3.shipment_summary.high_risk_items}}",
      "operator": "greater_than",
      "value": 0
    }
  },
  "action_5a_high_risk": {
    "app": "Slack",
    "event": "Send Channel Message",
    "channel": "#customs-compliance",
    "message_blocks": [
      {
        "type": "header",
        "text": ":warning: High-Risk Customs Classification Alert"
      },
      {
        "type": "section",
        "fields": [
          {"type": "mrkdwn", "text": "*Shipment:* {{trigger.shipment_id}}"},
          {"type": "mrkdwn", "text": "*BOL:* {{trigger.bill_of_lading}}"},
          {"type": "mrkdwn", "text": "*High-Risk Items:* {{action_3.shipment_summary.high_risk_items}}"},
          {"type": "mrkdwn", "text": "*Est. Duty:* \${{action_3.shipment_summary.total_estimated_duty}}"}
        ]
      },
      {
        "type": "section",
        "text": "*Compliance Concerns:*\\n{{action_3.compliance_summary}}"
      },
      {
        "type": "actions",
        "elements": [
          {
            "type": "button",
            "text": "Review Classifications",
            "url": "{{google_sheets_url}}"
          },
          {
            "type": "button",
            "text": "Approve & Generate Docs",
            "style": "primary",
            "action_id": "approve_customs_{{trigger.shipment_id}}"
          }
        ]
      }
    ]
  },
  "action_6": {
    "app": "Google Docs",
    "event": "Create Document from Template",
    "template_id": "{{cbp_entry_template_id}}",
    "name": "CBP_Entry_{{trigger.shipment_id}}_{{current_date}}",
    "folder_id": "{{customs_docs_folder}}",
    "merge_fields": {
      "shipment_id": "{{trigger.shipment_id}}",
      "bill_of_lading": "{{trigger.bill_of_lading}}",
      "port_of_entry": "{{trigger.port_of_entry}}",
      "arrival_date": "{{trigger.estimated_arrival}}",
      "supplier_name": "{{trigger.supplier_name}}",
      "country_of_origin": "{{trigger.country_of_origin}}",
      "total_value": "{{trigger.total_declared_value}}",
      "total_duty": "{{action_3.shipment_summary.total_estimated_duty}}",
      "line_items_table": "{{action_3.formatted_line_items_table}}",
      "generated_date": "{{current_date}}"
    }
  },
  "action_7": {
    "app": "Gmail",
    "event": "Send Email",
    "to": "customs-broker@company.com",
    "cc": "logistics@company.com",
    "subject": "Customs Entry Ready: {{trigger.shipment_id}} - {{trigger.bill_of_lading}}",
    "body_html": "<h2>Customs Entry Documentation</h2><p>Classification complete for shipment {{trigger.shipment_id}}.</p><p><strong>Estimated Duty:</strong> \${{action_3.shipment_summary.total_estimated_duty}}</p><p><strong>High-Risk Items:</strong> {{action_3.shipment_summary.high_risk_items}}</p><p><a href='{{action_6.document_url}}'>View Entry Document</a></p>",
    "attachments": ["{{action_6.document_url}}"]
  },
  "action_8": {
    "app": "Google Sheets",
    "event": "Update Spreadsheet Row",
    "spreadsheet_id": "{{customs_processing_sheet}}",
    "worksheet": "Pending_Classification",
    "row": "{{trigger.row_number}}",
    "updates": {
      "status": "classified",
      "total_estimated_duty": "{{action_3.shipment_summary.total_estimated_duty}}",
      "entry_document_url": "{{action_6.document_url}}",
      "classified_at": "{{current_timestamp}}"
    }
  }
}`,
              },
            ],
          },
        ],
      },
      aiAdvanced: {
        overview:
          'Deploy a multi-agent system using CrewAI and LangGraph that continuously learns from customs outcomes, automatically classifies products using ML with tariff database integration, generates compliant documentation, and optimizes duty exposure through FTA analysis and binding ruling recommendations.',
        estimatedMonthlyCost: '$800 - $1,500/month',
        architecture:
          'A Supervisor agent coordinates four specialist agents: Classification Agent uses ML to classify products into HS codes with confidence scoring, Document Agent generates compliant customs entry forms and validates completeness, Compliance Agent monitors for anti-dumping orders and license requirements, and Optimization Agent identifies FTA opportunities and recommends duty savings strategies.',
        agents: [
          {
            name: 'Classification Agent',
            role: 'ML-Powered HS Code Classifier',
            goal: 'Classify products into accurate HS codes using trained ML models, provide confidence scores, identify ambiguous cases requiring human review, and continuously improve accuracy through feedback learning.',
            tools: ['TF-IDF Classifier', 'Tariff Schedule DB', 'Embedding Search', 'Feedback Loop'],
          },
          {
            name: 'Document Agent',
            role: 'Customs Documentation Generator',
            goal: 'Generate compliant CBP entry forms, commercial invoices, and supporting documentation by assembling shipment data, classifications, and valuations into the required formats.',
            tools: ['Template Engine', 'PDF Generator', 'Document Validator', 'CBP ACE Integration'],
          },
          {
            name: 'Compliance Agent',
            role: 'Trade Compliance Monitor',
            goal: 'Screen shipments against anti-dumping orders, export control lists, sanction databases, and license requirements to identify compliance risks before entry.',
            tools: ['AD/CVD Database', 'Denied Party Screener', 'License Database', 'OFAC API'],
          },
          {
            name: 'Optimization Agent',
            role: 'Duty Optimization Strategist',
            goal: 'Analyze shipments for FTA eligibility, identify tariff engineering opportunities, recommend binding rulings for recurring products, and calculate total landed cost scenarios.',
            tools: ['FTA Rules Engine', 'Tariff Calculator', 'Binding Ruling Database', 'Landed Cost Model'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Hierarchical',
          stateManagement: 'PostgreSQL-backed state with shipment-level checkpointing and full audit trail for customs compliance',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent customs automation system with CrewAI, establishing each agent\'s role, tools, and the hierarchical workflow for classification, compliance, and documentation.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'CrewAI Customs Agent Definitions',
                description:
                  'Complete CrewAI agent setup for the four-agent customs automation system with detailed role definitions.',
                code: `from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomsAgentConfig:
    """Configuration for customs automation agents."""

    def __init__(self, openai_api_key: str, model: str = "gpt-4-turbo"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0.1,
        )

    def create_classification_agent(self, tools: List[Any]) -> Agent:
        """Create the HS code classification agent."""
        return Agent(
            role="Customs Classification Specialist",
            goal="""Accurately classify products into HS codes using ML models
            trained on historical classifications and the Harmonized Tariff
            Schedule. Provide confidence scores, identify ambiguous cases that
            need human review, and learn from customs feedback to improve.""",
            backstory="""You are a licensed customs broker with 20 years of
            experience classifying goods for import into the United States.
            You've processed over 100,000 entries and have an encyclopedic
            knowledge of the Harmonized Tariff Schedule. You understand the
            General Rules of Interpretation and know exactly when to apply
            GRI 1 vs GRI 3 vs GRI 6. You've seen every possible product
            description variation and can identify the correct classification
            even from ambiguous descriptions.""",
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            max_iter=15,
        )

    def create_document_agent(self, tools: List[Any]) -> Agent:
        """Create the customs document generation agent."""
        return Agent(
            role="Customs Documentation Specialist",
            goal="""Generate complete, accurate, and compliant customs entry
            documentation including CBP Form 7501, commercial invoices, packing
            lists, and certificates of origin. Validate all required fields
            and ensure consistency across documents.""",
            backstory="""You are a documentation expert who has prepared
            customs entries for Fortune 500 importers. You know every field
            on CBP Form 7501 and understand exactly what information CBP
            requires. You've dealt with rejected entries and know the common
            mistakes that cause delays. You ensure every document is complete
            before submission because you know that a missing field can delay
            clearance by days.""",
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=10,
        )

    def create_compliance_agent(self, tools: List[Any]) -> Agent:
        """Create the trade compliance screening agent."""
        return Agent(
            role="Trade Compliance Officer",
            goal="""Screen all shipments and parties against denied party
            lists, anti-dumping/CVD orders, export control regulations, and
            sanctions databases. Identify compliance risks before entry and
            recommend mitigation strategies.""",
            backstory="""You are a trade compliance expert who has built
            compliance programs for global importers. You understand the severe
            consequences of compliance failures - from penalties to criminal
            prosecution. You screen every shipment against OFAC SDN, BIS Entity
            List, and CBP AD/CVD orders. You know that a single missed screening
            can result in millions in fines and reputational damage. You take
            a conservative approach: when in doubt, flag for review.""",
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            max_iter=12,
        )

    def create_optimization_agent(self, tools: List[Any]) -> Agent:
        """Create the duty optimization agent."""
        return Agent(
            role="Duty Optimization Strategist",
            goal="""Analyze shipments for duty savings opportunities including
            FTA eligibility, Foreign Trade Zone benefits, duty drawback, and
            tariff engineering. Calculate total landed cost scenarios and
            recommend binding rulings for high-volume products.""",
            backstory="""You are a trade strategist who has saved companies
            millions through intelligent duty planning. You know every FTA
            that the US has signed and understand their rules of origin. You
            can identify when a product qualifies for USMCA vs. when it needs
            a tariff shift or regional value content. You understand that duty
            optimization is legal and expected - the key is proper documentation
            and consistent application of the rules.""",
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            max_iter=15,
        )


def create_customs_crew(config: CustomsAgentConfig, tools: Dict[str, List[Any]]) -> Crew:
    """Create the complete customs automation crew."""

    classification_agent = config.create_classification_agent(tools["classification"])
    document_agent = config.create_document_agent(tools["document"])
    compliance_agent = config.create_compliance_agent(tools["compliance"])
    optimization_agent = config.create_optimization_agent(tools["optimization"])

    # Define hierarchical tasks
    classification_task = Task(
        description="""Classify all line items in the shipment:

        Shipment: {shipment_id}
        Supplier: {supplier_name}
        Country of Origin: {country_of_origin}

        Line Items:
        {line_items}

        For each item:
        1. Query the ML classifier for top-3 HS code predictions
        2. Validate against the tariff schedule database
        3. Apply GRI rules to select the correct classification
        4. Calculate duty rate and estimated duty
        5. Flag items with confidence < 85% for human review

        Output JSON with classifications, confidence scores, and duty estimates.""",
        expected_output="JSON with HS codes, confidence scores, duty rates, and review flags",
        agent=classification_agent,
    )

    compliance_task = Task(
        description="""Screen the shipment for compliance risks:

        Shipment: {shipment_id}
        Supplier: {supplier_name} ({supplier_country})
        Classifications: {classifications}

        Screening required:
        1. OFAC SDN List - screen supplier name and address
        2. BIS Entity List - check for export control concerns
        3. AD/CVD Orders - check HS codes against active orders
        4. CBP WITHRO - check for Uyghur Forced Labor Prevention Act
        5. Required licenses - check if any HS codes need permits

        Flag any hits with severity (block/review/info) and recommended action.""",
        expected_output="JSON with screening results, flags, and recommended actions",
        agent=compliance_agent,
        context=[classification_task],
    )

    optimization_task = Task(
        description="""Analyze duty optimization opportunities:

        Shipment: {shipment_id}
        Classifications: {classifications}
        Compliance Status: {compliance_status}

        Analysis required:
        1. FTA Eligibility - check USMCA, CAFTA-DR, etc. rules of origin
        2. Section 301 Exclusions - check if any exclusions apply
        3. Duty Drawback - check if any inputs are re-exported
        4. Binding Ruling Candidates - high-volume items without rulings
        5. First Sale Valuation - potential for middleman transactions

        Calculate savings potential for each opportunity.""",
        expected_output="JSON with optimization opportunities and savings estimates",
        agent=optimization_agent,
        context=[classification_task, compliance_task],
    )

    document_task = Task(
        description="""Generate customs entry documentation:

        Shipment: {shipment_id}
        Classifications: {classifications}
        Compliance: {compliance_status}
        Optimizations: {optimizations}

        Documents to generate:
        1. CBP Form 7501 - Entry Summary with all classifications
        2. Commercial Invoice - with declared values and INCOTERMS
        3. Packing List - quantities and weights
        4. Certificate of Origin - if claiming FTA benefits

        Validate all required fields are populated and consistent.""",
        expected_output="JSON with document URLs and validation status",
        agent=document_agent,
        context=[classification_task, compliance_task, optimization_task],
    )

    return Crew(
        agents=[classification_agent, compliance_agent, optimization_agent, document_agent],
        tasks=[classification_task, compliance_task, optimization_task, document_task],
        process=Process.sequential,  # Hierarchical with dependencies
        verbose=True,
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the Classification Agent with ML-based HS code prediction, tariff database integration, and confidence scoring for accurate customs classification.',
            toolsUsed: ['scikit-learn', 'sentence-transformers', 'PostgreSQL', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'ML Classification Agent Implementation',
                description:
                  'Production-ready implementation of the Classification Agent with TF-IDF and embedding-based HS code prediction.',
                code: `import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sentence_transformers import SentenceTransformer
import asyncpg
import redis.asyncio as redis
from langchain.tools import tool

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of HS code classification for a product."""
    line_number: int
    product_description: str
    hs_code_6: str
    hs_code_10: str
    confidence_score: float
    duty_rate_pct: float
    estimated_duty_usd: float
    rationale: str
    gri_rules_applied: List[str]
    alternative_codes: List[Dict[str, Any]]
    requires_review: bool
    classified_at: str


@dataclass
class TariffEntry:
    """Entry from the Harmonized Tariff Schedule."""
    hs_code_10: str
    hs_code_6: str
    description: str
    duty_rate_general: float
    duty_rate_special: Dict[str, float]  # FTA rates
    unit_of_quantity: str
    special_provisions: List[str]
    ad_cvd_flag: bool


class HSCodeClassifier:
    """ML-based HS code classifier with embedding search fallback."""

    def __init__(
        self,
        model_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        postgres_dsn: str = "",
        redis_url: str = "redis://localhost:6379",
    ):
        self.model_path = model_path
        self.postgres_dsn = postgres_dsn
        self.redis_url = redis_url

        # Load ML model
        self.classifier = None
        self.vectorizer = None
        self._load_model()

        # Load embedding model for semantic search
        self.embedding_model = SentenceTransformer(embedding_model)

        # Connections
        self.pool: Optional[asyncpg.Pool] = None
        self.redis: Optional[redis.Redis] = None

    def _load_model(self) -> None:
        """Load the trained classifier model."""
        try:
            model_data = joblib.load(self.model_path)
            self.classifier = model_data["classifier"]
            self.vectorizer = model_data["vectorizer"]
            logger.info(f"Loaded classifier from {self.model_path}")
        except FileNotFoundError:
            logger.warning(f"Model not found at {self.model_path}, will use embedding search only")

    async def connect(self) -> None:
        """Initialize database connections."""
        if self.postgres_dsn:
            self.pool = await asyncpg.create_pool(self.postgres_dsn, min_size=2, max_size=10)
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)
        logger.info("Classification agent connected to databases")

    async def close(self) -> None:
        """Close database connections."""
        if self.pool:
            await self.pool.close()
        if self.redis:
            await self.redis.close()

    def _preprocess_description(self, description: str, material: str = "", origin: str = "") -> str:
        """Preprocess product description for classification."""
        text = f"{description.lower()} {material.lower()} {origin.lower()}"
        # Remove special characters but keep spaces
        text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
        # Normalize whitespace
        text = " ".join(text.split())
        return text

    async def get_tariff_entry(self, hs_code_10: str) -> Optional[TariffEntry]:
        """Fetch tariff entry from database."""
        cache_key = f"tariff:{hs_code_10}"

        # Check cache first
        cached = await self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return TariffEntry(**data)

        # Query database
        if not self.pool:
            return None

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT hs_code_10, hs_code_6, description,
                       duty_rate_general, duty_rate_special,
                       unit_of_quantity, special_provisions, ad_cvd_flag
                FROM tariff_schedule
                WHERE hs_code_10 = $1
                """,
                hs_code_10,
            )

            if not row:
                return None

            entry = TariffEntry(
                hs_code_10=row["hs_code_10"],
                hs_code_6=row["hs_code_6"],
                description=row["description"],
                duty_rate_general=float(row["duty_rate_general"]),
                duty_rate_special=json.loads(row["duty_rate_special"]) if row["duty_rate_special"] else {},
                unit_of_quantity=row["unit_of_quantity"],
                special_provisions=json.loads(row["special_provisions"]) if row["special_provisions"] else [],
                ad_cvd_flag=row["ad_cvd_flag"],
            )

            # Cache for 24 hours
            await self.redis.setex(cache_key, 86400, json.dumps(entry.__dict__))

            return entry

    async def classify_with_ml(
        self, description: str, material: str, origin: str
    ) -> List[Tuple[str, float]]:
        """Classify using trained ML model."""
        if not self.classifier or not self.vectorizer:
            return []

        text = self._preprocess_description(description, material, origin)
        features = self.vectorizer.transform([text])
        probas = self.classifier.predict_proba(features)[0]
        classes = self.classifier.classes_

        # Get top 5 predictions
        top_indices = np.argsort(probas)[-5:][::-1]
        predictions = [(classes[i], float(probas[i])) for i in top_indices]

        return predictions

    async def classify_with_embeddings(
        self, description: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Classify using semantic embedding search."""
        if not self.pool:
            return []

        # Generate embedding for query
        query_embedding = self.embedding_model.encode(description)

        # Search tariff descriptions using pgvector
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT hs_code_10, description,
                       1 - (embedding <=> $1::vector) as similarity
                FROM tariff_schedule_embeddings
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                query_embedding.tolist(),
                top_k,
            )

            return [(row["hs_code_10"], float(row["similarity"])) for row in rows]

    def _determine_gri_rules(
        self, description: str, hs_code: str, alternatives: List[Tuple[str, float]]
    ) -> List[str]:
        """Determine which GRI rules were applied."""
        rules = ["GRI 1"]  # Always start with GRI 1

        # If multiple candidates are close, GRI 3 may apply
        if len(alternatives) >= 2:
            top_confidence = alternatives[0][1]
            second_confidence = alternatives[1][1]
            if second_confidence > top_confidence * 0.9:
                rules.append("GRI 3(a)")

        # Check if it's a subheading decision
        if len(hs_code) == 10:
            rules.append("GRI 6")

        return rules

    async def classify_product(
        self,
        line_number: int,
        description: str,
        material: str,
        origin: str,
        declared_value: float,
    ) -> ClassificationResult:
        """Classify a single product and return full result."""
        # Get ML predictions
        ml_predictions = await self.classify_with_ml(description, material, origin)

        # Get embedding predictions
        embedding_predictions = await self.classify_with_embeddings(description)

        # Combine predictions (weighted average)
        combined_scores: Dict[str, float] = {}
        for hs_code, score in ml_predictions:
            combined_scores[hs_code] = combined_scores.get(hs_code, 0) + score * 0.7
        for hs_code, score in embedding_predictions:
            combined_scores[hs_code] = combined_scores.get(hs_code, 0) + score * 0.3

        # Sort by combined score
        sorted_predictions = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_predictions:
            # Fallback for no predictions
            return ClassificationResult(
                line_number=line_number,
                product_description=description,
                hs_code_6="999999",
                hs_code_10="9999.99.9999",
                confidence_score=0.0,
                duty_rate_pct=0.0,
                estimated_duty_usd=0.0,
                rationale="Unable to classify - insufficient data",
                gri_rules_applied=["GRI 1"],
                alternative_codes=[],
                requires_review=True,
                classified_at=datetime.now(timezone.utc).isoformat(),
            )

        # Get top prediction
        top_hs_code, top_confidence = sorted_predictions[0]

        # Fetch tariff details
        tariff_entry = await self.get_tariff_entry(top_hs_code)
        duty_rate = tariff_entry.duty_rate_general if tariff_entry else 0.0

        # Calculate estimated duty
        estimated_duty = declared_value * duty_rate / 100

        # Build alternative codes
        alternatives = [
            {"hs_code_10": hs, "confidence": round(conf, 4), "rationale": "Alternative classification"}
            for hs, conf in sorted_predictions[1:4]
        ]

        # Determine GRI rules
        gri_rules = self._determine_gri_rules(description, top_hs_code, sorted_predictions)

        # Generate rationale
        rationale = f"Classified as {top_hs_code} based on product description matching tariff heading"
        if tariff_entry:
            rationale += f": '{tariff_entry.description[:100]}...'"

        return ClassificationResult(
            line_number=line_number,
            product_description=description,
            hs_code_6=top_hs_code[:6] if len(top_hs_code) >= 6 else top_hs_code,
            hs_code_10=top_hs_code,
            confidence_score=round(top_confidence, 4),
            duty_rate_pct=duty_rate,
            estimated_duty_usd=round(estimated_duty, 2),
            rationale=rationale,
            gri_rules_applied=gri_rules,
            alternative_codes=alternatives,
            requires_review=top_confidence < 0.85,
            classified_at=datetime.now(timezone.utc).isoformat(),
        )

    async def record_feedback(
        self,
        line_number: int,
        predicted_hs_code: str,
        actual_hs_code: str,
        was_accepted: bool,
    ) -> None:
        """Record classification feedback for model improvement."""
        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO classification_feedback
                    (predicted_hs_code, actual_hs_code, was_accepted, recorded_at)
                VALUES ($1, $2, $3, $4)
                """,
                predicted_hs_code,
                actual_hs_code,
                was_accepted,
                datetime.now(timezone.utc),
            )


# LangChain tool for CrewAI integration
@tool
def classify_products(shipment_data: str) -> str:
    """
    Classify products in a shipment into HS codes.

    Args:
        shipment_data: JSON string with shipment line items including
                      description, material, origin, and value

    Returns:
        JSON string with classification results for each line item
    """
    # Implementation would parse JSON and call HSCodeClassifier
    pass


@tool
def get_tariff_rate(hs_code: str, origin_country: str) -> str:
    """
    Get the tariff rate for an HS code considering country of origin.

    Args:
        hs_code: 10-digit HTS code
        origin_country: ISO country code for origin

    Returns:
        JSON with duty rates including general and FTA rates
    """
    pass`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Compliance and Optimization agents with denied party screening, AD/CVD checking, FTA eligibility analysis, and duty savings calculations.',
            toolsUsed: ['OFAC API', 'CBP API', 'PostgreSQL', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Compliance & Optimization Agents',
                description:
                  'Implementation of the Compliance Agent with denied party screening and the Optimization Agent with FTA analysis.',
                code: `import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import httpx
import asyncpg
from langchain.tools import tool

logger = logging.getLogger(__name__)


@dataclass
class ComplianceFlag:
    """Compliance screening result."""
    flag_type: str  # 'ofac_sdn', 'bis_entity', 'ad_cvd', 'wro', 'license'
    severity: str  # 'block', 'review', 'info'
    entity_matched: str
    match_score: float
    details: str
    recommended_action: str


@dataclass
class ComplianceResult:
    """Complete compliance screening result for a shipment."""
    shipment_id: str
    screened_at: str
    overall_status: str  # 'clear', 'review_required', 'blocked'
    flags: List[ComplianceFlag]
    screening_summary: Dict[str, int]


@dataclass
class OptimizationOpportunity:
    """Duty optimization opportunity."""
    opportunity_type: str  # 'fta', 'section_301_exclusion', 'ftz', 'drawback'
    hs_codes_affected: List[str]
    current_duty_total: float
    optimized_duty_total: float
    savings_potential: float
    requirements: List[str]
    complexity: str  # 'low', 'medium', 'high'
    recommendation: str


class ComplianceAgent:
    """Agent for trade compliance screening."""

    def __init__(
        self,
        postgres_dsn: str,
        ofac_api_key: Optional[str] = None,
    ):
        self.postgres_dsn = postgres_dsn
        self.ofac_api_key = ofac_api_key
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self.postgres_dsn, min_size=2, max_size=10)
        logger.info("Compliance agent connected to database")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def screen_ofac(self, entity_name: str, country: str) -> List[ComplianceFlag]:
        """Screen entity against OFAC SDN list."""
        flags = []

        async with self.pool.acquire() as conn:
            # Fuzzy match against local SDN copy
            matches = await conn.fetch(
                """
                SELECT name, program, score
                FROM ofac_sdn
                WHERE similarity(name, $1) > 0.7
                   OR name ILIKE '%' || $1 || '%'
                ORDER BY similarity(name, $1) DESC
                LIMIT 5
                """,
                entity_name,
            )

            for match in matches:
                severity = "block" if match["score"] > 0.9 else "review"
                flags.append(ComplianceFlag(
                    flag_type="ofac_sdn",
                    severity=severity,
                    entity_matched=match["name"],
                    match_score=float(match["score"]),
                    details=f"OFAC program: {match['program']}",
                    recommended_action="Do not proceed" if severity == "block" else "Manual review required",
                ))

        return flags

    async def check_ad_cvd(self, hs_codes: List[str], origin_country: str) -> List[ComplianceFlag]:
        """Check HS codes against anti-dumping/CVD orders."""
        flags = []

        async with self.pool.acquire() as conn:
            for hs_code in hs_codes:
                # Check first 6 digits against AD/CVD orders
                hs_6 = hs_code[:6]
                orders = await conn.fetch(
                    """
                    SELECT case_number, merchandise_description,
                           ad_rate, cvd_rate, country
                    FROM ad_cvd_orders
                    WHERE hs_code_prefix = $1
                      AND country = $2
                      AND status = 'active'
                    """,
                    hs_6,
                    origin_country,
                )

                for order in orders:
                    total_rate = (order["ad_rate"] or 0) + (order["cvd_rate"] or 0)
                    flags.append(ComplianceFlag(
                        flag_type="ad_cvd",
                        severity="review",
                        entity_matched=hs_code,
                        match_score=1.0,
                        details=f"AD/CVD Order {order['case_number']}: {order['merchandise_description'][:50]}",
                        recommended_action=f"Additional duty of {total_rate}% applies. Ensure proper bonding.",
                    ))

        return flags

    async def check_wro(self, supplier_name: str, origin_country: str) -> List[ComplianceFlag]:
        """Check against Withhold Release Orders (forced labor)."""
        flags = []

        async with self.pool.acquire() as conn:
            wros = await conn.fetch(
                """
                SELECT wro_id, entity_name, merchandise, country
                FROM withhold_release_orders
                WHERE (entity_name ILIKE '%' || $1 || '%'
                       OR similarity(entity_name, $1) > 0.6)
                  AND country = $2
                  AND status = 'active'
                """,
                supplier_name,
                origin_country,
            )

            for wro in wros:
                flags.append(ComplianceFlag(
                    flag_type="wro",
                    severity="block",
                    entity_matched=wro["entity_name"],
                    match_score=0.8,
                    details=f"WRO {wro['wro_id']}: {wro['merchandise']}",
                    recommended_action="Shipment cannot be imported. Forced labor concern.",
                ))

        return flags

    async def screen_shipment(
        self,
        shipment_id: str,
        supplier_name: str,
        supplier_country: str,
        hs_codes: List[str],
    ) -> ComplianceResult:
        """Perform complete compliance screening."""
        all_flags = []

        # OFAC screening
        ofac_flags = await self.screen_ofac(supplier_name, supplier_country)
        all_flags.extend(ofac_flags)

        # AD/CVD check
        ad_cvd_flags = await self.check_ad_cvd(hs_codes, supplier_country)
        all_flags.extend(ad_cvd_flags)

        # WRO check
        wro_flags = await self.check_wro(supplier_name, supplier_country)
        all_flags.extend(wro_flags)

        # Determine overall status
        if any(f.severity == "block" for f in all_flags):
            overall_status = "blocked"
        elif any(f.severity == "review" for f in all_flags):
            overall_status = "review_required"
        else:
            overall_status = "clear"

        # Summary counts
        summary = {
            "total_flags": len(all_flags),
            "block_flags": len([f for f in all_flags if f.severity == "block"]),
            "review_flags": len([f for f in all_flags if f.severity == "review"]),
            "info_flags": len([f for f in all_flags if f.severity == "info"]),
        }

        return ComplianceResult(
            shipment_id=shipment_id,
            screened_at=datetime.now(timezone.utc).isoformat(),
            overall_status=overall_status,
            flags=all_flags,
            screening_summary=summary,
        )


class OptimizationAgent:
    """Agent for duty optimization analysis."""

    def __init__(self, postgres_dsn: str):
        self.postgres_dsn = postgres_dsn
        self.pool: Optional[asyncpg.Pool] = None

        # FTA country mappings
        self.fta_countries = {
            "USMCA": ["CA", "MX"],
            "CAFTA-DR": ["CR", "DO", "GT", "HN", "NI", "SV"],
            "Korea": ["KR"],
            "Australia": ["AU"],
            "Singapore": ["SG"],
        }

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self.postgres_dsn, min_size=2, max_size=10)
        logger.info("Optimization agent connected to database")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    def _get_applicable_ftas(self, origin_country: str) -> List[str]:
        """Get FTAs applicable to the origin country."""
        applicable = []
        for fta_name, countries in self.fta_countries.items():
            if origin_country in countries:
                applicable.append(fta_name)
        return applicable

    async def check_fta_eligibility(
        self,
        hs_codes: List[str],
        origin_country: str,
        declared_values: Dict[str, float],
    ) -> List[OptimizationOpportunity]:
        """Check FTA eligibility for duty savings."""
        opportunities = []
        applicable_ftas = self._get_applicable_ftas(origin_country)

        if not applicable_ftas:
            return opportunities

        async with self.pool.acquire() as conn:
            for fta in applicable_ftas:
                eligible_hs = []
                current_duty = 0.0
                fta_duty = 0.0

                for hs_code in hs_codes:
                    # Check FTA rate
                    row = await conn.fetchrow(
                        """
                        SELECT general_rate, fta_rates
                        FROM tariff_schedule
                        WHERE hs_code_10 = $1
                        """,
                        hs_code,
                    )

                    if not row:
                        continue

                    general_rate = float(row["general_rate"])
                    fta_rates = json.loads(row["fta_rates"]) if row["fta_rates"] else {}
                    fta_rate = fta_rates.get(fta, general_rate)

                    value = declared_values.get(hs_code, 0)
                    current_duty += value * general_rate / 100
                    fta_duty += value * fta_rate / 100

                    if fta_rate < general_rate:
                        eligible_hs.append(hs_code)

                if eligible_hs and current_duty > fta_duty:
                    savings = current_duty - fta_duty
                    opportunities.append(OptimizationOpportunity(
                        opportunity_type="fta",
                        hs_codes_affected=eligible_hs,
                        current_duty_total=round(current_duty, 2),
                        optimized_duty_total=round(fta_duty, 2),
                        savings_potential=round(savings, 2),
                        requirements=[
                            f"Certificate of Origin for {fta}",
                            "Rule of origin documentation",
                            f"Importer must claim {fta} preference on entry",
                        ],
                        complexity="low" if fta == "USMCA" else "medium",
                        recommendation=f"Claim {fta} preferential treatment for {len(eligible_hs)} items",
                    ))

        return opportunities

    async def check_section_301_exclusions(
        self,
        hs_codes: List[str],
        origin_country: str,
    ) -> List[OptimizationOpportunity]:
        """Check for Section 301 exclusions (China tariffs)."""
        opportunities = []

        if origin_country != "CN":
            return opportunities

        async with self.pool.acquire() as conn:
            excluded_hs = []

            for hs_code in hs_codes:
                exclusion = await conn.fetchrow(
                    """
                    SELECT exclusion_number, expiration_date, product_description
                    FROM section_301_exclusions
                    WHERE hs_code_10 = $1
                      AND status = 'active'
                      AND expiration_date > CURRENT_DATE
                    """,
                    hs_code,
                )

                if exclusion:
                    excluded_hs.append(hs_code)

            if excluded_hs:
                opportunities.append(OptimizationOpportunity(
                    opportunity_type="section_301_exclusion",
                    hs_codes_affected=excluded_hs,
                    current_duty_total=0,  # Would be calculated
                    optimized_duty_total=0,
                    savings_potential=0,  # Would be 7.5-25% of value
                    requirements=[
                        "Cite exclusion number on entry",
                        "Monitor expiration dates for renewals",
                    ],
                    complexity="low",
                    recommendation=f"Claim Section 301 exclusion for {len(excluded_hs)} items from China",
                ))

        return opportunities

    async def analyze_shipment(
        self,
        shipment_id: str,
        hs_codes: List[str],
        origin_country: str,
        declared_values: Dict[str, float],
    ) -> List[OptimizationOpportunity]:
        """Perform complete optimization analysis."""
        opportunities = []

        # Check FTA eligibility
        fta_opps = await self.check_fta_eligibility(hs_codes, origin_country, declared_values)
        opportunities.extend(fta_opps)

        # Check Section 301 exclusions
        s301_opps = await self.check_section_301_exclusions(hs_codes, origin_country)
        opportunities.extend(s301_opps)

        # Sort by savings potential
        opportunities.sort(key=lambda x: x.savings_potential, reverse=True)

        return opportunities


# LangChain tools for CrewAI
@tool
def screen_compliance(shipment_data: str) -> str:
    """
    Screen a shipment for trade compliance issues.

    Args:
        shipment_data: JSON with supplier_name, supplier_country, hs_codes

    Returns:
        JSON with compliance flags and overall status
    """
    pass


@tool
def analyze_duty_optimization(shipment_data: str) -> str:
    """
    Analyze a shipment for duty optimization opportunities.

    Args:
        shipment_data: JSON with hs_codes, origin_country, declared_values

    Returns:
        JSON with optimization opportunities and savings estimates
    """
    pass`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement the LangGraph state machine that orchestrates the hierarchical workflow from classification through documentation with compliance gates and approval routing.',
            toolsUsed: ['LangGraph', 'PostgreSQL', 'asyncio'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Hierarchical Customs Orchestration',
                description:
                  'Complete LangGraph implementation for orchestrating the customs automation pipeline with compliance gates.',
                code: `import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Dict, List, Literal, Optional, Sequence, TypedDict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import asyncpg

logger = logging.getLogger(__name__)


class CustomsState(TypedDict):
    """State schema for the customs processing workflow."""
    messages: Annotated[Sequence[BaseMessage], "Conversation messages"]
    current_phase: str
    shipment_id: str
    shipment_data: Dict[str, Any]
    classifications: List[Dict[str, Any]]
    compliance_result: Optional[Dict[str, Any]]
    optimization_result: Optional[Dict[str, Any]]
    documents_generated: List[Dict[str, str]]
    approval_status: str  # 'pending', 'approved', 'rejected', 'review_required'
    review_notes: List[str]
    errors: List[Dict[str, str]]
    started_at: str
    completed_at: Optional[str]


class CustomsOrchestrator:
    """LangGraph-based orchestrator for customs automation pipeline."""

    def __init__(
        self,
        openai_api_key: str,
        postgres_dsn: str,
        model: str = "gpt-4-turbo",
    ):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0,
        )
        self.postgres_dsn = postgres_dsn
        self.pool: Optional[asyncpg.Pool] = None
        self.checkpointer: Optional[AsyncPostgresSaver] = None
        self.graph = self._build_graph()

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self.postgres_dsn, min_size=2, max_size=10)
        self.checkpointer = AsyncPostgresSaver(self.pool)
        await self.checkpointer.setup()
        logger.info("Customs orchestrator connected to PostgreSQL")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for customs processing."""
        workflow = StateGraph(CustomsState)

        # Add nodes for each processing phase
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("compliance_check", self._compliance_node)
        workflow.add_node("optimize", self._optimize_node)
        workflow.add_node("generate_documents", self._document_node)
        workflow.add_node("human_review", self._human_review_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Set entry point
        workflow.set_entry_point("classify")

        # Define edges with compliance gates
        workflow.add_edge("classify", "compliance_check")

        workflow.add_conditional_edges(
            "compliance_check",
            self._route_after_compliance,
            {
                "optimize": "optimize",
                "review": "human_review",
                "blocked": "error_handler",
            },
        )

        workflow.add_edge("optimize", "generate_documents")

        workflow.add_conditional_edges(
            "generate_documents",
            self._route_after_documents,
            {
                "finalize": "finalize",
                "review": "human_review",
                "error": "error_handler",
            },
        )

        workflow.add_conditional_edges(
            "human_review",
            self._route_after_review,
            {
                "approved": "generate_documents",
                "rejected": "error_handler",
                "finalize": "finalize",
            },
        )

        workflow.add_edge("finalize", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def _classify_node(self, state: CustomsState) -> Dict:
        """Execute classification phase."""
        logger.info(f"Classifying shipment {state.get('shipment_id')}")

        try:
            shipment = state.get("shipment_data", {})
            line_items = shipment.get("line_items", [])

            # This would call the Classification Agent
            classifications = []
            for item in line_items:
                classifications.append({
                    "line_number": item.get("line_number"),
                    "hs_code_10": "8471.30.0100",  # Simulated
                    "confidence": 0.92,
                    "duty_rate": 0.0,
                })

            return {
                "classifications": classifications,
                "current_phase": "classified",
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Classified {len(classifications)} line items")
                ],
            }
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {
                "errors": state.get("errors", []) + [{
                    "phase": "classification",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            }

    async def _compliance_node(self, state: CustomsState) -> Dict:
        """Execute compliance screening phase."""
        logger.info(f"Screening compliance for shipment {state.get('shipment_id')}")

        try:
            shipment = state.get("shipment_data", {})
            classifications = state.get("classifications", [])
            hs_codes = [c["hs_code_10"] for c in classifications]

            # This would call the Compliance Agent
            compliance_result = {
                "overall_status": "clear",
                "flags": [],
                "screening_summary": {"total_flags": 0},
            }

            return {
                "compliance_result": compliance_result,
                "current_phase": "compliance_complete",
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Compliance screening complete: {compliance_result['overall_status']}")
                ],
            }
        except Exception as e:
            logger.error(f"Compliance error: {e}")
            return {
                "errors": state.get("errors", []) + [{
                    "phase": "compliance",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            }

    async def _optimize_node(self, state: CustomsState) -> Dict:
        """Execute optimization analysis phase."""
        logger.info(f"Analyzing optimization for shipment {state.get('shipment_id')}")

        try:
            # This would call the Optimization Agent
            optimization_result = {
                "opportunities": [],
                "total_savings_potential": 0,
            }

            return {
                "optimization_result": optimization_result,
                "current_phase": "optimization_complete",
            }
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {
                "errors": state.get("errors", []) + [{
                    "phase": "optimization",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            }

    async def _document_node(self, state: CustomsState) -> Dict:
        """Execute document generation phase."""
        logger.info(f"Generating documents for shipment {state.get('shipment_id')}")

        try:
            # This would call the Document Agent
            documents = [
                {"type": "cbp_7501", "url": "https://docs.example.com/7501_xxx.pdf"},
                {"type": "commercial_invoice", "url": "https://docs.example.com/ci_xxx.pdf"},
            ]

            return {
                "documents_generated": documents,
                "current_phase": "documents_complete",
            }
        except Exception as e:
            logger.error(f"Document generation error: {e}")
            return {
                "errors": state.get("errors", []) + [{
                    "phase": "documents",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            }

    async def _human_review_node(self, state: CustomsState) -> Dict:
        """Handle human review requirements."""
        logger.info(f"Routing shipment {state.get('shipment_id')} for human review")

        # In production, this would:
        # 1. Send notification to compliance team
        # 2. Create review task in workflow system
        # 3. Wait for approval (via webhook or polling)

        return {
            "approval_status": "review_required",
            "current_phase": "pending_review",
        }

    async def _finalize_node(self, state: CustomsState) -> Dict:
        """Finalize the customs processing."""
        logger.info(f"Finalizing shipment {state.get('shipment_id')}")

        # Record completion
        if self.pool:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO customs_processing_log
                        (shipment_id, status, classifications, compliance_result,
                         documents, completed_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    state.get("shipment_id"),
                    "complete",
                    json.dumps(state.get("classifications", [])),
                    json.dumps(state.get("compliance_result", {})),
                    json.dumps(state.get("documents_generated", [])),
                    datetime.now(timezone.utc),
                )

        return {
            "approval_status": "approved",
            "current_phase": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _error_handler_node(self, state: CustomsState) -> Dict:
        """Handle errors and blocked shipments."""
        errors = state.get("errors", [])
        compliance = state.get("compliance_result", {})

        logger.error(f"Error handler for shipment {state.get('shipment_id')}")

        # Record failure
        if self.pool:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO customs_processing_errors
                        (shipment_id, errors, compliance_flags, recorded_at)
                    VALUES ($1, $2, $3, $4)
                    """,
                    state.get("shipment_id"),
                    json.dumps(errors),
                    json.dumps(compliance.get("flags", [])),
                    datetime.now(timezone.utc),
                )

        return {
            "approval_status": "rejected",
            "current_phase": "error",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _route_after_compliance(self, state: CustomsState) -> str:
        """Route based on compliance screening results."""
        compliance = state.get("compliance_result", {})
        status = compliance.get("overall_status", "clear")

        if status == "blocked":
            return "blocked"
        if status == "review_required":
            return "review"
        return "optimize"

    def _route_after_documents(self, state: CustomsState) -> str:
        """Route after document generation."""
        errors = state.get("errors", [])
        if errors:
            return "error"

        # Check if any classification needs review
        classifications = state.get("classifications", [])
        low_confidence = [c for c in classifications if c.get("confidence", 1) < 0.85]
        if low_confidence:
            return "review"

        return "finalize"

    def _route_after_review(self, state: CustomsState) -> str:
        """Route based on human review decision."""
        approval = state.get("approval_status", "pending")

        if approval == "approved":
            # Check if documents already generated
            docs = state.get("documents_generated", [])
            if docs:
                return "finalize"
            return "approved"
        if approval == "rejected":
            return "rejected"
        return "finalize"

    async def process_shipment(
        self,
        shipment_id: str,
        shipment_data: Dict[str, Any],
    ) -> Dict:
        """Process a complete customs shipment."""
        await self.connect()

        initial_state = {
            "messages": [HumanMessage(content=f"Processing customs for shipment {shipment_id}")],
            "current_phase": "init",
            "shipment_id": shipment_id,
            "shipment_data": shipment_data,
            "classifications": [],
            "compliance_result": None,
            "optimization_result": None,
            "documents_generated": [],
            "approval_status": "pending",
            "review_notes": [],
            "errors": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        }

        config = {"configurable": {"thread_id": shipment_id}}

        try:
            final_state = await self.graph.ainvoke(initial_state, config)
            return {
                "status": final_state.get("approval_status"),
                "shipment_id": shipment_id,
                "classifications": final_state.get("classifications", []),
                "compliance": final_state.get("compliance_result"),
                "documents": final_state.get("documents_generated", []),
                "completed_at": final_state.get("completed_at"),
            }
        finally:
            await self.close()`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the customs automation system with Docker, implement CBP ACE integration, and set up monitoring for classification accuracy and compliance screening.',
            toolsUsed: ['Docker', 'CBP ACE API', 'LangSmith', 'Prometheus'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose for Customs Automation',
                description:
                  'Complete Docker Compose configuration for deploying the customs automation agents with compliance databases.',
                code: `version: '3.8'

services:
  # Core Database
  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: customs_automation
      POSTGRES_USER: customs_svc
      POSTGRES_PASSWORD: \${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-customs.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U customs_svc -d customs_automation"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Agent Services
  classification-agent:
    build:
      context: .
      dockerfile: Dockerfile.agents
    command: python -m agents.classification
    environment:
      - POSTGRES_DSN=postgresql://customs_svc:\${POSTGRES_PASSWORD}@postgres:5432/customs_automation
      - REDIS_URL=redis://redis:6379
      - MODEL_PATH=/models/hs_classifier_v2.joblib
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=customs-automation
    volumes:
      - ./models:/models:ro
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  compliance-agent:
    build:
      context: .
      dockerfile: Dockerfile.agents
    command: python -m agents.compliance
    environment:
      - POSTGRES_DSN=postgresql://customs_svc:\${POSTGRES_PASSWORD}@postgres:5432/customs_automation
      - OFAC_API_KEY=\${OFAC_API_KEY}
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

  orchestrator:
    build:
      context: .
      dockerfile: Dockerfile.agents
    command: python -m agents.orchestrator
    ports:
      - "8080:8080"
    environment:
      - POSTGRES_DSN=postgresql://customs_svc:\${POSTGRES_PASSWORD}@postgres:5432/customs_automation
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=customs-automation
      - CBP_ACE_ENDPOINT=\${CBP_ACE_ENDPOINT}
      - CBP_ACE_CLIENT_ID=\${CBP_ACE_CLIENT_ID}
    depends_on:
      postgres:
        condition: service_healthy
      classification-agent:
        condition: service_started
      compliance-agent:
        condition: service_started
    restart: unless-stopped

  # Compliance Data Sync (daily update of OFAC, AD/CVD, etc.)
  compliance-sync:
    build:
      context: .
      dockerfile: Dockerfile.sync
    command: python -m sync.compliance_data
    environment:
      - POSTGRES_DSN=postgresql://customs_svc:\${POSTGRES_PASSWORD}@postgres:5432/customs_automation
      - OFAC_FEED_URL=https://www.treasury.gov/ofac/downloads/sdn.csv
      - AD_CVD_FEED_URL=https://www.usitc.gov/tariff_affairs/documents/
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

  # Observability
  prometheus:
    image: prom/prometheus:v2.47.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.1.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/customs-dashboard.json:/etc/grafana/provisioning/dashboards/customs.json
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:`,
              },
              {
                language: 'python',
                title: 'Customs Metrics & Classification Accuracy Tracking',
                description:
                  'Prometheus metrics and accuracy tracking for the customs automation system.',
                code: `import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import asyncpg
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

# Prometheus Metrics
SHIPMENTS_PROCESSED = Counter(
    "customs_shipments_processed_total",
    "Total shipments processed",
    ["status"],  # approved, rejected, review_required
)

CLASSIFICATION_CONFIDENCE = Histogram(
    "customs_classification_confidence",
    "Distribution of classification confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
)

CLASSIFICATION_ACCURACY = Gauge(
    "customs_classification_accuracy",
    "Rolling 30-day classification accuracy based on feedback",
)

COMPLIANCE_FLAGS = Counter(
    "customs_compliance_flags_total",
    "Total compliance flags raised",
    ["flag_type", "severity"],
)

DUTY_SAVINGS = Counter(
    "customs_duty_savings_usd_total",
    "Total duty savings achieved through optimization",
    ["opportunity_type"],
)

PROCESSING_TIME = Histogram(
    "customs_processing_time_seconds",
    "End-to-end processing time per shipment",
    buckets=[30, 60, 120, 300, 600, 1200, 3600],
)

DOCUMENTS_GENERATED = Counter(
    "customs_documents_generated_total",
    "Total customs documents generated",
    ["document_type"],
)


class CustomsMetricsManager:
    """Manages metrics and accuracy tracking for customs automation."""

    def __init__(self, postgres_dsn: str, metrics_port: int = 8000):
        self.postgres_dsn = postgres_dsn
        self.metrics_port = metrics_port
        self.pool: Optional[asyncpg.Pool] = None
        self._started = False

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self.postgres_dsn, min_size=2, max_size=5)
        logger.info("Metrics manager connected to PostgreSQL")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    def start_server(self) -> None:
        if not self._started:
            start_http_server(self.metrics_port)
            self._started = True
            logger.info(f"Metrics server started on port {self.metrics_port}")

    def record_shipment(self, status: str, processing_time_seconds: float) -> None:
        SHIPMENTS_PROCESSED.labels(status=status).inc()
        PROCESSING_TIME.observe(processing_time_seconds)

    def record_classification(self, confidence: float) -> None:
        CLASSIFICATION_CONFIDENCE.observe(confidence)

    def record_compliance_flag(self, flag_type: str, severity: str) -> None:
        COMPLIANCE_FLAGS.labels(flag_type=flag_type, severity=severity).inc()

    def record_duty_savings(self, opportunity_type: str, amount: float) -> None:
        DUTY_SAVINGS.labels(opportunity_type=opportunity_type).inc(amount)

    def record_document(self, document_type: str) -> None:
        DOCUMENTS_GENERATED.labels(document_type=document_type).inc()

    async def calculate_classification_accuracy(self) -> float:
        """Calculate 30-day rolling classification accuracy from feedback."""
        if not self.pool:
            return 0.0

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) FILTER (WHERE was_accepted) AS correct,
                    COUNT(*) AS total
                FROM classification_feedback
                WHERE recorded_at >= NOW() - INTERVAL '30 days'
                """
            )

            if row["total"] == 0:
                return 1.0  # No feedback yet

            accuracy = row["correct"] / row["total"]
            CLASSIFICATION_ACCURACY.set(accuracy)
            return accuracy

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary metrics for dashboard."""
        if not self.pool:
            return {}

        async with self.pool.acquire() as conn:
            # Shipment summary (last 24 hours)
            shipment_stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE status = 'complete') AS approved,
                    COUNT(*) FILTER (WHERE status = 'error') AS rejected,
                    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) AS avg_processing_time
                FROM customs_processing_log
                WHERE started_at >= NOW() - INTERVAL '24 hours'
                """
            )

            # Classification accuracy
            accuracy = await self.calculate_classification_accuracy()

            # Compliance flags (last 24 hours)
            compliance_stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) AS total_flags,
                    COUNT(*) FILTER (WHERE severity = 'block') AS blocks,
                    COUNT(*) FILTER (WHERE severity = 'review') AS reviews
                FROM compliance_flags
                WHERE screened_at >= NOW() - INTERVAL '24 hours'
                """
            )

            # Duty savings (last 30 days)
            savings = await conn.fetchrow(
                """
                SELECT
                    SUM(savings_achieved) AS total_savings,
                    COUNT(DISTINCT shipment_id) AS shipments_optimized
                FROM optimization_results
                WHERE applied_at >= NOW() - INTERVAL '30 days'
                """
            )

            return {
                "period": "last_24_hours",
                "shipments": {
                    "total": shipment_stats["total"] or 0,
                    "approved": shipment_stats["approved"] or 0,
                    "rejected": shipment_stats["rejected"] or 0,
                    "avg_processing_time_seconds": round(
                        shipment_stats["avg_processing_time"] or 0, 1
                    ),
                },
                "classification": {
                    "accuracy_30d": round(accuracy, 4),
                },
                "compliance": {
                    "total_flags": compliance_stats["total_flags"] or 0,
                    "blocks": compliance_stats["blocks"] or 0,
                    "reviews": compliance_stats["reviews"] or 0,
                },
                "optimization": {
                    "savings_30d": round(savings["total_savings"] or 0, 2),
                    "shipments_optimized": savings["shipments_optimized"] or 0,
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

    async def run_accuracy_update_loop(self, interval_seconds: int = 300) -> None:
        """Periodically update accuracy metrics."""
        while True:
            try:
                await self.calculate_classification_accuracy()
                logger.debug("Updated classification accuracy metric")
            except Exception as e:
                logger.error(f"Error updating accuracy: {e}")
            await asyncio.sleep(interval_seconds)


async def main():
    """Example usage of metrics manager."""
    metrics = CustomsMetricsManager(
        postgres_dsn="postgresql://customs_svc:password@localhost:5432/customs_automation",
        metrics_port=8000,
    )
    await metrics.connect()
    metrics.start_server()

    # Record some sample metrics
    metrics.record_shipment("approved", 180.5)
    metrics.record_classification(0.92)
    metrics.record_compliance_flag("ad_cvd", "review")
    metrics.record_duty_savings("fta", 1250.00)
    metrics.record_document("cbp_7501")

    # Get summary
    summary = await metrics.get_metrics_summary()
    print(json.dumps(summary, indent=2))

    await metrics.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())`,
              },
            ],
          },
        ],
      },
    },
  ],
};
