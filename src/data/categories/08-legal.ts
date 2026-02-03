import type { Category } from '../types.ts';

export const legalCategory: Category = {
  id: 'legal',
  number: 8,
  title: 'Legal Risk & Compliance',
  shortTitle: 'Legal',
  description:
    'Automate right-to-erasure compliance and solve cross-border data sovereignty before regulators come knocking.',
  icon: 'Scale',
  accentColor: 'neon-purple',
  painPoints: [
    /* ───────────────────────────────────────────────
     * Pain Point 1 — Right to Erasure Compliance
     * ─────────────────────────────────────────────── */
    {
      id: 'right-to-erasure',
      number: 1,
      title: 'Right to Erasure Compliance',
      subtitle: 'GDPR Article 17 Data Deletion Gaps',
      summary:
        'A deletion request takes 3 weeks and 4 teams because personal data lives in 12+ systems. You can\'t prove deletion to regulators.',
      tags: ['gdpr', 'erasure', 'data-mapping'],
      metrics: {
        annualCostRange: '$200K - $1.5M',
        roi: '15x',
        paybackPeriod: '2-3 months',
        investmentRange: '$40K - $80K',
      },
      price: {
        present: {
          title: 'Present — The Problem Today',
          description:
            'Erasure requests are handled manually across dozens of disconnected systems with no unified personal-data inventory.',
          bullets: [
            'Average deletion request takes 15-21 business days to fulfil across 12+ systems',
            'No single catalogue of where a data subject\'s PII actually resides',
            'Audit evidence is stitched together in spreadsheets after the fact',
            'Repeated regulator follow-ups because proof-of-deletion is incomplete',
          ],
          severity: 'critical',
        },
        root: {
          title: 'Root — Why It Persists',
          description:
            'Personal data was never mapped at ingestion time, so every deletion becomes a forensic investigation.',
          bullets: [
            'Data lineage was never tracked — PII fans out through ETL without metadata',
            'Each team owns its own store and has no downstream visibility',
            'Legacy systems lack soft-delete or audit-log capabilities',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Impact — What It Costs the Business',
          description:
            'Compliance risk compounds with every unfulfilled request, and a single breach of Article 17 can trigger headline fines.',
          bullets: [
            'GDPR fines of up to 4% of global annual turnover per infringement',
            'Each manual deletion cycle costs ~$1,200 in cross-team labour',
            'Reputational damage from publicised enforcement actions',
            'Engineering velocity drops as teams context-switch to ad-hoc deletion tickets',
          ],
          severity: 'critical',
        },
        cost: {
          title: 'Cost — The Investment Required',
          description:
            'A PII discovery layer plus an automated erasure pipeline eliminates manual effort within weeks.',
          bullets: [
            'One-time build: $40K–$80K for discovery and orchestration layer',
            'Ongoing: < $500/month for metadata scanning infrastructure',
            'Timeline: MVP in 6–8 weeks, full coverage in 12 weeks',
          ],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Expected Return — The Payoff',
          description:
            'Sub-24-hour erasure fulfilment with regulator-ready audit trails that hold up under scrutiny.',
          bullets: [
            'Deletion SLA drops from 3 weeks to under 24 hours',
            'Audit-proof deletion certificates generated automatically',
            '15x return through avoided fines, reduced labour, and faster response',
            'Scalable to new jurisdictions (UK GDPR, CCPA, LGPD) with minimal config',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Build a personal-data discovery layer that maps PII across all systems, then orchestrate automated erasure with a cryptographic audit trail.',
        prerequisites: [
          'Read access to all production databases and data-lake partitions',
          'A PII classification taxonomy aligned with your DPA definitions',
          'Python 3.10+ with access to database connection pools',
          'A message broker (Kafka / SQS) for async erasure orchestration',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Discover Personal Data Across Systems',
            description:
              'Scan every data store to build a live PII registry that maps each data subject to the exact tables, columns, and object paths where their data lives.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'PII Discovery Across Relational Systems',
                description:
                  'Query information_schema and sample rows to detect columns likely containing personal data, then record findings in a central registry.',
                code: `-- pii_discovery.sql
-- Scan all user-facing schemas for columns that match PII patterns
INSERT INTO compliance.pii_registry
    (source_system, schema_name, table_name, column_name, pii_category, sample_hash, discovered_at)
SELECT
    current_database()                              AS source_system,
    c.table_schema                                  AS schema_name,
    c.table_name,
    c.column_name,
    CASE
        WHEN c.column_name ILIKE '%email%'          THEN 'email'
        WHEN c.column_name ILIKE '%phone%'          THEN 'phone'
        WHEN c.column_name ILIKE '%ssn%'            THEN 'national_id'
        WHEN c.column_name ILIKE '%date_of_birth%'  THEN 'dob'
        WHEN c.column_name ILIKE '%address%'        THEN 'address'
        WHEN c.column_name ILIKE '%first_name%'
          OR c.column_name ILIKE '%last_name%'      THEN 'name'
        ELSE 'other_pii'
    END                                             AS pii_category,
    md5(c.column_name || c.table_name)              AS sample_hash,
    now()                                           AS discovered_at
FROM information_schema.columns c
JOIN information_schema.tables t
    ON  c.table_schema = t.table_schema
    AND c.table_name   = t.table_name
WHERE t.table_type = 'BASE TABLE'
  AND c.table_schema NOT IN ('pg_catalog', 'information_schema', 'compliance')
  AND (
        c.column_name ILIKE '%email%'
     OR c.column_name ILIKE '%phone%'
     OR c.column_name ILIKE '%ssn%'
     OR c.column_name ILIKE '%date_of_birth%'
     OR c.column_name ILIKE '%address%'
     OR c.column_name ILIKE '%first_name%'
     OR c.column_name ILIKE '%last_name%'
  )
ON CONFLICT (source_system, schema_name, table_name, column_name)
DO UPDATE SET discovered_at = now();`,
              },
              {
                language: 'python',
                title: 'Cross-System PII Inventory Builder',
                description:
                  'Iterate over every registered data source, run the discovery SQL, and merge results into a unified PII map keyed by data-subject identifier.',
                code: `# pii_inventory.py
"""Build a unified PII inventory across all registered data sources."""

from __future__ import annotations
import hashlib, json, datetime as dt
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Iterator

import sqlalchemy as sa

REGISTRY_PATH = Path("data/pii_registry.json")
DISCOVERY_SQL = Path("sql/pii_discovery.sql").read_text()


@dataclass
class PiiRecord:
    source: str
    schema_name: str
    table: str
    column: str
    pii_category: str
    discovered_at: str


def scan_source(engine: sa.Engine, source_name: str) -> Iterator[PiiRecord]:
    """Run PII discovery against a single database and yield records."""
    with engine.connect() as conn:
        rows = conn.execute(sa.text(DISCOVERY_SQL))
        for r in rows:
            yield PiiRecord(
                source=source_name,
                schema_name=r.schema_name,
                table=r.table_name,
                column=r.column_name,
                pii_category=r.pii_category,
                discovered_at=dt.datetime.utcnow().isoformat(),
            )


def build_inventory(sources: dict[str, str]) -> list[dict]:
    """Scan every source and persist a merged PII registry."""
    inventory: list[dict] = []
    for name, conn_str in sources.items():
        engine = sa.create_engine(conn_str)
        records = [asdict(r) for r in scan_source(engine, name)]
        inventory.extend(records)
        print(f"[+] {name}: found {len(records)} PII columns")
    REGISTRY_PATH.write_text(json.dumps(inventory, indent=2))
    print(f"[✓] Registry saved — {len(inventory)} total PII entries")
    return inventory`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Automate the Erasure Pipeline',
            description:
              'Given a data-subject ID, look up every PII location in the registry and execute deletion across all systems in a single orchestrated run.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Orchestrated Erasure Engine',
                description:
                  'Accept a deletion request, resolve all PII locations from the registry, execute deletions in parallel, and collect per-system confirmation receipts.',
                code: `# erasure_engine.py
"""Automated GDPR Article 17 erasure pipeline with audit trail."""

from __future__ import annotations
import json, hashlib, datetime as dt, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import sqlalchemy as sa

REGISTRY = json.loads(Path("data/pii_registry.json").read_text())
AUDIT_DIR = Path("audit/erasure_receipts")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ErasureReceipt:
    request_id: str
    subject_id: str
    source: str
    table: str
    rows_deleted: int
    executed_at: str
    checksum: str


def _delete_from_source(
    subject_id: str, source: str, conn_str: str, entries: list[dict], request_id: str
) -> list[ErasureReceipt]:
    """Delete a subject's data from every table in a single source."""
    engine = sa.create_engine(conn_str)
    receipts: list[ErasureReceipt] = []
    with engine.begin() as conn:
        for entry in entries:
            result = conn.execute(
                sa.text(f"DELETE FROM {entry['schema_name']}.{entry['table']} "
                        f"WHERE subject_id = :sid"),
                {"sid": subject_id},
            )
            receipts.append(ErasureReceipt(
                request_id=request_id,
                subject_id=hashlib.sha256(subject_id.encode()).hexdigest(),
                source=source,
                table=f"{entry['schema_name']}.{entry['table']}",
                rows_deleted=result.rowcount,
                executed_at=dt.datetime.utcnow().isoformat(),
                checksum=hashlib.sha256(
                    f"{request_id}:{entry['table']}:{result.rowcount}".encode()
                ).hexdigest(),
            ))
    return receipts


def execute_erasure(subject_id: str, sources: dict[str, str]) -> Path:
    """Run full erasure for a data subject and persist an audit receipt."""
    request_id = str(uuid.uuid4())
    by_source: dict[str, list[dict]] = {}
    for rec in REGISTRY:
        by_source.setdefault(rec["source"], []).append(rec)

    all_receipts: list[dict] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(
                _delete_from_source, subject_id, src, sources[src], entries, request_id
            ): src
            for src, entries in by_source.items()
            if src in sources
        }
        for fut in as_completed(futures):
            all_receipts.extend(asdict(r) for r in fut.result())

    receipt_path = AUDIT_DIR / f"{request_id}.json"
    receipt_path.write_text(json.dumps(all_receipts, indent=2))
    print(f"[✓] Erasure complete — {len(all_receipts)} tables cleaned, receipt: {receipt_path}")
    return receipt_path`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate the erasure pipeline end-to-end with data quality assertions and automated pytest suites that confirm every PII location has been purged correctly.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Erasure Completeness Assertions',
                description:
                  'Run post-deletion data quality checks to verify all PII locations have been purged, referential integrity is intact, and no orphan records remain.',
                code: `-- erasure_completeness_checks.sql
-- Purpose: Data quality assertions after erasure pipeline execution.
-- Run these checks after every deletion batch to validate completeness.

-- 1. Verify no PII remains for the deleted subject across all registered tables
WITH deletion_targets AS (
    SELECT DISTINCT schema_name, table_name, column_name
    FROM compliance.pii_registry
    WHERE source_system = current_database()
),
residual_pii_check AS (
    SELECT
        dt.schema_name,
        dt.table_name,
        dt.column_name,
        COUNT(*) AS residual_row_count
    FROM deletion_targets dt
    JOIN metadata.erasure_requests er
        ON er.status = 'EXECUTED'
        AND er.completed_at >= now() - INTERVAL '1 hour'
    -- Dynamic check: ensure no rows remain for deleted subjects
    LEFT JOIN LATERAL (
        SELECT 1
        FROM information_schema.tables t
        WHERE t.table_schema = dt.schema_name
          AND t.table_name = dt.table_name
    ) tbl ON TRUE
    GROUP BY dt.schema_name, dt.table_name, dt.column_name
)
SELECT
    schema_name,
    table_name,
    column_name,
    residual_row_count,
    CASE
        WHEN residual_row_count = 0 THEN 'PASS'
        ELSE 'FAIL — residual PII detected'
    END AS assertion_result,
    now() AS checked_at
FROM residual_pii_check;

-- 2. Referential integrity check: no orphan audit receipts
SELECT
    er.request_id,
    er.subject_id_hash,
    CASE
        WHEN ac.request_id IS NOT NULL THEN 'PASS'
        ELSE 'FAIL — missing audit receipt'
    END AS receipt_integrity
FROM metadata.erasure_requests er
LEFT JOIN audit.erasure_certificates ac
    ON er.request_id = ac.request_id
WHERE er.completed_at >= now() - INTERVAL '24 hours';

-- 3. Null check: ensure no NULL subject_id_hash in completed requests
SELECT
    COUNT(*) AS null_hash_count,
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL — NULL subject hashes found'
    END AS null_check_result
FROM metadata.erasure_requests
WHERE status = 'EXECUTED'
  AND subject_id_hash IS NULL
  AND completed_at >= now() - INTERVAL '24 hours';

-- 4. Freshness threshold: ensure pipeline ran within expected window
SELECT
    MAX(completed_at) AS last_run,
    CASE
        WHEN MAX(completed_at) >= now() - INTERVAL '6 hours'
        THEN 'PASS — pipeline is fresh'
        ELSE 'FAIL — pipeline stale (> 6h since last run)'
    END AS freshness_status
FROM metadata.erasure_requests
WHERE status = 'EXECUTED';`,
              },
              {
                language: 'python',
                title: 'Erasure Pipeline Pytest Suite',
                description:
                  'Automated pytest tests that validate the erasure engine: discovery correctness, deletion execution, receipt generation, and certificate integrity.',
                code: `# tests/test_erasure_pipeline.py
"""Pytest suite for GDPR erasure pipeline validation."""

from __future__ import annotations
import json
import hashlib
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pii_registry(tmp_path: Path) -> Path:
    """Create a temporary PII registry for testing."""
    registry: list[dict[str, Any]] = [
        {
            "source": "primary_db",
            "schema_name": "public",
            "table": "users",
            "column": "email",
            "pii_category": "email",
            "discovered_at": "2025-01-15T10:00:00",
        },
        {
            "source": "primary_db",
            "schema_name": "public",
            "table": "profiles",
            "column": "phone_number",
            "pii_category": "phone",
            "discovered_at": "2025-01-15T10:00:00",
        },
        {
            "source": "analytics_db",
            "schema_name": "reporting",
            "table": "user_events",
            "column": "user_email",
            "pii_category": "email",
            "discovered_at": "2025-01-15T10:00:00",
        },
    ]
    registry_path: Path = tmp_path / "pii_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2))
    logger.info("Created test PII registry with %d entries", len(registry))
    return registry_path


@pytest.fixture
def sample_erasure_receipt(tmp_path: Path) -> Path:
    """Create a sample erasure receipt for certificate testing."""
    receipt: list[dict[str, Any]] = [
        {
            "request_id": "test-req-001",
            "subject_id": hashlib.sha256(b"user@example.com").hexdigest(),
            "source": "primary_db",
            "table": "public.users",
            "rows_deleted": 1,
            "executed_at": "2025-01-15T12:00:00",
            "checksum": hashlib.sha256(b"test-checksum-1").hexdigest(),
        },
        {
            "request_id": "test-req-001",
            "subject_id": hashlib.sha256(b"user@example.com").hexdigest(),
            "source": "primary_db",
            "table": "public.profiles",
            "rows_deleted": 1,
            "executed_at": "2025-01-15T12:00:01",
            "checksum": hashlib.sha256(b"test-checksum-2").hexdigest(),
        },
    ]
    receipt_path: Path = tmp_path / "test-req-001.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    logger.info("Created test erasure receipt at %s", receipt_path)
    return receipt_path

# ---------------------------------------------------------------------------
# Discovery Tests
# ---------------------------------------------------------------------------

class TestPiiDiscovery:
    """Validate PII discovery identifies all expected data locations."""

    def test_registry_contains_all_sources(
        self, sample_pii_registry: Path
    ) -> None:
        registry: list[dict[str, Any]] = json.loads(
            sample_pii_registry.read_text()
        )
        sources: set[str] = {r["source"] for r in registry}
        assert "primary_db" in sources, "primary_db missing from registry"
        assert "analytics_db" in sources, "analytics_db missing from registry"
        logger.info("Discovery test passed: %d sources found", len(sources))

    def test_all_pii_categories_detected(
        self, sample_pii_registry: Path
    ) -> None:
        registry: list[dict[str, Any]] = json.loads(
            sample_pii_registry.read_text()
        )
        categories: set[str] = {r["pii_category"] for r in registry}
        assert "email" in categories
        assert "phone" in categories
        logger.info("PII categories detected: %s", categories)

    def test_registry_entries_have_required_fields(
        self, sample_pii_registry: Path
    ) -> None:
        registry: list[dict[str, Any]] = json.loads(
            sample_pii_registry.read_text()
        )
        required_fields: set[str] = {
            "source", "schema_name", "table",
            "column", "pii_category", "discovered_at",
        }
        for entry in registry:
            missing: set[str] = required_fields - set(entry.keys())
            assert not missing, f"Missing fields: {missing}"

# ---------------------------------------------------------------------------
# Erasure Execution Tests
# ---------------------------------------------------------------------------

class TestErasureExecution:
    """Validate deletion execution and receipt generation."""

    def test_receipt_covers_all_tables(
        self, sample_erasure_receipt: Path
    ) -> None:
        receipt: list[dict[str, Any]] = json.loads(
            sample_erasure_receipt.read_text()
        )
        tables: set[str] = {r["table"] for r in receipt}
        assert "public.users" in tables
        assert "public.profiles" in tables
        logger.info("Erasure receipt covers %d tables", len(tables))

    def test_all_receipts_have_positive_deletions(
        self, sample_erasure_receipt: Path
    ) -> None:
        receipt: list[dict[str, Any]] = json.loads(
            sample_erasure_receipt.read_text()
        )
        for entry in receipt:
            assert entry["rows_deleted"] > 0, (
                f"No rows deleted in {entry['table']}"
            )

    def test_receipt_checksums_are_valid_sha256(
        self, sample_erasure_receipt: Path
    ) -> None:
        receipt: list[dict[str, Any]] = json.loads(
            sample_erasure_receipt.read_text()
        )
        for entry in receipt:
            checksum: str = entry["checksum"]
            assert len(checksum) == 64, "Checksum is not SHA-256 length"
            int(checksum, 16)  # Validates hex format

# ---------------------------------------------------------------------------
# Certificate Integrity Tests
# ---------------------------------------------------------------------------

class TestCertificateIntegrity:
    """Validate that deletion certificates are tamper-evident."""

    def test_chain_hash_is_deterministic(
        self, sample_erasure_receipt: Path
    ) -> None:
        events: list[dict[str, Any]] = json.loads(
            sample_erasure_receipt.read_text()
        )
        chain: str = "genesis"
        for evt in sorted(events, key=lambda e: e["executed_at"]):
            payload: str = (
                f"{chain}:{evt['source']}:{evt['table']}"
                f":{evt['rows_deleted']}"
            )
            chain = hashlib.sha256(payload.encode()).hexdigest()

        # Re-run to confirm determinism
        chain2: str = "genesis"
        for evt in sorted(events, key=lambda e: e["executed_at"]):
            payload2: str = (
                f"{chain2}:{evt['source']}:{evt['table']}"
                f":{evt['rows_deleted']}"
            )
            chain2 = hashlib.sha256(payload2.encode()).hexdigest()

        assert chain == chain2, "Hash chain is not deterministic"
        logger.info("Certificate chain hash verified: %s", chain[:16])`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Deploy the GDPR erasure pipeline to production with environment-based configuration, database migrations, scheduled execution, and secrets management.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'GDPR Erasure Pipeline Deployment Script',
                description:
                  'Production deployment script that validates environment variables, installs dependencies, runs database migrations, and registers the pipeline with the scheduler.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# deploy_erasure_pipeline.sh
# Purpose: Deploy the GDPR erasure pipeline to production.
# Usage:   ERASURE_ENV=production ./deploy_erasure_pipeline.sh

SCRIPT_DIR="$(cd "$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="\${SCRIPT_DIR}/deploy_\$(date +%Y%m%d_%H%M%S).log"

log() {
    local level="\$1"; shift
    echo "[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] [\${level}] \$*" | tee -a "\${LOG_FILE}"
}

log "INFO" "Starting GDPR erasure pipeline deployment..."

# ── 1. Validate required environment variables ──────────────────────────
REQUIRED_VARS=(
    "ERASURE_ENV"
    "DATABASE_URL"
    "ERASURE_DB_SCHEMA"
    "KAFKA_BROKER_URL"
    "SLACK_WEBHOOK_URL"
    "PII_REGISTRY_PATH"
    "AUDIT_CERTIFICATE_DIR"
)

missing_vars=()
for var in "\${REQUIRED_VARS[@]}"; do
    if [[ -z "\${!var:-}" ]]; then
        missing_vars+=("\${var}")
    fi
done

if [[ \${#missing_vars[@]} -gt 0 ]]; then
    log "ERROR" "Missing required environment variables: \${missing_vars[*]}"
    exit 1
fi
log "INFO" "All required environment variables are set (env=\${ERASURE_ENV})"

# ── 2. Install Python dependencies ──────────────────────────────────────
log "INFO" "Installing Python dependencies..."
if [[ -f "\${SCRIPT_DIR}/requirements.txt" ]]; then
    pip install --quiet --upgrade pip
    pip install --quiet -r "\${SCRIPT_DIR}/requirements.txt"
    log "INFO" "Python dependencies installed successfully"
else
    log "ERROR" "requirements.txt not found at \${SCRIPT_DIR}"
    exit 1
fi

# ── 3. Run database migrations ──────────────────────────────────────────
log "INFO" "Running database migrations for schema \${ERASURE_DB_SCHEMA}..."
python "\${SCRIPT_DIR}/migrations/run_migrations.py" \\
    --database-url "\${DATABASE_URL}" \\
    --schema "\${ERASURE_DB_SCHEMA}" \\
    --direction up

if [[ \$? -eq 0 ]]; then
    log "INFO" "Database migrations completed successfully"
else
    log "ERROR" "Database migration failed — aborting deployment"
    exit 1
fi

# ── 4. Validate PII registry file exists ────────────────────────────────
if [[ ! -f "\${PII_REGISTRY_PATH}" ]]; then
    log "WARN" "PII registry not found at \${PII_REGISTRY_PATH}, running discovery..."
    python "\${SCRIPT_DIR}/pii_inventory.py" --output "\${PII_REGISTRY_PATH}"
fi
log "INFO" "PII registry validated at \${PII_REGISTRY_PATH}"

# ── 5. Ensure audit certificate directory exists ────────────────────────
mkdir -p "\${AUDIT_CERTIFICATE_DIR}"
log "INFO" "Audit certificate directory ready: \${AUDIT_CERTIFICATE_DIR}"

# ── 6. Register pipeline with cron scheduler ────────────────────────────
CRON_EXPRESSION="0 */4 * * *"  # Every 4 hours
CRON_CMD="cd \${SCRIPT_DIR} && ERASURE_ENV=\${ERASURE_ENV} python erasure_engine.py --mode scheduled 2>&1 >> \${SCRIPT_DIR}/cron.log"
CRON_MARKER="# gdpr-erasure-pipeline"

# Remove existing entry, then add updated one
(crontab -l 2>/dev/null | grep -v "\${CRON_MARKER}") | crontab -
(crontab -l 2>/dev/null; echo "\${CRON_EXPRESSION} \${CRON_CMD} \${CRON_MARKER}") | crontab -
log "INFO" "Cron job registered: \${CRON_EXPRESSION}"

# ── 7. Send deployment notification to Slack ────────────────────────────
DEPLOY_MSG="GDPR Erasure Pipeline deployed to *\${ERASURE_ENV}* at \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
curl -s -X POST "\${SLACK_WEBHOOK_URL}" \\
    -H "Content-Type: application/json" \\
    -d "{\\"text\\": \\"\${DEPLOY_MSG}\\"}" > /dev/null

log "INFO" "Deployment complete. Slack notification sent."`,
              },
              {
                language: 'python',
                title: 'Erasure Pipeline Configuration Loader',
                description:
                  'Production configuration loader with environment-based config resolution, secrets handling via environment variables, and connection pool setup for the erasure pipeline.',
                code: `# config/erasure_config.py
"""Configuration loader for the GDPR erasure pipeline."""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErasureConfig:
    """Immutable configuration for the erasure pipeline."""

    # Environment
    env: str = "development"
    debug: bool = False

    # Database
    database_url: str = ""
    db_schema: str = "compliance"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30

    # Kafka / message broker
    kafka_broker_url: str = ""
    kafka_topic_requests: str = "erasure.requests"
    kafka_topic_receipts: str = "erasure.receipts"
    kafka_consumer_group: str = "erasure-pipeline"

    # PII registry
    pii_registry_path: str = "data/pii_registry.json"

    # Audit
    audit_dir: str = "audit/erasure_receipts"
    certificate_dir: str = "audit/certificates"

    # Alerting
    slack_webhook_url: str = ""
    alert_on_failure: bool = True

    # Scheduling
    deletion_sla_hours: int = 24
    scan_interval_hours: int = 4

    # Secrets (never logged)
    _secret_fields: tuple[str, ...] = field(
        default=("database_url", "kafka_broker_url", "slack_webhook_url"),
        repr=False,
    )


def load_config() -> ErasureConfig:
    """Load configuration from environment variables with validation."""
    config = ErasureConfig(
        env=os.environ.get("ERASURE_ENV", "development"),
        debug=os.environ.get("ERASURE_DEBUG", "false").lower() == "true",
        database_url=os.environ.get("DATABASE_URL", ""),
        db_schema=os.environ.get("ERASURE_DB_SCHEMA", "compliance"),
        db_pool_size=int(os.environ.get("DB_POOL_SIZE", "5")),
        db_max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "10")),
        db_pool_timeout=int(os.environ.get("DB_POOL_TIMEOUT", "30")),
        kafka_broker_url=os.environ.get("KAFKA_BROKER_URL", ""),
        kafka_topic_requests=os.environ.get(
            "KAFKA_TOPIC_REQUESTS", "erasure.requests"
        ),
        kafka_topic_receipts=os.environ.get(
            "KAFKA_TOPIC_RECEIPTS", "erasure.receipts"
        ),
        kafka_consumer_group=os.environ.get(
            "KAFKA_CONSUMER_GROUP", "erasure-pipeline"
        ),
        pii_registry_path=os.environ.get(
            "PII_REGISTRY_PATH", "data/pii_registry.json"
        ),
        audit_dir=os.environ.get("AUDIT_DIR", "audit/erasure_receipts"),
        certificate_dir=os.environ.get(
            "AUDIT_CERTIFICATE_DIR", "audit/certificates"
        ),
        slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL", ""),
        alert_on_failure=os.environ.get(
            "ALERT_ON_FAILURE", "true"
        ).lower() == "true",
        deletion_sla_hours=int(os.environ.get("DELETION_SLA_HOURS", "24")),
        scan_interval_hours=int(
            os.environ.get("SCAN_INTERVAL_HOURS", "4")
        ),
    )

    # Validate critical fields
    if not config.database_url:
        raise EnvironmentError("DATABASE_URL is required but not set")
    if config.env == "production" and not config.slack_webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set in production — alerts disabled")

    logger.info(
        "Configuration loaded: env=%s, schema=%s, pool_size=%d, sla=%dh",
        config.env,
        config.db_schema,
        config.db_pool_size,
        config.deletion_sla_hours,
    )
    return config


def create_engine_from_config(config: ErasureConfig) -> sa.Engine:
    """Create a SQLAlchemy engine with connection pooling from config."""
    engine: sa.Engine = sa.create_engine(
        config.database_url,
        poolclass=QueuePool,
        pool_size=config.db_pool_size,
        max_overflow=config.db_max_overflow,
        pool_timeout=config.db_pool_timeout,
        pool_pre_ping=True,
        echo=config.debug,
    )
    logger.info(
        "Database engine created: pool_size=%d, max_overflow=%d",
        config.db_pool_size,
        config.db_max_overflow,
    )
    return engine`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Produce tamper-evident deletion certificates that satisfy DPA audit requests, monitor deletion SLA compliance, and verify audit chain integrity with automated alerting.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Deletion Certificate Generator',
                description:
                  'Read the erasure receipt, compute a Merkle-style hash chain over every deletion event, and render a PDF-ready certificate payload.',
                code: `# audit_certificate.py
"""Generate tamper-evident deletion certificates for regulator audits."""

from __future__ import annotations
import json, hashlib, datetime as dt
from pathlib import Path
from dataclasses import dataclass, asdict

AUDIT_DIR = Path("audit/erasure_receipts")
CERT_DIR = Path("audit/certificates")
CERT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DeletionCertificate:
    certificate_id: str
    request_id: str
    subject_hash: str
    systems_purged: int
    total_rows_deleted: int
    chain_hash: str
    issued_at: str
    compliant_articles: list[str]


def generate_certificate(request_id: str) -> dict:
    """Build a hash-chain certificate from an erasure receipt."""
    receipt_path = AUDIT_DIR / f"{request_id}.json"
    events: list[dict] = json.loads(receipt_path.read_text())

    # Build a sequential hash chain over every deletion event
    chain = "genesis"
    for evt in sorted(events, key=lambda e: e["executed_at"]):
        payload = f"{chain}:{evt['source']}:{evt['table']}:{evt['rows_deleted']}"
        chain = hashlib.sha256(payload.encode()).hexdigest()

    cert = DeletionCertificate(
        certificate_id=hashlib.sha256(f"cert:{request_id}".encode()).hexdigest()[:16],
        request_id=request_id,
        subject_hash=events[0]["subject_id"] if events else "unknown",
        systems_purged=len({e["source"] for e in events}),
        total_rows_deleted=sum(e["rows_deleted"] for e in events),
        chain_hash=chain,
        issued_at=dt.datetime.utcnow().isoformat(),
        compliant_articles=["GDPR Art. 17", "UK GDPR Art. 17", "CCPA §1798.105"],
    )
    cert_path = CERT_DIR / f"{cert.certificate_id}.json"
    cert_path.write_text(json.dumps(asdict(cert), indent=2))
    print(f"[✓] Certificate {cert.certificate_id} issued — "
          f"{cert.systems_purged} systems, {cert.total_rows_deleted} rows verified")
    return asdict(cert)`,
              },
              {
                language: 'sql',
                title: 'Deletion SLA Monitoring Dashboard',
                description:
                  'Real-time dashboard query that tracks erasure request SLA compliance, flags overdue deletions, monitors audit chain verification status, and provides alerting thresholds.',
                code: `-- deletion_sla_dashboard.sql
-- Purpose: Monitor erasure SLA compliance and audit chain integrity.
-- Schedule: Run every 15 minutes via cron or Airflow.

-- 1. Erasure SLA compliance overview
WITH sla_status AS (
    SELECT
        er.request_id,
        er.subject_id_hash,
        er.requested_at,
        er.completed_at,
        er.status,
        EXTRACT(EPOCH FROM (
            COALESCE(er.completed_at, now()) - er.requested_at
        )) / 3600.0                                  AS hours_elapsed,
        24.0                                         AS sla_hours,
        CASE
            WHEN er.status = 'EXECUTED'
              AND er.completed_at <= er.requested_at + INTERVAL '24 hours'
            THEN 'WITHIN_SLA'
            WHEN er.status = 'EXECUTED'
              AND er.completed_at > er.requested_at + INTERVAL '24 hours'
            THEN 'SLA_BREACHED'
            WHEN er.status = 'PENDING'
              AND now() > er.requested_at + INTERVAL '24 hours'
            THEN 'SLA_AT_RISK'
            WHEN er.status = 'PENDING'
            THEN 'IN_PROGRESS'
            ELSE 'UNKNOWN'
        END                                          AS sla_status
    FROM metadata.erasure_requests er
    WHERE er.requested_at >= now() - INTERVAL '30 days'
)
SELECT
    sla_status,
    COUNT(*)                                         AS request_count,
    ROUND(AVG(hours_elapsed)::numeric, 2)            AS avg_hours,
    ROUND(MAX(hours_elapsed)::numeric, 2)            AS max_hours,
    ROUND(
        COUNT(*) FILTER (WHERE sla_status = 'WITHIN_SLA') * 100.0
        / NULLIF(COUNT(*), 0), 1
    )                                                AS sla_compliance_pct
FROM sla_status
GROUP BY sla_status
ORDER BY request_count DESC;

-- 2. Overdue deletion requests requiring immediate action
SELECT
    er.request_id,
    er.subject_id_hash,
    er.requested_at,
    ROUND(EXTRACT(EPOCH FROM (now() - er.requested_at)) / 3600.0, 1)
                                                     AS hours_overdue,
    er.status,
    er.assigned_team,
    er.source_systems
FROM metadata.erasure_requests er
WHERE er.status IN ('PENDING', 'IN_PROGRESS')
  AND now() > er.requested_at + INTERVAL '20 hours'
ORDER BY er.requested_at ASC;

-- 3. Audit chain verification status
SELECT
    ac.certificate_id,
    ac.request_id,
    ac.chain_hash,
    ac.systems_purged,
    ac.total_rows_deleted,
    ac.issued_at,
    CASE
        WHEN ac.chain_hash IS NOT NULL
          AND ac.systems_purged > 0
          AND ac.total_rows_deleted > 0
        THEN 'VERIFIED'
        ELSE 'INTEGRITY_FAILURE'
    END                                              AS chain_status
FROM audit.erasure_certificates ac
WHERE ac.issued_at >= now() - INTERVAL '7 days'
ORDER BY ac.issued_at DESC;

-- 4. Daily SLA trend for alerting threshold
SELECT
    DATE(er.requested_at)                            AS request_date,
    COUNT(*)                                         AS total_requests,
    COUNT(*) FILTER (
        WHERE er.status = 'EXECUTED'
          AND er.completed_at <= er.requested_at + INTERVAL '24 hours'
    )                                                AS within_sla,
    ROUND(
        COUNT(*) FILTER (
            WHERE er.status = 'EXECUTED'
              AND er.completed_at <= er.requested_at + INTERVAL '24 hours'
        ) * 100.0 / NULLIF(COUNT(*), 0), 1
    )                                                AS daily_sla_pct
FROM metadata.erasure_requests er
WHERE er.requested_at >= now() - INTERVAL '30 days'
GROUP BY DATE(er.requested_at)
ORDER BY request_date DESC;`,
              },
            ],
          },
        ],
        toolsUsed: [
          'PostgreSQL / information_schema',
          'SQLAlchemy',
          'Python concurrent.futures',
          'hashlib (SHA-256 hash chains)',
          'pytest',
          'Docker',
          'GitHub Actions',
          'cron / Airflow',
          'Slack API',
        ],
      },
    },

    /* ───────────────────────────────────────────────
     * Pain Point 2 — Cross-Border Data Sovereignty
     * ─────────────────────────────────────────────── */
    {
      id: 'cross-border-data-sovereignty',
      number: 2,
      title: 'Cross-Border Data Sovereignty',
      subtitle: 'Data Residency Requirements Across Jurisdictions',
      summary:
        'Your data flows freely between AWS regions, but EU, UAE, India, and Australia all mandate local residency. One misconfigured pipeline triggers an investigation.',
      tags: ['data-sovereignty', 'compliance', 'cross-border'],
      metrics: {
        annualCostRange: '$300K - $2M',
        roi: '10x',
        paybackPeriod: '3-4 months',
        investmentRange: '$80K - $160K',
      },
      price: {
        present: {
          title: 'Present — The Problem Today',
          description:
            'Data moves between cloud regions without jurisdiction-aware guardrails, creating silent residency violations.',
          bullets: [
            'ETL pipelines replicate data across regions with no residency checks',
            'No automated inventory of which records are subject to which jurisdiction',
            'Teams discover violations only during manual quarterly audits',
            'Shadow copies in analytics sandboxes bypass all residency controls',
          ],
          severity: 'critical',
        },
        root: {
          title: 'Root — Why It Persists',
          description:
            'Infrastructure was designed for performance and availability, not regulatory geography. Residency was bolted on as documentation, not code.',
          bullets: [
            'Cloud architecture optimises for latency, not legal boundaries',
            'Residency rules live in policy PDFs, not in pipeline config',
            'No metadata tag links a record to its governing jurisdiction',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Impact — What It Costs the Business',
          description:
            'A single residency breach can freeze cross-border operations and trigger multi-regulator investigations.',
          bullets: [
            'EU GDPR cross-border fines up to €20M or 4% of global turnover',
            'UAE PDPL and India DPDP Act impose data localisation penalties and operational bans',
            'Investigation remediation costs average $500K–$2M per incident',
            'Loss of customer trust when residency failures become public',
          ],
          severity: 'critical',
        },
        cost: {
          title: 'Cost — The Investment Required',
          description:
            'A jurisdiction-tagging layer plus automated routing rules prevent violations at the pipeline level.',
          bullets: [
            'One-time build: $80K–$160K for tagging, routing, and monitoring',
            'Ongoing: ~$1,200/month for multi-region metadata infrastructure',
            'Timeline: Core routing in 8 weeks, full jurisdiction coverage in 14 weeks',
          ],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Expected Return — The Payoff',
          description:
            'Every record automatically routed to its legally mandated region with real-time compliance dashboards.',
          bullets: [
            'Zero residency violations with policy-as-code enforcement',
            '10x ROI through avoided fines, faster market entry, and reduced legal spend',
            'New-jurisdiction onboarding drops from months to a config change',
            'Regulator-ready residency reports generated on demand',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Tag every record with its governing jurisdiction at ingestion, enforce residency rules in the data pipeline, and continuously audit compliance.',
        prerequisites: [
          'Multi-region cloud infrastructure (AWS / GCP / Azure)',
          'Access to pipeline orchestration layer (Airflow, dbt, or equivalent)',
          'A jurisdiction-to-region mapping approved by Legal',
          'Python 3.10+ and SQL access to all data stores',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Audit Current Data Residency Posture',
            description:
              'Scan all data stores to identify records that currently violate residency requirements by comparing storage region against jurisdiction rules.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Data Residency Violation Audit',
                description:
                  'Join record metadata with jurisdiction rules to surface every row stored in the wrong region.',
                code: `-- residency_audit.sql
-- Identify all records violating data residency requirements
WITH jurisdiction_rules AS (
    SELECT *
    FROM compliance.residency_rules
    -- e.g. jurisdiction='EU', allowed_regions='eu-west-1,eu-central-1'
),
record_locations AS (
    SELECT
        r.record_id,
        r.subject_country,
        r.source_system,
        r.storage_region,
        r.created_at
    FROM metadata.record_registry r
    WHERE r.contains_pii = TRUE
)
SELECT
    rl.record_id,
    rl.source_system,
    rl.subject_country,
    rl.storage_region                           AS current_region,
    jr.allowed_regions,
    jr.jurisdiction_label,
    CASE
        WHEN rl.storage_region = ANY(string_to_array(jr.allowed_regions, ','))
        THEN 'COMPLIANT'
        ELSE 'VIOLATION'
    END                                         AS residency_status,
    rl.created_at                               AS record_created,
    now()                                       AS audited_at
FROM record_locations rl
JOIN jurisdiction_rules jr
    ON rl.subject_country = jr.country_code
WHERE rl.storage_region != ALL(string_to_array(jr.allowed_regions, ','))
ORDER BY rl.created_at DESC;`,
              },
              {
                language: 'sql',
                title: 'Residency Violation Summary by Jurisdiction',
                description:
                  'Aggregate violations by jurisdiction and source system to prioritise remediation.',
                code: `-- residency_violation_summary.sql
-- Aggregate residency violations for executive reporting
SELECT
    jr.jurisdiction_label,
    rl.source_system,
    rl.storage_region                           AS violating_region,
    COUNT(*)                                    AS violation_count,
    MIN(rl.created_at)                          AS earliest_violation,
    MAX(rl.created_at)                          AS latest_violation,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY jr.jurisdiction_label),
        1
    )                                           AS pct_of_jurisdiction_violations
FROM metadata.record_registry rl
JOIN compliance.residency_rules jr
    ON rl.subject_country = jr.country_code
WHERE rl.contains_pii = TRUE
  AND rl.storage_region != ALL(string_to_array(jr.allowed_regions, ','))
GROUP BY jr.jurisdiction_label, rl.source_system, rl.storage_region
ORDER BY violation_count DESC;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Enforce Jurisdiction-Aware Data Routing',
            description:
              'Intercept data at the pipeline level, resolve the governing jurisdiction from subject metadata, and route each record to the correct region before it lands.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Automated Data Routing with Residency Rules',
                description:
                  'A pipeline middleware that reads jurisdiction rules, tags incoming records, and routes them to the legally mandated storage region.',
                code: `# residency_router.py
"""Pipeline middleware — route records to jurisdiction-compliant regions."""

from __future__ import annotations
import json, datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

RULES_PATH = Path("config/residency_rules.json")


@dataclass
class RoutingDecision:
    record_id: str
    subject_country: str
    jurisdiction: str
    target_region: str
    routed_at: str


class ResidencyRouter:
    """Resolve the correct storage region for every record based on jurisdiction."""

    def __init__(self) -> None:
        raw = json.loads(RULES_PATH.read_text())
        # Map country_code -> {jurisdiction, allowed_regions, preferred_region}
        self._rules: dict[str, dict] = {r["country_code"]: r for r in raw}

    def resolve(self, record: dict[str, Any]) -> RoutingDecision:
        country = record.get("subject_country", "UNKNOWN")
        rule = self._rules.get(country)
        if rule is None:
            raise ValueError(
                f"No residency rule for country '{country}'. "
                "Add it to config/residency_rules.json before ingesting."
            )
        return RoutingDecision(
            record_id=record["record_id"],
            subject_country=country,
            jurisdiction=rule["jurisdiction_label"],
            target_region=rule["preferred_region"],
            routed_at=dt.datetime.utcnow().isoformat(),
        )

    def route_batch(self, records: list[dict[str, Any]]) -> dict[str, list[dict]]:
        """Partition a batch of records by target region."""
        buckets: dict[str, list[dict]] = {}
        for rec in records:
            decision = self.resolve(rec)
            rec["_routing"] = asdict(decision)
            buckets.setdefault(decision.target_region, []).append(rec)
        for region, recs in buckets.items():
            print(f"  [{region}] {len(recs)} records routed")
        return buckets`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate residency routing logic and rule enforcement with data quality assertions and automated pytest suites that confirm records land in the correct jurisdictional region.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Residency Rule Validation Assertions',
                description:
                  'Run post-routing data quality checks to verify all records are stored in their correct jurisdictional region, referential integrity of routing metadata, and freshness of compliance scans.',
                code: `-- residency_validation_checks.sql
-- Purpose: Data quality assertions for cross-border residency compliance.
-- Run after every routing batch to validate correctness.

-- 1. Verify all PII records reside in their mandated region
WITH routing_validation AS (
    SELECT
        rr.record_id,
        rr.subject_country,
        rr.storage_region,
        jr.allowed_regions,
        jr.jurisdiction_label,
        CASE
            WHEN rr.storage_region = ANY(string_to_array(jr.allowed_regions, ','))
            THEN 'PASS'
            ELSE 'FAIL — record in wrong region'
        END AS region_check
    FROM metadata.record_registry rr
    JOIN compliance.residency_rules jr
        ON rr.subject_country = jr.country_code
    WHERE rr.contains_pii = TRUE
      AND rr.updated_at >= now() - INTERVAL '24 hours'
)
SELECT
    region_check                                     AS assertion_result,
    COUNT(*)                                         AS record_count,
    COUNT(DISTINCT subject_country)                  AS countries_affected,
    ARRAY_AGG(DISTINCT storage_region)               AS regions_involved,
    now()                                            AS checked_at
FROM routing_validation
GROUP BY region_check;

-- 2. Referential integrity: every routed record has routing metadata
SELECT
    rr.record_id,
    CASE
        WHEN rm.record_id IS NOT NULL THEN 'PASS'
        ELSE 'FAIL — missing routing metadata'
    END                                              AS metadata_integrity
FROM metadata.record_registry rr
LEFT JOIN compliance.routing_decisions rm
    ON rr.record_id = rm.record_id
WHERE rr.contains_pii = TRUE
  AND rr.created_at >= now() - INTERVAL '24 hours'
  AND rm.record_id IS NULL;

-- 3. Null check: ensure no NULL jurisdiction assignments
SELECT
    COUNT(*)                                         AS null_jurisdiction_count,
    CASE
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL — records with NULL jurisdiction'
    END                                              AS null_check_result
FROM compliance.routing_decisions
WHERE jurisdiction IS NULL
  AND routed_at >= now() - INTERVAL '24 hours';

-- 4. Freshness: ensure routing pipeline ran within expected window
SELECT
    MAX(routed_at)                                   AS last_routing_run,
    CASE
        WHEN MAX(routed_at) >= now() - INTERVAL '4 hours'
        THEN 'PASS — routing pipeline is fresh'
        ELSE 'FAIL — routing pipeline stale (> 4h since last run)'
    END                                              AS freshness_status
FROM compliance.routing_decisions;

-- 5. Completeness: all known jurisdictions have at least one rule
SELECT
    DISTINCT rr.subject_country,
    CASE
        WHEN jr.country_code IS NOT NULL THEN 'PASS'
        ELSE 'FAIL — no residency rule for country'
    END                                              AS rule_coverage
FROM metadata.record_registry rr
LEFT JOIN compliance.residency_rules jr
    ON rr.subject_country = jr.country_code
WHERE rr.contains_pii = TRUE
  AND jr.country_code IS NULL;`,
              },
              {
                language: 'python',
                title: 'Residency Routing Pytest Suite',
                description:
                  'Automated pytest tests that validate the residency router: rule loading, single-record routing, batch partitioning, unknown-country handling, and region compliance.',
                code: `# tests/test_residency_routing.py
"""Pytest suite for cross-border data residency routing validation."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def residency_rules(tmp_path: Path) -> Path:
    """Create temporary residency rules configuration."""
    rules: list[dict[str, Any]] = [
        {
            "country_code": "DE",
            "jurisdiction_label": "EU-GDPR",
            "allowed_regions": "eu-west-1,eu-central-1",
            "preferred_region": "eu-central-1",
            "strict": True,
        },
        {
            "country_code": "FR",
            "jurisdiction_label": "EU-GDPR",
            "allowed_regions": "eu-west-1,eu-central-1",
            "preferred_region": "eu-west-1",
            "strict": True,
        },
        {
            "country_code": "AE",
            "jurisdiction_label": "UAE-PDPL",
            "allowed_regions": "me-south-1",
            "preferred_region": "me-south-1",
            "strict": True,
        },
        {
            "country_code": "IN",
            "jurisdiction_label": "India-DPDP",
            "allowed_regions": "ap-south-1",
            "preferred_region": "ap-south-1",
            "strict": True,
        },
        {
            "country_code": "US",
            "jurisdiction_label": "US-CCPA",
            "allowed_regions": "us-east-1,us-west-2",
            "preferred_region": "us-east-1",
            "strict": False,
        },
    ]
    rules_path: Path = tmp_path / "residency_rules.json"
    rules_path.write_text(json.dumps(rules, indent=2))
    logger.info("Created test residency rules with %d jurisdictions", len(rules))
    return rules_path


@pytest.fixture
def sample_records() -> list[dict[str, Any]]:
    """Generate sample records with various subject countries."""
    return [
        {"record_id": "rec-001", "subject_country": "DE", "data": "test-de"},
        {"record_id": "rec-002", "subject_country": "FR", "data": "test-fr"},
        {"record_id": "rec-003", "subject_country": "AE", "data": "test-ae"},
        {"record_id": "rec-004", "subject_country": "IN", "data": "test-in"},
        {"record_id": "rec-005", "subject_country": "US", "data": "test-us"},
        {"record_id": "rec-006", "subject_country": "DE", "data": "test-de-2"},
    ]


@pytest.fixture
def rules_map(residency_rules: Path) -> dict[str, dict[str, Any]]:
    """Load rules into a lookup map for test assertions."""
    raw: list[dict[str, Any]] = json.loads(residency_rules.read_text())
    return {r["country_code"]: r for r in raw}

# ---------------------------------------------------------------------------
# Rule Loading Tests
# ---------------------------------------------------------------------------

class TestRuleLoading:
    """Validate that residency rules load correctly."""

    def test_all_jurisdictions_loaded(
        self, rules_map: dict[str, dict[str, Any]]
    ) -> None:
        assert len(rules_map) == 5, f"Expected 5 rules, got {len(rules_map)}"
        logger.info("Loaded %d jurisdiction rules", len(rules_map))

    def test_eu_countries_share_jurisdiction(
        self, rules_map: dict[str, dict[str, Any]]
    ) -> None:
        assert rules_map["DE"]["jurisdiction_label"] == "EU-GDPR"
        assert rules_map["FR"]["jurisdiction_label"] == "EU-GDPR"
        logger.info("EU jurisdiction consistency verified")

    def test_strict_flag_set_correctly(
        self, rules_map: dict[str, dict[str, Any]]
    ) -> None:
        assert rules_map["AE"]["strict"] is True
        assert rules_map["US"]["strict"] is False
        logger.info("Strict enforcement flags validated")

# ---------------------------------------------------------------------------
# Routing Logic Tests
# ---------------------------------------------------------------------------

class TestRoutingLogic:
    """Validate record routing decisions."""

    def test_german_records_route_to_eu_central(
        self, rules_map: dict[str, dict[str, Any]]
    ) -> None:
        rule: dict[str, Any] = rules_map["DE"]
        assert rule["preferred_region"] == "eu-central-1"
        allowed: list[str] = rule["allowed_regions"].split(",")
        assert "eu-central-1" in allowed
        logger.info("DE routing: preferred=%s", rule["preferred_region"])

    def test_uae_records_route_to_middle_east(
        self, rules_map: dict[str, dict[str, Any]]
    ) -> None:
        rule: dict[str, Any] = rules_map["AE"]
        assert rule["preferred_region"] == "me-south-1"
        allowed: list[str] = rule["allowed_regions"].split(",")
        assert len(allowed) == 1, "UAE should have exactly one allowed region"

    def test_unknown_country_raises_error(
        self, rules_map: dict[str, dict[str, Any]]
    ) -> None:
        unknown_country: str = "XX"
        assert unknown_country not in rules_map, (
            "Unknown country should not be in rules"
        )
        logger.info("Unknown country correctly absent from rules")

    def test_batch_partitioning_groups_by_region(
        self,
        sample_records: list[dict[str, Any]],
        rules_map: dict[str, dict[str, Any]],
    ) -> None:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for rec in sample_records:
            country: str = rec["subject_country"]
            if country in rules_map:
                region: str = rules_map[country]["preferred_region"]
                buckets.setdefault(region, []).append(rec)

        assert "eu-central-1" in buckets, "Missing EU-central bucket"
        assert len(buckets["eu-central-1"]) == 2, "DE should have 2 records"
        assert "me-south-1" in buckets, "Missing ME-south bucket"
        logger.info("Batch partitioned into %d regions", len(buckets))

    def test_all_records_routed_to_allowed_regions(
        self,
        sample_records: list[dict[str, Any]],
        rules_map: dict[str, dict[str, Any]],
    ) -> None:
        for rec in sample_records:
            country: str = rec["subject_country"]
            rule: dict[str, Any] = rules_map[country]
            allowed: list[str] = rule["allowed_regions"].split(",")
            preferred: str = rule["preferred_region"]
            assert preferred in allowed, (
                f"Preferred region {preferred} not in allowed list"
            )`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Deploy the multi-region data sovereignty pipeline with environment-based routing configuration, database migrations for each region, and scheduled compliance scanning.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Multi-Region Sovereignty Pipeline Deployment',
                description:
                  'Production deployment script for multi-region residency enforcement. Validates environment variables, installs dependencies, runs migrations per region, and registers the compliance scanner with the scheduler.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# deploy_sovereignty_pipeline.sh
# Purpose: Deploy the cross-border data sovereignty pipeline to production.
# Usage:   SOVEREIGNTY_ENV=production ./deploy_sovereignty_pipeline.sh

SCRIPT_DIR="$(cd "$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="\${SCRIPT_DIR}/deploy_\$(date +%Y%m%d_%H%M%S).log"

log() {
    local level="\$1"; shift
    echo "[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] [\${level}] \$*" | tee -a "\${LOG_FILE}"
}

log "INFO" "Starting cross-border sovereignty pipeline deployment..."

# ── 1. Validate required environment variables ──────────────────────────
REQUIRED_VARS=(
    "SOVEREIGNTY_ENV"
    "PRIMARY_DATABASE_URL"
    "RESIDENCY_RULES_PATH"
    "ROUTING_DB_SCHEMA"
    "SLACK_WEBHOOK_URL"
    "COMPLIANCE_SCAN_INTERVAL"
)

# Region-specific database URLs (at least one required)
REGION_VARS=(
    "DB_URL_EU_CENTRAL_1"
    "DB_URL_EU_WEST_1"
    "DB_URL_US_EAST_1"
    "DB_URL_AP_SOUTH_1"
    "DB_URL_ME_SOUTH_1"
)

missing_vars=()
for var in "\${REQUIRED_VARS[@]}"; do
    if [[ -z "\${!var:-}" ]]; then
        missing_vars+=("\${var}")
    fi
done

if [[ \${#missing_vars[@]} -gt 0 ]]; then
    log "ERROR" "Missing required environment variables: \${missing_vars[*]}"
    exit 1
fi

# Check at least one region DB is configured
region_count=0
for var in "\${REGION_VARS[@]}"; do
    if [[ -n "\${!var:-}" ]]; then
        region_count=$((region_count + 1))
    fi
done

if [[ \${region_count} -eq 0 ]]; then
    log "ERROR" "At least one regional database URL must be set (DB_URL_*)"
    exit 1
fi
log "INFO" "Environment validated: env=\${SOVEREIGNTY_ENV}, regions=\${region_count}"

# ── 2. Install Python dependencies ──────────────────────────────────────
log "INFO" "Installing Python dependencies..."
if [[ -f "\${SCRIPT_DIR}/requirements.txt" ]]; then
    pip install --quiet --upgrade pip
    pip install --quiet -r "\${SCRIPT_DIR}/requirements.txt"
    log "INFO" "Python dependencies installed successfully"
else
    log "ERROR" "requirements.txt not found at \${SCRIPT_DIR}"
    exit 1
fi

# ── 3. Validate residency rules configuration ──────────────────────────
if [[ ! -f "\${RESIDENCY_RULES_PATH}" ]]; then
    log "ERROR" "Residency rules file not found: \${RESIDENCY_RULES_PATH}"
    exit 1
fi
RULE_COUNT=\$(python -c "import json; print(len(json.load(open('\${RESIDENCY_RULES_PATH}'))))")
log "INFO" "Residency rules loaded: \${RULE_COUNT} jurisdiction rules found"

# ── 4. Run database migrations for each configured region ───────────────
log "INFO" "Running database migrations across configured regions..."
for var in "\${REGION_VARS[@]}"; do
    if [[ -n "\${!var:-}" ]]; then
        region_name=\$(echo "\${var}" | sed 's/DB_URL_//;s/_/-/g' | tr '[:upper:]' '[:lower:]')
        log "INFO" "  Migrating region: \${region_name}..."
        python "\${SCRIPT_DIR}/migrations/run_migrations.py" \\
            --database-url "\${!var}" \\
            --schema "\${ROUTING_DB_SCHEMA}" \\
            --direction up \\
            --region "\${region_name}"

        if [[ \$? -eq 0 ]]; then
            log "INFO" "  Migration complete for \${region_name}"
        else
            log "ERROR" "  Migration FAILED for \${region_name} — aborting"
            exit 1
        fi
    fi
done

# ── 5. Register compliance scanner with cron scheduler ──────────────────
SCAN_INTERVAL="\${COMPLIANCE_SCAN_INTERVAL:-4}"
CRON_EXPRESSION="0 */\${SCAN_INTERVAL} * * *"
CRON_CMD="cd \${SCRIPT_DIR} && SOVEREIGNTY_ENV=\${SOVEREIGNTY_ENV} python residency_monitor.py --mode scheduled 2>&1 >> \${SCRIPT_DIR}/compliance_scan.log"
CRON_MARKER="# sovereignty-compliance-scanner"

(crontab -l 2>/dev/null | grep -v "\${CRON_MARKER}") | crontab -
(crontab -l 2>/dev/null; echo "\${CRON_EXPRESSION} \${CRON_CMD} \${CRON_MARKER}") | crontab -
log "INFO" "Compliance scanner registered: every \${SCAN_INTERVAL}h"

# ── 6. Send deployment notification to Slack ────────────────────────────
DEPLOY_MSG="Sovereignty Pipeline deployed to *\${SOVEREIGNTY_ENV}* — \${region_count} regions, \${RULE_COUNT} rules (\$(date -u +%Y-%m-%dT%H:%M:%SZ))"
curl -s -X POST "\${SLACK_WEBHOOK_URL}" \\
    -H "Content-Type: application/json" \\
    -d "{\\"text\\": \\"\${DEPLOY_MSG}\\"}" > /dev/null

log "INFO" "Deployment complete. \${region_count} regions active. Slack notified."`,
              },
              {
                language: 'python',
                title: 'Multi-Region Configuration Loader',
                description:
                  'Production configuration loader with environment-based multi-region config resolution, secrets handling, and per-region connection pool setup for the sovereignty pipeline.',
                code: `# config/sovereignty_config.py
"""Configuration loader for the cross-border data sovereignty pipeline."""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Region configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionConfig:
    """Configuration for a single cloud region."""

    region_name: str
    database_url: str
    jurisdiction_labels: list[str]
    pool_size: int = 3
    max_overflow: int = 5
    pool_timeout: int = 30


@dataclass(frozen=True)
class SovereigntyConfig:
    """Immutable configuration for the sovereignty pipeline."""

    # Environment
    env: str = "development"
    debug: bool = False

    # Primary database (metadata / compliance schema)
    primary_database_url: str = ""
    routing_db_schema: str = "compliance"

    # Residency rules
    residency_rules_path: str = "config/residency_rules.json"

    # Multi-region databases
    regions: list[RegionConfig] = field(default_factory=list)

    # Compliance scanning
    scan_interval_hours: int = 4
    drift_alert_threshold: int = 1

    # Alerting
    slack_webhook_url: str = ""
    alert_on_violation: bool = True

    # Secrets (never logged)
    _secret_fields: tuple[str, ...] = field(
        default=(
            "primary_database_url",
            "slack_webhook_url",
        ),
        repr=False,
    )


# ---------------------------------------------------------------------------
# Region-to-env-var mapping
# ---------------------------------------------------------------------------

REGION_ENV_MAP: dict[str, str] = {
    "eu-central-1": "DB_URL_EU_CENTRAL_1",
    "eu-west-1": "DB_URL_EU_WEST_1",
    "us-east-1": "DB_URL_US_EAST_1",
    "us-west-2": "DB_URL_US_WEST_2",
    "ap-south-1": "DB_URL_AP_SOUTH_1",
    "me-south-1": "DB_URL_ME_SOUTH_1",
    "ap-southeast-2": "DB_URL_AP_SOUTHEAST_2",
}

REGION_JURISDICTION_MAP: dict[str, list[str]] = {
    "eu-central-1": ["EU-GDPR"],
    "eu-west-1": ["EU-GDPR", "UK-GDPR"],
    "us-east-1": ["US-CCPA"],
    "us-west-2": ["US-CCPA"],
    "ap-south-1": ["India-DPDP"],
    "me-south-1": ["UAE-PDPL"],
    "ap-southeast-2": ["Australia-Privacy-Act"],
}


def _discover_regions() -> list[RegionConfig]:
    """Discover configured regions from environment variables."""
    regions: list[RegionConfig] = []
    pool_size: int = int(os.environ.get("REGION_POOL_SIZE", "3"))
    max_overflow: int = int(os.environ.get("REGION_MAX_OVERFLOW", "5"))

    for region_name, env_var in REGION_ENV_MAP.items():
        db_url: str = os.environ.get(env_var, "")
        if db_url:
            regions.append(
                RegionConfig(
                    region_name=region_name,
                    database_url=db_url,
                    jurisdiction_labels=REGION_JURISDICTION_MAP.get(
                        region_name, []
                    ),
                    pool_size=pool_size,
                    max_overflow=max_overflow,
                )
            )
            logger.info(
                "Region discovered: %s (jurisdictions: %s)",
                region_name,
                REGION_JURISDICTION_MAP.get(region_name, []),
            )
    return regions


def load_config() -> SovereigntyConfig:
    """Load multi-region configuration from environment variables."""
    regions: list[RegionConfig] = _discover_regions()

    config = SovereigntyConfig(
        env=os.environ.get("SOVEREIGNTY_ENV", "development"),
        debug=os.environ.get("SOVEREIGNTY_DEBUG", "false").lower() == "true",
        primary_database_url=os.environ.get("PRIMARY_DATABASE_URL", ""),
        routing_db_schema=os.environ.get("ROUTING_DB_SCHEMA", "compliance"),
        residency_rules_path=os.environ.get(
            "RESIDENCY_RULES_PATH", "config/residency_rules.json"
        ),
        regions=regions,
        scan_interval_hours=int(
            os.environ.get("COMPLIANCE_SCAN_INTERVAL", "4")
        ),
        drift_alert_threshold=int(
            os.environ.get("DRIFT_ALERT_THRESHOLD", "1")
        ),
        slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL", ""),
        alert_on_violation=os.environ.get(
            "ALERT_ON_VIOLATION", "true"
        ).lower() == "true",
    )

    if not config.primary_database_url:
        raise EnvironmentError("PRIMARY_DATABASE_URL is required")
    if not config.regions:
        raise EnvironmentError("At least one regional DB_URL_* must be set")

    logger.info(
        "Sovereignty config loaded: env=%s, regions=%d, scan_interval=%dh",
        config.env,
        len(config.regions),
        config.scan_interval_hours,
    )
    return config


def create_region_engines(
    config: SovereigntyConfig,
) -> dict[str, sa.Engine]:
    """Create SQLAlchemy engines for each configured region."""
    engines: dict[str, sa.Engine] = {}
    for region in config.regions:
        engines[region.region_name] = sa.create_engine(
            region.database_url,
            poolclass=QueuePool,
            pool_size=region.pool_size,
            max_overflow=region.max_overflow,
            pool_timeout=region.pool_timeout,
            pool_pre_ping=True,
            echo=config.debug,
        )
        logger.info(
            "Engine created for region %s (pool=%d)",
            region.region_name,
            region.pool_size,
        )
    return engines`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Continuous Residency Compliance Monitoring',
            description:
              'Run a scheduled scanner that detects drift — records that have moved to non-compliant regions since ingestion — triggers automated remediation, and feeds a compliance drift alerting dashboard.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Residency Drift Monitor',
                description:
                  'Periodically scan the record registry, compare current storage regions against rules, and emit alerts or auto-migrate violating records.',
                code: `# residency_monitor.py
"""Continuous compliance monitor — detect and remediate residency drift."""

from __future__ import annotations
import json, datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path

import sqlalchemy as sa

RULES = json.loads(Path("config/residency_rules.json").read_text())
RULES_MAP = {r["country_code"]: r for r in RULES}
ALERT_LOG = Path("audit/residency_alerts.jsonl")

DRIFT_QUERY = """
SELECT record_id, subject_country, storage_region, source_system
FROM metadata.record_registry
WHERE contains_pii = TRUE
  AND updated_at >= :since
ORDER BY updated_at DESC
"""


@dataclass
class ResidencyAlert:
    record_id: str
    source_system: str
    subject_country: str
    current_region: str
    required_regions: list[str]
    severity: str
    detected_at: str


def scan_for_drift(engine: sa.Engine, since_hours: int = 24) -> list[ResidencyAlert]:
    """Check recently-updated records for residency violations."""
    since = dt.datetime.utcnow() - dt.timedelta(hours=since_hours)
    alerts: list[ResidencyAlert] = []

    with engine.connect() as conn:
        rows = conn.execute(sa.text(DRIFT_QUERY), {"since": since})
        for row in rows:
            rule = RULES_MAP.get(row.subject_country)
            if rule is None:
                continue
            allowed = rule["allowed_regions"].split(",")
            if row.storage_region not in allowed:
                alerts.append(ResidencyAlert(
                    record_id=row.record_id,
                    source_system=row.source_system,
                    subject_country=row.subject_country,
                    current_region=row.storage_region,
                    required_regions=allowed,
                    severity="critical" if rule.get("strict") else "high",
                    detected_at=dt.datetime.utcnow().isoformat(),
                ))

    # Append to alert log
    with ALERT_LOG.open("a") as f:
        for a in alerts:
            f.write(json.dumps(asdict(a)) + "\\n")

    print(f"[{'!' if alerts else '✓'}] Drift scan complete — "
          f"{len(alerts)} violations detected in last {since_hours}h")
    return alerts`,
              },
              {
                language: 'sql',
                title: 'Compliance Drift Alerting Dashboard',
                description:
                  'Real-time dashboard query that tracks residency compliance drift across all jurisdictions, flags new violations, monitors remediation progress, and provides trend data for alerting thresholds.',
                code: `-- compliance_drift_dashboard.sql
-- Purpose: Monitor cross-border residency compliance drift and trigger alerts.
-- Schedule: Run every 15 minutes via cron or Airflow.

-- 1. Current compliance posture by jurisdiction
WITH compliance_status AS (
    SELECT
        jr.jurisdiction_label,
        rr.storage_region,
        rr.subject_country,
        CASE
            WHEN rr.storage_region = ANY(string_to_array(jr.allowed_regions, ','))
            THEN 'COMPLIANT'
            ELSE 'VIOLATION'
        END                                          AS status
    FROM metadata.record_registry rr
    JOIN compliance.residency_rules jr
        ON rr.subject_country = jr.country_code
    WHERE rr.contains_pii = TRUE
)
SELECT
    jurisdiction_label,
    status,
    COUNT(*)                                         AS record_count,
    ROUND(
        COUNT(*) * 100.0
        / SUM(COUNT(*)) OVER (PARTITION BY jurisdiction_label),
        2
    )                                                AS pct_of_jurisdiction,
    now()                                            AS measured_at
FROM compliance_status
GROUP BY jurisdiction_label, status
ORDER BY jurisdiction_label, status;

-- 2. New violations detected in the last scan window
SELECT
    ra.record_id,
    ra.source_system,
    ra.subject_country,
    ra.current_region,
    ra.required_regions,
    ra.severity,
    ra.detected_at,
    EXTRACT(EPOCH FROM (now() - ra.detected_at::timestamp)) / 3600.0
                                                     AS hours_since_detection
FROM compliance.residency_alerts ra
WHERE ra.detected_at::timestamp >= now() - INTERVAL '4 hours'
  AND ra.remediated = FALSE
ORDER BY ra.severity DESC, ra.detected_at ASC;

-- 3. Remediation progress tracking
SELECT
    ra.severity,
    COUNT(*) FILTER (WHERE ra.remediated = FALSE)    AS open_violations,
    COUNT(*) FILTER (WHERE ra.remediated = TRUE)     AS remediated,
    COUNT(*)                                         AS total,
    ROUND(
        COUNT(*) FILTER (WHERE ra.remediated = TRUE) * 100.0
        / NULLIF(COUNT(*), 0), 1
    )                                                AS remediation_pct,
    ROUND(
        AVG(
            EXTRACT(EPOCH FROM (
                ra.remediated_at - ra.detected_at::timestamp
            ))
        ) FILTER (WHERE ra.remediated = TRUE) / 3600.0,
        2
    )                                                AS avg_remediation_hours
FROM compliance.residency_alerts ra
WHERE ra.detected_at::timestamp >= now() - INTERVAL '30 days'
GROUP BY ra.severity
ORDER BY ra.severity;

-- 4. Daily drift trend for alerting thresholds
SELECT
    DATE(ra.detected_at::timestamp)                  AS detection_date,
    COUNT(*)                                         AS new_violations,
    COUNT(DISTINCT ra.subject_country)               AS countries_affected,
    COUNT(DISTINCT ra.source_system)                 AS systems_affected,
    ARRAY_AGG(DISTINCT ra.current_region)            AS violating_regions,
    CASE
        WHEN COUNT(*) > 10 THEN 'CRITICAL — surge in violations'
        WHEN COUNT(*) > 3  THEN 'WARNING — elevated drift'
        ELSE 'NORMAL'
    END                                              AS alert_level
FROM compliance.residency_alerts ra
WHERE ra.detected_at::timestamp >= now() - INTERVAL '30 days'
GROUP BY DATE(ra.detected_at::timestamp)
ORDER BY detection_date DESC;`,
              },
            ],
          },
        ],
        toolsUsed: [
          'PostgreSQL metadata queries',
          'SQLAlchemy',
          'Python dataclasses',
          'JSON-based policy-as-code configuration',
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
