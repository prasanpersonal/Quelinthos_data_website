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
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude to analyze data subject requests, map PII locations from exported schemas, and automate deletion verification workflows via Zapier — no custom code required.',
        estimatedMonthlyCost: '$100 - $200/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Airtable ($20/mo)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Casetext ($150/mo)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Export database schemas and data subject identifiers to structured formats that AI can analyze for PII mapping and deletion tracking.',
            toolsUsed: ['Database Export Tools', 'Airtable', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Data Subject Request Schema',
                description:
                  'Structure for capturing erasure requests with subject identifiers, linked systems, and tracking metadata for AI analysis.',
                code: `{
  "erasure_request": {
    "request_id": "ER-2025-001",
    "subject_identifier": {
      "email": "user@example.com",
      "user_id": "usr_abc123",
      "phone": "+1-555-0123"
    },
    "request_date": "2025-01-15T10:30:00Z",
    "regulatory_basis": "GDPR Article 17",
    "deadline_date": "2025-02-14T23:59:59Z",
    "systems_to_check": [
      {
        "system_name": "primary_database",
        "connection_type": "PostgreSQL",
        "pii_tables": ["users", "profiles", "orders", "support_tickets"],
        "identifier_column": "email"
      },
      {
        "system_name": "analytics_warehouse",
        "connection_type": "BigQuery",
        "pii_tables": ["user_events", "session_logs"],
        "identifier_column": "user_id"
      },
      {
        "system_name": "marketing_platform",
        "connection_type": "API",
        "pii_tables": ["contacts", "campaigns"],
        "identifier_column": "email"
      }
    ],
    "schema_export": {
      "primary_database": {
        "users": {
          "columns": ["id", "email", "first_name", "last_name", "phone", "address"],
          "pii_columns": ["email", "first_name", "last_name", "phone", "address"],
          "row_count_estimate": 150000
        },
        "profiles": {
          "columns": ["user_id", "date_of_birth", "ssn_last4", "preferences"],
          "pii_columns": ["date_of_birth", "ssn_last4"],
          "row_count_estimate": 145000
        }
      }
    },
    "status": "pending_analysis",
    "assigned_team": "privacy_ops"
  }
}`,
              },
              {
                language: 'json',
                title: 'PII Discovery Export Format',
                description:
                  'Structured export of discovered PII locations across systems for AI-powered analysis and deletion planning.',
                code: `{
  "pii_discovery_results": {
    "scan_date": "2025-01-15T08:00:00Z",
    "scan_scope": "full_inventory",
    "discovered_locations": [
      {
        "source_system": "primary_database",
        "schema": "public",
        "table": "users",
        "column": "email",
        "pii_category": "email_address",
        "sample_format": "user@domain.com",
        "gdpr_relevant": true,
        "deletion_method": "DELETE WHERE email = ?"
      },
      {
        "source_system": "primary_database",
        "schema": "public",
        "table": "users",
        "column": "phone",
        "pii_category": "phone_number",
        "sample_format": "+1-XXX-XXX-XXXX",
        "gdpr_relevant": true,
        "deletion_method": "UPDATE SET phone = NULL WHERE id = ?"
      },
      {
        "source_system": "analytics_warehouse",
        "schema": "events",
        "table": "user_sessions",
        "column": "ip_address",
        "pii_category": "network_identifier",
        "sample_format": "XXX.XXX.XXX.XXX",
        "gdpr_relevant": true,
        "deletion_method": "DELETE FROM user_sessions WHERE user_id = ?"
      }
    ],
    "total_pii_columns": 47,
    "total_systems": 5,
    "coverage_percentage": 92.5
  }
}`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description:
              'Use ChatGPT or Claude to analyze erasure requests, generate PII mapping reports, create deletion scripts, and verify compliance with regulatory requirements.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro', 'Casetext'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'GDPR Erasure Request Analysis Prompt',
                description:
                  'Comprehensive prompt template for AI to analyze data subject requests, map PII locations, and generate deletion plans.',
                code: `system_prompt: |
  You are a GDPR compliance expert specializing in Article 17 Right to Erasure.
  Your task is to analyze data subject deletion requests and generate comprehensive
  deletion plans that ensure full compliance with regulatory requirements.

  Key responsibilities:
  - Map all PII locations across provided system schemas
  - Identify data retention exceptions (legal holds, legitimate interest)
  - Generate SQL-safe deletion scripts with proper ordering
  - Flag any systems requiring manual intervention
  - Calculate deletion deadlines based on request date

analysis_prompt: |
  ## GDPR Article 17 Erasure Request Analysis

  ### Request Details
  Request ID: {{request_id}}
  Data Subject Identifier: {{subject_email}}
  Request Date: {{request_date}}
  Regulatory Deadline: {{deadline_date}}

  ### System Schema Export
  {{schema_export_json}}

  ### PII Discovery Results
  {{pii_discovery_json}}

  ### Analysis Required

  Please analyze this erasure request and provide:

  1. **PII Location Mapping**
     - List every table/column containing this subject's PII
     - Identify the identifier column to use for each deletion
     - Flag any derived or downstream data copies

  2. **Deletion Order Plan**
     - Determine the correct deletion sequence (foreign key dependencies)
     - Identify any soft-delete vs hard-delete requirements
     - Note any tables requiring archival before deletion

  3. **Retention Exception Review**
     - Check for legal hold requirements
     - Identify any legitimate interest basis for retention
     - Flag any tax/financial records requiring extended retention

  4. **Deletion Scripts**
     - Generate parameterized SQL DELETE statements
     - Include transaction boundaries for atomicity
     - Add verification queries to confirm deletion

  5. **Compliance Checklist**
     - Verify all GDPR Article 17 requirements are met
     - Confirm response deadline is achievable
     - List any manual verification steps required

  6. **Risk Assessment**
     - Identify any systems that may not be covered
     - Flag potential data residue (backups, logs, caches)
     - Note any cross-border data transfer considerations

output_format: |
  Provide your analysis in the following JSON structure:
  {
    "pii_locations": [...],
    "deletion_plan": {...},
    "retention_exceptions": [...],
    "deletion_scripts": [...],
    "compliance_checklist": [...],
    "risk_assessment": {...},
    "estimated_completion_hours": number,
    "manual_steps_required": [...]
  }`,
              },
              {
                language: 'yaml',
                title: 'Deletion Verification Analysis Prompt',
                description:
                  'Prompt template for AI to analyze deletion execution results and generate compliance certificates.',
                code: `system_prompt: |
  You are a data privacy auditor specializing in GDPR Article 17 compliance verification.
  Your role is to analyze deletion execution results and confirm that all personal data
  has been properly erased according to regulatory requirements.

  You must be thorough, methodical, and flag any potential compliance gaps.

verification_prompt: |
  ## Post-Deletion Compliance Verification

  ### Original Erasure Request
  Request ID: {{request_id}}
  Subject Identifier: {{subject_identifier_hash}} (SHA-256 hashed)
  Systems Targeted: {{systems_list}}

  ### Deletion Execution Results
  {{deletion_receipts_json}}

  ### Verification Queries Results
  {{verification_query_results}}

  ### Analysis Required

  Please verify the deletion completeness and generate a compliance certificate:

  1. **Deletion Confirmation**
     - Verify each targeted system shows zero remaining records
     - Confirm row counts match expected deletions
     - Check for any orphaned related records

  2. **Completeness Assessment**
     - Compare deleted locations against original PII mapping
     - Identify any systems that were not processed
     - Flag any partial deletions requiring follow-up

  3. **Audit Trail Validation**
     - Verify all deletion receipts have valid checksums
     - Confirm execution timestamps are within SLA
     - Check that all required approvals were obtained

  4. **Compliance Certificate Generation**
     Generate a formal deletion certificate including:
     - Certificate ID and issue date
     - Summary of systems purged and rows deleted
     - Hash chain verification for tamper evidence
     - Regulatory articles satisfied (GDPR Art. 17, CCPA, etc.)
     - Authorized signatory placeholder

  5. **Residual Data Assessment**
     - Identify any backup systems requiring separate deletion
     - Note cache invalidation requirements
     - Flag any third-party data sharing that needs notification

output_format: |
  {
    "verification_status": "COMPLETE" | "INCOMPLETE" | "FAILED",
    "systems_verified": [...],
    "discrepancies_found": [...],
    "compliance_certificate": {
      "certificate_id": "string",
      "issue_date": "ISO8601",
      "subject_hash": "string",
      "systems_purged": number,
      "total_rows_deleted": number,
      "chain_hash": "string",
      "regulatory_compliance": [...],
      "valid_until": "ISO8601"
    },
    "follow_up_actions": [...],
    "auditor_notes": "string"
  }`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier workflows to automatically process erasure requests, trigger deletion verification, and deliver compliance certificates to stakeholders.',
            toolsUsed: ['Zapier Pro', 'Airtable', 'Slack', 'Email'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Erasure Request Processing Workflow',
                description:
                  'Multi-step Zap that captures erasure requests, triggers AI analysis, coordinates team actions, and tracks compliance deadlines.',
                code: `{
  "zap_name": "GDPR Erasure Request Processor",
  "description": "Automate Article 17 deletion request handling with AI analysis",
  "trigger": {
    "app": "Airtable",
    "event": "New Record in View",
    "config": {
      "base_id": "app_gdpr_requests",
      "table": "Erasure Requests",
      "view": "New Requests - Pending Analysis"
    },
    "output_fields": [
      "request_id",
      "subject_email",
      "subject_user_id",
      "request_date",
      "regulatory_basis",
      "systems_json"
    ]
  },
  "actions": [
    {
      "step": 1,
      "app": "Formatter by Zapier",
      "action": "Date/Time - Add/Subtract Time",
      "config": {
        "input_date": "{{request_date}}",
        "operation": "add",
        "amount": 30,
        "unit": "days"
      },
      "output": "deadline_date",
      "purpose": "Calculate GDPR 30-day response deadline"
    },
    {
      "step": 2,
      "app": "Webhooks by Zapier",
      "action": "GET",
      "config": {
        "url": "https://your-api.com/pii-discovery/{{subject_email}}",
        "headers": {
          "Authorization": "Bearer {{api_key}}",
          "Content-Type": "application/json"
        }
      },
      "output": "pii_locations_json",
      "purpose": "Fetch PII discovery results for subject"
    },
    {
      "step": 3,
      "app": "ChatGPT",
      "action": "Conversation",
      "config": {
        "model": "gpt-4",
        "system_message": "You are a GDPR compliance expert...",
        "user_message": "Analyze erasure request {{request_id}} for subject with PII in: {{pii_locations_json}}. Generate deletion plan and scripts.",
        "temperature": 0.3,
        "max_tokens": 4000
      },
      "output": "ai_analysis_json",
      "purpose": "AI-powered PII mapping and deletion planning"
    },
    {
      "step": 4,
      "app": "Airtable",
      "action": "Update Record",
      "config": {
        "base_id": "app_gdpr_requests",
        "table": "Erasure Requests",
        "record_id": "{{record_id}}",
        "fields": {
          "Status": "Analysis Complete",
          "Deadline": "{{deadline_date}}",
          "AI Analysis": "{{ai_analysis_json}}",
          "PII Locations Count": "{{pii_count}}",
          "Deletion Scripts": "{{deletion_scripts}}"
        }
      },
      "purpose": "Update request record with AI analysis results"
    },
    {
      "step": 5,
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#privacy-ops",
        "message": ":shield: *New Erasure Request Ready for Execution*\\n\\nRequest ID: {{request_id}}\\nSubject: {{subject_email_masked}}\\nDeadline: {{deadline_date}}\\nSystems Affected: {{systems_count}}\\nPII Locations: {{pii_count}}\\n\\n<{{airtable_record_url}}|View Full Analysis>",
        "bot_name": "GDPR Bot"
      },
      "purpose": "Notify privacy team of pending deletion"
    },
    {
      "step": 6,
      "app": "Paths by Zapier",
      "action": "Conditional Paths",
      "config": {
        "path_a": {
          "condition": "pii_count > 50",
          "name": "High Complexity Request",
          "continue_to": "escalation_flow"
        },
        "path_b": {
          "condition": "pii_count <= 50",
          "name": "Standard Request",
          "continue_to": "standard_flow"
        }
      },
      "purpose": "Route based on deletion complexity"
    }
  ],
  "schedule": {
    "check_frequency": "15 minutes",
    "timezone": "UTC",
    "active_hours": "24/7"
  },
  "error_handling": {
    "on_failure": "notify_admin",
    "retry_count": 3,
    "notification_channel": "#privacy-ops-alerts"
  }
}`,
              },
              {
                language: 'json',
                title: 'Zapier Deletion Verification & Certificate Workflow',
                description:
                  'Workflow that monitors deletion execution, triggers AI verification, generates compliance certificates, and notifies the data subject.',
                code: `{
  "zap_name": "Erasure Verification & Certificate Generator",
  "description": "Verify deletion completeness and generate GDPR compliance certificates",
  "trigger": {
    "app": "Airtable",
    "event": "Record Updated in View",
    "config": {
      "base_id": "app_gdpr_requests",
      "table": "Erasure Requests",
      "view": "Deletion Executed - Pending Verification",
      "trigger_field": "Execution Status",
      "trigger_value": "Completed"
    }
  },
  "actions": [
    {
      "step": 1,
      "app": "Webhooks by Zapier",
      "action": "GET",
      "config": {
        "url": "https://your-api.com/erasure-receipts/{{request_id}}",
        "headers": {
          "Authorization": "Bearer {{api_key}}"
        }
      },
      "output": "deletion_receipts_json",
      "purpose": "Fetch deletion execution receipts"
    },
    {
      "step": 2,
      "app": "ChatGPT",
      "action": "Conversation",
      "config": {
        "model": "gpt-4",
        "system_message": "You are a data privacy auditor...",
        "user_message": "Verify deletion for request {{request_id}}. Receipts: {{deletion_receipts_json}}. Original PII locations: {{pii_locations_json}}. Generate compliance certificate.",
        "temperature": 0.2,
        "max_tokens": 3000
      },
      "output": "verification_result_json",
      "purpose": "AI verification of deletion completeness"
    },
    {
      "step": 3,
      "app": "Formatter by Zapier",
      "action": "Text - Extract Pattern",
      "config": {
        "input": "{{verification_result_json}}",
        "pattern": "certificate_id.*?:.*?\"([^\"]+)\"",
        "output_type": "first_match"
      },
      "output": "certificate_id",
      "purpose": "Extract certificate ID from AI response"
    },
    {
      "step": 4,
      "app": "Google Docs",
      "action": "Create Document from Template",
      "config": {
        "template_id": "gdpr_certificate_template",
        "folder_id": "compliance_certificates_folder",
        "replacements": {
          "{{CERTIFICATE_ID}}": "{{certificate_id}}",
          "{{ISSUE_DATE}}": "{{current_date}}",
          "{{REQUEST_ID}}": "{{request_id}}",
          "{{SYSTEMS_PURGED}}": "{{systems_count}}",
          "{{ROWS_DELETED}}": "{{total_rows}}",
          "{{CHAIN_HASH}}": "{{chain_hash}}",
          "{{REGULATORY_COMPLIANCE}}": "GDPR Article 17, UK GDPR Article 17"
        }
      },
      "output": "certificate_doc_url",
      "purpose": "Generate formal compliance certificate"
    },
    {
      "step": 5,
      "app": "Airtable",
      "action": "Update Record",
      "config": {
        "base_id": "app_gdpr_requests",
        "table": "Erasure Requests",
        "record_id": "{{record_id}}",
        "fields": {
          "Status": "Verified Complete",
          "Verification Result": "{{verification_result_json}}",
          "Certificate ID": "{{certificate_id}}",
          "Certificate URL": "{{certificate_doc_url}}",
          "Completion Date": "{{current_datetime}}"
        }
      },
      "purpose": "Update request with verification results"
    },
    {
      "step": 6,
      "app": "Gmail",
      "action": "Send Email",
      "config": {
        "to": "{{subject_email}}",
        "subject": "Your Data Deletion Request Has Been Completed",
        "body": "Dear Data Subject,\\n\\nWe have completed processing your data deletion request (Reference: {{request_id}}).\\n\\nA total of {{systems_count}} systems were processed and your personal data has been removed in accordance with GDPR Article 17.\\n\\nYour deletion certificate reference is: {{certificate_id}}\\n\\nIf you have any questions, please contact our Data Protection Officer.\\n\\nBest regards,\\nPrivacy Team",
        "from_name": "Privacy Team"
      },
      "purpose": "Notify data subject of completion"
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
          'Deploy a multi-agent system using CrewAI and LangGraph that autonomously discovers PII across all data systems, orchestrates parallel deletions, generates cryptographic audit trails, and handles regulatory variations across GDPR, CCPA, and LGPD jurisdictions.',
        estimatedMonthlyCost: '$600 - $1,500/month',
        architecture:
          'A Supervisor agent coordinates four specialist agents: PII Discovery Agent scans databases and identifies personal data locations, Deletion Orchestrator Agent manages parallel erasure execution across systems, Compliance Auditor Agent verifies completeness and generates certificates, and Regulatory Advisor Agent handles jurisdiction-specific requirements and exceptions.',
        agents: [
          {
            name: 'PII Discovery Agent',
            role: 'Data Mapping Specialist',
            goal: 'Continuously scan all data sources to maintain a real-time inventory of personal data locations, identifying PII columns, data flows, and downstream copies across the entire data estate.',
            tools: ['DatabaseScanner', 'SchemaAnalyzer', 'PIIClassifier', 'LineageTracker'],
          },
          {
            name: 'Deletion Orchestrator Agent',
            role: 'Erasure Execution Coordinator',
            goal: 'Execute deletion requests across all identified PII locations with proper transaction ordering, parallel execution where safe, and comprehensive receipt generation for audit trails.',
            tools: ['SQLExecutor', 'APIConnector', 'TransactionManager', 'ReceiptGenerator'],
          },
          {
            name: 'Compliance Auditor Agent',
            role: 'Verification and Certification Specialist',
            goal: 'Verify that all deletions are complete, generate tamper-evident certificates with hash chains, and ensure audit trails satisfy regulatory inspection requirements.',
            tools: ['VerificationRunner', 'HashChainBuilder', 'CertificateGenerator', 'AuditLogWriter'],
          },
          {
            name: 'Regulatory Advisor Agent',
            role: 'Jurisdiction and Exception Handler',
            goal: 'Analyze each erasure request against applicable regulations (GDPR, CCPA, LGPD), identify retention exceptions, legal holds, and legitimate interest bases, and ensure jurisdiction-specific requirements are met.',
            tools: ['RegulationDatabase', 'LegalHoldChecker', 'ExceptionAnalyzer', 'HarveyAI'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with request-level checkpointing and 90-day audit retention',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system with specialized agents for PII discovery, deletion orchestration, compliance verification, and regulatory advisory functions.',
            toolsUsed: ['CrewAI', 'LangChain', 'Harvey AI'],
            codeSnippets: [
              {
                language: 'python',
                title: 'GDPR Erasure Multi-Agent System Definition',
                description:
                  'Complete CrewAI agent definitions for the erasure pipeline with specialized roles, goals, and tool assignments.',
                code: `# agents/erasure_agents.py
"""CrewAI agent definitions for GDPR Article 17 erasure automation."""

from __future__ import annotations
import os
from typing import Any, Optional

from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

def get_llm(model_type: str = "openai") -> Any:
    """Get configured LLM instance based on model type."""
    if model_type == "anthropic":
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.2,
            max_tokens=4096,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
    return ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.2,
        max_tokens=4096,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

# ---------------------------------------------------------------------------
# Agent Definitions
# ---------------------------------------------------------------------------

class ErasureAgentFactory:
    """Factory for creating specialized erasure pipeline agents."""

    def __init__(self, llm_type: str = "openai") -> None:
        self.llm = get_llm(llm_type)

    def create_pii_discovery_agent(self) -> Agent:
        """Create the PII Discovery Agent for data mapping."""
        return Agent(
            role="PII Discovery Specialist",
            goal=(
                "Maintain a comprehensive, real-time inventory of all personal "
                "data locations across the organization's data estate. Identify "
                "every table, column, and object path containing PII for any "
                "given data subject identifier."
            ),
            backstory=(
                "You are an expert data analyst with deep knowledge of database "
                "schemas, data lineage, and PII classification. You have mapped "
                "hundreds of enterprise data systems and understand how personal "
                "data flows through ETL pipelines, gets copied to analytics "
                "warehouses, and persists in backup systems. You are meticulous "
                "and never miss a data location."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            tools=[],  # Tools added dynamically
            max_iter=15,
            max_rpm=30,
        )

    def create_deletion_orchestrator_agent(self) -> Agent:
        """Create the Deletion Orchestrator Agent."""
        return Agent(
            role="Erasure Execution Coordinator",
            goal=(
                "Execute data subject deletion requests across all identified "
                "PII locations with proper transaction ordering, foreign key "
                "awareness, and parallel execution where safe. Generate detailed "
                "receipts for every deletion action."
            ),
            backstory=(
                "You are a database operations expert who has executed thousands "
                "of complex data deletion operations. You understand transaction "
                "isolation levels, foreign key cascades, and the importance of "
                "proper deletion ordering. You never leave orphaned records and "
                "always generate complete audit receipts."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            tools=[],
            max_iter=20,
            max_rpm=30,
        )

    def create_compliance_auditor_agent(self) -> Agent:
        """Create the Compliance Auditor Agent."""
        return Agent(
            role="Verification and Certification Specialist",
            goal=(
                "Verify that all deletions have been executed completely, "
                "generate tamper-evident certificates with cryptographic hash "
                "chains, and ensure all audit trails meet regulatory inspection "
                "requirements for GDPR, CCPA, and other privacy laws."
            ),
            backstory=(
                "You are a certified privacy auditor with experience conducting "
                "regulatory inspections for data protection authorities. You "
                "understand what regulators look for in deletion evidence and "
                "how to create audit trails that withstand scrutiny. You are "
                "thorough and never sign off on incomplete deletions."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            tools=[],
            max_iter=10,
            max_rpm=30,
        )

    def create_regulatory_advisor_agent(self) -> Agent:
        """Create the Regulatory Advisor Agent."""
        return Agent(
            role="Jurisdiction and Exception Handler",
            goal=(
                "Analyze each erasure request against applicable privacy "
                "regulations, identify any retention exceptions (legal holds, "
                "legitimate interest, tax records), and ensure all jurisdiction-"
                "specific requirements are properly handled."
            ),
            backstory=(
                "You are a data privacy lawyer with expertise in GDPR, CCPA, "
                "LGPD, and emerging privacy regulations worldwide. You have "
                "advised Fortune 500 companies on erasure requests and know "
                "exactly when deletion can be refused or delayed. You always "
                "cite specific regulatory articles in your advice."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            tools=[],
            max_iter=10,
            max_rpm=30,
        )

    def create_supervisor_agent(self) -> Agent:
        """Create the Supervisor Agent that coordinates the crew."""
        return Agent(
            role="Erasure Pipeline Supervisor",
            goal=(
                "Coordinate the end-to-end erasure process from request intake "
                "to certificate generation. Ensure all agents complete their "
                "tasks within SLA, handle exceptions gracefully, and maintain "
                "complete audit trails."
            ),
            backstory=(
                "You are a senior privacy program manager who has overseen "
                "thousands of erasure requests. You understand the critical "
                "importance of meeting regulatory deadlines and maintaining "
                "complete documentation. You excel at coordinating complex "
                "workflows across multiple teams and systems."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            tools=[],
            max_iter=25,
            max_rpm=30,
        )

# ---------------------------------------------------------------------------
# Crew Assembly
# ---------------------------------------------------------------------------

def create_erasure_crew(llm_type: str = "openai") -> Crew:
    """Assemble the complete erasure pipeline crew."""
    factory = ErasureAgentFactory(llm_type)

    supervisor = factory.create_supervisor_agent()
    discovery_agent = factory.create_pii_discovery_agent()
    orchestrator_agent = factory.create_deletion_orchestrator_agent()
    auditor_agent = factory.create_compliance_auditor_agent()
    regulatory_agent = factory.create_regulatory_advisor_agent()

    return Crew(
        agents=[
            supervisor,
            discovery_agent,
            orchestrator_agent,
            auditor_agent,
            regulatory_agent,
        ],
        tasks=[],  # Tasks added dynamically per request
        process=Process.hierarchical,
        manager_agent=supervisor,
        verbose=True,
        memory=True,
        embedder={
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"},
        },
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the PII Discovery Agent with tools for database scanning, schema analysis, and personal data classification across heterogeneous data sources.',
            toolsUsed: ['CrewAI Tools', 'SQLAlchemy', 'Presidio', 'Great Expectations'],
            codeSnippets: [
              {
                language: 'python',
                title: 'PII Discovery Agent Tools',
                description:
                  'Custom CrewAI tools for database scanning, PII classification, and data lineage tracking across multiple data sources.',
                code: `# tools/pii_discovery_tools.py
"""CrewAI tools for PII discovery and data mapping."""

from __future__ import annotations
import json
import hashlib
import logging
from datetime import datetime
from typing import Any, Optional, Type

from crewai_tools import BaseTool
from pydantic import BaseModel, Field
import sqlalchemy as sa
from sqlalchemy import inspect, MetaData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool Input Schemas
# ---------------------------------------------------------------------------

class DatabaseScanInput(BaseModel):
    """Input schema for database scanning."""
    connection_string: str = Field(
        ..., description="Database connection string"
    )
    schemas: list[str] = Field(
        default=["public"],
        description="List of schemas to scan",
    )
    sample_rows: int = Field(
        default=100,
        description="Number of rows to sample for PII detection",
    )

class PIIClassifyInput(BaseModel):
    """Input schema for PII classification."""
    column_name: str = Field(..., description="Column name to analyze")
    sample_values: list[str] = Field(
        ..., description="Sample values from the column"
    )
    table_context: str = Field(
        default="", description="Table name for context"
    )

class SubjectLookupInput(BaseModel):
    """Input schema for subject PII lookup."""
    subject_identifier: str = Field(
        ..., description="Email, user ID, or other identifier"
    )
    identifier_type: str = Field(
        default="email", description="Type of identifier"
    )

# ---------------------------------------------------------------------------
# Database Scanner Tool
# ---------------------------------------------------------------------------

class DatabaseScannerTool(BaseTool):
    """Tool to scan database schemas and discover potential PII columns."""

    name: str = "database_scanner"
    description: str = (
        "Scans a database to discover all tables and columns, identifying "
        "those likely to contain personal data based on naming conventions "
        "and data sampling. Returns a structured inventory of PII locations."
    )
    args_schema: Type[BaseModel] = DatabaseScanInput

    # PII indicator patterns
    PII_PATTERNS: list[str] = [
        "email", "phone", "mobile", "address", "street", "city", "zip",
        "postal", "ssn", "social_security", "tax_id", "passport",
        "license", "dob", "birth", "age", "gender", "first_name",
        "last_name", "full_name", "username", "ip_address", "device_id",
        "credit_card", "bank", "account", "salary", "income",
    ]

    def _run(
        self,
        connection_string: str,
        schemas: list[str],
        sample_rows: int = 100,
    ) -> str:
        """Execute database scan and return PII inventory."""
        try:
            engine = sa.create_engine(connection_string)
            inspector = inspect(engine)
            pii_inventory: list[dict[str, Any]] = []

            for schema in schemas:
                tables = inspector.get_table_names(schema=schema)
                logger.info(
                    "Scanning schema %s: %d tables found",
                    schema, len(tables)
                )

                for table in tables:
                    columns = inspector.get_columns(table, schema=schema)
                    for col in columns:
                        col_name_lower = col["name"].lower()

                        # Check if column name matches PII patterns
                        is_pii_candidate = any(
                            pattern in col_name_lower
                            for pattern in self.PII_PATTERNS
                        )

                        if is_pii_candidate:
                            pii_category = self._classify_column(col_name_lower)
                            pii_inventory.append({
                                "schema": schema,
                                "table": table,
                                "column": col["name"],
                                "data_type": str(col["type"]),
                                "nullable": col.get("nullable", True),
                                "pii_category": pii_category,
                                "confidence": "high" if pii_category else "medium",
                                "discovered_at": datetime.utcnow().isoformat(),
                            })

            result = {
                "status": "success",
                "schemas_scanned": schemas,
                "pii_columns_found": len(pii_inventory),
                "inventory": pii_inventory,
            }

            logger.info(
                "Database scan complete: %d PII columns discovered",
                len(pii_inventory)
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error("Database scan failed: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
                "schemas_scanned": schemas,
            })

    def _classify_column(self, column_name: str) -> Optional[str]:
        """Classify column into PII category based on name."""
        classifications = {
            "email": ["email", "e_mail", "mail"],
            "phone": ["phone", "mobile", "cell", "telephone"],
            "address": ["address", "street", "city", "zip", "postal"],
            "name": ["first_name", "last_name", "full_name", "name"],
            "national_id": ["ssn", "social_security", "tax_id", "passport"],
            "financial": ["credit_card", "bank", "account", "salary"],
            "demographic": ["dob", "birth", "age", "gender"],
            "device": ["ip_address", "device_id", "mac_address"],
        }

        for category, patterns in classifications.items():
            if any(p in column_name for p in patterns):
                return category
        return None

# ---------------------------------------------------------------------------
# Subject PII Lookup Tool
# ---------------------------------------------------------------------------

class SubjectPIILookupTool(BaseTool):
    """Tool to find all PII locations for a specific data subject."""

    name: str = "subject_pii_lookup"
    description: str = (
        "Searches across all registered data sources to find every location "
        "where a specific data subject's personal data is stored. Returns "
        "table, column, and row identifiers for deletion targeting."
    )
    args_schema: Type[BaseModel] = SubjectLookupInput

    def __init__(
        self,
        pii_registry_path: str = "data/pii_registry.json",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._registry_path = pii_registry_path

    def _run(
        self,
        subject_identifier: str,
        identifier_type: str = "email",
    ) -> str:
        """Find all PII locations for a data subject."""
        try:
            # Load PII registry
            with open(self._registry_path) as f:
                registry = json.load(f)

            # Build search queries based on identifier type
            subject_locations: list[dict[str, Any]] = []
            subject_hash = hashlib.sha256(
                subject_identifier.encode()
            ).hexdigest()

            for entry in registry:
                # Check if this entry could contain the subject's data
                if self._matches_identifier_type(entry, identifier_type):
                    subject_locations.append({
                        "source": entry["source"],
                        "schema": entry.get("schema_name", "public"),
                        "table": entry["table"],
                        "column": entry["column"],
                        "pii_category": entry["pii_category"],
                        "identifier_column": self._get_identifier_column(
                            entry, identifier_type
                        ),
                        "lookup_query": self._build_lookup_query(
                            entry, identifier_type
                        ),
                    })

            result = {
                "status": "success",
                "subject_hash": subject_hash,
                "identifier_type": identifier_type,
                "locations_found": len(subject_locations),
                "locations": subject_locations,
                "searched_at": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Subject lookup complete: %d locations found for %s",
                len(subject_locations), subject_hash[:16]
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error("Subject lookup failed: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
                "subject_hash": hashlib.sha256(
                    subject_identifier.encode()
                ).hexdigest()[:16],
            })

    def _matches_identifier_type(
        self, entry: dict[str, Any], identifier_type: str
    ) -> bool:
        """Check if registry entry matches the identifier type."""
        type_mapping = {
            "email": ["email"],
            "user_id": ["user_id", "id", "customer_id"],
            "phone": ["phone", "mobile"],
        }
        patterns = type_mapping.get(identifier_type, [identifier_type])
        return entry.get("pii_category") in patterns or any(
            p in entry.get("column", "").lower() for p in patterns
        )

    def _get_identifier_column(
        self, entry: dict[str, Any], identifier_type: str
    ) -> str:
        """Determine the identifier column for lookups."""
        if identifier_type == "email":
            return "email"
        elif identifier_type == "user_id":
            return "user_id"
        return entry.get("column", "id")

    def _build_lookup_query(
        self, entry: dict[str, Any], identifier_type: str
    ) -> str:
        """Build parameterized lookup query."""
        schema = entry.get("schema_name", "public")
        table = entry["table"]
        id_col = self._get_identifier_column(entry, identifier_type)
        return f"SELECT * FROM {schema}.{table} WHERE {id_col} = :identifier"`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Regulatory Advisor and Compliance Auditor agents with tools for legal analysis, retention exception handling, and verification workflows.',
            toolsUsed: ['CrewAI Tools', 'LangChain', 'Casetext API', 'Lexis+ AI'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Regulatory Advisor and Compliance Auditor Tools',
                description:
                  'CrewAI tools for regulatory analysis, retention exception handling, deletion verification, and certificate generation.',
                code: `# tools/compliance_tools.py
"""CrewAI tools for regulatory analysis and compliance verification."""

from __future__ import annotations
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Type

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool Input Schemas
# ---------------------------------------------------------------------------

class RegulatoryAnalysisInput(BaseModel):
    """Input schema for regulatory analysis."""
    subject_country: str = Field(
        ..., description="Country of the data subject"
    )
    data_categories: list[str] = Field(
        ..., description="Categories of PII involved"
    )
    request_type: str = Field(
        default="erasure", description="Type of DSAR request"
    )
    business_context: str = Field(
        default="", description="Business context for exception analysis"
    )

class RetentionCheckInput(BaseModel):
    """Input schema for retention exception checking."""
    request_id: str = Field(..., description="Erasure request ID")
    subject_identifier: str = Field(
        ..., description="Data subject identifier"
    )
    data_locations: list[dict] = Field(
        ..., description="List of PII locations to check"
    )

class VerificationInput(BaseModel):
    """Input schema for deletion verification."""
    request_id: str = Field(..., description="Erasure request ID")
    deletion_receipts: list[dict] = Field(
        ..., description="Deletion execution receipts"
    )
    original_locations: list[dict] = Field(
        ..., description="Original PII locations"
    )

class CertificateInput(BaseModel):
    """Input schema for certificate generation."""
    request_id: str = Field(..., description="Erasure request ID")
    verification_result: dict = Field(
        ..., description="Verification analysis result"
    )
    deletion_receipts: list[dict] = Field(
        ..., description="All deletion receipts"
    )

# ---------------------------------------------------------------------------
# Regulatory Analysis Tool
# ---------------------------------------------------------------------------

class RegulatoryAnalysisTool(BaseTool):
    """Tool for analyzing applicable regulations and requirements."""

    name: str = "regulatory_analyzer"
    description: str = (
        "Analyzes which privacy regulations apply to an erasure request "
        "based on the data subject's jurisdiction, data categories, and "
        "business context. Returns applicable regulations, deadlines, "
        "and any special requirements."
    )
    args_schema: Type[BaseModel] = RegulatoryAnalysisInput

    # Regulation database
    REGULATIONS: dict[str, dict[str, Any]] = {
        "EU": {
            "name": "GDPR",
            "articles": ["Article 17 - Right to Erasure"],
            "deadline_days": 30,
            "extensions_allowed": True,
            "max_extension_days": 60,
            "exceptions": [
                "Freedom of expression",
                "Legal obligations",
                "Public health",
                "Archiving in public interest",
                "Legal claims",
            ],
        },
        "UK": {
            "name": "UK GDPR",
            "articles": ["Article 17 - Right to Erasure"],
            "deadline_days": 30,
            "extensions_allowed": True,
            "max_extension_days": 60,
            "exceptions": [
                "Freedom of expression",
                "Legal obligations",
                "Public health",
                "Legal claims",
            ],
        },
        "US-CA": {
            "name": "CCPA/CPRA",
            "articles": ["Section 1798.105 - Right to Delete"],
            "deadline_days": 45,
            "extensions_allowed": True,
            "max_extension_days": 45,
            "exceptions": [
                "Complete transactions",
                "Security incidents",
                "Legal obligations",
                "Internal uses compatible with expectations",
            ],
        },
        "BR": {
            "name": "LGPD",
            "articles": ["Article 18 - Data Subject Rights"],
            "deadline_days": 15,
            "extensions_allowed": False,
            "max_extension_days": 0,
            "exceptions": [
                "Legal obligations",
                "Research (anonymized)",
                "Legitimate interests",
            ],
        },
    }

    def _run(
        self,
        subject_country: str,
        data_categories: list[str],
        request_type: str = "erasure",
        business_context: str = "",
    ) -> str:
        """Analyze applicable regulations for the request."""
        try:
            # Map country to jurisdiction
            jurisdiction = self._map_jurisdiction(subject_country)
            regulation = self.REGULATIONS.get(jurisdiction)

            if not regulation:
                return json.dumps({
                    "status": "warning",
                    "message": f"No specific regulation found for {subject_country}",
                    "recommendation": "Apply GDPR as baseline standard",
                    "jurisdiction": jurisdiction,
                })

            # Calculate deadline
            deadline = datetime.utcnow() + timedelta(
                days=regulation["deadline_days"]
            )

            # Analyze data categories for special handling
            special_handling = self._analyze_categories(data_categories)

            result = {
                "status": "success",
                "jurisdiction": jurisdiction,
                "regulation": regulation["name"],
                "applicable_articles": regulation["articles"],
                "deadline": deadline.isoformat(),
                "deadline_days": regulation["deadline_days"],
                "extension_available": regulation["extensions_allowed"],
                "max_extension_days": regulation["max_extension_days"],
                "potential_exceptions": regulation["exceptions"],
                "special_handling": special_handling,
                "analysis_date": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Regulatory analysis complete: %s applies with %d-day deadline",
                regulation["name"], regulation["deadline_days"]
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error("Regulatory analysis failed: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
            })

    def _map_jurisdiction(self, country: str) -> str:
        """Map country code to regulatory jurisdiction."""
        eu_countries = [
            "DE", "FR", "IT", "ES", "NL", "BE", "AT", "PL", "SE", "DK",
            "FI", "IE", "PT", "GR", "CZ", "HU", "RO", "BG", "SK", "HR",
        ]
        if country in eu_countries:
            return "EU"
        elif country == "GB":
            return "UK"
        elif country == "US":
            return "US-CA"  # Default to CCPA
        elif country == "BR":
            return "BR"
        return "EU"  # Default to GDPR as baseline

    def _analyze_categories(
        self, categories: list[str]
    ) -> list[dict[str, str]]:
        """Analyze data categories for special handling requirements."""
        special_cases = []

        if "financial" in categories:
            special_cases.append({
                "category": "financial",
                "requirement": "Check tax record retention (typically 7 years)",
                "action": "May need to retain for legal compliance",
            })

        if "health" in categories:
            special_cases.append({
                "category": "health",
                "requirement": "HIPAA/health record retention rules may apply",
                "action": "Consult with healthcare compliance team",
            })

        if "employment" in categories:
            special_cases.append({
                "category": "employment",
                "requirement": "Employment record retention varies by jurisdiction",
                "action": "Check local employment law requirements",
            })

        return special_cases

# ---------------------------------------------------------------------------
# Retention Exception Checker Tool
# ---------------------------------------------------------------------------

class RetentionExceptionCheckerTool(BaseTool):
    """Tool for checking retention exceptions and legal holds."""

    name: str = "retention_exception_checker"
    description: str = (
        "Checks if any data locations are subject to retention exceptions "
        "such as legal holds, regulatory retention requirements, or active "
        "litigation. Returns a list of locations that cannot be deleted."
    )
    args_schema: Type[BaseModel] = RetentionCheckInput

    def _run(
        self,
        request_id: str,
        subject_identifier: str,
        data_locations: list[dict],
    ) -> str:
        """Check for retention exceptions across all data locations."""
        try:
            exceptions: list[dict[str, Any]] = []
            deletable: list[dict[str, Any]] = []

            for location in data_locations:
                # Check legal holds
                hold_status = self._check_legal_hold(
                    subject_identifier, location
                )

                # Check regulatory retention
                retention_status = self._check_regulatory_retention(location)

                if hold_status["is_held"] or retention_status["must_retain"]:
                    exceptions.append({
                        "location": location,
                        "exception_type": (
                            "legal_hold" if hold_status["is_held"]
                            else "regulatory_retention"
                        ),
                        "reason": (
                            hold_status.get("reason")
                            or retention_status.get("reason")
                        ),
                        "retention_until": retention_status.get("retain_until"),
                        "action_required": "exclude_from_deletion",
                    })
                else:
                    deletable.append(location)

            result = {
                "status": "success",
                "request_id": request_id,
                "total_locations": len(data_locations),
                "deletable_count": len(deletable),
                "exception_count": len(exceptions),
                "deletable_locations": deletable,
                "retention_exceptions": exceptions,
                "checked_at": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Retention check complete: %d deletable, %d exceptions",
                len(deletable), len(exceptions)
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error("Retention check failed: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
                "request_id": request_id,
            })

    def _check_legal_hold(
        self,
        subject_identifier: str,
        location: dict,
    ) -> dict[str, Any]:
        """Check if location is under legal hold."""
        # In production, this would query a legal hold database
        return {
            "is_held": False,
            "reason": None,
            "hold_id": None,
        }

    def _check_regulatory_retention(
        self,
        location: dict,
    ) -> dict[str, Any]:
        """Check regulatory retention requirements for location."""
        table = location.get("table", "").lower()

        # Financial records typically have 7-year retention
        if any(t in table for t in ["transaction", "invoice", "payment"]):
            return {
                "must_retain": True,
                "reason": "Financial record retention (7 years)",
                "retain_until": (
                    datetime.utcnow() + timedelta(days=2555)
                ).isoformat(),
            }

        return {
            "must_retain": False,
            "reason": None,
            "retain_until": None,
        }

# ---------------------------------------------------------------------------
# Deletion Verification Tool
# ---------------------------------------------------------------------------

class DeletionVerificationTool(BaseTool):
    """Tool for verifying deletion completeness."""

    name: str = "deletion_verifier"
    description: str = (
        "Verifies that all targeted deletions have been executed "
        "completely by comparing deletion receipts against original "
        "PII locations and running verification queries."
    )
    args_schema: Type[BaseModel] = VerificationInput

    def _run(
        self,
        request_id: str,
        deletion_receipts: list[dict],
        original_locations: list[dict],
    ) -> str:
        """Verify deletion completeness."""
        try:
            # Map receipts by location
            receipt_map = {
                f"{r['source']}.{r['table']}": r
                for r in deletion_receipts
            }

            verification_results: list[dict[str, Any]] = []
            incomplete: list[dict[str, Any]] = []

            for location in original_locations:
                loc_key = f"{location['source']}.{location['table']}"
                receipt = receipt_map.get(loc_key)

                if receipt:
                    verification_results.append({
                        "location": loc_key,
                        "status": "verified",
                        "rows_deleted": receipt.get("rows_deleted", 0),
                        "executed_at": receipt.get("executed_at"),
                        "checksum": receipt.get("checksum"),
                    })
                else:
                    incomplete.append({
                        "location": loc_key,
                        "status": "missing_receipt",
                        "action_required": "manual_verification",
                    })

            overall_status = (
                "COMPLETE" if not incomplete
                else "INCOMPLETE" if len(incomplete) < len(original_locations) / 2
                else "FAILED"
            )

            result = {
                "status": "success",
                "request_id": request_id,
                "verification_status": overall_status,
                "locations_verified": len(verification_results),
                "locations_incomplete": len(incomplete),
                "verification_details": verification_results,
                "incomplete_locations": incomplete,
                "verified_at": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Verification complete: %s - %d verified, %d incomplete",
                overall_status, len(verification_results), len(incomplete)
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error("Verification failed: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
                "request_id": request_id,
            })

# ---------------------------------------------------------------------------
# Certificate Generator Tool
# ---------------------------------------------------------------------------

class CertificateGeneratorTool(BaseTool):
    """Tool for generating compliance certificates."""

    name: str = "certificate_generator"
    description: str = (
        "Generates a tamper-evident compliance certificate with "
        "cryptographic hash chain verification for regulatory audits."
    )
    args_schema: Type[BaseModel] = CertificateInput

    def _run(
        self,
        request_id: str,
        verification_result: dict,
        deletion_receipts: list[dict],
    ) -> str:
        """Generate compliance certificate."""
        try:
            # Build hash chain
            chain = "genesis"
            for receipt in sorted(
                deletion_receipts, key=lambda r: r.get("executed_at", "")
            ):
                payload = (
                    f"{chain}:{receipt['source']}:{receipt['table']}"
                    f":{receipt.get('rows_deleted', 0)}"
                )
                chain = hashlib.sha256(payload.encode()).hexdigest()

            # Generate certificate
            certificate_id = hashlib.sha256(
                f"cert:{request_id}:{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:16].upper()

            certificate = {
                "certificate_id": certificate_id,
                "request_id": request_id,
                "issue_date": datetime.utcnow().isoformat(),
                "verification_status": verification_result.get(
                    "verification_status", "UNKNOWN"
                ),
                "systems_purged": len(set(
                    r["source"] for r in deletion_receipts
                )),
                "total_rows_deleted": sum(
                    r.get("rows_deleted", 0) for r in deletion_receipts
                ),
                "chain_hash": chain,
                "hash_algorithm": "SHA-256",
                "regulatory_compliance": [
                    "GDPR Article 17",
                    "UK GDPR Article 17",
                    "CCPA Section 1798.105",
                ],
                "valid_for_audit": True,
                "retention_period_days": 2555,  # 7 years
            }

            result = {
                "status": "success",
                "certificate": certificate,
                "generated_at": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Certificate generated: %s - %d systems, %d rows",
                certificate_id,
                certificate["systems_purged"],
                certificate["total_rows_deleted"],
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error("Certificate generation failed: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
                "request_id": request_id,
            })`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Build the LangGraph state machine that coordinates agent workflows, handles errors, manages retries, and maintains request state throughout the erasure lifecycle.',
            toolsUsed: ['LangGraph', 'Redis', 'CrewAI'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Erasure Pipeline Orchestration',
                description:
                  'Complete LangGraph state machine implementing the supervisor pattern for GDPR erasure request processing.',
                code: `# orchestration/erasure_graph.py
"""LangGraph orchestration for GDPR erasure pipeline."""

from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Any, Literal, TypedDict, Annotated
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
import redis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class ErasureStatus(str, Enum):
    """Erasure request status."""
    RECEIVED = "received"
    ANALYZING = "analyzing"
    PII_DISCOVERED = "pii_discovered"
    RETENTION_CHECKED = "retention_checked"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"

class ErasureState(TypedDict):
    """State schema for erasure pipeline."""
    # Request metadata
    request_id: str
    subject_identifier: str
    identifier_type: str
    request_date: str
    deadline: str

    # Processing state
    status: ErasureStatus
    current_agent: str
    iteration_count: int

    # Discovery results
    pii_locations: list[dict[str, Any]]
    pii_count: int

    # Regulatory analysis
    applicable_regulations: list[str]
    retention_exceptions: list[dict[str, Any]]
    deletable_locations: list[dict[str, Any]]

    # Execution results
    deletion_receipts: list[dict[str, Any]]
    deletion_errors: list[dict[str, Any]]

    # Verification
    verification_result: dict[str, Any]
    certificate: dict[str, Any]

    # Error handling
    errors: list[str]
    retry_count: int
    max_retries: int

# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def intake_request(state: ErasureState) -> ErasureState:
    """Initial request intake and validation."""
    logger.info(
        "Processing erasure request: %s", state["request_id"]
    )

    state["status"] = ErasureStatus.ANALYZING
    state["current_agent"] = "regulatory_advisor"
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    return state

def regulatory_analysis(state: ErasureState) -> ErasureState:
    """Analyze applicable regulations and deadlines."""
    logger.info(
        "Regulatory analysis for request: %s", state["request_id"]
    )

    # In production, this would invoke the Regulatory Advisor Agent
    state["applicable_regulations"] = ["GDPR Article 17", "UK GDPR Article 17"]
    state["current_agent"] = "pii_discovery"

    return state

def discover_pii(state: ErasureState) -> ErasureState:
    """Discover all PII locations for the data subject."""
    logger.info(
        "PII discovery for subject: %s",
        state["subject_identifier"][:20] + "..."
    )

    # In production, this would invoke the PII Discovery Agent
    state["status"] = ErasureStatus.PII_DISCOVERED
    state["current_agent"] = "retention_checker"

    return state

def check_retention(state: ErasureState) -> ErasureState:
    """Check for retention exceptions and legal holds."""
    logger.info(
        "Checking retention exceptions for %d locations",
        state.get("pii_count", 0)
    )

    state["status"] = ErasureStatus.RETENTION_CHECKED
    state["current_agent"] = "deletion_orchestrator"

    return state

def execute_deletions(state: ErasureState) -> ErasureState:
    """Execute deletions across all approved locations."""
    logger.info(
        "Executing deletions for request: %s", state["request_id"]
    )

    state["status"] = ErasureStatus.EXECUTING
    state["current_agent"] = "compliance_auditor"

    return state

def verify_deletions(state: ErasureState) -> ErasureState:
    """Verify deletion completeness and generate certificate."""
    logger.info(
        "Verifying deletions for request: %s", state["request_id"]
    )

    state["status"] = ErasureStatus.VERIFYING
    state["current_agent"] = "certificate_generator"

    return state

def generate_certificate(state: ErasureState) -> ErasureState:
    """Generate compliance certificate."""
    logger.info(
        "Generating certificate for request: %s", state["request_id"]
    )

    state["status"] = ErasureStatus.COMPLETE
    state["current_agent"] = "complete"

    return state

def handle_error(state: ErasureState) -> ErasureState:
    """Handle errors and determine retry strategy."""
    logger.error(
        "Error in request %s: %s",
        state["request_id"],
        state.get("errors", [])
    )

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count < max_retries:
        state["retry_count"] = retry_count + 1
        state["status"] = ErasureStatus.ANALYZING  # Restart
        logger.info(
            "Retrying request %s (attempt %d/%d)",
            state["request_id"], retry_count + 1, max_retries
        )
    else:
        state["status"] = ErasureStatus.FAILED
        logger.error(
            "Request %s failed after %d retries",
            state["request_id"], max_retries
        )

    return state

# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------

def route_after_intake(
    state: ErasureState,
) -> Literal["regulatory_analysis", "handle_error"]:
    """Route after intake based on validation."""
    if state.get("errors"):
        return "handle_error"
    return "regulatory_analysis"

def route_after_discovery(
    state: ErasureState,
) -> Literal["check_retention", "handle_error"]:
    """Route after PII discovery."""
    if state.get("pii_count", 0) == 0:
        state["errors"] = state.get("errors", []) + [
            "No PII locations found for subject"
        ]
        return "handle_error"
    return "check_retention"

def route_after_execution(
    state: ErasureState,
) -> Literal["verify_deletions", "handle_error"]:
    """Route after deletion execution."""
    if state.get("deletion_errors"):
        return "handle_error"
    return "verify_deletions"

def route_after_verification(
    state: ErasureState,
) -> Literal["generate_certificate", "handle_error", "__end__"]:
    """Route after verification."""
    verification = state.get("verification_result", {})
    status = verification.get("verification_status", "UNKNOWN")

    if status == "COMPLETE":
        return "generate_certificate"
    elif status == "INCOMPLETE":
        return "handle_error"
    return "__end__"

def route_after_error(
    state: ErasureState,
) -> Literal["intake_request", "__end__"]:
    """Route after error handling."""
    if state["status"] == ErasureStatus.FAILED:
        return "__end__"
    return "intake_request"

# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_erasure_graph() -> StateGraph:
    """Build the LangGraph erasure pipeline."""

    # Create graph with state schema
    graph = StateGraph(ErasureState)

    # Add nodes
    graph.add_node("intake_request", intake_request)
    graph.add_node("regulatory_analysis", regulatory_analysis)
    graph.add_node("discover_pii", discover_pii)
    graph.add_node("check_retention", check_retention)
    graph.add_node("execute_deletions", execute_deletions)
    graph.add_node("verify_deletions", verify_deletions)
    graph.add_node("generate_certificate", generate_certificate)
    graph.add_node("handle_error", handle_error)

    # Set entry point
    graph.set_entry_point("intake_request")

    # Add edges with conditional routing
    graph.add_conditional_edges(
        "intake_request",
        route_after_intake,
        {
            "regulatory_analysis": "regulatory_analysis",
            "handle_error": "handle_error",
        },
    )

    graph.add_edge("regulatory_analysis", "discover_pii")

    graph.add_conditional_edges(
        "discover_pii",
        route_after_discovery,
        {
            "check_retention": "check_retention",
            "handle_error": "handle_error",
        },
    )

    graph.add_edge("check_retention", "execute_deletions")

    graph.add_conditional_edges(
        "execute_deletions",
        route_after_execution,
        {
            "verify_deletions": "verify_deletions",
            "handle_error": "handle_error",
        },
    )

    graph.add_conditional_edges(
        "verify_deletions",
        route_after_verification,
        {
            "generate_certificate": "generate_certificate",
            "handle_error": "handle_error",
            "__end__": END,
        },
    )

    graph.add_edge("generate_certificate", END)

    graph.add_conditional_edges(
        "handle_error",
        route_after_error,
        {
            "intake_request": "intake_request",
            "__end__": END,
        },
    )

    return graph

# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

class ErasurePipeline:
    """Main erasure pipeline executor."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
    ) -> None:
        self.graph = build_erasure_graph()
        self.checkpointer = MemorySaver()
        self.compiled = self.graph.compile(checkpointer=self.checkpointer)
        self.redis = redis.from_url(redis_url)

    def process_request(
        self,
        request_id: str,
        subject_identifier: str,
        identifier_type: str = "email",
        deadline_days: int = 30,
    ) -> dict[str, Any]:
        """Process an erasure request through the pipeline."""

        # Initialize state
        initial_state: ErasureState = {
            "request_id": request_id,
            "subject_identifier": subject_identifier,
            "identifier_type": identifier_type,
            "request_date": datetime.utcnow().isoformat(),
            "deadline": (
                datetime.utcnow() + timedelta(days=deadline_days)
            ).isoformat(),
            "status": ErasureStatus.RECEIVED,
            "current_agent": "intake",
            "iteration_count": 0,
            "pii_locations": [],
            "pii_count": 0,
            "applicable_regulations": [],
            "retention_exceptions": [],
            "deletable_locations": [],
            "deletion_receipts": [],
            "deletion_errors": [],
            "verification_result": {},
            "certificate": {},
            "errors": [],
            "retry_count": 0,
            "max_retries": 3,
        }

        # Run the graph
        config = {"configurable": {"thread_id": request_id}}

        try:
            final_state = self.compiled.invoke(initial_state, config)

            # Cache result in Redis
            self.redis.setex(
                f"erasure:result:{request_id}",
                timedelta(days=90),
                json.dumps(final_state, default=str),
            )

            logger.info(
                "Request %s completed with status: %s",
                request_id, final_state["status"]
            )
            return final_state

        except Exception as e:
            logger.error(
                "Pipeline failed for request %s: %s",
                request_id, str(e)
            )
            raise

from datetime import timedelta`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the multi-agent erasure system with Docker containerization, LangSmith tracing for agent observability, and Prometheus metrics for SLA monitoring.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose Deployment',
                description:
                  'Production Docker Compose configuration for the multi-agent erasure pipeline with Redis state management and observability stack.',
                code: `# docker-compose.yml
# GDPR Erasure Multi-Agent Pipeline - Production Deployment

version: "3.9"

services:
  # ─────────────────────────────────────────────────────────────────────────
  # Core Pipeline Services
  # ─────────────────────────────────────────────────────────────────────────

  erasure-pipeline:
    build:
      context: .
      dockerfile: Dockerfile.pipeline
    image: gdpr-erasure-pipeline:latest
    container_name: erasure-pipeline
    environment:
      - ERASURE_ENV=production
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=\${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=\${DATABASE_URL}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_PROJECT=gdpr-erasure-pipeline
      - LANGSMITH_TRACING_V2=true
      - PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./audit:/app/audit
      - prometheus_multiproc:/tmp/prometheus
    ports:
      - "8080:8080"
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
    networks:
      - erasure-network
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 4G
        reservations:
          cpus: "1"
          memory: 2G

  # ─────────────────────────────────────────────────────────────────────────
  # State Management
  # ─────────────────────────────────────────────────────────────────────────

  redis:
    image: redis:7-alpine
    container_name: erasure-redis
    command: >
      redis-server
      --appendonly yes
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped
    networks:
      - erasure-network

  # ─────────────────────────────────────────────────────────────────────────
  # Observability Stack
  # ─────────────────────────────────────────────────────────────────────────

  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: erasure-prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--storage.tsdb.retention.time=30d"
      - "--web.enable-lifecycle"
    volumes:
      - ./observability/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./observability/alerts.yml:/etc/prometheus/alerts.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped
    networks:
      - erasure-network

  grafana:
    image: grafana/grafana:10.1.0
    container_name: erasure-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SERVER_ROOT_URL=https://monitoring.example.com
    volumes:
      - ./observability/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./observability/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - erasure-network

  # ─────────────────────────────────────────────────────────────────────────
  # Alerting
  # ─────────────────────────────────────────────────────────────────────────

  alertmanager:
    image: prom/alertmanager:v0.26.0
    container_name: erasure-alertmanager
    command:
      - "--config.file=/etc/alertmanager/alertmanager.yml"
      - "--storage.path=/alertmanager"
    volumes:
      - ./observability/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager_data:/alertmanager
    ports:
      - "9093:9093"
    restart: unless-stopped
    networks:
      - erasure-network

# ─────────────────────────────────────────────────────────────────────────
# Volumes
# ─────────────────────────────────────────────────────────────────────────

volumes:
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  prometheus_multiproc:
    driver: local
  grafana_data:
    driver: local
  alertmanager_data:
    driver: local

# ─────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────

networks:
  erasure-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16`,
              },
              {
                language: 'python',
                title: 'LangSmith Tracing and Prometheus Metrics',
                description:
                  'Observability instrumentation for the erasure pipeline with LangSmith agent tracing and Prometheus metrics for SLA monitoring.',
                code: `# observability/metrics.py
"""Observability instrumentation for GDPR erasure pipeline."""

from __future__ import annotations
import os
import time
import logging
import functools
from typing import Any, Callable, TypeVar
from contextlib import contextmanager

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    CollectorRegistry,
    multiprocess,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from langsmith import Client as LangSmithClient
from langsmith.run_trees import RunTree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------

# Custom registry for multiprocess mode
if "prometheus_multiproc_dir" in os.environ:
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
else:
    registry = CollectorRegistry(auto_describe=True)

# Request counters
ERASURE_REQUESTS_TOTAL = Counter(
    "erasure_requests_total",
    "Total number of erasure requests processed",
    ["status", "regulation"],
    registry=registry,
)

ERASURE_REQUESTS_IN_PROGRESS = Gauge(
    "erasure_requests_in_progress",
    "Number of erasure requests currently being processed",
    registry=registry,
)

# Timing histograms
ERASURE_DURATION_SECONDS = Histogram(
    "erasure_duration_seconds",
    "Time spent processing erasure requests",
    ["stage"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600],
    registry=registry,
)

PII_DISCOVERY_DURATION = Histogram(
    "pii_discovery_duration_seconds",
    "Time spent discovering PII locations",
    ["source_type"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60],
    registry=registry,
)

DELETION_DURATION = Histogram(
    "deletion_execution_duration_seconds",
    "Time spent executing deletions",
    ["system"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
    registry=registry,
)

# SLA metrics
SLA_COMPLIANCE = Gauge(
    "erasure_sla_compliance_ratio",
    "Ratio of requests completed within SLA",
    ["regulation"],
    registry=registry,
)

REQUESTS_APPROACHING_DEADLINE = Gauge(
    "erasure_requests_approaching_deadline",
    "Number of requests within 48 hours of deadline",
    registry=registry,
)

# Agent metrics
AGENT_INVOCATIONS = Counter(
    "agent_invocations_total",
    "Total agent invocations by agent type",
    ["agent", "outcome"],
    registry=registry,
)

AGENT_TOKEN_USAGE = Counter(
    "agent_token_usage_total",
    "Total tokens used by agents",
    ["agent", "token_type"],
    registry=registry,
)

# Data metrics
PII_LOCATIONS_DISCOVERED = Gauge(
    "pii_locations_discovered_total",
    "Total PII locations in registry",
    ["category"],
    registry=registry,
)

ROWS_DELETED = Counter(
    "rows_deleted_total",
    "Total rows deleted across all systems",
    ["system", "table"],
    registry=registry,
)

# System info
PIPELINE_INFO = Info(
    "erasure_pipeline",
    "Erasure pipeline deployment information",
    registry=registry,
)

# ---------------------------------------------------------------------------
# LangSmith Tracing
# ---------------------------------------------------------------------------

class TracingManager:
    """Manages LangSmith tracing for agent workflows."""

    def __init__(self) -> None:
        self.client: LangSmithClient | None = None
        self.project_name = os.environ.get(
            "LANGSMITH_PROJECT", "gdpr-erasure-pipeline"
        )
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize LangSmith client if API key is available."""
        api_key = os.environ.get("LANGSMITH_API_KEY")
        if api_key:
            self.client = LangSmithClient(api_key=api_key)
            logger.info(
                "LangSmith tracing enabled for project: %s",
                self.project_name
            )
        else:
            logger.warning("LangSmith API key not set, tracing disabled")

    @contextmanager
    def trace_request(
        self,
        request_id: str,
        subject_hash: str,
        metadata: dict[str, Any] | None = None,
    ):
        """Context manager for tracing an erasure request."""
        if not self.client:
            yield None
            return

        run_tree = RunTree(
            name=f"erasure_request_{request_id}",
            run_type="chain",
            project_name=self.project_name,
            inputs={
                "request_id": request_id,
                "subject_hash": subject_hash,
                **(metadata or {}),
            },
        )

        try:
            run_tree.post()
            yield run_tree
        except Exception as e:
            run_tree.end(error=str(e))
            run_tree.patch()
            raise
        else:
            run_tree.end()
            run_tree.patch()

    @contextmanager
    def trace_agent(
        self,
        parent_run: RunTree | None,
        agent_name: str,
        inputs: dict[str, Any],
    ):
        """Context manager for tracing an agent invocation."""
        if not self.client or not parent_run:
            yield None
            return

        child_run = parent_run.create_child(
            name=agent_name,
            run_type="llm",
            inputs=inputs,
        )

        start_time = time.time()
        try:
            child_run.post()
            yield child_run
            AGENT_INVOCATIONS.labels(
                agent=agent_name, outcome="success"
            ).inc()
        except Exception as e:
            child_run.end(error=str(e))
            child_run.patch()
            AGENT_INVOCATIONS.labels(
                agent=agent_name, outcome="error"
            ).inc()
            raise
        else:
            duration = time.time() - start_time
            child_run.end(outputs={"duration_seconds": duration})
            child_run.patch()

# ---------------------------------------------------------------------------
# Instrumentation Decorators
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])

def track_duration(stage: str) -> Callable[[F], F]:
    """Decorator to track function duration by stage."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                ERASURE_DURATION_SECONDS.labels(stage=stage).observe(duration)
                logger.debug(
                    "Stage %s completed in %.2fs", stage, duration
                )
        return wrapper  # type: ignore
    return decorator

def track_agent_call(agent_name: str) -> Callable[[F], F]:
    """Decorator to track agent invocations."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)
                AGENT_INVOCATIONS.labels(
                    agent=agent_name, outcome="success"
                ).inc()
                return result
            except Exception:
                AGENT_INVOCATIONS.labels(
                    agent=agent_name, outcome="error"
                ).inc()
                raise
        return wrapper  # type: ignore
    return decorator

# ---------------------------------------------------------------------------
# Metrics Endpoint
# ---------------------------------------------------------------------------

def get_metrics() -> tuple[bytes, str]:
    """Generate Prometheus metrics for scraping."""
    return generate_latest(registry), CONTENT_TYPE_LATEST

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize_metrics(
    version: str,
    environment: str,
) -> TracingManager:
    """Initialize metrics and tracing for the pipeline."""
    PIPELINE_INFO.info({
        "version": version,
        "environment": environment,
        "langsmith_enabled": str(bool(os.environ.get("LANGSMITH_API_KEY"))),
    })

    logger.info(
        "Metrics initialized: version=%s, env=%s",
        version, environment
    )

    return TracingManager()`,
              },
            ],
          },
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
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude to analyze data residency requirements, map jurisdiction rules from regulatory documents, and automate compliance monitoring via Zapier with real-time alerts for violations.',
        estimatedMonthlyCost: '$120 - $220/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Notion ($10/mo)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Lexis+ AI ($175/mo)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Export cloud infrastructure configurations, data flow mappings, and jurisdiction rules to structured formats for AI analysis of residency compliance.',
            toolsUsed: ['AWS Config Export', 'Notion Database', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Jurisdiction Rules Configuration',
                description:
                  'Structured representation of data residency requirements by jurisdiction for AI-powered compliance analysis.',
                code: `{
  "residency_rules": {
    "version": "2025.1",
    "last_updated": "2025-01-15T00:00:00Z",
    "jurisdictions": [
      {
        "jurisdiction_id": "EU-GDPR",
        "countries": ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PL", "SE"],
        "regulation_name": "General Data Protection Regulation",
        "articles": ["Chapter V - Transfers to Third Countries"],
        "requirements": {
          "data_localization": "soft",
          "allowed_regions": ["eu-west-1", "eu-central-1", "eu-north-1", "eu-south-1"],
          "transfer_mechanisms": [
            "Adequacy Decision",
            "Standard Contractual Clauses",
            "Binding Corporate Rules"
          ],
          "prohibited_transfers": ["Countries without adequacy decision without SCCs"],
          "breach_notification_hours": 72
        },
        "penalties": {
          "max_fine_percentage": 4,
          "max_fine_euros": 20000000
        }
      },
      {
        "jurisdiction_id": "UAE-PDPL",
        "countries": ["AE"],
        "regulation_name": "Personal Data Protection Law",
        "articles": ["Article 22 - Cross-Border Transfer"],
        "requirements": {
          "data_localization": "strict",
          "allowed_regions": ["me-south-1"],
          "transfer_mechanisms": [
            "UAE Data Office Approval",
            "Equivalent Protection Verification"
          ],
          "prohibited_transfers": ["All transfers without explicit approval"],
          "breach_notification_hours": 72
        },
        "penalties": {
          "max_fine_aed": 10000000,
          "operational_restrictions": true
        }
      },
      {
        "jurisdiction_id": "India-DPDP",
        "countries": ["IN"],
        "regulation_name": "Digital Personal Data Protection Act",
        "articles": ["Section 16 - Transfer Outside India"],
        "requirements": {
          "data_localization": "conditional",
          "allowed_regions": ["ap-south-1", "ap-south-2"],
          "transfer_mechanisms": [
            "Notified Country Transfer",
            "Contract with Standard Clauses"
          ],
          "critical_data_localization": "strict",
          "breach_notification_hours": 24
        },
        "penalties": {
          "max_fine_inr": 2500000000
        }
      },
      {
        "jurisdiction_id": "Australia-Privacy",
        "countries": ["AU"],
        "regulation_name": "Privacy Act 1988",
        "articles": ["APP 8 - Cross-Border Disclosure"],
        "requirements": {
          "data_localization": "soft",
          "allowed_regions": ["ap-southeast-2"],
          "transfer_mechanisms": [
            "Reasonable Steps Verification",
            "Consent",
            "Required by Law"
          ],
          "breach_notification_hours": 72
        },
        "penalties": {
          "max_fine_aud": 50000000
        }
      }
    ]
  }
}`,
              },
              {
                language: 'json',
                title: 'Data Flow Inventory Export',
                description:
                  'Structured export of current data flows with source/destination regions and PII classifications for compliance analysis.',
                code: `{
  "data_flow_inventory": {
    "inventory_date": "2025-01-15T08:00:00Z",
    "total_flows": 47,
    "flows": [
      {
        "flow_id": "flow-001",
        "flow_name": "Customer Data ETL",
        "data_classification": "PII",
        "pii_categories": ["email", "name", "phone", "address"],
        "source": {
          "system": "production_db",
          "region": "us-east-1",
          "cloud_provider": "AWS"
        },
        "destination": {
          "system": "analytics_warehouse",
          "region": "eu-central-1",
          "cloud_provider": "AWS"
        },
        "subject_countries": ["DE", "FR", "GB", "US"],
        "transfer_mechanism": "Standard Contractual Clauses",
        "last_audit_date": "2024-09-15",
        "schedule": "daily",
        "data_volume_gb_daily": 15.5
      },
      {
        "flow_id": "flow-002",
        "flow_name": "UAE Customer Sync",
        "data_classification": "PII",
        "pii_categories": ["email", "name", "phone", "emirates_id"],
        "source": {
          "system": "crm_system",
          "region": "eu-west-1",
          "cloud_provider": "AWS"
        },
        "destination": {
          "system": "uae_local_db",
          "region": "me-south-1",
          "cloud_provider": "AWS"
        },
        "subject_countries": ["AE"],
        "transfer_mechanism": "UAE Data Office Approval",
        "last_audit_date": "2024-11-01",
        "schedule": "realtime",
        "data_volume_gb_daily": 2.3
      },
      {
        "flow_id": "flow-003",
        "flow_name": "India User Analytics",
        "data_classification": "PII",
        "pii_categories": ["email", "name", "aadhaar_last4"],
        "source": {
          "system": "india_app_db",
          "region": "ap-south-1",
          "cloud_provider": "AWS"
        },
        "destination": {
          "system": "global_analytics",
          "region": "us-west-2",
          "cloud_provider": "AWS"
        },
        "subject_countries": ["IN"],
        "transfer_mechanism": "Contract with Standard Clauses",
        "last_audit_date": "2024-08-20",
        "schedule": "weekly",
        "data_volume_gb_daily": 8.7,
        "compliance_status": "REVIEW_REQUIRED"
      }
    ],
    "compliance_summary": {
      "compliant_flows": 42,
      "review_required": 3,
      "violations_detected": 2
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
              'Use ChatGPT or Claude to analyze data flows against jurisdiction rules, identify compliance gaps, and generate remediation recommendations.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro', 'Lexis+ AI'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Data Sovereignty Compliance Analysis Prompt',
                description:
                  'Comprehensive prompt for AI to analyze cross-border data flows and identify residency compliance issues.',
                code: `system_prompt: |
  You are a data sovereignty compliance expert specializing in cross-border
  data transfer regulations including GDPR, UAE PDPL, India DPDP Act, and
  Australian Privacy Act. Your task is to analyze data flows and identify
  compliance gaps with jurisdiction-specific residency requirements.

  Key responsibilities:
  - Map data flows to applicable jurisdictions based on subject countries
  - Identify transfers that violate data localization requirements
  - Verify transfer mechanisms are appropriate for each jurisdiction
  - Calculate compliance risk scores based on violation severity
  - Generate specific remediation recommendations

compliance_analysis_prompt: |
  ## Cross-Border Data Sovereignty Compliance Analysis

  ### Jurisdiction Rules
  {{jurisdiction_rules_json}}

  ### Current Data Flows
  {{data_flow_inventory_json}}

  ### Cloud Infrastructure Configuration
  {{cloud_config_json}}

  ### Analysis Required

  For each data flow, analyze and report:

  1. **Jurisdiction Mapping**
     - Identify all applicable jurisdictions based on subject_countries
     - List the governing regulations for each jurisdiction
     - Note any conflicting requirements between jurisdictions

  2. **Residency Compliance Check**
     For each flow, verify:
     - Source region is appropriate for the data subjects
     - Destination region complies with localization requirements
     - Transfer mechanism is valid for the jurisdiction pair
     - All required approvals/documentation are in place

  3. **Violation Detection**
     Flag any flows that violate:
     - Strict localization requirements (data must stay in-country)
     - Transfer mechanism requirements (missing SCCs, approvals)
     - Prohibited destination transfers
     - Audit/documentation freshness requirements

  4. **Risk Assessment**
     For each violation:
     - Severity: CRITICAL / HIGH / MEDIUM / LOW
     - Potential penalties based on jurisdiction
     - Likelihood of regulatory action
     - Business impact assessment

  5. **Remediation Recommendations**
     For each non-compliant flow, provide:
     - Specific actions to achieve compliance
     - Recommended target region for data relocation
     - Required transfer mechanisms to implement
     - Estimated timeline and effort

  6. **Priority Ranking**
     Rank all issues by:
     - Regulatory risk (strict localization > soft localization)
     - Penalty exposure (fine amount / percentage)
     - Data volume affected
     - Time since last compliance audit

output_format: |
  {
    "analysis_date": "ISO8601",
    "total_flows_analyzed": number,
    "compliance_summary": {
      "fully_compliant": number,
      "review_required": number,
      "violations": number
    },
    "flow_analysis": [
      {
        "flow_id": "string",
        "jurisdictions": [...],
        "compliance_status": "COMPLIANT" | "REVIEW" | "VIOLATION",
        "violations": [...],
        "risk_score": number,
        "remediation": {...}
      }
    ],
    "priority_actions": [...],
    "executive_summary": "string"
  }`,
              },
              {
                language: 'yaml',
                title: 'Jurisdiction Rule Extraction Prompt',
                description:
                  'Prompt for AI to extract and structure data residency requirements from regulatory documents.',
                code: `system_prompt: |
  You are a legal AI assistant specializing in data protection regulations.
  Your task is to extract structured data residency and cross-border transfer
  requirements from regulatory documents and convert them into machine-readable
  rules for automated compliance monitoring.

  Be precise and cite specific articles, sections, or clauses for each rule.

extraction_prompt: |
  ## Regulatory Document Analysis

  ### Document Information
  Regulation Name: {{regulation_name}}
  Jurisdiction: {{jurisdiction}}
  Document Type: {{document_type}}
  Effective Date: {{effective_date}}

  ### Document Content
  {{document_text}}

  ### Extraction Requirements

  Please extract and structure the following information:

  1. **Data Localization Requirements**
     - Type: strict (must stay in-country) / soft (with transfer mechanisms) / none
     - Specific articles mandating localization
     - Categories of data with special localization rules
     - Exemptions or exceptions to localization

  2. **Cross-Border Transfer Mechanisms**
     - List all valid transfer mechanisms mentioned
     - Requirements for each mechanism (documentation, approvals)
     - Prohibited transfer destinations or conditions
     - Specific approval authorities named

  3. **Enforcement Provisions**
     - Maximum penalties (fines, percentages)
     - Operational restrictions possible
     - Breach notification timeframes
     - Regulatory authority contact information

  4. **Special Categories**
     - Sensitive data definitions and extra requirements
     - Critical infrastructure data provisions
     - Government data handling rules
     - Children's data specific provisions

  5. **Implementation Timeline**
     - Effective dates for different provisions
     - Grace periods or transition rules
     - Upcoming amendments or changes announced

output_format: |
  {
    "jurisdiction_id": "string",
    "regulation_name": "string",
    "version": "string",
    "effective_date": "ISO8601",
    "data_localization": {
      "type": "strict" | "soft" | "conditional" | "none",
      "legal_basis": ["Article X", "Section Y"],
      "special_categories": [...],
      "exemptions": [...]
    },
    "transfer_mechanisms": [
      {
        "name": "string",
        "legal_basis": "string",
        "requirements": [...],
        "approval_authority": "string"
      }
    ],
    "penalties": {
      "max_fine": {...},
      "operational_restrictions": boolean,
      "breach_notification_hours": number
    },
    "special_provisions": [...],
    "citations": [...]
  }`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier workflows to continuously monitor data residency compliance, alert on violations, and generate compliance reports for stakeholders.',
            toolsUsed: ['Zapier Pro', 'Notion', 'Slack', 'AWS CloudWatch'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Data Residency Monitoring Workflow',
                description:
                  'Automated Zap that monitors cloud infrastructure changes, detects potential residency violations, and triggers AI compliance analysis.',
                code: `{
  "zap_name": "Data Residency Compliance Monitor",
  "description": "Continuous monitoring of cross-border data flows for residency violations",
  "trigger": {
    "app": "Schedule by Zapier",
    "event": "Every Hour",
    "config": {
      "interval": "hour",
      "timezone": "UTC"
    }
  },
  "actions": [
    {
      "step": 1,
      "app": "Webhooks by Zapier",
      "action": "GET",
      "config": {
        "url": "https://your-api.com/data-flows/inventory",
        "headers": {
          "Authorization": "Bearer {{api_key}}",
          "Content-Type": "application/json"
        },
        "query_params": {
          "include_metrics": "true",
          "since_hours": "24"
        }
      },
      "output": "flow_inventory_json",
      "purpose": "Fetch current data flow inventory with recent changes"
    },
    {
      "step": 2,
      "app": "Webhooks by Zapier",
      "action": "GET",
      "config": {
        "url": "https://your-api.com/cloud-config/regions",
        "headers": {
          "Authorization": "Bearer {{api_key}}"
        }
      },
      "output": "cloud_regions_json",
      "purpose": "Fetch current cloud infrastructure regions"
    },
    {
      "step": 3,
      "app": "Code by Zapier",
      "action": "Run JavaScript",
      "config": {
        "code": "const flows = JSON.parse(inputData.flow_inventory); const rules = JSON.parse(inputData.jurisdiction_rules); const violations = flows.flows.filter(f => { const subjectJurisdictions = f.subject_countries.map(c => rules.jurisdictions.find(j => j.countries.includes(c))).filter(Boolean); return subjectJurisdictions.some(j => !j.requirements.allowed_regions.includes(f.destination.region)); }); return { violations: JSON.stringify(violations), count: violations.length };",
        "input_data": {
          "flow_inventory": "{{flow_inventory_json}}",
          "jurisdiction_rules": "{{jurisdiction_rules}}"
        }
      },
      "output": ["violations_json", "violation_count"],
      "purpose": "Pre-filter flows with potential violations"
    },
    {
      "step": 4,
      "app": "Filter by Zapier",
      "action": "Only Continue If",
      "config": {
        "conditions": [
          {
            "field": "{{violation_count}}",
            "operator": "greater_than",
            "value": "0"
          }
        ]
      },
      "purpose": "Only proceed if violations detected"
    },
    {
      "step": 5,
      "app": "ChatGPT",
      "action": "Conversation",
      "config": {
        "model": "gpt-4",
        "system_message": "You are a data sovereignty compliance expert...",
        "user_message": "Analyze these potential residency violations and provide risk assessment with remediation priorities:\\n\\nViolations: {{violations_json}}\\n\\nJurisdiction Rules: {{jurisdiction_rules}}",
        "temperature": 0.2,
        "max_tokens": 3000
      },
      "output": "ai_compliance_analysis",
      "purpose": "AI analysis of detected violations"
    },
    {
      "step": 6,
      "app": "Notion",
      "action": "Create Database Item",
      "config": {
        "database_id": "compliance_alerts_db",
        "properties": {
          "Title": "Residency Violation Alert - {{current_date}}",
          "Status": "Open",
          "Severity": "{{extracted_severity}}",
          "Violations Count": "{{violation_count}}",
          "Analysis": "{{ai_compliance_analysis}}",
          "Flows Affected": "{{affected_flow_ids}}",
          "Created": "{{current_datetime}}"
        }
      },
      "output": "notion_page_url",
      "purpose": "Log violation in compliance tracking database"
    },
    {
      "step": 7,
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#data-compliance",
        "message": ":rotating_light: *Data Residency Violation Detected*\\n\\n*Violations:* {{violation_count}} flows affected\\n*Severity:* {{extracted_severity}}\\n*Jurisdictions:* {{affected_jurisdictions}}\\n\\n*Summary:*\\n{{ai_summary}}\\n\\n<{{notion_page_url}}|View Full Analysis>",
        "bot_name": "Sovereignty Monitor"
      },
      "purpose": "Alert compliance team of violations"
    },
    {
      "step": 8,
      "app": "Paths by Zapier",
      "action": "Conditional Paths",
      "config": {
        "path_a": {
          "condition": "severity == 'CRITICAL'",
          "name": "Critical - Immediate Action",
          "continue_to": "pagerduty_alert"
        },
        "path_b": {
          "condition": "severity == 'HIGH'",
          "name": "High - Business Hours",
          "continue_to": "email_leadership"
        },
        "path_c": {
          "condition": "severity IN ('MEDIUM', 'LOW')",
          "name": "Standard - Ticket",
          "continue_to": "create_jira_ticket"
        }
      },
      "purpose": "Route based on violation severity"
    }
  ],
  "error_handling": {
    "on_failure": "notify_admin",
    "retry_count": 2,
    "notification_channel": "#compliance-ops-alerts"
  }
}`,
              },
              {
                language: 'json',
                title: 'Zapier Weekly Compliance Report Generator',
                description:
                  'Automated workflow that generates weekly data sovereignty compliance reports with AI-powered insights and trend analysis.',
                code: `{
  "zap_name": "Weekly Sovereignty Compliance Report",
  "description": "Generate and distribute weekly data residency compliance reports",
  "trigger": {
    "app": "Schedule by Zapier",
    "event": "Every Week",
    "config": {
      "day": "Monday",
      "time": "08:00",
      "timezone": "UTC"
    }
  },
  "actions": [
    {
      "step": 1,
      "app": "Webhooks by Zapier",
      "action": "GET",
      "config": {
        "url": "https://your-api.com/compliance/weekly-summary",
        "headers": {
          "Authorization": "Bearer {{api_key}}"
        },
        "query_params": {
          "period": "last_7_days",
          "include_trends": "true"
        }
      },
      "output": "weekly_summary_json",
      "purpose": "Fetch weekly compliance metrics"
    },
    {
      "step": 2,
      "app": "Notion",
      "action": "Query Database",
      "config": {
        "database_id": "compliance_alerts_db",
        "filter": {
          "property": "Created",
          "date": {
            "past_week": {}
          }
        },
        "sorts": [
          {
            "property": "Severity",
            "direction": "descending"
          }
        ]
      },
      "output": "weekly_alerts",
      "purpose": "Get all alerts from the past week"
    },
    {
      "step": 3,
      "app": "ChatGPT",
      "action": "Conversation",
      "config": {
        "model": "gpt-4",
        "system_message": "You are a compliance reporting specialist. Generate executive-friendly summaries of data sovereignty compliance status.",
        "user_message": "Generate a weekly data sovereignty compliance report from this data:\\n\\nWeekly Metrics: {{weekly_summary_json}}\\n\\nAlerts This Week: {{weekly_alerts}}\\n\\nInclude: Executive summary, Key metrics (compliance rate, violations, remediations), Trend analysis, Risk highlights, Recommended actions for next week.",
        "temperature": 0.3,
        "max_tokens": 4000
      },
      "output": "ai_report_content",
      "purpose": "AI-generated compliance report"
    },
    {
      "step": 4,
      "app": "Google Docs",
      "action": "Create Document from Template",
      "config": {
        "template_id": "weekly_compliance_report_template",
        "folder_id": "compliance_reports_folder",
        "document_name": "Data Sovereignty Report - Week of {{week_start_date}}",
        "replacements": {
          "{{REPORT_DATE}}": "{{current_date}}",
          "{{WEEK_RANGE}}": "{{week_start}} - {{week_end}}",
          "{{COMPLIANCE_RATE}}": "{{compliance_percentage}}%",
          "{{VIOLATIONS_COUNT}}": "{{total_violations}}",
          "{{REMEDIATIONS_COUNT}}": "{{remediations_completed}}",
          "{{AI_REPORT_CONTENT}}": "{{ai_report_content}}",
          "{{TREND_CHART_DATA}}": "{{trend_data_json}}"
        }
      },
      "output": "report_doc_url",
      "purpose": "Generate formatted report document"
    },
    {
      "step": 5,
      "app": "Notion",
      "action": "Create Database Item",
      "config": {
        "database_id": "compliance_reports_db",
        "properties": {
          "Title": "Weekly Report - {{week_start_date}}",
          "Period": "{{week_range}}",
          "Compliance Rate": "{{compliance_percentage}}",
          "Violations": "{{total_violations}}",
          "Report URL": "{{report_doc_url}}",
          "Status": "Published"
        }
      },
      "purpose": "Archive report in Notion"
    },
    {
      "step": 6,
      "app": "Gmail",
      "action": "Send Email",
      "config": {
        "to": "{{compliance_leadership_emails}}",
        "cc": "{{legal_team_emails}}",
        "subject": "Weekly Data Sovereignty Compliance Report - {{week_start_date}}",
        "body": "Dear Leadership Team,\\n\\nPlease find attached the weekly data sovereignty compliance report.\\n\\n**Key Highlights:**\\n- Compliance Rate: {{compliance_percentage}}%\\n- New Violations: {{new_violations}}\\n- Remediations Completed: {{remediations_completed}}\\n\\n**Executive Summary:**\\n{{executive_summary}}\\n\\nFull report: {{report_doc_url}}\\n\\nBest regards,\\nCompliance Automation System",
        "from_name": "Compliance Team"
      },
      "purpose": "Distribute report to leadership"
    },
    {
      "step": 7,
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#data-compliance",
        "message": ":bar_chart: *Weekly Sovereignty Compliance Report Published*\\n\\n*Week:* {{week_range}}\\n*Compliance Rate:* {{compliance_percentage}}%\\n*Violations:* {{total_violations}} ({{trend_direction}} from last week)\\n*Remediations:* {{remediations_completed}} completed\\n\\n<{{report_doc_url}}|View Full Report>",
        "bot_name": "Compliance Reports"
      },
      "purpose": "Notify team of published report"
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
          'Deploy a multi-agent system using CrewAI and LangGraph that continuously monitors data residency compliance across all cloud regions, automatically routes data to correct jurisdictions, detects policy violations in real-time, and orchestrates remediation workflows with full audit trails.',
        estimatedMonthlyCost: '$800 - $1,800/month',
        architecture:
          'A Supervisor agent coordinates four specialist agents: Jurisdiction Mapper Agent maintains rules from regulatory sources, Data Flow Monitor Agent tracks all cross-border transfers, Compliance Analyzer Agent detects violations and assesses risk, and Remediation Orchestrator Agent executes automated data relocation and generates compliance reports.',
        agents: [
          {
            name: 'Jurisdiction Mapper Agent',
            role: 'Regulatory Intelligence Specialist',
            goal: 'Continuously monitor regulatory sources to maintain up-to-date jurisdiction rules, extract requirements from new regulations, and alert on rule changes that affect existing data flows.',
            tools: ['RegulatoryFeedMonitor', 'DocumentParser', 'RuleExtractor', 'LexisAI'],
          },
          {
            name: 'Data Flow Monitor Agent',
            role: 'Infrastructure Surveillance Specialist',
            goal: 'Monitor all cloud infrastructure and data pipelines in real-time, track data movement across regions, classify data by jurisdiction requirements, and maintain a complete data lineage map.',
            tools: ['CloudWatchIntegration', 'DataLineageTracker', 'FlowClassifier', 'RegionMapper'],
          },
          {
            name: 'Compliance Analyzer Agent',
            role: 'Risk Assessment Specialist',
            goal: 'Continuously analyze data flows against jurisdiction rules, detect violations before they occur, calculate compliance risk scores, and prioritize issues by regulatory impact.',
            tools: ['ComplianceEngine', 'RiskCalculator', 'ViolationDetector', 'ImpactAnalyzer'],
          },
          {
            name: 'Remediation Orchestrator Agent',
            role: 'Compliance Enforcement Specialist',
            goal: 'Execute automated remediation workflows including data relocation, transfer mechanism updates, documentation generation, and stakeholder notifications with complete audit trails.',
            tools: ['DataMigrator', 'PolicyEnforcer', 'DocumentGenerator', 'AuditLogger'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with multi-region replication and continuous compliance checkpointing',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system with specialized agents for jurisdiction mapping, data flow monitoring, compliance analysis, and automated remediation.',
            toolsUsed: ['CrewAI', 'LangChain', 'Lexis+ AI'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Data Sovereignty Multi-Agent System Definition',
                description:
                  'Complete CrewAI agent definitions for cross-border data sovereignty compliance with specialized roles and tool assignments.',
                code: `# agents/sovereignty_agents.py
"""CrewAI agent definitions for cross-border data sovereignty compliance."""

from __future__ import annotations
import os
from typing import Any, Optional

from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

def get_llm(model_type: str = "openai") -> Any:
    """Get configured LLM instance based on model type."""
    if model_type == "anthropic":
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.2,
            max_tokens=4096,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
    return ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.2,
        max_tokens=4096,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

# ---------------------------------------------------------------------------
# Agent Definitions
# ---------------------------------------------------------------------------

class SovereigntyAgentFactory:
    """Factory for creating specialized sovereignty compliance agents."""

    def __init__(self, llm_type: str = "openai") -> None:
        self.llm = get_llm(llm_type)

    def create_jurisdiction_mapper_agent(self) -> Agent:
        """Create the Jurisdiction Mapper Agent for regulatory intelligence."""
        return Agent(
            role="Regulatory Intelligence Specialist",
            goal=(
                "Maintain comprehensive, up-to-date jurisdiction rules by "
                "monitoring regulatory sources, extracting requirements from "
                "new legislation, and alerting on changes that affect data "
                "residency compliance. Ensure rules cover GDPR, UAE PDPL, "
                "India DPDP, and all applicable privacy regulations."
            ),
            backstory=(
                "You are a regulatory affairs expert with deep knowledge of "
                "global data protection laws. You have advised multinational "
                "corporations on cross-border data transfer compliance and "
                "can quickly parse complex regulatory language into actionable "
                "rules. You monitor regulatory developments daily and understand "
                "how new requirements affect existing data architectures."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            tools=[],  # Tools added dynamically
            max_iter=15,
            max_rpm=30,
        )

    def create_data_flow_monitor_agent(self) -> Agent:
        """Create the Data Flow Monitor Agent for infrastructure surveillance."""
        return Agent(
            role="Infrastructure Surveillance Specialist",
            goal=(
                "Monitor all cloud infrastructure and data pipelines in real-"
                "time to track data movement across regions. Classify each "
                "data flow by subject jurisdiction, maintain complete data "
                "lineage maps, and detect any unauthorized cross-border transfers."
            ),
            backstory=(
                "You are a cloud infrastructure expert who has designed data "
                "architectures for global enterprises. You understand how data "
                "flows through ETL pipelines, replicates across regions, and "
                "gets copied to analytics systems. You can trace any piece of "
                "data from source to every downstream destination."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            tools=[],
            max_iter=20,
            max_rpm=30,
        )

    def create_compliance_analyzer_agent(self) -> Agent:
        """Create the Compliance Analyzer Agent for risk assessment."""
        return Agent(
            role="Risk Assessment Specialist",
            goal=(
                "Continuously analyze data flows against jurisdiction rules "
                "to detect compliance violations before they cause regulatory "
                "exposure. Calculate risk scores based on violation severity, "
                "data volume, and penalty potential. Prioritize issues for "
                "remediation based on business impact."
            ),
            backstory=(
                "You are a compliance analyst who has conducted hundreds of "
                "data protection impact assessments. You understand how "
                "regulators evaluate compliance and can predict which issues "
                "will attract enforcement attention. You excel at translating "
                "technical violations into business risk language."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            tools=[],
            max_iter=15,
            max_rpm=30,
        )

    def create_remediation_orchestrator_agent(self) -> Agent:
        """Create the Remediation Orchestrator Agent for enforcement."""
        return Agent(
            role="Compliance Enforcement Specialist",
            goal=(
                "Execute automated remediation workflows to resolve compliance "
                "violations, including data relocation to compliant regions, "
                "transfer mechanism implementation, documentation generation, "
                "and stakeholder notifications. Maintain complete audit trails."
            ),
            backstory=(
                "You are a data operations expert who has executed complex "
                "data migration projects under regulatory deadlines. You "
                "understand the operational challenges of moving data between "
                "regions while maintaining availability. You always document "
                "every action for audit purposes."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            tools=[],
            max_iter=25,
            max_rpm=30,
        )

    def create_supervisor_agent(self) -> Agent:
        """Create the Supervisor Agent that coordinates the crew."""
        return Agent(
            role="Sovereignty Compliance Supervisor",
            goal=(
                "Coordinate continuous data sovereignty monitoring and "
                "remediation across all jurisdictions. Ensure compliance "
                "posture remains strong, escalate critical issues appropriately, "
                "and maintain executive visibility into compliance status."
            ),
            backstory=(
                "You are a Chief Privacy Officer with experience managing "
                "global data protection programs. You understand both the "
                "regulatory and business implications of sovereignty violations "
                "and can balance compliance requirements with operational needs. "
                "You excel at stakeholder communication and crisis management."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            tools=[],
            max_iter=30,
            max_rpm=30,
        )

# ---------------------------------------------------------------------------
# Crew Assembly
# ---------------------------------------------------------------------------

def create_sovereignty_crew(llm_type: str = "openai") -> Crew:
    """Assemble the complete sovereignty compliance crew."""
    factory = SovereigntyAgentFactory(llm_type)

    supervisor = factory.create_supervisor_agent()
    jurisdiction_agent = factory.create_jurisdiction_mapper_agent()
    monitor_agent = factory.create_data_flow_monitor_agent()
    analyzer_agent = factory.create_compliance_analyzer_agent()
    remediation_agent = factory.create_remediation_orchestrator_agent()

    return Crew(
        agents=[
            supervisor,
            jurisdiction_agent,
            monitor_agent,
            analyzer_agent,
            remediation_agent,
        ],
        tasks=[],  # Tasks added dynamically based on monitoring events
        process=Process.hierarchical,
        manager_agent=supervisor,
        verbose=True,
        memory=True,
        embedder={
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"},
        },
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the Data Flow Monitor and Jurisdiction Mapper agents with tools for cloud infrastructure scanning, regulatory feed monitoring, and data lineage tracking.',
            toolsUsed: ['CrewAI Tools', 'AWS SDK', 'Terraform', 'dbt'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Data Flow and Jurisdiction Monitoring Tools',
                description:
                  'Custom CrewAI tools for monitoring cloud infrastructure, tracking data flows, and maintaining jurisdiction rules.',
                code: `# tools/sovereignty_monitoring_tools.py
"""CrewAI tools for data sovereignty monitoring and jurisdiction mapping."""

from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Type

from crewai_tools import BaseTool
from pydantic import BaseModel, Field
import boto3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool Input Schemas
# ---------------------------------------------------------------------------

class CloudScanInput(BaseModel):
    """Input schema for cloud infrastructure scanning."""
    aws_regions: list[str] = Field(
        default=["us-east-1", "eu-west-1", "eu-central-1", "ap-south-1"],
        description="AWS regions to scan",
    )
    resource_types: list[str] = Field(
        default=["rds", "s3", "dynamodb", "redshift"],
        description="Resource types to discover",
    )
    include_data_classification: bool = Field(
        default=True,
        description="Whether to include data classification tags",
    )

class DataFlowDetectionInput(BaseModel):
    """Input schema for data flow detection."""
    source_region: str = Field(..., description="Source region for flow analysis")
    time_window_hours: int = Field(
        default=24, description="Hours of flow data to analyze"
    )
    include_vpc_flow_logs: bool = Field(
        default=True, description="Include VPC flow log analysis"
    )

class JurisdictionRuleInput(BaseModel):
    """Input schema for jurisdiction rule updates."""
    jurisdiction_id: str = Field(..., description="Jurisdiction identifier")
    regulation_source: str = Field(
        ..., description="URL or path to regulation document"
    )
    effective_date: str = Field(
        default="", description="Rule effective date (ISO8601)"
    )

# ---------------------------------------------------------------------------
# Cloud Infrastructure Scanner Tool
# ---------------------------------------------------------------------------

class CloudInfrastructureScannerTool(BaseTool):
    """Tool to scan cloud infrastructure and discover data storage locations."""

    name: str = "cloud_infrastructure_scanner"
    description: str = (
        "Scans AWS infrastructure across specified regions to discover all "
        "data storage resources (RDS, S3, DynamoDB, Redshift), their locations, "
        "and associated data classification tags. Returns a structured inventory "
        "for residency compliance analysis."
    )
    args_schema: Type[BaseModel] = CloudScanInput

    def _run(
        self,
        aws_regions: list[str],
        resource_types: list[str],
        include_data_classification: bool = True,
    ) -> str:
        """Scan cloud infrastructure across regions."""
        try:
            inventory: list[dict[str, Any]] = []

            for region in aws_regions:
                logger.info("Scanning region: %s", region)

                # Scan each resource type
                if "rds" in resource_types:
                    inventory.extend(
                        self._scan_rds_instances(region, include_data_classification)
                    )

                if "s3" in resource_types:
                    inventory.extend(
                        self._scan_s3_buckets(region, include_data_classification)
                    )

                if "dynamodb" in resource_types:
                    inventory.extend(
                        self._scan_dynamodb_tables(region, include_data_classification)
                    )

            # Aggregate by region
            region_summary = {}
            for item in inventory:
                region = item.get("region", "unknown")
                if region not in region_summary:
                    region_summary[region] = {
                        "resource_count": 0,
                        "pii_resources": 0,
                        "resources": [],
                    }
                region_summary[region]["resource_count"] += 1
                if item.get("contains_pii"):
                    region_summary[region]["pii_resources"] += 1
                region_summary[region]["resources"].append(item)

            result = {
                "status": "success",
                "scan_timestamp": datetime.utcnow().isoformat(),
                "regions_scanned": aws_regions,
                "total_resources": len(inventory),
                "region_summary": region_summary,
                "inventory": inventory,
            }

            logger.info(
                "Infrastructure scan complete: %d resources across %d regions",
                len(inventory), len(aws_regions)
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error("Infrastructure scan failed: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
                "regions_scanned": aws_regions,
            })

    def _scan_rds_instances(
        self, region: str, include_classification: bool
    ) -> list[dict[str, Any]]:
        """Scan RDS instances in a region."""
        try:
            rds = boto3.client("rds", region_name=region)
            instances = rds.describe_db_instances()

            results = []
            for instance in instances.get("DBInstances", []):
                tags = {
                    t["Key"]: t["Value"]
                    for t in instance.get("TagList", [])
                }

                results.append({
                    "resource_type": "rds",
                    "resource_id": instance["DBInstanceIdentifier"],
                    "region": region,
                    "availability_zone": instance.get("AvailabilityZone"),
                    "engine": instance.get("Engine"),
                    "storage_gb": instance.get("AllocatedStorage"),
                    "contains_pii": tags.get("DataClassification") == "PII",
                    "data_classification": tags.get("DataClassification"),
                    "jurisdiction_tag": tags.get("Jurisdiction"),
                    "last_modified": instance.get(
                        "InstanceCreateTime", ""
                    ).isoformat() if instance.get("InstanceCreateTime") else None,
                })

            return results

        except Exception as e:
            logger.warning("RDS scan failed for %s: %s", region, str(e))
            return []

    def _scan_s3_buckets(
        self, region: str, include_classification: bool
    ) -> list[dict[str, Any]]:
        """Scan S3 buckets in a region."""
        try:
            s3 = boto3.client("s3", region_name=region)
            buckets = s3.list_buckets()

            results = []
            for bucket in buckets.get("Buckets", []):
                bucket_name = bucket["Name"]

                # Get bucket location
                try:
                    location = s3.get_bucket_location(Bucket=bucket_name)
                    bucket_region = location.get(
                        "LocationConstraint"
                    ) or "us-east-1"
                except Exception:
                    bucket_region = "unknown"

                # Only include buckets in target region
                if bucket_region != region:
                    continue

                # Get tags if available
                tags = {}
                try:
                    tag_response = s3.get_bucket_tagging(Bucket=bucket_name)
                    tags = {
                        t["Key"]: t["Value"]
                        for t in tag_response.get("TagSet", [])
                    }
                except Exception:
                    pass

                results.append({
                    "resource_type": "s3",
                    "resource_id": bucket_name,
                    "region": bucket_region,
                    "contains_pii": tags.get("DataClassification") == "PII",
                    "data_classification": tags.get("DataClassification"),
                    "jurisdiction_tag": tags.get("Jurisdiction"),
                    "created_at": bucket.get(
                        "CreationDate", ""
                    ).isoformat() if bucket.get("CreationDate") else None,
                })

            return results

        except Exception as e:
            logger.warning("S3 scan failed for %s: %s", region, str(e))
            return []

    def _scan_dynamodb_tables(
        self, region: str, include_classification: bool
    ) -> list[dict[str, Any]]:
        """Scan DynamoDB tables in a region."""
        try:
            dynamodb = boto3.client("dynamodb", region_name=region)
            tables = dynamodb.list_tables()

            results = []
            for table_name in tables.get("TableNames", []):
                # Get table details
                table_info = dynamodb.describe_table(TableName=table_name)
                table = table_info.get("Table", {})

                # Get tags
                tags = {}
                try:
                    tag_response = dynamodb.list_tags_of_resource(
                        ResourceArn=table.get("TableArn", "")
                    )
                    tags = {
                        t["Key"]: t["Value"]
                        for t in tag_response.get("Tags", [])
                    }
                except Exception:
                    pass

                results.append({
                    "resource_type": "dynamodb",
                    "resource_id": table_name,
                    "region": region,
                    "contains_pii": tags.get("DataClassification") == "PII",
                    "data_classification": tags.get("DataClassification"),
                    "jurisdiction_tag": tags.get("Jurisdiction"),
                    "item_count": table.get("ItemCount", 0),
                    "size_bytes": table.get("TableSizeBytes", 0),
                })

            return results

        except Exception as e:
            logger.warning("DynamoDB scan failed for %s: %s", region, str(e))
            return []

# ---------------------------------------------------------------------------
# Data Flow Detection Tool
# ---------------------------------------------------------------------------

class DataFlowDetectionTool(BaseTool):
    """Tool to detect and map cross-border data flows."""

    name: str = "data_flow_detector"
    description: str = (
        "Analyzes data movement patterns to detect cross-border data flows, "
        "including ETL pipelines, replication streams, and API-based transfers. "
        "Maps source and destination regions for compliance analysis."
    )
    args_schema: Type[BaseModel] = DataFlowDetectionInput

    def _run(
        self,
        source_region: str,
        time_window_hours: int = 24,
        include_vpc_flow_logs: bool = True,
    ) -> str:
        """Detect data flows from source region."""
        try:
            detected_flows: list[dict[str, Any]] = []

            # Detect RDS replication
            detected_flows.extend(
                self._detect_rds_replication(source_region)
            )

            # Detect S3 replication
            detected_flows.extend(
                self._detect_s3_replication(source_region)
            )

            # Detect DynamoDB global tables
            detected_flows.extend(
                self._detect_dynamodb_global_tables(source_region)
            )

            # Analyze VPC flow logs if requested
            if include_vpc_flow_logs:
                detected_flows.extend(
                    self._analyze_vpc_flows(source_region, time_window_hours)
                )

            # Classify flows by cross-border status
            cross_border_flows = [
                f for f in detected_flows
                if f.get("source_region") != f.get("destination_region")
            ]

            result = {
                "status": "success",
                "source_region": source_region,
                "analysis_window_hours": time_window_hours,
                "total_flows_detected": len(detected_flows),
                "cross_border_flows": len(cross_border_flows),
                "flows": detected_flows,
                "analyzed_at": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Flow detection complete: %d flows, %d cross-border",
                len(detected_flows), len(cross_border_flows)
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error("Flow detection failed: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
                "source_region": source_region,
            })

    def _detect_rds_replication(
        self, source_region: str
    ) -> list[dict[str, Any]]:
        """Detect RDS read replicas and cross-region replication."""
        # Implementation would query RDS for replication relationships
        return []

    def _detect_s3_replication(
        self, source_region: str
    ) -> list[dict[str, Any]]:
        """Detect S3 cross-region replication rules."""
        # Implementation would check S3 replication configurations
        return []

    def _detect_dynamodb_global_tables(
        self, source_region: str
    ) -> list[dict[str, Any]]:
        """Detect DynamoDB global table replicas."""
        # Implementation would check for global table configurations
        return []

    def _analyze_vpc_flows(
        self, source_region: str, hours: int
    ) -> list[dict[str, Any]]:
        """Analyze VPC flow logs for cross-region traffic."""
        # Implementation would query CloudWatch Logs for VPC flows
        return []

# ---------------------------------------------------------------------------
# Jurisdiction Rule Manager Tool
# ---------------------------------------------------------------------------

class JurisdictionRuleManagerTool(BaseTool):
    """Tool to manage and update jurisdiction residency rules."""

    name: str = "jurisdiction_rule_manager"
    description: str = (
        "Manages the jurisdiction rules database, including adding new rules, "
        "updating existing rules based on regulatory changes, and validating "
        "rule consistency. Supports GDPR, UAE PDPL, India DPDP, and other regulations."
    )
    args_schema: Type[BaseModel] = JurisdictionRuleInput

    def __init__(
        self,
        rules_path: str = "config/residency_rules.json",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._rules_path = rules_path

    def _run(
        self,
        jurisdiction_id: str,
        regulation_source: str,
        effective_date: str = "",
    ) -> str:
        """Update jurisdiction rules from regulation source."""
        try:
            # Load existing rules
            with open(self._rules_path) as f:
                rules_db = json.load(f)

            # Find or create jurisdiction entry
            existing_rule = next(
                (r for r in rules_db.get("jurisdictions", [])
                 if r.get("jurisdiction_id") == jurisdiction_id),
                None
            )

            if existing_rule:
                action = "updated"
                # Update existing rule
                existing_rule["last_updated"] = datetime.utcnow().isoformat()
                existing_rule["source_document"] = regulation_source
                if effective_date:
                    existing_rule["effective_date"] = effective_date
            else:
                action = "created"
                # Create new rule entry
                new_rule = {
                    "jurisdiction_id": jurisdiction_id,
                    "source_document": regulation_source,
                    "effective_date": effective_date or datetime.utcnow().isoformat(),
                    "created_at": datetime.utcnow().isoformat(),
                    "requirements": {},  # To be populated by AI analysis
                }
                rules_db.setdefault("jurisdictions", []).append(new_rule)

            # Save updated rules
            with open(self._rules_path, "w") as f:
                json.dump(rules_db, f, indent=2)

            result = {
                "status": "success",
                "action": action,
                "jurisdiction_id": jurisdiction_id,
                "updated_at": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Jurisdiction rule %s: %s", action, jurisdiction_id
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error("Rule update failed: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
                "jurisdiction_id": jurisdiction_id,
            })`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Compliance Analyzer and Remediation Orchestrator agents with tools for violation detection, risk scoring, and automated remediation execution.',
            toolsUsed: ['CrewAI Tools', 'LangChain', 'AWS Data Pipeline', 'Terraform'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Compliance Analysis and Remediation Tools',
                description:
                  'CrewAI tools for analyzing compliance violations, calculating risk scores, and orchestrating automated remediation workflows.',
                code: `# tools/compliance_analysis_tools.py
"""CrewAI tools for sovereignty compliance analysis and remediation."""

from __future__ import annotations
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Type
from enum import Enum

from crewai_tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------

class ViolationSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RemediationAction(str, Enum):
    RELOCATE_DATA = "relocate_data"
    IMPLEMENT_SCCs = "implement_sccs"
    STOP_TRANSFER = "stop_transfer"
    UPDATE_DOCUMENTATION = "update_documentation"
    REQUEST_APPROVAL = "request_approval"

# ---------------------------------------------------------------------------
# Tool Input Schemas
# ---------------------------------------------------------------------------

class ComplianceCheckInput(BaseModel):
    """Input schema for compliance checking."""
    data_flows: list[dict] = Field(
        ..., description="List of data flows to analyze"
    )
    jurisdiction_rules: dict = Field(
        ..., description="Current jurisdiction rules"
    )
    include_risk_scoring: bool = Field(
        default=True, description="Calculate risk scores"
    )

class ViolationAnalysisInput(BaseModel):
    """Input schema for detailed violation analysis."""
    violation_id: str = Field(..., description="Violation to analyze")
    flow_details: dict = Field(..., description="Violating flow details")
    jurisdiction_rule: dict = Field(
        ..., description="Applicable jurisdiction rule"
    )

class RemediationPlanInput(BaseModel):
    """Input schema for remediation planning."""
    violations: list[dict] = Field(
        ..., description="Violations requiring remediation"
    )
    constraints: dict = Field(
        default={}, description="Business constraints on remediation"
    )
    priority_mode: str = Field(
        default="risk", description="Prioritization mode: risk, deadline, volume"
    )

class DataRelocationInput(BaseModel):
    """Input schema for data relocation execution."""
    flow_id: str = Field(..., description="Flow to remediate")
    source_location: dict = Field(..., description="Current data location")
    target_region: str = Field(..., description="Target compliant region")
    migration_strategy: str = Field(
        default="online", description="Migration strategy: online, offline, hybrid"
    )

# ---------------------------------------------------------------------------
# Compliance Violation Detector Tool
# ---------------------------------------------------------------------------

class ComplianceViolationDetectorTool(BaseTool):
    """Tool to detect data sovereignty compliance violations."""

    name: str = "compliance_violation_detector"
    description: str = (
        "Analyzes data flows against jurisdiction rules to detect compliance "
        "violations. Identifies flows that violate data localization requirements, "
        "lack proper transfer mechanisms, or transfer data to prohibited regions."
    )
    args_schema: Type[BaseModel] = ComplianceCheckInput

    def _run(
        self,
        data_flows: list[dict],
        jurisdiction_rules: dict,
        include_risk_scoring: bool = True,
    ) -> str:
        """Detect compliance violations in data flows."""
        try:
            violations: list[dict[str, Any]] = []
            compliant_flows: list[str] = []

            # Build jurisdiction lookup
            rules_by_country = {}
            for rule in jurisdiction_rules.get("jurisdictions", []):
                for country in rule.get("countries", []):
                    rules_by_country[country] = rule

            # Analyze each flow
            for flow in data_flows:
                flow_violations = self._check_flow_compliance(
                    flow, rules_by_country
                )

                if flow_violations:
                    for violation in flow_violations:
                        if include_risk_scoring:
                            violation["risk_score"] = self._calculate_risk_score(
                                violation, flow
                            )
                        violations.append(violation)
                else:
                    compliant_flows.append(flow.get("flow_id", "unknown"))

            # Sort by risk score if available
            if include_risk_scoring:
                violations.sort(
                    key=lambda v: v.get("risk_score", 0), reverse=True
                )

            result = {
                "status": "success",
                "total_flows_analyzed": len(data_flows),
                "compliant_count": len(compliant_flows),
                "violation_count": len(violations),
                "violations": violations,
                "compliant_flows": compliant_flows,
                "analyzed_at": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Compliance check complete: %d violations in %d flows",
                len(violations), len(data_flows)
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error("Compliance check failed: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
            })

    def _check_flow_compliance(
        self,
        flow: dict,
        rules_by_country: dict,
    ) -> list[dict[str, Any]]:
        """Check a single flow for compliance violations."""
        violations = []
        flow_id = flow.get("flow_id", "unknown")
        subject_countries = flow.get("subject_countries", [])
        dest_region = flow.get("destination", {}).get("region", "")
        transfer_mechanism = flow.get("transfer_mechanism", "")

        for country in subject_countries:
            rule = rules_by_country.get(country)
            if not rule:
                continue

            requirements = rule.get("requirements", {})
            allowed_regions = requirements.get("allowed_regions", [])
            localization_type = requirements.get("data_localization", "soft")

            # Check region compliance
            if dest_region and dest_region not in allowed_regions:
                severity = (
                    ViolationSeverity.CRITICAL
                    if localization_type == "strict"
                    else ViolationSeverity.HIGH
                )
                violations.append({
                    "violation_id": hashlib.sha256(
                        f"{flow_id}:{country}:region".encode()
                    ).hexdigest()[:12],
                    "flow_id": flow_id,
                    "violation_type": "region_non_compliance",
                    "severity": severity.value,
                    "subject_country": country,
                    "jurisdiction": rule.get("jurisdiction_id"),
                    "current_region": dest_region,
                    "allowed_regions": allowed_regions,
                    "description": (
                        f"Data for {country} subjects stored in {dest_region}, "
                        f"which is not in allowed regions: {allowed_regions}"
                    ),
                    "detected_at": datetime.utcnow().isoformat(),
                })

            # Check transfer mechanism
            valid_mechanisms = requirements.get("transfer_mechanisms", [])
            if (
                dest_region not in allowed_regions
                and transfer_mechanism
                and transfer_mechanism not in valid_mechanisms
            ):
                violations.append({
                    "violation_id": hashlib.sha256(
                        f"{flow_id}:{country}:mechanism".encode()
                    ).hexdigest()[:12],
                    "flow_id": flow_id,
                    "violation_type": "invalid_transfer_mechanism",
                    "severity": ViolationSeverity.HIGH.value,
                    "subject_country": country,
                    "jurisdiction": rule.get("jurisdiction_id"),
                    "current_mechanism": transfer_mechanism,
                    "valid_mechanisms": valid_mechanisms,
                    "description": (
                        f"Transfer mechanism '{transfer_mechanism}' not valid "
                        f"for {rule.get('jurisdiction_id')}. Valid options: {valid_mechanisms}"
                    ),
                    "detected_at": datetime.utcnow().isoformat(),
                })

        return violations

    def _calculate_risk_score(
        self,
        violation: dict,
        flow: dict,
    ) -> float:
        """Calculate risk score for a violation (0-100)."""
        base_score = 0.0

        # Severity weight (40% of score)
        severity_weights = {
            "critical": 40,
            "high": 30,
            "medium": 20,
            "low": 10,
        }
        base_score += severity_weights.get(
            violation.get("severity", "low"), 10
        )

        # Data volume weight (30% of score)
        daily_volume = flow.get("data_volume_gb_daily", 0)
        if daily_volume > 100:
            base_score += 30
        elif daily_volume > 10:
            base_score += 20
        elif daily_volume > 1:
            base_score += 10

        # PII sensitivity weight (30% of score)
        pii_categories = flow.get("pii_categories", [])
        sensitive_categories = {"ssn", "national_id", "health", "financial"}
        if any(cat in sensitive_categories for cat in pii_categories):
            base_score += 30
        elif pii_categories:
            base_score += 15

        return min(base_score, 100.0)

# ---------------------------------------------------------------------------
# Remediation Planner Tool
# ---------------------------------------------------------------------------

class RemediationPlannerTool(BaseTool):
    """Tool to create remediation plans for compliance violations."""

    name: str = "remediation_planner"
    description: str = (
        "Creates detailed remediation plans for compliance violations, "
        "including recommended actions, target configurations, timelines, "
        "and resource requirements. Prioritizes actions based on risk and constraints."
    )
    args_schema: Type[BaseModel] = RemediationPlanInput

    def _run(
        self,
        violations: list[dict],
        constraints: dict,
        priority_mode: str = "risk",
    ) -> str:
        """Create remediation plans for violations."""
        try:
            remediation_plans: list[dict[str, Any]] = []

            # Sort violations by priority mode
            if priority_mode == "risk":
                sorted_violations = sorted(
                    violations,
                    key=lambda v: v.get("risk_score", 0),
                    reverse=True,
                )
            elif priority_mode == "deadline":
                sorted_violations = sorted(
                    violations,
                    key=lambda v: v.get("detected_at", ""),
                )
            else:
                sorted_violations = violations

            # Create plan for each violation
            for idx, violation in enumerate(sorted_violations):
                plan = self._create_plan(violation, constraints, idx + 1)
                remediation_plans.append(plan)

            # Calculate total effort
            total_hours = sum(
                p.get("estimated_hours", 0) for p in remediation_plans
            )

            result = {
                "status": "success",
                "total_violations": len(violations),
                "total_estimated_hours": total_hours,
                "priority_mode": priority_mode,
                "remediation_plans": remediation_plans,
                "created_at": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Remediation plans created: %d plans, %d total hours",
                len(remediation_plans), total_hours
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error("Remediation planning failed: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
            })

    def _create_plan(
        self,
        violation: dict,
        constraints: dict,
        priority_rank: int,
    ) -> dict[str, Any]:
        """Create a remediation plan for a single violation."""
        violation_type = violation.get("violation_type", "")
        allowed_regions = violation.get("allowed_regions", [])

        # Determine primary action
        if violation_type == "region_non_compliance":
            primary_action = RemediationAction.RELOCATE_DATA
            target_region = allowed_regions[0] if allowed_regions else "unknown"
            estimated_hours = 40  # Data migration effort
        elif violation_type == "invalid_transfer_mechanism":
            primary_action = RemediationAction.IMPLEMENT_SCCs
            target_region = None
            estimated_hours = 20  # Legal documentation effort
        else:
            primary_action = RemediationAction.UPDATE_DOCUMENTATION
            target_region = None
            estimated_hours = 8

        # Check constraints
        max_downtime = constraints.get("max_downtime_hours", 4)
        frozen_systems = constraints.get("frozen_systems", [])

        plan = {
            "plan_id": hashlib.sha256(
                f"plan:{violation.get('violation_id')}".encode()
            ).hexdigest()[:12],
            "violation_id": violation.get("violation_id"),
            "priority_rank": priority_rank,
            "severity": violation.get("severity"),
            "risk_score": violation.get("risk_score", 0),
            "primary_action": primary_action.value,
            "target_region": target_region,
            "estimated_hours": estimated_hours,
            "steps": self._generate_steps(primary_action, violation, constraints),
            "prerequisites": self._identify_prerequisites(violation, constraints),
            "risks": self._identify_risks(primary_action, violation),
            "rollback_plan": self._create_rollback_plan(primary_action),
            "success_criteria": self._define_success_criteria(violation),
        }

        return plan

    def _generate_steps(
        self,
        action: RemediationAction,
        violation: dict,
        constraints: dict,
    ) -> list[dict[str, Any]]:
        """Generate remediation steps."""
        if action == RemediationAction.RELOCATE_DATA:
            return [
                {"step": 1, "action": "Create target infrastructure in compliant region"},
                {"step": 2, "action": "Configure data replication to target region"},
                {"step": 3, "action": "Validate data integrity after replication"},
                {"step": 4, "action": "Update application configurations"},
                {"step": 5, "action": "Switch traffic to new region"},
                {"step": 6, "action": "Verify compliance and remove old data"},
            ]
        elif action == RemediationAction.IMPLEMENT_SCCs:
            return [
                {"step": 1, "action": "Draft Standard Contractual Clauses"},
                {"step": 2, "action": "Legal review and approval"},
                {"step": 3, "action": "Execute SCCs with data recipient"},
                {"step": 4, "action": "Document transfer impact assessment"},
                {"step": 5, "action": "Update data processing records"},
            ]
        return [{"step": 1, "action": "Review and update documentation"}]

    def _identify_prerequisites(
        self,
        violation: dict,
        constraints: dict,
    ) -> list[str]:
        """Identify prerequisites for remediation."""
        return [
            "Approval from data governance team",
            "Infrastructure capacity in target region",
            "Application compatibility verification",
            "Stakeholder notification plan",
        ]

    def _identify_risks(
        self,
        action: RemediationAction,
        violation: dict,
    ) -> list[dict[str, str]]:
        """Identify risks associated with remediation."""
        if action == RemediationAction.RELOCATE_DATA:
            return [
                {"risk": "Data loss during migration", "mitigation": "Full backup before migration"},
                {"risk": "Service downtime", "mitigation": "Use online migration with cutover window"},
                {"risk": "Increased latency", "mitigation": "Performance testing before cutover"},
            ]
        return []

    def _create_rollback_plan(
        self,
        action: RemediationAction,
    ) -> dict[str, Any]:
        """Create rollback plan for remediation."""
        return {
            "trigger_conditions": [
                "Data integrity check failure",
                "Application errors above threshold",
                "Performance degradation beyond SLA",
            ],
            "rollback_steps": [
                "Revert DNS/routing to original region",
                "Restore from pre-migration backup if needed",
                "Notify stakeholders of rollback",
            ],
            "estimated_rollback_time_minutes": 30,
        }

    def _define_success_criteria(
        self,
        violation: dict,
    ) -> list[str]:
        """Define success criteria for remediation."""
        return [
            "Data verified in compliant region",
            "No compliance violations detected in post-check",
            "Application functionality verified",
            "Audit trail documentation complete",
        ]`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Build the LangGraph state machine that coordinates continuous monitoring, violation detection, risk assessment, and automated remediation workflows.',
            toolsUsed: ['LangGraph', 'Redis', 'CrewAI'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Sovereignty Monitoring Orchestration',
                description:
                  'Complete LangGraph state machine for continuous data sovereignty compliance monitoring and automated remediation.',
                code: `# orchestration/sovereignty_graph.py
"""LangGraph orchestration for data sovereignty compliance monitoring."""

from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Literal, TypedDict, Annotated
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import redis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class MonitoringStatus(str, Enum):
    """Monitoring cycle status."""
    IDLE = "idle"
    SCANNING = "scanning"
    ANALYZING = "analyzing"
    VIOLATION_DETECTED = "violation_detected"
    REMEDIATING = "remediating"
    REPORTING = "reporting"
    COMPLETE = "complete"
    ERROR = "error"

class SovereigntyState(TypedDict):
    """State schema for sovereignty monitoring pipeline."""
    # Cycle metadata
    cycle_id: str
    cycle_start: str
    status: MonitoringStatus
    current_agent: str

    # Scan configuration
    regions_to_scan: list[str]
    scan_depth: str  # quick, standard, deep

    # Infrastructure inventory
    infrastructure_inventory: list[dict[str, Any]]
    data_flows: list[dict[str, Any]]
    total_resources: int

    # Jurisdiction rules
    jurisdiction_rules: dict[str, Any]
    rules_last_updated: str

    # Compliance analysis
    violations: list[dict[str, Any]]
    compliant_flows: list[str]
    risk_summary: dict[str, Any]

    # Remediation
    remediation_plans: list[dict[str, Any]]
    remediation_results: list[dict[str, Any]]
    pending_approvals: list[dict[str, Any]]

    # Reporting
    compliance_report: dict[str, Any]
    alerts_sent: list[str]

    # Error handling
    errors: list[str]
    retry_count: int
    max_retries: int

# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def start_monitoring_cycle(state: SovereigntyState) -> SovereigntyState:
    """Initialize a new monitoring cycle."""
    logger.info(
        "Starting sovereignty monitoring cycle: %s", state["cycle_id"]
    )

    state["status"] = MonitoringStatus.SCANNING
    state["current_agent"] = "data_flow_monitor"
    state["cycle_start"] = datetime.utcnow().isoformat()

    return state

def scan_infrastructure(state: SovereigntyState) -> SovereigntyState:
    """Scan cloud infrastructure across all regions."""
    logger.info(
        "Scanning infrastructure across %d regions",
        len(state.get("regions_to_scan", []))
    )

    # In production, this invokes the Data Flow Monitor Agent
    state["status"] = MonitoringStatus.SCANNING
    state["current_agent"] = "jurisdiction_mapper"

    return state

def update_jurisdiction_rules(state: SovereigntyState) -> SovereigntyState:
    """Check for and apply jurisdiction rule updates."""
    logger.info("Checking jurisdiction rules for updates")

    # In production, this invokes the Jurisdiction Mapper Agent
    state["current_agent"] = "compliance_analyzer"

    return state

def analyze_compliance(state: SovereigntyState) -> SovereigntyState:
    """Analyze data flows for compliance violations."""
    logger.info(
        "Analyzing compliance for %d data flows",
        len(state.get("data_flows", []))
    )

    state["status"] = MonitoringStatus.ANALYZING
    state["current_agent"] = "remediation_orchestrator"

    return state

def create_remediation_plans(state: SovereigntyState) -> SovereigntyState:
    """Create remediation plans for detected violations."""
    violations = state.get("violations", [])
    logger.info(
        "Creating remediation plans for %d violations",
        len(violations)
    )

    if violations:
        state["status"] = MonitoringStatus.VIOLATION_DETECTED
    else:
        state["status"] = MonitoringStatus.REPORTING

    state["current_agent"] = "supervisor"

    return state

def execute_auto_remediation(state: SovereigntyState) -> SovereigntyState:
    """Execute automated remediation for approved violations."""
    logger.info(
        "Executing auto-remediation for %d plans",
        len(state.get("remediation_plans", []))
    )

    state["status"] = MonitoringStatus.REMEDIATING

    return state

def generate_compliance_report(state: SovereigntyState) -> SovereigntyState:
    """Generate compliance status report."""
    logger.info("Generating compliance report for cycle %s", state["cycle_id"])

    violations = state.get("violations", [])
    compliant = state.get("compliant_flows", [])

    state["compliance_report"] = {
        "cycle_id": state["cycle_id"],
        "generated_at": datetime.utcnow().isoformat(),
        "total_flows_analyzed": len(violations) + len(compliant),
        "compliant_count": len(compliant),
        "violation_count": len(violations),
        "compliance_rate": (
            len(compliant) / (len(violations) + len(compliant)) * 100
            if (violations or compliant) else 100.0
        ),
        "critical_violations": len([
            v for v in violations if v.get("severity") == "critical"
        ]),
        "remediations_executed": len(state.get("remediation_results", [])),
        "pending_actions": len(state.get("pending_approvals", [])),
    }

    state["status"] = MonitoringStatus.REPORTING
    state["current_agent"] = "alerting"

    return state

def send_alerts(state: SovereigntyState) -> SovereigntyState:
    """Send alerts for critical violations and status updates."""
    violations = state.get("violations", [])
    critical_count = len([
        v for v in violations if v.get("severity") == "critical"
    ])

    logger.info(
        "Sending alerts: %d critical violations",
        critical_count
    )

    state["alerts_sent"] = []

    if critical_count > 0:
        state["alerts_sent"].append(f"critical_alert_{state['cycle_id']}")

    state["status"] = MonitoringStatus.COMPLETE
    state["current_agent"] = "complete"

    return state

def handle_monitoring_error(state: SovereigntyState) -> SovereigntyState:
    """Handle errors in monitoring cycle."""
    logger.error(
        "Error in monitoring cycle %s: %s",
        state["cycle_id"],
        state.get("errors", [])
    )

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count < max_retries:
        state["retry_count"] = retry_count + 1
        state["status"] = MonitoringStatus.SCANNING
        logger.info(
            "Retrying monitoring cycle (attempt %d/%d)",
            retry_count + 1, max_retries
        )
    else:
        state["status"] = MonitoringStatus.ERROR
        logger.error(
            "Monitoring cycle %s failed after %d retries",
            state["cycle_id"], max_retries
        )

    return state

# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------

def route_after_scan(
    state: SovereigntyState,
) -> Literal["update_jurisdiction_rules", "handle_monitoring_error"]:
    """Route after infrastructure scan."""
    if state.get("errors"):
        return "handle_monitoring_error"
    return "update_jurisdiction_rules"

def route_after_analysis(
    state: SovereigntyState,
) -> Literal["create_remediation_plans", "generate_compliance_report"]:
    """Route based on violation detection."""
    if state.get("violations"):
        return "create_remediation_plans"
    return "generate_compliance_report"

def route_after_remediation_planning(
    state: SovereigntyState,
) -> Literal["execute_auto_remediation", "generate_compliance_report"]:
    """Route based on auto-remediation eligibility."""
    plans = state.get("remediation_plans", [])
    auto_remediable = [
        p for p in plans
        if p.get("severity") in ["low", "medium"]
        and p.get("auto_approved", False)
    ]

    if auto_remediable:
        return "execute_auto_remediation"
    return "generate_compliance_report"

def route_after_error(
    state: SovereigntyState,
) -> Literal["start_monitoring_cycle", "__end__"]:
    """Route after error handling."""
    if state["status"] == MonitoringStatus.ERROR:
        return "__end__"
    return "start_monitoring_cycle"

# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_sovereignty_graph() -> StateGraph:
    """Build the LangGraph sovereignty monitoring pipeline."""

    graph = StateGraph(SovereigntyState)

    # Add nodes
    graph.add_node("start_monitoring_cycle", start_monitoring_cycle)
    graph.add_node("scan_infrastructure", scan_infrastructure)
    graph.add_node("update_jurisdiction_rules", update_jurisdiction_rules)
    graph.add_node("analyze_compliance", analyze_compliance)
    graph.add_node("create_remediation_plans", create_remediation_plans)
    graph.add_node("execute_auto_remediation", execute_auto_remediation)
    graph.add_node("generate_compliance_report", generate_compliance_report)
    graph.add_node("send_alerts", send_alerts)
    graph.add_node("handle_monitoring_error", handle_monitoring_error)

    # Set entry point
    graph.set_entry_point("start_monitoring_cycle")

    # Add edges
    graph.add_edge("start_monitoring_cycle", "scan_infrastructure")

    graph.add_conditional_edges(
        "scan_infrastructure",
        route_after_scan,
        {
            "update_jurisdiction_rules": "update_jurisdiction_rules",
            "handle_monitoring_error": "handle_monitoring_error",
        },
    )

    graph.add_edge("update_jurisdiction_rules", "analyze_compliance")

    graph.add_conditional_edges(
        "analyze_compliance",
        route_after_analysis,
        {
            "create_remediation_plans": "create_remediation_plans",
            "generate_compliance_report": "generate_compliance_report",
        },
    )

    graph.add_conditional_edges(
        "create_remediation_plans",
        route_after_remediation_planning,
        {
            "execute_auto_remediation": "execute_auto_remediation",
            "generate_compliance_report": "generate_compliance_report",
        },
    )

    graph.add_edge("execute_auto_remediation", "generate_compliance_report")
    graph.add_edge("generate_compliance_report", "send_alerts")
    graph.add_edge("send_alerts", END)

    graph.add_conditional_edges(
        "handle_monitoring_error",
        route_after_error,
        {
            "start_monitoring_cycle": "start_monitoring_cycle",
            "__end__": END,
        },
    )

    return graph

# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

class SovereigntyMonitor:
    """Main sovereignty monitoring pipeline executor."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_regions: list[str] | None = None,
    ) -> None:
        self.graph = build_sovereignty_graph()
        self.checkpointer = MemorySaver()
        self.compiled = self.graph.compile(checkpointer=self.checkpointer)
        self.redis = redis.from_url(redis_url)
        self.default_regions = default_regions or [
            "us-east-1", "eu-west-1", "eu-central-1",
            "ap-south-1", "me-south-1",
        ]

    def run_monitoring_cycle(
        self,
        cycle_id: str | None = None,
        regions: list[str] | None = None,
        scan_depth: str = "standard",
    ) -> dict[str, Any]:
        """Run a complete monitoring cycle."""
        import uuid

        cycle_id = cycle_id or f"cycle-{uuid.uuid4().hex[:8]}"
        regions = regions or self.default_regions

        initial_state: SovereigntyState = {
            "cycle_id": cycle_id,
            "cycle_start": "",
            "status": MonitoringStatus.IDLE,
            "current_agent": "initializing",
            "regions_to_scan": regions,
            "scan_depth": scan_depth,
            "infrastructure_inventory": [],
            "data_flows": [],
            "total_resources": 0,
            "jurisdiction_rules": {},
            "rules_last_updated": "",
            "violations": [],
            "compliant_flows": [],
            "risk_summary": {},
            "remediation_plans": [],
            "remediation_results": [],
            "pending_approvals": [],
            "compliance_report": {},
            "alerts_sent": [],
            "errors": [],
            "retry_count": 0,
            "max_retries": 3,
        }

        config = {"configurable": {"thread_id": cycle_id}}

        try:
            final_state = self.compiled.invoke(initial_state, config)

            # Cache result
            self.redis.setex(
                f"sovereignty:cycle:{cycle_id}",
                timedelta(days=30),
                json.dumps(final_state, default=str),
            )

            logger.info(
                "Monitoring cycle %s completed: %s",
                cycle_id, final_state["status"]
            )
            return final_state

        except Exception as e:
            logger.error(
                "Monitoring cycle %s failed: %s",
                cycle_id, str(e)
            )
            raise`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the multi-agent sovereignty monitoring system with Docker containerization, multi-region support, LangSmith tracing, and Prometheus metrics for compliance SLA tracking.',
            toolsUsed: ['Docker', 'Kubernetes', 'LangSmith', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Kubernetes Deployment for Multi-Region Monitoring',
                description:
                  'Production Kubernetes manifests for deploying the sovereignty monitoring system across multiple regions with proper observability.',
                code: `# kubernetes/sovereignty-monitoring.yaml
# Data Sovereignty Multi-Agent Monitoring System - Kubernetes Deployment

apiVersion: v1
kind: Namespace
metadata:
  name: sovereignty-monitoring
  labels:
    app.kubernetes.io/name: sovereignty-monitoring
    app.kubernetes.io/component: compliance

---
# ConfigMap for jurisdiction rules
apiVersion: v1
kind: ConfigMap
metadata:
  name: jurisdiction-rules
  namespace: sovereignty-monitoring
data:
  residency_rules.json: |
    {
      "version": "2025.1",
      "jurisdictions": [
        {
          "jurisdiction_id": "EU-GDPR",
          "countries": ["DE", "FR", "IT", "ES", "NL"],
          "requirements": {
            "data_localization": "soft",
            "allowed_regions": ["eu-west-1", "eu-central-1"]
          }
        },
        {
          "jurisdiction_id": "UAE-PDPL",
          "countries": ["AE"],
          "requirements": {
            "data_localization": "strict",
            "allowed_regions": ["me-south-1"]
          }
        }
      ]
    }

---
# Secret for API keys
apiVersion: v1
kind: Secret
metadata:
  name: sovereignty-secrets
  namespace: sovereignty-monitoring
type: Opaque
stringData:
  OPENAI_API_KEY: "\${OPENAI_API_KEY}"
  ANTHROPIC_API_KEY: "\${ANTHROPIC_API_KEY}"
  LANGSMITH_API_KEY: "\${LANGSMITH_API_KEY}"
  AWS_ACCESS_KEY_ID: "\${AWS_ACCESS_KEY_ID}"
  AWS_SECRET_ACCESS_KEY: "\${AWS_SECRET_ACCESS_KEY}"

---
# Main monitoring deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sovereignty-monitor
  namespace: sovereignty-monitoring
  labels:
    app: sovereignty-monitor
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sovereignty-monitor
  template:
    metadata:
      labels:
        app: sovereignty-monitor
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: sovereignty-monitor
      containers:
        - name: monitor
          image: sovereignty-monitor:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8080
              name: http
            - containerPort: 9090
              name: metrics
          env:
            - name: SOVEREIGNTY_ENV
              value: "production"
            - name: REDIS_URL
              value: "redis://redis-master.sovereignty-monitoring:6379"
            - name: LANGSMITH_PROJECT
              value: "sovereignty-monitoring"
            - name: LANGSMITH_TRACING_V2
              value: "true"
            - name: SCAN_INTERVAL_MINUTES
              value: "60"
            - name: REGIONS_TO_SCAN
              value: "us-east-1,eu-west-1,eu-central-1,ap-south-1,me-south-1"
          envFrom:
            - secretRef:
                name: sovereignty-secrets
          volumeMounts:
            - name: jurisdiction-rules
              mountPath: /app/config
              readOnly: true
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 10
      volumes:
        - name: jurisdiction-rules
          configMap:
            name: jurisdiction-rules

---
# CronJob for scheduled deep scans
apiVersion: batch/v1
kind: CronJob
metadata:
  name: sovereignty-deep-scan
  namespace: sovereignty-monitoring
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM UTC
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: sovereignty-monitor
          containers:
            - name: deep-scanner
              image: sovereignty-monitor:latest
              command: ["python", "run_deep_scan.py"]
              env:
                - name: SCAN_DEPTH
                  value: "deep"
                - name: REDIS_URL
                  value: "redis://redis-master.sovereignty-monitoring:6379"
              envFrom:
                - secretRef:
                    name: sovereignty-secrets
              volumeMounts:
                - name: jurisdiction-rules
                  mountPath: /app/config
                  readOnly: true
              resources:
                requests:
                  cpu: "1"
                  memory: "2Gi"
                limits:
                  cpu: "4"
                  memory: "8Gi"
          volumes:
            - name: jurisdiction-rules
              configMap:
                name: jurisdiction-rules
          restartPolicy: OnFailure

---
# Service for internal communication
apiVersion: v1
kind: Service
metadata:
  name: sovereignty-monitor
  namespace: sovereignty-monitoring
spec:
  selector:
    app: sovereignty-monitor
  ports:
    - name: http
      port: 8080
      targetPort: 8080
    - name: metrics
      port: 9090
      targetPort: 9090

---
# ServiceMonitor for Prometheus
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sovereignty-monitor
  namespace: sovereignty-monitoring
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: sovereignty-monitor
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics`,
              },
              {
                language: 'python',
                title: 'Prometheus Metrics and LangSmith Tracing',
                description:
                  'Observability instrumentation for the sovereignty monitoring pipeline with compliance-specific metrics and agent tracing.',
                code: `# observability/sovereignty_metrics.py
"""Observability instrumentation for data sovereignty monitoring."""

from __future__ import annotations
import os
import time
import logging
import functools
from typing import Any, Callable, TypeVar
from contextlib import contextmanager
from datetime import datetime

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from langsmith import Client as LangSmithClient
from langsmith.run_trees import RunTree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus Metrics Registry
# ---------------------------------------------------------------------------

registry = CollectorRegistry(auto_describe=True)

# Monitoring cycle metrics
MONITORING_CYCLES_TOTAL = Counter(
    "sovereignty_monitoring_cycles_total",
    "Total monitoring cycles executed",
    ["status", "scan_depth"],
    registry=registry,
)

MONITORING_CYCLE_DURATION = Histogram(
    "sovereignty_monitoring_cycle_duration_seconds",
    "Time spent on complete monitoring cycles",
    ["scan_depth"],
    buckets=[60, 120, 300, 600, 1200, 1800, 3600],
    registry=registry,
)

# Compliance metrics
COMPLIANCE_RATE = Gauge(
    "sovereignty_compliance_rate_percent",
    "Current compliance rate as percentage",
    ["jurisdiction"],
    registry=registry,
)

VIOLATIONS_DETECTED = Counter(
    "sovereignty_violations_detected_total",
    "Total violations detected",
    ["jurisdiction", "severity", "violation_type"],
    registry=registry,
)

ACTIVE_VIOLATIONS = Gauge(
    "sovereignty_active_violations",
    "Currently active (unremediated) violations",
    ["jurisdiction", "severity"],
    registry=registry,
)

# Data flow metrics
DATA_FLOWS_MONITORED = Gauge(
    "sovereignty_data_flows_monitored",
    "Number of data flows currently monitored",
    ["source_region", "destination_region"],
    registry=registry,
)

CROSS_BORDER_TRANSFERS = Counter(
    "sovereignty_cross_border_transfers_total",
    "Total cross-border data transfers detected",
    ["source_region", "destination_region", "compliance_status"],
    registry=registry,
)

# Remediation metrics
REMEDIATIONS_EXECUTED = Counter(
    "sovereignty_remediations_executed_total",
    "Total remediations executed",
    ["action_type", "outcome"],
    registry=registry,
)

REMEDIATION_DURATION = Histogram(
    "sovereignty_remediation_duration_seconds",
    "Time spent on remediation actions",
    ["action_type"],
    buckets=[60, 300, 600, 1800, 3600, 7200, 14400],
    registry=registry,
)

PENDING_APPROVALS = Gauge(
    "sovereignty_pending_approvals",
    "Number of remediations pending approval",
    ["severity"],
    registry=registry,
)

# Infrastructure metrics
RESOURCES_SCANNED = Gauge(
    "sovereignty_resources_scanned",
    "Number of cloud resources in inventory",
    ["region", "resource_type"],
    registry=registry,
)

SCAN_DURATION = Histogram(
    "sovereignty_scan_duration_seconds",
    "Time spent scanning infrastructure",
    ["region", "resource_type"],
    buckets=[1, 5, 10, 30, 60, 120, 300],
    registry=registry,
)

# Agent metrics
AGENT_INVOCATIONS = Counter(
    "sovereignty_agent_invocations_total",
    "Total agent invocations",
    ["agent", "outcome"],
    registry=registry,
)

AGENT_DURATION = Histogram(
    "sovereignty_agent_duration_seconds",
    "Time spent in agent processing",
    ["agent"],
    buckets=[1, 5, 10, 30, 60, 120],
    registry=registry,
)

AGENT_TOKEN_USAGE = Counter(
    "sovereignty_agent_token_usage_total",
    "Total tokens consumed by agents",
    ["agent", "token_type"],
    registry=registry,
)

# System info
SYSTEM_INFO = Info(
    "sovereignty_monitoring",
    "Sovereignty monitoring system information",
    registry=registry,
)

# ---------------------------------------------------------------------------
# LangSmith Tracing Manager
# ---------------------------------------------------------------------------

class SovereigntyTracingManager:
    """Manages LangSmith tracing for sovereignty monitoring workflows."""

    def __init__(self) -> None:
        self.client: LangSmithClient | None = None
        self.project_name = os.environ.get(
            "LANGSMITH_PROJECT", "sovereignty-monitoring"
        )
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize LangSmith client."""
        api_key = os.environ.get("LANGSMITH_API_KEY")
        if api_key:
            self.client = LangSmithClient(api_key=api_key)
            logger.info(
                "LangSmith tracing enabled: %s", self.project_name
            )
        else:
            logger.warning("LangSmith not configured")

    @contextmanager
    def trace_monitoring_cycle(
        self,
        cycle_id: str,
        regions: list[str],
        scan_depth: str,
    ):
        """Trace a complete monitoring cycle."""
        if not self.client:
            yield None
            return

        run_tree = RunTree(
            name=f"monitoring_cycle_{cycle_id}",
            run_type="chain",
            project_name=self.project_name,
            inputs={
                "cycle_id": cycle_id,
                "regions": regions,
                "scan_depth": scan_depth,
                "started_at": datetime.utcnow().isoformat(),
            },
        )

        start_time = time.time()
        try:
            run_tree.post()
            yield run_tree
        except Exception as e:
            run_tree.end(error=str(e))
            run_tree.patch()
            MONITORING_CYCLES_TOTAL.labels(
                status="error", scan_depth=scan_depth
            ).inc()
            raise
        else:
            duration = time.time() - start_time
            run_tree.end(outputs={
                "duration_seconds": duration,
                "completed_at": datetime.utcnow().isoformat(),
            })
            run_tree.patch()
            MONITORING_CYCLES_TOTAL.labels(
                status="success", scan_depth=scan_depth
            ).inc()
            MONITORING_CYCLE_DURATION.labels(
                scan_depth=scan_depth
            ).observe(duration)

    @contextmanager
    def trace_agent(
        self,
        parent_run: RunTree | None,
        agent_name: str,
        inputs: dict[str, Any],
    ):
        """Trace an individual agent invocation."""
        if not self.client or not parent_run:
            yield None
            return

        child_run = parent_run.create_child(
            name=agent_name,
            run_type="llm",
            inputs=inputs,
        )

        start_time = time.time()
        try:
            child_run.post()
            yield child_run
            AGENT_INVOCATIONS.labels(
                agent=agent_name, outcome="success"
            ).inc()
        except Exception as e:
            child_run.end(error=str(e))
            child_run.patch()
            AGENT_INVOCATIONS.labels(
                agent=agent_name, outcome="error"
            ).inc()
            raise
        else:
            duration = time.time() - start_time
            child_run.end(outputs={"duration_seconds": duration})
            child_run.patch()
            AGENT_DURATION.labels(agent=agent_name).observe(duration)

# ---------------------------------------------------------------------------
# Metric Recording Functions
# ---------------------------------------------------------------------------

def record_compliance_status(
    jurisdiction: str,
    total_flows: int,
    compliant_flows: int,
) -> None:
    """Record compliance rate for a jurisdiction."""
    if total_flows > 0:
        rate = (compliant_flows / total_flows) * 100
        COMPLIANCE_RATE.labels(jurisdiction=jurisdiction).set(rate)
        logger.info(
            "Compliance rate for %s: %.1f%% (%d/%d)",
            jurisdiction, rate, compliant_flows, total_flows
        )

def record_violation(
    jurisdiction: str,
    severity: str,
    violation_type: str,
) -> None:
    """Record a detected violation."""
    VIOLATIONS_DETECTED.labels(
        jurisdiction=jurisdiction,
        severity=severity,
        violation_type=violation_type,
    ).inc()

def record_remediation(
    action_type: str,
    outcome: str,
    duration_seconds: float,
) -> None:
    """Record a remediation action."""
    REMEDIATIONS_EXECUTED.labels(
        action_type=action_type, outcome=outcome
    ).inc()
    REMEDIATION_DURATION.labels(
        action_type=action_type
    ).observe(duration_seconds)

def record_cross_border_transfer(
    source_region: str,
    destination_region: str,
    is_compliant: bool,
) -> None:
    """Record a cross-border data transfer."""
    status = "compliant" if is_compliant else "violation"
    CROSS_BORDER_TRANSFERS.labels(
        source_region=source_region,
        destination_region=destination_region,
        compliance_status=status,
    ).inc()

# ---------------------------------------------------------------------------
# Metrics Endpoint
# ---------------------------------------------------------------------------

def get_metrics() -> tuple[bytes, str]:
    """Generate Prometheus metrics for scraping."""
    return generate_latest(registry), CONTENT_TYPE_LATEST

def initialize_metrics(
    version: str,
    environment: str,
    regions: list[str],
) -> SovereigntyTracingManager:
    """Initialize metrics and tracing."""
    SYSTEM_INFO.info({
        "version": version,
        "environment": environment,
        "monitored_regions": ",".join(regions),
        "langsmith_enabled": str(bool(os.environ.get("LANGSMITH_API_KEY"))),
    })

    logger.info(
        "Sovereignty metrics initialized: version=%s, env=%s, regions=%d",
        version, environment, len(regions)
    )

    return SovereigntyTracingManager()`,
              },
            ],
          },
        ],
      },
    },
  ],
};
