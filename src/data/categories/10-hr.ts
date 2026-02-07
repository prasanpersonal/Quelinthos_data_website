import type { Category } from '../types.ts';

export const hrCategory: Category = {
  id: 'hr',
  number: 10,
  title: 'Human Resources',
  shortTitle: 'HR',
  description:
    'Fix people analytics that mislead leadership and unify employee data scattered across HRIS, payroll, and benefits systems.',
  icon: 'Users',
  accentColor: 'neon-blue',
  painPoints: [
    // ── Pain Point 1: HR Metric Validity Crisis ──────────────────────────
    {
      id: 'hr-metric-validity',
      number: 1,
      title: 'HR Metric Validity Crisis',
      subtitle: 'People Analytics That Mislead Leadership',
      summary:
        'Your turnover rate calculation excludes contractors, your engagement scores have 30% response bias, and headcount reports differ between HR and Finance by 15%.',
      tags: ['hr-analytics', 'people-data', 'metrics'],
      metrics: {
        annualCostRange: '$300K - $1.5M',
        roi: '5x',
        paybackPeriod: '3-4 months',
        investmentRange: '$60K - $120K',
      },
      price: {
        present: {
          title: 'Current State of HR Metrics',
          description:
            'People analytics dashboards report numbers that do not withstand basic scrutiny, yet leadership relies on them for headcount planning and retention strategy.',
          bullets: [
            'Turnover rate calculations exclude contractors and contingent workers, understating actual attrition by 20-35%',
            'Engagement survey scores carry 30% non-response bias — disengaged employees rarely participate',
            'Headcount reports from HR and Finance differ by 10-15% due to inconsistent definitions of "active employee"',
            'Time-to-fill metrics reset when requisitions are reposted, masking chronic hiring bottlenecks',
          ],
          severity: 'high',
        },
        root: {
          title: 'Why HR Metrics Are Unreliable',
          description:
            'Metric definitions live in tribal knowledge rather than standardized SQL logic, and each HRIS report applies its own filtering assumptions.',
          bullets: [
            'No single-source-of-truth definition for "employee" — each system (HRIS, payroll, badge access) counts differently',
            'Turnover formulas are embedded in spreadsheets maintained by individual analysts with no version control',
            'Survey platforms report raw scores without adjusting for response-rate bias or department-size weighting',
            'Finance uses payroll headcount while HR uses HRIS active records, creating a permanent reconciliation gap',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Business Impact of Misleading People Data',
          description:
            'Flawed metrics lead directly to bad workforce decisions — over-hiring in some departments while under-investing in retention where it matters most.',
          bullets: [
            'Leadership approved a costly retention program targeting the wrong departments because turnover data excluded contract-to-hire conversions',
            'Board-reported headcount was challenged by auditors, requiring 3 weeks of emergency reconciliation',
            'Engagement "improvements" were illusory — scores rose because disengaged employees stopped responding entirely',
            'Compensation benchmarking used incorrect tenure calculations, resulting in $400K in misallocated merit increases',
          ],
          severity: 'high',
        },
        cost: {
          title: 'Cost of Inaccurate HR Analytics',
          description:
            'The direct and indirect costs of unreliable people metrics span misallocated budgets, audit risk, and eroded credibility with the C-suite.',
          bullets: [
            'HR analysts spend 15-20 hours per month manually reconciling headcount across systems',
            'Misguided retention programs waste $150K-$500K annually by targeting the wrong populations',
            'Board-reporting errors create compliance and reputational risk during audits and IPO preparation',
          ],
          severity: 'medium',
        },
        expectedReturn: {
          title: 'Value of Standardized People Analytics',
          description:
            'Implementing governed metric definitions and automated reconciliation restores trust in HR data and enables confident workforce planning.',
          bullets: [
            'Single headcount definition eliminates HR-Finance reconciliation — saving 200+ analyst hours per year',
            'Bias-adjusted engagement scores reveal true hotspots, improving retention program ROI by 3-5x',
            'Auditable metric lineage reduces board-reporting preparation from weeks to hours',
            'Standardized turnover calculations enable meaningful benchmarking against industry data for the first time',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Build standardized SQL views for core HR metrics with explicit inclusion/exclusion rules, then automate headcount reconciliation between HR and Finance systems.',
        prerequisites: [
          'Read access to HRIS database (BambooHR, Workday, or similar)',
          'Read access to payroll system (ADP, Paychex, or similar)',
          'Python 3.9+ with pandas and sqlalchemy',
          'A shared analytics database (PostgreSQL or Snowflake)',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduled reconciliation',
          'Slack incoming webhook URL for alerting',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Standardized HR Metrics Views',
            description:
              'Create SQL views that codify official definitions for headcount, turnover, and tenure — eliminating spreadsheet-based calculations.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Governed Headcount & Turnover Views',
                description:
                  'SQL views that enforce a single definition of active headcount and calculate turnover consistently across all reporting.',
                code: `-- Canonical headcount: the ONE definition everyone uses
CREATE OR REPLACE VIEW hr_analytics.v_active_headcount AS
SELECT
    e.employee_id,
    e.full_name,
    e.department,
    e.location,
    e.hire_date,
    e.employment_type,          -- 'FTE', 'contractor', 'intern'
    e.payroll_status,
    CASE
        WHEN e.employment_type IN ('FTE', 'contractor')
             AND e.termination_date IS NULL
             AND e.payroll_status = 'active'
        THEN TRUE
        ELSE FALSE
    END AS is_active,
    DATE_PART('year', AGE(CURRENT_DATE, e.hire_date)) AS tenure_years,
    e.cost_center,
    cc.finance_dept_name          -- join to Finance mapping
FROM hris.employees e
LEFT JOIN finance.cost_center_map cc
    ON e.cost_center = cc.cost_center_code
WHERE e.hire_date <= CURRENT_DATE;

-- Monthly turnover rate with contractor inclusion flag
CREATE OR REPLACE VIEW hr_analytics.v_monthly_turnover AS
WITH monthly_terms AS (
    SELECT
        DATE_TRUNC('month', termination_date) AS term_month,
        employment_type,
        department,
        COUNT(*) AS terminations
    FROM hris.employees
    WHERE termination_date IS NOT NULL
    GROUP BY 1, 2, 3
),
monthly_headcount AS (
    SELECT
        DATE_TRUNC('month', snapshot_date) AS hc_month,
        department,
        COUNT(*) AS avg_headcount
    FROM hr_analytics.daily_headcount_snapshot
    GROUP BY 1, 2
)
SELECT
    t.term_month,
    t.department,
    t.employment_type,
    t.terminations,
    h.avg_headcount,
    ROUND(t.terminations::NUMERIC / NULLIF(h.avg_headcount, 0) * 100, 2)
        AS turnover_pct
FROM monthly_terms t
JOIN monthly_headcount h
    ON t.term_month = h.hc_month AND t.department = h.department;`,
              },
              {
                language: 'sql',
                title: 'Engagement Score Bias Adjustment',
                description:
                  'Adjust raw engagement survey scores for non-response bias by weighting department scores against their response rates.',
                code: `-- Bias-adjusted engagement scores
CREATE OR REPLACE VIEW hr_analytics.v_engagement_adjusted AS
WITH survey_stats AS (
    SELECT
        s.survey_period,
        s.department,
        COUNT(DISTINCT s.employee_id)   AS respondents,
        hc.active_count                 AS dept_headcount,
        AVG(s.overall_score)            AS raw_avg_score,
        ROUND(
            COUNT(DISTINCT s.employee_id)::NUMERIC
            / NULLIF(hc.active_count, 0) * 100, 1
        ) AS response_rate_pct
    FROM surveys.engagement_responses s
    JOIN (
        SELECT department, COUNT(*) AS active_count
        FROM hr_analytics.v_active_headcount
        WHERE is_active = TRUE
        GROUP BY department
    ) hc ON s.department = hc.department
    GROUP BY s.survey_period, s.department, hc.active_count
)
SELECT
    survey_period,
    department,
    respondents,
    dept_headcount,
    response_rate_pct,
    raw_avg_score,
    -- Penalise low-response departments: assume non-respondents
    -- score at the 25th percentile of respondent scores
    ROUND(
        raw_avg_score * (response_rate_pct / 100.0)
        + (SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY overall_score)
           FROM surveys.engagement_responses) * (1 - response_rate_pct / 100.0),
        2
    ) AS adjusted_score,
    CASE
        WHEN response_rate_pct < 50 THEN 'LOW_CONFIDENCE'
        WHEN response_rate_pct < 75 THEN 'MODERATE_CONFIDENCE'
        ELSE 'HIGH_CONFIDENCE'
    END AS confidence_flag
FROM survey_stats;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Automated Headcount Reconciliation',
            description:
              'Python pipeline that pulls headcount from both HRIS and payroll, identifies discrepancies, and generates an exception report for rapid resolution.',
            codeSnippets: [
              {
                language: 'python',
                title: 'HRIS-to-Payroll Headcount Reconciler',
                description:
                  'Compares active employee lists between HRIS and payroll, flags mismatches, and writes a reconciliation report to the analytics database.',
                code: `"""Automated headcount reconciliation: HRIS vs Payroll."""
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import date

engine = create_engine("postgresql://analytics:***@db-host:5432/people_analytics")

def pull_hris_headcount() -> pd.DataFrame:
    query = text("""
        SELECT employee_id, full_name, department, employment_type,
               hire_date, termination_date, payroll_status
        FROM hris.employees
        WHERE termination_date IS NULL
          AND payroll_status = 'active'
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

def pull_payroll_headcount() -> pd.DataFrame:
    query = text("""
        SELECT employee_id, legal_name, dept_code, pay_status,
               pay_frequency, last_pay_date
        FROM payroll.active_employees
        WHERE pay_status IN ('active', 'leave')
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

def reconcile(hris: pd.DataFrame, payroll: pd.DataFrame) -> pd.DataFrame:
    merged = hris.merge(
        payroll, on="employee_id", how="outer", indicator=True, suffixes=("_hris", "_payroll")
    )
    discrepancies = merged[merged["_merge"] != "both"].copy()
    discrepancies["discrepancy_type"] = discrepancies["_merge"].map({
        "left_only": "IN_HRIS_NOT_PAYROLL",
        "right_only": "IN_PAYROLL_NOT_HRIS",
    })
    discrepancies["detected_date"] = date.today()
    return discrepancies

def run_reconciliation():
    hris_df = pull_hris_headcount()
    payroll_df = pull_payroll_headcount()
    issues = reconcile(hris_df, payroll_df)

    print(f"HRIS active: {len(hris_df)} | Payroll active: {len(payroll_df)}")
    print(f"Discrepancies found: {len(issues)}")

    if not issues.empty:
        issues.to_sql(
            "headcount_reconciliation_log",
            engine,
            schema="hr_analytics",
            if_exists="append",
            index=False,
        )
    return issues

if __name__ == "__main__":
    run_reconciliation()`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Data quality assertion queries and automated pytest-based pipeline validation ensure headcount reconciliation accuracy before metrics reach dashboards.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Headcount Reconciliation Data Quality Assertions',
                description:
                  'SQL assertions that validate headcount accuracy, null completeness, and turnover calculation integrity across the HR analytics layer.',
                code: `-- Assertion 1: No NULL employee_id in active headcount view
SELECT
    'active_headcount_null_employee_id' AS assertion,
    COUNT(*) AS violations
FROM hr_analytics.v_active_headcount
WHERE employee_id IS NULL
   OR full_name IS NULL
   OR department IS NULL;
-- Expected: 0 violations

-- Assertion 2: Headcount difference between HRIS and payroll < 2%
WITH hris_count AS (
    SELECT COUNT(*) AS hris_total
    FROM hr_analytics.v_active_headcount
    WHERE is_active = TRUE
),
payroll_count AS (
    SELECT COUNT(*) AS payroll_total
    FROM payroll.active_employees
    WHERE pay_status IN ('active', 'leave')
)
SELECT
    'hris_payroll_gap_pct' AS assertion,
    hris_total,
    payroll_total,
    ROUND(
        ABS(hris_total - payroll_total)::NUMERIC
        / NULLIF(GREATEST(hris_total, payroll_total), 0) * 100, 2
    ) AS gap_pct,
    CASE
        WHEN ABS(hris_total - payroll_total)::NUMERIC
             / NULLIF(GREATEST(hris_total, payroll_total), 0) * 100 > 2.0
        THEN 'FAIL'
        ELSE 'PASS'
    END AS result
FROM hris_count, payroll_count;

-- Assertion 3: Turnover rate sanity — no department > 25% monthly
SELECT
    'excessive_monthly_turnover' AS assertion,
    department,
    term_month,
    turnover_pct
FROM hr_analytics.v_monthly_turnover
WHERE turnover_pct > 25.0
  AND term_month >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '6 months');
-- Expected: 0 rows

-- Assertion 4: Engagement scores within valid range [1.0, 5.0]
SELECT
    'engagement_score_out_of_range' AS assertion,
    COUNT(*) AS violations
FROM hr_analytics.v_engagement_adjusted
WHERE adjusted_score < 1.0
   OR adjusted_score > 5.0
   OR raw_avg_score < 1.0
   OR raw_avg_score > 5.0;
-- Expected: 0 violations`,
              },
              {
                language: 'python',
                title: 'Pytest Pipeline Validation for HR Metrics',
                description:
                  'Automated pytest suite that validates metric calculation logic, reconciliation accuracy, and data completeness for the HR analytics pipeline.',
                code: `"""Pytest suite for HR metric pipeline validation."""
import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

ENGINE = create_engine(
    "postgresql://analytics:***@db-host:5432/people_analytics"
)


@pytest.fixture(scope="module")
def db_connection():
    """Provide a reusable database connection for all tests."""
    conn = ENGINE.connect()
    yield conn
    conn.close()


class TestHeadcountReconciliation:
    """Validate HRIS-to-payroll headcount reconciliation accuracy."""

    def test_no_null_employee_ids(self, db_connection) -> None:
        """Active headcount must have no NULL employee_id values."""
        result: pd.DataFrame = pd.read_sql(
            text(
                "SELECT COUNT(*) AS nulls "
                "FROM hr_analytics.v_active_headcount "
                "WHERE employee_id IS NULL"
            ),
            db_connection,
        )
        null_count: int = int(result.iloc[0]["nulls"])
        logger.info("NULL employee_id count: %d", null_count)
        assert null_count == 0, f"Found {null_count} NULL employee_ids"

    def test_hris_payroll_gap_within_threshold(self, db_connection) -> None:
        """HRIS and payroll headcount must agree within 2%."""
        hris: pd.DataFrame = pd.read_sql(
            text(
                "SELECT COUNT(*) AS cnt "
                "FROM hr_analytics.v_active_headcount "
                "WHERE is_active = TRUE"
            ),
            db_connection,
        )
        payroll: pd.DataFrame = pd.read_sql(
            text(
                "SELECT COUNT(*) AS cnt "
                "FROM payroll.active_employees "
                "WHERE pay_status IN ('active', 'leave')"
            ),
            db_connection,
        )
        hris_n: int = int(hris.iloc[0]["cnt"])
        payroll_n: int = int(payroll.iloc[0]["cnt"])
        gap_pct: float = (
            abs(hris_n - payroll_n) / max(hris_n, payroll_n, 1) * 100
        )
        logger.info(
            "HRIS=%d, Payroll=%d, gap=%.2f%%", hris_n, payroll_n, gap_pct
        )
        assert gap_pct <= 2.0, (
            f"Headcount gap {gap_pct:.2f}% exceeds 2% threshold"
        )

    def test_turnover_rate_sanity(self, db_connection) -> None:
        """No department should exceed 25% monthly turnover."""
        result: pd.DataFrame = pd.read_sql(
            text(
                "SELECT department, turnover_pct "
                "FROM hr_analytics.v_monthly_turnover "
                "WHERE turnover_pct > 25.0 "
                "  AND term_month >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '3 months')"
            ),
            db_connection,
        )
        logger.info("Departments with >25%% turnover: %d", len(result))
        assert result.empty, (
            f"Departments with excessive turnover: "
            f"{result['department'].tolist()}"
        )


class TestEngagementScores:
    """Validate engagement score bias adjustment logic."""

    def test_scores_within_valid_range(self, db_connection) -> None:
        """All adjusted engagement scores must be between 1.0 and 5.0."""
        result: pd.DataFrame = pd.read_sql(
            text(
                "SELECT COUNT(*) AS violations "
                "FROM hr_analytics.v_engagement_adjusted "
                "WHERE adjusted_score < 1.0 OR adjusted_score > 5.0"
            ),
            db_connection,
        )
        violations: int = int(result.iloc[0]["violations"])
        logger.info("Out-of-range engagement scores: %d", violations)
        assert violations == 0, f"{violations} scores out of [1.0, 5.0] range"

    def test_confidence_flags_populated(self, db_connection) -> None:
        """Every engagement row must have a valid confidence flag."""
        valid_flags: list[str] = [
            "LOW_CONFIDENCE",
            "MODERATE_CONFIDENCE",
            "HIGH_CONFIDENCE",
        ]
        result: pd.DataFrame = pd.read_sql(
            text(
                "SELECT COUNT(*) AS violations "
                "FROM hr_analytics.v_engagement_adjusted "
                "WHERE confidence_flag IS NULL "
                "   OR confidence_flag NOT IN "
                "       ('LOW_CONFIDENCE','MODERATE_CONFIDENCE','HIGH_CONFIDENCE')"
            ),
            db_connection,
        )
        violations: int = int(result.iloc[0]["violations"])
        logger.info("Invalid confidence flags: %d", violations)
        assert violations == 0, f"{violations} rows with invalid confidence_flag"`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Deployment scripts and configuration management for scheduling the headcount reconciliation pipeline in production with proper environment isolation.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Scheduled Reconciliation Deployment Script',
                description:
                  'Bash deployment script that builds the Docker image, runs validation tests, installs the cron schedule, and sends a Slack notification on completion.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
PROJECT_DIR="/opt/hr-metrics-pipeline"
DOCKER_IMAGE="hr-metrics-reconciler:latest"
CRON_SCHEDULE="30 6 * * *"   # 6:30 AM daily
LOG_DIR="/var/log/hr-metrics"
SLACK_WEBHOOK="\${SLACK_WEBHOOK_URL:?'SLACK_WEBHOOK_URL must be set'}"

echo "[INFO] Starting HR metrics pipeline deployment..."
echo "[INFO] Project directory: \${PROJECT_DIR}"
echo "[INFO] Docker image: \${DOCKER_IMAGE}"

# --- Ensure directories exist ---
mkdir -p "\${LOG_DIR}"
mkdir -p "\${PROJECT_DIR}/config"

# --- Build Docker image ---
echo "[INFO] Building Docker image..."
docker build \\
    --tag "\${DOCKER_IMAGE}" \\
    --build-arg BUILD_DATE="\$(date -u +%Y-%m-%dT%H:%M:%SZ)" \\
    --file "\${PROJECT_DIR}/Dockerfile" \\
    "\${PROJECT_DIR}"

# --- Run pytest validation inside container ---
echo "[INFO] Running pipeline validation tests..."
docker run --rm \\
    --env-file "\${PROJECT_DIR}/config/.env" \\
    "\${DOCKER_IMAGE}" \\
    python -m pytest tests/ -v --tb=short --junitxml=/tmp/test-results.xml

echo "[INFO] All validation tests passed."

# --- Install cron job ---
CRON_CMD="\${CRON_SCHEDULE} docker run --rm --env-file \${PROJECT_DIR}/config/.env \${DOCKER_IMAGE} python -m hr_metrics.reconciler >> \${LOG_DIR}/reconciliation.log 2>&1"
CRON_MARKER="# hr-metrics-reconciliation"

# Remove old cron entry if present, then add new one
(crontab -l 2>/dev/null | grep -v "\${CRON_MARKER}" || true; echo "\${CRON_CMD} \${CRON_MARKER}") | crontab -
echo "[INFO] Cron job installed: \${CRON_SCHEDULE}"

# --- Send Slack deployment notification ---
DEPLOY_MSG="HR Metrics Pipeline deployed successfully at \$(date -u +%Y-%m-%dT%H:%M:%SZ). Cron: \${CRON_SCHEDULE}."
curl -sf -X POST "\${SLACK_WEBHOOK}" \\
    -H "Content-Type: application/json" \\
    -d "{\\"text\\": \\"\${DEPLOY_MSG}\\"}" \\
    || echo "[WARN] Slack notification failed (non-fatal)"

echo "[INFO] Deployment complete."`,
              },
              {
                language: 'python',
                title: 'Environment-Based Configuration Loader',
                description:
                  'Configuration loader that reads database credentials and pipeline settings from environment variables with validation and sensible defaults.',
                code: `"""Environment-based configuration loader for HR metrics pipeline."""
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection settings."""
    host: str
    port: int
    database: str
    username: str
    password: str
    schema: str = "hr_analytics"

    @property
    def connection_url(self) -> str:
        return (
            f"postgresql://\${self.username}:\${self.password}"
            f"@\${self.host}:\${self.port}/\${self.database}"
        )


@dataclass(frozen=True)
class AlertConfig:
    """Alerting and notification settings."""
    slack_webhook_url: str
    alert_channel: str = "#hr-data-alerts"
    headcount_change_threshold_pct: float = 5.0
    turnover_rate_max_pct: float = 8.0
    engagement_drop_threshold: float = 0.3
    reconciliation_gap_max_pct: float = 2.0


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""
    db: DatabaseConfig
    alerts: AlertConfig
    environment: str = "production"
    dry_run: bool = False
    log_level: str = "INFO"


def _require_env(key: str) -> str:
    """Read a required environment variable or raise."""
    value: Optional[str] = os.environ.get(key)
    if not value:
        raise EnvironmentError(f"Required env var {key} is not set")
    return value


def _env_float(key: str, default: float) -> float:
    """Read an optional float environment variable."""
    raw: Optional[str] = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%s, using default %s", key, raw, default)
        return default


def load_config() -> PipelineConfig:
    """Load pipeline configuration from environment variables."""
    db_config = DatabaseConfig(
        host=_require_env("DB_HOST"),
        port=int(os.environ.get("DB_PORT", "5432")),
        database=_require_env("DB_NAME"),
        username=_require_env("DB_USER"),
        password=_require_env("DB_PASSWORD"),
        schema=os.environ.get("DB_SCHEMA", "hr_analytics"),
    )

    alert_config = AlertConfig(
        slack_webhook_url=_require_env("SLACK_WEBHOOK_URL"),
        alert_channel=os.environ.get("ALERT_CHANNEL", "#hr-data-alerts"),
        headcount_change_threshold_pct=_env_float("HC_CHANGE_THRESHOLD", 5.0),
        turnover_rate_max_pct=_env_float("TURNOVER_MAX", 8.0),
        engagement_drop_threshold=_env_float("ENGAGEMENT_DROP", 0.3),
        reconciliation_gap_max_pct=_env_float("RECON_GAP_MAX", 2.0),
    )

    config = PipelineConfig(
        db=db_config,
        alerts=alert_config,
        environment=os.environ.get("ENVIRONMENT", "production"),
        dry_run=os.environ.get("DRY_RUN", "false").lower() == "true",
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )

    logger.info(
        "Config loaded: env=%s, db=%s/%s, dry_run=%s",
        config.environment, db_config.host, db_config.database, config.dry_run,
    )
    return config


if __name__ == "__main__":
    cfg: PipelineConfig = load_config()
    logger.info("Database URL: %s", cfg.db.connection_url)
    logger.info("Alert channel: %s", cfg.alerts.alert_channel)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description:
              'Automated checks that flag when key HR metrics drift outside expected ranges, plus a headcount reconciliation SLA dashboard for ongoing governance.',
            codeSnippets: [
              {
                language: 'python',
                title: 'HR Metric Drift Detector',
                description:
                  'Compares current metric values against historical baselines and sends alerts when turnover, headcount, or engagement scores breach thresholds.',
                code: `"""Detect anomalous HR metric values before dashboard refresh."""
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import date, timedelta

engine = create_engine("postgresql://analytics:***@db-host:5432/people_analytics")

THRESHOLDS = {
    "headcount_change_pct": 5.0,       # flag if monthly HC swings > 5%
    "turnover_rate_max": 8.0,          # flag monthly turnover above 8%
    "engagement_drop": 0.3,            # flag if adjusted score drops > 0.3
    "reconciliation_gap_pct": 2.0,     # flag HRIS/payroll gap > 2%
}

def check_headcount_drift() -> list[dict]:
    query = text("""
        SELECT hc_month, SUM(avg_headcount) AS total_hc
        FROM hr_analytics.v_monthly_turnover
        WHERE hc_month >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '3 months')
        GROUP BY hc_month ORDER BY hc_month
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    alerts = []
    if len(df) >= 2:
        prev, curr = df.iloc[-2]["total_hc"], df.iloc[-1]["total_hc"]
        change_pct = abs(curr - prev) / prev * 100 if prev else 0
        if change_pct > THRESHOLDS["headcount_change_pct"]:
            alerts.append({
                "metric": "headcount_change",
                "value": round(change_pct, 2),
                "threshold": THRESHOLDS["headcount_change_pct"],
                "severity": "high" if change_pct > 10 else "medium",
                "detail": f"Headcount shifted {change_pct:.1f}% month-over-month",
            })
    return alerts

def check_engagement_drop() -> list[dict]:
    query = text("""
        SELECT survey_period, AVG(adjusted_score) AS avg_adj
        FROM hr_analytics.v_engagement_adjusted
        GROUP BY survey_period ORDER BY survey_period DESC LIMIT 2
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    alerts = []
    if len(df) == 2:
        drop = df.iloc[1]["avg_adj"] - df.iloc[0]["avg_adj"]
        if drop > THRESHOLDS["engagement_drop"]:
            alerts.append({
                "metric": "engagement_score_drop",
                "value": round(drop, 2),
                "threshold": THRESHOLDS["engagement_drop"],
                "severity": "high",
                "detail": f"Adjusted engagement fell {drop:.2f} pts between periods",
            })
    return alerts

def run_all_checks():
    all_alerts = check_headcount_drift() + check_engagement_drop()
    if all_alerts:
        pd.DataFrame(all_alerts).to_sql(
            "metric_alerts", engine, schema="hr_analytics",
            if_exists="append", index=False,
        )
        print(f"[ALERT] {len(all_alerts)} metric anomalies detected")
    else:
        print("[OK] All HR metrics within expected ranges")
    return all_alerts

if __name__ == "__main__":
    run_all_checks()`,
              },
              {
                language: 'sql',
                title: 'Headcount Reconciliation SLA Dashboard',
                description:
                  'SQL view powering a real-time dashboard that tracks reconciliation SLA compliance, open discrepancies by age, and alert history for leadership visibility.',
                code: `-- Headcount reconciliation SLA monitoring dashboard
-- Shows daily reconciliation status, breach history, and current gaps

-- 1. Daily reconciliation run status (SLA: gap < 2%, run by 8 AM)
CREATE OR REPLACE VIEW hr_analytics.v_reconciliation_sla AS
WITH daily_runs AS (
    SELECT
        detected_date,
        COUNT(*) AS total_discrepancies,
        SUM(CASE WHEN discrepancy_type = 'IN_HRIS_NOT_PAYROLL' THEN 1 ELSE 0 END)
            AS hris_only,
        SUM(CASE WHEN discrepancy_type = 'IN_PAYROLL_NOT_HRIS' THEN 1 ELSE 0 END)
            AS payroll_only,
        MIN(detected_date) AS run_date
    FROM hr_analytics.headcount_reconciliation_log
    WHERE detected_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY detected_date
),
headcount_snapshot AS (
    SELECT COUNT(*) AS current_headcount
    FROM hr_analytics.v_active_headcount
    WHERE is_active = TRUE
)
SELECT
    dr.detected_date,
    dr.total_discrepancies,
    dr.hris_only,
    dr.payroll_only,
    hs.current_headcount,
    ROUND(
        dr.total_discrepancies::NUMERIC
        / NULLIF(hs.current_headcount, 0) * 100, 2
    ) AS gap_pct,
    CASE
        WHEN dr.total_discrepancies::NUMERIC
             / NULLIF(hs.current_headcount, 0) * 100 <= 2.0
        THEN 'WITHIN_SLA'
        ELSE 'SLA_BREACH'
    END AS sla_status
FROM daily_runs dr
CROSS JOIN headcount_snapshot hs
ORDER BY dr.detected_date DESC;

-- 2. Open discrepancies aged by days outstanding
CREATE OR REPLACE VIEW hr_analytics.v_open_discrepancies_aged AS
SELECT
    rl.employee_id,
    rl.discrepancy_type,
    rl.detected_date,
    CURRENT_DATE - rl.detected_date AS days_open,
    CASE
        WHEN CURRENT_DATE - rl.detected_date <= 3  THEN 'fresh'
        WHEN CURRENT_DATE - rl.detected_date <= 7  THEN 'aging'
        WHEN CURRENT_DATE - rl.detected_date <= 14 THEN 'stale'
        ELSE 'critical'
    END AS age_bucket,
    rl.full_name_hris,
    rl.department
FROM hr_analytics.headcount_reconciliation_log rl
LEFT JOIN hr_analytics.reconciliation_resolutions rr
    ON rl.employee_id = rr.employee_id
    AND rl.detected_date = rr.detected_date
WHERE rr.resolved_date IS NULL
ORDER BY days_open DESC;

-- 3. Alert trend summary for executive reporting
SELECT
    DATE_TRUNC('week', detected_date) AS week,
    sla_status,
    COUNT(*) AS days_in_status,
    AVG(gap_pct) AS avg_gap_pct,
    MAX(total_discrepancies) AS max_discrepancies
FROM hr_analytics.v_reconciliation_sla
GROUP BY 1, 2
ORDER BY week DESC, sla_status;`,
              },
            ],
          },
        ],
        toolsUsed: ['PostgreSQL', 'Python', 'pandas', 'SQLAlchemy', 'pytest', 'Docker', 'GitHub Actions', 'cron / Airflow', 'Slack API'],
      },
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude to analyze HR metric discrepancies from exported data, then automate reconciliation alerts with Zapier connecting your HRIS, payroll, and Slack for drift notifications without custom code.',
        estimatedMonthlyCost: '$100 - $200/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Google Sheets (free)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Lattice AI ($6/user/mo)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Export headcount data from HRIS and payroll systems into structured formats for AI analysis. Set up automated weekly exports via Zapier to Google Sheets.',
            toolsUsed: ['Zapier', 'Google Sheets', 'BambooHR', 'ADP'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier HRIS Export Configuration',
                description:
                  'Zapier trigger configuration to automatically export BambooHR employee data weekly to Google Sheets for reconciliation analysis.',
                code: `{
  "zapier_workflow": {
    "name": "Weekly HRIS Headcount Export",
    "trigger": {
      "app": "Schedule by Zapier",
      "event": "Every Week",
      "config": {
        "day_of_week": "Monday",
        "time": "06:00",
        "timezone": "America/New_York"
      }
    },
    "actions": [
      {
        "step": 1,
        "app": "BambooHR",
        "event": "Get All Employees",
        "config": {
          "fields": [
            "id",
            "displayName",
            "department",
            "jobTitle",
            "employmentStatus",
            "hireDate",
            "terminationDate"
          ],
          "filter": "status=active"
        }
      },
      {
        "step": 2,
        "app": "Google Sheets",
        "event": "Create Spreadsheet Row(s)",
        "config": {
          "spreadsheet_id": "{{HR_METRICS_SHEET_ID}}",
          "worksheet": "HRIS_Headcount",
          "clear_existing": true,
          "add_timestamp_column": true,
          "columns": {
            "A": "{{employee_id}}",
            "B": "{{display_name}}",
            "C": "{{department}}",
            "D": "{{employment_status}}",
            "E": "{{hire_date}}",
            "F": "{{export_timestamp}}"
          }
        }
      }
    ]
  }
}`,
              },
              {
                language: 'json',
                title: 'Payroll Data Export Configuration',
                description:
                  'Companion Zapier workflow to export ADP payroll headcount data to the same Google Sheet for cross-system comparison.',
                code: `{
  "zapier_workflow": {
    "name": "Weekly Payroll Headcount Export",
    "trigger": {
      "app": "Schedule by Zapier",
      "event": "Every Week",
      "config": {
        "day_of_week": "Monday",
        "time": "06:30",
        "timezone": "America/New_York"
      }
    },
    "actions": [
      {
        "step": 1,
        "app": "ADP Workforce Now",
        "event": "Get Workers",
        "config": {
          "status_filter": ["active", "leave"],
          "fields": [
            "workerID",
            "legalName",
            "departmentCode",
            "payStatus",
            "lastPayDate"
          ]
        }
      },
      {
        "step": 2,
        "app": "Google Sheets",
        "event": "Create Spreadsheet Row(s)",
        "config": {
          "spreadsheet_id": "{{HR_METRICS_SHEET_ID}}",
          "worksheet": "Payroll_Headcount",
          "clear_existing": true,
          "add_timestamp_column": true,
          "columns": {
            "A": "{{worker_id}}",
            "B": "{{legal_name}}",
            "C": "{{department_code}}",
            "D": "{{pay_status}}",
            "E": "{{last_pay_date}}",
            "F": "{{export_timestamp}}"
          }
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
              'Use ChatGPT or Claude to analyze the exported headcount data, identify discrepancies between HRIS and payroll, and generate actionable reconciliation reports.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'HR Metric Reconciliation Prompt Template',
                description:
                  'Structured prompt for ChatGPT/Claude to analyze headcount discrepancies and generate a prioritized reconciliation report.',
                code: `hr_metric_reconciliation_prompt:
  system_context: |
    You are an HR Analytics specialist helping reconcile headcount
    discrepancies between HRIS and payroll systems. Your analysis should
    be thorough, actionable, and prioritized by business impact.

  user_prompt_template: |
    ## HR Metric Reconciliation Analysis Request

    I have exported headcount data from two systems that should match:

    ### HRIS Data (BambooHR) - {{hris_count}} records
    \`\`\`csv
    {{hris_data_sample}}
    \`\`\`

    ### Payroll Data (ADP) - {{payroll_count}} records
    \`\`\`csv
    {{payroll_data_sample}}
    \`\`\`

    Please analyze and provide:

    1. **Headcount Variance Summary**
       - Total HRIS count vs Payroll count
       - Percentage difference and severity assessment
       - Department-level breakdown of mismatches

    2. **Discrepancy Categories**
       - Employees in HRIS but not Payroll (potential payroll setup issues)
       - Employees in Payroll but not HRIS (potential ghost employees)
       - Status mismatches (active in one, terminated in other)

    3. **Root Cause Hypotheses**
       - Timing differences (recent hires/terms not synced)
       - Definition differences (contractor classification)
       - Data entry errors

    4. **Prioritized Action Items**
       - High priority: Compliance/audit risks
       - Medium priority: Cost implications
       - Low priority: Data hygiene improvements

    5. **Recommended Process Improvements**
       - Preventive measures to reduce future discrepancies

  expected_output_format: |
    Structured markdown report with:
    - Executive summary (2-3 sentences)
    - Detailed findings by category
    - Action items table with owner, priority, deadline
    - Metric definitions for standardization`,
              },
              {
                language: 'yaml',
                title: 'Engagement Score Bias Analysis Prompt',
                description:
                  'Prompt template for analyzing engagement survey response bias and calculating adjusted scores.',
                code: `engagement_bias_analysis_prompt:
  system_context: |
    You are a People Analytics expert specializing in survey methodology
    and response bias correction. Help identify and adjust for non-response
    bias in employee engagement surveys.

  user_prompt_template: |
    ## Engagement Survey Bias Analysis

    Our latest engagement survey has potential response bias issues:

    ### Survey Response Data by Department
    \`\`\`csv
    department,headcount,respondents,response_rate,raw_avg_score
    {{survey_data}}
    \`\`\`

    ### Historical Benchmark
    - Company-wide average response rate: {{historical_response_rate}}%
    - Previous period average score: {{previous_score}}

    Please analyze:

    1. **Response Rate Analysis**
       - Identify departments with statistically low response rates
       - Flag departments where results may not be representative
       - Calculate confidence intervals for each department score

    2. **Bias Adjustment Calculation**
       - Apply non-response bias correction using the assumption that
         non-respondents score at the 25th percentile of respondents
       - Show original vs adjusted scores
       - Highlight departments where adjustment changes the narrative

    3. **Confidence Flags**
       - HIGH_CONFIDENCE: >75% response rate
       - MODERATE_CONFIDENCE: 50-75% response rate
       - LOW_CONFIDENCE: <50% response rate

    4. **Recommendations**
       - Which department scores should leadership trust?
       - Where should we investigate further before acting?
       - Suggestions to improve response rates next cycle

  output_format: |
    Provide a table with columns:
    Department | Raw Score | Adjusted Score | Confidence | Action Required`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Set up Zapier workflows to automatically detect metric drift, trigger AI analysis, and deliver reconciliation alerts to Slack and stakeholders.',
            toolsUsed: ['Zapier', 'Slack', 'Google Sheets', 'Email'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Automated Drift Detection & Alert Workflow',
                description:
                  'Zapier workflow that detects when HRIS-Payroll headcount gap exceeds threshold and sends prioritized alerts to HR operations.',
                code: `{
  "zapier_workflow": {
    "name": "HR Metric Drift Alert System",
    "trigger": {
      "app": "Google Sheets",
      "event": "New or Updated Spreadsheet Row",
      "config": {
        "spreadsheet_id": "{{HR_METRICS_SHEET_ID}}",
        "worksheet": "Reconciliation_Summary",
        "trigger_column": "gap_percentage"
      }
    },
    "actions": [
      {
        "step": 1,
        "app": "Filter by Zapier",
        "event": "Only Continue If",
        "config": {
          "conditions": [
            {
              "field": "{{gap_percentage}}",
              "operator": "greater_than",
              "value": 2.0
            }
          ]
        }
      },
      {
        "step": 2,
        "app": "Paths by Zapier",
        "event": "Route by Severity",
        "config": {
          "paths": [
            {
              "name": "Critical Alert",
              "condition": "{{gap_percentage}} > 5",
              "continue_to": "step_3a"
            },
            {
              "name": "Warning Alert",
              "condition": "{{gap_percentage}} > 2 AND {{gap_percentage}} <= 5",
              "continue_to": "step_3b"
            }
          ]
        }
      },
      {
        "step": "3a",
        "app": "Slack",
        "event": "Send Channel Message",
        "config": {
          "channel": "#hr-ops-critical",
          "message_template": ":rotating_light: *CRITICAL: HR Metric Drift Detected*\\n\\n*Gap:* {{gap_percentage}}% between HRIS and Payroll\\n*HRIS Count:* {{hris_count}}\\n*Payroll Count:* {{payroll_count}}\\n*Detected:* {{timestamp}}\\n\\n*Immediate Action Required:* Review reconciliation report\\n<{{reconciliation_sheet_url}}|View Full Report>",
          "mention": "@hr-ops-oncall"
        }
      },
      {
        "step": "3b",
        "app": "Slack",
        "event": "Send Channel Message",
        "config": {
          "channel": "#hr-data-alerts",
          "message_template": ":warning: *Warning: HR Metric Variance*\\n\\nHeadcount gap of {{gap_percentage}}% detected.\\n\\n*Details:*\\n- HRIS: {{hris_count}} employees\\n- Payroll: {{payroll_count}} employees\\n- Delta: {{absolute_difference}}\\n\\n<{{reconciliation_sheet_url}}|Review Discrepancies>"
        }
      },
      {
        "step": 4,
        "app": "Google Sheets",
        "event": "Update Spreadsheet Row",
        "config": {
          "spreadsheet_id": "{{HR_METRICS_SHEET_ID}}",
          "worksheet": "Alert_Log",
          "row_data": {
            "alert_timestamp": "{{zap_timestamp}}",
            "gap_percentage": "{{gap_percentage}}",
            "severity": "{{path_taken}}",
            "notification_sent": true
          }
        }
      }
    ]
  }
}`,
              },
              {
                language: 'json',
                title: 'Weekly Executive Summary Delivery',
                description:
                  'Zapier workflow that compiles weekly HR metrics into a summary email with AI-generated insights for leadership.',
                code: `{
  "zapier_workflow": {
    "name": "Weekly HR Metrics Executive Summary",
    "trigger": {
      "app": "Schedule by Zapier",
      "event": "Every Week",
      "config": {
        "day_of_week": "Friday",
        "time": "16:00",
        "timezone": "America/New_York"
      }
    },
    "actions": [
      {
        "step": 1,
        "app": "Google Sheets",
        "event": "Get Many Spreadsheet Rows",
        "config": {
          "spreadsheet_id": "{{HR_METRICS_SHEET_ID}}",
          "worksheet": "Weekly_Summary",
          "filter": "week_ending = {{current_week_end}}"
        }
      },
      {
        "step": 2,
        "app": "ChatGPT",
        "event": "Conversation",
        "config": {
          "model": "gpt-4",
          "system_message": "You are an HR analytics assistant. Summarize the weekly HR metrics data into a concise executive brief.",
          "user_message": "Generate a 3-paragraph executive summary of this week's HR metrics:\\n\\nHeadcount: {{current_headcount}} ({{headcount_change}} from last week)\\nTurnover Rate: {{turnover_rate}}%\\nOpen Positions: {{open_positions}}\\nReconciliation Status: {{reconciliation_status}}\\nEngagement Score: {{engagement_score}}\\n\\nHighlight any metrics that need attention.",
          "max_tokens": 500
        }
      },
      {
        "step": 3,
        "app": "Gmail",
        "event": "Send Email",
        "config": {
          "to": "{{executive_distribution_list}}",
          "subject": "Weekly HR Metrics Summary - Week Ending {{week_end_date}}",
          "body_template": "<h2>HR Metrics Weekly Brief</h2>\\n\\n{{chatgpt_summary}}\\n\\n<hr>\\n\\n<h3>Key Metrics</h3>\\n<table>\\n<tr><td><b>Active Headcount:</b></td><td>{{current_headcount}}</td></tr>\\n<tr><td><b>Monthly Turnover:</b></td><td>{{turnover_rate}}%</td></tr>\\n<tr><td><b>HRIS-Payroll Gap:</b></td><td>{{gap_percentage}}%</td></tr>\\n<tr><td><b>Engagement Score:</b></td><td>{{engagement_score}}/5.0</td></tr>\\n</table>\\n\\n<p><a href='{{full_dashboard_url}}'>View Full Dashboard</a></p>",
          "is_html": true
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
          'Deploy a multi-agent system where specialized AI agents handle source reconciliation, metric drift detection, bias adjustment calculations, and automated alerting, coordinated by a supervisor agent that ensures consistent HR metric governance.',
        estimatedMonthlyCost: '$500 - $1,200/month',
        architecture:
          'Supervisor agent coordinates four specialist agents: Source Reconciliation Agent (HRIS/payroll matching), Drift Detection Agent (anomaly identification), Bias Adjustment Agent (engagement score correction), and Reporting Agent (stakeholder communications). LangGraph orchestrates the daily workflow with Redis-backed state persistence.',
        agents: [
          {
            name: 'Source Reconciliation Agent',
            role: 'Data Matching Specialist',
            goal: 'Match employee records across HRIS, payroll, and benefits systems using fuzzy matching and identity resolution algorithms to maintain the master employee crosswalk.',
            tools: ['pandas', 'fuzzywuzzy', 'recordlinkage', 'sqlalchemy', 'BambooHR API', 'ADP API'],
          },
          {
            name: 'Drift Detection Agent',
            role: 'Anomaly Detection Specialist',
            goal: 'Monitor HR metrics for statistically significant deviations from historical baselines and trigger alerts when turnover, headcount, or engagement metrics breach thresholds.',
            tools: ['scipy.stats', 'pandas', 'numpy', 'sklearn.ensemble', 'prometheus_client'],
          },
          {
            name: 'Bias Adjustment Agent',
            role: 'Statistical Analysis Specialist',
            goal: 'Apply non-response bias corrections to engagement survey scores and calculate confidence intervals for all HR metrics to ensure statistically sound reporting.',
            tools: ['scipy.stats', 'statsmodels', 'pandas', 'numpy'],
          },
          {
            name: 'Reporting Agent',
            role: 'Stakeholder Communications Specialist',
            goal: 'Generate executive summaries, reconciliation reports, and audit-ready documentation with appropriate detail levels for different audiences.',
            tools: ['jinja2', 'markdown', 'slack_sdk', 'sendgrid', 'google-api-python-client'],
          },
          {
            name: 'Supervisor Agent',
            role: 'Workflow Orchestrator',
            goal: 'Coordinate the specialist agents, manage workflow state, handle errors gracefully, and ensure daily HR metric governance runs complete successfully.',
            tools: ['langgraph', 'redis', 'langchain', 'langsmith'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with daily checkpointing and 30-day audit trail retention',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent architecture with CrewAI, specifying each agent role, goals, and tool access for the HR metric validation pipeline.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'HR Metric Validation Agent Definitions',
                description:
                  'CrewAI agent definitions for the source reconciliation, drift detection, bias adjustment, and reporting agents with their specialized tools.',
                code: `"""HR Metric Validation Multi-Agent System - Agent Definitions."""
from typing import List, Optional
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class HRMetricConfig(BaseModel):
    """Configuration for HR metric validation thresholds."""
    headcount_gap_threshold_pct: float = Field(default=2.0, description="Max allowed HRIS-payroll gap")
    turnover_rate_max_pct: float = Field(default=8.0, description="Monthly turnover alert threshold")
    engagement_drop_threshold: float = Field(default=0.3, description="Point drop triggering alert")
    response_rate_min_pct: float = Field(default=50.0, description="Min response rate for confidence")


class ReconciliationResult(BaseModel):
    """Output schema for reconciliation agent."""
    hris_count: int
    payroll_count: int
    gap_percentage: float
    discrepancies: List[dict]
    severity: str
    recommendations: List[str]


# Initialize the LLM with appropriate settings for HR domain
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.1,  # Low temperature for consistent analysis
    max_tokens=4096,
)


# --- Source Reconciliation Agent ---
source_reconciliation_agent = Agent(
    role="HR Data Reconciliation Specialist",
    goal="""Match employee records across HRIS (BambooHR), payroll (ADP), and
    benefits systems. Identify discrepancies, classify their severity, and
    provide actionable resolution steps.""",
    backstory="""You are an expert in HR data integration with 10+ years of
    experience reconciling employee records across enterprise systems. You
    understand common causes of HRIS-payroll mismatches including timing
    differences, contractor classification, and data entry errors. You prioritize
    compliance risks and can identify ghost employees or missing payroll setups.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[],  # Tools added separately
    max_iter=5,
    memory=True,
)


# --- Drift Detection Agent ---
drift_detection_agent = Agent(
    role="HR Metric Anomaly Detection Specialist",
    goal="""Monitor HR metrics for statistically significant deviations from
    historical baselines. Detect unusual turnover spikes, headcount swings,
    and engagement score changes that require investigation.""",
    backstory="""You are a data scientist specializing in anomaly detection
    for people analytics. You apply statistical tests to distinguish normal
    variation from true metric drift. You understand seasonality in HR data
    (Q4 turnover spikes, post-survey engagement dips) and adjust thresholds
    accordingly. You never cry wolf on normal fluctuations.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[],
    max_iter=5,
    memory=True,
)


# --- Bias Adjustment Agent ---
bias_adjustment_agent = Agent(
    role="HR Survey Methodology Specialist",
    goal="""Apply rigorous statistical corrections to engagement survey scores
    to account for non-response bias. Calculate confidence intervals and flag
    results that should not be trusted due to low response rates.""",
    backstory="""You are a PhD statistician specializing in survey methodology
    and response bias correction. You understand that disengaged employees are
    less likely to respond to surveys, creating systematic upward bias in raw
    scores. You apply conservative adjustments and always communicate uncertainty
    to prevent overconfident decision-making on flawed data.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[],
    max_iter=3,
    memory=True,
)


# --- Reporting Agent ---
reporting_agent = Agent(
    role="HR Analytics Communications Specialist",
    goal="""Generate clear, actionable reports tailored to different audiences:
    executive summaries for leadership, detailed reconciliation logs for HR ops,
    and audit-ready documentation for compliance.""",
    backstory="""You are a seasoned HR analytics communicator who knows how to
    translate complex data findings into business impact. For executives, you
    lead with the 'so what' and recommended actions. For operations teams, you
    provide step-by-step resolution guides. For auditors, you document methodology
    and maintain chain of custody for all metric calculations.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[],
    max_iter=3,
    memory=True,
)


# --- Supervisor Agent ---
supervisor_agent = Agent(
    role="HR Metric Governance Coordinator",
    goal="""Orchestrate the daily HR metric validation workflow, ensuring all
    specialist agents complete their tasks successfully. Handle errors gracefully,
    escalate critical issues, and maintain audit trails.""",
    backstory="""You are the chief of staff for HR analytics operations. You
    coordinate the specialist agents, manage dependencies between their work,
    and ensure the daily metric governance pipeline runs to completion. When
    issues arise, you know when to retry, when to skip, and when to escalate
    to human operators.""",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[],
    max_iter=10,
    memory=True,
)


def create_hr_metric_crew(config: HRMetricConfig) -> Crew:
    """Create the HR metric validation crew with configured agents."""

    # Define tasks for each agent
    reconciliation_task = Task(
        description="""Pull headcount data from HRIS and payroll systems.
        Match records using employee ID and fuzzy name matching.
        Identify and classify all discrepancies.
        Threshold for alerting: >{threshold}% gap.""".format(
            threshold=config.headcount_gap_threshold_pct
        ),
        expected_output="ReconciliationResult with discrepancy list and severity",
        agent=source_reconciliation_agent,
    )

    drift_task = Task(
        description="""Analyze this month's HR metrics against 6-month baseline.
        Flag any metric breaching thresholds:
        - Turnover > {turnover}%
        - Headcount change > 5%
        - Engagement drop > {engagement} points""".format(
            turnover=config.turnover_rate_max_pct,
            engagement=config.engagement_drop_threshold,
        ),
        expected_output="List of metric anomalies with statistical confidence",
        agent=drift_detection_agent,
    )

    bias_task = Task(
        description="""Review engagement survey results. Calculate bias-adjusted
        scores for departments with <{min_response}% response rate.
        Assign confidence flags to all scores.""".format(
            min_response=config.response_rate_min_pct
        ),
        expected_output="Adjusted engagement scores with confidence intervals",
        agent=bias_adjustment_agent,
    )

    reporting_task = Task(
        description="""Compile findings from reconciliation, drift detection,
        and bias adjustment into three reports:
        1. Executive summary (3 paragraphs max)
        2. HR Ops action items (prioritized list)
        3. Audit log entry (methodology + results)""",
        expected_output="Three formatted reports for different audiences",
        agent=reporting_agent,
        context=[reconciliation_task, drift_task, bias_task],
    )

    return Crew(
        agents=[
            source_reconciliation_agent,
            drift_detection_agent,
            bias_adjustment_agent,
            reporting_agent,
        ],
        tasks=[reconciliation_task, drift_task, bias_task, reporting_task],
        process=Process.sequential,
        manager_agent=supervisor_agent,
        verbose=True,
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the Source Reconciliation Agent with tools for pulling data from BambooHR, ADP, and other HR systems, plus fuzzy matching capabilities for identity resolution.',
            toolsUsed: ['CrewAI', 'LangChain', 'pandas', 'fuzzywuzzy', 'sqlalchemy'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Source Reconciliation Agent Tools',
                description:
                  'Custom CrewAI tools for the Source Reconciliation Agent to pull HRIS/payroll data and perform fuzzy matching for identity resolution.',
                code: `"""Source Reconciliation Agent Tools for HR Data Integration."""
import hashlib
import os
from typing import Any, Dict, List, Optional, Type

import pandas as pd
import requests
from crewai.tools import BaseTool
from fuzzywuzzy import fuzz, process
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text


class BambooHRFetchInput(BaseModel):
    """Input schema for BambooHR employee fetch."""
    include_terminated: bool = Field(
        default=False,
        description="Whether to include terminated employees"
    )
    department_filter: Optional[str] = Field(
        default=None,
        description="Filter by department name"
    )


class BambooHRFetchTool(BaseTool):
    """Tool to fetch employee data from BambooHR API."""

    name: str = "bamboohr_fetch"
    description: str = """Fetches employee directory from BambooHR HRIS.
    Returns employee ID, name, department, status, and hire date."""
    args_schema: Type[BaseModel] = BambooHRFetchInput

    def _run(
        self,
        include_terminated: bool = False,
        department_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute BambooHR API call to fetch employees."""
        api_key = os.environ.get("BAMBOOHR_API_KEY")
        subdomain = os.environ.get("BAMBOOHR_SUBDOMAIN")

        if not api_key or not subdomain:
            return {"error": "BambooHR credentials not configured"}

        url = f"https://api.bamboohr.com/api/gateway.php/{subdomain}/v1/employees/directory"

        try:
            response = requests.get(
                url,
                headers={"Accept": "application/json"},
                auth=(api_key, "x"),
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            employees = data.get("employees", [])
            df = pd.DataFrame(employees)

            # Filter terminated if requested
            if not include_terminated and "status" in df.columns:
                df = df[df["status"] == "Active"]

            # Filter by department if specified
            if department_filter and "department" in df.columns:
                df = df[df["department"].str.contains(
                    department_filter, case=False, na=False
                )]

            return {
                "success": True,
                "count": len(df),
                "employees": df.to_dict(orient="records"),
                "columns": list(df.columns),
            }

        except requests.RequestException as e:
            return {"error": f"BambooHR API error: {str(e)}"}


class ADPFetchInput(BaseModel):
    """Input schema for ADP payroll fetch."""
    status_filter: List[str] = Field(
        default=["active", "leave"],
        description="Worker status codes to include"
    )


class ADPFetchTool(BaseTool):
    """Tool to fetch worker data from ADP Workforce Now API."""

    name: str = "adp_fetch"
    description: str = """Fetches worker records from ADP payroll system.
    Returns worker ID, legal name, pay status, and department code."""
    args_schema: Type[BaseModel] = ADPFetchInput

    def _run(self, status_filter: List[str] = None) -> Dict[str, Any]:
        """Execute ADP API call to fetch workers."""
        client_id = os.environ.get("ADP_CLIENT_ID")
        client_secret = os.environ.get("ADP_CLIENT_SECRET")

        if not client_id or not client_secret:
            return {"error": "ADP credentials not configured"}

        status_filter = status_filter or ["active", "leave"]

        try:
            # Get OAuth token
            token_response = requests.post(
                "https://accounts.adp.com/auth/oauth/v2/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=30,
            )
            token_response.raise_for_status()
            token = token_response.json()["access_token"]

            # Fetch workers
            workers_response = requests.get(
                "https://api.adp.com/hr/v2/workers",
                headers={"Authorization": f"Bearer {token}"},
                timeout=60,
            )
            workers_response.raise_for_status()

            workers = workers_response.json().get("workers", [])

            records = []
            for w in workers:
                status = w.get("workerStatus", {}).get("statusCode", {}).get("codeValue", "")
                if status.lower() in [s.lower() for s in status_filter]:
                    records.append({
                        "adp_worker_id": w.get("workerID", {}).get("idValue"),
                        "legal_name": w.get("person", {}).get("legalName", {}).get("formattedName"),
                        "pay_status": status,
                        "department_code": w.get("organizationalUnits", [{}])[0].get("nameCode", {}).get("codeValue"),
                        "ssn_hash": hashlib.sha256(
                            w.get("person", {}).get("governmentIDs", [{}])[0]
                            .get("idValue", "").encode()
                        ).hexdigest() if w.get("person", {}).get("governmentIDs") else None,
                    })

            return {
                "success": True,
                "count": len(records),
                "workers": records,
            }

        except requests.RequestException as e:
            return {"error": f"ADP API error: {str(e)}"}


class FuzzyMatchInput(BaseModel):
    """Input schema for fuzzy matching tool."""
    hris_records: List[Dict] = Field(description="Records from HRIS system")
    payroll_records: List[Dict] = Field(description="Records from payroll system")
    match_threshold: int = Field(default=85, description="Minimum fuzzy match score (0-100)")


class FuzzyMatchTool(BaseTool):
    """Tool for fuzzy matching employee records across systems."""

    name: str = "fuzzy_match_employees"
    description: str = """Matches employee records between HRIS and payroll using
    fuzzy name matching when exact ID matches fail. Returns matched pairs,
    HRIS-only records, and payroll-only records."""
    args_schema: Type[BaseModel] = FuzzyMatchInput

    def _run(
        self,
        hris_records: List[Dict],
        payroll_records: List[Dict],
        match_threshold: int = 85,
    ) -> Dict[str, Any]:
        """Perform fuzzy matching between HRIS and payroll records."""

        hris_df = pd.DataFrame(hris_records)
        payroll_df = pd.DataFrame(payroll_records)

        # Standardize name columns
        hris_name_col = next(
            (c for c in hris_df.columns if "name" in c.lower()),
            hris_df.columns[0]
        )
        payroll_name_col = next(
            (c for c in payroll_df.columns if "name" in c.lower()),
            payroll_df.columns[0]
        )

        matched = []
        hris_only = []
        payroll_matched_indices = set()

        for idx, hris_row in hris_df.iterrows():
            hris_name = str(hris_row[hris_name_col]).strip()

            # Find best match in payroll
            payroll_names = payroll_df[payroll_name_col].tolist()
            best_match = process.extractOne(
                hris_name,
                payroll_names,
                scorer=fuzz.token_sort_ratio
            )

            if best_match and best_match[1] >= match_threshold:
                payroll_idx = payroll_names.index(best_match[0])
                payroll_matched_indices.add(payroll_idx)
                matched.append({
                    "hris_record": hris_row.to_dict(),
                    "payroll_record": payroll_df.iloc[payroll_idx].to_dict(),
                    "match_score": best_match[1],
                    "match_type": "exact" if best_match[1] == 100 else "fuzzy",
                })
            else:
                hris_only.append({
                    "record": hris_row.to_dict(),
                    "best_match_score": best_match[1] if best_match else 0,
                    "discrepancy_type": "IN_HRIS_NOT_PAYROLL",
                })

        # Find payroll-only records
        payroll_only = [
            {
                "record": payroll_df.iloc[i].to_dict(),
                "discrepancy_type": "IN_PAYROLL_NOT_HRIS",
            }
            for i in range(len(payroll_df))
            if i not in payroll_matched_indices
        ]

        return {
            "success": True,
            "summary": {
                "total_hris": len(hris_df),
                "total_payroll": len(payroll_df),
                "matched": len(matched),
                "hris_only": len(hris_only),
                "payroll_only": len(payroll_only),
                "match_rate_pct": round(len(matched) / max(len(hris_df), 1) * 100, 2),
            },
            "matched_records": matched,
            "hris_only_records": hris_only,
            "payroll_only_records": payroll_only,
        }


# Register tools with the Source Reconciliation Agent
reconciliation_tools = [
    BambooHRFetchTool(),
    ADPFetchTool(),
    FuzzyMatchTool(),
]`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Drift Detection and Bias Adjustment agents with statistical analysis tools for anomaly detection and survey score correction.',
            toolsUsed: ['CrewAI', 'scipy', 'statsmodels', 'pandas', 'numpy'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Drift Detection Agent Tools',
                description:
                  'Statistical analysis tools for the Drift Detection Agent to identify anomalous HR metric values using z-scores and trend analysis.',
                code: `"""Drift Detection Agent Tools for HR Metric Anomaly Detection."""
from typing import Any, Dict, List, Type

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from scipy import stats
from sqlalchemy import create_engine, text


class MetricDriftInput(BaseModel):
    """Input schema for metric drift detection."""
    metric_name: str = Field(description="Name of the metric to analyze")
    current_value: float = Field(description="Current period's metric value")
    historical_values: List[float] = Field(description="Historical values for baseline")
    threshold_z_score: float = Field(default=2.0, description="Z-score threshold for anomaly")


class MetricDriftDetectionTool(BaseTool):
    """Tool for detecting statistically significant metric drift."""

    name: str = "detect_metric_drift"
    description: str = """Analyzes a metric value against historical baseline
    using z-score analysis. Flags values that deviate significantly from the norm."""
    args_schema: Type[BaseModel] = MetricDriftInput

    def _run(
        self,
        metric_name: str,
        current_value: float,
        historical_values: List[float],
        threshold_z_score: float = 2.0,
    ) -> Dict[str, Any]:
        """Detect if current metric value represents significant drift."""

        if len(historical_values) < 3:
            return {
                "error": "Insufficient historical data (need at least 3 periods)",
                "metric_name": metric_name,
            }

        historical = np.array(historical_values)
        mean = np.mean(historical)
        std = np.std(historical, ddof=1)  # Sample standard deviation

        # Avoid division by zero
        if std == 0:
            z_score = 0 if current_value == mean else float('inf')
        else:
            z_score = (current_value - mean) / std

        # Calculate percentile of current value
        percentile = stats.percentileofscore(historical, current_value)

        # Determine severity
        abs_z = abs(z_score)
        if abs_z >= 3.0:
            severity = "critical"
            is_anomaly = True
        elif abs_z >= threshold_z_score:
            severity = "warning"
            is_anomaly = True
        else:
            severity = "normal"
            is_anomaly = False

        # Direction of drift
        direction = "increase" if current_value > mean else "decrease"

        return {
            "metric_name": metric_name,
            "current_value": current_value,
            "historical_mean": round(mean, 4),
            "historical_std": round(std, 4),
            "z_score": round(z_score, 2),
            "percentile": round(percentile, 1),
            "is_anomaly": is_anomaly,
            "severity": severity,
            "direction": direction,
            "threshold_used": threshold_z_score,
            "interpretation": f"{metric_name} is {abs_z:.1f} standard deviations "
                            f"{'above' if z_score > 0 else 'below'} the historical mean. "
                            f"This {'IS' if is_anomaly else 'is NOT'} a statistically "
                            f"significant deviation.",
        }


class TurnoverTrendInput(BaseModel):
    """Input schema for turnover trend analysis."""
    monthly_turnover_rates: List[Dict] = Field(
        description="List of {month, department, turnover_pct} records"
    )
    alert_threshold_pct: float = Field(default=8.0, description="Turnover % triggering alert")


class TurnoverTrendAnalysisTool(BaseTool):
    """Tool for analyzing turnover trends and detecting hotspots."""

    name: str = "analyze_turnover_trends"
    description: str = """Analyzes monthly turnover rates by department to detect
    concerning trends and identify departments with abnormal attrition."""
    args_schema: Type[BaseModel] = TurnoverTrendInput

    def _run(
        self,
        monthly_turnover_rates: List[Dict],
        alert_threshold_pct: float = 8.0,
    ) -> Dict[str, Any]:
        """Analyze turnover trends across departments."""

        df = pd.DataFrame(monthly_turnover_rates)

        if df.empty:
            return {"error": "No turnover data provided"}

        # Aggregate by department
        dept_summary = df.groupby("department").agg({
            "turnover_pct": ["mean", "std", "max", "count"]
        }).round(2)
        dept_summary.columns = ["avg_turnover", "std_turnover", "max_turnover", "months"]
        dept_summary = dept_summary.reset_index()

        # Identify hotspots
        hotspots = dept_summary[
            dept_summary["avg_turnover"] > alert_threshold_pct
        ].to_dict(orient="records")

        # Calculate company-wide trend
        monthly_trend = df.groupby("month")["turnover_pct"].mean().reset_index()
        monthly_trend = monthly_trend.sort_values("month")

        # Check for increasing trend using linear regression
        if len(monthly_trend) >= 3:
            x = np.arange(len(monthly_trend))
            y = monthly_trend["turnover_pct"].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            trend_direction = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
            trend_significant = p_value < 0.05
        else:
            slope, trend_direction, trend_significant = 0, "insufficient_data", False

        return {
            "summary": {
                "total_departments": len(dept_summary),
                "hotspot_count": len(hotspots),
                "company_avg_turnover": round(df["turnover_pct"].mean(), 2),
                "company_max_turnover": round(df["turnover_pct"].max(), 2),
            },
            "trend": {
                "direction": trend_direction,
                "slope_per_month": round(slope, 3),
                "statistically_significant": trend_significant,
            },
            "hotspot_departments": hotspots,
            "department_summary": dept_summary.to_dict(orient="records"),
            "alert_threshold_used": alert_threshold_pct,
        }


class BiasAdjustmentInput(BaseModel):
    """Input schema for engagement score bias adjustment."""
    survey_results: List[Dict] = Field(
        description="List of {department, headcount, respondents, raw_score} records"
    )
    assumed_nonrespondent_percentile: float = Field(
        default=25.0,
        description="Percentile to assume for non-respondents"
    )


class EngagementBiasAdjustmentTool(BaseTool):
    """Tool for calculating bias-adjusted engagement scores."""

    name: str = "adjust_engagement_bias"
    description: str = """Adjusts raw engagement survey scores for non-response
    bias by assuming non-respondents score at a specified percentile."""
    args_schema: Type[BaseModel] = BiasAdjustmentInput

    def _run(
        self,
        survey_results: List[Dict],
        assumed_nonrespondent_percentile: float = 25.0,
    ) -> Dict[str, Any]:
        """Calculate bias-adjusted engagement scores."""

        df = pd.DataFrame(survey_results)

        if df.empty:
            return {"error": "No survey data provided"}

        # Calculate response rates
        df["response_rate"] = (df["respondents"] / df["headcount"] * 100).round(1)

        # Get overall 25th percentile of raw scores for non-respondent assumption
        nonrespondent_score = np.percentile(
            df["raw_score"],
            assumed_nonrespondent_percentile
        )

        # Calculate adjusted scores
        # adjusted = raw * response_rate + nonrespondent_score * (1 - response_rate)
        df["adjusted_score"] = (
            df["raw_score"] * (df["response_rate"] / 100) +
            nonrespondent_score * (1 - df["response_rate"] / 100)
        ).round(2)

        # Assign confidence flags
        def assign_confidence(rate: float) -> str:
            if rate >= 75:
                return "HIGH_CONFIDENCE"
            elif rate >= 50:
                return "MODERATE_CONFIDENCE"
            else:
                return "LOW_CONFIDENCE"

        df["confidence_flag"] = df["response_rate"].apply(assign_confidence)

        # Calculate impact of adjustment
        df["adjustment_delta"] = (df["raw_score"] - df["adjusted_score"]).round(2)

        # Summary statistics
        low_confidence_depts = df[df["confidence_flag"] == "LOW_CONFIDENCE"]

        return {
            "methodology": {
                "assumed_nonrespondent_score": round(nonrespondent_score, 2),
                "assumed_percentile": assumed_nonrespondent_percentile,
                "formula": "adjusted = raw * response_rate + nonrespondent_score * (1 - response_rate)",
            },
            "summary": {
                "total_departments": len(df),
                "avg_response_rate": round(df["response_rate"].mean(), 1),
                "avg_raw_score": round(df["raw_score"].mean(), 2),
                "avg_adjusted_score": round(df["adjusted_score"].mean(), 2),
                "avg_adjustment": round(df["adjustment_delta"].mean(), 2),
                "low_confidence_count": len(low_confidence_depts),
            },
            "adjusted_scores": df[[
                "department", "headcount", "respondents", "response_rate",
                "raw_score", "adjusted_score", "adjustment_delta", "confidence_flag"
            ]].to_dict(orient="records"),
            "warnings": [
                f"Department '{row['department']}' has only {row['response_rate']}% response rate - results may not be representative"
                for _, row in low_confidence_depts.iterrows()
            ],
        }


# Tool registrations
drift_detection_tools = [
    MetricDriftDetectionTool(),
    TurnoverTrendAnalysisTool(),
]

bias_adjustment_tools = [
    EngagementBiasAdjustmentTool(),
]`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement the LangGraph state machine that coordinates the multi-agent HR metric validation workflow with Redis-backed state persistence.',
            toolsUsed: ['LangGraph', 'Redis', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph HR Metric Validation Workflow',
                description:
                  'LangGraph state machine orchestrating the daily HR metric validation pipeline with supervisor coordination and checkpoint persistence.',
                code: `"""LangGraph Orchestration for HR Metric Validation Pipeline."""
import json
import logging
import operator
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

import redis
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HRMetricState(TypedDict):
    """State schema for HR metric validation workflow."""

    # Workflow metadata
    run_id: str
    run_date: str
    status: Literal["pending", "running", "completed", "failed"]

    # Agent outputs
    reconciliation_result: Optional[Dict[str, Any]]
    drift_detection_result: Optional[Dict[str, Any]]
    bias_adjustment_result: Optional[Dict[str, Any]]
    reports_generated: Optional[Dict[str, str]]

    # Error tracking
    errors: Annotated[List[str], operator.add]

    # Message history for agents
    messages: Annotated[List[BaseMessage], operator.add]

    # Control flow
    current_agent: str
    next_agent: Optional[str]
    requires_escalation: bool
    escalation_reason: Optional[str]


class HRMetricWorkflow:
    """LangGraph workflow for HR metric validation."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the workflow with Redis checkpointing."""
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)
        self.redis_client = redis.from_url(redis_url)
        self.checkpointer = MemorySaver()  # Use Redis in production
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph state machine."""

        workflow = StateGraph(HRMetricState)

        # Add nodes for each agent
        workflow.add_node("initialize", self._initialize_run)
        workflow.add_node("reconciliation_agent", self._run_reconciliation)
        workflow.add_node("drift_detection_agent", self._run_drift_detection)
        workflow.add_node("bias_adjustment_agent", self._run_bias_adjustment)
        workflow.add_node("reporting_agent", self._generate_reports)
        workflow.add_node("supervisor_check", self._supervisor_check)
        workflow.add_node("escalate", self._escalate_to_human)
        workflow.add_node("finalize", self._finalize_run)

        # Define edges
        workflow.set_entry_point("initialize")

        workflow.add_edge("initialize", "reconciliation_agent")
        workflow.add_edge("reconciliation_agent", "supervisor_check")

        # Conditional routing from supervisor
        workflow.add_conditional_edges(
            "supervisor_check",
            self._route_after_supervisor,
            {
                "drift_detection": "drift_detection_agent",
                "bias_adjustment": "bias_adjustment_agent",
                "reporting": "reporting_agent",
                "escalate": "escalate",
                "finalize": "finalize",
            }
        )

        workflow.add_edge("drift_detection_agent", "supervisor_check")
        workflow.add_edge("bias_adjustment_agent", "supervisor_check")
        workflow.add_edge("reporting_agent", "finalize")
        workflow.add_edge("escalate", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def _initialize_run(self, state: HRMetricState) -> Dict[str, Any]:
        """Initialize a new validation run."""
        logger.info("Initializing HR metric validation run")

        run_id = f"hr-validation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        return {
            "run_id": run_id,
            "run_date": datetime.now().isoformat(),
            "status": "running",
            "current_agent": "reconciliation_agent",
            "errors": [],
            "requires_escalation": False,
            "messages": [
                SystemMessage(content="Starting HR metric validation pipeline"),
            ],
        }

    def _run_reconciliation(self, state: HRMetricState) -> Dict[str, Any]:
        """Execute source reconciliation agent."""
        logger.info("Running Source Reconciliation Agent")

        try:
            # In production, this calls the actual CrewAI agent
            # Simulated result for demonstration
            result = {
                "hris_count": 1247,
                "payroll_count": 1239,
                "gap_percentage": 0.64,
                "discrepancies": [
                    {"type": "IN_HRIS_NOT_PAYROLL", "count": 12},
                    {"type": "IN_PAYROLL_NOT_HRIS", "count": 4},
                ],
                "severity": "low" if 0.64 < 2.0 else "high",
                "recommendations": [
                    "Review 12 HRIS records missing from payroll",
                    "Investigate 4 payroll ghost employees",
                ],
            }

            return {
                "reconciliation_result": result,
                "current_agent": "supervisor_check",
                "next_agent": "drift_detection",
                "messages": [
                    HumanMessage(content=f"Reconciliation complete: {result['gap_percentage']}% gap"),
                ],
            }

        except Exception as e:
            logger.error(f"Reconciliation agent failed: {e}")
            return {
                "errors": [f"Reconciliation failed: {str(e)}"],
                "requires_escalation": True,
                "escalation_reason": str(e),
            }

    def _run_drift_detection(self, state: HRMetricState) -> Dict[str, Any]:
        """Execute drift detection agent."""
        logger.info("Running Drift Detection Agent")

        try:
            result = {
                "metrics_analyzed": ["turnover_rate", "headcount", "engagement"],
                "anomalies_detected": [
                    {
                        "metric": "turnover_rate",
                        "department": "Engineering",
                        "z_score": 2.3,
                        "severity": "warning",
                    }
                ],
                "overall_status": "warning",
            }

            return {
                "drift_detection_result": result,
                "current_agent": "supervisor_check",
                "next_agent": "bias_adjustment",
                "messages": [
                    HumanMessage(content=f"Drift detection: {len(result['anomalies_detected'])} anomalies found"),
                ],
            }

        except Exception as e:
            logger.error(f"Drift detection agent failed: {e}")
            return {
                "errors": [f"Drift detection failed: {str(e)}"],
                "next_agent": "bias_adjustment",  # Continue despite error
            }

    def _run_bias_adjustment(self, state: HRMetricState) -> Dict[str, Any]:
        """Execute bias adjustment agent."""
        logger.info("Running Bias Adjustment Agent")

        try:
            result = {
                "departments_adjusted": 12,
                "avg_adjustment": -0.18,
                "low_confidence_departments": ["Sales", "Marketing"],
                "adjusted_company_score": 3.72,
            }

            return {
                "bias_adjustment_result": result,
                "current_agent": "supervisor_check",
                "next_agent": "reporting",
                "messages": [
                    HumanMessage(content=f"Bias adjustment: company score {result['adjusted_company_score']}"),
                ],
            }

        except Exception as e:
            logger.error(f"Bias adjustment agent failed: {e}")
            return {
                "errors": [f"Bias adjustment failed: {str(e)}"],
                "next_agent": "reporting",
            }

    def _generate_reports(self, state: HRMetricState) -> Dict[str, Any]:
        """Execute reporting agent to generate outputs."""
        logger.info("Running Reporting Agent")

        reports = {
            "executive_summary": "HR metrics within acceptable ranges. Minor reconciliation discrepancies identified.",
            "hr_ops_actions": "12 HRIS records require payroll setup verification.",
            "audit_log": json.dumps({
                "run_id": state["run_id"],
                "timestamp": datetime.now().isoformat(),
                "reconciliation": state.get("reconciliation_result"),
                "drift": state.get("drift_detection_result"),
                "bias": state.get("bias_adjustment_result"),
            }),
        }

        return {
            "reports_generated": reports,
            "current_agent": "finalize",
        }

    def _supervisor_check(self, state: HRMetricState) -> Dict[str, Any]:
        """Supervisor agent evaluates progress and routes next steps."""
        logger.info(f"Supervisor checking after: {state.get('current_agent')}")

        # Check for critical issues requiring escalation
        recon = state.get("reconciliation_result", {})
        if recon.get("gap_percentage", 0) > 5.0:
            return {
                "requires_escalation": True,
                "escalation_reason": f"Critical headcount gap: {recon['gap_percentage']}%",
            }

        # Route to next agent
        return {"current_agent": "supervisor_check"}

    def _route_after_supervisor(self, state: HRMetricState) -> str:
        """Determine next node after supervisor check."""

        if state.get("requires_escalation"):
            return "escalate"

        next_agent = state.get("next_agent")

        if next_agent == "drift_detection":
            return "drift_detection"
        elif next_agent == "bias_adjustment":
            return "bias_adjustment"
        elif next_agent == "reporting":
            return "reporting"
        else:
            return "finalize"

    def _escalate_to_human(self, state: HRMetricState) -> Dict[str, Any]:
        """Handle escalation to human operators."""
        logger.warning(f"Escalating to human: {state.get('escalation_reason')}")

        # In production, send Slack/email alert
        return {
            "status": "escalated",
            "messages": [
                HumanMessage(content=f"ESCALATED: {state.get('escalation_reason')}"),
            ],
        }

    def _finalize_run(self, state: HRMetricState) -> Dict[str, Any]:
        """Finalize the validation run and persist results."""
        logger.info(f"Finalizing run: {state.get('run_id')}")

        # Persist to Redis for audit trail
        self.redis_client.setex(
            f"hr-validation:{state['run_id']}",
            86400 * 30,  # 30 day retention
            json.dumps({
                "run_id": state["run_id"],
                "run_date": state["run_date"],
                "status": "completed" if not state.get("errors") else "completed_with_errors",
                "reconciliation": state.get("reconciliation_result"),
                "drift": state.get("drift_detection_result"),
                "bias": state.get("bias_adjustment_result"),
                "reports": state.get("reports_generated"),
                "errors": state.get("errors", []),
            })
        )

        return {"status": "completed"}

    def run(self, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute the full validation workflow."""
        config = config or {}
        thread_id = config.get("thread_id", f"thread-{datetime.now().strftime('%Y%m%d')}")

        initial_state: HRMetricState = {
            "run_id": "",
            "run_date": "",
            "status": "pending",
            "reconciliation_result": None,
            "drift_detection_result": None,
            "bias_adjustment_result": None,
            "reports_generated": None,
            "errors": [],
            "messages": [],
            "current_agent": "",
            "next_agent": None,
            "requires_escalation": False,
            "escalation_reason": None,
        }

        result = self.graph.invoke(
            initial_state,
            {"configurable": {"thread_id": thread_id}},
        )

        return result


# Usage
if __name__ == "__main__":
    workflow = HRMetricWorkflow()
    result = workflow.run()
    print(f"Workflow completed: {result['status']}")`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Production deployment configuration with Docker, LangSmith tracing for agent observability, and Prometheus metrics for operational monitoring.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose Production Deployment',
                description:
                  'Docker Compose configuration for deploying the HR metric validation multi-agent system with Redis state management and observability stack.',
                code: `version: '3.8'

services:
  hr-metric-agents:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: hr-metric-agents
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGCHAIN_API_KEY=\${LANGCHAIN_API_KEY}
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_PROJECT=hr-metric-validation
      - BAMBOOHR_API_KEY=\${BAMBOOHR_API_KEY}
      - BAMBOOHR_SUBDOMAIN=\${BAMBOOHR_SUBDOMAIN}
      - ADP_CLIENT_ID=\${ADP_CLIENT_ID}
      - ADP_CLIENT_SECRET=\${ADP_CLIENT_SECRET}
      - DATABASE_URL=postgresql://\${DB_USER}:\${DB_PASSWORD}@postgres:5432/hr_analytics
      - REDIS_URL=redis://redis:6379
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
      - LOG_LEVEL=INFO
    depends_on:
      - redis
      - postgres
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
    restart: unless-stopped
    networks:
      - hr-metrics-net
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  redis:
    image: redis:7-alpine
    container_name: hr-metrics-redis
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    networks:
      - hr-metrics-net
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: hr-metrics-postgres
    environment:
      - POSTGRES_USER=\${DB_USER}
      - POSTGRES_PASSWORD=\${DB_PASSWORD}
      - POSTGRES_DB=hr_analytics
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./sql/init:/docker-entrypoint-initdb.d:ro
    networks:
      - hr-metrics-net
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: hr-metrics-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "9090:9090"
    networks:
      - hr-metrics-net
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: hr-metrics-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    networks:
      - hr-metrics-net
    restart: unless-stopped

  # Scheduler for daily runs
  scheduler:
    image: mcuadros/ofelia:latest
    container_name: hr-metrics-scheduler
    depends_on:
      - hr-metric-agents
    command: daemon --docker
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    labels:
      ofelia.job-exec.hr-validation.schedule: "0 6 * * *"
      ofelia.job-exec.hr-validation.command: "python -m hr_metrics.run_validation"
      ofelia.job-exec.hr-validation.container: "hr-metric-agents"
    networks:
      - hr-metrics-net
    restart: unless-stopped

volumes:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:

networks:
  hr-metrics-net:
    driver: bridge`,
              },
              {
                language: 'python',
                title: 'LangSmith Tracing & Prometheus Metrics',
                description:
                  'Observability instrumentation for tracking agent execution, latency, and success rates with LangSmith tracing and Prometheus metrics.',
                code: `"""Observability instrumentation for HR Metric Validation Agents."""
import functools
import logging
import time
from typing import Any, Callable, Dict, Optional

from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Prometheus Metrics ---

AGENT_RUNS_TOTAL = Counter(
    "hr_agent_runs_total",
    "Total number of agent executions",
    ["agent_name", "status"],
)

AGENT_DURATION_SECONDS = Histogram(
    "hr_agent_duration_seconds",
    "Agent execution duration in seconds",
    ["agent_name"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

RECONCILIATION_GAP_PCT = Gauge(
    "hr_reconciliation_gap_percentage",
    "Current HRIS-payroll headcount gap percentage",
)

ENGAGEMENT_SCORE = Gauge(
    "hr_engagement_score_adjusted",
    "Company-wide bias-adjusted engagement score",
)

ANOMALIES_DETECTED = Gauge(
    "hr_metric_anomalies_detected",
    "Number of metric anomalies detected in last run",
)

WORKFLOW_STATUS = Gauge(
    "hr_workflow_last_status",
    "Status of last workflow run (1=success, 0=failure)",
)


# --- LangSmith Tracing ---

langsmith_client = Client()


def traced_agent(agent_name: str) -> Callable:
    """Decorator for tracing agent executions with LangSmith and Prometheus."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        @traceable(name=agent_name, run_type="chain")
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            status = "success"

            try:
                result = func(*args, **kwargs)

                # Record agent-specific metrics
                if agent_name == "reconciliation_agent" and result:
                    gap = result.get("gap_percentage", 0)
                    RECONCILIATION_GAP_PCT.set(gap)

                elif agent_name == "bias_adjustment_agent" and result:
                    score = result.get("adjusted_company_score", 0)
                    ENGAGEMENT_SCORE.set(score)

                elif agent_name == "drift_detection_agent" and result:
                    anomalies = len(result.get("anomalies_detected", []))
                    ANOMALIES_DETECTED.set(anomalies)

                return result

            except Exception as e:
                status = "error"
                logger.error(f"{agent_name} failed: {e}")

                # Add error to LangSmith trace
                run_tree = get_current_run_tree()
                if run_tree:
                    run_tree.end(error=str(e))

                raise

            finally:
                duration = time.time() - start_time
                AGENT_RUNS_TOTAL.labels(agent_name=agent_name, status=status).inc()
                AGENT_DURATION_SECONDS.labels(agent_name=agent_name).observe(duration)
                logger.info(f"{agent_name} completed in {duration:.2f}s with status={status}")

        return wrapper
    return decorator


class WorkflowObserver:
    """Observer for tracking overall workflow execution."""

    def __init__(self, prometheus_port: int = 8000):
        """Initialize observer and start Prometheus metrics server."""
        self.prometheus_port = prometheus_port
        self._metrics_started = False

    def start_metrics_server(self) -> None:
        """Start Prometheus metrics HTTP server."""
        if not self._metrics_started:
            start_http_server(self.prometheus_port)
            self._metrics_started = True
            logger.info(f"Prometheus metrics available at :{self.prometheus_port}/metrics")

    @traceable(name="hr_metric_validation_workflow", run_type="chain")
    def observe_workflow(
        self,
        workflow_func: Callable,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute and observe a complete workflow run."""

        start_time = time.time()

        try:
            result = workflow_func(*args, **kwargs)

            status = result.get("status", "unknown")
            WORKFLOW_STATUS.set(1 if status == "completed" else 0)

            # Log summary to LangSmith
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.metadata = {
                    "run_id": result.get("run_id"),
                    "duration_seconds": time.time() - start_time,
                    "reconciliation_gap": result.get("reconciliation_result", {}).get("gap_percentage"),
                    "anomalies_detected": len(result.get("drift_detection_result", {}).get("anomalies_detected", [])),
                    "engagement_score": result.get("bias_adjustment_result", {}).get("adjusted_company_score"),
                }

            return result

        except Exception as e:
            WORKFLOW_STATUS.set(0)
            logger.error(f"Workflow failed: {e}")
            raise

    def record_custom_metric(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a custom metric value for dashboards."""
        # Extend with custom Prometheus gauges as needed
        logger.info(f"Custom metric: {metric_name}={value}, labels={labels}")


# Usage example
if __name__ == "__main__":
    observer = WorkflowObserver(prometheus_port=8000)
    observer.start_metrics_server()

    @traced_agent("reconciliation_agent")
    def run_reconciliation():
        # Simulated reconciliation
        time.sleep(1)
        return {"gap_percentage": 1.5, "discrepancies": []}

    result = run_reconciliation()
    print(f"Reconciliation result: {result}")`,
              },
            ],
          },
        ],
      },
    },

    // ── Pain Point 2: Employee Data Fragmentation ────────────────────────
    {
      id: 'employee-data-fragmentation',
      number: 2,
      title: 'Employee Data Fragmentation',
      subtitle: 'HRIS, Payroll, Benefits, and LMS Data Silos',
      summary:
        'Employee records exist in BambooHR, ADP, benefits portals, and your LMS. A simple "how many employees completed training?" takes 2 days to answer.',
      tags: ['hris', 'integration', 'employee-data'],
      metrics: {
        annualCostRange: '$200K - $1M',
        roi: '6x',
        paybackPeriod: '3-5 months',
        investmentRange: '$50K - $100K',
      },
      price: {
        present: {
          title: 'Current State of Employee Data',
          description:
            'Every HR question that spans more than one system triggers a multi-day scavenger hunt through disconnected platforms.',
          bullets: [
            'BambooHR holds demographics and org structure, ADP holds payroll and tax, benefits portal holds elections — none share a common key reliably',
            'Answering "which employees in Engineering completed compliance training?" requires manual exports from 3 systems and VLOOKUP matching',
            'New hire onboarding creates records in 5+ systems; mismatches are discovered weeks later during the first payroll run',
            'Terminated employees linger as active in the LMS and benefits portal for months because there is no automated deprovisioning',
          ],
          severity: 'high',
        },
        root: {
          title: 'Why Employee Data Is Fragmented',
          description:
            'Each HR sub-function selected its own best-of-breed platform, and no integration layer was ever built to keep them in sync.',
          bullets: [
            'No master employee identifier — BambooHR uses its own ID, ADP uses a payroll ID, and the LMS uses email addresses',
            'Point-to-point integrations were built ad hoc and break silently when any system updates its API',
            'HR operations team lacks engineering support; "integrations" are manual CSV uploads on a weekly schedule',
            'No single owner is accountable for cross-system data consistency — each platform has a different admin',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Business Impact of Fragmented Employee Data',
          description:
            'Data silos slow down every people process — from onboarding and compliance to workforce planning and offboarding.',
          bullets: [
            'Compliance auditors flagged the company because 12% of "active" employees in the benefits system had actually been terminated',
            'HR business partners spend 8-10 hours per week assembling cross-system reports instead of advising managers',
            'A department re-org took 6 weeks to propagate across all systems, creating 3 months of misattributed cost-center charges',
            'Executives receive conflicting training-completion rates depending on which system the report was pulled from',
          ],
          severity: 'high',
        },
        cost: {
          title: 'Cost of Employee Data Silos',
          description:
            'The labour cost of manual reconciliation and the risk cost of stale records add up quickly across the HR function.',
          bullets: [
            'HR team spends an estimated 500+ hours per year on manual cross-system data pulls and reconciliation',
            'Benefits over-payment for terminated employees costs $50K-$200K annually before detection',
            'Compliance gaps from stale LMS records create audit findings that cost $25K-$100K to remediate',
          ],
          severity: 'medium',
        },
        expectedReturn: {
          title: 'Value of a Unified Employee Data Layer',
          description:
            'A single integration hub with a master employee record eliminates reconciliation work and enables real-time cross-system reporting.',
          bullets: [
            'Master employee ID mapping eliminates VLOOKUP gymnastics — any cross-system question answered in minutes',
            'Automated termination propagation closes benefits and LMS access within 24 hours, stopping over-payments immediately',
            'HR business partners reclaim 8+ hours per week for strategic advising instead of data wrangling',
            'Single pane of glass for compliance: training completion, benefits enrollment, and payroll status in one query',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Build a unified employee data model with a master identity crosswalk, then create an automated integration pipeline that syncs records across HRIS, payroll, benefits, and LMS.',
        prerequisites: [
          'API access to BambooHR (or primary HRIS)',
          'API or SFTP access to ADP payroll exports',
          'LMS database read access or API credentials',
          'PostgreSQL or Snowflake for the unified data layer',
          'Python 3.9+ with requests, pandas, sqlalchemy',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduled reconciliation',
          'Slack incoming webhook URL for alerting',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Unified Employee Data Model',
            description:
              'Create a master identity crosswalk and a canonical employee profile that stitches together records from every HR system.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Master Employee Identity & Profile',
                description:
                  'Schema for the employee identity crosswalk and the unified profile view that joins HRIS, payroll, benefits, and LMS data.',
                code: `-- Master identity crosswalk: maps every system ID to one person
CREATE TABLE IF NOT EXISTS hr_unified.employee_identity (
    master_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bamboohr_id      INTEGER UNIQUE,
    adp_payroll_id   VARCHAR(20) UNIQUE,
    benefits_ssn_hash VARCHAR(64),     -- SHA-256 of SSN for matching
    lms_email        VARCHAR(255) UNIQUE,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_identity_bamboo ON hr_unified.employee_identity(bamboohr_id);
CREATE INDEX idx_identity_adp    ON hr_unified.employee_identity(adp_payroll_id);
CREATE INDEX idx_identity_lms    ON hr_unified.employee_identity(lms_email);

-- Unified employee profile: single view across all systems
CREATE OR REPLACE VIEW hr_unified.v_employee_profile AS
SELECT
    ei.master_id,
    -- HRIS fields
    h.full_name,
    h.department,
    h.job_title,
    h.hire_date,
    h.termination_date,
    h.employment_status  AS hris_status,
    -- Payroll fields
    p.pay_frequency,
    p.annual_salary,
    p.last_pay_date,
    p.pay_status         AS payroll_status,
    -- Benefits fields
    b.medical_plan,
    b.dental_enrolled,
    b.benefits_status,
    -- LMS fields
    l.courses_completed,
    l.last_training_date,
    l.compliance_current,
    -- Data quality flags
    CASE WHEN h.employment_status = 'terminated'
              AND b.benefits_status = 'active'
         THEN TRUE ELSE FALSE
    END AS benefits_leak_flag,
    CASE WHEN h.employment_status = 'terminated'
              AND l.compliance_current = TRUE
         THEN TRUE ELSE FALSE
    END AS lms_stale_flag
FROM hr_unified.employee_identity ei
LEFT JOIN hris.employees h        ON ei.bamboohr_id    = h.id
LEFT JOIN payroll.employees p     ON ei.adp_payroll_id = p.employee_id
LEFT JOIN benefits.enrollments b  ON ei.benefits_ssn_hash = b.ssn_hash
LEFT JOIN lms.learner_profiles l  ON ei.lms_email      = l.email;`,
              },
              {
                language: 'sql',
                title: 'Cross-System Compliance Query',
                description:
                  'Example query that answers "which active Engineering employees have NOT completed compliance training?" in seconds instead of days.',
                code: `-- Previously a 2-day, 3-spreadsheet effort — now a single query
SELECT
    ep.master_id,
    ep.full_name,
    ep.department,
    ep.job_title,
    ep.hris_status,
    ep.compliance_current,
    ep.last_training_date,
    CURRENT_DATE - ep.last_training_date AS days_since_training
FROM hr_unified.v_employee_profile ep
WHERE ep.department = 'Engineering'
  AND ep.hris_status = 'active'
  AND (ep.compliance_current = FALSE OR ep.compliance_current IS NULL)
ORDER BY ep.last_training_date ASC NULLS FIRST;

-- Stale-record cleanup report: terminated employees still active elsewhere
SELECT
    ep.master_id,
    ep.full_name,
    ep.termination_date,
    ep.benefits_leak_flag,
    ep.lms_stale_flag,
    ep.benefits_status,
    ep.payroll_status
FROM hr_unified.v_employee_profile ep
WHERE ep.hris_status = 'terminated'
  AND (ep.benefits_leak_flag = TRUE OR ep.lms_stale_flag = TRUE)
ORDER BY ep.termination_date ASC;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'HRIS Integration Pipeline',
            description:
              'Python pipeline that pulls data from BambooHR and ADP APIs, resolves identities, and loads into the unified data layer on a daily schedule.',
            codeSnippets: [
              {
                language: 'python',
                title: 'BambooHR & ADP Sync Pipeline',
                description:
                  'Fetches employee records from BambooHR and ADP, matches them via the identity crosswalk, and upserts into the unified schema.',
                code: `"""Daily HRIS integration pipeline: BambooHR + ADP -> unified layer."""
import os, hashlib
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import date

BAMBOO_API = os.environ["BAMBOOHR_API_KEY"]
BAMBOO_DOMAIN = os.environ["BAMBOOHR_SUBDOMAIN"]
ADP_CLIENT_ID = os.environ["ADP_CLIENT_ID"]
ADP_CLIENT_SECRET = os.environ["ADP_CLIENT_SECRET"]

engine = create_engine(os.environ["DATABASE_URL"])

def fetch_bamboohr_employees() -> pd.DataFrame:
    url = f"https://api.bamboohr.com/api/gateway.php/{BAMBOO_DOMAIN}/v1/employees/directory"
    resp = requests.get(url, headers={"Accept": "application/json"},
                        auth=(BAMBOO_API, "x"), timeout=30)
    resp.raise_for_status()
    employees = resp.json().get("employees", [])
    df = pd.json_normalize(employees)
    df = df.rename(columns={
        "id": "bamboohr_id", "displayName": "full_name",
        "department": "department", "jobTitle": "job_title",
        "workEmail": "work_email", "hireDate": "hire_date",
    })
    return df[["bamboohr_id", "full_name", "department",
               "job_title", "work_email", "hire_date"]]

def fetch_adp_employees() -> pd.DataFrame:
    token_resp = requests.post(
        "https://accounts.adp.com/auth/oauth/v2/token",
        data={"grant_type": "client_credentials",
               "client_id": ADP_CLIENT_ID,
               "client_secret": ADP_CLIENT_SECRET},
        timeout=30,
    )
    token_resp.raise_for_status()
    token = token_resp.json()["access_token"]
    emp_resp = requests.get(
        "https://api.adp.com/hr/v2/workers",
        headers={"Authorization": f"Bearer {token}"}, timeout=60,
    )
    emp_resp.raise_for_status()
    workers = emp_resp.json().get("workers", [])
    records = [{
        "adp_payroll_id": w["workerID"]["idValue"],
        "legal_name": w["person"]["legalName"]["formattedName"],
        "ssn_hash": hashlib.sha256(
            w["person"].get("governmentIDs", [{}])[0]
            .get("idValue", "").encode()
        ).hexdigest(),
        "pay_status": w["workerStatus"]["statusCode"]["codeValue"],
    } for w in workers]
    return pd.DataFrame(records)

def upsert_identities(bamboo_df: pd.DataFrame, adp_df: pd.DataFrame):
    merged = bamboo_df.merge(
        adp_df, left_on="work_email", right_on="legal_name",
        how="outer", suffixes=("_b", "_a"),
    )
    with engine.begin() as conn:
        for _, row in merged.iterrows():
            conn.execute(text("""
                INSERT INTO hr_unified.employee_identity
                    (bamboohr_id, adp_payroll_id, benefits_ssn_hash, lms_email)
                VALUES (:bid, :aid, :ssn, :email)
                ON CONFLICT (bamboohr_id) DO UPDATE SET
                    adp_payroll_id = EXCLUDED.adp_payroll_id,
                    benefits_ssn_hash = EXCLUDED.benefits_ssn_hash,
                    lms_email = EXCLUDED.lms_email,
                    updated_at = NOW()
            """), {
                "bid": row.get("bamboohr_id"),
                "aid": row.get("adp_payroll_id"),
                "ssn": row.get("ssn_hash"),
                "email": row.get("work_email"),
            })

def run_daily_sync():
    print(f"[{date.today()}] Starting HRIS sync...")
    bamboo = fetch_bamboohr_employees()
    adp = fetch_adp_employees()
    print(f"  BambooHR: {len(bamboo)} | ADP: {len(adp)}")
    upsert_identities(bamboo, adp)
    print("  Identity crosswalk updated.")

if __name__ == "__main__":
    run_daily_sync()`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Data quality assertions for the identity crosswalk and automated pytest validation of BambooHR/ADP sync logic ensure unified records are trustworthy.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Identity Crosswalk Validation Assertions',
                description:
                  'SQL assertions that detect orphan records, duplicate identity mappings, and missing cross-system keys in the employee identity crosswalk.',
                code: `-- Assertion 1: No orphan crosswalk entries (master_id without any source key)
SELECT
    'orphan_crosswalk_entry' AS assertion,
    COUNT(*) AS violations
FROM hr_unified.employee_identity
WHERE bamboohr_id    IS NULL
  AND adp_payroll_id IS NULL
  AND lms_email      IS NULL;
-- Expected: 0 — every master_id must link to at least one source system

-- Assertion 2: No duplicate BambooHR IDs in crosswalk
SELECT
    'duplicate_bamboohr_id' AS assertion,
    bamboohr_id,
    COUNT(*) AS occurrences
FROM hr_unified.employee_identity
WHERE bamboohr_id IS NOT NULL
GROUP BY bamboohr_id
HAVING COUNT(*) > 1;
-- Expected: 0 rows

-- Assertion 3: No duplicate ADP payroll IDs in crosswalk
SELECT
    'duplicate_adp_payroll_id' AS assertion,
    adp_payroll_id,
    COUNT(*) AS occurrences
FROM hr_unified.employee_identity
WHERE adp_payroll_id IS NOT NULL
GROUP BY adp_payroll_id
HAVING COUNT(*) > 1;
-- Expected: 0 rows

-- Assertion 4: Active HRIS employees must have a payroll match
SELECT
    'active_without_payroll_link' AS assertion,
    ep.master_id,
    ep.full_name,
    ep.department,
    ep.hris_status,
    ep.payroll_status
FROM hr_unified.v_employee_profile ep
WHERE ep.hris_status = 'active'
  AND ep.payroll_status IS NULL;
-- Expected: 0 rows — all active employees should be in payroll

-- Assertion 5: Terminated employees with stale downstream access
SELECT
    'terminated_still_active_downstream' AS assertion,
    COUNT(*) AS violations,
    SUM(CASE WHEN benefits_leak_flag THEN 1 ELSE 0 END) AS benefits_leaks,
    SUM(CASE WHEN lms_stale_flag THEN 1 ELSE 0 END) AS lms_stale
FROM hr_unified.v_employee_profile
WHERE hris_status = 'terminated'
  AND (benefits_leak_flag = TRUE OR lms_stale_flag = TRUE);
-- Expected: 0 violations (target SLA)`,
              },
              {
                language: 'python',
                title: 'Pytest Validation for BambooHR/ADP Sync Logic',
                description:
                  'Automated pytest suite that validates identity crosswalk integrity, sync pipeline correctness, and cross-system record matching after each daily run.',
                code: `"""Pytest suite for BambooHR/ADP sync pipeline validation."""
import logging
from datetime import date
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

ENGINE = create_engine(
    "postgresql://analytics:***@db-host:5432/people_analytics"
)


@pytest.fixture(scope="module")
def db_connection():
    """Provide a reusable database connection for all tests."""
    conn = ENGINE.connect()
    yield conn
    conn.close()


class TestIdentityCrosswalk:
    """Validate identity crosswalk integrity after sync."""

    def test_no_orphan_master_ids(self, db_connection) -> None:
        """Every master_id must link to at least one source system."""
        result: pd.DataFrame = pd.read_sql(
            text(
                "SELECT COUNT(*) AS orphans "
                "FROM hr_unified.employee_identity "
                "WHERE bamboohr_id IS NULL "
                "  AND adp_payroll_id IS NULL "
                "  AND lms_email IS NULL"
            ),
            db_connection,
        )
        orphans: int = int(result.iloc[0]["orphans"])
        logger.info("Orphan crosswalk entries: %d", orphans)
        assert orphans == 0, f"Found {orphans} orphan master_id entries"

    def test_no_duplicate_bamboohr_ids(self, db_connection) -> None:
        """BambooHR IDs must be unique across the crosswalk."""
        result: pd.DataFrame = pd.read_sql(
            text(
                "SELECT bamboohr_id, COUNT(*) AS cnt "
                "FROM hr_unified.employee_identity "
                "WHERE bamboohr_id IS NOT NULL "
                "GROUP BY bamboohr_id HAVING COUNT(*) > 1"
            ),
            db_connection,
        )
        logger.info("Duplicate BambooHR IDs: %d", len(result))
        assert result.empty, (
            f"Duplicate bamboohr_ids: {result['bamboohr_id'].tolist()}"
        )

    def test_no_duplicate_adp_ids(self, db_connection) -> None:
        """ADP payroll IDs must be unique across the crosswalk."""
        result: pd.DataFrame = pd.read_sql(
            text(
                "SELECT adp_payroll_id, COUNT(*) AS cnt "
                "FROM hr_unified.employee_identity "
                "WHERE adp_payroll_id IS NOT NULL "
                "GROUP BY adp_payroll_id HAVING COUNT(*) > 1"
            ),
            db_connection,
        )
        logger.info("Duplicate ADP IDs: %d", len(result))
        assert result.empty, (
            f"Duplicate adp_payroll_ids: {result['adp_payroll_id'].tolist()}"
        )

    def test_active_employees_have_payroll(self, db_connection) -> None:
        """Active HRIS employees must be linked to a payroll record."""
        result: pd.DataFrame = pd.read_sql(
            text(
                "SELECT COUNT(*) AS unlinked "
                "FROM hr_unified.v_employee_profile "
                "WHERE hris_status = 'active' "
                "  AND payroll_status IS NULL"
            ),
            db_connection,
        )
        unlinked: int = int(result.iloc[0]["unlinked"])
        logger.info("Active employees without payroll link: %d", unlinked)
        assert unlinked == 0, (
            f"{unlinked} active employees missing payroll record"
        )


class TestSyncPipeline:
    """Validate the daily BambooHR/ADP sync pipeline output."""

    def test_crosswalk_updated_today(self, db_connection) -> None:
        """At least one crosswalk record should be updated today."""
        result: pd.DataFrame = pd.read_sql(
            text(
                "SELECT COUNT(*) AS updated "
                "FROM hr_unified.employee_identity "
                "WHERE updated_at::DATE = CURRENT_DATE"
            ),
            db_connection,
        )
        updated: int = int(result.iloc[0]["updated"])
        logger.info("Crosswalk records updated today: %d", updated)
        assert updated > 0, "Sync pipeline did not update any records today"

    def test_no_stale_terminated_records(self, db_connection) -> None:
        """Terminated employees should not have active downstream access."""
        result: pd.DataFrame = pd.read_sql(
            text(
                "SELECT COUNT(*) AS stale "
                "FROM hr_unified.v_employee_profile "
                "WHERE hris_status = 'terminated' "
                "  AND (benefits_leak_flag = TRUE OR lms_stale_flag = TRUE)"
            ),
            db_connection,
        )
        stale: int = int(result.iloc[0]["stale"])
        logger.info("Stale terminated records: %d", stale)
        # Warn but do not fail — deprovisioning may lag by SLA
        if stale > 0:
            logger.warning(
                "%d terminated employees still active in downstream systems",
                stale,
            )`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Production deployment scripts and configuration management for the BambooHR/ADP sync pipeline with secure API credential handling.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'BambooHR/ADP Sync Cron Deployment Script',
                description:
                  'Bash deployment script that builds the sync pipeline container, runs integration tests, installs the daily cron schedule, and notifies Slack on deployment.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
PROJECT_DIR="/opt/hr-unified-sync"
DOCKER_IMAGE="hr-unified-sync:latest"
CRON_SCHEDULE="0 5 * * *"    # 5:00 AM daily — before business hours
LOG_DIR="/var/log/hr-sync"
SLACK_WEBHOOK="\${SLACK_WEBHOOK_URL:?'SLACK_WEBHOOK_URL must be set'}"
ENV_FILE="\${PROJECT_DIR}/config/.env"

echo "[INFO] Starting HR unified sync pipeline deployment..."
echo "[INFO] Project directory: \${PROJECT_DIR}"
echo "[INFO] Docker image: \${DOCKER_IMAGE}"

# --- Validate required secrets exist ---
if [[ ! -f "\${ENV_FILE}" ]]; then
    echo "[ERROR] Environment file not found: \${ENV_FILE}"
    exit 1
fi

# Check required API keys are present in env file
for KEY in BAMBOOHR_API_KEY BAMBOOHR_SUBDOMAIN ADP_CLIENT_ID ADP_CLIENT_SECRET DATABASE_URL; do
    if ! grep -q "^\${KEY}=" "\${ENV_FILE}"; then
        echo "[ERROR] Missing required key in \${ENV_FILE}: \${KEY}"
        exit 1
    fi
done
echo "[INFO] All required API credentials validated."

# --- Ensure directories exist ---
mkdir -p "\${LOG_DIR}"
mkdir -p "\${PROJECT_DIR}/config"

# --- Build Docker image ---
echo "[INFO] Building Docker image..."
docker build \\
    --tag "\${DOCKER_IMAGE}" \\
    --build-arg BUILD_DATE="\$(date -u +%Y-%m-%dT%H:%M:%SZ)" \\
    --file "\${PROJECT_DIR}/Dockerfile" \\
    "\${PROJECT_DIR}"

# --- Run pytest integration tests inside container ---
echo "[INFO] Running sync pipeline validation tests..."
docker run --rm \\
    --env-file "\${ENV_FILE}" \\
    "\${DOCKER_IMAGE}" \\
    python -m pytest tests/ -v --tb=short --junitxml=/tmp/test-results.xml

echo "[INFO] All validation tests passed."

# --- Install cron job for daily sync ---
CRON_CMD="\${CRON_SCHEDULE} docker run --rm --env-file \${ENV_FILE} \${DOCKER_IMAGE} python -m hr_sync.pipeline >> \${LOG_DIR}/daily_sync.log 2>&1"
CRON_MARKER="# hr-unified-daily-sync"

(crontab -l 2>/dev/null | grep -v "\${CRON_MARKER}" || true; echo "\${CRON_CMD} \${CRON_MARKER}") | crontab -
echo "[INFO] Cron job installed: \${CRON_SCHEDULE}"

# --- Send Slack deployment notification ---
DEPLOY_MSG="HR Unified Sync Pipeline deployed at \$(date -u +%Y-%m-%dT%H:%M:%SZ). Daily sync: \${CRON_SCHEDULE}. Systems: BambooHR, ADP."
curl -sf -X POST "\${SLACK_WEBHOOK}" \\
    -H "Content-Type: application/json" \\
    -d "{\\"text\\": \\"\${DEPLOY_MSG}\\"}" \\
    || echo "[WARN] Slack notification failed (non-fatal)"

echo "[INFO] Deployment complete."`,
              },
              {
                language: 'python',
                title: 'Configuration Loader with API Credential Management',
                description:
                  'Configuration loader that securely reads BambooHR/ADP API credentials and pipeline settings from environment variables with validation.',
                code: `"""Configuration loader with API credential management for HR sync."""
import logging
import os
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection settings."""
    host: str
    port: int
    database: str
    username: str
    password: str
    schema: str = "hr_unified"

    @property
    def connection_url(self) -> str:
        return (
            f"postgresql://\${self.username}:\${self.password}"
            f"@\${self.host}:\${self.port}/\${self.database}"
        )


@dataclass(frozen=True)
class BambooHRConfig:
    """BambooHR API credentials and settings."""
    api_key: str
    subdomain: str
    base_url: str = "https://api.bamboohr.com/api/gateway.php"

    @property
    def directory_url(self) -> str:
        return f"\${self.base_url}/\${self.subdomain}/v1/employees/directory"


@dataclass(frozen=True)
class ADPConfig:
    """ADP API credentials and settings."""
    client_id: str
    client_secret: str
    token_url: str = "https://accounts.adp.com/auth/oauth/v2/token"
    workers_url: str = "https://api.adp.com/hr/v2/workers"


@dataclass(frozen=True)
class AlertConfig:
    """Alerting and notification settings."""
    slack_webhook_url: str
    alert_channel: str = "#hr-data-alerts"
    stale_record_threshold_days: int = 7
    sync_failure_notify: bool = True


@dataclass(frozen=True)
class SyncPipelineConfig:
    """Top-level configuration for the HR sync pipeline."""
    db: DatabaseConfig
    bamboohr: BambooHRConfig
    adp: ADPConfig
    alerts: AlertConfig
    environment: str = "production"
    dry_run: bool = False
    log_level: str = "INFO"


def _require_env(key: str) -> str:
    """Read a required environment variable or raise."""
    value: Optional[str] = os.environ.get(key)
    if not value:
        raise EnvironmentError(f"Required env var {key} is not set")
    return value


def load_sync_config() -> SyncPipelineConfig:
    """Load sync pipeline configuration from environment variables."""
    db_config = DatabaseConfig(
        host=_require_env("DB_HOST"),
        port=int(os.environ.get("DB_PORT", "5432")),
        database=_require_env("DB_NAME"),
        username=_require_env("DB_USER"),
        password=_require_env("DB_PASSWORD"),
        schema=os.environ.get("DB_SCHEMA", "hr_unified"),
    )

    bamboo_config = BambooHRConfig(
        api_key=_require_env("BAMBOOHR_API_KEY"),
        subdomain=_require_env("BAMBOOHR_SUBDOMAIN"),
    )

    adp_config = ADPConfig(
        client_id=_require_env("ADP_CLIENT_ID"),
        client_secret=_require_env("ADP_CLIENT_SECRET"),
    )

    alert_config = AlertConfig(
        slack_webhook_url=_require_env("SLACK_WEBHOOK_URL"),
        alert_channel=os.environ.get("ALERT_CHANNEL", "#hr-data-alerts"),
        stale_record_threshold_days=int(
            os.environ.get("STALE_THRESHOLD_DAYS", "7")
        ),
        sync_failure_notify=os.environ.get(
            "SYNC_FAILURE_NOTIFY", "true"
        ).lower() == "true",
    )

    config = SyncPipelineConfig(
        db=db_config,
        bamboohr=bamboo_config,
        adp=adp_config,
        alerts=alert_config,
        environment=os.environ.get("ENVIRONMENT", "production"),
        dry_run=os.environ.get("DRY_RUN", "false").lower() == "true",
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )

    logger.info(
        "Sync config loaded: env=%s, db=%s/%s, bamboo=%s, dry_run=%s",
        config.environment,
        db_config.host,
        db_config.database,
        bamboo_config.subdomain,
        config.dry_run,
    )
    return config


if __name__ == "__main__":
    cfg: SyncPipelineConfig = load_sync_config()
    logger.info("Database URL: %s", cfg.db.connection_url)
    logger.info("BambooHR directory: %s", cfg.bamboohr.directory_url)
    logger.info("ADP workers endpoint: %s", cfg.adp.workers_url)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Termination Propagation & Data Quality Checks',
            description:
              'Detect terminated employees still active in downstream systems, generate deprovisioning tasks, and monitor sync health via an alerting dashboard.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Stale Record Detector & Deprovisioning Trigger',
                description:
                  'Scans the unified profile for data quality issues — especially terminated employees with active benefits or LMS access — and queues deprovisioning actions.',
                code: `"""Detect stale employee records and trigger deprovisioning."""
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import date

engine = create_engine("postgresql://analytics:***@db-host:5432/people_analytics")

def find_stale_records() -> pd.DataFrame:
    query = text("""
        SELECT master_id, full_name, termination_date,
               benefits_leak_flag, lms_stale_flag,
               benefits_status, payroll_status
        FROM hr_unified.v_employee_profile
        WHERE hris_status = 'terminated'
          AND (benefits_leak_flag = TRUE OR lms_stale_flag = TRUE)
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

def create_deprovision_tasks(stale: pd.DataFrame) -> pd.DataFrame:
    tasks = []
    for _, row in stale.iterrows():
        if row["benefits_leak_flag"]:
            tasks.append({
                "master_id": row["master_id"],
                "employee_name": row["full_name"],
                "system": "benefits_portal",
                "action": "terminate_enrollment",
                "priority": "high",
                "detected_date": date.today(),
                "terminated_date": row["termination_date"],
                "days_overdue": (date.today() - row["termination_date"]).days,
            })
        if row["lms_stale_flag"]:
            tasks.append({
                "master_id": row["master_id"],
                "employee_name": row["full_name"],
                "system": "lms",
                "action": "deactivate_learner",
                "priority": "medium",
                "detected_date": date.today(),
                "terminated_date": row["termination_date"],
                "days_overdue": (date.today() - row["termination_date"]).days,
            })
    return pd.DataFrame(tasks)

def run_stale_check():
    stale = find_stale_records()
    print(f"Stale records found: {len(stale)}")
    if stale.empty:
        print("All terminated employees properly deprovisioned.")
        return
    tasks = create_deprovision_tasks(stale)
    tasks.to_sql(
        "deprovision_queue", engine, schema="hr_unified",
        if_exists="append", index=False,
    )
    overdue_30 = tasks[tasks["days_overdue"] > 30]
    print(f"  Deprovision tasks queued: {len(tasks)}")
    print(f"  Critical (>30 days overdue): {len(overdue_30)}")

if __name__ == "__main__":
    run_stale_check()`,
              },
              {
                language: 'sql',
                title: 'Sync Failure & Stale Record Alerting Dashboard',
                description:
                  'SQL views powering a dashboard that tracks daily sync health, stale record trends, deprovisioning SLA compliance, and sync failure alerting.',
                code: `-- Sync health and stale record alerting dashboard
-- Monitors BambooHR/ADP sync failures and deprovisioning SLA compliance

-- 1. Daily sync run status — detect sync failures and data gaps
CREATE OR REPLACE VIEW hr_unified.v_sync_health AS
WITH daily_sync_stats AS (
    SELECT
        updated_at::DATE AS sync_date,
        COUNT(*) AS records_synced,
        COUNT(DISTINCT bamboohr_id) AS bamboo_records,
        COUNT(DISTINCT adp_payroll_id) AS adp_records,
        COUNT(*) FILTER (
            WHERE bamboohr_id IS NOT NULL AND adp_payroll_id IS NOT NULL
        ) AS fully_linked,
        COUNT(*) FILTER (
            WHERE bamboohr_id IS NOT NULL AND adp_payroll_id IS NULL
        ) AS bamboo_only,
        COUNT(*) FILTER (
            WHERE bamboohr_id IS NULL AND adp_payroll_id IS NOT NULL
        ) AS adp_only
    FROM hr_unified.employee_identity
    WHERE updated_at >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY updated_at::DATE
)
SELECT
    sync_date,
    records_synced,
    bamboo_records,
    adp_records,
    fully_linked,
    bamboo_only,
    adp_only,
    ROUND(
        fully_linked::NUMERIC / NULLIF(records_synced, 0) * 100, 1
    ) AS link_rate_pct,
    CASE
        WHEN records_synced = 0 THEN 'SYNC_FAILURE'
        WHEN fully_linked::NUMERIC / NULLIF(records_synced, 0) * 100 < 90
        THEN 'DEGRADED'
        ELSE 'HEALTHY'
    END AS sync_status
FROM daily_sync_stats
ORDER BY sync_date DESC;

-- 2. Stale record trend — terminated employees still active downstream
CREATE OR REPLACE VIEW hr_unified.v_stale_record_trend AS
SELECT
    dq.detected_date,
    dq.system,
    COUNT(*) AS pending_tasks,
    COUNT(*) FILTER (WHERE dq.days_overdue > 30) AS critical_overdue,
    COUNT(*) FILTER (WHERE dq.days_overdue BETWEEN 8 AND 30) AS warning_overdue,
    COUNT(*) FILTER (WHERE dq.days_overdue <= 7) AS within_sla,
    AVG(dq.days_overdue)::INTEGER AS avg_days_overdue
FROM hr_unified.deprovision_queue dq
LEFT JOIN hr_unified.deprovision_completions dc
    ON dq.master_id = dc.master_id
    AND dq.system = dc.system
WHERE dc.completed_date IS NULL   -- still open
GROUP BY dq.detected_date, dq.system
ORDER BY dq.detected_date DESC, dq.system;

-- 3. Deprovisioning SLA compliance — percentage resolved within 7 days
CREATE OR REPLACE VIEW hr_unified.v_deprovision_sla AS
WITH task_resolution AS (
    SELECT
        dq.master_id,
        dq.system,
        dq.detected_date,
        dc.completed_date,
        COALESCE(dc.completed_date, CURRENT_DATE) - dq.detected_date
            AS resolution_days,
        CASE
            WHEN dc.completed_date IS NOT NULL
                 AND (dc.completed_date - dq.detected_date) <= 7
            THEN 'MET_SLA'
            WHEN dc.completed_date IS NOT NULL
            THEN 'BREACHED_SLA'
            ELSE 'OPEN'
        END AS sla_status
    FROM hr_unified.deprovision_queue dq
    LEFT JOIN hr_unified.deprovision_completions dc
        ON dq.master_id = dc.master_id
        AND dq.system = dc.system
    WHERE dq.detected_date >= CURRENT_DATE - INTERVAL '90 days'
)
SELECT
    DATE_TRUNC('week', detected_date) AS week,
    system,
    COUNT(*) AS total_tasks,
    COUNT(*) FILTER (WHERE sla_status = 'MET_SLA') AS met_sla,
    COUNT(*) FILTER (WHERE sla_status = 'BREACHED_SLA') AS breached_sla,
    COUNT(*) FILTER (WHERE sla_status = 'OPEN') AS still_open,
    ROUND(
        COUNT(*) FILTER (WHERE sla_status = 'MET_SLA')::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE sla_status != 'OPEN'), 0) * 100, 1
    ) AS sla_compliance_pct
FROM task_resolution
GROUP BY 1, 2
ORDER BY week DESC, system;`,
              },
            ],
          },
        ],
        toolsUsed: ['PostgreSQL', 'Python', 'pandas', 'SQLAlchemy', 'BambooHR API', 'ADP API', 'pytest', 'Docker', 'GitHub Actions', 'cron / Airflow', 'Slack API'],
      },
      aiEasyWin: {
        overview:
          'Use ChatGPT or Claude to analyze employee data exports from multiple HR systems, identify identity mismatches, and automate cross-system sync alerts with Zapier connecting HRIS, payroll, LMS, and Slack.',
        estimatedMonthlyCost: '$120 - $200/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Google Sheets (free)', 'Airtable ($20/mo)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Workday AI', 'Culture Amp'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Set up automated exports from BambooHR, ADP, benefits portal, and LMS into a central Google Sheet or Airtable base using Zapier scheduled workflows.',
            toolsUsed: ['Zapier', 'Google Sheets', 'Airtable', 'BambooHR', 'ADP'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Multi-System Employee Data Export Workflow',
                description:
                  'Zapier multi-step workflow that pulls employee records from HRIS, payroll, and LMS into a unified Airtable base for cross-system analysis.',
                code: `{
  "zapier_workflow": {
    "name": "Daily Multi-System Employee Sync",
    "trigger": {
      "app": "Schedule by Zapier",
      "event": "Every Day",
      "config": {
        "time": "05:00",
        "timezone": "America/New_York"
      }
    },
    "actions": [
      {
        "step": 1,
        "app": "BambooHR",
        "event": "Get All Employees",
        "config": {
          "fields": [
            "id",
            "displayName",
            "workEmail",
            "department",
            "jobTitle",
            "employmentStatus",
            "hireDate",
            "terminationDate"
          ],
          "output_key": "hris_employees"
        }
      },
      {
        "step": 2,
        "app": "ADP Workforce Now",
        "event": "Get Workers",
        "config": {
          "status_filter": ["active", "leave", "terminated"],
          "fields": [
            "workerID",
            "legalName",
            "workEmail",
            "departmentCode",
            "payStatus",
            "lastPayDate"
          ],
          "output_key": "payroll_workers"
        }
      },
      {
        "step": 3,
        "app": "Webhooks by Zapier",
        "event": "GET",
        "config": {
          "url": "{{LMS_API_ENDPOINT}}/users",
          "headers": {
            "Authorization": "Bearer {{LMS_API_TOKEN}}"
          },
          "query_params": {
            "status": "all",
            "fields": "email,name,courses_completed,compliance_status"
          },
          "output_key": "lms_learners"
        }
      },
      {
        "step": 4,
        "app": "Airtable",
        "event": "Create or Update Record",
        "config": {
          "base_id": "{{EMPLOYEE_DATA_HUB_BASE}}",
          "table": "HRIS_Records",
          "match_field": "employee_id",
          "fields": {
            "employee_id": "{{hris_employees.id}}",
            "full_name": "{{hris_employees.displayName}}",
            "work_email": "{{hris_employees.workEmail}}",
            "department": "{{hris_employees.department}}",
            "hris_status": "{{hris_employees.employmentStatus}}",
            "last_sync": "{{zap_timestamp}}"
          }
        }
      },
      {
        "step": 5,
        "app": "Airtable",
        "event": "Create or Update Record",
        "config": {
          "base_id": "{{EMPLOYEE_DATA_HUB_BASE}}",
          "table": "Payroll_Records",
          "match_field": "worker_id",
          "fields": {
            "worker_id": "{{payroll_workers.workerID}}",
            "legal_name": "{{payroll_workers.legalName}}",
            "work_email": "{{payroll_workers.workEmail}}",
            "payroll_status": "{{payroll_workers.payStatus}}",
            "last_pay_date": "{{payroll_workers.lastPayDate}}",
            "last_sync": "{{zap_timestamp}}"
          }
        }
      },
      {
        "step": 6,
        "app": "Airtable",
        "event": "Create or Update Record",
        "config": {
          "base_id": "{{EMPLOYEE_DATA_HUB_BASE}}",
          "table": "LMS_Records",
          "match_field": "email",
          "fields": {
            "email": "{{lms_learners.email}}",
            "learner_name": "{{lms_learners.name}}",
            "courses_completed": "{{lms_learners.courses_completed}}",
            "compliance_status": "{{lms_learners.compliance_status}}",
            "last_sync": "{{zap_timestamp}}"
          }
        }
      }
    ]
  }
}`,
              },
              {
                language: 'json',
                title: 'Airtable Unified Employee View Configuration',
                description:
                  'Airtable formula configuration for creating a unified employee view that links records across HRIS, payroll, and LMS tables.',
                code: `{
  "airtable_schema": {
    "base_name": "Employee Data Hub",
    "tables": {
      "Unified_Employee_View": {
        "description": "Linked view combining all employee data sources",
        "fields": [
          {
            "name": "Primary Email",
            "type": "email",
            "description": "Primary key for cross-system matching"
          },
          {
            "name": "HRIS Record",
            "type": "linked_record",
            "linked_table": "HRIS_Records",
            "lookup_field": "work_email"
          },
          {
            "name": "Payroll Record",
            "type": "linked_record",
            "linked_table": "Payroll_Records",
            "lookup_field": "work_email"
          },
          {
            "name": "LMS Record",
            "type": "linked_record",
            "linked_table": "LMS_Records",
            "lookup_field": "email"
          },
          {
            "name": "Full Name",
            "type": "rollup",
            "rollup_table": "HRIS Record",
            "rollup_field": "full_name"
          },
          {
            "name": "Department",
            "type": "rollup",
            "rollup_table": "HRIS Record",
            "rollup_field": "department"
          },
          {
            "name": "HRIS Status",
            "type": "rollup",
            "rollup_table": "HRIS Record",
            "rollup_field": "hris_status"
          },
          {
            "name": "Payroll Status",
            "type": "rollup",
            "rollup_table": "Payroll Record",
            "rollup_field": "payroll_status"
          },
          {
            "name": "Compliance Current",
            "type": "rollup",
            "rollup_table": "LMS Record",
            "rollup_field": "compliance_status"
          },
          {
            "name": "Has Data Mismatch",
            "type": "formula",
            "formula": "OR(AND({HRIS Status}='terminated', {Payroll Status}='active'), AND({HRIS Status}='terminated', {Compliance Current}='compliant'), BLANK({Payroll Record}), BLANK({LMS Record}))"
          },
          {
            "name": "Mismatch Type",
            "type": "formula",
            "formula": "IF(AND({HRIS Status}='terminated', {Payroll Status}='active'), 'PAYROLL_LEAK', IF(AND({HRIS Status}='terminated', {Compliance Current}='compliant'), 'LMS_STALE', IF(BLANK({Payroll Record}), 'MISSING_PAYROLL', IF(BLANK({LMS Record}), 'MISSING_LMS', 'OK'))))"
          }
        ],
        "views": [
          {
            "name": "Data Mismatches",
            "type": "grid",
            "filter": "{Has Data Mismatch} = TRUE()",
            "sort": [
              {"field": "Mismatch Type", "direction": "asc"},
              {"field": "Full Name", "direction": "asc"}
            ]
          },
          {
            "name": "Terminated Still Active",
            "type": "grid",
            "filter": "AND({HRIS Status}='terminated', OR({Payroll Status}='active', {Compliance Current}='compliant'))"
          }
        ]
      }
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
              'Use ChatGPT or Claude to analyze the unified employee data, identify cross-system discrepancies, and generate actionable remediation recommendations.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Employee Data Fragmentation Analysis Prompt',
                description:
                  'Structured prompt for ChatGPT/Claude to analyze cross-system employee data and identify identity mismatches and data quality issues.',
                code: `employee_data_analysis_prompt:
  system_context: |
    You are an HR data integration specialist with expertise in identity
    resolution and data quality. You help organizations unify employee
    records across multiple HR systems including HRIS, payroll, benefits,
    and learning management systems.

  user_prompt_template: |
    ## Cross-System Employee Data Analysis

    I need help analyzing employee data from multiple HR systems to identify
    fragmentation issues and data quality problems.

    ### HRIS Data (BambooHR) - {{hris_count}} records
    \`\`\`csv
    employee_id,full_name,work_email,department,status,hire_date,term_date
    {{hris_data_sample}}
    \`\`\`

    ### Payroll Data (ADP) - {{payroll_count}} records
    \`\`\`csv
    worker_id,legal_name,work_email,dept_code,pay_status,last_pay_date
    {{payroll_data_sample}}
    \`\`\`

    ### LMS Data - {{lms_count}} records
    \`\`\`csv
    email,learner_name,courses_completed,compliance_status,last_login
    {{lms_data_sample}}
    \`\`\`

    Please analyze and provide:

    1. **Identity Matching Assessment**
       - How many records can be matched across all 3 systems?
       - Which records exist in some systems but not others?
       - Are there potential duplicates or near-matches?

    2. **Status Inconsistencies**
       - Terminated in HRIS but active in payroll (cost leak)
       - Terminated in HRIS but compliant in LMS (stale access)
       - Active in HRIS but missing from payroll (setup issue)

    3. **Data Quality Issues**
       - Name mismatches between systems (e.g., "Bob" vs "Robert")
       - Missing or inconsistent email addresses
       - Department/cost center mapping gaps

    4. **Compliance Risks**
       - Ghost employees (in payroll but not HRIS)
       - Unauthorized system access (terminated but LMS active)
       - Audit-trail gaps

    5. **Prioritized Action Plan**
       - High: Immediate termination propagation needed
       - Medium: Identity resolution required
       - Low: Data hygiene improvements

  expected_output_format: |
    Provide a structured report with:
    - Executive summary (impact statement)
    - Detailed findings by category
    - Specific records requiring action (include IDs)
    - Recommended process improvements
    - Estimated cost savings from resolution`,
              },
              {
                language: 'yaml',
                title: 'Identity Resolution Prompt Template',
                description:
                  'Prompt for AI-assisted identity matching when email addresses do not align across systems.',
                code: `identity_resolution_prompt:
  system_context: |
    You are an expert in probabilistic record matching and identity resolution.
    You help match employee records across systems when exact key matches fail,
    using name similarity, department alignment, and temporal proximity.

  user_prompt_template: |
    ## Identity Resolution Request

    I have employee records that could not be matched by email address.
    Help me determine which records likely represent the same person.

    ### Unmatched HRIS Records
    \`\`\`csv
    {{unmatched_hris}}
    \`\`\`

    ### Unmatched Payroll Records
    \`\`\`csv
    {{unmatched_payroll}}
    \`\`\`

    For each potential match, consider:

    1. **Name Similarity**
       - Exact match, nickname variations (Bob/Robert, Bill/William)
       - Maiden name vs married name
       - Name order differences (First Last vs Last, First)
       - Typos and phonetic similarity

    2. **Contextual Signals**
       - Same department/cost center
       - Similar hire dates (within 30 days)
       - Same job title or level

    3. **Confidence Scoring**
       - HIGH (90%+): Strong name match + contextual alignment
       - MEDIUM (70-89%): Partial name match OR strong contextual signals
       - LOW (50-69%): Weak signals, manual review recommended
       - NO MATCH (<50%): Likely different individuals

    Please provide:

    1. **Proposed Matches Table**
       | HRIS ID | HRIS Name | Payroll ID | Payroll Name | Confidence | Reasoning |
       |---------|-----------|------------|--------------|------------|-----------|

    2. **Records Requiring Manual Review**
       - List with specific questions for HR to verify

    3. **Confirmed Non-Matches**
       - HRIS-only records (need payroll setup)
       - Payroll-only records (potential ghost employees)

  output_format: |
    Structured markdown with match confidence percentages
    and clear action items for each category`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Configure Zapier workflows to automatically detect data fragmentation issues, trigger alerts for terminated employees with stale access, and deliver daily sync status reports.',
            toolsUsed: ['Zapier', 'Slack', 'Airtable', 'Email'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Stale Access Alert Workflow',
                description:
                  'Zapier workflow that detects terminated employees still active in downstream systems and triggers immediate deprovisioning alerts.',
                code: `{
  "zapier_workflow": {
    "name": "Terminated Employee Stale Access Alert",
    "trigger": {
      "app": "Airtable",
      "event": "New Record in View",
      "config": {
        "base_id": "{{EMPLOYEE_DATA_HUB_BASE}}",
        "table": "Unified_Employee_View",
        "view": "Terminated Still Active"
      }
    },
    "actions": [
      {
        "step": 1,
        "app": "Formatter by Zapier",
        "event": "Date/Time",
        "config": {
          "input": "{{trigger.HRIS_term_date}}",
          "transform": "compare",
          "compare_to": "{{zap_today}}",
          "output_key": "days_since_termination"
        }
      },
      {
        "step": 2,
        "app": "Paths by Zapier",
        "event": "Route by Urgency",
        "config": {
          "paths": [
            {
              "name": "Critical (>30 days)",
              "condition": "{{days_since_termination}} > 30",
              "continue_to": "step_3a"
            },
            {
              "name": "Warning (7-30 days)",
              "condition": "{{days_since_termination}} >= 7 AND {{days_since_termination}} <= 30",
              "continue_to": "step_3b"
            },
            {
              "name": "New (< 7 days)",
              "condition": "{{days_since_termination}} < 7",
              "continue_to": "step_3c"
            }
          ]
        }
      },
      {
        "step": "3a",
        "app": "Slack",
        "event": "Send Channel Message",
        "config": {
          "channel": "#hr-ops-critical",
          "message_template": ":rotating_light: *CRITICAL: Terminated Employee Still Has Active Access*\\n\\n*Employee:* {{trigger.Full_Name}}\\n*Email:* {{trigger.Primary_Email}}\\n*Terminated:* {{trigger.HRIS_term_date}} ({{days_since_termination}} days ago)\\n*Active In:* {{trigger.Mismatch_Type}}\\n\\n*Immediate Action Required:*\\n- If PAYROLL_LEAK: Process final pay and terminate payroll record\\n- If LMS_STALE: Revoke LMS access and mark courses incomplete\\n\\n<{{airtable_record_url}}|View Full Record>",
          "mention": "@hr-ops-oncall"
        }
      },
      {
        "step": "3b",
        "app": "Slack",
        "event": "Send Channel Message",
        "config": {
          "channel": "#hr-data-alerts",
          "message_template": ":warning: *Stale Employee Access Detected*\\n\\n*Employee:* {{trigger.Full_Name}}\\n*Terminated:* {{trigger.HRIS_term_date}} ({{days_since_termination}} days ago)\\n*Issue:* {{trigger.Mismatch_Type}}\\n\\nPlease resolve within 48 hours.\\n<{{airtable_record_url}}|View Record>"
        }
      },
      {
        "step": "3c",
        "app": "Airtable",
        "event": "Update Record",
        "config": {
          "base_id": "{{EMPLOYEE_DATA_HUB_BASE}}",
          "table": "Unified_Employee_View",
          "record_id": "{{trigger.record_id}}",
          "fields": {
            "Alert_Sent": true,
            "Alert_Timestamp": "{{zap_timestamp}}",
            "SLA_Due_Date": "{{zap_today_plus_7_days}}"
          }
        }
      },
      {
        "step": 4,
        "app": "Airtable",
        "event": "Create Record",
        "config": {
          "base_id": "{{EMPLOYEE_DATA_HUB_BASE}}",
          "table": "Deprovisioning_Tasks",
          "fields": {
            "Employee_Email": "{{trigger.Primary_Email}}",
            "Employee_Name": "{{trigger.Full_Name}}",
            "Task_Type": "{{trigger.Mismatch_Type}}",
            "Priority": "{{path_taken}}",
            "Created_Date": "{{zap_timestamp}}",
            "SLA_Due": "{{sla_due_date}}",
            "Status": "Open"
          }
        }
      }
    ]
  }
}`,
              },
              {
                language: 'json',
                title: 'Daily Cross-System Sync Status Report',
                description:
                  'Zapier workflow that generates and delivers a daily employee data sync health report with match rates and outstanding issues.',
                code: `{
  "zapier_workflow": {
    "name": "Daily Employee Data Sync Status Report",
    "trigger": {
      "app": "Schedule by Zapier",
      "event": "Every Day",
      "config": {
        "time": "08:00",
        "timezone": "America/New_York"
      }
    },
    "actions": [
      {
        "step": 1,
        "app": "Airtable",
        "event": "Find Records",
        "config": {
          "base_id": "{{EMPLOYEE_DATA_HUB_BASE}}",
          "table": "Unified_Employee_View",
          "formula": "TRUE()",
          "output_key": "all_records"
        }
      },
      {
        "step": 2,
        "app": "Airtable",
        "event": "Find Records",
        "config": {
          "base_id": "{{EMPLOYEE_DATA_HUB_BASE}}",
          "table": "Unified_Employee_View",
          "formula": "{Has Data Mismatch} = TRUE()",
          "output_key": "mismatch_records"
        }
      },
      {
        "step": 3,
        "app": "Airtable",
        "event": "Find Records",
        "config": {
          "base_id": "{{EMPLOYEE_DATA_HUB_BASE}}",
          "table": "Deprovisioning_Tasks",
          "formula": "AND({Status} = 'Open', {Created_Date} < DATEADD(TODAY(), -7, 'days'))",
          "output_key": "overdue_tasks"
        }
      },
      {
        "step": 4,
        "app": "Code by Zapier",
        "event": "Run JavaScript",
        "config": {
          "code": "const total = inputData.total_count;\\nconst mismatches = inputData.mismatch_count;\\nconst overdue = inputData.overdue_count;\\nconst matchRate = ((total - mismatches) / total * 100).toFixed(1);\\nconst healthStatus = matchRate >= 98 ? 'HEALTHY' : matchRate >= 95 ? 'WARNING' : 'CRITICAL';\\noutput = { matchRate, healthStatus, total, mismatches, overdue };",
          "input_data": {
            "total_count": "{{all_records.count}}",
            "mismatch_count": "{{mismatch_records.count}}",
            "overdue_count": "{{overdue_tasks.count}}"
          }
        }
      },
      {
        "step": 5,
        "app": "ChatGPT",
        "event": "Conversation",
        "config": {
          "model": "gpt-4",
          "system_message": "You are an HR data analyst. Generate a brief daily status summary.",
          "user_message": "Generate a 2-paragraph daily sync status summary:\\n\\nMetrics:\\n- Total employees tracked: {{step4.total}}\\n- Cross-system match rate: {{step4.matchRate}}%\\n- Active data mismatches: {{step4.mismatches}}\\n- Overdue deprovisioning tasks: {{step4.overdue}}\\n- Health status: {{step4.healthStatus}}\\n\\nHighlight any concerns and recommend priorities for today.",
          "max_tokens": 300
        }
      },
      {
        "step": 6,
        "app": "Slack",
        "event": "Send Channel Message",
        "config": {
          "channel": "#hr-data-sync",
          "message_template": ":chart_with_upwards_trend: *Daily Employee Data Sync Report*\\n*Date:* {{zap_today}}\\n\\n*Health Status:* {{step4.healthStatus}} {{health_emoji}}\\n\\n*Key Metrics:*\\n- Total Employees: {{step4.total}}\\n- Match Rate: {{step4.matchRate}}%\\n- Active Mismatches: {{step4.mismatches}}\\n- Overdue Tasks: {{step4.overdue}}\\n\\n*AI Summary:*\\n{{step5.response}}\\n\\n<{{airtable_dashboard_url}}|View Full Dashboard>"
        }
      },
      {
        "step": 7,
        "app": "Gmail",
        "event": "Send Email",
        "config": {
          "to": "{{hr_leadership_email}}",
          "subject": "Employee Data Sync Status - {{zap_today}} - {{step4.healthStatus}}",
          "body_template": "<h2>Daily Employee Data Integration Report</h2>\\n<p><strong>Status:</strong> {{step4.healthStatus}}</p>\\n<table border='1' cellpadding='8'>\\n<tr><td>Total Employees</td><td>{{step4.total}}</td></tr>\\n<tr><td>Cross-System Match Rate</td><td>{{step4.matchRate}}%</td></tr>\\n<tr><td>Active Mismatches</td><td>{{step4.mismatches}}</td></tr>\\n<tr><td>Overdue Deprovisioning</td><td>{{step4.overdue}}</td></tr>\\n</table>\\n<h3>Summary</h3>\\n<p>{{step5.response}}</p>\\n<p><a href='{{airtable_dashboard_url}}'>View Full Dashboard</a></p>",
          "is_html": true
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
          'Deploy a multi-agent system where specialized AI agents handle identity matching, sync orchestration, termination propagation, and compliance monitoring, coordinated by a supervisor agent that ensures seamless employee data unification across all HR systems.',
        estimatedMonthlyCost: '$600 - $1,200/month',
        architecture:
          'Supervisor agent coordinates five specialist agents: Identity Resolution Agent (fuzzy matching and master ID management), Sync Orchestration Agent (cross-system data flow), Termination Propagation Agent (downstream access revocation), Compliance Monitor Agent (audit trail and risk detection), and Reporting Agent (stakeholder communications). LangGraph orchestrates continuous sync with Redis-backed state persistence.',
        agents: [
          {
            name: 'Identity Resolution Agent',
            role: 'Master Identity Specialist',
            goal: 'Match employee records across HRIS, payroll, benefits, and LMS using probabilistic matching algorithms. Maintain the master identity crosswalk and resolve ambiguous matches.',
            tools: ['recordlinkage', 'fuzzywuzzy', 'pandas', 'dedupe', 'sqlalchemy'],
          },
          {
            name: 'Sync Orchestration Agent',
            role: 'Data Flow Coordinator',
            goal: 'Coordinate data extraction from all HR systems, detect changes since last sync, and route updates to the unified employee data layer with conflict resolution.',
            tools: ['BambooHR API', 'ADP API', 'LMS API', 'Benefits API', 'Redis', 'pandas'],
          },
          {
            name: 'Termination Propagation Agent',
            role: 'Access Revocation Specialist',
            goal: 'Detect terminated employees in HRIS and ensure their access is revoked across all downstream systems within SLA. Queue deprovisioning tasks and track completion.',
            tools: ['slack_sdk', 'jira-python', 'pandas', 'sqlalchemy'],
          },
          {
            name: 'Compliance Monitor Agent',
            role: 'Audit & Risk Specialist',
            goal: 'Monitor for compliance risks including ghost employees, unauthorized access, and audit-trail gaps. Generate compliance reports and flag violations.',
            tools: ['pandas', 'sqlalchemy', 'jinja2', 'sendgrid'],
          },
          {
            name: 'Reporting Agent',
            role: 'Stakeholder Communications Specialist',
            goal: 'Generate sync status reports, data quality scorecards, and executive summaries tailored to different audiences from HR ops to leadership.',
            tools: ['jinja2', 'markdown', 'slack_sdk', 'google-api-python-client'],
          },
          {
            name: 'Supervisor Agent',
            role: 'Workflow Orchestrator',
            goal: 'Coordinate the specialist agents, manage sync dependencies, handle API failures gracefully, and ensure the employee data unification pipeline runs continuously.',
            tools: ['langgraph', 'redis', 'langchain', 'langsmith'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with event sourcing for audit trail and 90-day retention',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent architecture with CrewAI, specifying each agent role, goals, and tool access for the employee data unification pipeline.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Employee Data Unification Agent Definitions',
                description:
                  'CrewAI agent definitions for identity resolution, sync orchestration, termination propagation, and compliance monitoring with their specialized tools.',
                code: `"""Employee Data Unification Multi-Agent System - Agent Definitions."""
from typing import List, Optional
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class DataUnificationConfig(BaseModel):
    """Configuration for employee data unification pipeline."""
    identity_match_threshold: float = Field(default=0.85, description="Min confidence for auto-match")
    termination_sla_days: int = Field(default=7, description="Days to propagate termination")
    sync_frequency_hours: int = Field(default=6, description="Hours between full syncs")
    ghost_employee_threshold_days: int = Field(default=30, description="Days in payroll without HRIS")


class IdentityMatch(BaseModel):
    """Schema for identity matching results."""
    master_id: str
    hris_id: Optional[str]
    payroll_id: Optional[str]
    lms_email: Optional[str]
    benefits_ssn_hash: Optional[str]
    confidence: float
    match_type: str  # 'exact', 'fuzzy', 'manual_review'


# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.1,
    max_tokens=4096,
)


# --- Identity Resolution Agent ---
identity_resolution_agent = Agent(
    role="Master Identity Resolution Specialist",
    goal="""Match employee records across HRIS, payroll, benefits, and LMS systems
    using probabilistic matching. Maintain the master identity crosswalk, resolve
    ambiguous matches, and prevent duplicate master IDs.""",
    backstory="""You are an expert in data integration and entity resolution with
    deep experience in HR systems. You understand the challenges of matching records
    when employees have nicknames, name changes, inconsistent email domains, and
    different ID formats across systems. You balance automation with human review
    to ensure data quality while maintaining operational efficiency.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[],  # Tools added separately
    max_iter=5,
    memory=True,
)


# --- Sync Orchestration Agent ---
sync_orchestration_agent = Agent(
    role="Cross-System Data Flow Coordinator",
    goal="""Coordinate data extraction from all HR systems, detect changes since
    last sync, and route updates to the unified employee data layer. Handle API
    failures, rate limits, and data conflicts gracefully.""",
    backstory="""You are a data engineering specialist who has built dozens of
    HR system integrations. You understand the quirks of each vendor's API,
    common failure modes, and best practices for incremental syncing. You
    prioritize data freshness while respecting API limits and ensuring no
    records are lost during sync failures.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[],
    max_iter=5,
    memory=True,
)


# --- Termination Propagation Agent ---
termination_propagation_agent = Agent(
    role="Employee Offboarding Access Revocation Specialist",
    goal="""Detect terminated employees in HRIS and ensure their access is revoked
    across all downstream systems within the SLA. Create deprovisioning tasks,
    track completion, and escalate overdue items.""",
    backstory="""You are a security-focused HR operations specialist who
    understands the compliance and financial risks of delayed offboarding.
    You have seen companies pay benefits to terminated employees for months
    and face audit findings for stale LMS access. You are relentless about
    closing access gaps quickly and documenting everything.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[],
    max_iter=5,
    memory=True,
)


# --- Compliance Monitor Agent ---
compliance_monitor_agent = Agent(
    role="HR Data Compliance & Audit Specialist",
    goal="""Monitor for compliance risks including ghost employees, unauthorized
    system access, and audit-trail gaps. Generate compliance reports, flag
    violations, and maintain evidence for auditors.""",
    backstory="""You are a compliance expert who has guided companies through
    SOX audits, SOC 2 certifications, and HR regulatory reviews. You know what
    auditors look for and proactively identify issues before they become findings.
    You document everything meticulously and can explain the data lineage of
    any employee record.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[],
    max_iter=3,
    memory=True,
)


# --- Reporting Agent ---
reporting_agent = Agent(
    role="Employee Data Analytics Communications Specialist",
    goal="""Generate sync status reports, data quality scorecards, and executive
    summaries tailored to different audiences. Translate technical findings into
    business impact and actionable recommendations.""",
    backstory="""You are a seasoned HR analytics communicator who bridges the
    gap between data engineers and business stakeholders. You know that executives
    want the 'so what' in three sentences, while HR ops needs step-by-step
    resolution guides. You create compelling visualizations and write clearly.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[],
    max_iter=3,
    memory=True,
)


# --- Supervisor Agent ---
supervisor_agent = Agent(
    role="Employee Data Unification Pipeline Orchestrator",
    goal="""Coordinate the specialist agents, manage sync dependencies, handle
    API failures gracefully, and ensure the employee data unification pipeline
    runs continuously. Escalate critical issues to human operators.""",
    backstory="""You are the chief architect of the employee data integration
    platform. You understand the dependencies between agents, know when to
    retry vs skip vs escalate, and maintain the overall health of the pipeline.
    You communicate proactively about issues and never let silent failures
    accumulate.""",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[],
    max_iter=10,
    memory=True,
)


def create_data_unification_crew(config: DataUnificationConfig) -> Crew:
    """Create the employee data unification crew with configured agents."""

    # Define tasks for each agent
    identity_task = Task(
        description="""Pull new/changed records from all HR systems.
        Match records using email, name similarity, and department alignment.
        Update master identity crosswalk. Flag records needing manual review.
        Confidence threshold: >{threshold}.""".format(
            threshold=config.identity_match_threshold
        ),
        expected_output="Updated crosswalk with match confidence scores",
        agent=identity_resolution_agent,
    )

    sync_task = Task(
        description="""Extract incremental changes from BambooHR, ADP, benefits,
        and LMS since last sync. Detect conflicts and apply resolution rules.
        Update unified employee profiles. Log all changes for audit.""",
        expected_output="Sync summary with records updated, conflicts resolved",
        agent=sync_orchestration_agent,
    )

    termination_task = Task(
        description="""Scan for employees terminated in HRIS within last 24 hours.
        Check downstream system status. Create deprovisioning tasks for any
        with active access. SLA: {sla} days for full propagation.""".format(
            sla=config.termination_sla_days
        ),
        expected_output="Deprovisioning task list with priorities and SLAs",
        agent=termination_propagation_agent,
        context=[sync_task],  # Depends on fresh sync data
    )

    compliance_task = Task(
        description="""Audit the unified data for compliance risks:
        - Ghost employees (payroll > {ghost_days} days without HRIS)
        - Stale access (terminated > {sla} days with downstream active)
        - Audit trail gaps (missing sync logs)
        Generate compliance scorecard.""".format(
            ghost_days=config.ghost_employee_threshold_days,
            sla=config.termination_sla_days,
        ),
        expected_output="Compliance scorecard with risk items and remediation steps",
        agent=compliance_monitor_agent,
        context=[sync_task, termination_task],
    )

    reporting_task = Task(
        description="""Compile findings into three deliverables:
        1. Executive dashboard update (3 KPIs with trends)
        2. HR Ops daily action list (prioritized tasks)
        3. Weekly compliance summary (for audit file)""",
        expected_output="Three formatted reports for different audiences",
        agent=reporting_agent,
        context=[identity_task, sync_task, termination_task, compliance_task],
    )

    return Crew(
        agents=[
            identity_resolution_agent,
            sync_orchestration_agent,
            termination_propagation_agent,
            compliance_monitor_agent,
            reporting_agent,
        ],
        tasks=[identity_task, sync_task, termination_task, compliance_task, reporting_task],
        process=Process.sequential,
        manager_agent=supervisor_agent,
        verbose=True,
    )`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the Identity Resolution and Sync Orchestration agents with tools for pulling data from multiple HR systems and performing probabilistic record matching.',
            toolsUsed: ['CrewAI', 'LangChain', 'recordlinkage', 'pandas', 'sqlalchemy'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Identity Resolution Agent Tools',
                description:
                  'Custom CrewAI tools for probabilistic record matching across HR systems using the recordlinkage library.',
                code: `"""Identity Resolution Agent Tools for Employee Data Unification."""
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import recordlinkage
from crewai.tools import BaseTool
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field
from recordlinkage.index import Full, Block
from sqlalchemy import create_engine, text


class ProbabilisticMatchInput(BaseModel):
    """Input schema for probabilistic record matching."""
    source_a_records: List[Dict] = Field(description="Records from first system")
    source_b_records: List[Dict] = Field(description="Records from second system")
    blocking_fields: List[str] = Field(
        default=["department"],
        description="Fields to use for blocking (reduce comparison space)"
    )
    match_threshold: float = Field(default=0.85, description="Min score for auto-match")


class ProbabilisticMatchTool(BaseTool):
    """Tool for probabilistic record matching using recordlinkage."""

    name: str = "probabilistic_match"
    description: str = """Matches employee records between two systems using
    probabilistic record linkage. Handles name variations, typos, and missing
    data. Returns matches with confidence scores."""
    args_schema: Type[BaseModel] = ProbabilisticMatchInput

    def _run(
        self,
        source_a_records: List[Dict],
        source_b_records: List[Dict],
        blocking_fields: List[str] = None,
        match_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """Execute probabilistic record matching."""

        blocking_fields = blocking_fields or ["department"]

        # Convert to DataFrames
        df_a = pd.DataFrame(source_a_records)
        df_b = pd.DataFrame(source_b_records)

        if df_a.empty or df_b.empty:
            return {"error": "One or both record sets are empty"}

        # Standardize column names
        df_a.columns = df_a.columns.str.lower().str.replace(" ", "_")
        df_b.columns = df_b.columns.str.lower().str.replace(" ", "_")

        # Find name columns
        name_col_a = next(
            (c for c in df_a.columns if "name" in c.lower()),
            df_a.columns[0]
        )
        name_col_b = next(
            (c for c in df_b.columns if "name" in c.lower()),
            df_b.columns[0]
        )

        # Find email columns
        email_col_a = next(
            (c for c in df_a.columns if "email" in c.lower()),
            None
        )
        email_col_b = next(
            (c for c in df_b.columns if "email" in c.lower()),
            None
        )

        # Create indexer with blocking
        indexer = recordlinkage.Index()

        # Use blocking on available fields to reduce comparisons
        valid_blocking = [
            f for f in blocking_fields
            if f in df_a.columns and f in df_b.columns
        ]
        if valid_blocking:
            indexer.block(valid_blocking[0])
        else:
            indexer.full()  # Compare all pairs if no blocking field

        candidate_pairs = indexer.index(df_a, df_b)

        # Create comparison features
        compare = recordlinkage.Compare()

        # Name comparison with multiple methods
        compare.string(
            name_col_a, name_col_b,
            method="jarowinkler",
            threshold=0.8,
            label="name_jaro"
        )
        compare.string(
            name_col_a, name_col_b,
            method="levenshtein",
            threshold=0.8,
            label="name_lev"
        )

        # Email comparison if available
        if email_col_a and email_col_b:
            compare.exact(email_col_a, email_col_b, label="email_exact")
            compare.string(
                email_col_a, email_col_b,
                method="jarowinkler",
                label="email_jaro"
            )

        # Department comparison if available
        if "department" in df_a.columns and "department" in df_b.columns:
            compare.exact("department", "department", label="dept_exact")

        # Compute comparison features
        features = compare.compute(candidate_pairs, df_a, df_b)

        # Calculate overall match score (weighted average)
        weights = {
            "name_jaro": 0.3,
            "name_lev": 0.2,
            "email_exact": 0.25,
            "email_jaro": 0.15,
            "dept_exact": 0.1,
        }

        # Apply weights to available features
        available_weights = {k: v for k, v in weights.items() if k in features.columns}
        total_weight = sum(available_weights.values())
        normalized_weights = {k: v/total_weight for k, v in available_weights.items()}

        features["match_score"] = sum(
            features[col] * weight
            for col, weight in normalized_weights.items()
            if col in features.columns
        )

        # Classify matches
        auto_matches = features[features["match_score"] >= match_threshold]
        review_needed = features[
            (features["match_score"] >= 0.6) &
            (features["match_score"] < match_threshold)
        ]
        non_matches = features[features["match_score"] < 0.6]

        # Build results
        matched_pairs = []
        for (idx_a, idx_b), row in auto_matches.iterrows():
            matched_pairs.append({
                "source_a_index": int(idx_a),
                "source_b_index": int(idx_b),
                "source_a_record": source_a_records[idx_a],
                "source_b_record": source_b_records[idx_b],
                "match_score": round(float(row["match_score"]), 3),
                "match_type": "auto",
            })

        review_pairs = []
        for (idx_a, idx_b), row in review_needed.iterrows():
            review_pairs.append({
                "source_a_index": int(idx_a),
                "source_b_index": int(idx_b),
                "source_a_record": source_a_records[idx_a],
                "source_b_record": source_b_records[idx_b],
                "match_score": round(float(row["match_score"]), 3),
                "match_type": "manual_review",
            })

        # Find unmatched records
        matched_a_indices = set(auto_matches.index.get_level_values(0))
        matched_b_indices = set(auto_matches.index.get_level_values(1))

        unmatched_a = [
            {"index": i, "record": source_a_records[i]}
            for i in range(len(df_a))
            if i not in matched_a_indices
        ]
        unmatched_b = [
            {"index": i, "record": source_b_records[i]}
            for i in range(len(df_b))
            if i not in matched_b_indices
        ]

        return {
            "success": True,
            "summary": {
                "source_a_count": len(df_a),
                "source_b_count": len(df_b),
                "auto_matched": len(matched_pairs),
                "needs_review": len(review_pairs),
                "unmatched_a": len(unmatched_a),
                "unmatched_b": len(unmatched_b),
                "match_rate_pct": round(
                    len(matched_pairs) / max(len(df_a), len(df_b), 1) * 100, 2
                ),
            },
            "auto_matches": matched_pairs,
            "manual_review": review_pairs,
            "unmatched_source_a": unmatched_a[:50],  # Limit for response size
            "unmatched_source_b": unmatched_b[:50],
            "methodology": {
                "blocking_fields": valid_blocking,
                "match_threshold": match_threshold,
                "features_used": list(features.columns),
                "weights": normalized_weights,
            },
        }


class CrosswalkUpdateInput(BaseModel):
    """Input schema for updating the master identity crosswalk."""
    matches: List[Dict] = Field(description="Matched record pairs with scores")
    source_a_system: str = Field(description="Name of first system (e.g., 'hris')")
    source_b_system: str = Field(description="Name of second system (e.g., 'payroll')")


class CrosswalkUpdateTool(BaseTool):
    """Tool for updating the master identity crosswalk."""

    name: str = "update_identity_crosswalk"
    description: str = """Updates the master identity crosswalk with new matches.
    Creates new master IDs for unlinked records, links existing IDs for matched
    records, and logs all changes for audit."""
    args_schema: Type[BaseModel] = CrosswalkUpdateInput

    def __init__(self, database_url: Optional[str] = None):
        super().__init__()
        self._database_url = database_url or os.environ.get("DATABASE_URL")

    def _run(
        self,
        matches: List[Dict],
        source_a_system: str,
        source_b_system: str,
    ) -> Dict[str, Any]:
        """Update the identity crosswalk with matched records."""

        if not self._database_url:
            return {"error": "Database URL not configured"}

        engine = create_engine(self._database_url)
        created_count = 0
        linked_count = 0
        errors = []

        id_field_map = {
            "hris": "bamboohr_id",
            "payroll": "adp_payroll_id",
            "lms": "lms_email",
            "benefits": "benefits_ssn_hash",
        }

        source_a_field = id_field_map.get(source_a_system.lower())
        source_b_field = id_field_map.get(source_b_system.lower())

        if not source_a_field or not source_b_field:
            return {"error": f"Unknown system: {source_a_system} or {source_b_system}"}

        with engine.begin() as conn:
            for match in matches:
                try:
                    source_a_id = match.get("source_a_record", {}).get("id") or \
                                  match.get("source_a_record", {}).get("employee_id")
                    source_b_id = match.get("source_b_record", {}).get("id") or \
                                  match.get("source_b_record", {}).get("worker_id") or \
                                  match.get("source_b_record", {}).get("email")

                    # Check if either ID already exists in crosswalk
                    existing = conn.execute(text(f"""
                        SELECT master_id, {source_a_field}, {source_b_field}
                        FROM hr_unified.employee_identity
                        WHERE {source_a_field} = :a_id OR {source_b_field} = :b_id
                    """), {"a_id": source_a_id, "b_id": source_b_id}).fetchone()

                    if existing:
                        # Update existing record to link both IDs
                        conn.execute(text(f"""
                            UPDATE hr_unified.employee_identity
                            SET {source_a_field} = COALESCE({source_a_field}, :a_id),
                                {source_b_field} = COALESCE({source_b_field}, :b_id),
                                updated_at = NOW()
                            WHERE master_id = :master_id
                        """), {
                            "a_id": source_a_id,
                            "b_id": source_b_id,
                            "master_id": existing[0],
                        })
                        linked_count += 1
                    else:
                        # Create new crosswalk entry
                        conn.execute(text(f"""
                            INSERT INTO hr_unified.employee_identity
                                ({source_a_field}, {source_b_field})
                            VALUES (:a_id, :b_id)
                        """), {"a_id": source_a_id, "b_id": source_b_id})
                        created_count += 1

                except Exception as e:
                    errors.append({
                        "match": match,
                        "error": str(e),
                    })

        return {
            "success": len(errors) == 0,
            "summary": {
                "total_matches_processed": len(matches),
                "new_master_ids_created": created_count,
                "existing_records_linked": linked_count,
                "errors": len(errors),
            },
            "errors": errors[:10],  # Limit error list
        }


# Tool registrations
identity_resolution_tools = [
    ProbabilisticMatchTool(),
    CrosswalkUpdateTool(),
]`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the Termination Propagation and Compliance Monitor agents with tools for detecting stale access and generating compliance reports.',
            toolsUsed: ['CrewAI', 'pandas', 'sqlalchemy', 'slack_sdk'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Termination Propagation & Compliance Tools',
                description:
                  'Tools for detecting terminated employees with stale downstream access and generating compliance audit reports.',
                code: `"""Termination Propagation & Compliance Monitor Agent Tools."""
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from sqlalchemy import create_engine, text


class StaleAccessDetectionInput(BaseModel):
    """Input schema for stale access detection."""
    sla_days: int = Field(default=7, description="Days allowed for termination propagation")
    include_resolved: bool = Field(default=False, description="Include already-resolved issues")


class StaleAccessDetectionTool(BaseTool):
    """Tool for detecting terminated employees with active downstream access."""

    name: str = "detect_stale_access"
    description: str = """Scans for terminated employees who still have active
    access in payroll, benefits, or LMS systems beyond the allowed SLA."""
    args_schema: Type[BaseModel] = StaleAccessDetectionInput

    def __init__(self, database_url: Optional[str] = None):
        super().__init__()
        self._database_url = database_url or os.environ.get("DATABASE_URL")

    def _run(
        self,
        sla_days: int = 7,
        include_resolved: bool = False,
    ) -> Dict[str, Any]:
        """Detect stale access for terminated employees."""

        if not self._database_url:
            return {"error": "Database URL not configured"}

        engine = create_engine(self._database_url)

        query = text("""
            SELECT
                ep.master_id,
                ep.full_name,
                ep.work_email,
                ep.department,
                ep.termination_date,
                CURRENT_DATE - ep.termination_date AS days_since_termination,
                ep.payroll_status,
                ep.benefits_status,
                ep.compliance_current AS lms_active,
                CASE
                    WHEN ep.payroll_status = 'active' THEN 'PAYROLL_LEAK'
                    WHEN ep.benefits_status = 'active' THEN 'BENEFITS_LEAK'
                    WHEN ep.compliance_current = TRUE THEN 'LMS_STALE'
                    ELSE 'UNKNOWN'
                END AS violation_type,
                CASE
                    WHEN CURRENT_DATE - ep.termination_date > 30 THEN 'critical'
                    WHEN CURRENT_DATE - ep.termination_date > :sla_days THEN 'overdue'
                    ELSE 'within_sla'
                END AS urgency
            FROM hr_unified.v_employee_profile ep
            WHERE ep.hris_status = 'terminated'
              AND ep.termination_date IS NOT NULL
              AND (
                  ep.payroll_status = 'active'
                  OR ep.benefits_status = 'active'
                  OR ep.compliance_current = TRUE
              )
            ORDER BY ep.termination_date ASC
        """)

        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sla_days": sla_days})

        if df.empty:
            return {
                "success": True,
                "message": "No stale access detected",
                "summary": {
                    "total_violations": 0,
                    "critical": 0,
                    "overdue": 0,
                    "within_sla": 0,
                },
                "violations": [],
            }

        # Summarize by urgency
        urgency_counts = df["urgency"].value_counts().to_dict()

        # Summarize by violation type
        type_counts = df["violation_type"].value_counts().to_dict()

        return {
            "success": True,
            "summary": {
                "total_violations": len(df),
                "critical": urgency_counts.get("critical", 0),
                "overdue": urgency_counts.get("overdue", 0),
                "within_sla": urgency_counts.get("within_sla", 0),
                "by_type": type_counts,
            },
            "violations": df.to_dict(orient="records"),
            "sla_used_days": sla_days,
            "detection_timestamp": datetime.now().isoformat(),
        }


class DeprovisioningTaskInput(BaseModel):
    """Input schema for creating deprovisioning tasks."""
    violations: List[Dict] = Field(description="Stale access violations to create tasks for")
    notify_slack: bool = Field(default=True, description="Send Slack notifications")


class DeprovisioningTaskTool(BaseTool):
    """Tool for creating deprovisioning tasks from stale access violations."""

    name: str = "create_deprovisioning_tasks"
    description: str = """Creates deprovisioning tasks for each stale access
    violation, assigns priorities based on urgency, and optionally notifies
    via Slack."""
    args_schema: Type[BaseModel] = DeprovisioningTaskInput

    def __init__(
        self,
        database_url: Optional[str] = None,
        slack_token: Optional[str] = None,
    ):
        super().__init__()
        self._database_url = database_url or os.environ.get("DATABASE_URL")
        self._slack_token = slack_token or os.environ.get("SLACK_BOT_TOKEN")

    def _run(
        self,
        violations: List[Dict],
        notify_slack: bool = True,
    ) -> Dict[str, Any]:
        """Create deprovisioning tasks and notify stakeholders."""

        if not self._database_url:
            return {"error": "Database URL not configured"}

        engine = create_engine(self._database_url)
        created_tasks = []
        slack_notifications = []

        sla_map = {
            "critical": 1,  # 1 day SLA for critical
            "overdue": 3,   # 3 days for overdue
            "within_sla": 7,  # 7 days for within SLA
        }

        with engine.begin() as conn:
            for violation in violations:
                urgency = violation.get("urgency", "within_sla")
                sla_days = sla_map.get(urgency, 7)
                due_date = date.today() + timedelta(days=sla_days)

                # Insert task
                result = conn.execute(text("""
                    INSERT INTO hr_unified.deprovision_queue
                        (master_id, employee_name, employee_email, system,
                         action, priority, detected_date, sla_due_date)
                    VALUES
                        (:master_id, :name, :email, :system,
                         :action, :priority, CURRENT_DATE, :due_date)
                    ON CONFLICT (master_id, system) DO UPDATE SET
                        priority = EXCLUDED.priority,
                        sla_due_date = LEAST(deprovision_queue.sla_due_date, EXCLUDED.sla_due_date)
                    RETURNING id
                """), {
                    "master_id": violation["master_id"],
                    "name": violation["full_name"],
                    "email": violation.get("work_email"),
                    "system": violation["violation_type"].lower().split("_")[0],
                    "action": f"Terminate {violation['violation_type']}",
                    "priority": urgency,
                    "due_date": due_date,
                })

                task_id = result.fetchone()[0]
                created_tasks.append({
                    "task_id": task_id,
                    "employee": violation["full_name"],
                    "violation": violation["violation_type"],
                    "urgency": urgency,
                    "sla_due": str(due_date),
                })

        # Send Slack notifications for critical items
        if notify_slack and self._slack_token:
            critical_violations = [v for v in violations if v.get("urgency") == "critical"]
            if critical_violations:
                try:
                    client = WebClient(token=self._slack_token)
                    message = self._format_slack_alert(critical_violations)
                    response = client.chat_postMessage(
                        channel="#hr-ops-critical",
                        text=message,
                        mrkdwn=True,
                    )
                    slack_notifications.append({
                        "channel": "#hr-ops-critical",
                        "message_ts": response["ts"],
                        "violations_count": len(critical_violations),
                    })
                except SlackApiError as e:
                    slack_notifications.append({
                        "error": str(e),
                    })

        return {
            "success": True,
            "summary": {
                "tasks_created": len(created_tasks),
                "critical_alerts_sent": len([n for n in slack_notifications if "error" not in n]),
            },
            "tasks": created_tasks,
            "slack_notifications": slack_notifications,
        }

    def _format_slack_alert(self, violations: List[Dict]) -> str:
        """Format critical violations for Slack notification."""
        header = ":rotating_light: *CRITICAL: Stale Access Requires Immediate Action*\\n\\n"
        rows = []
        for v in violations[:10]:  # Limit to 10 in message
            rows.append(
                f"- *{v['full_name']}* | Terminated {v['days_since_termination']} days ago | "
                f"{v['violation_type']}"
            )
        footer = f"\\n\\n_{len(violations)} total critical violations. <{{dashboard_url}}|View All>_"
        return header + "\\n".join(rows) + footer


class ComplianceReportInput(BaseModel):
    """Input schema for compliance report generation."""
    report_period_days: int = Field(default=30, description="Days to include in report")
    include_resolved: bool = Field(default=True, description="Include resolved issues")


class ComplianceReportTool(BaseTool):
    """Tool for generating compliance audit reports."""

    name: str = "generate_compliance_report"
    description: str = """Generates a comprehensive compliance report covering
    ghost employees, stale access, audit trail completeness, and SLA performance."""
    args_schema: Type[BaseModel] = ComplianceReportInput

    def __init__(self, database_url: Optional[str] = None):
        super().__init__()
        self._database_url = database_url or os.environ.get("DATABASE_URL")

    def _run(
        self,
        report_period_days: int = 30,
        include_resolved: bool = True,
    ) -> Dict[str, Any]:
        """Generate compliance audit report."""

        if not self._database_url:
            return {"error": "Database URL not configured"}

        engine = create_engine(self._database_url)
        report_data = {}

        with engine.connect() as conn:
            # 1. Ghost employee check (in payroll but not HRIS)
            ghost_query = text("""
                SELECT COUNT(*) as count
                FROM hr_unified.v_employee_profile
                WHERE hris_status IS NULL
                  AND payroll_status = 'active'
            """)
            ghost_count = conn.execute(ghost_query).scalar()

            # 2. Stale access summary
            stale_query = text("""
                SELECT
                    COUNT(*) as total_violations,
                    COUNT(*) FILTER (WHERE urgency = 'critical') as critical,
                    COUNT(*) FILTER (WHERE urgency = 'overdue') as overdue,
                    AVG(days_since_termination) as avg_days_stale
                FROM (
                    SELECT
                        CURRENT_DATE - termination_date as days_since_termination,
                        CASE
                            WHEN CURRENT_DATE - termination_date > 30 THEN 'critical'
                            WHEN CURRENT_DATE - termination_date > 7 THEN 'overdue'
                            ELSE 'within_sla'
                        END as urgency
                    FROM hr_unified.v_employee_profile
                    WHERE hris_status = 'terminated'
                      AND (payroll_status = 'active' OR benefits_status = 'active')
                ) violations
            """)
            stale_result = conn.execute(stale_query).fetchone()

            # 3. SLA compliance rate
            sla_query = text("""
                SELECT
                    COUNT(*) as total_tasks,
                    COUNT(*) FILTER (
                        WHERE completed_date IS NOT NULL
                          AND completed_date <= sla_due_date
                    ) as met_sla,
                    COUNT(*) FILTER (
                        WHERE completed_date IS NOT NULL
                          AND completed_date > sla_due_date
                    ) as breached_sla
                FROM hr_unified.deprovision_queue
                WHERE detected_date >= CURRENT_DATE - :days
            """)
            sla_result = conn.execute(sla_query, {"days": report_period_days}).fetchone()

            # 4. Sync health check
            sync_query = text("""
                SELECT
                    COUNT(*) as total_records,
                    COUNT(*) FILTER (WHERE updated_at >= CURRENT_DATE - INTERVAL '1 day') as synced_today,
                    COUNT(*) FILTER (WHERE updated_at < CURRENT_DATE - INTERVAL '7 days') as stale_sync
                FROM hr_unified.employee_identity
            """)
            sync_result = conn.execute(sync_query).fetchone()

        # Calculate SLA compliance rate
        total_completed = (sla_result[1] or 0) + (sla_result[2] or 0)
        sla_compliance_pct = (
            round((sla_result[1] or 0) / total_completed * 100, 1)
            if total_completed > 0 else 100.0
        )

        report_data = {
            "report_period": f"Last {report_period_days} days",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": {
                "overall_status": "HEALTHY" if ghost_count == 0 and sla_compliance_pct >= 95 else "NEEDS_ATTENTION",
                "ghost_employees": ghost_count,
                "stale_access_violations": stale_result[0] or 0,
                "sla_compliance_pct": sla_compliance_pct,
            },
            "detailed_findings": {
                "ghost_employees": {
                    "count": ghost_count,
                    "risk": "HIGH" if ghost_count > 0 else "LOW",
                    "description": "Employees in payroll system without corresponding HRIS record",
                },
                "stale_access": {
                    "total": stale_result[0] or 0,
                    "critical": stale_result[1] or 0,
                    "overdue": stale_result[2] or 0,
                    "avg_days_outstanding": round(stale_result[3] or 0, 1),
                },
                "deprovisioning_sla": {
                    "total_tasks": sla_result[0] or 0,
                    "met_sla": sla_result[1] or 0,
                    "breached_sla": sla_result[2] or 0,
                    "compliance_pct": sla_compliance_pct,
                },
                "sync_health": {
                    "total_identities": sync_result[0] or 0,
                    "synced_last_24h": sync_result[1] or 0,
                    "stale_over_7_days": sync_result[2] or 0,
                },
            },
            "recommendations": self._generate_recommendations(
                ghost_count, stale_result, sla_compliance_pct
            ),
        }

        return {
            "success": True,
            "report": report_data,
        }

    def _generate_recommendations(
        self,
        ghost_count: int,
        stale_result: tuple,
        sla_compliance_pct: float,
    ) -> List[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = []

        if ghost_count > 0:
            recommendations.append(
                f"URGENT: Investigate {ghost_count} potential ghost employees in payroll. "
                "These represent immediate compliance and financial risk."
            )

        if (stale_result[1] or 0) > 0:  # Critical stale access
            recommendations.append(
                f"CRITICAL: {stale_result[1]} terminated employees have had active "
                "downstream access for >30 days. Immediate deprovisioning required."
            )

        if sla_compliance_pct < 95:
            recommendations.append(
                f"Process improvement needed: SLA compliance at {sla_compliance_pct}%. "
                "Review deprovisioning workflow for bottlenecks."
            )

        if not recommendations:
            recommendations.append(
                "All compliance metrics within acceptable ranges. Continue monitoring."
            )

        return recommendations


# Tool registrations
termination_compliance_tools = [
    StaleAccessDetectionTool(),
    DeprovisioningTaskTool(),
    ComplianceReportTool(),
]`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement the LangGraph state machine that coordinates the employee data unification pipeline with continuous sync and event-driven termination propagation.',
            toolsUsed: ['LangGraph', 'Redis', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Employee Data Unification Workflow',
                description:
                  'LangGraph state machine orchestrating continuous employee data sync, identity resolution, and termination propagation with Redis persistence.',
                code: `"""LangGraph Orchestration for Employee Data Unification Pipeline."""
import json
import logging
import operator
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

import redis
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmployeeDataState(TypedDict):
    """State schema for employee data unification workflow."""

    # Workflow metadata
    run_id: str
    run_type: Literal["scheduled_sync", "termination_event", "manual"]
    run_timestamp: str
    status: Literal["pending", "running", "completed", "failed"]

    # Agent outputs
    identity_resolution_result: Optional[Dict[str, Any]]
    sync_result: Optional[Dict[str, Any]]
    termination_result: Optional[Dict[str, Any]]
    compliance_result: Optional[Dict[str, Any]]
    reports_generated: Optional[Dict[str, str]]

    # Error tracking
    errors: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]

    # Message history
    messages: Annotated[List[BaseMessage], operator.add]

    # Control flow
    current_agent: str
    pending_agents: List[str]
    requires_escalation: bool
    escalation_reason: Optional[str]

    # Event triggers (for event-driven runs)
    trigger_event: Optional[Dict[str, Any]]


class EmployeeDataWorkflow:
    """LangGraph workflow for employee data unification."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the workflow with Redis checkpointing."""
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)
        self.redis_client = redis.from_url(redis_url)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph state machine."""

        workflow = StateGraph(EmployeeDataState)

        # Add nodes
        workflow.add_node("initialize", self._initialize_run)
        workflow.add_node("identity_resolution", self._run_identity_resolution)
        workflow.add_node("sync_orchestration", self._run_sync)
        workflow.add_node("termination_check", self._check_terminations)
        workflow.add_node("compliance_monitor", self._run_compliance)
        workflow.add_node("generate_reports", self._generate_reports)
        workflow.add_node("supervisor_review", self._supervisor_review)
        workflow.add_node("escalate", self._escalate)
        workflow.add_node("finalize", self._finalize)

        # Entry point
        workflow.set_entry_point("initialize")

        # Conditional routing from initialize based on run type
        workflow.add_conditional_edges(
            "initialize",
            self._route_from_initialize,
            {
                "full_sync": "identity_resolution",
                "termination_only": "termination_check",
                "compliance_only": "compliance_monitor",
            }
        )

        # Sequential flow for full sync
        workflow.add_edge("identity_resolution", "sync_orchestration")
        workflow.add_edge("sync_orchestration", "termination_check")
        workflow.add_edge("termination_check", "supervisor_review")

        # Supervisor routing
        workflow.add_conditional_edges(
            "supervisor_review",
            self._route_from_supervisor,
            {
                "compliance": "compliance_monitor",
                "reports": "generate_reports",
                "escalate": "escalate",
                "finalize": "finalize",
            }
        )

        workflow.add_edge("compliance_monitor", "supervisor_review")
        workflow.add_edge("generate_reports", "finalize")
        workflow.add_edge("escalate", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def _initialize_run(self, state: EmployeeDataState) -> Dict[str, Any]:
        """Initialize a new unification run."""
        run_id = f"emp-sync-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_type = state.get("run_type", "scheduled_sync")

        logger.info(f"Initializing {run_type} run: {run_id}")

        return {
            "run_id": run_id,
            "run_timestamp": datetime.now().isoformat(),
            "status": "running",
            "current_agent": "initialize",
            "pending_agents": ["identity_resolution", "sync_orchestration",
                             "termination_check", "compliance_monitor", "generate_reports"],
            "errors": [],
            "warnings": [],
            "requires_escalation": False,
            "messages": [
                SystemMessage(content=f"Starting {run_type} workflow"),
            ],
        }

    def _route_from_initialize(self, state: EmployeeDataState) -> str:
        """Route based on run type."""
        run_type = state.get("run_type", "scheduled_sync")

        if run_type == "termination_event":
            return "termination_only"
        elif run_type == "compliance_check":
            return "compliance_only"
        else:
            return "full_sync"

    def _run_identity_resolution(self, state: EmployeeDataState) -> Dict[str, Any]:
        """Execute identity resolution agent."""
        logger.info("Running Identity Resolution Agent")

        try:
            # Simulated result - in production, calls CrewAI agent
            result = {
                "systems_compared": ["hris", "payroll", "lms"],
                "total_records_processed": 1250,
                "auto_matched": 1180,
                "manual_review_needed": 45,
                "new_master_ids_created": 12,
                "duplicates_flagged": 3,
            }

            return {
                "identity_resolution_result": result,
                "current_agent": "sync_orchestration",
                "pending_agents": [a for a in state.get("pending_agents", [])
                                  if a != "identity_resolution"],
                "messages": [
                    HumanMessage(content=f"Identity resolution: {result['auto_matched']} matched, "
                                        f"{result['manual_review_needed']} need review"),
                ],
            }

        except Exception as e:
            logger.error(f"Identity resolution failed: {e}")
            return {
                "errors": [f"Identity resolution failed: {str(e)}"],
                "warnings": ["Continuing with existing identity mappings"],
                "current_agent": "sync_orchestration",
            }

    def _run_sync(self, state: EmployeeDataState) -> Dict[str, Any]:
        """Execute sync orchestration agent."""
        logger.info("Running Sync Orchestration Agent")

        try:
            result = {
                "systems_synced": ["bamboohr", "adp", "lms", "benefits"],
                "records_updated": 156,
                "records_created": 8,
                "conflicts_resolved": 3,
                "sync_duration_seconds": 45,
            }

            return {
                "sync_result": result,
                "current_agent": "termination_check",
                "pending_agents": [a for a in state.get("pending_agents", [])
                                  if a != "sync_orchestration"],
                "messages": [
                    HumanMessage(content=f"Sync complete: {result['records_updated']} updated"),
                ],
            }

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return {
                "errors": [f"Sync failed: {str(e)}"],
                "requires_escalation": True,
                "escalation_reason": "Sync failure - data may be stale",
            }

    def _check_terminations(self, state: EmployeeDataState) -> Dict[str, Any]:
        """Execute termination propagation agent."""
        logger.info("Running Termination Propagation Agent")

        try:
            result = {
                "terminated_employees_scanned": 23,
                "stale_access_detected": 5,
                "critical_violations": 1,
                "deprovisioning_tasks_created": 5,
                "notifications_sent": 1,
            }

            # Check for critical issues
            if result["critical_violations"] > 0:
                return {
                    "termination_result": result,
                    "current_agent": "supervisor_review",
                    "requires_escalation": True,
                    "escalation_reason": f"{result['critical_violations']} critical stale access violations",
                    "messages": [
                        HumanMessage(content=f"ALERT: {result['critical_violations']} critical violations found"),
                    ],
                }

            return {
                "termination_result": result,
                "current_agent": "supervisor_review",
                "messages": [
                    HumanMessage(content=f"Termination check: {result['stale_access_detected']} stale access found"),
                ],
            }

        except Exception as e:
            logger.error(f"Termination check failed: {e}")
            return {
                "errors": [f"Termination check failed: {str(e)}"],
                "current_agent": "supervisor_review",
            }

    def _supervisor_review(self, state: EmployeeDataState) -> Dict[str, Any]:
        """Supervisor reviews progress and determines next steps."""
        logger.info("Supervisor reviewing workflow state")

        # Update pending agents
        completed = []
        if state.get("identity_resolution_result"):
            completed.append("identity_resolution")
        if state.get("sync_result"):
            completed.append("sync_orchestration")
        if state.get("termination_result"):
            completed.append("termination_check")
        if state.get("compliance_result"):
            completed.append("compliance_monitor")

        remaining = [a for a in state.get("pending_agents", []) if a not in completed]

        return {
            "current_agent": "supervisor_review",
            "pending_agents": remaining,
        }

    def _route_from_supervisor(self, state: EmployeeDataState) -> str:
        """Route from supervisor based on state."""

        if state.get("requires_escalation"):
            return "escalate"

        pending = state.get("pending_agents", [])

        if "compliance_monitor" in pending:
            return "compliance"
        elif "generate_reports" in pending:
            return "reports"
        else:
            return "finalize"

    def _run_compliance(self, state: EmployeeDataState) -> Dict[str, Any]:
        """Execute compliance monitor agent."""
        logger.info("Running Compliance Monitor Agent")

        try:
            result = {
                "ghost_employees": 0,
                "stale_access_total": 5,
                "sla_compliance_pct": 94.2,
                "audit_trail_complete": True,
                "recommendations": [
                    "Review 5 stale access cases within 48 hours",
                ],
            }

            return {
                "compliance_result": result,
                "current_agent": "supervisor_review",
                "pending_agents": [a for a in state.get("pending_agents", [])
                                  if a != "compliance_monitor"],
                "messages": [
                    HumanMessage(content=f"Compliance check: SLA at {result['sla_compliance_pct']}%"),
                ],
            }

        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
            return {
                "errors": [f"Compliance check failed: {str(e)}"],
                "current_agent": "supervisor_review",
            }

    def _generate_reports(self, state: EmployeeDataState) -> Dict[str, Any]:
        """Execute reporting agent."""
        logger.info("Running Reporting Agent")

        reports = {
            "executive_summary": f"Employee data sync completed. {state.get('sync_result', {}).get('records_updated', 0)} records updated.",
            "hr_ops_actions": f"{state.get('termination_result', {}).get('deprovisioning_tasks_created', 0)} deprovisioning tasks pending.",
            "compliance_scorecard": json.dumps(state.get("compliance_result", {})),
        }

        return {
            "reports_generated": reports,
            "current_agent": "finalize",
            "pending_agents": [],
        }

    def _escalate(self, state: EmployeeDataState) -> Dict[str, Any]:
        """Handle escalation to human operators."""
        logger.warning(f"Escalating: {state.get('escalation_reason')}")

        # In production, send Slack/email alert
        return {
            "status": "escalated",
            "messages": [
                HumanMessage(content=f"ESCALATED: {state.get('escalation_reason')}"),
            ],
        }

    def _finalize(self, state: EmployeeDataState) -> Dict[str, Any]:
        """Finalize the workflow run."""
        logger.info(f"Finalizing run: {state.get('run_id')}")

        # Persist to Redis
        run_summary = {
            "run_id": state["run_id"],
            "run_type": state.get("run_type", "scheduled_sync"),
            "timestamp": state["run_timestamp"],
            "status": "completed" if not state.get("errors") else "completed_with_errors",
            "identity_resolution": state.get("identity_resolution_result"),
            "sync": state.get("sync_result"),
            "terminations": state.get("termination_result"),
            "compliance": state.get("compliance_result"),
            "reports": state.get("reports_generated"),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
        }

        self.redis_client.setex(
            f"emp-sync:{state['run_id']}",
            86400 * 90,  # 90 day retention for audit
            json.dumps(run_summary),
        )

        return {"status": "completed"}

    def run_scheduled_sync(self) -> Dict[str, Any]:
        """Execute a scheduled full sync."""
        return self._execute(run_type="scheduled_sync")

    def run_termination_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Execute termination propagation for a specific event."""
        return self._execute(run_type="termination_event", trigger_event=event)

    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the workflow with given parameters."""
        initial_state: EmployeeDataState = {
            "run_id": "",
            "run_type": kwargs.get("run_type", "scheduled_sync"),
            "run_timestamp": "",
            "status": "pending",
            "identity_resolution_result": None,
            "sync_result": None,
            "termination_result": None,
            "compliance_result": None,
            "reports_generated": None,
            "errors": [],
            "warnings": [],
            "messages": [],
            "current_agent": "",
            "pending_agents": [],
            "requires_escalation": False,
            "escalation_reason": None,
            "trigger_event": kwargs.get("trigger_event"),
        }

        thread_id = f"thread-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        return self.graph.invoke(
            initial_state,
            {"configurable": {"thread_id": thread_id}},
        )


# Usage
if __name__ == "__main__":
    workflow = EmployeeDataWorkflow()
    result = workflow.run_scheduled_sync()
    print(f"Workflow completed: {result['status']}")`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Production deployment configuration with Docker, LangSmith tracing, and Prometheus metrics for monitoring the employee data unification pipeline.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'Grafana'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose Production Deployment',
                description:
                  'Docker Compose configuration for deploying the employee data unification multi-agent system with event-driven sync triggers.',
                code: `version: '3.8'

services:
  employee-data-agents:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: employee-data-agents
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGCHAIN_API_KEY=\${LANGCHAIN_API_KEY}
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_PROJECT=employee-data-unification
      - BAMBOOHR_API_KEY=\${BAMBOOHR_API_KEY}
      - BAMBOOHR_SUBDOMAIN=\${BAMBOOHR_SUBDOMAIN}
      - ADP_CLIENT_ID=\${ADP_CLIENT_ID}
      - ADP_CLIENT_SECRET=\${ADP_CLIENT_SECRET}
      - LMS_API_ENDPOINT=\${LMS_API_ENDPOINT}
      - LMS_API_TOKEN=\${LMS_API_TOKEN}
      - BENEFITS_API_URL=\${BENEFITS_API_URL}
      - DATABASE_URL=postgresql://\${DB_USER}:\${DB_PASSWORD}@postgres:5432/hr_unified
      - REDIS_URL=redis://redis:6379
      - SLACK_BOT_TOKEN=\${SLACK_BOT_TOKEN}
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
      - LOG_LEVEL=INFO
    depends_on:
      - redis
      - postgres
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
    ports:
      - "8080:8080"  # API for manual triggers
    restart: unless-stopped
    networks:
      - employee-data-net
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

  # Event listener for real-time termination propagation
  termination-listener:
    build:
      context: .
      dockerfile: Dockerfile.listener
    container_name: termination-listener
    environment:
      - BAMBOOHR_WEBHOOK_SECRET=\${BAMBOOHR_WEBHOOK_SECRET}
      - REDIS_URL=redis://redis:6379
      - AGENT_API_URL=http://employee-data-agents:8080
    ports:
      - "8081:8081"
    depends_on:
      - employee-data-agents
      - redis
    networks:
      - employee-data-net
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: employee-data-redis
    command: >
      redis-server
      --appendonly yes
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
      --save 900 1
      --save 300 10
    volumes:
      - redis-data:/data
    networks:
      - employee-data-net
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: employee-data-postgres
    environment:
      - POSTGRES_USER=\${DB_USER}
      - POSTGRES_PASSWORD=\${DB_PASSWORD}
      - POSTGRES_DB=hr_unified
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./sql/init:/docker-entrypoint-initdb.d:ro
    networks:
      - employee-data-net
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: employee-data-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=90d'
    ports:
      - "9090:9090"
    networks:
      - employee-data-net
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: employee-data-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    networks:
      - employee-data-net
    restart: unless-stopped

  # Scheduler for periodic syncs
  scheduler:
    image: mcuadros/ofelia:latest
    container_name: employee-data-scheduler
    depends_on:
      - employee-data-agents
    command: daemon --docker
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    labels:
      # Full sync every 6 hours
      ofelia.job-exec.full-sync.schedule: "0 */6 * * *"
      ofelia.job-exec.full-sync.command: "python -m employee_data.run_sync --type full"
      ofelia.job-exec.full-sync.container: "employee-data-agents"
      # Compliance check daily
      ofelia.job-exec.compliance.schedule: "0 7 * * *"
      ofelia.job-exec.compliance.command: "python -m employee_data.run_sync --type compliance"
      ofelia.job-exec.compliance.container: "employee-data-agents"
    networks:
      - employee-data-net
    restart: unless-stopped

volumes:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:

networks:
  employee-data-net:
    driver: bridge`,
              },
              {
                language: 'python',
                title: 'Observability & Metrics Instrumentation',
                description:
                  'Prometheus metrics and LangSmith tracing for monitoring sync health, identity match rates, and SLA compliance.',
                code: `"""Observability instrumentation for Employee Data Unification Pipeline."""
import functools
import logging
import time
from typing import Any, Callable, Dict, Optional

from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree
from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Prometheus Metrics ---

# Sync metrics
SYNC_RUNS_TOTAL = Counter(
    "employee_sync_runs_total",
    "Total number of sync runs",
    ["run_type", "status"],
)

SYNC_DURATION_SECONDS = Histogram(
    "employee_sync_duration_seconds",
    "Sync run duration in seconds",
    ["run_type"],
    buckets=[10, 30, 60, 120, 300, 600, 1200],
)

RECORDS_SYNCED = Counter(
    "employee_records_synced_total",
    "Total records synced across systems",
    ["system", "operation"],  # operation: created, updated, deleted
)

# Identity resolution metrics
IDENTITY_MATCH_RATE = Gauge(
    "employee_identity_match_rate",
    "Percentage of records auto-matched",
)

MANUAL_REVIEW_QUEUE = Gauge(
    "employee_manual_review_queue_size",
    "Number of records pending manual identity review",
)

MASTER_IDS_TOTAL = Gauge(
    "employee_master_ids_total",
    "Total number of master employee identities",
)

# Termination propagation metrics
STALE_ACCESS_VIOLATIONS = Gauge(
    "employee_stale_access_violations",
    "Current count of stale access violations",
    ["urgency"],  # critical, overdue, within_sla
)

DEPROVISIONING_TASKS = Gauge(
    "employee_deprovisioning_tasks",
    "Current deprovisioning task queue size",
    ["status"],  # open, completed, overdue
)

SLA_COMPLIANCE_PCT = Gauge(
    "employee_deprovisioning_sla_compliance_pct",
    "Percentage of terminations propagated within SLA",
)

# Compliance metrics
GHOST_EMPLOYEES = Gauge(
    "employee_ghost_employees_count",
    "Number of potential ghost employees detected",
)

DATA_QUALITY_SCORE = Gauge(
    "employee_data_quality_score",
    "Overall data quality score (0-100)",
)

# System health
LAST_SUCCESSFUL_SYNC = Gauge(
    "employee_last_successful_sync_timestamp",
    "Timestamp of last successful sync",
)

API_ERRORS = Counter(
    "employee_api_errors_total",
    "API errors by system",
    ["system", "error_type"],
)


# --- Traced Agent Decorator ---

def traced_agent(agent_name: str) -> Callable:
    """Decorator for tracing agent executions with metrics."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        @traceable(name=agent_name, run_type="chain")
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            status = "success"

            try:
                result = func(*args, **kwargs)

                # Record agent-specific metrics
                if agent_name == "identity_resolution_agent" and result:
                    auto_matched = result.get("auto_matched", 0)
                    total = result.get("total_records_processed", 1)
                    IDENTITY_MATCH_RATE.set(auto_matched / total * 100 if total else 0)
                    MANUAL_REVIEW_QUEUE.set(result.get("manual_review_needed", 0))
                    MASTER_IDS_TOTAL.inc(result.get("new_master_ids_created", 0))

                elif agent_name == "sync_orchestration_agent" and result:
                    for system in result.get("systems_synced", []):
                        RECORDS_SYNCED.labels(
                            system=system, operation="updated"
                        ).inc(result.get("records_updated", 0) // len(result.get("systems_synced", [1])))
                    LAST_SUCCESSFUL_SYNC.set(time.time())

                elif agent_name == "termination_propagation_agent" and result:
                    STALE_ACCESS_VIOLATIONS.labels(urgency="critical").set(
                        result.get("critical_violations", 0)
                    )
                    DEPROVISIONING_TASKS.labels(status="open").set(
                        result.get("deprovisioning_tasks_created", 0)
                    )

                elif agent_name == "compliance_monitor_agent" and result:
                    GHOST_EMPLOYEES.set(result.get("ghost_employees", 0))
                    SLA_COMPLIANCE_PCT.set(result.get("sla_compliance_pct", 0))

                return result

            except Exception as e:
                status = "error"
                logger.error(f"{agent_name} failed: {e}")
                API_ERRORS.labels(system=agent_name, error_type=type(e).__name__).inc()
                raise

            finally:
                duration = time.time() - start_time
                SYNC_DURATION_SECONDS.labels(run_type=agent_name).observe(duration)
                logger.info(f"{agent_name} completed in {duration:.2f}s")

        return wrapper
    return decorator


class UnificationObserver:
    """Observer for the employee data unification pipeline."""

    def __init__(self, prometheus_port: int = 8000):
        self.prometheus_port = prometheus_port
        self._started = False

    def start_metrics_server(self) -> None:
        """Start Prometheus metrics HTTP server."""
        if not self._started:
            start_http_server(self.prometheus_port)
            self._started = True
            logger.info(f"Prometheus metrics at :{self.prometheus_port}/metrics")

    @traceable(name="employee_data_unification_workflow", run_type="chain")
    def observe_workflow(
        self,
        workflow_func: Callable,
        run_type: str = "scheduled_sync",
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute and observe a complete workflow run."""

        start_time = time.time()

        try:
            result = workflow_func(*args, **kwargs)

            status = result.get("status", "unknown")
            SYNC_RUNS_TOTAL.labels(run_type=run_type, status=status).inc()

            if status == "completed":
                LAST_SUCCESSFUL_SYNC.set(time.time())

            # Add metadata to LangSmith trace
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.metadata = {
                    "run_id": result.get("run_id"),
                    "run_type": run_type,
                    "duration_seconds": time.time() - start_time,
                    "records_synced": result.get("sync_result", {}).get("records_updated", 0),
                    "stale_access_found": result.get("termination_result", {}).get("stale_access_detected", 0),
                }

            return result

        except Exception as e:
            SYNC_RUNS_TOTAL.labels(run_type=run_type, status="failed").inc()
            logger.error(f"Workflow failed: {e}")
            raise

    def calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score based on current metrics."""
        # Weighted scoring of quality indicators
        scores = {
            "match_rate": IDENTITY_MATCH_RATE._value.get() if hasattr(IDENTITY_MATCH_RATE, '_value') else 95,
            "sla_compliance": SLA_COMPLIANCE_PCT._value.get() if hasattr(SLA_COMPLIANCE_PCT, '_value') else 95,
            "no_ghost_employees": 100 if GHOST_EMPLOYEES._value.get() == 0 else 50 if hasattr(GHOST_EMPLOYEES, '_value') else 100,
        }

        weights = {"match_rate": 0.4, "sla_compliance": 0.4, "no_ghost_employees": 0.2}
        total_score = sum(scores[k] * weights[k] for k in weights)

        DATA_QUALITY_SCORE.set(total_score)
        return total_score


# Usage
if __name__ == "__main__":
    observer = UnificationObserver(prometheus_port=8000)
    observer.start_metrics_server()

    @traced_agent("sync_orchestration_agent")
    def test_sync():
        time.sleep(1)
        return {"records_updated": 100, "systems_synced": ["hris", "payroll"]}

    result = test_sync()
    print(f"Sync result: {result}")
    print(f"Data quality score: {observer.calculate_data_quality_score()}")`,
              },
            ],
          },
        ],
      },
    },
  ],
};
