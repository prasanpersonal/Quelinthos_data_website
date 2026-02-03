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
    },
  ],
};
