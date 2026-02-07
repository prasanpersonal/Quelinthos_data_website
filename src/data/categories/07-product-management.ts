import type { Category } from '../types.ts';

export const productManagementCategory: Category = {
  id: 'product-management',
  number: 7,
  title: 'Product Management',
  shortTitle: 'Product',
  description:
    'Aggregate scattered feedback into actionable insights and fix analytics implementations that lie to your product team.',
  icon: 'Boxes',
  accentColor: 'neon-blue',
  painPoints: [
    // ── Pain Point 1: Feedback Aggregation Failure ──────────────────────
    {
      id: 'feedback-aggregation',
      number: 1,
      title: 'Feedback Aggregation Failure',
      subtitle: 'Customer Feedback Scattered Across 10+ Channels',
      summary:
        'Feature requests live in Intercom, Zendesk, Slack, email, and sales call notes. Your product roadmap is built on whoever shouts loudest, not data.',
      tags: ['feedback', 'product-management', 'nlp'],
      metrics: {
        annualCostRange: '$300K - $1.5M',
        roi: '6x',
        paybackPeriod: '3-4 months',
        investmentRange: '$60K - $120K',
      },
      price: {
        present: {
          title: 'Current State of Feedback Chaos',
          description:
            'Customer feedback is fragmented across Intercom, Zendesk, Slack, email threads, sales call transcripts, and social media — with no single source of truth.',
          bullets: [
            'Feature requests duplicated across 10+ channels with no deduplication',
            'Product managers spend 12+ hours per week manually triaging feedback',
            'Roadmap decisions driven by recency bias and loudest stakeholders',
            'No quantitative signal on which features impact retention or expansion',
          ],
          severity: 'high',
        },
        root: {
          title: 'Why Feedback Stays Scattered',
          description:
            'Each team adopted its own tool for capturing customer voice, and no integration layer was ever built to unify, classify, or prioritize the signals.',
          bullets: [
            'Support, sales, and CS teams use different platforms with no shared taxonomy',
            'No NLP pipeline to extract themes, sentiment, or feature requests from unstructured text',
            'Feedback volume exceeds human capacity to read, tag, and synthesize',
            'Legacy tools lack APIs or export formats needed for consolidation',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Business Impact of Misaligned Roadmaps',
          description:
            'Building the wrong features because of incomplete feedback leads to churn, missed expansion revenue, and engineering waste.',
          bullets: [
            'Engineering cycles wasted on low-impact features that customers never asked for',
            'Churn increases when high-value accounts feel their feedback is ignored',
            'Competitive deals lost because requested features were deprioritized without data',
            'Product-market fit signals buried under noise from vocal minorities',
          ],
          severity: 'high',
        },
        cost: {
          title: 'Investment to Unify Feedback',
          description:
            'Building an NLP-powered feedback aggregation pipeline requires integration work, model tuning, and a unified repository — but the tooling is mature.',
          bullets: [
            'API integrations with 5-10 feedback sources (Intercom, Zendesk, Slack, email, CRM)',
            'NLP clustering model for theme extraction and sentiment scoring',
            'Unified feedback warehouse with deduplication logic',
            'Dashboard and alerting layer for product team consumption',
          ],
          severity: 'medium',
        },
        expectedReturn: {
          title: 'Data-Driven Roadmap Prioritization',
          description:
            'A unified feedback system replaces gut-feel prioritization with quantified demand signals tied to revenue, retention, and customer segments.',
          bullets: [
            'Reduce PM triage time by 70%, freeing 8+ hours per week for strategic work',
            'Identify top-requested features by ARR-weighted customer segment in real time',
            'Decrease feature miss rate by 40% — build what customers actually need',
            'Surface churn-risk signals 4-6 weeks earlier through sentiment trend detection',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Build an NLP-powered feedback aggregation pipeline that ingests customer feedback from all channels, clusters it into themes, scores sentiment, and surfaces prioritized insights to the product team.',
        prerequisites: [
          'API access to feedback sources (Intercom, Zendesk, Slack, email)',
          'Python 3.10+ with scikit-learn, sentence-transformers, and pandas',
          'PostgreSQL or Snowflake for the unified feedback repository',
          'Basic understanding of NLP concepts (embeddings, clustering)',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Build Unified Feedback Repository',
            description:
              'Create a normalized schema that captures feedback from every source with consistent metadata, enabling cross-channel analysis and deduplication.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Unified Feedback Repository Schema',
                description:
                  'Central table that normalizes feedback from all channels into a single queryable format with source tracking and NLP-ready fields.',
                code: `-- Unified feedback repository for cross-channel aggregation
CREATE TABLE unified_feedback (
    feedback_id        UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    source_platform    VARCHAR(50) NOT NULL,        -- intercom, zendesk, slack, email, salesforce
    source_id          VARCHAR(255) NOT NULL,        -- original ID in source system
    customer_id        UUID REFERENCES customers(customer_id),
    account_id         UUID REFERENCES accounts(account_id),
    account_arr        NUMERIC(12,2),                -- for ARR-weighted prioritization
    raw_text           TEXT NOT NULL,
    cleaned_text       TEXT,                          -- preprocessed for NLP
    feedback_type      VARCHAR(30),                   -- feature_request, bug, complaint, praise
    sentiment_score    NUMERIC(4,3),                  -- -1.000 to 1.000
    cluster_id         INTEGER,                       -- assigned by clustering pipeline
    cluster_label      VARCHAR(200),                  -- human-readable theme
    embedding_vector   VECTOR(384),                   -- sentence-transformer embedding
    is_duplicate       BOOLEAN DEFAULT FALSE,
    duplicate_of       UUID REFERENCES unified_feedback(feedback_id),
    submitted_at       TIMESTAMPTZ NOT NULL,
    ingested_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_platform, source_id)
);

CREATE INDEX idx_feedback_cluster ON unified_feedback (cluster_id)
    WHERE is_duplicate = FALSE;
CREATE INDEX idx_feedback_sentiment ON unified_feedback (sentiment_score, submitted_at);
CREATE INDEX idx_feedback_account ON unified_feedback (account_id, submitted_at);`,
              },
              {
                language: 'python',
                title: 'Multi-Channel Feedback Ingestion',
                description:
                  'Ingest and normalize feedback from Intercom, Zendesk, and Slack into the unified repository with consistent schema mapping.',
                code: `import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass
import requests
import psycopg2
from psycopg2.extras import execute_values

@dataclass
class NormalizedFeedback:
    source_platform: str
    source_id: str
    customer_email: str
    raw_text: str
    submitted_at: datetime

class FeedbackIngestionPipeline:
    """Ingests feedback from multiple channels into the unified repository."""

    def __init__(self, db_conn, config: dict):
        self.conn = db_conn
        self.config = config

    def ingest_intercom(self, since_hours: int = 24) -> list[NormalizedFeedback]:
        headers = {"Authorization": f"Bearer {self.config['intercom_token']}"}
        resp = requests.get(
            "https://api.intercom.io/conversations",
            headers=headers,
            params={"updated_after": int((datetime.now(timezone.utc).timestamp()) - since_hours * 3600)},
        )
        conversations = resp.json().get("conversations", [])
        results = []
        for convo in conversations:
            body = convo.get("source", {}).get("body", "")
            if not body or len(body.strip()) < 10:
                continue
            results.append(NormalizedFeedback(
                source_platform="intercom",
                source_id=convo["id"],
                customer_email=convo.get("source", {}).get("author", {}).get("email", ""),
                raw_text=body,
                submitted_at=datetime.fromtimestamp(convo["created_at"], tz=timezone.utc),
            ))
        return results

    def ingest_zendesk(self, since_hours: int = 24) -> list[NormalizedFeedback]:
        auth = (self.config["zendesk_email"], self.config["zendesk_token"])
        resp = requests.get(
            f"https://{self.config['zendesk_subdomain']}.zendesk.com/api/v2/tickets/recent.json",
            auth=auth,
        )
        tickets = resp.json().get("tickets", [])
        return [
            NormalizedFeedback(
                source_platform="zendesk",
                source_id=str(t["id"]),
                customer_email=t.get("requester", {}).get("email", ""),
                raw_text=t.get("description", ""),
                submitted_at=datetime.fromisoformat(t["created_at"]),
            )
            for t in tickets if t.get("description")
        ]

    def write_to_repository(self, items: list[NormalizedFeedback]) -> int:
        query = """
            INSERT INTO unified_feedback (source_platform, source_id, raw_text, submitted_at)
            VALUES %s
            ON CONFLICT (source_platform, source_id) DO NOTHING
        """
        rows = [(f.source_platform, f.source_id, f.raw_text, f.submitted_at) for f in items]
        with self.conn.cursor() as cur:
            execute_values(cur, query, rows)
        self.conn.commit()
        return len(rows)`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'NLP Feedback Clustering & Sentiment Pipeline',
            description:
              'Apply sentence embeddings and clustering to group similar feedback into themes, then score sentiment to surface urgent signals.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Embedding Generation & Theme Clustering',
                description:
                  'Generate sentence embeddings for each feedback item, cluster them into themes using HDBSCAN, and assign human-readable labels to each cluster.',
                code: `import numpy as np
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import psycopg2

class FeedbackClusteringPipeline:
    """Clusters feedback into themes using sentence embeddings + HDBSCAN."""

    def __init__(self, db_conn, model_name: str = "all-MiniLM-L6-v2"):
        self.conn = db_conn
        self.model = SentenceTransformer(model_name)

    def fetch_unclustered(self, limit: int = 5000) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT feedback_id, cleaned_text
                FROM unified_feedback
                WHERE cluster_id IS NULL
                  AND is_duplicate = FALSE
                  AND cleaned_text IS NOT NULL
                ORDER BY submitted_at DESC
                LIMIT %s
            """, (limit,))
            return [{"id": row[0], "text": row[1]} for row in cur.fetchall()]

    def generate_embeddings(self, items: list[dict]) -> np.ndarray:
        texts = [item["text"] for item in items]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=64)
        return embeddings

    def cluster_and_label(self, items: list[dict], embeddings: np.ndarray) -> dict:
        clusterer = HDBSCAN(min_cluster_size=5, min_samples=3, metric="euclidean")
        labels = clusterer.fit_predict(embeddings)
        cluster_texts: dict[int, list[str]] = {}
        for item, label in zip(items, labels):
            if label == -1:
                continue
            cluster_texts.setdefault(label, []).append(item["text"])

        # Generate human-readable labels via TF-IDF top terms
        cluster_labels = {}
        for cid, texts in cluster_texts.items():
            tfidf = TfidfVectorizer(max_features=5, stop_words="english")
            tfidf.fit(texts)
            top_terms = tfidf.get_feature_names_out()
            cluster_labels[cid] = " / ".join(top_terms[:3]).title()

        return {"labels": labels, "cluster_labels": cluster_labels}

    def persist_clusters(self, items: list[dict], result: dict) -> int:
        updated = 0
        with self.conn.cursor() as cur:
            for item, label in zip(items, result["labels"]):
                cluster_label = result["cluster_labels"].get(int(label), "Uncategorized")
                cur.execute("""
                    UPDATE unified_feedback
                    SET cluster_id = %s, cluster_label = %s
                    WHERE feedback_id = %s
                """, (int(label), cluster_label, item["id"]))
                updated += 1
        self.conn.commit()
        return updated`,
              },
              {
                language: 'python',
                title: 'Sentiment Scoring Pipeline',
                description:
                  'Score each feedback item for sentiment intensity, flag negative trends, and detect urgency signals for the product team.',
                code: `from transformers import pipeline as hf_pipeline
from datetime import datetime, timedelta, timezone
import psycopg2

class SentimentScoringPipeline:
    """Scores feedback sentiment and detects negative trend spikes."""

    def __init__(self, db_conn):
        self.conn = db_conn
        self.sentiment_model = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )

    def score_unscored_feedback(self, batch_size: int = 200) -> int:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT feedback_id, cleaned_text
                FROM unified_feedback
                WHERE sentiment_score IS NULL AND cleaned_text IS NOT NULL
                ORDER BY submitted_at DESC
                LIMIT %s
            """, (batch_size,))
            rows = cur.fetchall()

        if not rows:
            return 0

        texts = [row[1][:512] for row in rows]
        predictions = self.sentiment_model(texts, batch_size=32)

        with self.conn.cursor() as cur:
            for (fid, _), pred in zip(rows, predictions):
                score = pred["score"] if pred["label"] == "POSITIVE" else -pred["score"]
                cur.execute(
                    "UPDATE unified_feedback SET sentiment_score = %s WHERE feedback_id = %s",
                    (round(score, 3), fid),
                )
        self.conn.commit()
        return len(rows)

    def detect_sentiment_drops(self, window_days: int = 7, threshold: float = -0.15) -> list[dict]:
        """Find clusters where average sentiment dropped significantly in the last window."""
        with self.conn.cursor() as cur:
            cur.execute("""
                WITH recent AS (
                    SELECT cluster_id, cluster_label, AVG(sentiment_score) AS avg_recent
                    FROM unified_feedback
                    WHERE submitted_at >= NOW() - INTERVAL '%s days'
                      AND sentiment_score IS NOT NULL AND cluster_id IS NOT NULL
                    GROUP BY cluster_id, cluster_label
                ),
                baseline AS (
                    SELECT cluster_id, AVG(sentiment_score) AS avg_baseline
                    FROM unified_feedback
                    WHERE submitted_at < NOW() - INTERVAL '%s days'
                      AND sentiment_score IS NOT NULL AND cluster_id IS NOT NULL
                    GROUP BY cluster_id
                )
                SELECT r.cluster_id, r.cluster_label,
                       r.avg_recent, b.avg_baseline,
                       (r.avg_recent - b.avg_baseline) AS sentiment_delta
                FROM recent r JOIN baseline b ON r.cluster_id = b.cluster_id
                WHERE (r.avg_recent - b.avg_baseline) < %s
                ORDER BY sentiment_delta ASC
            """, (window_days, window_days, threshold))
            return [
                {"cluster_id": r[0], "label": r[1], "recent": float(r[2]),
                 "baseline": float(r[3]), "delta": float(r[4])}
                for r in cur.fetchall()
            ]`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate the feedback pipeline with data quality assertions and automated pytest checks that ensure deduplication accuracy, embedding quality, and NLP clustering integrity.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Feedback Deduplication & Embedding Quality Assertions',
                description:
                  'Data quality assertions that verify deduplication accuracy, embedding completeness, and feedback classification consistency across the unified repository.',
                code: `-- Feedback pipeline data quality assertions
-- Assert 1: Deduplication accuracy - no exact-text duplicates remain unmarked
WITH duplicate_check AS (
    SELECT
        cleaned_text,
        COUNT(*)                                          AS occurrence_count,
        COUNT(*) FILTER (WHERE is_duplicate = FALSE)      AS unmarked_count,
        ARRAY_AGG(feedback_id ORDER BY ingested_at)       AS feedback_ids
    FROM unified_feedback
    WHERE cleaned_text IS NOT NULL
      AND LENGTH(cleaned_text) > 20
    GROUP BY cleaned_text
    HAVING COUNT(*) > 1
       AND COUNT(*) FILTER (WHERE is_duplicate = FALSE) > 1
),
-- Assert 2: Embedding coverage - all cleaned feedback has embeddings
embedding_coverage AS (
    SELECT
        COUNT(*)                                          AS total_cleaned,
        COUNT(*) FILTER (WHERE embedding_vector IS NULL)  AS missing_embeddings,
        ROUND(
            COUNT(*) FILTER (WHERE embedding_vector IS NOT NULL)::NUMERIC
            / NULLIF(COUNT(*), 0) * 100, 2
        )                                                 AS coverage_pct
    FROM unified_feedback
    WHERE cleaned_text IS NOT NULL
      AND is_duplicate = FALSE
      AND ingested_at < NOW() - INTERVAL '1 hour'
),
-- Assert 3: Cluster assignment completeness
cluster_coverage AS (
    SELECT
        COUNT(*)                                          AS total_with_embeddings,
        COUNT(*) FILTER (WHERE cluster_id IS NULL)        AS unassigned_count,
        COUNT(DISTINCT cluster_id)                        AS active_clusters,
        ROUND(
            COUNT(*) FILTER (WHERE cluster_id IS NOT NULL)::NUMERIC
            / NULLIF(COUNT(*), 0) * 100, 2
        )                                                 AS assignment_pct
    FROM unified_feedback
    WHERE embedding_vector IS NOT NULL
      AND is_duplicate = FALSE
),
-- Assert 4: Sentiment scoring completeness
sentiment_coverage AS (
    SELECT
        COUNT(*)                                          AS total_feedback,
        COUNT(*) FILTER (WHERE sentiment_score IS NULL)   AS unscored_count,
        ROUND(AVG(sentiment_score), 3)                    AS global_avg_sentiment,
        ROUND(STDDEV(sentiment_score), 3)                 AS sentiment_stddev
    FROM unified_feedback
    WHERE cleaned_text IS NOT NULL
      AND is_duplicate = FALSE
)
SELECT
    'dedup_violations'       AS assertion,
    (SELECT COUNT(*) FROM duplicate_check) AS violation_count,
    CASE WHEN (SELECT COUNT(*) FROM duplicate_check) = 0
         THEN 'PASS' ELSE 'FAIL' END AS status
UNION ALL
SELECT
    'embedding_coverage',
    (SELECT missing_embeddings FROM embedding_coverage),
    CASE WHEN (SELECT coverage_pct FROM embedding_coverage) >= 98.0
         THEN 'PASS' ELSE 'FAIL' END
UNION ALL
SELECT
    'cluster_assignment',
    (SELECT unassigned_count FROM cluster_coverage),
    CASE WHEN (SELECT assignment_pct FROM cluster_coverage) >= 90.0
         THEN 'PASS' ELSE 'FAIL' END
UNION ALL
SELECT
    'sentiment_coverage',
    (SELECT unscored_count FROM sentiment_coverage),
    CASE WHEN (SELECT unscored_count FROM sentiment_coverage)::NUMERIC
              / NULLIF((SELECT total_feedback FROM sentiment_coverage), 0) < 0.05
         THEN 'PASS' ELSE 'FAIL' END;`,
              },
              {
                language: 'python',
                title: 'Pytest NLP Pipeline Validation Suite',
                description:
                  'Automated pytest suite that validates NLP pipeline accuracy including embedding generation, clustering quality metrics, and feedback classification consistency.',
                code: `import logging
import numpy as np
import pytest
from typing import Any
from dataclasses import dataclass
from unittest.mock import MagicMock
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class PipelineTestFixtures:
    """Shared fixtures for NLP pipeline validation."""
    model: SentenceTransformer
    sample_feedback: list[str]
    expected_clusters: int


@pytest.fixture(scope="module")
def pipeline_fixtures() -> PipelineTestFixtures:
    """Load model and sample data for pipeline tests."""
    logger.info("Loading sentence-transformer model for validation")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sample_feedback: list[str] = [
        "We need better CSV export functionality",
        "CSV export is broken and missing columns",
        "Please add CSV download to the reports page",
        "The dashboard is too slow to load",
        "Dashboard performance is terrible with large datasets",
        "Loading times on the dashboard are unacceptable",
        "Add SSO support for enterprise customers",
        "We require SAML-based single sign-on integration",
        "Enterprise SSO is a blocker for our procurement",
    ]
    return PipelineTestFixtures(model=model, sample_feedback=sample_feedback, expected_clusters=3)


@pytest.fixture(scope="module")
def embeddings(pipeline_fixtures: PipelineTestFixtures) -> np.ndarray:
    """Generate embeddings for sample feedback."""
    logger.info("Generating embeddings for %d samples", len(pipeline_fixtures.sample_feedback))
    return pipeline_fixtures.model.encode(pipeline_fixtures.sample_feedback)


class TestEmbeddingQuality:
    """Validate embedding generation produces consistent, meaningful vectors."""

    def test_embedding_dimensions(self, embeddings: np.ndarray) -> None:
        logger.info("Checking embedding dimensions: shape=%s", embeddings.shape)
        assert embeddings.shape[1] == 384, f"Expected 384-dim vectors, got {embeddings.shape[1]}"

    def test_embedding_norms_reasonable(self, embeddings: np.ndarray) -> None:
        norms: np.ndarray = np.linalg.norm(embeddings, axis=1)
        logger.info("Embedding norms: min=%.3f, max=%.3f, mean=%.3f", norms.min(), norms.max(), norms.mean())
        assert np.all(norms > 0.1), "Found near-zero embeddings indicating encoding failure"
        assert np.all(norms < 50.0), "Found abnormally large embedding norms"

    def test_similar_feedback_has_high_cosine(self, embeddings: np.ndarray) -> None:
        # First 3 items are all about CSV export - should be similar
        csv_embeddings: np.ndarray = embeddings[:3]
        normed: np.ndarray = csv_embeddings / np.linalg.norm(csv_embeddings, axis=1, keepdims=True)
        cosine_matrix: np.ndarray = normed @ normed.T
        min_similarity: float = float(cosine_matrix[np.triu_indices(3, k=1)].min())
        logger.info("Min cosine similarity within CSV-export cluster: %.3f", min_similarity)
        assert min_similarity > 0.5, f"Similar feedback cosine too low: {min_similarity}"


class TestClusteringQuality:
    """Validate clustering produces meaningful, well-separated groups."""

    def test_silhouette_score_acceptable(self, embeddings: np.ndarray) -> None:
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=3, min_samples=2)
        labels: np.ndarray = clusterer.fit_predict(embeddings)
        valid_mask: np.ndarray = labels >= 0
        if valid_mask.sum() < 4:
            pytest.skip("Too few clustered points for silhouette score")
        score: float = float(silhouette_score(embeddings[valid_mask], labels[valid_mask]))
        logger.info("Silhouette score: %.3f (threshold: 0.3)", score)
        assert score > 0.3, f"Silhouette score {score} below 0.3 threshold"

    def test_noise_ratio_acceptable(self, embeddings: np.ndarray) -> None:
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=3, min_samples=2)
        labels: np.ndarray = clusterer.fit_predict(embeddings)
        noise_ratio: float = float(np.sum(labels == -1)) / len(labels)
        logger.info("Noise ratio: %.2f (threshold: 0.40)", noise_ratio)
        assert noise_ratio < 0.40, f"Too many noise points: {noise_ratio:.0%}"`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Deploy the feedback aggregation pipeline as a containerized service with health checks, scheduled runs, and configuration management for multi-source ingestion.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Feedback Worker Deployment Script',
                description:
                  'Production deployment script for the feedback aggregation worker with Docker builds, health checks, rolling restarts, and rollback support.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------
# Feedback Aggregation Worker — Deployment Script
# Builds, tests, and deploys the feedback pipeline container
# -----------------------------------------------------------

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="\${SCRIPT_DIR}/.."
IMAGE_NAME="feedback-aggregation-worker"
REGISTRY="\${DOCKER_REGISTRY:-ghcr.io/myorg}"
DEPLOY_ENV="\${DEPLOY_ENV:-staging}"
COMPOSE_FILE="\${PROJECT_ROOT}/docker/docker-compose.\${DEPLOY_ENV}.yml"
HEALTH_ENDPOINT="http://localhost:8091/health"
HEALTH_TIMEOUT=60
ROLLBACK_TAG=""

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] \$*"; }
err() { log "ERROR: \$*" >&2; }

cleanup() {
    local exit_code=\$?
    if [[ \$exit_code -ne 0 && -n "\${ROLLBACK_TAG}" ]]; then
        err "Deployment failed (exit \${exit_code}). Rolling back to \${ROLLBACK_TAG}"
        docker tag "\${REGISTRY}/\${IMAGE_NAME}:\${ROLLBACK_TAG}" \\
                    "\${REGISTRY}/\${IMAGE_NAME}:latest"
        docker-compose -f "\${COMPOSE_FILE}" up -d --no-build feedback-worker
        log "Rollback complete."
    fi
}
trap cleanup EXIT

# Capture current image tag for rollback
ROLLBACK_TAG=\$(docker inspect --format='{{.Config.Image}}' feedback-worker 2>/dev/null | awk -F: '{print \$2}') || true
log "Current tag for rollback: \${ROLLBACK_TAG:-none}"

# Build and tag
GIT_SHA=\$(git -C "\${PROJECT_ROOT}" rev-parse --short HEAD)
BUILD_TAG="\${DEPLOY_ENV}-\${GIT_SHA}-\$(date +%Y%m%d%H%M%S)"
log "Building image: \${IMAGE_NAME}:\${BUILD_TAG}"
docker build \\
    --file "\${PROJECT_ROOT}/docker/Dockerfile.feedback-worker" \\
    --build-arg DEPLOY_ENV="\${DEPLOY_ENV}" \\
    --tag "\${REGISTRY}/\${IMAGE_NAME}:\${BUILD_TAG}" \\
    --tag "\${REGISTRY}/\${IMAGE_NAME}:latest" \\
    "\${PROJECT_ROOT}"

# Run unit tests inside container
log "Running in-container tests"
docker run --rm \\
    "\${REGISTRY}/\${IMAGE_NAME}:\${BUILD_TAG}" \\
    python -m pytest tests/ -x --tb=short -q

# Push to registry
log "Pushing to registry"
docker push "\${REGISTRY}/\${IMAGE_NAME}:\${BUILD_TAG}"
docker push "\${REGISTRY}/\${IMAGE_NAME}:latest"

# Deploy with rolling restart
log "Deploying to \${DEPLOY_ENV}"
docker-compose -f "\${COMPOSE_FILE}" pull feedback-worker
docker-compose -f "\${COMPOSE_FILE}" up -d --no-build feedback-worker

# Health check loop
log "Waiting for health check (timeout: \${HEALTH_TIMEOUT}s)"
elapsed=0
while [[ \$elapsed -lt \$HEALTH_TIMEOUT ]]; do
    if curl -sf "\${HEALTH_ENDPOINT}" > /dev/null 2>&1; then
        log "Health check passed after \${elapsed}s"
        break
    fi
    sleep 3
    elapsed=\$((elapsed + 3))
done

if [[ \$elapsed -ge \$HEALTH_TIMEOUT ]]; then
    err "Health check timed out after \${HEALTH_TIMEOUT}s"
    exit 1
fi

log "Deployment complete: \${IMAGE_NAME}:\${BUILD_TAG}"`,
              },
              {
                language: 'python',
                title: 'Multi-Source Feedback Ingestion Configuration',
                description:
                  'Configuration loader that manages API credentials, source-specific settings, and scheduling parameters for the multi-channel feedback ingestion pipeline.',
                code: `import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class SourceConfig:
    """Configuration for a single feedback source."""
    name: str
    enabled: bool
    api_base_url: str
    auth_token: str
    poll_interval_minutes: int = 60
    batch_size: int = 100
    custom_headers: dict[str, str] = field(default_factory=dict)
    field_mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level configuration for the feedback aggregation pipeline."""
    db_connection_string: str
    sources: list[SourceConfig]
    embedding_model: str = "all-MiniLM-L6-v2"
    clustering_min_size: int = 5
    sentiment_batch_size: int = 200
    dedup_similarity_threshold: float = 0.92
    slack_webhook_url: Optional[str] = None
    log_level: str = "INFO"
    max_retries: int = 3
    retry_delay_seconds: int = 30


class ConfigLoader:
    """Loads and validates pipeline configuration from file and environment."""

    ENV_PREFIX: str = "FEEDBACK_"

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path: Optional[Path] = Path(config_path) if config_path else None
        self._raw: dict[str, Any] = {}
        logger.info("ConfigLoader initialized with path=%s", self.config_path)

    def load(self) -> PipelineConfig:
        """Load config from file, then overlay environment variable overrides."""
        if self.config_path and self.config_path.exists():
            logger.info("Loading config from %s", self.config_path)
            with open(self.config_path) as f:
                self._raw = json.load(f)
        else:
            logger.warning("No config file found at %s, using env-only config", self.config_path)
            self._raw = {}

        # Environment variable overrides
        self._raw["db_connection_string"] = self._env(
            "DB_CONNECTION_STRING",
            self._raw.get("db_connection_string", ""),
        )
        self._raw["slack_webhook_url"] = self._env(
            "SLACK_WEBHOOK_URL",
            self._raw.get("slack_webhook_url"),
        )
        self._raw["log_level"] = self._env("LOG_LEVEL", self._raw.get("log_level", "INFO"))

        sources: list[SourceConfig] = self._load_sources(self._raw.get("sources", []))
        config = PipelineConfig(
            db_connection_string=self._raw["db_connection_string"],
            sources=sources,
            embedding_model=self._raw.get("embedding_model", "all-MiniLM-L6-v2"),
            clustering_min_size=int(self._raw.get("clustering_min_size", 5)),
            sentiment_batch_size=int(self._raw.get("sentiment_batch_size", 200)),
            dedup_similarity_threshold=float(self._raw.get("dedup_similarity_threshold", 0.92)),
            slack_webhook_url=self._raw.get("slack_webhook_url"),
            log_level=self._raw["log_level"],
        )
        self._validate(config)
        logger.info("Config loaded: %d sources, model=%s", len(config.sources), config.embedding_model)
        return config

    def _load_sources(self, raw_sources: list[dict[str, Any]]) -> list[SourceConfig]:
        """Parse source configurations with env override support for tokens."""
        sources: list[SourceConfig] = []
        for src in raw_sources:
            name: str = src["name"]
            token_env_key: str = f"{self.ENV_PREFIX}{name.upper()}_TOKEN"
            auth_token: str = os.environ.get(token_env_key, src.get("auth_token", ""))
            sources.append(SourceConfig(
                name=name,
                enabled=src.get("enabled", True),
                api_base_url=src["api_base_url"],
                auth_token=auth_token,
                poll_interval_minutes=int(src.get("poll_interval_minutes", 60)),
                batch_size=int(src.get("batch_size", 100)),
                custom_headers=src.get("custom_headers", {}),
                field_mapping=src.get("field_mapping", {}),
            ))
            logger.info("Source '%s': enabled=%s, poll=%dm", name, sources[-1].enabled, sources[-1].poll_interval_minutes)
        return sources

    def _validate(self, config: PipelineConfig) -> None:
        """Validate required configuration values."""
        if not config.db_connection_string:
            raise ValueError("db_connection_string is required")
        if not config.sources:
            logger.warning("No feedback sources configured — pipeline will have no input")
        active: int = sum(1 for s in config.sources if s.enabled)
        if active == 0:
            logger.warning("All feedback sources are disabled")

    def _env(self, key: str, default: Any = None) -> Any:
        """Read environment variable with prefix."""
        return os.environ.get(f"{self.ENV_PREFIX}{key}", default)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'ARR-Weighted Prioritization Dashboard',
            description:
              'Surface the highest-impact feedback themes weighted by customer ARR, request frequency, and sentiment urgency so the product team can prioritize with confidence. Includes monitoring for feedback volume and classification drift.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'ARR-Weighted Feature Request Rankings',
                description:
                  'Rank feedback clusters by total ARR of requesting accounts, request volume, and sentiment to produce a data-driven prioritization view.',
                code: `-- ARR-weighted feedback prioritization for product roadmap
WITH cluster_metrics AS (
    SELECT
        uf.cluster_id,
        uf.cluster_label,
        COUNT(DISTINCT uf.feedback_id)                   AS request_count,
        COUNT(DISTINCT uf.account_id)                    AS unique_accounts,
        COALESCE(SUM(DISTINCT uf.account_arr), 0)        AS total_arr_requesting,
        ROUND(AVG(uf.sentiment_score), 3)                AS avg_sentiment,
        MIN(uf.submitted_at)                             AS first_seen,
        MAX(uf.submitted_at)                             AS last_seen,
        COUNT(*) FILTER (
            WHERE uf.submitted_at >= NOW() - INTERVAL '30 days'
        )                                                 AS requests_last_30d
    FROM unified_feedback uf
    WHERE uf.cluster_id IS NOT NULL
      AND uf.is_duplicate = FALSE
      AND uf.feedback_type = 'feature_request'
    GROUP BY uf.cluster_id, uf.cluster_label
),
scored AS (
    SELECT *,
        -- Composite priority score: ARR weight + volume + momentum + urgency
        ROUND(
            (total_arr_requesting / NULLIF(MAX(total_arr_requesting) OVER (), 0)) * 40
            + (request_count::NUMERIC / NULLIF(MAX(request_count) OVER (), 0)) * 25
            + (requests_last_30d::NUMERIC / NULLIF(MAX(requests_last_30d) OVER (), 0)) * 20
            + (1 - LEAST(GREATEST(avg_sentiment, -1), 1)) * 15  -- lower sentiment = higher urgency
        , 2) AS priority_score
    FROM cluster_metrics
)
SELECT
    cluster_id,
    cluster_label                       AS theme,
    priority_score,
    request_count                       AS total_requests,
    unique_accounts,
    '$' || TO_CHAR(total_arr_requesting, 'FM999,999,999') AS arr_at_stake,
    avg_sentiment,
    requests_last_30d                   AS recent_momentum,
    first_seen::DATE                    AS first_reported,
    last_seen::DATE                     AS last_reported
FROM scored
ORDER BY priority_score DESC
LIMIT 25;`,
              },
              {
                language: 'python',
                title: 'Feedback Volume & Classification Drift Monitor',
                description:
                  'Monitors feedback ingestion volume, detects classification drift in NLP clusters, and sends Slack alerts when anomalies are detected.',
                code: `import json
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
import psycopg2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class DriftAlert:
    """Represents a detected drift or volume anomaly."""
    alert_type: str          # volume_drop, volume_spike, cluster_drift, classification_shift
    severity: str            # warning, critical
    metric_name: str
    expected_value: float
    actual_value: float
    deviation_pct: float
    details: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackPipelineMonitor:
    """Monitors feedback volume, classification drift, and pipeline health."""

    def __init__(
        self,
        db_conn: Any,
        slack_webhook_url: Optional[str] = None,
        volume_z_threshold: float = 2.5,
        drift_threshold: float = 0.15,
    ) -> None:
        self.conn = db_conn
        self.slack_webhook_url = slack_webhook_url
        self.volume_z_threshold = volume_z_threshold
        self.drift_threshold = drift_threshold
        logger.info(
            "Monitor initialized: z_threshold=%.1f, drift_threshold=%.2f",
            volume_z_threshold, drift_threshold,
        )

    def check_ingestion_volume(self, lookback_days: int = 28) -> list[DriftAlert]:
        """Detect anomalies in daily feedback ingestion volume per source."""
        alerts: list[DriftAlert] = []
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT source_platform,
                       DATE(ingested_at)       AS ingest_date,
                       COUNT(*)                AS daily_count
                FROM unified_feedback
                WHERE ingested_at >= CURRENT_DATE - INTERVAL '%s days'
                GROUP BY source_platform, DATE(ingested_at)
                ORDER BY source_platform, ingest_date
            """, (lookback_days,))
            rows = cur.fetchall()

        source_volumes: dict[str, list[float]] = {}
        source_today: dict[str, float] = {}
        today_str: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for platform, ingest_date, count in rows:
            date_str: str = ingest_date.strftime("%Y-%m-%d")
            if date_str == today_str:
                source_today[platform] = float(count)
            else:
                source_volumes.setdefault(platform, []).append(float(count))

        for platform, volumes in source_volumes.items():
            if len(volumes) < 7:
                continue
            arr: np.ndarray = np.array(volumes)
            median: float = float(np.median(arr))
            mad: float = float(np.median(np.abs(arr - median))) * 1.4826
            mad = max(mad, 1.0)
            actual: float = source_today.get(platform, 0.0)
            z_score: float = (actual - median) / mad

            if abs(z_score) >= self.volume_z_threshold:
                deviation_pct: float = round((actual - median) / median * 100, 1)
                alert_type: str = "volume_drop" if z_score < 0 else "volume_spike"
                severity: str = "critical" if abs(z_score) > 4.0 else "warning"
                alerts.append(DriftAlert(
                    alert_type=alert_type,
                    severity=severity,
                    metric_name=f"feedback_volume_{platform}",
                    expected_value=median,
                    actual_value=actual,
                    deviation_pct=deviation_pct,
                    details=f"{platform}: expected ~{median:.0f}/day, got {actual:.0f} (z={z_score:.1f})",
                ))
                logger.warning("Volume anomaly: %s", alerts[-1].details)

        return alerts

    def check_classification_drift(self, window_days: int = 7) -> list[DriftAlert]:
        """Detect drift in cluster size distributions indicating model degradation."""
        alerts: list[DriftAlert] = []
        with self.conn.cursor() as cur:
            cur.execute("""
                WITH recent AS (
                    SELECT cluster_id, cluster_label, COUNT(*) AS cnt
                    FROM unified_feedback
                    WHERE ingested_at >= CURRENT_DATE - INTERVAL '%s days'
                      AND cluster_id IS NOT NULL AND is_duplicate = FALSE
                    GROUP BY cluster_id, cluster_label
                ),
                baseline AS (
                    SELECT cluster_id, cluster_label, COUNT(*) AS cnt
                    FROM unified_feedback
                    WHERE ingested_at < CURRENT_DATE - INTERVAL '%s days'
                      AND ingested_at >= CURRENT_DATE - INTERVAL '%s days'
                      AND cluster_id IS NOT NULL AND is_duplicate = FALSE
                    GROUP BY cluster_id, cluster_label
                )
                SELECT
                    COALESCE(r.cluster_id, b.cluster_id)   AS cid,
                    COALESCE(r.cluster_label, b.cluster_label) AS label,
                    COALESCE(b.cnt, 0)                      AS baseline_cnt,
                    COALESCE(r.cnt, 0)                      AS recent_cnt
                FROM recent r
                FULL OUTER JOIN baseline b ON r.cluster_id = b.cluster_id
            """, (window_days, window_days, window_days * 4))
            rows = cur.fetchall()

        total_baseline: float = max(sum(r[2] for r in rows), 1.0)
        total_recent: float = max(sum(r[3] for r in rows), 1.0)
        for cid, label, baseline_cnt, recent_cnt in rows:
            baseline_pct: float = baseline_cnt / total_baseline
            recent_pct: float = recent_cnt / total_recent
            drift: float = abs(recent_pct - baseline_pct)
            if drift >= self.drift_threshold:
                alerts.append(DriftAlert(
                    alert_type="classification_shift",
                    severity="warning" if drift < 0.25 else "critical",
                    metric_name=f"cluster_distribution_{cid}",
                    expected_value=round(baseline_pct * 100, 1),
                    actual_value=round(recent_pct * 100, 1),
                    deviation_pct=round(drift * 100, 1),
                    details=f"Cluster '{label}' shifted from {baseline_pct:.1%} to {recent_pct:.1%}",
                ))
                logger.warning("Classification drift: %s", alerts[-1].details)

        return alerts

    def send_slack_alerts(self, alerts: list[DriftAlert]) -> bool:
        """Send aggregated alerts to Slack via incoming webhook."""
        if not alerts or not self.slack_webhook_url:
            logger.info("No alerts to send or no webhook configured")
            return False

        critical: list[DriftAlert] = [a for a in alerts if a.severity == "critical"]
        warnings: list[DriftAlert] = [a for a in alerts if a.severity == "warning"]
        header: str = f":rotating_light: *Feedback Pipeline Alert* — {len(alerts)} issues detected"
        blocks: list[str] = [header]

        if critical:
            blocks.append("\\n*Critical:*")
            for a in critical:
                blocks.append(f"  :red_circle: [{a.alert_type}] {a.details}")
        if warnings:
            blocks.append("\\n*Warnings:*")
            for a in warnings:
                blocks.append(f"  :warning: [{a.alert_type}] {a.details}")

        payload: dict[str, str] = {"text": "\\n".join(blocks)}
        try:
            resp = requests.post(self.slack_webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info("Slack alert sent successfully (%d alerts)", len(alerts))
            return True
        except requests.RequestException as exc:
            logger.error("Failed to send Slack alert: %s", exc)
            return False

    def run_all_checks(self) -> list[DriftAlert]:
        """Run all monitoring checks and alert on findings."""
        logger.info("Running feedback pipeline health checks")
        all_alerts: list[DriftAlert] = []
        all_alerts.extend(self.check_ingestion_volume())
        all_alerts.extend(self.check_classification_drift())
        if all_alerts:
            self.send_slack_alerts(all_alerts)
        logger.info("Health check complete: %d alerts", len(all_alerts))
        return all_alerts`,
              },
            ],
          },
        ],
        toolsUsed: [
          'Python (sentence-transformers, HDBSCAN, transformers)',
          'PostgreSQL with pgvector extension',
          'Intercom API / Zendesk API / Slack API',
          'scikit-learn (TF-IDF vectorizer)',
          'pytest',
          'Docker',
          'GitHub Actions',
          'cron / Airflow',
          'Slack API',
        ],
      },
      aiEasyWin: {
        overview:
          'Use ChatGPT/Claude with Zapier to automatically aggregate feedback from multiple channels, extract themes using AI prompts, and deliver weekly prioritized insights to your product team without any custom code.',
        estimatedMonthlyCost: '$100 - $200/month',
        primaryTools: [
          'ChatGPT Plus ($20/mo)',
          'Zapier Pro ($29.99/mo)',
          'Notion ($10/mo)',
        ],
        alternativeTools: [
          'Claude Pro ($20/mo)',
          'Make ($10.59/mo)',
          'Productboard AI ($25/mo)',
          'Airtable ($20/mo)',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Set up automated data extraction from your feedback sources (Intercom, Zendesk, Slack, email) using Zapier webhooks and scheduled triggers to collect feedback into a central Notion database or Google Sheet.',
            toolsUsed: ['Zapier', 'Notion', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Multi-Source Feedback Collector Configuration',
                description:
                  'Zapier workflow configuration that collects feedback from Intercom conversations, Zendesk tickets, and Slack messages into a unified Notion database.',
                code: `{
  "workflow_name": "Multi-Channel Feedback Aggregator",
  "triggers": [
    {
      "name": "intercom_conversation_closed",
      "app": "Intercom",
      "event": "Conversation Closed",
      "filter": {
        "field": "conversation.source.type",
        "condition": "contains",
        "value": "user"
      }
    },
    {
      "name": "zendesk_ticket_solved",
      "app": "Zendesk",
      "event": "Ticket Solved",
      "filter": {
        "field": "ticket.tags",
        "condition": "contains_any",
        "value": ["feature-request", "feedback", "suggestion"]
      }
    },
    {
      "name": "slack_reaction_added",
      "app": "Slack",
      "event": "New Reaction on Message",
      "filter": {
        "field": "reaction",
        "condition": "equals",
        "value": "feedback"
      }
    }
  ],
  "actions": [
    {
      "name": "add_to_notion",
      "app": "Notion",
      "action": "Create Database Item",
      "database_id": "{{FEEDBACK_DATABASE_ID}}",
      "properties": {
        "Title": "{{trigger.subject_or_first_50_chars}}",
        "Source": "{{trigger.source_name}}",
        "Raw Text": "{{trigger.body_or_message}}",
        "Customer Email": "{{trigger.customer_email}}",
        "Account Name": "{{trigger.company_name}}",
        "ARR": "{{trigger.account_arr}}",
        "Submitted Date": "{{trigger.created_at}}",
        "Status": "Pending Analysis"
      }
    }
  ],
  "schedule": {
    "intercom": "every_15_minutes",
    "zendesk": "every_30_minutes",
    "slack": "real_time"
  }
}`,
              },
              {
                language: 'yaml',
                title: 'Notion Feedback Database Schema',
                description:
                  'Schema for the unified Notion database that stores all collected feedback with fields optimized for AI analysis.',
                code: `# Notion Database Schema for Feedback Aggregation
database_name: "Product Feedback Hub"

properties:
  - name: "Title"
    type: "title"
    description: "Brief summary of the feedback"

  - name: "Source"
    type: "select"
    options:
      - Intercom
      - Zendesk
      - Slack
      - Email
      - Sales Call
      - G2 Review

  - name: "Raw Text"
    type: "text"
    description: "Full feedback content for AI analysis"

  - name: "Customer Email"
    type: "email"

  - name: "Account Name"
    type: "text"

  - name: "ARR"
    type: "number"
    format: "dollar"
    description: "Account ARR for weighted prioritization"

  - name: "Submitted Date"
    type: "date"

  - name: "Status"
    type: "select"
    options:
      - Pending Analysis
      - Analyzed
      - Actionable
      - Archived

  - name: "AI Theme"
    type: "select"
    description: "Theme assigned by AI analysis"

  - name: "AI Sentiment"
    type: "select"
    options:
      - Very Positive
      - Positive
      - Neutral
      - Negative
      - Very Negative

  - name: "AI Priority Score"
    type: "number"
    description: "1-10 priority score from AI analysis"

  - name: "AI Summary"
    type: "text"
    description: "One-line AI-generated summary"

views:
  - name: "Pending Analysis"
    filter: "Status = Pending Analysis"
    sort: "Submitted Date desc"

  - name: "By Theme"
    group_by: "AI Theme"
    sort: "AI Priority Score desc"

  - name: "High Priority"
    filter: "AI Priority Score >= 7"
    sort: "ARR desc"`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description:
              'Use ChatGPT or Claude to analyze batches of feedback, extract themes, score sentiment, identify feature requests, and generate prioritized summaries using structured prompts.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Feedback Theme Extraction Prompt Template',
                description:
                  'Structured prompt for ChatGPT/Claude to analyze feedback batches and extract themes, sentiment, and actionable insights.',
                code: `# Feedback Analysis Prompt Template
# Use with ChatGPT Plus or Claude Pro via Zapier

system_prompt: |
  You are a product management analyst specializing in customer feedback analysis.
  Your role is to analyze customer feedback and extract actionable insights.
  Always respond in valid JSON format.

user_prompt_template: |
  Analyze the following batch of customer feedback and provide structured insights.

  ## Feedback Items:
  {{feedback_batch}}

  ## Analysis Requirements:
  1. Identify the PRIMARY THEME for each feedback item
  2. Score SENTIMENT from -1.0 (very negative) to 1.0 (very positive)
  3. Classify as: feature_request, bug_report, complaint, praise, question
  4. Extract the CORE ASK in one sentence
  5. Assign PRIORITY SCORE (1-10) based on:
     - Urgency of the need
     - Specificity of the request
     - Potential business impact

  ## Response Format (JSON):
  {
    "analyzed_items": [
      {
        "feedback_id": "{{id}}",
        "theme": "string - e.g., 'Export Functionality', 'Dashboard Performance'",
        "sentiment_score": number,
        "feedback_type": "feature_request|bug_report|complaint|praise|question",
        "core_ask": "One sentence summary of what the customer wants",
        "priority_score": number,
        "suggested_tags": ["tag1", "tag2"]
      }
    ],
    "theme_summary": {
      "top_themes": [
        {"theme": "string", "count": number, "avg_sentiment": number}
      ],
      "emerging_patterns": ["pattern1", "pattern2"],
      "urgent_items": ["feedback_id1", "feedback_id2"]
    }
  }

batch_size: 20
model_settings:
  model: "gpt-4-turbo"
  temperature: 0.3
  max_tokens: 2000`,
              },
              {
                language: 'yaml',
                title: 'Weekly Prioritization Summary Prompt',
                description:
                  'Prompt template for generating a weekly executive summary of feedback themes weighted by customer ARR.',
                code: `# Weekly Feedback Prioritization Summary Prompt
# Run weekly via scheduled Zapier workflow

system_prompt: |
  You are a senior product strategist creating an executive summary of customer feedback.
  Focus on actionable insights and ARR-weighted prioritization.
  Be concise but comprehensive.

user_prompt_template: |
  Generate a weekly product feedback summary from the following analyzed data.

  ## This Week's Feedback ({{feedback_count}} items):
  {{weekly_feedback_json}}

  ## Account ARR Data:
  {{account_arr_mapping}}

  ## Generate:

  ### 1. Executive Summary (3-5 bullet points)
  - Top customer pain points this week
  - Notable sentiment shifts
  - Urgent items requiring immediate attention

  ### 2. ARR-Weighted Feature Priority
  Rank the top 5 feature requests by:
  - Total ARR of requesting customers
  - Number of unique accounts
  - Average sentiment score

  Format:
  | Rank | Feature Request | ARR at Stake | # Accounts | Sentiment |

  ### 3. Churn Risk Signals
  Identify feedback from high-ARR accounts with negative sentiment:
  - Account name
  - Specific concern
  - Recommended action

  ### 4. Quick Wins
  List 3 low-effort improvements that would address multiple complaints

  ### 5. Themes to Watch
  Emerging patterns that aren't urgent yet but show growth

  Keep the total summary under 500 words.

schedule: "Every Monday 8:00 AM"
output_destination: "Slack #product-team channel"`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Automate the entire workflow with Zapier: trigger AI analysis on new feedback batches, update the Notion database with insights, and deliver weekly summaries to Slack and email.',
            toolsUsed: ['Zapier', 'Slack', 'Email'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier AI Analysis Automation Workflow',
                description:
                  'Complete Zapier workflow that triggers ChatGPT analysis on new feedback batches and updates the database with results.',
                code: `{
  "workflow_name": "AI Feedback Analysis Pipeline",
  "trigger": {
    "type": "schedule",
    "schedule": "every_6_hours",
    "description": "Process pending feedback every 6 hours"
  },
  "steps": [
    {
      "step": 1,
      "name": "fetch_pending_feedback",
      "app": "Notion",
      "action": "Find Database Items",
      "config": {
        "database_id": "{{FEEDBACK_DATABASE_ID}}",
        "filter": {
          "property": "Status",
          "select": { "equals": "Pending Analysis" }
        },
        "page_size": 20
      }
    },
    {
      "step": 2,
      "name": "format_for_ai",
      "app": "Formatter by Zapier",
      "action": "Utilities - Line Itemizer",
      "config": {
        "input": "{{step1.results}}",
        "output_format": "json_array",
        "fields": ["id", "Raw Text", "Account Name", "ARR", "Source"]
      }
    },
    {
      "step": 3,
      "name": "analyze_with_chatgpt",
      "app": "ChatGPT",
      "action": "Conversation",
      "config": {
        "model": "gpt-4-turbo",
        "system_message": "You are a product feedback analyst. Respond only in valid JSON.",
        "user_message": "Analyze this feedback batch and return theme, sentiment (-1 to 1), type, priority (1-10), and one-line summary for each: {{step2.output}}",
        "temperature": 0.3
      }
    },
    {
      "step": 4,
      "name": "parse_ai_response",
      "app": "Code by Zapier",
      "action": "Run Javascript",
      "config": {
        "code": "const response = JSON.parse(inputData.aiResponse); return { items: response.analyzed_items };"
      }
    },
    {
      "step": 5,
      "name": "update_notion_items",
      "app": "Notion",
      "action": "Update Database Item",
      "config": {
        "loop": "{{step4.items}}",
        "page_id": "{{loop.feedback_id}}",
        "properties": {
          "AI Theme": "{{loop.theme}}",
          "AI Sentiment": "{{loop.sentiment_score > 0.3 ? 'Positive' : loop.sentiment_score < -0.3 ? 'Negative' : 'Neutral'}}",
          "AI Priority Score": "{{loop.priority_score}}",
          "AI Summary": "{{loop.core_ask}}",
          "Status": "Analyzed"
        }
      }
    },
    {
      "step": 6,
      "name": "log_completion",
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#product-ops",
        "message": ":white_check_mark: Analyzed {{step1.results.length}} feedback items. Top theme: {{step4.items[0].theme}}"
      }
    }
  ]
}`,
              },
              {
                language: 'json',
                title: 'Weekly Summary Delivery Workflow',
                description:
                  'Zapier workflow that generates and delivers the weekly feedback summary to stakeholders via Slack and email.',
                code: `{
  "workflow_name": "Weekly Feedback Summary Delivery",
  "trigger": {
    "type": "schedule",
    "schedule": "every_monday_8am",
    "timezone": "America/New_York"
  },
  "steps": [
    {
      "step": 1,
      "name": "fetch_weekly_feedback",
      "app": "Notion",
      "action": "Find Database Items",
      "config": {
        "database_id": "{{FEEDBACK_DATABASE_ID}}",
        "filter": {
          "and": [
            { "property": "Status", "select": { "equals": "Analyzed" } },
            { "property": "Submitted Date", "date": { "past_week": {} } }
          ]
        },
        "sorts": [{ "property": "AI Priority Score", "direction": "descending" }]
      }
    },
    {
      "step": 2,
      "name": "aggregate_by_theme",
      "app": "Code by Zapier",
      "action": "Run Javascript",
      "config": {
        "code": "const items = inputData.feedbackItems; const themes = {}; items.forEach(i => { const t = i.aiTheme; if (!themes[t]) themes[t] = {count: 0, totalARR: 0, accounts: new Set()}; themes[t].count++; themes[t].totalARR += i.arr || 0; themes[t].accounts.add(i.accountName); }); return Object.entries(themes).map(([theme, data]) => ({theme, count: data.count, totalARR: data.totalARR, uniqueAccounts: data.accounts.size})).sort((a,b) => b.totalARR - a.totalARR);"
      }
    },
    {
      "step": 3,
      "name": "generate_summary",
      "app": "ChatGPT",
      "action": "Conversation",
      "config": {
        "model": "gpt-4-turbo",
        "user_message": "Create a concise weekly product feedback summary for executives. Include: 1) Top 3 themes by ARR, 2) Urgent items, 3) Quick wins. Data: {{step2.output}}"
      }
    },
    {
      "step": 4,
      "name": "post_to_slack",
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#product-team",
        "message": ":chart_with_upwards_trend: *Weekly Feedback Summary*\\n\\n{{step3.response}}\\n\\n_Based on {{step1.results.length}} feedback items from the past week_",
        "unfurl_links": false
      }
    },
    {
      "step": 5,
      "name": "email_stakeholders",
      "app": "Email by Zapier",
      "action": "Send Outbound Email",
      "config": {
        "to": "product-leadership@company.com",
        "subject": "Weekly Product Feedback Summary - {{current_date}}",
        "body": "{{step3.response}}",
        "reply_to": "product@company.com"
      }
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
          'Deploy a multi-agent system using CrewAI and LangGraph that continuously ingests feedback from all channels, runs sophisticated NLP analysis for theme clustering and sentiment detection, maintains a real-time prioritization engine, and proactively alerts the product team to emerging patterns and churn risks.',
        estimatedMonthlyCost: '$500 - $1,500/month',
        architecture:
          'Supervisor agent orchestrates specialized agents for ingestion, NLP analysis, prioritization, and alerting. Agents communicate through a shared state graph with Redis-backed persistence and daily checkpointing for trend analysis.',
        agents: [
          {
            name: 'FeedbackIngestionAgent',
            role: 'Data Collector',
            goal: 'Continuously poll and normalize feedback from Intercom, Zendesk, Slack, email, and CRM systems into a unified format',
            tools: ['IntercomAPI', 'ZendeskAPI', 'SlackAPI', 'GmailAPI', 'SalesforceAPI', 'PostgresWriter'],
          },
          {
            name: 'NLPAnalysisAgent',
            role: 'Theme Extractor',
            goal: 'Generate embeddings, cluster feedback into themes, score sentiment, and detect emerging patterns using transformer models',
            tools: ['SentenceTransformers', 'HDBSCANClustering', 'SentimentModel', 'TopicModeling'],
          },
          {
            name: 'PrioritizationAgent',
            role: 'Product Strategist',
            goal: 'Score and rank feedback by ARR-weighted impact, urgency, and strategic alignment with product roadmap',
            tools: ['ARRLookup', 'RoadmapMatcher', 'ImpactScorer', 'TrendAnalyzer'],
          },
          {
            name: 'AlertingAgent',
            role: 'Signal Detector',
            goal: 'Monitor for churn risk signals, sentiment drops, and volume anomalies, and proactively notify the product team',
            tools: ['AnomalyDetector', 'ChurnPredictor', 'SlackNotifier', 'PagerDutyIntegration'],
          },
          {
            name: 'ReportingAgent',
            role: 'Insight Synthesizer',
            goal: 'Generate executive summaries, trend reports, and actionable recommendations for product leadership',
            tools: ['ReportGenerator', 'VisualizationEngine', 'NotionWriter', 'EmailSender'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with hourly snapshots and 90-day retention for trend analysis',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system architecture using CrewAI with specialized agents for feedback ingestion, NLP analysis, prioritization, alerting, and reporting.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'CrewAI Feedback Analysis Agent Definitions',
                description:
                  'Define the specialized agents for the feedback aggregation system using CrewAI, including their roles, goals, and tools.',
                code: `"""
Feedback Aggregation Multi-Agent System
Agent definitions using CrewAI framework
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass, field

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class AgentConfig:
    """Configuration for a feedback analysis agent."""
    name: str
    role: str
    goal: str
    backstory: str
    tools: list[Tool] = field(default_factory=list)
    verbose: bool = True
    allow_delegation: bool = False
    max_iterations: int = 10


class FeedbackAgentFactory:
    """Factory for creating specialized feedback analysis agents."""

    def __init__(self, llm: Optional[ChatOpenAI] = None) -> None:
        self.llm = llm or ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.2,
        )
        logger.info("FeedbackAgentFactory initialized with model: %s", self.llm.model_name)

    def create_ingestion_agent(self, tools: list[Tool]) -> Agent:
        """Create the feedback ingestion agent."""
        config = AgentConfig(
            name="FeedbackIngestionAgent",
            role="Data Collector",
            goal="Continuously poll and normalize customer feedback from all channels into a unified format",
            backstory="""You are an expert data engineer specializing in customer feedback systems.
            You have deep experience with Intercom, Zendesk, Slack, and CRM APIs.
            Your mission is to ensure no piece of customer feedback is ever lost or duplicated.
            You normalize data from disparate sources into a consistent schema for analysis.""",
            tools=tools,
        )
        return self._create_agent(config)

    def create_nlp_agent(self, tools: list[Tool]) -> Agent:
        """Create the NLP analysis agent."""
        config = AgentConfig(
            name="NLPAnalysisAgent",
            role="Theme Extractor & Sentiment Analyst",
            goal="Analyze feedback using NLP to extract themes, score sentiment, and identify emerging patterns",
            backstory="""You are a senior NLP engineer with expertise in customer feedback analysis.
            You use transformer models for embeddings, HDBSCAN for clustering, and fine-tuned
            sentiment models. You excel at finding the signal in noisy customer communications
            and grouping similar requests into actionable themes.""",
            tools=tools,
            allow_delegation=True,
        )
        return self._create_agent(config)

    def create_prioritization_agent(self, tools: list[Tool]) -> Agent:
        """Create the prioritization agent."""
        config = AgentConfig(
            name="PrioritizationAgent",
            role="Product Strategist",
            goal="Rank and prioritize feedback by ARR-weighted impact, urgency, and strategic alignment",
            backstory="""You are a veteran product strategist who has built prioritization frameworks
            at multiple successful SaaS companies. You understand that not all feedback is equal -
            you weight by customer ARR, account health, strategic fit, and engineering feasibility.
            You produce data-driven priority rankings that product teams can act on immediately.""",
            tools=tools,
        )
        return self._create_agent(config)

    def create_alerting_agent(self, tools: list[Tool]) -> Agent:
        """Create the alerting agent."""
        config = AgentConfig(
            name="AlertingAgent",
            role="Signal Detector",
            goal="Monitor for churn risks, sentiment drops, and anomalies, alerting the team proactively",
            backstory="""You are a customer success expert with a keen eye for early warning signals.
            You monitor sentiment trends, feedback volume patterns, and customer health indicators.
            When you detect a high-ARR account showing frustration or a sudden spike in complaints
            about a feature, you immediately alert the right team members.""",
            tools=tools,
        )
        return self._create_agent(config)

    def create_reporting_agent(self, tools: list[Tool]) -> Agent:
        """Create the reporting agent."""
        config = AgentConfig(
            name="ReportingAgent",
            role="Insight Synthesizer",
            goal="Generate executive summaries and actionable reports for product leadership",
            backstory="""You are a product analytics leader who has presented to C-suites and boards.
            You distill complex feedback data into clear, actionable insights. Your reports
            answer three questions: What are customers asking for? Why does it matter? What
            should we do about it? You always include ARR impact and recommended actions.""",
            tools=tools,
            allow_delegation=True,
        )
        return self._create_agent(config)

    def _create_agent(self, config: AgentConfig) -> Agent:
        """Create an agent from configuration."""
        logger.info("Creating agent: %s (%s)", config.name, config.role)
        return Agent(
            role=config.role,
            goal=config.goal,
            backstory=config.backstory,
            tools=config.tools,
            llm=self.llm,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation,
            max_iter=config.max_iterations,
        )


def create_feedback_crew(agent_factory: FeedbackAgentFactory, tool_registry: dict[str, list[Tool]]) -> Crew:
    """Create the complete feedback analysis crew."""

    # Create agents with their specialized tools
    ingestion_agent = agent_factory.create_ingestion_agent(tool_registry.get("ingestion", []))
    nlp_agent = agent_factory.create_nlp_agent(tool_registry.get("nlp", []))
    prioritization_agent = agent_factory.create_prioritization_agent(tool_registry.get("prioritization", []))
    alerting_agent = agent_factory.create_alerting_agent(tool_registry.get("alerting", []))
    reporting_agent = agent_factory.create_reporting_agent(tool_registry.get("reporting", []))

    # Define tasks
    ingest_task = Task(
        description="Poll all feedback sources and ingest new items into the unified repository",
        expected_output="List of newly ingested feedback items with normalized schema",
        agent=ingestion_agent,
    )

    analyze_task = Task(
        description="Analyze new feedback: generate embeddings, cluster into themes, score sentiment",
        expected_output="Analyzed feedback with theme assignments and sentiment scores",
        agent=nlp_agent,
        context=[ingest_task],
    )

    prioritize_task = Task(
        description="Score and rank feedback themes by ARR-weighted impact and strategic alignment",
        expected_output="Prioritized list of themes with ARR impact and recommended actions",
        agent=prioritization_agent,
        context=[analyze_task],
    )

    alert_task = Task(
        description="Check for churn risk signals, sentiment anomalies, and urgent items requiring immediate attention",
        expected_output="List of alerts with severity, affected accounts, and recommended response",
        agent=alerting_agent,
        context=[analyze_task, prioritize_task],
    )

    report_task = Task(
        description="Generate executive summary of feedback insights for product leadership",
        expected_output="Executive report with top themes, ARR impact, risks, and recommendations",
        agent=reporting_agent,
        context=[prioritize_task, alert_task],
    )

    crew = Crew(
        agents=[ingestion_agent, nlp_agent, prioritization_agent, alerting_agent, reporting_agent],
        tasks=[ingest_task, analyze_task, prioritize_task, alert_task, report_task],
        process=Process.sequential,
        verbose=True,
    )

    logger.info("Feedback analysis crew created with %d agents and %d tasks", len(crew.agents), len(crew.tasks))
    return crew`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the feedback ingestion agent with connectors for all feedback sources, real-time polling, and deduplication logic.',
            toolsUsed: ['LangChain', 'Intercom API', 'Zendesk API', 'Slack API'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Multi-Channel Feedback Ingestion Tools',
                description:
                  'LangChain tools for the ingestion agent to poll feedback from Intercom, Zendesk, Slack, and other sources.',
                code: `"""
Feedback Ingestion Tools for CrewAI Agent
Provides connectors to all feedback source APIs
"""

import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from dataclasses import dataclass, field

import requests
from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class NormalizedFeedback:
    """Normalized feedback item from any source."""
    source_id: str
    source_platform: str
    customer_email: str
    account_name: Optional[str]
    account_arr: Optional[float]
    raw_text: str
    submitted_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        """Generate deduplication key based on content hash."""
        content = f"{self.source_platform}:{self.customer_email}:{self.raw_text[:200]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class IntercomIngestionInput(BaseModel):
    """Input schema for Intercom ingestion."""
    since_hours: int = Field(default=24, description="Hours to look back for conversations")
    max_items: int = Field(default=100, description="Maximum items to fetch")


class IntercomIngestionTool:
    """Tool for ingesting feedback from Intercom conversations."""

    def __init__(self, api_token: str, account_lookup: Optional[dict[str, float]] = None) -> None:
        self.api_token = api_token
        self.account_lookup = account_lookup or {}
        self.base_url = "https://api.intercom.io"
        logger.info("IntercomIngestionTool initialized")

    def fetch_conversations(self, since_hours: int = 24, max_items: int = 100) -> list[NormalizedFeedback]:
        """Fetch and normalize conversations from Intercom."""
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
        }

        since_timestamp = int((datetime.now(timezone.utc) - timedelta(hours=since_hours)).timestamp())

        try:
            response = requests.get(
                f"{self.base_url}/conversations",
                headers=headers,
                params={"updated_after": since_timestamp, "per_page": min(max_items, 50)},
                timeout=30,
            )
            response.raise_for_status()
            conversations = response.json().get("conversations", [])
        except requests.RequestException as e:
            logger.error("Failed to fetch Intercom conversations: %s", e)
            return []

        results: list[NormalizedFeedback] = []
        for convo in conversations[:max_items]:
            body = convo.get("source", {}).get("body", "")
            if not body or len(body.strip()) < 10:
                continue

            email = convo.get("source", {}).get("author", {}).get("email", "")
            company = convo.get("source", {}).get("author", {}).get("company", {}).get("name", "")

            results.append(NormalizedFeedback(
                source_id=convo["id"],
                source_platform="intercom",
                customer_email=email,
                account_name=company,
                account_arr=self.account_lookup.get(company),
                raw_text=body,
                submitted_at=datetime.fromtimestamp(convo["created_at"], tz=timezone.utc),
                metadata={"conversation_id": convo["id"], "tags": convo.get("tags", [])},
            ))

        logger.info("Fetched %d conversations from Intercom (since %d hours ago)", len(results), since_hours)
        return results

    def as_langchain_tool(self) -> StructuredTool:
        """Convert to LangChain tool for agent use."""
        return StructuredTool.from_function(
            func=lambda since_hours, max_items: [f.__dict__ for f in self.fetch_conversations(since_hours, max_items)],
            name="intercom_fetch_feedback",
            description="Fetch customer feedback from Intercom conversations",
            args_schema=IntercomIngestionInput,
        )


class ZendeskIngestionTool:
    """Tool for ingesting feedback from Zendesk tickets."""

    def __init__(self, subdomain: str, email: str, api_token: str, account_lookup: Optional[dict[str, float]] = None) -> None:
        self.subdomain = subdomain
        self.auth = (f"{email}/token", api_token)
        self.account_lookup = account_lookup or {}
        logger.info("ZendeskIngestionTool initialized for subdomain: %s", subdomain)

    def fetch_tickets(self, since_hours: int = 24, tags: Optional[list[str]] = None) -> list[NormalizedFeedback]:
        """Fetch and normalize tickets from Zendesk."""
        tags = tags or ["feature-request", "feedback", "suggestion"]

        try:
            response = requests.get(
                f"https://{self.subdomain}.zendesk.com/api/v2/search.json",
                auth=self.auth,
                params={
                    "query": f"type:ticket tags:{' '.join(tags)} created>{since_hours}hours",
                    "sort_by": "created_at",
                    "sort_order": "desc",
                },
                timeout=30,
            )
            response.raise_for_status()
            tickets = response.json().get("results", [])
        except requests.RequestException as e:
            logger.error("Failed to fetch Zendesk tickets: %s", e)
            return []

        results: list[NormalizedFeedback] = []
        for ticket in tickets:
            description = ticket.get("description", "")
            if not description or len(description.strip()) < 10:
                continue

            requester_email = ticket.get("requester", {}).get("email", "")
            org_name = ticket.get("organization", {}).get("name", "")

            results.append(NormalizedFeedback(
                source_id=str(ticket["id"]),
                source_platform="zendesk",
                customer_email=requester_email,
                account_name=org_name,
                account_arr=self.account_lookup.get(org_name),
                raw_text=description,
                submitted_at=datetime.fromisoformat(ticket["created_at"].replace("Z", "+00:00")),
                metadata={"ticket_id": ticket["id"], "subject": ticket.get("subject", ""), "tags": ticket.get("tags", [])},
            ))

        logger.info("Fetched %d tickets from Zendesk (since %d hours ago)", len(results), since_hours)
        return results


class SlackIngestionTool:
    """Tool for ingesting feedback from Slack channels."""

    def __init__(self, bot_token: str, feedback_channels: list[str], account_lookup: Optional[dict[str, float]] = None) -> None:
        self.bot_token = bot_token
        self.feedback_channels = feedback_channels
        self.account_lookup = account_lookup or {}
        logger.info("SlackIngestionTool initialized for %d channels", len(feedback_channels))

    def fetch_messages(self, since_hours: int = 24) -> list[NormalizedFeedback]:
        """Fetch feedback messages from Slack channels."""
        headers = {"Authorization": f"Bearer {self.bot_token}"}
        oldest_timestamp = str((datetime.now(timezone.utc) - timedelta(hours=since_hours)).timestamp())

        results: list[NormalizedFeedback] = []

        for channel_id in self.feedback_channels:
            try:
                response = requests.get(
                    "https://slack.com/api/conversations.history",
                    headers=headers,
                    params={"channel": channel_id, "oldest": oldest_timestamp, "limit": 100},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                if not data.get("ok"):
                    logger.warning("Slack API error for channel %s: %s", channel_id, data.get("error"))
                    continue

                messages = data.get("messages", [])
                for msg in messages:
                    text = msg.get("text", "")
                    if not text or len(text.strip()) < 10 or msg.get("subtype"):
                        continue

                    results.append(NormalizedFeedback(
                        source_id=msg["ts"],
                        source_platform="slack",
                        customer_email=msg.get("user", "unknown"),
                        account_name=None,
                        account_arr=None,
                        raw_text=text,
                        submitted_at=datetime.fromtimestamp(float(msg["ts"]), tz=timezone.utc),
                        metadata={"channel": channel_id, "thread_ts": msg.get("thread_ts")},
                    ))

            except requests.RequestException as e:
                logger.error("Failed to fetch Slack messages from %s: %s", channel_id, e)

        logger.info("Fetched %d messages from Slack (since %d hours ago)", len(results), since_hours)
        return results


def create_ingestion_tools(config: dict[str, Any]) -> list[Tool]:
    """Create all ingestion tools from configuration."""
    tools: list[Tool] = []
    account_lookup = config.get("account_arr_lookup", {})

    if config.get("intercom_token"):
        intercom_tool = IntercomIngestionTool(config["intercom_token"], account_lookup)
        tools.append(intercom_tool.as_langchain_tool())

    if config.get("zendesk_subdomain"):
        zendesk_tool = ZendeskIngestionTool(
            config["zendesk_subdomain"],
            config["zendesk_email"],
            config["zendesk_token"],
            account_lookup,
        )
        tools.append(StructuredTool.from_function(
            func=lambda since_hours=24: [f.__dict__ for f in zendesk_tool.fetch_tickets(since_hours)],
            name="zendesk_fetch_feedback",
            description="Fetch customer feedback from Zendesk tickets tagged with feature-request or feedback",
        ))

    if config.get("slack_bot_token"):
        slack_tool = SlackIngestionTool(
            config["slack_bot_token"],
            config.get("slack_feedback_channels", []),
            account_lookup,
        )
        tools.append(StructuredTool.from_function(
            func=lambda since_hours=24: [f.__dict__ for f in slack_tool.fetch_messages(since_hours)],
            name="slack_fetch_feedback",
            description="Fetch customer feedback messages from designated Slack channels",
        ))

    logger.info("Created %d ingestion tools", len(tools))
    return tools`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the NLP analysis and prioritization agents with embedding generation, clustering, sentiment scoring, and ARR-weighted ranking.',
            toolsUsed: ['LangChain', 'SentenceTransformers', 'HDBSCAN', 'Transformers'],
            codeSnippets: [
              {
                language: 'python',
                title: 'NLP Analysis and Prioritization Tools',
                description:
                  'Tools for the NLP and prioritization agents including embedding generation, theme clustering, sentiment analysis, and ARR-weighted scoring.',
                code: `"""
NLP Analysis and Prioritization Tools for CrewAI Agents
Provides theme extraction, sentiment scoring, and ARR-weighted prioritization
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline as hf_pipeline
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class AnalyzedFeedback:
    """Feedback item with NLP analysis results."""
    feedback_id: str
    raw_text: str
    cluster_id: int
    cluster_label: str
    sentiment_score: float
    feedback_type: str
    embedding: Optional[list[float]] = None


@dataclass
class ThemeCluster:
    """A cluster of related feedback items."""
    cluster_id: int
    label: str
    feedback_count: int
    avg_sentiment: float
    total_arr: float
    unique_accounts: int
    sample_feedback: list[str]
    priority_score: float = 0.0


class EmbeddingGeneratorInput(BaseModel):
    """Input schema for embedding generation."""
    texts: list[str] = Field(description="List of texts to embed")


class EmbeddingGenerator:
    """Generate sentence embeddings for feedback clustering."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info("EmbeddingGenerator initialized with model: %s (dim=%d)", model_name, self.embedding_dim)

    def generate(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        logger.info("Generated embeddings for %d texts", len(texts))
        return embeddings

    def as_langchain_tool(self) -> StructuredTool:
        """Convert to LangChain tool."""
        return StructuredTool.from_function(
            func=lambda texts: self.generate(texts).tolist(),
            name="generate_embeddings",
            description="Generate sentence embeddings for feedback texts for clustering",
            args_schema=EmbeddingGeneratorInput,
        )


class ThemeClusteringInput(BaseModel):
    """Input schema for theme clustering."""
    embeddings: list[list[float]] = Field(description="List of embedding vectors")
    texts: list[str] = Field(description="Corresponding texts for label generation")
    min_cluster_size: int = Field(default=5, description="Minimum cluster size for HDBSCAN")


class ThemeClusterer:
    """Cluster feedback into themes using HDBSCAN."""

    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        logger.info("ThemeClusterer initialized: min_cluster_size=%d", min_cluster_size)

    def cluster_and_label(
        self,
        embeddings: np.ndarray,
        texts: list[str],
    ) -> tuple[np.ndarray, dict[int, str]]:
        """Cluster embeddings and generate human-readable labels for each cluster."""

        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)

        # Group texts by cluster
        cluster_texts: dict[int, list[str]] = {}
        for text, label in zip(texts, labels):
            if label >= 0:  # Ignore noise points (-1)
                cluster_texts.setdefault(label, []).append(text)

        # Generate labels using TF-IDF top terms
        cluster_labels: dict[int, str] = {}
        for cluster_id, cluster_text_list in cluster_texts.items():
            try:
                tfidf = TfidfVectorizer(max_features=5, stop_words="english", ngram_range=(1, 2))
                tfidf.fit(cluster_text_list)
                top_terms = tfidf.get_feature_names_out()[:3]
                cluster_labels[cluster_id] = " / ".join(term.title() for term in top_terms)
            except ValueError:
                cluster_labels[cluster_id] = f"Theme {cluster_id}"

        logger.info("Clustered %d items into %d themes (noise: %d)",
                   len(texts), len(cluster_labels), np.sum(labels == -1))
        return labels, cluster_labels

    def as_langchain_tool(self) -> StructuredTool:
        """Convert to LangChain tool."""
        def _cluster(embeddings: list[list[float]], texts: list[str], min_cluster_size: int = 5) -> dict[str, Any]:
            self.min_cluster_size = min_cluster_size
            emb_array = np.array(embeddings)
            labels, cluster_labels = self.cluster_and_label(emb_array, texts)
            return {
                "labels": labels.tolist(),
                "cluster_labels": cluster_labels,
                "num_clusters": len(cluster_labels),
                "noise_count": int(np.sum(labels == -1)),
            }

        return StructuredTool.from_function(
            func=_cluster,
            name="cluster_feedback_themes",
            description="Cluster feedback embeddings into themes and generate human-readable labels",
            args_schema=ThemeClusteringInput,
        )


class SentimentScorerInput(BaseModel):
    """Input schema for sentiment scoring."""
    texts: list[str] = Field(description="List of texts to score for sentiment")


class SentimentScorer:
    """Score sentiment of feedback using transformer model."""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> None:
        self.model = hf_pipeline(
            "sentiment-analysis",
            model=model_name,
            truncation=True,
            max_length=512,
        )
        logger.info("SentimentScorer initialized with model: %s", model_name)

    def score(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """Score sentiment for a list of texts. Returns scores from -1.0 to 1.0."""
        if not texts:
            return []

        # Truncate long texts
        truncated = [t[:512] for t in texts]
        predictions = self.model(truncated, batch_size=batch_size)

        scores: list[float] = []
        for pred in predictions:
            if pred["label"] == "POSITIVE":
                scores.append(pred["score"])
            else:
                scores.append(-pred["score"])

        logger.info("Scored sentiment for %d texts (avg: %.3f)", len(texts), np.mean(scores))
        return scores

    def as_langchain_tool(self) -> StructuredTool:
        """Convert to LangChain tool."""
        return StructuredTool.from_function(
            func=lambda texts: self.score(texts),
            name="score_sentiment",
            description="Score sentiment of feedback texts, returning values from -1.0 (negative) to 1.0 (positive)",
            args_schema=SentimentScorerInput,
        )


class ARRPrioritizationInput(BaseModel):
    """Input schema for ARR-weighted prioritization."""
    themes: list[dict[str, Any]] = Field(description="List of theme clusters with feedback data")
    account_arr: dict[str, float] = Field(description="Mapping of account names to ARR values")


class ARRPrioritizer:
    """Prioritize feedback themes by ARR-weighted impact."""

    def __init__(
        self,
        arr_weight: float = 0.4,
        volume_weight: float = 0.25,
        momentum_weight: float = 0.20,
        urgency_weight: float = 0.15,
    ) -> None:
        self.arr_weight = arr_weight
        self.volume_weight = volume_weight
        self.momentum_weight = momentum_weight
        self.urgency_weight = urgency_weight
        logger.info("ARRPrioritizer initialized: arr=%.2f, vol=%.2f, mom=%.2f, urg=%.2f",
                   arr_weight, volume_weight, momentum_weight, urgency_weight)

    def prioritize(
        self,
        themes: list[ThemeCluster],
        recent_window_days: int = 30,
    ) -> list[ThemeCluster]:
        """Calculate priority scores and rank themes."""
        if not themes:
            return []

        # Normalize metrics for scoring
        max_arr = max(t.total_arr for t in themes) or 1.0
        max_count = max(t.feedback_count for t in themes) or 1
        max_accounts = max(t.unique_accounts for t in themes) or 1

        for theme in themes:
            # ARR impact score (0-1)
            arr_score = theme.total_arr / max_arr

            # Volume score (0-1)
            volume_score = theme.feedback_count / max_count

            # Momentum score - using unique accounts as proxy (0-1)
            momentum_score = theme.unique_accounts / max_accounts

            # Urgency score - negative sentiment = higher urgency (0-1)
            urgency_score = max(0, (1 - theme.avg_sentiment) / 2)

            # Composite priority score
            theme.priority_score = round(
                (arr_score * self.arr_weight +
                 volume_score * self.volume_weight +
                 momentum_score * self.momentum_weight +
                 urgency_score * self.urgency_weight) * 100,
                2
            )

        # Sort by priority score descending
        ranked = sorted(themes, key=lambda t: t.priority_score, reverse=True)
        logger.info("Prioritized %d themes. Top: %s (score: %.2f)",
                   len(ranked), ranked[0].label if ranked else "N/A",
                   ranked[0].priority_score if ranked else 0)
        return ranked

    def as_langchain_tool(self) -> StructuredTool:
        """Convert to LangChain tool."""
        def _prioritize(themes: list[dict[str, Any]], account_arr: dict[str, float]) -> list[dict[str, Any]]:
            theme_clusters = [
                ThemeCluster(
                    cluster_id=t["cluster_id"],
                    label=t["label"],
                    feedback_count=t["feedback_count"],
                    avg_sentiment=t["avg_sentiment"],
                    total_arr=t.get("total_arr", 0),
                    unique_accounts=t["unique_accounts"],
                    sample_feedback=t.get("sample_feedback", []),
                )
                for t in themes
            ]
            ranked = self.prioritize(theme_clusters)
            return [
                {
                    "cluster_id": t.cluster_id,
                    "label": t.label,
                    "priority_score": t.priority_score,
                    "feedback_count": t.feedback_count,
                    "total_arr": t.total_arr,
                    "unique_accounts": t.unique_accounts,
                    "avg_sentiment": t.avg_sentiment,
                }
                for t in ranked
            ]

        return StructuredTool.from_function(
            func=_prioritize,
            name="prioritize_themes_by_arr",
            description="Prioritize feedback themes by ARR-weighted impact, volume, momentum, and urgency",
            args_schema=ARRPrioritizationInput,
        )


def create_nlp_tools() -> list[StructuredTool]:
    """Create NLP analysis tools for the agent."""
    embedding_gen = EmbeddingGenerator()
    theme_clusterer = ThemeClusterer()
    sentiment_scorer = SentimentScorer()

    return [
        embedding_gen.as_langchain_tool(),
        theme_clusterer.as_langchain_tool(),
        sentiment_scorer.as_langchain_tool(),
    ]


def create_prioritization_tools() -> list[StructuredTool]:
    """Create prioritization tools for the agent."""
    prioritizer = ARRPrioritizer()
    return [prioritizer.as_langchain_tool()]`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement the LangGraph state machine that orchestrates all agents, manages shared state, handles failures gracefully, and enables continuous feedback processing.',
            toolsUsed: ['LangGraph', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Feedback Analysis Orchestrator',
                description:
                  'State machine implementation using LangGraph to orchestrate the feedback analysis workflow with proper error handling and state persistence.',
                code: `"""
LangGraph Orchestrator for Feedback Analysis Pipeline
Manages agent coordination, state transitions, and error handling
"""

import logging
from typing import Any, TypedDict, Annotated, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json

from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import redis

logger = logging.getLogger(__name__)


class FeedbackPipelineState(TypedDict):
    """State schema for the feedback analysis pipeline."""
    # Pipeline metadata
    run_id: str
    started_at: str
    current_stage: str
    error_count: int

    # Ingestion state
    raw_feedback: list[dict[str, Any]]
    ingestion_complete: bool

    # Analysis state
    embeddings: list[list[float]]
    cluster_assignments: list[int]
    cluster_labels: dict[str, str]
    sentiment_scores: list[float]
    analysis_complete: bool

    # Prioritization state
    prioritized_themes: list[dict[str, Any]]
    prioritization_complete: bool

    # Alerting state
    alerts: list[dict[str, Any]]
    alerting_complete: bool

    # Reporting state
    executive_summary: str
    report_complete: bool

    # Messages for agent communication
    messages: Sequence[BaseMessage]


@dataclass
class RedisStateManager:
    """Manages pipeline state persistence in Redis."""

    client: redis.Redis
    key_prefix: str = "feedback_pipeline"
    ttl_days: int = 90

    def save_state(self, run_id: str, state: FeedbackPipelineState) -> None:
        """Persist pipeline state to Redis."""
        key = f"{self.key_prefix}:{run_id}"
        # Convert non-serializable types
        serializable_state = {
            k: v for k, v in state.items()
            if k != "messages"
        }
        self.client.setex(
            key,
            self.ttl_days * 86400,
            json.dumps(serializable_state, default=str),
        )
        logger.debug("Saved state for run %s", run_id)

    def load_state(self, run_id: str) -> dict[str, Any] | None:
        """Load pipeline state from Redis."""
        key = f"{self.key_prefix}:{run_id}"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    def save_checkpoint(self, run_id: str, stage: str, data: dict[str, Any]) -> None:
        """Save a stage checkpoint for debugging and recovery."""
        key = f"{self.key_prefix}:checkpoint:{run_id}:{stage}"
        self.client.setex(key, self.ttl_days * 86400, json.dumps(data, default=str))
        logger.info("Checkpoint saved: %s/%s", run_id, stage)


class FeedbackPipelineOrchestrator:
    """Orchestrates the feedback analysis pipeline using LangGraph."""

    def __init__(
        self,
        ingestion_agent: Any,
        nlp_agent: Any,
        prioritization_agent: Any,
        alerting_agent: Any,
        reporting_agent: Any,
        redis_client: redis.Redis | None = None,
    ) -> None:
        self.ingestion_agent = ingestion_agent
        self.nlp_agent = nlp_agent
        self.prioritization_agent = prioritization_agent
        self.alerting_agent = alerting_agent
        self.reporting_agent = reporting_agent

        self.state_manager = RedisStateManager(redis_client) if redis_client else None
        self.graph = self._build_graph()
        logger.info("FeedbackPipelineOrchestrator initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for pipeline orchestration."""

        # Create the state graph
        workflow = StateGraph(FeedbackPipelineState)

        # Add nodes for each pipeline stage
        workflow.add_node("ingest", self._ingest_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("prioritize", self._prioritize_node)
        workflow.add_node("alert", self._alert_node)
        workflow.add_node("report", self._report_node)
        workflow.add_node("handle_error", self._error_handler_node)

        # Define edges (transitions)
        workflow.set_entry_point("ingest")

        workflow.add_conditional_edges(
            "ingest",
            self._should_continue_after_ingest,
            {
                "continue": "analyze",
                "error": "handle_error",
                "end": END,
            }
        )

        workflow.add_conditional_edges(
            "analyze",
            self._should_continue_after_analyze,
            {
                "continue": "prioritize",
                "error": "handle_error",
            }
        )

        workflow.add_edge("prioritize", "alert")
        workflow.add_edge("alert", "report")
        workflow.add_edge("report", END)
        workflow.add_edge("handle_error", END)

        # Compile with checkpointing
        memory = MemorySaver()
        compiled = workflow.compile(checkpointer=memory)

        logger.info("Pipeline graph built with %d nodes", len(workflow.nodes))
        return compiled

    def _ingest_node(self, state: FeedbackPipelineState) -> dict[str, Any]:
        """Execute the ingestion stage."""
        logger.info("Executing ingestion stage for run %s", state["run_id"])

        try:
            # Run ingestion agent
            result = self.ingestion_agent.execute(
                "Fetch new feedback from all configured sources (Intercom, Zendesk, Slack)"
            )

            raw_feedback = result.get("feedback_items", [])
            logger.info("Ingested %d feedback items", len(raw_feedback))

            return {
                "raw_feedback": raw_feedback,
                "ingestion_complete": True,
                "current_stage": "analyze",
                "messages": state["messages"] + [AIMessage(content=f"Ingested {len(raw_feedback)} items")],
            }

        except Exception as e:
            logger.error("Ingestion failed: %s", e)
            return {
                "error_count": state["error_count"] + 1,
                "current_stage": "error",
                "messages": state["messages"] + [AIMessage(content=f"Ingestion error: {e}")],
            }

    def _analyze_node(self, state: FeedbackPipelineState) -> dict[str, Any]:
        """Execute the NLP analysis stage."""
        logger.info("Executing analysis stage for run %s", state["run_id"])

        try:
            texts = [f["raw_text"] for f in state["raw_feedback"]]

            # Generate embeddings
            embeddings_result = self.nlp_agent.execute(
                f"Generate embeddings for {len(texts)} feedback texts",
                context={"texts": texts},
            )

            # Cluster into themes
            clustering_result = self.nlp_agent.execute(
                "Cluster the feedback embeddings into themes",
                context={
                    "embeddings": embeddings_result["embeddings"],
                    "texts": texts,
                },
            )

            # Score sentiment
            sentiment_result = self.nlp_agent.execute(
                "Score sentiment for all feedback texts",
                context={"texts": texts},
            )

            logger.info("Analysis complete: %d clusters, avg sentiment: %.3f",
                       clustering_result["num_clusters"],
                       sum(sentiment_result["scores"]) / len(sentiment_result["scores"]) if sentiment_result["scores"] else 0)

            return {
                "embeddings": embeddings_result["embeddings"],
                "cluster_assignments": clustering_result["labels"],
                "cluster_labels": clustering_result["cluster_labels"],
                "sentiment_scores": sentiment_result["scores"],
                "analysis_complete": True,
                "current_stage": "prioritize",
            }

        except Exception as e:
            logger.error("Analysis failed: %s", e)
            return {
                "error_count": state["error_count"] + 1,
                "current_stage": "error",
            }

    def _prioritize_node(self, state: FeedbackPipelineState) -> dict[str, Any]:
        """Execute the prioritization stage."""
        logger.info("Executing prioritization stage for run %s", state["run_id"])

        try:
            # Build theme summaries
            themes = self._aggregate_themes(state)

            # Run prioritization
            priority_result = self.prioritization_agent.execute(
                "Prioritize themes by ARR-weighted impact",
                context={"themes": themes},
            )

            logger.info("Prioritization complete: top theme is '%s'",
                       priority_result["ranked_themes"][0]["label"] if priority_result["ranked_themes"] else "N/A")

            return {
                "prioritized_themes": priority_result["ranked_themes"],
                "prioritization_complete": True,
                "current_stage": "alert",
            }

        except Exception as e:
            logger.error("Prioritization failed: %s", e)
            return {"error_count": state["error_count"] + 1, "current_stage": "error"}

    def _alert_node(self, state: FeedbackPipelineState) -> dict[str, Any]:
        """Execute the alerting stage."""
        logger.info("Executing alerting stage for run %s", state["run_id"])

        try:
            alert_result = self.alerting_agent.execute(
                "Check for churn risk signals and urgent items",
                context={
                    "feedback": state["raw_feedback"],
                    "sentiment_scores": state["sentiment_scores"],
                    "themes": state["prioritized_themes"],
                },
            )

            alerts = alert_result.get("alerts", [])
            logger.info("Alerting complete: %d alerts generated", len(alerts))

            return {
                "alerts": alerts,
                "alerting_complete": True,
                "current_stage": "report",
            }

        except Exception as e:
            logger.error("Alerting failed: %s", e)
            return {"alerts": [], "alerting_complete": True, "current_stage": "report"}

    def _report_node(self, state: FeedbackPipelineState) -> dict[str, Any]:
        """Execute the reporting stage."""
        logger.info("Executing reporting stage for run %s", state["run_id"])

        try:
            report_result = self.reporting_agent.execute(
                "Generate executive summary of feedback insights",
                context={
                    "themes": state["prioritized_themes"],
                    "alerts": state["alerts"],
                    "total_feedback": len(state["raw_feedback"]),
                },
            )

            logger.info("Reporting complete: summary generated")

            return {
                "executive_summary": report_result["summary"],
                "report_complete": True,
                "current_stage": "complete",
            }

        except Exception as e:
            logger.error("Reporting failed: %s", e)
            return {"executive_summary": f"Report generation failed: {e}", "report_complete": True}

    def _error_handler_node(self, state: FeedbackPipelineState) -> dict[str, Any]:
        """Handle pipeline errors."""
        logger.error("Pipeline error handler invoked. Error count: %d", state["error_count"])
        return {"current_stage": "failed"}

    def _should_continue_after_ingest(self, state: FeedbackPipelineState) -> str:
        """Determine next step after ingestion."""
        if state.get("error_count", 0) > 0:
            return "error"
        if not state.get("raw_feedback"):
            logger.info("No new feedback to process")
            return "end"
        return "continue"

    def _should_continue_after_analyze(self, state: FeedbackPipelineState) -> str:
        """Determine next step after analysis."""
        if state.get("error_count", 0) > 0:
            return "error"
        return "continue"

    def _aggregate_themes(self, state: FeedbackPipelineState) -> list[dict[str, Any]]:
        """Aggregate feedback into theme summaries."""
        themes: dict[int, dict[str, Any]] = {}

        for i, (feedback, cluster_id, sentiment) in enumerate(zip(
            state["raw_feedback"],
            state["cluster_assignments"],
            state["sentiment_scores"],
        )):
            if cluster_id < 0:  # Skip noise
                continue

            if cluster_id not in themes:
                themes[cluster_id] = {
                    "cluster_id": cluster_id,
                    "label": state["cluster_labels"].get(str(cluster_id), f"Theme {cluster_id}"),
                    "feedback_count": 0,
                    "sentiments": [],
                    "total_arr": 0.0,
                    "accounts": set(),
                    "sample_feedback": [],
                }

            themes[cluster_id]["feedback_count"] += 1
            themes[cluster_id]["sentiments"].append(sentiment)
            themes[cluster_id]["total_arr"] += feedback.get("account_arr", 0) or 0
            if feedback.get("account_name"):
                themes[cluster_id]["accounts"].add(feedback["account_name"])
            if len(themes[cluster_id]["sample_feedback"]) < 3:
                themes[cluster_id]["sample_feedback"].append(feedback["raw_text"][:200])

        # Finalize aggregations
        result = []
        for theme in themes.values():
            theme["avg_sentiment"] = sum(theme["sentiments"]) / len(theme["sentiments"]) if theme["sentiments"] else 0
            theme["unique_accounts"] = len(theme["accounts"])
            del theme["sentiments"]
            del theme["accounts"]
            result.append(theme)

        return result

    def run(self, run_id: str | None = None) -> FeedbackPipelineState:
        """Execute the complete feedback analysis pipeline."""
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        initial_state: FeedbackPipelineState = {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "current_stage": "ingest",
            "error_count": 0,
            "raw_feedback": [],
            "ingestion_complete": False,
            "embeddings": [],
            "cluster_assignments": [],
            "cluster_labels": {},
            "sentiment_scores": [],
            "analysis_complete": False,
            "prioritized_themes": [],
            "prioritization_complete": False,
            "alerts": [],
            "alerting_complete": False,
            "executive_summary": "",
            "report_complete": False,
            "messages": [HumanMessage(content=f"Starting feedback analysis pipeline run: {run_id}")],
        }

        logger.info("Starting pipeline run: %s", run_id)

        # Execute the graph
        final_state = self.graph.invoke(initial_state, {"configurable": {"thread_id": run_id}})

        # Persist final state
        if self.state_manager:
            self.state_manager.save_state(run_id, final_state)

        logger.info("Pipeline run %s completed. Stage: %s", run_id, final_state["current_stage"])
        return final_state`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the multi-agent system with Docker, implement comprehensive observability using LangSmith for agent tracing, and set up Prometheus metrics and Slack alerts for production monitoring.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'Grafana', 'Slack'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose Production Configuration',
                description:
                  'Docker Compose configuration for deploying the feedback analysis multi-agent system with Redis, Prometheus, and observability tooling.',
                code: `# docker-compose.production.yml
# Feedback Analysis Multi-Agent System - Production Deployment

version: "3.9"

services:
  # Main feedback analysis worker
  feedback-agent-worker:
    build:
      context: .
      dockerfile: Dockerfile.feedback-agents
    image: ghcr.io/myorg/feedback-agents:latest
    container_name: feedback-agent-worker
    restart: unless-stopped
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGCHAIN_API_KEY=\${LANGCHAIN_API_KEY}
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_PROJECT=feedback-analysis-prod
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=\${DATABASE_URL}
      - INTERCOM_TOKEN=\${INTERCOM_TOKEN}
      - ZENDESK_SUBDOMAIN=\${ZENDESK_SUBDOMAIN}
      - ZENDESK_EMAIL=\${ZENDESK_EMAIL}
      - ZENDESK_TOKEN=\${ZENDESK_TOKEN}
      - SLACK_BOT_TOKEN=\${SLACK_BOT_TOKEN}
      - SLACK_WEBHOOK_URL=\${SLACK_WEBHOOK_URL}
      - LOG_LEVEL=INFO
      - PIPELINE_SCHEDULE_HOURS=6
    volumes:
      - ./config:/app/config:ro
      - feedback-data:/app/data
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8080/health').raise_for_status()"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "1.0"
          memory: 2G
    networks:
      - feedback-network

  # Redis for state management and caching
  redis:
    image: redis:7-alpine
    container_name: feedback-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - feedback-network

  # Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: feedback-prometheus
    restart: unless-stopped
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--storage.tsdb.retention.time=30d"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - feedback-network

  # Grafana for dashboards
  grafana:
    image: grafana/grafana:10.0.0
    container_name: feedback-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_ADMIN_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    networks:
      - feedback-network

volumes:
  feedback-data:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  feedback-network:
    driver: bridge`,
              },
              {
                language: 'python',
                title: 'Observability and Metrics Module',
                description:
                  'Prometheus metrics, LangSmith tracing integration, and Slack alerting for production monitoring of the multi-agent system.',
                code: `"""
Observability Module for Feedback Analysis Pipeline
Provides Prometheus metrics, LangSmith tracing, and Slack alerting
"""

import logging
import time
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
import json

from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
import requests

logger = logging.getLogger(__name__)


# ─── Prometheus Metrics ───────────────────────────────────────────────────────

# Pipeline execution metrics
PIPELINE_RUNS_TOTAL = Counter(
    "feedback_pipeline_runs_total",
    "Total number of pipeline runs",
    ["status"],  # success, failure, partial
)

PIPELINE_DURATION_SECONDS = Histogram(
    "feedback_pipeline_duration_seconds",
    "Duration of complete pipeline runs",
    buckets=[30, 60, 120, 300, 600, 1200, 3600],
)

STAGE_DURATION_SECONDS = Histogram(
    "feedback_pipeline_stage_duration_seconds",
    "Duration of individual pipeline stages",
    ["stage"],  # ingest, analyze, prioritize, alert, report
    buckets=[5, 10, 30, 60, 120, 300],
)

# Feedback processing metrics
FEEDBACK_INGESTED_TOTAL = Counter(
    "feedback_items_ingested_total",
    "Total feedback items ingested",
    ["source"],  # intercom, zendesk, slack
)

FEEDBACK_ANALYZED_TOTAL = Counter(
    "feedback_items_analyzed_total",
    "Total feedback items analyzed",
)

THEMES_DETECTED = Gauge(
    "feedback_themes_detected",
    "Number of distinct themes in the current analysis",
)

AVG_SENTIMENT_SCORE = Gauge(
    "feedback_avg_sentiment_score",
    "Average sentiment score of recent feedback",
)

# Agent metrics
AGENT_INVOCATIONS_TOTAL = Counter(
    "feedback_agent_invocations_total",
    "Total agent invocations",
    ["agent_name"],
)

AGENT_ERRORS_TOTAL = Counter(
    "feedback_agent_errors_total",
    "Total agent errors",
    ["agent_name", "error_type"],
)

LLM_TOKENS_USED = Counter(
    "feedback_llm_tokens_used_total",
    "Total LLM tokens consumed",
    ["model", "token_type"],  # token_type: prompt, completion
)

# Alert metrics
ALERTS_GENERATED_TOTAL = Counter(
    "feedback_alerts_generated_total",
    "Total alerts generated",
    ["severity", "alert_type"],
)


@dataclass
class MetricsCollector:
    """Collects and exposes pipeline metrics."""

    port: int = 8080
    _started: bool = False

    def start_server(self) -> None:
        """Start the Prometheus metrics HTTP server."""
        if not self._started:
            start_http_server(self.port)
            self._started = True
            logger.info("Prometheus metrics server started on port %d", self.port)

    def record_pipeline_run(self, status: str, duration_seconds: float) -> None:
        """Record a complete pipeline run."""
        PIPELINE_RUNS_TOTAL.labels(status=status).inc()
        PIPELINE_DURATION_SECONDS.observe(duration_seconds)
        logger.info("Pipeline run recorded: status=%s, duration=%.2fs", status, duration_seconds)

    def record_stage_duration(self, stage: str, duration_seconds: float) -> None:
        """Record duration of a pipeline stage."""
        STAGE_DURATION_SECONDS.labels(stage=stage).observe(duration_seconds)

    def record_ingestion(self, source: str, count: int) -> None:
        """Record feedback ingestion from a source."""
        FEEDBACK_INGESTED_TOTAL.labels(source=source).inc(count)

    def record_analysis(self, count: int, num_themes: int, avg_sentiment: float) -> None:
        """Record analysis results."""
        FEEDBACK_ANALYZED_TOTAL.inc(count)
        THEMES_DETECTED.set(num_themes)
        AVG_SENTIMENT_SCORE.set(avg_sentiment)

    def record_agent_invocation(self, agent_name: str) -> None:
        """Record an agent invocation."""
        AGENT_INVOCATIONS_TOTAL.labels(agent_name=agent_name).inc()

    def record_agent_error(self, agent_name: str, error_type: str) -> None:
        """Record an agent error."""
        AGENT_ERRORS_TOTAL.labels(agent_name=agent_name, error_type=error_type).inc()

    def record_llm_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record LLM token usage."""
        LLM_TOKENS_USED.labels(model=model, token_type="prompt").inc(prompt_tokens)
        LLM_TOKENS_USED.labels(model=model, token_type="completion").inc(completion_tokens)

    def record_alert(self, severity: str, alert_type: str) -> None:
        """Record an alert generation."""
        ALERTS_GENERATED_TOTAL.labels(severity=severity, alert_type=alert_type).inc()


def timed_stage(stage_name: str, metrics: MetricsCollector):
    """Decorator to time and record pipeline stage duration."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                metrics.record_stage_duration(stage_name, duration)
                logger.debug("Stage '%s' completed in %.2fs", stage_name, duration)
        return wrapper
    return decorator


# ─── Slack Alerting ───────────────────────────────────────────────────────────

@dataclass
class SlackAlerter:
    """Sends alerts to Slack channels."""

    webhook_url: str
    channel_overrides: dict[str, str] = field(default_factory=dict)
    default_channel: str = "#product-ops"

    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "warning",
        fields: Optional[list[dict[str, str]]] = None,
        channel: Optional[str] = None,
    ) -> bool:
        """Send an alert to Slack."""
        target_channel = channel or self.channel_overrides.get(severity, self.default_channel)

        color_map = {
            "critical": "#dc3545",
            "warning": "#ffc107",
            "info": "#17a2b8",
            "success": "#28a745",
        }

        emoji_map = {
            "critical": ":rotating_light:",
            "warning": ":warning:",
            "info": ":information_source:",
            "success": ":white_check_mark:",
        }

        attachment = {
            "color": color_map.get(severity, "#6c757d"),
            "title": f"{emoji_map.get(severity, '')} {title}",
            "text": message,
            "footer": "Feedback Analysis Pipeline",
            "ts": int(datetime.now(timezone.utc).timestamp()),
        }

        if fields:
            attachment["fields"] = [
                {"title": f["title"], "value": f["value"], "short": f.get("short", True)}
                for f in fields
            ]

        payload = {
            "channel": target_channel,
            "attachments": [attachment],
        }

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Slack alert sent: %s (%s)", title, severity)
            return True
        except requests.RequestException as e:
            logger.error("Failed to send Slack alert: %s", e)
            return False

    def send_pipeline_summary(
        self,
        run_id: str,
        status: str,
        feedback_count: int,
        theme_count: int,
        alert_count: int,
        top_themes: list[dict[str, Any]],
        duration_seconds: float,
    ) -> bool:
        """Send a pipeline run summary to Slack."""
        status_emoji = ":white_check_mark:" if status == "success" else ":x:"

        theme_lines = "\\\\n".join(
            f"  {i+1}. *{t['label']}* (Score: {t['priority_score']}, ARR: {t.get('total_arr', 0)})"
            for i, t in enumerate(top_themes[:5])
        )

        message = f"""
*Pipeline Run Summary*
Run ID: {run_id}
Status: {status_emoji} {status.upper()}
Duration: {duration_seconds}s

*Metrics:*
- Feedback Processed: {feedback_count}
- Themes Identified: {theme_count}
- Alerts Generated: {alert_count}

*Top Themes by Priority:*
{theme_lines}
        """.strip()

        return self.send_alert(
            title="Feedback Pipeline Run Complete",
            message=message,
            severity="success" if status == "success" else "warning",
        )

    def send_churn_risk_alert(
        self,
        account_name: str,
        arr: float,
        risk_signals: list[str],
        recent_feedback: str,
    ) -> bool:
        """Send a churn risk alert for a high-value account."""
        return self.send_alert(
            title=f"Churn Risk Detected: {account_name}",
            message=f"*Recent Feedback:*\\\\n>{recent_feedback[:300]}...",
            severity="critical",
            fields=[
                {"title": "Account ARR", "value": f"USD {arr}"},
                {"title": "Risk Signals", "value": "\\\\n".join(f"- {s}" for s in risk_signals)},
            ],
            channel=self.channel_overrides.get("churn_risk", "#customer-success"),
        )


# ─── LangSmith Tracing Integration ────────────────────────────────────────────

@dataclass
class LangSmithTracer:
    """Integration with LangSmith for agent tracing."""

    project_name: str = "feedback-analysis"

    def trace_agent_run(
        self,
        agent_name: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log an agent run to LangSmith (handled automatically by LangChain)."""
        # LangSmith tracing is automatic when LANGCHAIN_TRACING_V2=true
        # This method provides a hook for additional custom logging
        logger.debug(
            "Agent trace: %s | inputs=%d keys | outputs=%d keys",
            agent_name,
            len(inputs),
            len(outputs),
        )

    def add_feedback(self, run_id: str, score: float, comment: Optional[str] = None) -> None:
        """Add human feedback to a traced run for fine-tuning."""
        # Requires LangSmith API for programmatic feedback
        logger.info("Feedback recorded for run %s: score=%.2f", run_id, score)


def create_observability_stack(config: dict[str, Any]) -> tuple[MetricsCollector, SlackAlerter, LangSmithTracer]:
    """Create the complete observability stack."""

    metrics = MetricsCollector(port=config.get("metrics_port", 8080))
    metrics.start_server()

    alerter = SlackAlerter(
        webhook_url=config["slack_webhook_url"],
        channel_overrides={
            "critical": config.get("slack_critical_channel", "#product-ops-critical"),
            "churn_risk": config.get("slack_cs_channel", "#customer-success"),
        },
    )

    tracer = LangSmithTracer(project_name=config.get("langsmith_project", "feedback-analysis"))

    logger.info("Observability stack initialized: metrics, alerting, tracing")
    return metrics, alerter, tracer`,
              },
            ],
          },
        ],
      },
    },

    // ── Pain Point 2: Analytics Implementation Failure ──────────────────
    {
      id: 'analytics-implementation-failure',
      number: 2,
      title: 'Analytics Implementation Failure',
      subtitle: 'Event Tracking That Lies to Product Teams',
      summary:
        'Your analytics setup tracks pageviews but misses 60% of user interactions. Funnel analysis is fiction because events fire incorrectly or not at all.',
      tags: ['analytics', 'event-tracking', 'product-analytics'],
      metrics: {
        annualCostRange: '$500K - $2.5M',
        roi: '7x',
        paybackPeriod: '2-3 months',
        investmentRange: '$70K - $140K',
      },
      price: {
        present: {
          title: 'Broken Event Tracking Across the Product',
          description:
            'Analytics events are misconfigured, missing, or duplicated — making funnel analysis, feature adoption metrics, and A/B test results unreliable.',
          bullets: [
            'Over 60% of critical user interactions have no event tracking or fire incorrectly',
            'Funnel conversion rates are off by 20-40% due to missing or duplicate events',
            'A/B test results are inconclusive because exposure events do not match assignments',
            'Product decisions are made on dashboards that show fabricated engagement numbers',
          ],
          severity: 'high',
        },
        root: {
          title: 'No Validation Layer for Analytics Code',
          description:
            'Event tracking is implemented ad-hoc by feature engineers with no schema contract, automated validation, or regression testing.',
          bullets: [
            'No centralized event schema — each engineer names and structures events differently',
            'Analytics code ships without tests; broken tracking is found weeks later (if at all)',
            'Frontend framework changes silently break event listeners without alerts',
            'No monitoring to detect when event volumes drop or spike anomalously',
          ],
          severity: 'high',
        },
        impact: {
          title: 'Decisions Built on False Data',
          description:
            'Unreliable analytics directly causes bad product decisions, wasted engineering effort, and inability to measure what matters.',
          bullets: [
            'Product launches deemed successful based on inflated metrics, masking real adoption issues',
            'Engineering builds features the data "suggests" users want — but the data is wrong',
            'Revenue impact of pricing experiments cannot be measured, stalling pricing optimization',
            'Customer health scores in CS tools are inaccurate, leading to missed churn signals',
          ],
          severity: 'high',
        },
        cost: {
          title: 'Investment in Analytics Reliability',
          description:
            'Fixing analytics requires an event schema registry, automated validation pipeline, and anomaly detection — a focused 2-3 month effort.',
          bullets: [
            'Event schema registry and contract enforcement layer',
            'Automated validation framework that runs in CI and production',
            'Anomaly detection for event volume and property drift',
            'Backfill and reconciliation for historically broken events',
          ],
          severity: 'medium',
        },
        expectedReturn: {
          title: 'Trustworthy Product Analytics',
          description:
            'A validated analytics pipeline means product teams can trust their data for the first time, enabling faster and more accurate decisions.',
          bullets: [
            'Increase event tracking coverage from 40% to 98% of critical user interactions',
            'Reduce dashboard error rate from 20-40% to under 2%',
            'Cut analytics debugging time by 80% with automated validation and alerts',
            'Enable reliable A/B testing, unlocking 15-25% conversion improvements',
          ],
          severity: 'high',
        },
      },
      implementation: {
        overview:
          'Audit the current event tracking implementation, build an automated validation framework that catches broken analytics in CI and production, and deploy anomaly detection to alert on tracking regressions.',
        prerequisites: [
          'Access to the analytics event stream (Segment, Amplitude, Mixpanel, or raw event logs)',
          'PostgreSQL or BigQuery for the event audit warehouse',
          'Python 3.10+ with pandas, pytest, and statistical libraries',
          'CI/CD pipeline access for integrating validation checks',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Event Tracking Audit & Gap Analysis',
            description:
              'Systematically audit every tracked event against the expected tracking plan, identify gaps, duplicates, and misconfigured properties.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Event Coverage Audit Query',
                description:
                  'Compare the expected tracking plan against actual events in the warehouse to surface missing events, property drift, and volume anomalies.',
                code: `-- Audit event tracking coverage against the expected tracking plan
WITH expected_events AS (
    -- Tracking plan: what SHOULD be tracked
    SELECT event_name, expected_properties, critical_funnel_step,
           min_expected_daily_volume, max_expected_daily_volume
    FROM analytics_tracking_plan
    WHERE is_active = TRUE
),
actual_events AS (
    -- What IS actually being tracked (last 30 days)
    SELECT
        event_name,
        COUNT(*)                                        AS total_fires,
        COUNT(*) / 30.0                                 AS avg_daily_volume,
        COUNT(DISTINCT user_id)                         AS unique_users,
        COUNT(DISTINCT DATE(event_timestamp))           AS days_with_data,
        ARRAY_AGG(DISTINCT jsonb_object_keys(properties)) AS observed_properties,
        MAX(event_timestamp)                            AS last_seen
    FROM raw_events
    WHERE event_timestamp >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY event_name
),
audit_result AS (
    SELECT
        e.event_name,
        e.critical_funnel_step,
        CASE
            WHEN a.event_name IS NULL THEN 'MISSING'
            WHEN a.days_with_data < 25 THEN 'INTERMITTENT'
            WHEN a.avg_daily_volume < e.min_expected_daily_volume * 0.5 THEN 'UNDER_FIRING'
            WHEN a.avg_daily_volume > e.max_expected_daily_volume * 2.0 THEN 'OVER_FIRING'
            ELSE 'HEALTHY'
        END AS tracking_status,
        e.min_expected_daily_volume,
        ROUND(a.avg_daily_volume, 1)                    AS actual_daily_volume,
        a.unique_users,
        a.days_with_data,
        a.last_seen,
        e.expected_properties,
        a.observed_properties
    FROM expected_events e
    LEFT JOIN actual_events a ON e.event_name = a.event_name
)
SELECT
    tracking_status,
    critical_funnel_step,
    event_name,
    min_expected_daily_volume   AS expected_vol,
    actual_daily_volume         AS actual_vol,
    days_with_data,
    last_seen::DATE
FROM audit_result
WHERE tracking_status != 'HEALTHY'
ORDER BY
    CASE tracking_status
        WHEN 'MISSING' THEN 1 WHEN 'UNDER_FIRING' THEN 2
        WHEN 'OVER_FIRING' THEN 3 WHEN 'INTERMITTENT' THEN 4
    END,
    critical_funnel_step DESC NULLS LAST;`,
              },
              {
                language: 'sql',
                title: 'Event Property Drift Detection',
                description:
                  'Detect events where tracked properties have drifted from the schema — missing required fields, unexpected types, or new unknown properties.',
                code: `-- Detect property schema drift in tracked events
WITH property_analysis AS (
    SELECT
        event_name,
        key                                              AS property_name,
        COUNT(*)                                         AS occurrences,
        COUNT(*) FILTER (WHERE value IS NULL)            AS null_count,
        ROUND(
            COUNT(*) FILTER (WHERE value IS NULL)::NUMERIC
            / NULLIF(COUNT(*), 0) * 100, 1
        )                                                AS null_pct,
        COUNT(DISTINCT jsonb_typeof(value))              AS distinct_types,
        MODE() WITHIN GROUP (ORDER BY jsonb_typeof(value)) AS dominant_type
    FROM raw_events,
    LATERAL jsonb_each(properties) AS kv(key, value)
    WHERE event_timestamp >= CURRENT_DATE - INTERVAL '14 days'
    GROUP BY event_name, key
),
schema_check AS (
    SELECT
        pa.event_name,
        pa.property_name,
        tp.expected_type,
        pa.dominant_type                                  AS actual_type,
        tp.is_required,
        pa.null_pct,
        CASE
            WHEN tp.property_name IS NULL THEN 'UNEXPECTED_PROPERTY'
            WHEN tp.expected_type != pa.dominant_type THEN 'TYPE_MISMATCH'
            WHEN tp.is_required AND pa.null_pct > 5 THEN 'REQUIRED_FIELD_MISSING'
            WHEN pa.distinct_types > 1 THEN 'MIXED_TYPES'
            ELSE 'OK'
        END AS drift_status
    FROM property_analysis pa
    LEFT JOIN tracking_plan_properties tp
        ON pa.event_name = tp.event_name AND pa.property_name = tp.property_name
)
SELECT event_name, property_name, drift_status,
       expected_type, actual_type, null_pct
FROM schema_check
WHERE drift_status != 'OK'
ORDER BY event_name, drift_status;`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Automated Event Validation Framework',
            description:
              'Build a Python validation framework that enforces the tracking plan schema, runs in CI to catch broken analytics before deploy, and validates events in production.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Event Schema Validator',
                description:
                  'A validation engine that checks every analytics event against the tracking plan schema, catching type mismatches, missing properties, and naming violations.',
                code: `import json
import re
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ValidationError:
    event_name: str
    error_type: str  # missing_property, type_mismatch, naming_violation, unknown_event
    detail: str
    severity: str    # error, warning

@dataclass
class TrackingPlanValidator:
    """Validates analytics events against a centralized tracking plan schema."""

    schema: dict = field(default_factory=dict)
    errors: list[ValidationError] = field(default_factory=list)

    @classmethod
    def from_file(cls, schema_path: str) -> "TrackingPlanValidator":
        with open(schema_path) as f:
            schema = json.load(f)
        return cls(schema=schema)

    def validate_event(self, event_name: str, properties: dict) -> list[ValidationError]:
        errors: list[ValidationError] = []

        # Check naming convention: snake_case
        if not re.match(r'^[a-z][a-z0-9]*(_[a-z0-9]+)*$', event_name):
            errors.append(ValidationError(
                event_name, "naming_violation",
                f"Event name '{event_name}' must be snake_case", "error",
            ))

        event_schema = self.schema.get("events", {}).get(event_name)
        if event_schema is None:
            errors.append(ValidationError(
                event_name, "unknown_event",
                f"Event '{event_name}' not found in tracking plan", "warning",
            ))
            return errors

        # Validate required properties exist
        for prop_name, prop_def in event_schema.get("properties", {}).items():
            if prop_def.get("required") and prop_name not in properties:
                errors.append(ValidationError(
                    event_name, "missing_property",
                    f"Required property '{prop_name}' missing", "error",
                ))

            if prop_name in properties and properties[prop_name] is not None:
                expected_type = prop_def.get("type", "string")
                actual_value = properties[prop_name]
                if not self._type_check(actual_value, expected_type):
                    errors.append(ValidationError(
                        event_name, "type_mismatch",
                        f"Property '{prop_name}': expected {expected_type}, "
                        f"got {type(actual_value).__name__}", "error",
                    ))

        # Flag unexpected properties not in schema
        known_props = set(event_schema.get("properties", {}).keys())
        for prop_name in properties:
            if prop_name not in known_props:
                errors.append(ValidationError(
                    event_name, "unexpected_property",
                    f"Property '{prop_name}' not defined in tracking plan", "warning",
                ))

        self.errors.extend(errors)
        return errors

    def _type_check(self, value, expected: str) -> bool:
        type_map = {"string": str, "number": (int, float), "boolean": bool, "array": list}
        return isinstance(value, type_map.get(expected, str))`,
              },
              {
                language: 'python',
                title: 'Production Event Volume Anomaly Detector',
                description:
                  'Monitor production event streams for volume anomalies — detecting when events stop firing, spike unexpectedly, or show gradual degradation.',
                code: `import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import psycopg2

@dataclass
class VolumeAnomaly:
    event_name: str
    anomaly_type: str    # drop, spike, flatline, gradual_decline
    expected_volume: float
    actual_volume: float
    deviation_pct: float
    detected_at: datetime

class EventVolumeMonitor:
    """Detects anomalies in analytics event volumes using statistical baselines."""

    def __init__(self, db_conn, z_threshold: float = 2.5):
        self.conn = db_conn
        self.z_threshold = z_threshold

    def compute_baselines(self, lookback_days: int = 28) -> dict[str, dict]:
        """Compute per-event daily volume baselines using median and MAD."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT event_name, DATE(event_timestamp) AS event_date, COUNT(*) AS daily_vol
                FROM raw_events
                WHERE event_timestamp >= CURRENT_DATE - INTERVAL '%s days'
                  AND event_timestamp < CURRENT_DATE
                GROUP BY event_name, DATE(event_timestamp)
            """, (lookback_days,))
            rows = cur.fetchall()

        volumes: dict[str, list[float]] = {}
        for event_name, _, daily_vol in rows:
            volumes.setdefault(event_name, []).append(float(daily_vol))

        baselines = {}
        for event_name, vols in volumes.items():
            arr = np.array(vols)
            median = float(np.median(arr))
            mad = float(np.median(np.abs(arr - median))) * 1.4826  # scaled MAD
            baselines[event_name] = {"median": median, "mad": max(mad, 1.0), "n_days": len(vols)}
        return baselines

    def detect_anomalies(self) -> list[VolumeAnomaly]:
        baselines = self.compute_baselines()
        anomalies: list[VolumeAnomaly] = []
        today_volumes = self._get_today_volumes()
        now = datetime.now(timezone.utc)

        for event_name, baseline in baselines.items():
            actual = today_volumes.get(event_name, 0.0)
            z_score = (actual - baseline["median"]) / baseline["mad"]

            if abs(z_score) < self.z_threshold:
                continue

            if actual < baseline["median"] * 0.1:
                anomaly_type = "flatline"
            elif z_score < -self.z_threshold:
                anomaly_type = "drop"
            else:
                anomaly_type = "spike"

            anomalies.append(VolumeAnomaly(
                event_name=event_name,
                anomaly_type=anomaly_type,
                expected_volume=baseline["median"],
                actual_volume=actual,
                deviation_pct=round((actual - baseline["median"]) / baseline["median"] * 100, 1),
                detected_at=now,
            ))
        return sorted(anomalies, key=lambda a: abs(a.deviation_pct), reverse=True)

    def _get_today_volumes(self) -> dict[str, float]:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT event_name, COUNT(*) AS vol
                FROM raw_events
                WHERE event_timestamp >= CURRENT_DATE
                GROUP BY event_name
            """)
            return {row[0]: float(row[1]) for row in cur.fetchall()}`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description:
              'Validate the analytics event pipeline with SQL-based schema conformance assertions and a comprehensive pytest framework that verifies event payloads, property types, and naming conventions.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Event Schema Conformance Assertions',
                description:
                  'Data quality assertions that verify event payloads conform to the tracking plan schema, catching type mismatches, missing required properties, and naming violations.',
                code: `-- Event schema conformance assertions for analytics pipeline
-- Validates actual events against the tracking plan contract

-- Assert 1: All tracked events exist in the tracking plan
WITH unknown_events AS (
    SELECT
        re.event_name,
        COUNT(*)                                           AS fire_count,
        MIN(re.event_timestamp)                            AS first_seen,
        MAX(re.event_timestamp)                            AS last_seen
    FROM raw_events re
    LEFT JOIN analytics_tracking_plan tp
        ON re.event_name = tp.event_name
    WHERE re.event_timestamp >= CURRENT_DATE - INTERVAL '7 days'
      AND tp.event_name IS NULL
    GROUP BY re.event_name
),
-- Assert 2: Required properties are present and non-null
missing_required AS (
    SELECT
        re.event_name,
        tpp.property_name                                  AS missing_property,
        COUNT(*)                                           AS violation_count,
        ROUND(
            COUNT(*)::NUMERIC / NULLIF(
                (SELECT COUNT(*) FROM raw_events r2
                 WHERE r2.event_name = re.event_name
                   AND r2.event_timestamp >= CURRENT_DATE - INTERVAL '7 days'), 0
            ) * 100, 2
        )                                                  AS violation_pct
    FROM raw_events re
    JOIN tracking_plan_properties tpp
        ON re.event_name = tpp.event_name
    WHERE re.event_timestamp >= CURRENT_DATE - INTERVAL '7 days'
      AND tpp.is_required = TRUE
      AND (
          re.properties ->> tpp.property_name IS NULL
          OR re.properties ->> tpp.property_name = ''
      )
    GROUP BY re.event_name, tpp.property_name
),
-- Assert 3: Property type conformance
type_mismatches AS (
    SELECT
        re.event_name,
        tpp.property_name,
        tpp.expected_type,
        jsonb_typeof(re.properties -> tpp.property_name)   AS actual_type,
        COUNT(*)                                           AS mismatch_count
    FROM raw_events re
    JOIN tracking_plan_properties tpp
        ON re.event_name = tpp.event_name
    WHERE re.event_timestamp >= CURRENT_DATE - INTERVAL '7 days'
      AND re.properties -> tpp.property_name IS NOT NULL
      AND jsonb_typeof(re.properties -> tpp.property_name) != tpp.expected_type
    GROUP BY re.event_name, tpp.property_name, tpp.expected_type,
             jsonb_typeof(re.properties -> tpp.property_name)
),
-- Assert 4: Naming convention check (must be snake_case)
naming_violations AS (
    SELECT DISTINCT event_name
    FROM raw_events
    WHERE event_timestamp >= CURRENT_DATE - INTERVAL '7 days'
      AND event_name !~ '^[a-z][a-z0-9]*(_[a-z0-9]+)*$'
)
SELECT 'unknown_events'       AS assertion,
       (SELECT COUNT(*) FROM unknown_events)    AS violation_count,
       CASE WHEN (SELECT COUNT(*) FROM unknown_events) = 0
            THEN 'PASS' ELSE 'FAIL' END        AS status
UNION ALL
SELECT 'missing_required_props',
       (SELECT COALESCE(SUM(violation_count), 0) FROM missing_required),
       CASE WHEN (SELECT COALESCE(SUM(violation_count), 0) FROM missing_required) = 0
            THEN 'PASS' ELSE 'FAIL' END
UNION ALL
SELECT 'type_conformance',
       (SELECT COALESCE(SUM(mismatch_count), 0) FROM type_mismatches),
       CASE WHEN (SELECT COALESCE(SUM(mismatch_count), 0) FROM type_mismatches) = 0
            THEN 'PASS' ELSE 'FAIL' END
UNION ALL
SELECT 'naming_convention',
       (SELECT COUNT(*) FROM naming_violations),
       CASE WHEN (SELECT COUNT(*) FROM naming_violations) = 0
            THEN 'PASS' ELSE 'FAIL' END;`,
              },
              {
                language: 'python',
                title: 'Pytest Event Validation Framework',
                description:
                  'Comprehensive pytest suite that validates event payloads against the tracking plan, checks property types, enforces naming conventions, and verifies event volume baselines.',
                code: `import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest
import psycopg2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TRACKING_PLAN_PATH = Path("analytics/tracking_plan.json")
DB_CONNECTION_STRING = "postgresql://analytics:pass@localhost:5432/events"


@dataclass
class EventValidationResult:
    """Result of validating a single event against the tracking plan."""
    event_name: str
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class EventValidator:
    """Validates analytics event payloads against the tracking plan schema."""

    SNAKE_CASE_RE: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
    TYPE_MAP: dict[str, type] = {
        "string": str,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def __init__(self, tracking_plan: dict[str, Any]) -> None:
        self.plan: dict[str, Any] = tracking_plan
        self.events_schema: dict[str, Any] = tracking_plan.get("events", {})
        logger.info("EventValidator loaded with %d event schemas", len(self.events_schema))

    def validate(self, event_name: str, properties: dict[str, Any]) -> EventValidationResult:
        """Validate a single event payload against the tracking plan."""
        result = EventValidationResult(event_name=event_name, is_valid=True)

        # Check naming convention
        if not self.SNAKE_CASE_RE.match(event_name):
            result.errors.append(f"Event name '{event_name}' violates snake_case convention")
            result.is_valid = False

        # Check event exists in plan
        schema: Optional[dict[str, Any]] = self.events_schema.get(event_name)
        if schema is None:
            result.warnings.append(f"Event '{event_name}' not found in tracking plan")
            return result

        # Check required properties
        prop_schemas: dict[str, Any] = schema.get("properties", {})
        for prop_name, prop_def in prop_schemas.items():
            if prop_def.get("required", False) and prop_name not in properties:
                result.errors.append(f"Missing required property: {prop_name}")
                result.is_valid = False

            if prop_name in properties and properties[prop_name] is not None:
                expected_type: str = prop_def.get("type", "string")
                actual_value: Any = properties[prop_name]
                if not isinstance(actual_value, self.TYPE_MAP.get(expected_type, str)):
                    result.errors.append(
                        f"Property '{prop_name}': expected {expected_type}, "
                        f"got {type(actual_value).__name__}"
                    )
                    result.is_valid = False

        # Flag unexpected properties
        known_props: set[str] = set(prop_schemas.keys())
        for prop_name in properties:
            if prop_name not in known_props:
                result.warnings.append(f"Unexpected property: {prop_name}")

        return result


@pytest.fixture(scope="module")
def tracking_plan() -> dict[str, Any]:
    """Load the tracking plan schema from file."""
    logger.info("Loading tracking plan from %s", TRACKING_PLAN_PATH)
    return json.loads(TRACKING_PLAN_PATH.read_text())


@pytest.fixture(scope="module")
def validator(tracking_plan: dict[str, Any]) -> EventValidator:
    """Create an EventValidator instance."""
    return EventValidator(tracking_plan)


@pytest.fixture(scope="module")
def recent_events() -> list[dict[str, Any]]:
    """Fetch recent events from the warehouse for validation."""
    logger.info("Fetching recent events for validation")
    conn = psycopg2.connect(DB_CONNECTION_STRING)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT event_name, properties
            FROM raw_events
            WHERE event_timestamp >= CURRENT_DATE - INTERVAL '1 day'
            ORDER BY RANDOM()
            LIMIT 5000
        """)
        rows = cur.fetchall()
    conn.close()
    logger.info("Fetched %d events for validation", len(rows))
    return [{"event_name": r[0], "properties": r[1] or {}} for r in rows]


class TestEventSchemaConformance:
    """Validate production events conform to the tracking plan schema."""

    def test_all_events_recognized(
        self, validator: EventValidator, recent_events: list[dict[str, Any]]
    ) -> None:
        unknown: list[str] = []
        for evt in recent_events:
            result: EventValidationResult = validator.validate(evt["event_name"], evt["properties"])
            if any("not found in tracking plan" in w for w in result.warnings):
                unknown.append(evt["event_name"])
        unique_unknown: set[str] = set(unknown)
        logger.info("Unknown events: %d unique out of %d total", len(unique_unknown), len(unknown))
        assert len(unique_unknown) == 0, f"Unknown events: {unique_unknown}"

    def test_required_properties_present(
        self, validator: EventValidator, recent_events: list[dict[str, Any]]
    ) -> None:
        failures: list[str] = []
        for evt in recent_events:
            result: EventValidationResult = validator.validate(evt["event_name"], evt["properties"])
            for err in result.errors:
                if "Missing required" in err:
                    failures.append(f"{evt['event_name']}: {err}")
        logger.info("Required property violations: %d", len(failures))
        assert len(failures) == 0, f"Missing required props:\\n" + "\\n".join(failures[:20])

    def test_property_types_match(
        self, validator: EventValidator, recent_events: list[dict[str, Any]]
    ) -> None:
        mismatches: list[str] = []
        for evt in recent_events:
            result: EventValidationResult = validator.validate(evt["event_name"], evt["properties"])
            for err in result.errors:
                if "expected" in err and "got" in err:
                    mismatches.append(f"{evt['event_name']}: {err}")
        logger.info("Type mismatches: %d", len(mismatches))
        assert len(mismatches) == 0, f"Type mismatches:\\n" + "\\n".join(mismatches[:20])

    def test_naming_conventions(self, recent_events: list[dict[str, Any]]) -> None:
        pattern: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
        violations: list[str] = [
            evt["event_name"] for evt in recent_events
            if not pattern.match(evt["event_name"])
        ]
        unique_violations: set[str] = set(violations)
        logger.info("Naming violations: %d unique", len(unique_violations))
        assert len(unique_violations) == 0, f"Naming violations: {unique_violations}"`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description:
              'Deploy the analytics validation pipeline as a containerized service with automated tag auditing, health checks, and configuration management for the event processing infrastructure.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Analytics Tag Deployment Script',
                description:
                  'Production deployment script for the analytics validation service with Docker builds, tracking plan validation, rolling deployment, and rollback support.',
                code: `#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------
# Analytics Validation Service — Deployment Script
# Builds, validates tracking plan, and deploys the container
# -----------------------------------------------------------

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="\${SCRIPT_DIR}/.."
IMAGE_NAME="analytics-validator"
REGISTRY="\${DOCKER_REGISTRY:-ghcr.io/myorg}"
DEPLOY_ENV="\${DEPLOY_ENV:-staging}"
COMPOSE_FILE="\${PROJECT_ROOT}/docker/docker-compose.\${DEPLOY_ENV}.yml"
TRACKING_PLAN="\${PROJECT_ROOT}/analytics/tracking_plan.json"
HEALTH_ENDPOINT="http://localhost:8092/health"
HEALTH_TIMEOUT=90
ROLLBACK_TAG=""

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] \$*"; }
err() { log "ERROR: \$*" >&2; }

cleanup() {
    local exit_code=\$?
    if [[ \$exit_code -ne 0 && -n "\${ROLLBACK_TAG}" ]]; then
        err "Deployment failed (exit \${exit_code}). Rolling back to \${ROLLBACK_TAG}"
        docker tag "\${REGISTRY}/\${IMAGE_NAME}:\${ROLLBACK_TAG}" \\
                    "\${REGISTRY}/\${IMAGE_NAME}:latest"
        docker-compose -f "\${COMPOSE_FILE}" up -d --no-build analytics-validator
        log "Rollback complete."
    fi
}
trap cleanup EXIT

# Validate tracking plan JSON before deployment
log "Validating tracking plan schema"
if ! python3 -c "
import json, sys
with open('\${TRACKING_PLAN}') as f:
    plan = json.load(f)
events = plan.get('events', {})
if not events:
    print('ERROR: No events defined in tracking plan', file=sys.stderr)
    sys.exit(1)
for name, schema in events.items():
    if not schema.get('properties'):
        print(f'WARNING: Event {name} has no properties defined', file=sys.stderr)
print(f'Tracking plan valid: {len(events)} events defined')
"; then
    err "Tracking plan validation failed"
    exit 1
fi

# Capture current tag for rollback
ROLLBACK_TAG=\$(docker inspect --format='{{.Config.Image}}' analytics-validator 2>/dev/null | awk -F: '{print \$2}') || true
log "Current tag for rollback: \${ROLLBACK_TAG:-none}"

# Build container image
GIT_SHA=\$(git -C "\${PROJECT_ROOT}" rev-parse --short HEAD)
BUILD_TAG="\${DEPLOY_ENV}-\${GIT_SHA}-\$(date +%Y%m%d%H%M%S)"
log "Building image: \${IMAGE_NAME}:\${BUILD_TAG}"
docker build \\
    --file "\${PROJECT_ROOT}/docker/Dockerfile.analytics-validator" \\
    --build-arg DEPLOY_ENV="\${DEPLOY_ENV}" \\
    --build-arg TRACKING_PLAN_VERSION="\${GIT_SHA}" \\
    --tag "\${REGISTRY}/\${IMAGE_NAME}:\${BUILD_TAG}" \\
    --tag "\${REGISTRY}/\${IMAGE_NAME}:latest" \\
    "\${PROJECT_ROOT}"

# Run validation test suite inside container
log "Running analytics validation tests"
docker run --rm \\
    -e DATABASE_URL="\${DATABASE_URL:-postgresql://localhost:5432/analytics}" \\
    "\${REGISTRY}/\${IMAGE_NAME}:\${BUILD_TAG}" \\
    python -m pytest tests/test_event_validation.py -x --tb=short -q

# Push to registry
log "Pushing to registry"
docker push "\${REGISTRY}/\${IMAGE_NAME}:\${BUILD_TAG}"
docker push "\${REGISTRY}/\${IMAGE_NAME}:latest"

# Deploy with rolling restart
log "Deploying to \${DEPLOY_ENV}"
docker-compose -f "\${COMPOSE_FILE}" pull analytics-validator
docker-compose -f "\${COMPOSE_FILE}" up -d --no-build analytics-validator

# Health check loop
log "Waiting for health check (timeout: \${HEALTH_TIMEOUT}s)"
elapsed=0
while [[ \$elapsed -lt \$HEALTH_TIMEOUT ]]; do
    if curl -sf "\${HEALTH_ENDPOINT}" > /dev/null 2>&1; then
        log "Health check passed after \${elapsed}s"
        break
    fi
    sleep 3
    elapsed=\$((elapsed + 3))
done

if [[ \$elapsed -ge \$HEALTH_TIMEOUT ]]; then
    err "Health check timed out after \${HEALTH_TIMEOUT}s"
    exit 1
fi

log "Deployment complete: \${IMAGE_NAME}:\${BUILD_TAG}"`,
              },
              {
                language: 'python',
                title: 'Analytics Pipeline Configuration',
                description:
                  'Configuration loader for the analytics validation pipeline with support for event stream connections, warehouse credentials, validation thresholds, and alerting configuration.',
                code: `import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class EventStreamConfig:
    """Configuration for an analytics event stream source."""
    name: str
    stream_type: str            # segment, amplitude, mixpanel, raw_kafka
    connection_url: str
    api_key: str
    batch_size: int = 500
    poll_interval_seconds: int = 30
    enabled: bool = True


@dataclass
class ValidationThresholds:
    """Thresholds for analytics validation checks."""
    max_unknown_event_pct: float = 0.02
    max_missing_property_pct: float = 0.01
    max_type_mismatch_pct: float = 0.005
    volume_z_score_threshold: float = 2.5
    min_daily_volume_per_event: int = 10
    freshness_max_lag_minutes: int = 15


@dataclass
class AnalyticsPipelineConfig:
    """Top-level configuration for the analytics validation pipeline."""
    db_connection_string: str
    tracking_plan_path: str
    event_streams: list[EventStreamConfig]
    thresholds: ValidationThresholds
    slack_webhook_url: Optional[str] = None
    log_level: str = "INFO"
    ci_mode: bool = False
    report_output_dir: str = "./reports"
    max_retries: int = 3
    retry_delay_seconds: int = 15


class AnalyticsConfigLoader:
    """Loads and validates analytics pipeline configuration from file and environment."""

    ENV_PREFIX: str = "ANALYTICS_"

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path: Optional[Path] = Path(config_path) if config_path else None
        self._raw: dict[str, Any] = {}
        logger.info("AnalyticsConfigLoader initialized: path=%s", self.config_path)

    def load(self) -> AnalyticsPipelineConfig:
        """Load config from JSON file with environment variable overrides."""
        if self.config_path and self.config_path.exists():
            logger.info("Loading config from %s", self.config_path)
            with open(self.config_path) as f:
                self._raw = json.load(f)
        else:
            logger.warning("Config file not found at %s, using env-only", self.config_path)
            self._raw = {}

        # Environment overrides for sensitive values
        db_conn: str = self._env("DB_CONNECTION_STRING", self._raw.get("db_connection_string", ""))
        slack_url: Optional[str] = self._env("SLACK_WEBHOOK_URL", self._raw.get("slack_webhook_url"))
        log_level: str = self._env("LOG_LEVEL", self._raw.get("log_level", "INFO"))
        ci_mode: bool = self._env("CI_MODE", str(self._raw.get("ci_mode", False))).lower() == "true"

        # Parse event streams
        streams: list[EventStreamConfig] = self._load_streams(self._raw.get("event_streams", []))

        # Parse validation thresholds
        thresh_raw: dict[str, Any] = self._raw.get("thresholds", {})
        thresholds = ValidationThresholds(
            max_unknown_event_pct=float(thresh_raw.get("max_unknown_event_pct", 0.02)),
            max_missing_property_pct=float(thresh_raw.get("max_missing_property_pct", 0.01)),
            max_type_mismatch_pct=float(thresh_raw.get("max_type_mismatch_pct", 0.005)),
            volume_z_score_threshold=float(thresh_raw.get("volume_z_score_threshold", 2.5)),
            min_daily_volume_per_event=int(thresh_raw.get("min_daily_volume_per_event", 10)),
            freshness_max_lag_minutes=int(thresh_raw.get("freshness_max_lag_minutes", 15)),
        )

        config = AnalyticsPipelineConfig(
            db_connection_string=db_conn,
            tracking_plan_path=self._raw.get("tracking_plan_path", "analytics/tracking_plan.json"),
            event_streams=streams,
            thresholds=thresholds,
            slack_webhook_url=slack_url,
            log_level=log_level,
            ci_mode=ci_mode,
            report_output_dir=self._raw.get("report_output_dir", "./reports"),
        )
        self._validate(config)
        logger.info(
            "Config loaded: %d streams, ci_mode=%s, thresholds=%s",
            len(config.event_streams), config.ci_mode, thresholds,
        )
        return config

    def _load_streams(self, raw_streams: list[dict[str, Any]]) -> list[EventStreamConfig]:
        """Parse event stream configurations with env overrides for API keys."""
        streams: list[EventStreamConfig] = []
        for src in raw_streams:
            name: str = src["name"]
            key_env: str = f"{self.ENV_PREFIX}{name.upper()}_API_KEY"
            api_key: str = os.environ.get(key_env, src.get("api_key", ""))
            streams.append(EventStreamConfig(
                name=name,
                stream_type=src.get("stream_type", "raw_kafka"),
                connection_url=src["connection_url"],
                api_key=api_key,
                batch_size=int(src.get("batch_size", 500)),
                poll_interval_seconds=int(src.get("poll_interval_seconds", 30)),
                enabled=src.get("enabled", True),
            ))
            logger.info("Stream '%s': type=%s, enabled=%s", name, streams[-1].stream_type, streams[-1].enabled)
        return streams

    def _validate(self, config: AnalyticsPipelineConfig) -> None:
        """Validate required configuration values are present."""
        if not config.db_connection_string:
            raise ValueError("db_connection_string is required")
        if not Path(config.tracking_plan_path).exists() and not config.ci_mode:
            logger.warning("Tracking plan not found at %s", config.tracking_plan_path)
        active: int = sum(1 for s in config.event_streams if s.enabled)
        if active == 0:
            logger.warning("No active event streams configured")

    def _env(self, key: str, default: Any = None) -> Any:
        """Read environment variable with prefix."""
        return os.environ.get(f"{self.ENV_PREFIX}{key}", default)`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'CI Integration & Tracking Plan Enforcement',
            description:
              'Wire the validation framework into CI so broken analytics never reach production, generate coverage reports for product team visibility, and monitor data freshness and event volumes with alerting.',
            codeSnippets: [
              {
                language: 'python',
                title: 'CI Validation Test Suite',
                description:
                  'Pytest-based test suite that validates all analytics call sites in the codebase against the tracking plan, running on every pull request.',
                code: `import json
import ast
import pytest
from pathlib import Path
from dataclasses import dataclass

TRACKING_PLAN_PATH = Path("analytics/tracking_plan.json")
SOURCE_DIRS = [Path("src/"), Path("app/")]

@dataclass
class AnalyticsCallSite:
    file_path: str
    line_number: int
    event_name: str
    properties: list[str]

def extract_analytics_calls(source_dirs: list[Path]) -> list[AnalyticsCallSite]:
    """Parse source files to find all analytics.track() call sites."""
    call_sites: list[AnalyticsCallSite] = []
    track_methods = {"track", "track_event", "analytics.track", "posthog.capture"}

    for src_dir in source_dirs:
        for py_file in src_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func_name = _get_call_name(node)
                if func_name not in track_methods:
                    continue
                event_name = _extract_string_arg(node, 0)
                prop_keys = _extract_dict_keys(node, 1)
                if event_name:
                    call_sites.append(AnalyticsCallSite(
                        file_path=str(py_file),
                        line_number=node.lineno,
                        event_name=event_name,
                        properties=prop_keys,
                    ))
    return call_sites

def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return f"{getattr(node.func.value, 'id', '')}.{node.func.attr}"
    return getattr(node.func, "id", "")

def _extract_string_arg(node: ast.Call, idx: int) -> str | None:
    if len(node.args) > idx and isinstance(node.args[idx], ast.Constant):
        return node.args[idx].value
    return None

def _extract_dict_keys(node: ast.Call, idx: int) -> list[str]:
    if len(node.args) > idx and isinstance(node.args[idx], ast.Dict):
        return [k.value for k in node.args[idx].keys if isinstance(k, ast.Constant)]
    return []

class TestAnalyticsTracking:
    """CI gate: validate all analytics calls match the tracking plan."""

    @pytest.fixture(scope="class")
    def tracking_plan(self) -> dict:
        return json.loads(TRACKING_PLAN_PATH.read_text())

    @pytest.fixture(scope="class")
    def call_sites(self) -> list[AnalyticsCallSite]:
        return extract_analytics_calls(SOURCE_DIRS)

    def test_all_events_in_tracking_plan(self, tracking_plan, call_sites):
        plan_events = set(tracking_plan.get("events", {}).keys())
        unknown = [cs for cs in call_sites if cs.event_name not in plan_events]
        assert not unknown, (
            f"Found {len(unknown)} analytics calls for events not in tracking plan: "
            + ", ".join(f"{cs.event_name} ({cs.file_path}:{cs.line_number})" for cs in unknown[:5])
        )

    def test_required_properties_present(self, tracking_plan, call_sites):
        failures = []
        for cs in call_sites:
            schema = tracking_plan.get("events", {}).get(cs.event_name, {})
            required = [p for p, d in schema.get("properties", {}).items() if d.get("required")]
            missing = [p for p in required if p not in cs.properties]
            if missing:
                failures.append(f"{cs.event_name} at {cs.file_path}:{cs.line_number} missing: {missing}")
        assert not failures, f"Missing required properties:\\n" + "\\n".join(failures[:10])`,
              },
              {
                language: 'sql',
                title: 'Data Freshness & Event Volume Alerting Dashboard',
                description:
                  'SQL dashboard query that monitors data freshness lag, event volume anomalies, and pipeline health metrics for operational alerting.',
                code: `-- Data freshness and event volume alerting dashboard
-- Surfaces stale pipelines, volume anomalies, and tracking gaps

-- Section 1: Data freshness per event (detect stale pipelines)
WITH freshness AS (
    SELECT
        event_name,
        MAX(event_timestamp)                               AS last_event_at,
        EXTRACT(EPOCH FROM (NOW() - MAX(event_timestamp))) / 60.0
                                                           AS minutes_since_last,
        COUNT(*) FILTER (
            WHERE event_timestamp >= NOW() - INTERVAL '1 hour'
        )                                                  AS events_last_hour,
        COUNT(*) FILTER (
            WHERE event_timestamp >= NOW() - INTERVAL '24 hours'
        )                                                  AS events_last_24h
    FROM raw_events
    WHERE event_timestamp >= NOW() - INTERVAL '48 hours'
    GROUP BY event_name
),
-- Section 2: Volume baselines and anomaly detection
volume_baselines AS (
    SELECT
        event_name,
        DATE(event_timestamp)                              AS event_date,
        COUNT(*)                                           AS daily_volume
    FROM raw_events
    WHERE event_timestamp >= CURRENT_DATE - INTERVAL '28 days'
      AND event_timestamp < CURRENT_DATE
    GROUP BY event_name, DATE(event_timestamp)
),
volume_stats AS (
    SELECT
        event_name,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY daily_volume) AS median_volume,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY daily_volume) AS p25_volume,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY daily_volume) AS p75_volume,
        AVG(daily_volume)                                  AS avg_volume,
        STDDEV(daily_volume)                               AS stddev_volume,
        COUNT(*)                                           AS baseline_days
    FROM volume_baselines
    GROUP BY event_name
    HAVING COUNT(*) >= 7
),
today_volume AS (
    SELECT
        event_name,
        COUNT(*)                                           AS today_count,
        COUNT(DISTINCT user_id)                            AS today_users
    FROM raw_events
    WHERE event_timestamp >= CURRENT_DATE
    GROUP BY event_name
),
-- Section 3: Combined alerting view
alerts AS (
    SELECT
        COALESCE(f.event_name, vs.event_name)              AS event_name,
        f.minutes_since_last,
        f.events_last_hour,
        f.events_last_24h,
        tv.today_count,
        vs.median_volume,
        vs.avg_volume,
        CASE
            WHEN f.minutes_since_last > 60
                THEN 'STALE'
            WHEN tv.today_count IS NULL
                THEN 'NO_DATA_TODAY'
            WHEN vs.stddev_volume > 0
                AND ABS(tv.today_count - vs.avg_volume) / vs.stddev_volume > 2.5
                AND tv.today_count < vs.median_volume * 0.5
                THEN 'VOLUME_DROP'
            WHEN vs.stddev_volume > 0
                AND ABS(tv.today_count - vs.avg_volume) / vs.stddev_volume > 2.5
                AND tv.today_count > vs.median_volume * 2.0
                THEN 'VOLUME_SPIKE'
            ELSE 'HEALTHY'
        END                                                AS alert_status,
        CASE
            WHEN f.minutes_since_last > 120 THEN 'critical'
            WHEN f.minutes_since_last > 60  THEN 'warning'
            WHEN tv.today_count < vs.median_volume * 0.3 THEN 'critical'
            WHEN tv.today_count < vs.median_volume * 0.5 THEN 'warning'
            ELSE 'info'
        END                                                AS severity,
        ROUND(
            CASE WHEN vs.median_volume > 0
                 THEN (COALESCE(tv.today_count, 0) - vs.median_volume)
                      / vs.median_volume * 100
                 ELSE 0
            END, 1
        )                                                  AS deviation_pct
    FROM freshness f
    FULL OUTER JOIN volume_stats vs ON f.event_name = vs.event_name
    LEFT JOIN today_volume tv ON f.event_name = tv.event_name
)
SELECT
    alert_status,
    severity,
    event_name,
    ROUND(minutes_since_last, 1)       AS lag_minutes,
    events_last_hour,
    today_count,
    ROUND(median_volume::NUMERIC, 0)   AS baseline_median,
    deviation_pct                      AS deviation_from_baseline_pct
FROM alerts
WHERE alert_status != 'HEALTHY'
ORDER BY
    CASE severity WHEN 'critical' THEN 1 WHEN 'warning' THEN 2 ELSE 3 END,
    CASE alert_status
        WHEN 'STALE' THEN 1 WHEN 'NO_DATA_TODAY' THEN 2
        WHEN 'VOLUME_DROP' THEN 3 WHEN 'VOLUME_SPIKE' THEN 4
    END;`,
              },
            ],
          },
        ],
        toolsUsed: [
          'PostgreSQL / BigQuery (event audit warehouse)',
          'Python (pandas, pytest, AST parsing)',
          'Segment / Amplitude / Mixpanel APIs',
          'CI/CD integration (GitHub Actions / GitLab CI)',
          'pytest',
          'Docker',
          'GitHub Actions',
          'cron / Airflow',
          'Slack API',
        ],
      },
      aiEasyWin: {
        overview:
          'Use ChatGPT/Claude with Zapier to automatically audit your analytics implementation, validate event schemas against your tracking plan, and receive weekly reports on tracking gaps and anomalies without building custom infrastructure.',
        estimatedMonthlyCost: '$120 - $200/month',
        primaryTools: [
          'ChatGPT Plus ($20/mo)',
          'Zapier Pro ($29.99/mo)',
          'Amplitude ($49/mo starter)',
        ],
        alternativeTools: [
          'Claude Pro ($20/mo)',
          'Make ($10.59/mo)',
          'Mixpanel ($25/mo)',
          'Segment ($120/mo)',
        ],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description:
              'Set up automated extraction of your event data from analytics platforms (Amplitude, Mixpanel, Segment) and your tracking plan documentation to prepare for AI-powered validation.',
            toolsUsed: ['Zapier', 'Google Sheets', 'Amplitude/Mixpanel API'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Event Data Extraction Configuration',
                description:
                  'Zapier workflow that exports event data from Amplitude/Mixpanel to Google Sheets for AI analysis.',
                code: `{
  "workflow_name": "Analytics Event Data Extractor",
  "triggers": [
    {
      "name": "daily_schedule",
      "type": "schedule",
      "schedule": "every_day_9am",
      "timezone": "America/New_York"
    }
  ],
  "steps": [
    {
      "step": 1,
      "name": "fetch_amplitude_events",
      "app": "Webhooks by Zapier",
      "action": "Custom Request",
      "config": {
        "method": "GET",
        "url": "https://amplitude.com/api/2/events/list",
        "headers": {
          "Authorization": "Basic {{base64_encode(API_KEY:SECRET_KEY)}}"
        },
        "query_params": {
          "start": "{{yesterday_date}}",
          "end": "{{today_date}}"
        }
      }
    },
    {
      "step": 2,
      "name": "fetch_event_volumes",
      "app": "Webhooks by Zapier",
      "action": "Custom Request",
      "config": {
        "method": "GET",
        "url": "https://amplitude.com/api/2/events/segmentation",
        "headers": {
          "Authorization": "Basic {{base64_encode(API_KEY:SECRET_KEY)}}"
        },
        "body": {
          "e": {"event_type": "_all"},
          "start": "{{7_days_ago}}",
          "end": "{{today_date}}",
          "m": "uniques",
          "g": "event_type"
        }
      }
    },
    {
      "step": 3,
      "name": "format_event_data",
      "app": "Code by Zapier",
      "action": "Run Javascript",
      "config": {
        "code": "const events = JSON.parse(inputData.eventsResponse); const volumes = JSON.parse(inputData.volumesResponse); return events.data.map(e => ({ event_name: e.event_type, properties: Object.keys(e.event_properties || {}), daily_volume: volumes.data[e.event_type]?.value || 0, last_seen: e.last_seen }));"
      }
    },
    {
      "step": 4,
      "name": "export_to_sheets",
      "app": "Google Sheets",
      "action": "Create Spreadsheet Row(s)",
      "config": {
        "spreadsheet_id": "{{ANALYTICS_AUDIT_SHEET_ID}}",
        "worksheet": "Event Inventory",
        "rows": "{{step3.output}}"
      }
    },
    {
      "step": 5,
      "name": "fetch_tracking_plan",
      "app": "Google Sheets",
      "action": "Get Many Spreadsheet Rows",
      "config": {
        "spreadsheet_id": "{{TRACKING_PLAN_SHEET_ID}}",
        "worksheet": "Tracking Plan"
      }
    }
  ]
}`,
              },
              {
                language: 'yaml',
                title: 'Tracking Plan Schema Template',
                description:
                  'YAML schema template for defining your tracking plan that AI will validate against actual events.',
                code: `# Tracking Plan Schema for AI Validation
# Define expected events, properties, and validation rules

tracking_plan:
  version: "2.0"
  last_updated: "2024-01-15"

  global_properties:
    # Properties that should be present on all events
    - name: "user_id"
      type: "string"
      required: true
      description: "Unique user identifier"

    - name: "session_id"
      type: "string"
      required: true

    - name: "timestamp"
      type: "datetime"
      required: true

    - name: "platform"
      type: "enum"
      values: ["web", "ios", "android"]
      required: true

  events:
    # ─── User Authentication Events ─────────────────
    - name: "user_signed_up"
      category: "authentication"
      description: "User completed registration"
      critical_funnel: true
      expected_daily_volume:
        min: 100
        max: 5000
      properties:
        - name: "signup_method"
          type: "enum"
          values: ["email", "google", "github", "sso"]
          required: true
        - name: "referral_source"
          type: "string"
          required: false

    - name: "user_logged_in"
      category: "authentication"
      description: "User authenticated successfully"
      critical_funnel: true
      expected_daily_volume:
        min: 1000
        max: 50000
      properties:
        - name: "login_method"
          type: "enum"
          values: ["email", "google", "github", "sso"]
          required: true

    # ─── Core Product Events ─────────────────────────
    - name: "feature_used"
      category: "product"
      description: "User interacted with a product feature"
      properties:
        - name: "feature_name"
          type: "string"
          required: true
        - name: "feature_category"
          type: "string"
          required: true
        - name: "time_spent_seconds"
          type: "number"
          required: false

    - name: "export_completed"
      category: "product"
      description: "User exported data from the platform"
      properties:
        - name: "export_format"
          type: "enum"
          values: ["csv", "json", "pdf", "xlsx"]
          required: true
        - name: "row_count"
          type: "number"
          required: true

    # ─── Conversion Events ────────────────────────────
    - name: "checkout_started"
      category: "conversion"
      critical_funnel: true
      properties:
        - name: "plan_name"
          type: "string"
          required: true
        - name: "plan_price"
          type: "number"
          required: true
        - name: "billing_period"
          type: "enum"
          values: ["monthly", "annual"]
          required: true

    - name: "subscription_created"
      category: "conversion"
      critical_funnel: true
      properties:
        - name: "plan_name"
          type: "string"
          required: true
        - name: "mrr"
          type: "number"
          required: true

  validation_rules:
    naming_convention: "snake_case"
    max_property_count: 20
    require_descriptions: true
    volume_anomaly_threshold: 2.5  # standard deviations`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description:
              'Use ChatGPT or Claude to compare your actual events against the tracking plan, identify gaps, validate property schemas, and generate prioritized remediation reports.',
            toolsUsed: ['ChatGPT Plus', 'Claude Pro'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Event Tracking Audit Prompt Template',
                description:
                  'Structured prompt for ChatGPT/Claude to audit analytics events against the tracking plan and identify implementation issues.',
                code: `# Analytics Event Audit Prompt Template
# Use with ChatGPT Plus or Claude Pro via Zapier

system_prompt: |
  You are a senior analytics engineer specializing in event tracking audits.
  Your role is to identify gaps, inconsistencies, and issues in analytics implementations.
  Always respond in valid JSON format with actionable recommendations.

user_prompt_template: |
  Audit the following analytics implementation against the tracking plan.

  ## Tracking Plan (Expected Events):
  {{tracking_plan_json}}

  ## Actual Events (From Analytics Platform):
  {{actual_events_json}}

  ## Perform These Checks:

  ### 1. Coverage Analysis
  - Which expected events are MISSING from actual implementation?
  - Which actual events are NOT in the tracking plan (unexpected)?
  - What is the overall coverage percentage?

  ### 2. Property Validation
  For each event present in both:
  - Are all REQUIRED properties present?
  - Are property TYPES correct?
  - Are there unexpected properties not in the plan?

  ### 3. Volume Anomaly Detection
  Using the expected_daily_volume ranges:
  - Which events are significantly under-firing?
  - Which events are significantly over-firing?
  - Flag any events with zero volume in the last 7 days

  ### 4. Critical Funnel Analysis
  For events marked critical_funnel=true:
  - Are all critical events present and firing correctly?
  - What is the drop-off between funnel steps?

  ## Response Format (JSON):
  {
    "audit_summary": {
      "coverage_percentage": number,
      "missing_events": ["event1", "event2"],
      "unexpected_events": ["event3"],
      "events_with_issues": number,
      "critical_issues": number
    },
    "event_issues": [
      {
        "event_name": "string",
        "issue_type": "missing_property|type_mismatch|volume_anomaly|not_in_plan",
        "severity": "critical|warning|info",
        "details": "string",
        "recommendation": "string"
      }
    ],
    "volume_anomalies": [
      {
        "event_name": "string",
        "expected_range": "min-max",
        "actual_volume": number,
        "deviation": "under_firing|over_firing|zero_volume"
      }
    ],
    "funnel_analysis": {
      "funnel_events": ["step1", "step2"],
      "conversion_rates": [number],
      "issues": ["issue description"]
    },
    "prioritized_actions": [
      {
        "priority": 1,
        "action": "string",
        "impact": "high|medium|low",
        "effort": "high|medium|low"
      }
    ]
  }

model_settings:
  model: "gpt-4-turbo"
  temperature: 0.2
  max_tokens: 3000`,
              },
              {
                language: 'yaml',
                title: 'Weekly Analytics Health Report Prompt',
                description:
                  'Prompt template for generating a weekly analytics health report with trends and recommendations.',
                code: `# Weekly Analytics Health Report Prompt
# Generates executive summary of tracking implementation health

system_prompt: |
  You are an analytics operations expert creating a weekly health report.
  Focus on actionable insights, trends, and clear recommendations.
  Present data in a format suitable for product and engineering leadership.

user_prompt_template: |
  Generate a weekly analytics health report from the audit data.

  ## This Week's Audit Results:
  {{current_audit_json}}

  ## Last Week's Audit Results (for trend comparison):
  {{previous_audit_json}}

  ## Event Volume Time Series (7 days):
  {{volume_timeseries_json}}

  ## Generate Report Sections:

  ### 1. Executive Summary (3 bullet points max)
  - Overall health score (0-100)
  - Key improvements or regressions from last week
  - Most urgent issue requiring attention

  ### 2. Coverage Scorecard
  | Metric | This Week | Last Week | Trend |
  |--------|-----------|-----------|-------|
  | Events Covered | X/Y | | |
  | Properties Valid | X% | | |
  | Critical Funnels OK | X/Y | | |

  ### 3. Issues by Severity
  - CRITICAL (blocking): List with event names
  - WARNING (degraded): List with event names
  - INFO (minor): Count only

  ### 4. Volume Anomalies
  Events with unusual volume patterns that may indicate tracking issues:
  - Under-firing (potential broken tracking)
  - Over-firing (potential duplicate events)
  - New events not in plan

  ### 5. Week-over-Week Trends
  - Events that improved
  - Events that degraded
  - New issues introduced

  ### 6. Recommended Actions (Top 5)
  Prioritized list of fixes with:
  - Event/property affected
  - Issue description
  - Suggested fix
  - Estimated effort (hours)

  ### 7. Funnel Health
  For each critical conversion funnel:
  - Current conversion rate
  - Change from last week
  - Any data quality issues affecting accuracy

  Keep the report concise and actionable.

schedule: "Every Monday 9:00 AM"
output_format: "markdown"
recipients: ["product-analytics@company.com", "#analytics-ops"]`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description:
              'Automate the complete audit workflow with Zapier: run daily schema validation, generate weekly health reports, and alert on critical tracking failures.',
            toolsUsed: ['Zapier', 'Slack', 'Email', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Analytics Audit Automation Workflow',
                description:
                  'Complete Zapier workflow that runs daily analytics audits using AI and stores results for trend analysis.',
                code: `{
  "workflow_name": "Daily Analytics Audit Pipeline",
  "trigger": {
    "type": "schedule",
    "schedule": "every_day_6am",
    "timezone": "America/New_York"
  },
  "steps": [
    {
      "step": 1,
      "name": "fetch_actual_events",
      "app": "Webhooks by Zapier",
      "action": "Custom Request",
      "config": {
        "method": "GET",
        "url": "https://amplitude.com/api/2/events/list",
        "headers": {
          "Authorization": "Basic {{AMPLITUDE_API_CREDENTIALS}}"
        }
      }
    },
    {
      "step": 2,
      "name": "fetch_event_volumes",
      "app": "Webhooks by Zapier",
      "action": "Custom Request",
      "config": {
        "method": "GET",
        "url": "https://amplitude.com/api/2/events/segmentation",
        "body": {
          "e": {"event_type": "_all"},
          "start": "{{7_days_ago}}",
          "end": "{{today}}",
          "m": "totals",
          "g": "event_type"
        }
      }
    },
    {
      "step": 3,
      "name": "fetch_tracking_plan",
      "app": "Google Sheets",
      "action": "Get Many Rows",
      "config": {
        "spreadsheet_id": "{{TRACKING_PLAN_SHEET_ID}}",
        "worksheet": "Events"
      }
    },
    {
      "step": 4,
      "name": "format_for_audit",
      "app": "Code by Zapier",
      "action": "Run Javascript",
      "config": {
        "code": "const actual = JSON.parse(inputData.actualEvents); const plan = inputData.trackingPlan; const volumes = JSON.parse(inputData.volumes); return { actual_events: actual.data.map(e => ({name: e.event_type, properties: e.event_properties, volume: volumes.data[e.event_type] || 0})), tracking_plan: plan.map(row => ({name: row.event_name, properties: JSON.parse(row.properties || '[]'), required: row.required === 'TRUE', expected_volume: {min: row.min_volume, max: row.max_volume}})) };"
      }
    },
    {
      "step": 5,
      "name": "run_ai_audit",
      "app": "ChatGPT",
      "action": "Conversation",
      "config": {
        "model": "gpt-4-turbo",
        "system_message": "You are an analytics engineer. Audit the events and respond only in valid JSON.",
        "user_message": "Compare these actual events against the tracking plan. Identify: 1) missing events, 2) property issues, 3) volume anomalies. Actual: {{step4.actual_events}} Plan: {{step4.tracking_plan}}",
        "temperature": 0.2
      }
    },
    {
      "step": 6,
      "name": "parse_audit_results",
      "app": "Code by Zapier",
      "action": "Run Javascript",
      "config": {
        "code": "const audit = JSON.parse(inputData.aiResponse); audit.audit_date = new Date().toISOString().split('T')[0]; audit.critical_count = audit.event_issues.filter(i => i.severity === 'critical').length; return audit;"
      }
    },
    {
      "step": 7,
      "name": "store_audit_results",
      "app": "Google Sheets",
      "action": "Create Spreadsheet Row",
      "config": {
        "spreadsheet_id": "{{AUDIT_HISTORY_SHEET_ID}}",
        "worksheet": "Audit History",
        "row": {
          "date": "{{step6.audit_date}}",
          "coverage_pct": "{{step6.audit_summary.coverage_percentage}}",
          "missing_events": "{{step6.audit_summary.missing_events.length}}",
          "issues_count": "{{step6.event_issues.length}}",
          "critical_count": "{{step6.critical_count}}",
          "full_results": "{{step5.response}}"
        }
      }
    },
    {
      "step": 8,
      "name": "check_critical_issues",
      "app": "Filter by Zapier",
      "config": {
        "condition": "{{step6.critical_count}} > 0"
      }
    },
    {
      "step": 9,
      "name": "alert_on_critical",
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#analytics-alerts",
        "message": ":rotating_light: *Analytics Audit Alert*\\n\\nCritical issues detected: {{step6.critical_count}}\\n\\n*Issues:*\\n{{step6.event_issues.filter(i => i.severity === 'critical').map(i => '- ' + i.event_name + ': ' + i.details).join('\\n')}}\\n\\n<{{AUDIT_DASHBOARD_URL}}|View Full Audit Report>"
      },
      "run_if": "step8_passed"
    }
  ]
}`,
              },
              {
                language: 'json',
                title: 'Weekly Analytics Health Report Delivery',
                description:
                  'Zapier workflow that generates and delivers the weekly analytics health report to stakeholders.',
                code: `{
  "workflow_name": "Weekly Analytics Health Report",
  "trigger": {
    "type": "schedule",
    "schedule": "every_monday_9am",
    "timezone": "America/New_York"
  },
  "steps": [
    {
      "step": 1,
      "name": "fetch_this_week_audits",
      "app": "Google Sheets",
      "action": "Get Many Rows",
      "config": {
        "spreadsheet_id": "{{AUDIT_HISTORY_SHEET_ID}}",
        "worksheet": "Audit History",
        "filter": {
          "date": { "gte": "{{7_days_ago}}" }
        }
      }
    },
    {
      "step": 2,
      "name": "fetch_last_week_audits",
      "app": "Google Sheets",
      "action": "Get Many Rows",
      "config": {
        "spreadsheet_id": "{{AUDIT_HISTORY_SHEET_ID}}",
        "worksheet": "Audit History",
        "filter": {
          "date": { "gte": "{{14_days_ago}}", "lt": "{{7_days_ago}}" }
        }
      }
    },
    {
      "step": 3,
      "name": "calculate_trends",
      "app": "Code by Zapier",
      "action": "Run Javascript",
      "config": {
        "code": "const thisWeek = inputData.thisWeekAudits; const lastWeek = inputData.lastWeekAudits; const avgCoverage = arr => arr.reduce((sum, a) => sum + parseFloat(a.coverage_pct), 0) / arr.length; const avgIssues = arr => arr.reduce((sum, a) => sum + parseInt(a.issues_count), 0) / arr.length; return { this_week_avg_coverage: avgCoverage(thisWeek).toFixed(1), last_week_avg_coverage: avgCoverage(lastWeek).toFixed(1), coverage_trend: avgCoverage(thisWeek) > avgCoverage(lastWeek) ? 'improved' : 'declined', this_week_avg_issues: avgIssues(thisWeek).toFixed(0), last_week_avg_issues: avgIssues(lastWeek).toFixed(0), latest_audit: thisWeek[thisWeek.length - 1] };"
      }
    },
    {
      "step": 4,
      "name": "generate_report",
      "app": "ChatGPT",
      "action": "Conversation",
      "config": {
        "model": "gpt-4-turbo",
        "user_message": "Generate a concise weekly analytics health report.\\n\\nMetrics:\\n- Coverage: {{step3.this_week_avg_coverage}}% (was {{step3.last_week_avg_coverage}}%)\\n- Avg issues/day: {{step3.this_week_avg_issues}} (was {{step3.last_week_avg_issues}})\\n- Trend: {{step3.coverage_trend}}\\n\\nLatest audit details: {{step3.latest_audit.full_results}}\\n\\nFormat as Slack-friendly markdown with emoji. Include: 1) Health score, 2) Key wins/losses, 3) Top 3 action items."
      }
    },
    {
      "step": 5,
      "name": "post_to_slack",
      "app": "Slack",
      "action": "Send Channel Message",
      "config": {
        "channel": "#product-analytics",
        "message": ":bar_chart: *Weekly Analytics Health Report*\\n_Week of {{current_week_start}}_\\n\\n{{step4.response}}\\n\\n---\\n:link: <{{AUDIT_DASHBOARD_URL}}|View Full Dashboard> | <{{TRACKING_PLAN_URL}}|Tracking Plan>"
      }
    },
    {
      "step": 6,
      "name": "email_stakeholders",
      "app": "Email by Zapier",
      "action": "Send Outbound Email",
      "config": {
        "to": "analytics-team@company.com, product-leadership@company.com",
        "subject": "Weekly Analytics Health Report - {{current_date}}",
        "body_type": "html",
        "body": "<h2>Weekly Analytics Health Report</h2><p>Week of {{current_week_start}}</p>{{step4.response | markdown_to_html}}<hr><p><a href='{{AUDIT_DASHBOARD_URL}}'>View Full Dashboard</a></p>"
      }
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
          'Deploy a multi-agent system using CrewAI and LangGraph that continuously validates your analytics implementation, automatically detects schema drift and tracking regressions, enforces your tracking plan contract in CI/CD, and provides real-time anomaly detection with intelligent alerting.',
        estimatedMonthlyCost: '$600 - $1,500/month',
        architecture:
          'Supervisor agent coordinates specialized agents for schema validation, volume monitoring, CI integration, and alerting. The system integrates with your CI/CD pipeline to block deployments with broken analytics and maintains continuous production monitoring.',
        agents: [
          {
            name: 'SchemaValidatorAgent',
            role: 'Tracking Plan Enforcer',
            goal: 'Validate all analytics events against the tracking plan schema, detecting property mismatches, type errors, and naming violations',
            tools: ['TrackingPlanLoader', 'SchemaValidator', 'EventFetcher', 'DiffGenerator'],
          },
          {
            name: 'VolumeMonitorAgent',
            role: 'Anomaly Detector',
            goal: 'Monitor event volumes in real-time, detect anomalies using statistical methods, and identify tracking regressions before they impact dashboards',
            tools: ['VolumeAnalyzer', 'AnomalyDetector', 'TrendCalculator', 'BaselineManager'],
          },
          {
            name: 'CIEnforcementAgent',
            role: 'Deployment Gatekeeper',
            goal: 'Analyze code changes for analytics impact, validate new tracking calls against the plan, and block deployments with schema violations',
            tools: ['ASTParser', 'CodeAnalyzer', 'GitHubAPI', 'CIStatusUpdater'],
          },
          {
            name: 'AlertingAgent',
            role: 'Incident Responder',
            goal: 'Triage analytics issues by severity, route alerts to appropriate teams, and provide intelligent context for faster resolution',
            tools: ['AlertRouter', 'SlackNotifier', 'PagerDutyIntegration', 'IncidentTracker'],
          },
          {
            name: 'ReportingAgent',
            role: 'Analytics Analyst',
            goal: 'Generate comprehensive audit reports, track health trends over time, and provide recommendations for improving tracking coverage',
            tools: ['ReportGenerator', 'TrendAnalyzer', 'CoverageCalculator', 'RecommendationEngine'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with 5-minute snapshots and 30-day retention for trend analysis',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description:
              'Define the multi-agent system architecture using CrewAI with specialized agents for schema validation, volume monitoring, CI enforcement, alerting, and reporting.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'CrewAI Analytics Validation Agent Definitions',
                description:
                  'Define the specialized agents for the analytics validation system using CrewAI, including their roles, goals, and tools.',
                code: `"""
Analytics Validation Multi-Agent System
Agent definitions using CrewAI framework
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass, field

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class AnalyticsAgentConfig:
    """Configuration for an analytics validation agent."""
    name: str
    role: str
    goal: str
    backstory: str
    tools: list[Tool] = field(default_factory=list)
    verbose: bool = True
    allow_delegation: bool = False
    max_iterations: int = 15


class AnalyticsAgentFactory:
    """Factory for creating specialized analytics validation agents."""

    def __init__(self, llm: Optional[ChatOpenAI] = None) -> None:
        self.llm = llm or ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.1,
        )
        logger.info("AnalyticsAgentFactory initialized with model: %s", self.llm.model_name)

    def create_schema_validator_agent(self, tools: list[Tool]) -> Agent:
        """Create the schema validation agent."""
        config = AnalyticsAgentConfig(
            name="SchemaValidatorAgent",
            role="Tracking Plan Enforcer",
            goal="Validate all analytics events against the tracking plan, detecting schema violations and drift",
            backstory="""You are a meticulous analytics engineer who has seen countless broken dashboards
            caused by tracking implementation issues. You validate every event property against the tracking
            plan schema, enforce naming conventions, and catch type mismatches before they corrupt data.
            Your mantra: 'Trust, but verify every single event.'""",
            tools=tools,
            allow_delegation=True,
        )
        return self._create_agent(config)

    def create_volume_monitor_agent(self, tools: list[Tool]) -> Agent:
        """Create the volume monitoring agent."""
        config = AnalyticsAgentConfig(
            name="VolumeMonitorAgent",
            role="Anomaly Detector",
            goal="Monitor event volumes, detect anomalies, and identify tracking regressions in real-time",
            backstory="""You are a statistician specializing in time series anomaly detection for analytics
            pipelines. You've prevented dozens of data quality incidents by catching volume drops before
            anyone noticed the dashboards were wrong. You use robust statistical methods (MAD, Z-scores,
            CUSUM) to distinguish real issues from normal variance.""",
            tools=tools,
        )
        return self._create_agent(config)

    def create_ci_enforcement_agent(self, tools: list[Tool]) -> Agent:
        """Create the CI enforcement agent."""
        config = AnalyticsAgentConfig(
            name="CIEnforcementAgent",
            role="Deployment Gatekeeper",
            goal="Analyze code changes for analytics impact and block deployments with tracking violations",
            backstory="""You are a senior platform engineer who has built CI/CD pipelines for analytics-heavy
            applications. You've seen the chaos that broken tracking causes - missed revenue attribution,
            incorrect A/B test results, angry product managers. Now you guard the gates, analyzing every
            PR for analytics changes and ensuring they match the tracking plan before merge.""",
            tools=tools,
        )
        return self._create_agent(config)

    def create_alerting_agent(self, tools: list[Tool]) -> Agent:
        """Create the alerting agent."""
        config = AnalyticsAgentConfig(
            name="AlertingAgent",
            role="Incident Responder",
            goal="Triage analytics issues by severity, route alerts intelligently, and provide actionable context",
            backstory="""You are an SRE with deep experience in observability and incident management for
            data systems. You know that alert fatigue is real, so you carefully triage issues - only
            paging for true emergencies, batching warnings, and always providing the context needed
            for fast resolution. Every alert you send includes: what broke, why it matters, and how to fix it.""",
            tools=tools,
        )
        return self._create_agent(config)

    def create_reporting_agent(self, tools: list[Tool]) -> Agent:
        """Create the reporting agent."""
        config = AnalyticsAgentConfig(
            name="ReportingAgent",
            role="Analytics Analyst",
            goal="Generate comprehensive audit reports, track health trends, and recommend improvements",
            backstory="""You are a product analytics lead who has built analytics practices at multiple
            companies. You understand that tracking quality directly impacts product decisions and revenue.
            Your reports are clear, actionable, and always tie issues to business impact. You track trends
            over time and celebrate wins when coverage improves.""",
            tools=tools,
            allow_delegation=True,
        )
        return self._create_agent(config)

    def _create_agent(self, config: AnalyticsAgentConfig) -> Agent:
        """Create an agent from configuration."""
        logger.info("Creating agent: %s (%s)", config.name, config.role)
        return Agent(
            role=config.role,
            goal=config.goal,
            backstory=config.backstory,
            tools=config.tools,
            llm=self.llm,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation,
            max_iter=config.max_iterations,
        )


def create_analytics_validation_crew(
    agent_factory: AnalyticsAgentFactory,
    tool_registry: dict[str, list[Tool]],
) -> Crew:
    """Create the complete analytics validation crew."""

    # Create agents with their specialized tools
    schema_agent = agent_factory.create_schema_validator_agent(tool_registry.get("schema", []))
    volume_agent = agent_factory.create_volume_monitor_agent(tool_registry.get("volume", []))
    ci_agent = agent_factory.create_ci_enforcement_agent(tool_registry.get("ci", []))
    alerting_agent = agent_factory.create_alerting_agent(tool_registry.get("alerting", []))
    reporting_agent = agent_factory.create_reporting_agent(tool_registry.get("reporting", []))

    # Define tasks
    schema_validation_task = Task(
        description="Validate all events from the last 24 hours against the tracking plan schema",
        expected_output="List of schema violations with event name, property, expected vs actual, and severity",
        agent=schema_agent,
    )

    volume_monitoring_task = Task(
        description="Analyze event volumes for the last 24 hours, detect anomalies using statistical baselines",
        expected_output="List of volume anomalies with event name, expected range, actual volume, and deviation type",
        agent=volume_agent,
    )

    alerting_task = Task(
        description="Triage all detected issues, determine severity, and send appropriate alerts",
        expected_output="Summary of alerts sent with severity levels and recipients",
        agent=alerting_agent,
        context=[schema_validation_task, volume_monitoring_task],
    )

    reporting_task = Task(
        description="Generate a comprehensive analytics health report with trends and recommendations",
        expected_output="Formatted report with health score, issues by severity, trends, and prioritized actions",
        agent=reporting_agent,
        context=[schema_validation_task, volume_monitoring_task, alerting_task],
    )

    crew = Crew(
        agents=[schema_agent, volume_agent, ci_agent, alerting_agent, reporting_agent],
        tasks=[schema_validation_task, volume_monitoring_task, alerting_task, reporting_task],
        process=Process.sequential,
        verbose=True,
    )

    logger.info("Analytics validation crew created with %d agents and %d tasks", len(crew.agents), len(crew.tasks))
    return crew`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description:
              'Implement the schema validation and event fetching agents with connectors to analytics platforms (Amplitude, Mixpanel, Segment) and tracking plan storage.',
            toolsUsed: ['LangChain', 'Amplitude API', 'Mixpanel API', 'Segment API'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Analytics Event Fetching and Schema Validation Tools',
                description:
                  'LangChain tools for fetching events from analytics platforms and validating against tracking plan schemas.',
                code: `"""
Analytics Event Fetching and Schema Validation Tools
Provides connectors to analytics platforms and schema validation logic
"""

import json
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

import requests
from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class EventSchema:
    """Schema definition for an analytics event."""
    name: str
    properties: dict[str, dict[str, Any]]  # property_name -> {type, required, enum_values}
    critical_funnel: bool = False
    expected_volume_min: Optional[int] = None
    expected_volume_max: Optional[int] = None


@dataclass
class TrackingPlan:
    """Complete tracking plan with all event schemas."""
    version: str
    events: dict[str, EventSchema]
    global_properties: dict[str, dict[str, Any]]
    naming_convention: str = "snake_case"


@dataclass
class SchemaViolation:
    """A detected schema violation."""
    event_name: str
    violation_type: str  # missing_property, type_mismatch, naming_violation, unexpected_property
    property_name: Optional[str]
    expected: str
    actual: str
    severity: str  # critical, warning, info
    sample_count: int = 1


class TrackingPlanLoader:
    """Loads and parses tracking plan from various sources."""

    def __init__(self) -> None:
        logger.info("TrackingPlanLoader initialized")

    def load_from_file(self, path: str) -> TrackingPlan:
        """Load tracking plan from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        events = {}
        for event_data in data.get("events", []):
            props = {}
            for prop in event_data.get("properties", []):
                props[prop["name"]] = {
                    "type": prop.get("type", "string"),
                    "required": prop.get("required", False),
                    "enum_values": prop.get("values"),
                }

            events[event_data["name"]] = EventSchema(
                name=event_data["name"],
                properties=props,
                critical_funnel=event_data.get("critical_funnel", False),
                expected_volume_min=event_data.get("expected_daily_volume", {}).get("min"),
                expected_volume_max=event_data.get("expected_daily_volume", {}).get("max"),
            )

        global_props = {}
        for prop in data.get("global_properties", []):
            global_props[prop["name"]] = {
                "type": prop.get("type", "string"),
                "required": prop.get("required", False),
            }

        return TrackingPlan(
            version=data.get("version", "1.0"),
            events=events,
            global_properties=global_props,
            naming_convention=data.get("validation_rules", {}).get("naming_convention", "snake_case"),
        )

    def as_langchain_tool(self) -> StructuredTool:
        """Convert to LangChain tool."""
        return StructuredTool.from_function(
            func=lambda path: self.load_from_file(path).__dict__,
            name="load_tracking_plan",
            description="Load the tracking plan schema from a file",
        )


class AmplitudeEventFetcherInput(BaseModel):
    """Input schema for Amplitude event fetcher."""
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    event_types: Optional[list[str]] = Field(default=None, description="Specific events to fetch, or None for all")


class AmplitudeEventFetcher:
    """Fetches events and their properties from Amplitude."""

    def __init__(self, api_key: str, secret_key: str) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://amplitude.com/api/2"
        logger.info("AmplitudeEventFetcher initialized")

    def fetch_event_schemas(self) -> list[dict[str, Any]]:
        """Fetch all event types and their property schemas from Amplitude."""
        try:
            response = requests.get(
                f"{self.base_url}/taxonomy/event",
                auth=(self.api_key, self.secret_key),
                timeout=30,
            )
            response.raise_for_status()
            events = response.json().get("data", [])

            result = []
            for event in events:
                result.append({
                    "name": event["event_type"],
                    "properties": list(event.get("event_properties", {}).keys()),
                    "first_seen": event.get("first_seen"),
                    "last_seen": event.get("last_seen"),
                    "volume": event.get("totals", 0),
                })

            logger.info("Fetched %d event schemas from Amplitude", len(result))
            return result

        except requests.RequestException as e:
            logger.error("Failed to fetch Amplitude events: %s", e)
            return []

    def fetch_event_volumes(self, start_date: str, end_date: str) -> dict[str, int]:
        """Fetch event volumes for a date range."""
        try:
            response = requests.get(
                f"{self.base_url}/events/segmentation",
                auth=(self.api_key, self.secret_key),
                params={
                    "e": json.dumps({"event_type": "_all"}),
                    "start": start_date.replace("-", ""),
                    "end": end_date.replace("-", ""),
                    "m": "totals",
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json().get("data", {})

            volumes = {}
            for series in data.get("series", []):
                event_name = series.get("event_type", "unknown")
                volumes[event_name] = sum(series.get("values", [0]))

            logger.info("Fetched volumes for %d events", len(volumes))
            return volumes

        except requests.RequestException as e:
            logger.error("Failed to fetch Amplitude volumes: %s", e)
            return {}

    def as_langchain_tool(self) -> StructuredTool:
        """Convert to LangChain tool."""
        return StructuredTool.from_function(
            func=lambda: self.fetch_event_schemas(),
            name="amplitude_fetch_event_schemas",
            description="Fetch all event schemas and their properties from Amplitude",
        )


class SchemaValidatorInput(BaseModel):
    """Input schema for schema validator."""
    actual_events: list[dict[str, Any]] = Field(description="List of actual events from analytics platform")
    tracking_plan: dict[str, Any] = Field(description="Tracking plan schema to validate against")


class SchemaValidator:
    """Validates events against tracking plan schema."""

    SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

    TYPE_MAP = {
        "string": str,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def __init__(self) -> None:
        logger.info("SchemaValidator initialized")

    def validate(
        self,
        actual_events: list[dict[str, Any]],
        tracking_plan: TrackingPlan,
    ) -> list[SchemaViolation]:
        """Validate actual events against the tracking plan."""
        violations: list[SchemaViolation] = []

        actual_event_names = {e["name"] for e in actual_events}
        expected_event_names = set(tracking_plan.events.keys())

        # Check for missing events
        missing_events = expected_event_names - actual_event_names
        for event_name in missing_events:
            schema = tracking_plan.events[event_name]
            violations.append(SchemaViolation(
                event_name=event_name,
                violation_type="missing_event",
                property_name=None,
                expected="Event should be tracked",
                actual="Event not found in last 24 hours",
                severity="critical" if schema.critical_funnel else "warning",
            ))

        # Check for unexpected events
        unexpected_events = actual_event_names - expected_event_names
        for event_name in unexpected_events:
            if not self.SNAKE_CASE_RE.match(event_name):
                violations.append(SchemaViolation(
                    event_name=event_name,
                    violation_type="naming_violation",
                    property_name=None,
                    expected="snake_case",
                    actual=event_name,
                    severity="warning",
                ))

            violations.append(SchemaViolation(
                event_name=event_name,
                violation_type="unexpected_event",
                property_name=None,
                expected="Event should be in tracking plan",
                actual="Event not defined in plan",
                severity="info",
            ))

        # Validate properties for events in both
        for actual_event in actual_events:
            event_name = actual_event["name"]
            if event_name not in tracking_plan.events:
                continue

            schema = tracking_plan.events[event_name]
            actual_props = set(actual_event.get("properties", []))
            expected_props = set(schema.properties.keys())

            # Check missing required properties
            for prop_name, prop_def in schema.properties.items():
                if prop_def.get("required") and prop_name not in actual_props:
                    violations.append(SchemaViolation(
                        event_name=event_name,
                        violation_type="missing_property",
                        property_name=prop_name,
                        expected=f"Required property of type {prop_def.get('type', 'string')}",
                        actual="Property not found",
                        severity="critical" if schema.critical_funnel else "warning",
                    ))

            # Check unexpected properties
            for prop_name in actual_props:
                if prop_name not in expected_props and prop_name not in tracking_plan.global_properties:
                    violations.append(SchemaViolation(
                        event_name=event_name,
                        violation_type="unexpected_property",
                        property_name=prop_name,
                        expected="Property should be in schema",
                        actual="Property not defined",
                        severity="info",
                    ))

        logger.info("Validation complete: %d violations found", len(violations))
        return violations

    def as_langchain_tool(self) -> StructuredTool:
        """Convert to LangChain tool."""
        def _validate(actual_events: list[dict], tracking_plan: dict) -> list[dict]:
            # Convert dict to TrackingPlan
            events = {}
            for name, event_data in tracking_plan.get("events", {}).items():
                events[name] = EventSchema(
                    name=name,
                    properties=event_data.get("properties", {}),
                    critical_funnel=event_data.get("critical_funnel", False),
                )

            plan = TrackingPlan(
                version=tracking_plan.get("version", "1.0"),
                events=events,
                global_properties=tracking_plan.get("global_properties", {}),
            )

            violations = self.validate(actual_events, plan)
            return [v.__dict__ for v in violations]

        return StructuredTool.from_function(
            func=_validate,
            name="validate_event_schemas",
            description="Validate actual events against the tracking plan schema",
            args_schema=SchemaValidatorInput,
        )


def create_schema_validation_tools(config: dict[str, Any]) -> list[Tool]:
    """Create all schema validation tools from configuration."""
    tools: list[Tool] = []

    # Tracking plan loader
    plan_loader = TrackingPlanLoader()
    tools.append(plan_loader.as_langchain_tool())

    # Amplitude fetcher
    if config.get("amplitude_api_key"):
        amplitude = AmplitudeEventFetcher(
            config["amplitude_api_key"],
            config["amplitude_secret_key"],
        )
        tools.append(amplitude.as_langchain_tool())

    # Schema validator
    validator = SchemaValidator()
    tools.append(validator.as_langchain_tool())

    logger.info("Created %d schema validation tools", len(tools))
    return tools`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description:
              'Implement the volume monitoring and anomaly detection agents with statistical baseline computation, trend analysis, and intelligent alerting logic.',
            toolsUsed: ['LangChain', 'NumPy', 'SciPy'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Volume Monitoring and Anomaly Detection Tools',
                description:
                  'Tools for statistical volume monitoring, anomaly detection, and trend analysis.',
                code: `"""
Volume Monitoring and Anomaly Detection Tools
Provides statistical analysis for event volume monitoring
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

import numpy as np
from scipy import stats
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of volume anomalies."""
    VOLUME_DROP = "volume_drop"
    VOLUME_SPIKE = "volume_spike"
    FLATLINE = "flatline"
    GRADUAL_DECLINE = "gradual_decline"
    MISSING_DATA = "missing_data"


@dataclass
class VolumeAnomaly:
    """A detected volume anomaly."""
    event_name: str
    anomaly_type: AnomalyType
    expected_volume: float
    actual_volume: float
    deviation_zscore: float
    deviation_pct: float
    severity: str  # critical, warning, info
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_critical(self) -> bool:
        """Check if this is a critical anomaly."""
        return self.severity == "critical" or abs(self.deviation_zscore) > 4.0


@dataclass
class VolumeBaseline:
    """Statistical baseline for an event's volume."""
    event_name: str
    median: float
    mad: float  # Median Absolute Deviation
    mean: float
    std: float
    min_observed: float
    max_observed: float
    sample_days: int
    last_updated: datetime


class VolumeBaselineManagerInput(BaseModel):
    """Input for baseline computation."""
    event_volumes: dict[str, list[float]] = Field(
        description="Dictionary mapping event names to list of daily volumes"
    )


class VolumeBaselineManager:
    """Computes and manages statistical baselines for event volumes."""

    def __init__(self, min_sample_days: int = 7) -> None:
        self.min_sample_days = min_sample_days
        self.baselines: dict[str, VolumeBaseline] = {}
        logger.info("VolumeBaselineManager initialized: min_sample_days=%d", min_sample_days)

    def compute_baseline(self, event_name: str, daily_volumes: list[float]) -> Optional[VolumeBaseline]:
        """Compute statistical baseline for an event."""
        if len(daily_volumes) < self.min_sample_days:
            logger.warning("Insufficient data for %s: %d days (need %d)",
                         event_name, len(daily_volumes), self.min_sample_days)
            return None

        arr = np.array(daily_volumes)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median))) * 1.4826  # Scaled MAD

        baseline = VolumeBaseline(
            event_name=event_name,
            median=median,
            mad=max(mad, 1.0),  # Prevent division by zero
            mean=float(np.mean(arr)),
            std=float(np.std(arr)) if len(arr) > 1 else 1.0,
            min_observed=float(np.min(arr)),
            max_observed=float(np.max(arr)),
            sample_days=len(daily_volumes),
            last_updated=datetime.now(timezone.utc),
        )

        self.baselines[event_name] = baseline
        logger.debug("Computed baseline for %s: median=%.1f, mad=%.1f", event_name, median, mad)
        return baseline

    def compute_all_baselines(self, event_volumes: dict[str, list[float]]) -> dict[str, VolumeBaseline]:
        """Compute baselines for all events."""
        results = {}
        for event_name, volumes in event_volumes.items():
            baseline = self.compute_baseline(event_name, volumes)
            if baseline:
                results[event_name] = baseline

        logger.info("Computed baselines for %d events", len(results))
        return results

    def as_langchain_tool(self) -> StructuredTool:
        """Convert to LangChain tool."""
        def _compute(event_volumes: dict[str, list[float]]) -> dict[str, dict]:
            baselines = self.compute_all_baselines(event_volumes)
            return {name: vars(b) for name, b in baselines.items()}

        return StructuredTool.from_function(
            func=_compute,
            name="compute_volume_baselines",
            description="Compute statistical baselines for event volumes",
            args_schema=VolumeBaselineManagerInput,
        )


class AnomalyDetectorInput(BaseModel):
    """Input for anomaly detection."""
    current_volumes: dict[str, float] = Field(description="Current day's volumes by event")
    baselines: dict[str, dict[str, Any]] = Field(description="Pre-computed baselines by event")
    z_threshold: float = Field(default=2.5, description="Z-score threshold for anomaly detection")


class AnomalyDetector:
    """Detects volume anomalies using statistical methods."""

    def __init__(self, z_threshold: float = 2.5, critical_z_threshold: float = 4.0) -> None:
        self.z_threshold = z_threshold
        self.critical_z_threshold = critical_z_threshold
        logger.info("AnomalyDetector initialized: z_threshold=%.1f", z_threshold)

    def detect_anomalies(
        self,
        current_volumes: dict[str, float],
        baselines: dict[str, VolumeBaseline],
        critical_events: Optional[set[str]] = None,
    ) -> list[VolumeAnomaly]:
        """Detect volume anomalies for all events."""
        anomalies: list[VolumeAnomaly] = []
        critical_events = critical_events or set()

        for event_name, baseline in baselines.items():
            actual = current_volumes.get(event_name, 0.0)

            # Calculate Z-score using robust statistics (MAD)
            z_score = (actual - baseline.median) / baseline.mad

            if abs(z_score) < self.z_threshold:
                continue

            # Determine anomaly type
            if actual < baseline.median * 0.05:
                anomaly_type = AnomalyType.FLATLINE
            elif z_score < -self.z_threshold:
                anomaly_type = AnomalyType.VOLUME_DROP
            else:
                anomaly_type = AnomalyType.VOLUME_SPIKE

            # Determine severity
            is_critical = event_name in critical_events
            if abs(z_score) > self.critical_z_threshold or (is_critical and abs(z_score) > self.z_threshold):
                severity = "critical"
            elif abs(z_score) > self.z_threshold:
                severity = "warning"
            else:
                severity = "info"

            deviation_pct = ((actual - baseline.median) / baseline.median * 100) if baseline.median > 0 else 0

            anomaly = VolumeAnomaly(
                event_name=event_name,
                anomaly_type=anomaly_type,
                expected_volume=baseline.median,
                actual_volume=actual,
                deviation_zscore=round(z_score, 2),
                deviation_pct=round(deviation_pct, 1),
                severity=severity,
            )
            anomalies.append(anomaly)
            logger.warning("Anomaly detected: %s %s (z=%.2f, %+.1f%%)",
                         event_name, anomaly_type.value, z_score, deviation_pct)

        # Sort by severity and absolute deviation
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        anomalies.sort(key=lambda a: (severity_order[a.severity], -abs(a.deviation_zscore)))

        logger.info("Detected %d anomalies (%d critical)",
                   len(anomalies), sum(1 for a in anomalies if a.severity == "critical"))
        return anomalies

    def detect_gradual_decline(
        self,
        daily_volumes: list[float],
        event_name: str,
        decline_threshold: float = -0.15,
    ) -> Optional[VolumeAnomaly]:
        """Detect gradual volume decline using linear regression."""
        if len(daily_volumes) < 7:
            return None

        arr = np.array(daily_volumes)
        x = np.arange(len(arr))

        slope, _, r_value, _, _ = stats.linregress(x, arr)

        # Calculate percentage decline over the period
        if arr[0] > 0:
            total_decline_pct = (slope * len(arr)) / arr[0]
        else:
            total_decline_pct = 0

        # Check if there's a significant declining trend
        if total_decline_pct < decline_threshold and r_value**2 > 0.5:
            return VolumeAnomaly(
                event_name=event_name,
                anomaly_type=AnomalyType.GRADUAL_DECLINE,
                expected_volume=float(arr[0]),
                actual_volume=float(arr[-1]),
                deviation_zscore=0,  # Not applicable for trend analysis
                deviation_pct=round(total_decline_pct * 100, 1),
                severity="warning",
            )

        return None

    def as_langchain_tool(self) -> StructuredTool:
        """Convert to LangChain tool."""
        def _detect(current_volumes: dict[str, float], baselines: dict[str, dict], z_threshold: float = 2.5) -> list[dict]:
            self.z_threshold = z_threshold

            # Convert dict baselines back to VolumeBaseline objects
            baseline_objs = {}
            for name, b in baselines.items():
                baseline_objs[name] = VolumeBaseline(
                    event_name=name,
                    median=b["median"],
                    mad=b["mad"],
                    mean=b["mean"],
                    std=b["std"],
                    min_observed=b["min_observed"],
                    max_observed=b["max_observed"],
                    sample_days=b["sample_days"],
                    last_updated=datetime.fromisoformat(b["last_updated"]) if isinstance(b["last_updated"], str) else b["last_updated"],
                )

            anomalies = self.detect_anomalies(current_volumes, baseline_objs)
            return [
                {
                    "event_name": a.event_name,
                    "anomaly_type": a.anomaly_type.value,
                    "expected_volume": a.expected_volume,
                    "actual_volume": a.actual_volume,
                    "deviation_zscore": a.deviation_zscore,
                    "deviation_pct": a.deviation_pct,
                    "severity": a.severity,
                }
                for a in anomalies
            ]

        return StructuredTool.from_function(
            func=_detect,
            name="detect_volume_anomalies",
            description="Detect volume anomalies using statistical baseline comparison",
            args_schema=AnomalyDetectorInput,
        )


def create_volume_monitoring_tools() -> list[StructuredTool]:
    """Create volume monitoring tools."""
    baseline_manager = VolumeBaselineManager()
    anomaly_detector = AnomalyDetector()

    return [
        baseline_manager.as_langchain_tool(),
        anomaly_detector.as_langchain_tool(),
    ]`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description:
              'Implement the LangGraph state machine that orchestrates the analytics validation pipeline with schema validation, volume monitoring, alerting, and reporting stages.',
            toolsUsed: ['LangGraph', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph Analytics Validation Orchestrator',
                description:
                  'State machine implementation using LangGraph to orchestrate the analytics validation workflow.',
                code: `"""
LangGraph Orchestrator for Analytics Validation Pipeline
Manages validation workflow with schema checking, volume monitoring, and alerting
"""

import logging
from typing import Any, TypedDict, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import json

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import redis

logger = logging.getLogger(__name__)


class AnalyticsValidationState(TypedDict):
    """State schema for the analytics validation pipeline."""
    # Pipeline metadata
    run_id: str
    started_at: str
    current_stage: str
    error_count: int

    # Schema validation state
    tracking_plan: dict[str, Any]
    actual_events: list[dict[str, Any]]
    schema_violations: list[dict[str, Any]]
    schema_validation_complete: bool

    # Volume monitoring state
    current_volumes: dict[str, float]
    baselines: dict[str, dict[str, Any]]
    volume_anomalies: list[dict[str, Any]]
    volume_check_complete: bool

    # Alerting state
    alerts_sent: list[dict[str, Any]]
    alerting_complete: bool

    # Reporting state
    health_score: float
    report: str
    report_complete: bool

    # Messages
    messages: list[BaseMessage]


@dataclass
class RedisStateManager:
    """Manages validation pipeline state in Redis."""

    client: redis.Redis
    key_prefix: str = "analytics_validation"
    ttl_days: int = 30

    def save_state(self, run_id: str, state: AnalyticsValidationState) -> None:
        """Persist state to Redis."""
        key = f"{self.key_prefix}:{run_id}"
        serializable = {k: v for k, v in state.items() if k != "messages"}
        self.client.setex(key, self.ttl_days * 86400, json.dumps(serializable, default=str))

    def load_state(self, run_id: str) -> Optional[dict[str, Any]]:
        """Load state from Redis."""
        key = f"{self.key_prefix}:{run_id}"
        data = self.client.get(key)
        return json.loads(data) if data else None

    def get_recent_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent validation runs for trend analysis."""
        pattern = f"{self.key_prefix}:*"
        keys = sorted(self.client.keys(pattern), reverse=True)[:limit]
        runs = []
        for key in keys:
            data = self.client.get(key)
            if data:
                runs.append(json.loads(data))
        return runs


class AnalyticsValidationOrchestrator:
    """Orchestrates the analytics validation pipeline using LangGraph."""

    def __init__(
        self,
        schema_validator_agent: Any,
        volume_monitor_agent: Any,
        alerting_agent: Any,
        reporting_agent: Any,
        redis_client: Optional[redis.Redis] = None,
    ) -> None:
        self.schema_validator = schema_validator_agent
        self.volume_monitor = volume_monitor_agent
        self.alerting_agent = alerting_agent
        self.reporting_agent = reporting_agent
        self.state_manager = RedisStateManager(redis_client) if redis_client else None
        self.graph = self._build_graph()
        logger.info("AnalyticsValidationOrchestrator initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AnalyticsValidationState)

        # Add nodes
        workflow.add_node("load_data", self._load_data_node)
        workflow.add_node("validate_schema", self._validate_schema_node)
        workflow.add_node("check_volumes", self._check_volumes_node)
        workflow.add_node("send_alerts", self._send_alerts_node)
        workflow.add_node("generate_report", self._generate_report_node)
        workflow.add_node("handle_error", self._error_handler_node)

        # Define edges
        workflow.set_entry_point("load_data")

        workflow.add_conditional_edges(
            "load_data",
            self._should_continue,
            {"continue": "validate_schema", "error": "handle_error"}
        )

        workflow.add_edge("validate_schema", "check_volumes")
        workflow.add_edge("check_volumes", "send_alerts")
        workflow.add_edge("send_alerts", "generate_report")
        workflow.add_edge("generate_report", END)
        workflow.add_edge("handle_error", END)

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _load_data_node(self, state: AnalyticsValidationState) -> dict[str, Any]:
        """Load tracking plan and fetch current events."""
        logger.info("Loading data for validation run %s", state["run_id"])

        try:
            # Load tracking plan
            plan_result = self.schema_validator.execute(
                "Load the tracking plan from the configured path"
            )

            # Fetch current events from analytics platform
            events_result = self.schema_validator.execute(
                "Fetch all events and their schemas from the analytics platform"
            )

            # Fetch current volumes
            volumes_result = self.volume_monitor.execute(
                "Fetch today's event volumes from the analytics platform"
            )

            # Load historical volumes for baseline
            baseline_result = self.volume_monitor.execute(
                "Load or compute volume baselines from the last 28 days"
            )

            return {
                "tracking_plan": plan_result.get("tracking_plan", {}),
                "actual_events": events_result.get("events", []),
                "current_volumes": volumes_result.get("volumes", {}),
                "baselines": baseline_result.get("baselines", {}),
                "current_stage": "validate_schema",
            }

        except Exception as e:
            logger.error("Data loading failed: %s", e)
            return {"error_count": state["error_count"] + 1, "current_stage": "error"}

    def _validate_schema_node(self, state: AnalyticsValidationState) -> dict[str, Any]:
        """Run schema validation."""
        logger.info("Running schema validation for %s", state["run_id"])

        try:
            result = self.schema_validator.execute(
                "Validate all events against the tracking plan schema",
                context={
                    "actual_events": state["actual_events"],
                    "tracking_plan": state["tracking_plan"],
                },
            )

            violations = result.get("violations", [])
            logger.info("Schema validation complete: %d violations", len(violations))

            return {
                "schema_violations": violations,
                "schema_validation_complete": True,
                "current_stage": "check_volumes",
            }

        except Exception as e:
            logger.error("Schema validation failed: %s", e)
            return {"error_count": state["error_count"] + 1}

    def _check_volumes_node(self, state: AnalyticsValidationState) -> dict[str, Any]:
        """Run volume anomaly detection."""
        logger.info("Checking volumes for %s", state["run_id"])

        try:
            result = self.volume_monitor.execute(
                "Detect volume anomalies using statistical baselines",
                context={
                    "current_volumes": state["current_volumes"],
                    "baselines": state["baselines"],
                },
            )

            anomalies = result.get("anomalies", [])
            logger.info("Volume check complete: %d anomalies", len(anomalies))

            return {
                "volume_anomalies": anomalies,
                "volume_check_complete": True,
                "current_stage": "send_alerts",
            }

        except Exception as e:
            logger.error("Volume check failed: %s", e)
            return {"volume_anomalies": [], "volume_check_complete": True}

    def _send_alerts_node(self, state: AnalyticsValidationState) -> dict[str, Any]:
        """Send alerts for detected issues."""
        logger.info("Sending alerts for %s", state["run_id"])

        try:
            all_issues = state["schema_violations"] + state["volume_anomalies"]
            critical_issues = [i for i in all_issues if i.get("severity") == "critical"]

            if not all_issues:
                return {"alerts_sent": [], "alerting_complete": True, "current_stage": "generate_report"}

            result = self.alerting_agent.execute(
                "Triage and send alerts for detected analytics issues",
                context={
                    "schema_violations": state["schema_violations"],
                    "volume_anomalies": state["volume_anomalies"],
                    "critical_count": len(critical_issues),
                },
            )

            logger.info("Alerting complete: %d alerts sent", len(result.get("alerts_sent", [])))

            return {
                "alerts_sent": result.get("alerts_sent", []),
                "alerting_complete": True,
                "current_stage": "generate_report",
            }

        except Exception as e:
            logger.error("Alerting failed: %s", e)
            return {"alerts_sent": [], "alerting_complete": True}

    def _generate_report_node(self, state: AnalyticsValidationState) -> dict[str, Any]:
        """Generate the health report."""
        logger.info("Generating report for %s", state["run_id"])

        try:
            # Calculate health score
            total_events = len(state["actual_events"])
            violation_count = len(state["schema_violations"])
            anomaly_count = len(state["volume_anomalies"])
            critical_count = sum(
                1 for i in state["schema_violations"] + state["volume_anomalies"]
                if i.get("severity") == "critical"
            )

            # Health score: start at 100, deduct for issues
            health_score = 100.0
            health_score -= critical_count * 10  # -10 per critical
            health_score -= (violation_count - critical_count) * 2  # -2 per non-critical violation
            health_score -= anomaly_count * 3  # -3 per anomaly
            health_score = max(0, health_score)

            result = self.reporting_agent.execute(
                "Generate comprehensive analytics health report",
                context={
                    "health_score": health_score,
                    "total_events": total_events,
                    "schema_violations": state["schema_violations"],
                    "volume_anomalies": state["volume_anomalies"],
                    "alerts_sent": state["alerts_sent"],
                },
            )

            return {
                "health_score": health_score,
                "report": result.get("report", ""),
                "report_complete": True,
                "current_stage": "complete",
            }

        except Exception as e:
            logger.error("Report generation failed: %s", e)
            return {"health_score": 0, "report": f"Error: {e}", "report_complete": True}

    def _error_handler_node(self, state: AnalyticsValidationState) -> dict[str, Any]:
        """Handle pipeline errors."""
        logger.error("Pipeline error: %d errors occurred", state["error_count"])
        return {"current_stage": "failed"}

    def _should_continue(self, state: AnalyticsValidationState) -> str:
        """Determine if pipeline should continue."""
        if state.get("error_count", 0) > 2:
            return "error"
        return "continue"

    def run(self, run_id: Optional[str] = None) -> AnalyticsValidationState:
        """Execute the validation pipeline."""
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        initial_state: AnalyticsValidationState = {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "current_stage": "load_data",
            "error_count": 0,
            "tracking_plan": {},
            "actual_events": [],
            "schema_violations": [],
            "schema_validation_complete": False,
            "current_volumes": {},
            "baselines": {},
            "volume_anomalies": [],
            "volume_check_complete": False,
            "alerts_sent": [],
            "alerting_complete": False,
            "health_score": 0.0,
            "report": "",
            "report_complete": False,
            "messages": [HumanMessage(content=f"Starting analytics validation: {run_id}")],
        }

        logger.info("Starting validation pipeline: %s", run_id)
        final_state = self.graph.invoke(initial_state, {"configurable": {"thread_id": run_id}})

        if self.state_manager:
            self.state_manager.save_state(run_id, final_state)

        logger.info("Pipeline %s complete. Health score: %.1f", run_id, final_state["health_score"])
        return final_state`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description:
              'Deploy the analytics validation system with Docker, CI/CD integration for deployment blocking, and comprehensive observability with LangSmith tracing and Prometheus metrics.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus', 'GitHub Actions'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'GitHub Actions CI Integration',
                description:
                  'GitHub Actions workflow that runs analytics validation on every PR and blocks merges with schema violations.',
                code: `# .github/workflows/analytics-validation.yml
# Run analytics schema validation on every PR

name: Analytics Schema Validation

on:
  pull_request:
    paths:
      - 'src/**/*.ts'
      - 'src/**/*.tsx'
      - 'app/**/*.py'
      - 'analytics/**'

jobs:
  validate-analytics:
    name: Validate Analytics Implementation
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for diff

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements-analytics.txt
          pip install ast-comments

      - name: Extract analytics calls from changed files
        id: extract
        run: |
          # Get changed files
          CHANGED_FILES=$(git diff --name-only origin/main...HEAD | grep -E '\\.(ts|tsx|py)$' || true)

          if [ -z "$CHANGED_FILES" ]; then
            echo "No analytics-relevant files changed"
            echo "has_changes=false" >> $GITHUB_OUTPUT
            exit 0
          fi

          echo "has_changes=true" >> $GITHUB_OUTPUT

          # Run analytics extraction
          python scripts/extract_analytics_calls.py \\
            --files $CHANGED_FILES \\
            --output analytics_calls.json

      - name: Load tracking plan
        if: steps.extract.outputs.has_changes == 'true'
        run: |
          # Fetch latest tracking plan from source of truth
          curl -s -H "Authorization: Bearer \${{ secrets.AMPLITUDE_API_KEY }}" \\
            "https://amplitude.com/api/2/taxonomy/event" > amplitude_events.json

          # Also load local tracking plan
          cp analytics/tracking_plan.json local_plan.json

      - name: Validate against tracking plan
        if: steps.extract.outputs.has_changes == 'true'
        id: validate
        run: |
          python scripts/validate_analytics.py \\
            --calls analytics_calls.json \\
            --plan local_plan.json \\
            --production amplitude_events.json \\
            --output validation_report.json

          # Check for blocking violations
          CRITICAL=$(jq '.critical_count' validation_report.json)
          if [ "$CRITICAL" -gt 0 ]; then
            echo "status=failed" >> $GITHUB_OUTPUT
            echo "::error::Found $CRITICAL critical analytics violations"
          else
            echo "status=passed" >> $GITHUB_OUTPUT
          fi

      - name: Generate PR comment
        if: steps.extract.outputs.has_changes == 'true'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = JSON.parse(fs.readFileSync('validation_report.json', 'utf8'));

            let body = '## Analytics Validation Report\\n\\n';

            if (report.critical_count > 0) {
              body += ':x: **BLOCKED**: Found critical schema violations\\n\\n';
            } else if (report.warning_count > 0) {
              body += ':warning: **WARNINGS**: Found non-critical issues\\n\\n';
            } else {
              body += ':white_check_mark: **PASSED**: All analytics calls validated\\n\\n';
            }

            body += '### Summary\\n';
            body += \`| Metric | Count |\\n|--------|-------|\\n\`;
            body += \`| Events validated | \${report.events_checked} |\\n\`;
            body += \`| Critical violations | \${report.critical_count} |\\n\`;
            body += \`| Warnings | \${report.warning_count} |\\n\\n\`;

            if (report.violations.length > 0) {
              body += '### Violations\\n';
              for (const v of report.violations.slice(0, 10)) {
                const icon = v.severity === 'critical' ? ':x:' : ':warning:';
                body += \`\${icon} **\${v.event_name}**: \${v.details}\\n\`;
                body += \`  - File: \\\`\${v.file}:\${v.line}\\\`\\n\`;
              }
            }

            body += '\\n---\\n_Validated by Analytics Schema CI_';

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });

      - name: Fail on critical violations
        if: steps.validate.outputs.status == 'failed'
        run: |
          echo "Critical analytics violations detected. Blocking merge."
          exit 1

      - name: Upload validation artifacts
        if: always() && steps.extract.outputs.has_changes == 'true'
        uses: actions/upload-artifact@v4
        with:
          name: analytics-validation
          path: |
            validation_report.json
            analytics_calls.json`,
              },
              {
                language: 'python',
                title: 'Observability and Alerting Module',
                description:
                  'Prometheus metrics, LangSmith integration, and intelligent alerting for the analytics validation system.',
                code: `"""
Observability Module for Analytics Validation Pipeline
Prometheus metrics, alerting, and LangSmith tracing
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
import time

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import requests

logger = logging.getLogger(__name__)


# ─── Prometheus Metrics ───────────────────────────────────────────────────────

VALIDATION_RUNS_TOTAL = Counter(
    "analytics_validation_runs_total",
    "Total validation runs",
    ["status"],
)

VALIDATION_DURATION_SECONDS = Histogram(
    "analytics_validation_duration_seconds",
    "Duration of validation runs",
    buckets=[5, 10, 30, 60, 120, 300],
)

SCHEMA_VIOLATIONS_TOTAL = Counter(
    "analytics_schema_violations_total",
    "Total schema violations detected",
    ["severity", "violation_type"],
)

VOLUME_ANOMALIES_TOTAL = Counter(
    "analytics_volume_anomalies_total",
    "Total volume anomalies detected",
    ["severity", "anomaly_type"],
)

HEALTH_SCORE_GAUGE = Gauge(
    "analytics_health_score",
    "Current analytics health score (0-100)",
)

TRACKING_COVERAGE_GAUGE = Gauge(
    "analytics_tracking_coverage_pct",
    "Percentage of expected events being tracked",
)

EVENTS_VALIDATED = Counter(
    "analytics_events_validated_total",
    "Total events validated",
)


@dataclass
class MetricsCollector:
    """Collects and exposes analytics validation metrics."""

    port: int = 8081
    _started: bool = False

    def start_server(self) -> None:
        """Start Prometheus metrics server."""
        if not self._started:
            start_http_server(self.port)
            self._started = True
            logger.info("Metrics server started on port %d", self.port)

    def record_validation_run(
        self,
        status: str,
        duration: float,
        health_score: float,
        violation_count: int,
        anomaly_count: int,
    ) -> None:
        """Record a validation run."""
        VALIDATION_RUNS_TOTAL.labels(status=status).inc()
        VALIDATION_DURATION_SECONDS.observe(duration)
        HEALTH_SCORE_GAUGE.set(health_score)

    def record_violation(self, severity: str, violation_type: str) -> None:
        """Record a schema violation."""
        SCHEMA_VIOLATIONS_TOTAL.labels(severity=severity, violation_type=violation_type).inc()

    def record_anomaly(self, severity: str, anomaly_type: str) -> None:
        """Record a volume anomaly."""
        VOLUME_ANOMALIES_TOTAL.labels(severity=severity, anomaly_type=anomaly_type).inc()

    def update_coverage(self, coverage_pct: float) -> None:
        """Update tracking coverage gauge."""
        TRACKING_COVERAGE_GAUGE.set(coverage_pct)


# ─── Slack Alerting ───────────────────────────────────────────────────────────

@dataclass
class AlertConfig:
    """Configuration for alert routing."""
    webhook_url: str
    channels: dict[str, str] = field(default_factory=lambda: {
        "critical": "#analytics-critical",
        "warning": "#analytics-alerts",
        "ci_failure": "#engineering",
        "report": "#product-analytics",
    })


class AnalyticsAlerter:
    """Sends analytics validation alerts to Slack."""

    def __init__(self, config: AlertConfig) -> None:
        self.config = config
        logger.info("AnalyticsAlerter initialized")

    def send_validation_alert(
        self,
        title: str,
        violations: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        health_score: float,
    ) -> bool:
        """Send validation results alert."""
        critical_count = sum(1 for v in violations + anomalies if v.get("severity") == "critical")

        if critical_count == 0 and health_score > 80:
            return False  # No alert needed

        severity = "critical" if critical_count > 0 else "warning"
        channel = self.config.channels.get(severity, "#analytics-alerts")

        # Build message
        emoji = ":rotating_light:" if severity == "critical" else ":warning:"
        color = "#dc3545" if severity == "critical" else "#ffc107"

        blocks = [
            f"{emoji} *Analytics Validation Alert*",
            f"Health Score: *{health_score:.0f}/100*",
            "",
            f"*Issues Detected:*",
            f"- Schema violations: {len(violations)} ({critical_count} critical)",
            f"- Volume anomalies: {len(anomalies)}",
        ]

        if violations:
            blocks.append("")
            blocks.append("*Top Violations:*")
            for v in sorted(violations, key=lambda x: x.get("severity", "") == "critical", reverse=True)[:5]:
                icon = ":x:" if v.get("severity") == "critical" else ":warning:"
                blocks.append(f"  {icon} {v['event_name']}: {v.get('details', v.get('violation_type', 'Unknown'))}")

        payload = {
            "channel": channel,
            "attachments": [{
                "color": color,
                "text": "\\n".join(blocks),
                "footer": "Analytics Validation Pipeline",
                "ts": int(datetime.now(timezone.utc).timestamp()),
            }],
        }

        try:
            response = requests.post(self.config.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Alert sent to %s", channel)
            return True
        except requests.RequestException as e:
            logger.error("Failed to send alert: %s", e)
            return False

    def send_ci_failure_alert(
        self,
        pr_number: int,
        pr_title: str,
        violations: list[dict[str, Any]],
        repo: str,
    ) -> bool:
        """Send CI failure alert for blocked PR."""
        channel = self.config.channels.get("ci_failure", "#engineering")

        message = f"""
:no_entry: *Analytics CI Check Failed*

PR #{pr_number}: {pr_title}
Repository: {repo}

*Blocking Violations:*
"""
        for v in violations[:5]:
            message += f"- {v['event_name']}: {v.get('details', 'Schema violation')}\\n"

        message += f"\\n<https://github.com/{repo}/pull/{pr_number}|View PR>"

        payload = {
            "channel": channel,
            "text": message,
        }

        try:
            response = requests.post(self.config.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error("Failed to send CI alert: %s", e)
            return False

    def send_weekly_report(
        self,
        health_score: float,
        coverage_pct: float,
        violations_by_week: dict[str, int],
        top_issues: list[dict[str, Any]],
    ) -> bool:
        """Send weekly analytics health report."""
        channel = self.config.channels.get("report", "#product-analytics")

        trend = "trending up" if health_score > 80 else "needs attention"
        emoji = ":chart_with_upwards_trend:" if health_score > 80 else ":chart_with_downwards_trend:"

        message = f"""
{emoji} *Weekly Analytics Health Report*

*Health Score:* {health_score:.0f}/100 ({trend})
*Tracking Coverage:* {coverage_pct:.1f}%

*Week-over-Week Violations:*
"""
        for week, count in violations_by_week.items():
            message += f"- {week}: {count} violations\\n"

        if top_issues:
            message += "\\n*Top Issues to Address:*\\n"
            for i, issue in enumerate(top_issues[:3], 1):
                message += f"{i}. {issue['event_name']}: {issue.get('recommendation', 'Fix schema violation')}\\n"

        payload = {
            "channel": channel,
            "text": message,
        }

        try:
            response = requests.post(self.config.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error("Failed to send report: %s", e)
            return False


def timed_validation(metrics: MetricsCollector):
    """Decorator to time validation operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                status = "success"
                return result
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.perf_counter() - start
                VALIDATION_DURATION_SECONDS.observe(duration)
                logger.info("Validation completed in %.2fs with status: %s", duration, status)
        return wrapper
    return decorator


def create_observability_stack(config: dict[str, Any]) -> tuple[MetricsCollector, AnalyticsAlerter]:
    """Create the observability stack."""
    metrics = MetricsCollector(port=config.get("metrics_port", 8081))
    metrics.start_server()

    alert_config = AlertConfig(
        webhook_url=config["slack_webhook_url"],
        channels=config.get("slack_channels", {}),
    )
    alerter = AnalyticsAlerter(alert_config)

    logger.info("Observability stack initialized")
    return metrics, alerter`,
              },
            ],
          },
        ],
      },
    },
  ],
};
