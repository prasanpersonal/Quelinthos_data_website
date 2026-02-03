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
    },
  ],
};
