/**
 * CATEGORY TEMPLATE — 5-Step Standard + AI Solutions
 *
 * How to add a new category:
 * 1. Copy this file and rename it: XX-your-category.ts (e.g. 11-cybersecurity.ts)
 * 2. Fill in all fields below — every field is required
 * 3. Register in index.ts:
 *    - Import: import { yourCategory } from './XX-your-category.ts';
 *    - Add to allCategories array
 * 4. Run `npm test` — validation will catch any missing/invalid fields
 *
 * Notes:
 * - icon must match a key in src/components/IconResolver.tsx (see AVAILABLE_ICON_NAMES)
 * - accentColor must be one of: 'neon-blue' | 'neon-purple' | 'neon-gold'
 * - Each pain point needs a full PRICE framework (5 sections)
 * - severity must be one of: 'low' | 'medium' | 'high' | 'critical'
 * - code snippet language must be one of: 'sql' | 'python' | 'bash'
 * - AI code snippet language must be one of: 'python' | 'yaml' | 'json'
 * - annualCostRange must match format: "$XK - $YM" (e.g. "$500K - $3M")
 *
 * 5-Step Implementation Standard:
 * Every pain point MUST have exactly 5 implementation steps:
 *   Step 1 — Data Foundation (SQL): Schema, tables, views, indexes
 *   Step 2 — Core Pipeline (Python): Ingestion, transformation, ML/NLP logic
 *   Step 3 — Testing & Validation (Python + SQL): Data quality assertions, pipeline tests
 *   Step 4 — Deployment & Ops (Bash + Python): Environment setup, migration, scheduling
 *   Step 5 — Monitoring & Alerting (SQL + Python): SLA dashboards, anomaly detection, alerts
 *
 * AI Solutions Standard:
 * Every pain point MUST have two AI solution guides:
 *
 * aiEasyWin (3-Step Pattern) — Low-code with ChatGPT/Claude + Zapier:
 *   Step 1 — Data Extraction & Preparation: Export data into AI-readable format
 *   Step 2 — AI-Powered Analysis: Use ChatGPT/Claude with custom prompts
 *   Step 3 — Automation & Delivery: Zapier/Make workflows for stakeholder delivery
 *   Cost: $50-300/month depending on complexity
 *
 * aiAdvanced (5-Step Multi-Agent Pattern) — LangGraph + CrewAI:
 *   Step 1 — Agent Architecture & Role Design: Define 4+ agents with CrewAI
 *   Step 2 — Data Ingestion Agent(s): Build API connectors with LangChain
 *   Step 3 — Analysis & Decision Agent(s): Domain-specific analysis agents
 *   Step 4 — Workflow Orchestration: Wire agents with LangGraph state machine
 *   Step 5 — Deployment & Observability: Docker + LangSmith tracing
 *   Cost: $300-3,000/month depending on scale
 */

import type { Category } from '../types.ts';

export const yourCategory: Category = {
  id: 'your-category-slug',
  number: 11, // next sequential number
  title: 'Your Category Title',
  shortTitle: 'Short',
  description: 'One-line description of this business domain.',
  icon: 'Briefcase', // must exist in IconResolver
  accentColor: 'neon-blue', // 'neon-blue' | 'neon-purple' | 'neon-gold'
  painPoints: [
    {
      id: 'your-pain-point-slug',
      number: 1,
      title: 'Pain Point Title',
      subtitle: 'Pain Point Subtitle',
      summary: 'A 1-2 sentence summary of the problem.',
      price: {
        present: {
          title: 'Current Situation',
          description: 'What is happening right now.',
          bullets: ['Bullet point 1', 'Bullet point 2', 'Bullet point 3'],
          severity: 'high',
        },
        root: {
          title: 'Root Cause',
          description: 'Why this problem exists.',
          bullets: ['Bullet point 1', 'Bullet point 2'],
          severity: 'high',
        },
        impact: {
          title: 'Business Impact',
          description: 'What this costs the business.',
          bullets: ['Bullet point 1', 'Bullet point 2', 'Bullet point 3'],
          severity: 'critical',
        },
        cost: {
          title: 'Cost of Inaction',
          description: 'What happens if nothing changes.',
          bullets: ['Bullet point 1', 'Bullet point 2'],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Expected Return',
          description: 'What the fix delivers.',
          bullets: ['Bullet point 1', 'Bullet point 2'],
          severity: 'medium',
        },
      },
      implementation: {
        overview: 'High-level description of the implementation approach.',
        prerequisites: [
          'Domain-specific prerequisite 1',
          'Domain-specific prerequisite 2',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        steps: [
          // Step 1 — Data Foundation
          {
            stepNumber: 1,
            title: 'Data Foundation',
            description: 'Schema design, tables, views, and indexes.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Core Schema',
                description: 'Database schema for the domain.',
                code: `SELECT 1;`,
              },
            ],
          },
          // Step 2 — Core Pipeline
          {
            stepNumber: 2,
            title: 'Core Pipeline',
            description: 'Ingestion, transformation, and business logic.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Pipeline Implementation',
                description: 'Main data processing pipeline.',
                code: `print("hello")`,
              },
            ],
          },
          // Step 3 — Testing & Validation
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description: 'Data quality assertions and pipeline tests.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Data Quality Assertions',
                description: 'Row counts, null checks, referential integrity, freshness.',
                code: `SELECT 1;`,
              },
              {
                language: 'python',
                title: 'Pipeline Test Suite',
                description: 'pytest-based validation of pipeline correctness.',
                code: `print("test")`,
              },
            ],
          },
          // Step 4 — Deployment & Ops
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description: 'Environment setup, migration, and scheduling.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Deployment Script',
                description: 'Environment checks, migration, and scheduler setup.',
                code: `echo "deploy"`,
              },
              {
                language: 'python',
                title: 'Configuration Loader',
                description: 'Env-var based config with secrets and connection pools.',
                code: `print("config")`,
              },
            ],
          },
          // Step 5 — Monitoring & Alerting
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description: 'SLA dashboards, anomaly detection, and Slack alerts.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Monitoring Dashboard',
                description: 'Health metrics and SLA tracking queries.',
                code: `SELECT 1;`,
              },
              {
                language: 'python',
                title: 'Alert Service',
                description: 'Slack webhook alerting with threshold monitoring.',
                code: `print("alert")`,
              },
            ],
          },
        ],
        toolsUsed: [
          'PostgreSQL',
          'Python',
          'pytest',
          'Docker',
          'GitHub Actions',
          'cron / Airflow',
          'Slack API',
        ],
      },
      // ─────────────────────────────────────────────────────────────────────
      // AI Easy Win — ChatGPT/Claude + Zapier (3-Step Pattern)
      // ─────────────────────────────────────────────────────────────────────
      aiEasyWin: {
        overview: 'Use ChatGPT/Claude to analyze exported data and Zapier to automate delivery without custom infrastructure.',
        estimatedMonthlyCost: '$70 - $150/month',
        primaryTools: ['ChatGPT Plus ($20/mo)', 'Zapier Pro ($29.99/mo)', 'Google Sheets (free)'],
        alternativeTools: ['Claude Pro ($20/mo)', 'Make ($10.59/mo)', 'Airtable ($20/mo)'],
        steps: [
          {
            stepNumber: 1,
            title: 'Data Extraction & Preparation',
            description: 'Export data from source systems into a structured format for AI analysis.',
            toolsUsed: ['Google Sheets', 'CSV Export', 'Zapier'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Data Structure Template',
                description: 'Standardized format for AI-ready data export.',
                code: `{
  "columns": ["id", "date", "category", "value", "status"],
  "data_types": {
    "id": "string",
    "date": "ISO8601",
    "category": "enum",
    "value": "number",
    "status": "enum"
  }
}`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'AI-Powered Analysis',
            description: 'Build reusable prompts that analyze your data and generate actionable insights.',
            toolsUsed: ['ChatGPT Plus', 'Custom GPT (optional)'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Analysis Prompt Template',
                description: 'Structured prompt for consistent AI analysis.',
                code: `system: |
  You are a domain expert analyst. Analyze the provided data
  and identify patterns, anomalies, and actionable insights.

user: |
  Analyze this data and provide:
  1. Summary statistics
  2. Key patterns identified
  3. Anomalies or concerns
  4. Recommended actions

  Data:
  {{paste_data_here}}`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Automation & Delivery',
            description: 'Configure Zapier to trigger analysis and deliver insights to stakeholders.',
            toolsUsed: ['Zapier', 'Slack', 'Google Sheets'],
            codeSnippets: [
              {
                language: 'json',
                title: 'Zapier Workflow Configuration',
                description: 'Automated workflow for scheduled analysis and delivery.',
                code: `{
  "trigger": {
    "app": "Schedule",
    "event": "Every Week on Monday at 9:00am"
  },
  "actions": [
    {"app": "Google Sheets", "action": "Get Spreadsheet Data"},
    {"app": "ChatGPT", "action": "Send Prompt"},
    {"app": "Slack", "action": "Send Channel Message"}
  ]
}`,
              },
            ],
          },
        ],
      },

      // ─────────────────────────────────────────────────────────────────────
      // AI Advanced — Multi-Agent with LangGraph + CrewAI (5-Step Pattern)
      // ─────────────────────────────────────────────────────────────────────
      aiAdvanced: {
        overview: 'Deploy a multi-agent system that autonomously ingests data, performs analysis, and delivers recommendations.',
        estimatedMonthlyCost: '$500 - $1,500/month',
        architecture: 'Supervisor agent coordinates specialist agents for data collection, analysis, and reporting',
        agents: [
          {
            name: 'DataCollectorAgent',
            role: 'Data Ingestion Specialist',
            goal: 'Extract and normalize data from all source systems daily',
            tools: ['API clients', 'Database connectors', 'File parsers'],
          },
          {
            name: 'AnalystAgent',
            role: 'Domain Analyst',
            goal: 'Analyze collected data and identify patterns, anomalies, and insights',
            tools: ['pandas', 'numpy', 'scikit-learn', 'custom analyzers'],
          },
          {
            name: 'DecisionAgent',
            role: 'Strategy Advisor',
            goal: 'Generate actionable recommendations based on analysis',
            tools: ['LLM reasoning', 'rule engine', 'optimization solver'],
          },
          {
            name: 'ReporterAgent',
            role: 'Stakeholder Communicator',
            goal: 'Create and deliver reports with visualizations',
            tools: ['matplotlib', 'Slack API', 'email sender', 'PDF generator'],
          },
        ],
        orchestration: {
          framework: 'LangGraph',
          pattern: 'Supervisor',
          stateManagement: 'Redis-backed state with daily checkpointing',
        },
        steps: [
          {
            stepNumber: 1,
            title: 'Agent Architecture & Role Design',
            description: 'Define agent team with roles, goals, and tool access using CrewAI.',
            toolsUsed: ['CrewAI', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'CrewAI Agent Definitions',
                description: 'Define the multi-agent team structure.',
                code: `from crewai import Agent, Crew, Task
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

data_collector = Agent(
    role="Data Ingestion Specialist",
    goal="Extract and normalize data from all sources",
    backstory="Expert at connecting to APIs and databases",
    llm=llm,
    tools=[api_tool, db_tool],
)

analyst = Agent(
    role="Domain Analyst",
    goal="Identify patterns and anomalies in data",
    backstory="Senior analyst with deep domain expertise",
    llm=llm,
    tools=[analysis_tool],
)`,
              },
            ],
          },
          {
            stepNumber: 2,
            title: 'Data Ingestion Agent(s)',
            description: 'Build agents that connect to source systems and extract data.',
            toolsUsed: ['LangChain', 'API SDKs', 'Database connectors'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Data Ingestion Pipeline',
                description: 'Agent tool for extracting data from sources.',
                code: `from langchain.tools import tool
from typing import Dict, List
import pandas as pd

@tool
def fetch_source_data(source: str, date_range: str) -> Dict:
    """Fetch data from the specified source system."""
    # Connect to source and extract data
    connector = get_connector(source)
    raw_data = connector.fetch(date_range)

    # Normalize to standard schema
    normalized = normalize_schema(raw_data)
    return {"status": "success", "rows": len(normalized)}`,
              },
            ],
          },
          {
            stepNumber: 3,
            title: 'Analysis & Decision Agent(s)',
            description: 'Build specialist agents for domain-specific analysis.',
            toolsUsed: ['pandas', 'scikit-learn', 'LangChain'],
            codeSnippets: [
              {
                language: 'python',
                title: 'Analysis Agent Tools',
                description: 'Domain-specific analysis capabilities.',
                code: `@tool
def analyze_patterns(data: pd.DataFrame) -> Dict:
    """Identify patterns and anomalies in the data."""
    results = {
        "summary": compute_summary_stats(data),
        "anomalies": detect_anomalies(data),
        "trends": identify_trends(data),
    }
    return results

@tool
def generate_recommendations(analysis: Dict) -> List[str]:
    """Generate actionable recommendations from analysis."""
    recommendations = []
    for anomaly in analysis["anomalies"]:
        rec = create_recommendation(anomaly)
        recommendations.append(rec)
    return recommendations`,
              },
            ],
          },
          {
            stepNumber: 4,
            title: 'Workflow Orchestration',
            description: 'Wire agents into a stateful workflow with LangGraph.',
            toolsUsed: ['LangGraph', 'Redis'],
            codeSnippets: [
              {
                language: 'python',
                title: 'LangGraph State Machine',
                description: 'Orchestrate agents with state management.',
                code: `from langgraph.graph import StateGraph, END
from typing import TypedDict

class WorkflowState(TypedDict):
    data: dict
    analysis: dict
    recommendations: list
    report: str

def build_workflow():
    workflow = StateGraph(WorkflowState)

    workflow.add_node("collect", data_collector_node)
    workflow.add_node("analyze", analyst_node)
    workflow.add_node("decide", decision_node)
    workflow.add_node("report", reporter_node)

    workflow.add_edge("collect", "analyze")
    workflow.add_edge("analyze", "decide")
    workflow.add_edge("decide", "report")
    workflow.add_edge("report", END)

    workflow.set_entry_point("collect")
    return workflow.compile()`,
              },
            ],
          },
          {
            stepNumber: 5,
            title: 'Deployment & Observability',
            description: 'Containerize and deploy with LangSmith tracing.',
            toolsUsed: ['Docker', 'LangSmith', 'Prometheus'],
            codeSnippets: [
              {
                language: 'yaml',
                title: 'Docker Compose Configuration',
                description: 'Production deployment with monitoring.',
                code: `version: '3.8'
services:
  agent-workflow:
    build: .
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=\${LANGSMITH_API_KEY}
      - LANGSMITH_TRACING=true
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data
volumes:
  redis_data:`,
              },
            ],
          },
        ],
      },

      metrics: {
        annualCostRange: '$500K - $3M',
        roi: '6x',
        paybackPeriod: '3-4 months',
        investmentRange: '$80K - $150K',
      },
      tags: ['tag-1', 'tag-2'],
    },
  ],
};
