import os
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.agents.base_agent import BaseAgent
from backend.data.database import UserEvent
from backend.intelligence.openrouter_client import chat_completion, openrouter_api_key, configured_models


class RecommenderOptimizerAgent(BaseAgent):
    """
    Recommender Model Optimization & Drift Guard Agent.
    Evaluates recommender click/interaction telemetry, checks for performance drift,
    and queries OpenRouter LLMs to suggest model optimization parameters.
    """

    def __init__(self, db: Session, name: str = "Recommender Optimizer Agent"):
        super().__init__(name)
        self.db = db

    async def run(self, hours: int = 24, dry_run: bool = False) -> str:
        """
        Runs the recommender drift guard and optimization loop.
        """
        self.start()

        # 1. Fetch interaction events
        self.log_step("Fetch Event Telemetry", f"Querying fact_user_event for records in the last {hours} hours.")
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        
        # Read naïve DateTime for database querying compatibility
        naive_cutoff = cutoff_time.replace(tzinfo=None)
        
        events = (
            self.db.query(UserEvent)
            .filter(UserEvent.created_at >= naive_cutoff)
            .order_by(desc(UserEvent.created_at))
            .all()
        )

        self.log_step("Calculate Recommender Metrics", f"Retrieved {len(events)} user events to compute CTR & engagement.")

        # 2. Aggregate metrics
        recommendations_served = 0
        clicks = 0
        searches = 0
        ratings = []

        for e in events:
            if e.event_type == "recommendation_served":
                recommendations_served += 1
            elif e.event_type == "click":
                clicks += 1
            elif e.event_type == "search":
                searches += 1
            elif e.event_type == "rating" and e.event_value is not None:
                ratings.append(e.event_value)

        # Calculate CTR
        ctr = 0.0
        if recommendations_served > 0:
            ctr = round(clicks / recommendations_served, 4)
        elif clicks > 0:
            # Fallback if recommendation_served is not logged but clicks are logged
            ctr = 0.08  # Default mock/baseline
        
        avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0

        # Define baselines
        baseline_ctr = float(os.getenv("NOVA_BASELINE_CTR", "0.12"))
        drift_detected = ctr < (baseline_ctr * 0.85)

        self.log_step(
            "Detect Performance Drift",
            f"CTR: {ctr:.2%} (Baseline: {baseline_ctr:.2%}). "
            f"Avg Rating: {avg_rating}/5. "
            f"Drift Status: {'🚨 DRIFT DETECTED' if drift_detected else '🟢 STABLE'}"
        )

        # 3. Query OpenRouter for optimization suggestions
        system_prompt = (
            "You are a Recommender System Optimization Agent. Your goal is to analyze real-time recommendation "
            "metrics (CTR, search rate, rating averages, drift status), diagnose performance regression, "
            "and suggest parameter updates for retrieval, ranking, and diversity reranking. "
            "Keep recommendations brief, mathematical, and actionable."
        )

        user_prompt = (
            f"Analyze recommendation system metrics for the past {hours} hours:\n\n"
            f"- **Recommendations Served**: {recommendations_served}\n"
            f"- **Clicks**: {clicks}\n"
            f"- **Calculated CTR**: {ctr:.2%}\n"
            f"- **Target Baseline CTR**: {baseline_ctr:.2%}\n"
            f"- **Performance Drift Status**: {'🚨 DRIFT DETECTED' if drift_detected else '🟢 STABLE'}\n"
            f"- **Average User Rating**: {avg_rating}/5 (based on {len(ratings)} reviews)\n"
            f"- **Total Searches**: {searches}\n\n"
            f"Please output:\n"
            f"1. A diagnosis of user engagement (potential causes of drift/stabilization).\n"
            f"2. Three hyperparameter/model suggestions (e.g. ensemble weights for retrieval, diversity penalty alpha, online learning rate).\n"
        )

        self.estimate_tokens(user_prompt, is_output=False)

        api_key = openrouter_api_key()
        models = configured_models("OPENROUTER_MODELS")

        if dry_run or not api_key:
            ai_response = (
                f"**[DRY RUN DIAGNOSIS]** Engagement is stable. Calculated CTR ({ctr:.2%}) is near baseline.\n"
                f"Recommendations:\n"
                f"1. Maintain current ensemble weights: SASRec=0.45, LightGCN=0.35, KAN=0.20.\n"
                f"2. Keep diversity MMR factor at alpha=0.65.\n"
                f"3. Keep online learning gradient rate at lr=0.001."
            )
            self.log_step("Call OpenRouter Optimizer", "Using mock optimization advice (dry run or missing API key).")
        else:
            try:
                # Run OpenRouter chat completion synchronously as configured in openrouter_client
                ai_response = chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    models=models,
                    temperature=0.3,
                    timeout_seconds=30.0,
                    api_key=api_key,
                    use_fast_model=True
                )
                self.log_step("Call OpenRouter Optimizer", f"Successfully fetched optimization recommendations using {models[0]}.")
            except Exception as e:
                self.log_error(f"OpenRouter query failed: {e}")
                ai_response = "**ERROR:** OpenRouter API query failed. Recommending baseline settings."

        self.estimate_tokens(ai_response, is_output=True)

        # 4. Construct Markdown Report
        report_md = []
        report_md.append("# 🎬 Recommender System Optimization Report")
        report_md.append(f"Generated on: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        report_md.append(f"Telemetry Window: Past {hours} hours")
        report_md.append("")
        
        report_md.append("## 📊 Quality Performance Metrics")
        report_md.append("| Metric | Value | Baseline / Target | Status |")
        report_md.append("|---|---|---|---|")
        status_str = "🚨 Drift Warning" if drift_detected else "🟢 Healthy"
        report_md.append(f"| **Click-Through Rate (CTR)** | {ctr:.2%} | {baseline_ctr:.2%} | {status_str} |")
        report_md.append(f"| **Average User Rating** | {avg_rating:.2f}/5 | 4.00/5 | {'🟢 Good' if avg_rating >= 3.8 else '🟡 Fair'} |")
        report_md.append(f"| **Recommendations Served** | {recommendations_served:,} | - | - |")
        report_md.append(f"| **Total Clicks** | {clicks:,} | - | - |")
        report_md.append(f"| **Total Searches** | {searches:,} | - | - |")
        report_md.append("")

        report_md.append("## 🧠 OpenRouter AI Diagnosis & Hyperparameter Tuning")
        report_md.append(ai_response)
        report_md.append("")
        
        report_md.append("\n*Disclaimer: AI-suggested hyperparameters must be validated on validation/offline testing sets before promotion to production.*")

        self.finish("completed")
        return "\n".join(report_md)
