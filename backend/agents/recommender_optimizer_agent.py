from datetime import UTC, datetime, timedelta
import os
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.agents.base_agent import BaseAgent
from backend.data.database import UserEvent
from backend.intelligence.openrouter_client import chat_completion, configured_models, openrouter_api_key


class RecommenderOptimizerAgent(BaseAgent):
    """
    Recommender Model Optimization & Drift Guard Agent.
    Evaluates recommender click/interaction telemetry, checks for performance drift,
    and queries OpenRouter LLMs to suggest model optimization parameters.
    """

    def __init__(self, db: Session, name: str = "Recommender Optimizer Agent"):
        super().__init__(name)
        self.db = db

    async def run(self, hours: int = 24, dry_run: bool = False) -> tuple[str, dict[str, Any]]:
        """
        Runs the recommender drift guard and optimization loop.
        Returns:
            A tuple of (report_markdown: str, report_json: dict)
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

        self.log_step(
            "Calculate Recommender Metrics", f"Retrieved {len(events)} user events to compute CTR & engagement."
        )

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
        ctr: float | None = None
        ctr_status = "unavailable_no_recommendation_served"
        if recommendations_served > 0:
            ctr = round(clicks / recommendations_served, 4)
            ctr_status = "observed"
        elif clicks > 0:
            ctr_status = "unavailable_missing_recommendation_served"

        avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0

        # Define baselines
        baseline_ctr = float(os.getenv("NOVA_BASELINE_CTR", "0.12"))
        drift_detected = ctr < (baseline_ctr * 0.85) if ctr is not None else None
        ctr_display = f"{ctr:.2%}" if ctr is not None else "Unavailable"
        drift_display = (
            "🚨 DRIFT DETECTED" if drift_detected is True else ("🟢 STABLE" if drift_detected is False else "⚪ UNKNOWN")
        )

        self.log_step(
            "Detect Performance Drift",
            f"CTR: {ctr_display} (Baseline: {baseline_ctr:.2%}; status: {ctr_status}). "
            f"Avg Rating: {avg_rating}/5. "
            f"Drift Status: {drift_display}",
        )

        # Build dynamic local heuristics
        if drift_detected is None:
            suggested_weights = None
            suggested_mmr_alpha = None
            suggested_learning_rate = None
            heuristic_diagnosis = (
                "CTR and performance drift are unavailable because recommendation_served impressions "
                "were not recorded. Restore impression telemetry before changing model parameters."
            )
        elif drift_detected:
            suggested_weights = {"sasrec": 0.50, "lightgcn": 0.30, "kan": 0.20}
            suggested_mmr_alpha = 0.55
            suggested_learning_rate = 0.002
            heuristic_diagnosis = (
                f"Engagement drift detected! CTR ({ctr:.2%}) has dropped below the baseline threshold. "
                f"Tuning actions: shift weights towards sequential SASRec model (0.50) to boost accuracy, "
                f"lower diversity penalty alpha to 0.55, and increase online learning rate to 0.002."
            )
        else:
            suggested_weights = {"sasrec": 0.45, "lightgcn": 0.35, "kan": 0.20}
            suggested_mmr_alpha = 0.65
            suggested_learning_rate = 0.001
            heuristic_diagnosis = (
                f"Engagement is stable. CTR ({ctr:.2%}) is healthy compared to baseline ({baseline_ctr:.2%}). "
                f"Maintaining default weights (SASRec=0.45, LightGCN=0.35, KAN=0.20) and standard parameters."
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
            f"- **CTR**: {ctr_display} ({ctr_status})\n"
            f"- **Target Baseline CTR**: {baseline_ctr:.2%}\n"
            f"- **Performance Drift Status**: {drift_display}\n"
            f"- **Average User Rating**: {avg_rating}/5 (based on {len(ratings)} reviews)\n"
            f"- **Total Searches**: {searches}\n\n"
            f"Please output:\n"
            f"1. A diagnosis of user engagement (potential causes of drift/stabilization).\n"
            f"2. Three hyperparameter/model suggestions (e.g. ensemble weights for retrieval, diversity penalty alpha, online learning rate).\n"
        )

        self.estimate_tokens(user_prompt, is_output=False)

        api_key = openrouter_api_key()
        models = configured_models("OPENROUTER_MODELS")

        if drift_detected is None:
            ai_response = (
                "**[TELEMETRY INCOMPLETE — NO TUNING GENERATED]**\n"
                f"Diagnosis: {heuristic_diagnosis}"
            )
            self.log_step("Call OpenRouter Optimizer", "Skipped tuning because CTR telemetry is incomplete.")
        elif dry_run or not api_key:
            ai_response = (
                f"**[HEURISTIC LOCAL ASSESSMENT (DRY RUN)]**\n"
                f"Diagnosis: {heuristic_diagnosis}\n\n"
                f"Recommendations:\n"
                f"1. Adjust ensemble weights to: SASRec={suggested_weights['sasrec']}, LightGCN={suggested_weights['lightgcn']}, KAN={suggested_weights['kan']}.\n"
                f"2. Set diversity MMR factor alpha to {suggested_mmr_alpha}.\n"
                f"3. Set online learning rate to {suggested_learning_rate}."
            )
            self.log_step("Call OpenRouter Optimizer", "Using mock optimization advice (dry run or missing API key).")
        else:
            try:
                ai_response = chat_completion(
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    models=models,
                    temperature=0.3,
                    timeout_seconds=30.0,
                    api_key=api_key,
                    use_fast_model=True,
                )
                self.log_step(
                    "Call OpenRouter Optimizer", f"Successfully fetched optimization recommendations using {models[0]}."
                )
            except Exception as e:
                self.log_error(f"OpenRouter query failed: {e}")
                ai_response = (
                    f"**[HEURISTIC LOCAL FALLBACK (API ERROR)]**\n"
                    f"Diagnosis: {heuristic_diagnosis}\n\n"
                    f"Recommendations:\n"
                    f"1. Fallback to weights: SASRec={suggested_weights['sasrec']}, LightGCN={suggested_weights['lightgcn']}, KAN={suggested_weights['kan']}.\n"
                    f"2. Fallback diversity alpha: {suggested_mmr_alpha}."
                )

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
        status_str = (
            "🚨 Drift Warning" if drift_detected is True else ("🟢 Healthy" if drift_detected is False else "⚪ Insufficient telemetry")
        )
        report_md.append(f"| **Click-Through Rate (CTR)** | {ctr_display} | {baseline_ctr:.2%} | {status_str} |")
        report_md.append(
            f"| **Average User Rating** | {avg_rating:.2f}/5 | 4.00/5 | {'🟢 Good' if avg_rating >= 3.8 else '🟡 Fair'} |"
        )
        report_md.append(f"| **Recommendations Served** | {recommendations_served:,} | - | - |")
        report_md.append(f"| **Total Clicks** | {clicks:,} | - | - |")
        report_md.append(f"| **Total Searches** | {searches:,} | - | - |")
        report_md.append("")

        report_md.append("## 🧠 OpenRouter AI Diagnosis & Hyperparameter Tuning")
        report_md.append(ai_response)
        report_md.append("")

        report_md.append(
            "\n*Disclaimer: AI-suggested hyperparameters must be validated on validation/offline testing sets before promotion to production.*"
        )

        self.finish("completed")

        report_json = {
            "name": self.name,
            "status": self.status,
            "duration_seconds": self.duration,
            "cost_usd": self.estimated_cost,
            "timestamp": datetime.now(UTC).isoformat(),
            "scope_hours": hours,
            "metrics": {
                "recommendations_served": recommendations_served,
                "clicks": clicks,
                "searches": searches,
                "ctr": ctr,
                "ctr_status": ctr_status,
                "baseline_ctr": baseline_ctr,
                "drift_detected": drift_detected,
                "avg_rating": avg_rating,
            },
            "suggested_hyperparameters": (
                {
                    "ensemble_weights": suggested_weights,
                    "diversity_alpha": suggested_mmr_alpha,
                    "online_learning_rate": suggested_learning_rate,
                }
                if drift_detected is not None
                else None
            ),
        }
        return "\n".join(report_md), report_json
