import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, UTC

# Add parent directory to python path to resolve backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.data.database import SessionLocal
from backend.agents.recommender_optimizer_agent import RecommenderOptimizerAgent


async def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="APEX Recommender Optimizer Agent CLI Runner")
    parser.add_argument("--hours", type=int, default=24, help="Scope of recent events to analyze (in hours)")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock/dry-run mode without OpenRouter queries")
    parser.add_argument("--output-dir", type=str, default="output/optimization_reports", help="Output directory for reports")
    args = parser.parse_args()

    print(f"[{datetime.now(UTC).isoformat()}] Starting Recommender Optimizer Agent...")
    print(f"Params: hours={args.hours}, dry_run={args.dry_run}, output_dir={args.output_dir}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize database session
    db = SessionLocal()
    try:
        agent = RecommenderOptimizerAgent(db, name="Recommender Performance & Drift Agent")
        
        # Run agent
        report_md = await agent.run(hours=args.hours, dry_run=args.dry_run)
        
        # Save output report with timestamped file
        timestamp = int(time.time())
        report_file_name = f"recommender_optimization_report_{timestamp}.md"
        report_path = os.path.join(args.output_dir, report_file_name)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Success: Optimization report written to: {report_path}")

        # Write latest report to fixed link
        latest_path = os.path.join(args.output_dir, "latest_optimization_report.md")
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Success: Latest optimization report updated at: {latest_path}")

        # Print agent reasoning telemetry
        print("\n" + "=" * 40 + "\n")
        print(agent.get_summary_markdown())
        print("\n" + "=" * 40 + "\n")

    except Exception as e:
        print(f"Fatal error during agent execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
