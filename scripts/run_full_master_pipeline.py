import sys
import os
import time
import subprocess
import requests

def log_step(step_num, step_name):
    print(f"\n==================================================")
    print(f" STEP {step_num}: {step_name}")
    print(f"==================================================\n")

def run_script(script_name, args=[]):
    cmd = [sys.executable, f"scripts/{script_name}"] + args
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"⚠️ Warning in {script_name}: {res.stderr[:300]}")
    else:
        print(f"✅ {script_name} completed successfully!")

def main():
    print("STARTING FULL MASTER END-TO-END RECOMMENDATION PIPELINE...")
    start_time = time.time()

    # Step 1: PySpark Medallion ETL & Data Ingestion
    log_step(1, "PySpark Medallion ETL & Data Ingestion")
    run_script("pyspark_medallion_pipeline.py")

    # Step 2: SOTA Multi-Model Suite Training (Two-Tower, MMoE, Generative Diffusion)
    log_step(2, "Train SOTA Model Suite (Two-Tower, MMoE Ranker, Diffusion)")
    run_script("train_mmoe_ranker.py")
    run_script("train_apex_models.py")

    # Step 3: Agentic AI Recommender Optimizer
    log_step(3, "Run Agentic AI Recommender Hyperparameter Optimizer Agent")
    run_script("run_optimizer_agent.py")

    # Step 4: Rebuild & Export Serving Artifacts (ONNX, TurboVec)
    log_step(4, "Rebuild & Quantize Serving Artifacts (ONNX, TurboVec)")
    run_script("rebuild_serving_artifacts.py")

    # Step 5: Sync to Hugging Face Model Hub & Space
    log_step(5, "Sync Codebase & Weights to Hugging Face Hub & Space")
    run_script("hf_upload.py")

    # Step 6: Deploy Cloudflare Workers AI Edge Gateway
    log_step(6, "Deploy Cloudflare Workers AI Edge Gateway")
    subprocess.run(["npx", "wrangler", "deploy"], capture_output=True, text=True)

    elapsed = time.time() - start_time
    print(f"\n==================================================")
    print(f" 🎉 FULL MASTER PIPELINE COMPLETED IN {elapsed:.1f} SECONDS!")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()
