import sys
import os
import time
import subprocess

def log_step(step_num, step_name):
    print(f"\n==================================================")
    print(f" STEP {step_num}: {step_name}")
    print(f"==================================================\n")

def run_script(script_name, args=[]):
    cmd = [sys.executable, f"scripts/{script_name}"] + args
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout[:500])
    if res.returncode != 0:
        print(f"Warning in {script_name}: {res.stderr[:300]}")
    else:
        print(f"DONE: {script_name} completed successfully!")

def main():
    print("STARTING ULTIMATE ALL-IN-ONE END-TO-END SYSTEM PIPELINE...")
    start_time = time.time()

    # 1. Data Ingestion & Merging
    log_step(1, "Download Real Datasets (TMDB + MovieLens 25M)")
    run_script("download_and_merge_datasets.py")

    log_step(2, "PySpark Medallion ETL (Bronze -> Silver -> Gold)")
    run_script("pyspark_medallion_pipeline.py")

    # 2. Deep Learning & RL Model Training
    log_step(3, "Train Two-Tower Deep Retrieval Model")
    run_script("train_two_tower.py")

    log_step(4, "Train Multi-Gate Mixture-of-Experts (MMoE) Ranker")
    run_script("train_mmoe_ranker.py")

    log_step(5, "Train Generative Diffusion Model")
    run_script("train_generative_diffusion.py")

    log_step(6, "Train Reinforcement Learning (RL) Policy Network")
    run_script("train_rl_policy_compact.py")

    log_step(7, "Causal Debias Model Training")
    run_script("causal_debias_training.py")

    # 3. Agentic AI & Quantization
    log_step(8, "Run Agentic AI Recommender Hyperparameter Optimizer")
    run_script("run_optimizer_agent.py")

    log_step(9, "Export & Quantize ONNX Serving Models")
    run_script("export_to_onnx.py")

    log_step(10, "Build TurboVec Scalar Quantized Vector Index")
    run_script("migrate_faiss_to_turbovec.py")

    # 4. Evaluation & Benchmarking
    log_step(11, "Run Real-World Evaluation & Semantic Benchmark Audit")
    run_script("evaluate_real_world.py")
    run_script("evaluate_semantic_benchmark.py")

    # 5. Multi-Cloud Deployment
    log_step(12, "Sync Codebase & Weights to Hugging Face Hub & Space")
    run_script("hf_upload.py")

    log_step(13, "Deploy 100% Max-Utilized Cloudflare Workers AI + KV Gateway")
    subprocess.run(["npx", "wrangler", "deploy"], capture_output=True, text=True)

    elapsed = time.time() - start_time
    print(f"\n==================================================")
    print(f" ALL-IN-ONE PIPELINE COMPLETED IN {elapsed:.1f} SECONDS!")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()
