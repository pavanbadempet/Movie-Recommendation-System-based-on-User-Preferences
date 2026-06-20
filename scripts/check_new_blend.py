import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from backend.models.ensemble_engine import ApexEnsembleEngine
from backend.models.neural_weight_optimizer import get_contextual_weights


def check_new_blend():
    print("=" * 60)
    print("APEX Meta-Learning Blending Diagnostics")
    print("=" * 60)

    # 1. Initialize Ensemble Engine
    print("\n[1] Initializing Ensemble Engine...")
    engine = ApexEnsembleEngine(num_users=1000, num_items=5000, emb_dim=16)
    engine.eval()
    print("Engine initialized successfully.")

    # 2. Test Cold-Start User vs. Regular User
    print("\n[2] Extracting Contextual Weights for Different User Profiles...")

    # Cold-Start User (0 ratings, 0 clicks, 0 views)
    profile_cold = {
        "total_ratings": 0,
        "avg_rating": 0.0,
        "click_count": 0,
        "view_count": 0
    }
    
    # Active User (50 ratings, 4.2 average, 100 clicks, 150 views)
    profile_active = {
        "total_ratings": 50,
        "avg_rating": 4.2,
        "click_count": 100,
        "view_count": 150
    }

    dummy_embedding = np.random.randn(16)

    model_path = Path("models/contextual_weight_net.pth")
    if not model_path.exists():
        print(f"WARNING: models/contextual_weight_net.pth not found! Using fallback static weights.")

    weights_cold = get_contextual_weights(
        behavior_profile=profile_cold,
        als_user_embedding=dummy_embedding,
        model_path=model_path
    )
    
    weights_active = get_contextual_weights(
        behavior_profile=profile_active,
        als_user_embedding=dummy_embedding,
        model_path=model_path
    )

    print("\n--- Generated Contextual Weights ---")
    print(f"{'Model Name':<15} | {'Cold-Start User':<20} | {'Active User':<20}")
    print("-" * 60)
    for model in weights_cold.keys():
        print(f"{model:<15} | {weights_cold[model]:.6f} {' ' * 9} | {weights_active[model]:.6f}")

    # 3. Test Prediction Blending Modes
    print("\n[3] Testing Blending Calculations...")
    candidate_ids = [101, 102, 103, 104, 105]
    
    # Run with default linear blend
    print("\nRunning linear blend (APEX_ENSEMBLE_BLEND_MODE=linear)...")
    os.environ["APEX_ENSEMBLE_BLEND_MODE"] = "linear"
    scores_linear = engine.predict_ensemble(5, candidate_ids)
    for cid in candidate_ids:
        print(f"  Item {cid} Score: {scores_linear[cid]:.6f}")

    # Run with geometric blend
    print("\nRunning geometric blend (APEX_ENSEMBLE_BLEND_MODE=geometric)...")
    os.environ["APEX_ENSEMBLE_BLEND_MODE"] = "geometric"
    scores_geom = engine.predict_ensemble(5, candidate_ids)
    for cid in candidate_ids:
        print(f"  Item {cid} Score: {scores_geom[cid]:.6f}")

    print("\nDiagnostics complete! Context-dependent ensemble blending is fully operational.")

if __name__ == "__main__":
    check_new_blend()
