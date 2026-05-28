import pytest
from scripts.replay_evaluation import CounterfactualReplayEngine, generate_comparison_report

def test_propensity_scoring():
    """Ensure position bias calculation correctly decreases probability for lower ranks."""
    engine = CounterfactualReplayEngine()
    
    p_rank_0 = engine._compute_propensity(0) # Top result
    p_rank_1 = engine._compute_propensity(1) # Second result
    p_rank_9 = engine._compute_propensity(9) # 10th result
    
    # 1/log2(2) = 1.0
    assert abs(p_rank_0 - 1.0) < 1e-5
    # 1/log2(3) = 0.63
    assert p_rank_1 < p_rank_0
    assert p_rank_9 < p_rank_1
    assert p_rank_9 > 0.0 # Never exactly 0

def test_ips_evaluation():
    """Test Inverse Propensity Scoring (IPS) with a mocked historical log."""
    engine = CounterfactualReplayEngine()
    
    mock_logs = [
        {
            "user_id": "u1",
            "recommended_items": [10, 20, 30], # Historical slate
            "clicked_item_id": 20 # User clicked the item at rank 1 (0-indexed)
        },
        {
            "user_id": "u2",
            "recommended_items": [40, 50, 60],
            "clicked_item_id": None # No click
        }
    ]
    
    # Mock Policy A (Identical to historical logging policy)
    def policy_a(user_id):
        if user_id == "u1": return [10, 20, 30]
        return [40, 50, 60]
        
    ips_a = engine.evaluate_ips(mock_logs, policy_a)
    # Expected IPS: 
    # Session 1: click on 20. hist_rank = 1. new_rank = 1. Weight = P(1)/P(1) = 1.0. Reward = 1.0
    # Session 2: no click. Reward = 0.0
    # Average = 1.0 / 2 = 0.5
    assert abs(ips_a - 0.5) < 1e-5

    # Mock Policy B (Better policy! Places the clicked item at Rank 0)
    def policy_b(user_id):
        if user_id == "u1": return [20, 10, 30] # Promoted 20 to the top
        return [60, 50, 40]
        
    ips_b = engine.evaluate_ips(mock_logs, policy_b)
    # Session 1: click on 20. hist_rank = 1. new_rank = 0. 
    # Weight = P(new=0)/P(hist=1) = 1.0 / 0.6309 = 1.5849
    # Average = 1.5849 / 2 = 0.7924
    assert ips_b > ips_a # The better policy should have a higher estimated CTR!

def test_comparison_report():
    """Ensure the text report generation correctly calculates relative lift."""
    logs = [{"user_id": "u1", "recommended_items": [1], "clicked_item_id": 1}]
    
    # Fake policies
    def p_base(uid): return [1]
    def p_treatment(uid): return [1] # Exact same, 0% lift
    
    report = generate_comparison_report("V1", "V2", logs, p_base, p_treatment)
    
    assert "0.00%" in report
    assert "REJECTED" in report # Needs > 1% lift to pass
