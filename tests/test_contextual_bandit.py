from backend.intelligence.contextual_bandit import BanditEngine


def test_bandit_initialization():
    bandit = BanditEngine()
    assert bandit.total_impressions == 0
    assert bandit.item_clicks[100] == 0


def test_bandit_reward_update():
    bandit = BanditEngine()
    bandit.update_reward(100, clicked=True)
    bandit.update_reward(100, clicked=False)
    assert bandit.item_impressions[100] == 2
    assert bandit.item_clicks[100] == 1
    assert bandit.total_impressions == 2


def test_ucb_unseen_item_exploration():
    bandit = BanditEngine()
    # Unseen item gets exploration boost
    score = bandit.get_ucb_score(999, base_score=0.5)
    assert score > 10.0


def test_thompson_sampling_scoring():
    bandit = BanditEngine()
    bandit.update_reward(50, clicked=True)
    bandit.update_reward(50, clicked=True)
    score = bandit.get_thompson_sample(50, base_score=1.0)
    assert score > 0.5


def test_apply_exploration_pipeline():
    bandit = BanditEngine()
    candidates = [
        {"id": 1, "title": "Known Hit", "similarity_score": 0.9},
        {"id": 2, "title": "Hidden Gem", "similarity_score": 0.8},
    ]
    explored = bandit.apply_exploration(candidates, strategy="thompson")
    assert len(explored) == 2
    assert "similarity_score" in explored[0]
