import numpy as np
import pytest
import torch

from backend.intelligence.multimodal_fusion import MultiModalFusionIndex
from backend.intelligence.vision_encoder import VisionEncoder


@pytest.fixture
def dummy_image_path(tmp_path):
    from PIL import Image

    path = tmp_path / "dummy.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(path)
    return path


def test_vision_encoder(dummy_image_path):
    """Test that the CLIP vision encoder correctly generates 512d vectors without NaN."""
    try:
        encoder = VisionEncoder(device="cpu")
    except Exception:
        pytest.skip("transformers/Pillow not fully installed or model unavailable.")

    embeddings = encoder.encode_images([dummy_image_path])

    assert embeddings.shape == (1, 512)
    assert not torch.isnan(embeddings).any()
    # Check L2 normalization (norm should be ~1.0)
    norm = torch.norm(embeddings, p=2, dim=-1).item()
    assert abs(norm - 1.0) < 1e-4


def test_multimodal_fusion_l2_normalization():
    """Test the MultiModalFusionIndex mathematical bounds."""
    fusion = MultiModalFusionIndex()

    # Create fake unnormalized text and vision vectors
    text_vec = np.random.rand(1, 768).astype(np.float32) * 10
    vision_vec = np.random.rand(1, 512).astype(np.float32) * 20

    # Simulate internal fusion logic
    t_norm = text_vec / np.linalg.norm(text_vec)
    v_norm = vision_vec / np.linalg.norm(vision_vec)

    fused = np.concatenate([t_norm * 0.6, v_norm * 0.4], axis=1)
    fused_norm = fusion._normalize(fused)

    assert fused_norm.shape == (1, 1280)
    assert not np.isnan(fused_norm).any()

    norm = np.linalg.norm(fused_norm)
    assert abs(norm - 1.0) < 1e-4
