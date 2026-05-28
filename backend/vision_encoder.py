import torch
import logging
from PIL import Image
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)

class VisionEncoder:
    """
    Vision Encoder using OpenAI's CLIP (Contrastive Language-Image Pretraining).
    Translates raw movie poster images into dense 512-dimensional semantic vectors,
    capturing color palettes, cinematography aesthetics, and visual themes.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing Vision Encoder ({model_name}) on {self.device}...")
        
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.embedding_dim = self.model.config.projection_dim
            logger.info("Vision Encoder initialized successfully.")
        except ImportError:
            logger.error("Failed to import transformers. Run: pip install transformers Pillow")
            raise

    def encode_images(self, image_paths: List[Union[str, Path]]) -> torch.Tensor:
        """
        Encode a batch of image paths into visual embeddings.
        Returns a tensor of shape (batch_size, embedding_dim).
        """
        images = []
        valid_indices = []
        
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(str(path)).convert("RGB")
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                
        if not images:
            return torch.zeros((len(image_paths), self.embedding_dim), device=self.device)
            
        # Process images
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        if not isinstance(image_features, torch.Tensor):
            if hasattr(image_features, "image_embeds") and image_features.image_embeds is not None:
                image_features = image_features.image_embeds
            elif hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
                image_features = image_features.pooler_output
            elif hasattr(image_features, "last_hidden_state"):
                image_features = image_features.last_hidden_state[:, 0, :]
            else:
                raise TypeError(f"Unsupported CLIP image feature output: {type(image_features)!r}")
            
        # Normalize the embeddings for cosine similarity
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        
        # Construct final tensor matching original batch size (filling failures with zeros)
        final_embeddings = torch.zeros((len(image_paths), self.embedding_dim), device=self.device)
        for idx, valid_idx in enumerate(valid_indices):
            final_embeddings[valid_idx] = image_features[idx]
            
        return final_embeddings
