"""
Configuration for LLaVA models.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class VisionConfig:
    """Configuration for the vision encoder."""
    model_name: str = "openai/clip-vit-large-patch14-336"
    image_size: int = 336
    patch_size: int = 14
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    projection_dim: int = 768
    dropout: float = 0.0
    attention_dropout: float = 0.0


@dataclass
class LanguageConfig:
    """Configuration for the language model."""
    model_name: str = "lmsys/vicuna-7b-v1.5"
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    max_position_embeddings: int = 2048
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_theta: float = 10000.0


@dataclass
class ProjectorConfig:
    """Configuration for the projection layer."""
    input_dim: int = 1024  # Vision encoder hidden size
    hidden_dim: int = 4096  # Projection hidden dimension
    output_dim: int = 4096  # Language model hidden size
    dropout: float = 0.1
    num_layers: int = 2
    activation: str = "gelu"


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_steps: int = 0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 3
    fp16: bool = True
    bf16: bool = False
    seed: int = 42
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch"


@dataclass
class LLaVAConfig:
    """Configuration for the LLaVA model."""
    vision: VisionConfig = VisionConfig()
    language: LanguageConfig = LanguageConfig()
    projector: ProjectorConfig = ProjectorConfig()
    training: TrainingConfig = TrainingConfig()
    
    # Additional configurations
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLaVAConfig":
        """Create a configuration from a dictionary."""
        vision_config = VisionConfig(**config_dict.get("vision", {}))
        language_config = LanguageConfig(**config_dict.get("language", {}))
        projector_config = ProjectorConfig(**config_dict.get("projector", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        
        # Get additional configurations
        additional_config = {k: v for k, v in config_dict.items() 
                            if k not in ["vision", "language", "projector", "training"]}
        
        # Create and return the configuration
        config = cls(
            vision=vision_config,
            language=language_config,
            projector=projector_config,
            training=training_config,
            **additional_config
        )
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        config_dict = {
            "vision": {
                "model_name": self.vision.model_name,
                "image_size": self.vision.image_size,
                "patch_size": self.vision.patch_size,
                "hidden_size": self.vision.hidden_size,
                "num_attention_heads": self.vision.num_attention_heads,
                "num_hidden_layers": self.vision.num_hidden_layers,
                "intermediate_size": self.vision.intermediate_size,
                "projection_dim": self.vision.projection_dim,
                "dropout": self.vision.dropout,
                "attention_dropout": self.vision.attention_dropout
            },
            "language": {
                "model_name": self.language.model_name,
                "hidden_size": self.language.hidden_size,
                "num_attention_heads": self.language.num_attention_heads,
                "num_hidden_layers": self.language.num_hidden_layers,
                "intermediate_size": self.language.intermediate_size,
                "max_position_embeddings": self.language.max_position_embeddings,
                "vocab_size": self.language.vocab_size,
                "rms_norm_eps": self.language.rms_norm_eps,
                "use_cache": self.language.use_cache,
                "rope_theta": self.language.rope_theta
            },
            "projector": {
                "input_dim": self.projector.input_dim,
                "hidden_dim": self.projector.hidden_dim,
                "output_dim": self.projector.output_dim,
                "dropout": self.projector.dropout,
                "num_layers": self.projector.num_layers,
                "activation": self.projector.activation
            },
            "training": {
                "batch_size": self.training.batch_size,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "num_train_epochs": self.training.num_train_epochs,
                "max_steps": self.training.max_steps,
                "warmup_steps": self.training.warmup_steps,
                "lr_scheduler_type": self.training.lr_scheduler_type,
                "logging_steps": self.training.logging_steps,
                "save_steps": self.training.save_steps,
                "eval_steps": self.training.eval_steps,
                "save_total_limit": self.training.save_total_limit,
                "fp16": self.training.fp16,
                "bf16": self.training.bf16,
                "seed": self.training.seed,
                "gradient_checkpointing": self.training.gradient_checkpointing,
                "optim": self.training.optim
            },
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty
        }
        
        return config_dict


# Default configurations for different model sizes
LLAVA_7B_CONFIG = LLaVAConfig(
    language=LanguageConfig(
        model_name="lmsys/vicuna-7b-v1.5",
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        intermediate_size=11008
    ),
    projector=ProjectorConfig(
        input_dim=1024,
        hidden_dim=4096,
        output_dim=4096
    )
)

LLAVA_13B_CONFIG = LLaVAConfig(
    language=LanguageConfig(
        model_name="lmsys/vicuna-13b-v1.5",
        hidden_size=5120,
        num_attention_heads=40,
        num_hidden_layers=40,
        intermediate_size=13824
    ),
    projector=ProjectorConfig(
        input_dim=1024,
        hidden_dim=5120,
        output_dim=5120
    )
)