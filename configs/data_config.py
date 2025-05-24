"""
Configuration for data processing in LLaVA.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    path: str
    image_folder: str
    annotation_file: str
    split: str = "train"
    max_samples: int = -1  # -1 means use all samples
    image_key: str = "image"
    caption_key: str = "caption"
    question_key: str = "question"
    answer_key: str = "answer"


@dataclass
class PretrainingDataConfig:
    """Configuration for pretraining data."""
    datasets: List[DatasetConfig] = field(default_factory=list)
    image_size: int = 336
    max_length: int = 2048
    pad_to_max_length: bool = False
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    
    @classmethod
    def default(cls) -> "PretrainingDataConfig":
        """Create a default pretraining data configuration."""
        return cls(
            datasets=[
                DatasetConfig(
                    name="laion-cc-sbu-558k",
                    path="liuhaotian/LLaVA-Pretrain",
                    image_folder="",
                    annotation_file="blip_laion_cc_sbu_558k.json",
                    image_key="image",
                    caption_key="caption"
                )
            ]
        )


@dataclass
class InstructionTuningDataConfig:
    """Configuration for instruction tuning data."""
    datasets: List[DatasetConfig] = field(default_factory=list)
    image_size: int = 336
    max_length: int = 2048
    pad_to_max_length: bool = False
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    
    @classmethod
    def default(cls) -> "InstructionTuningDataConfig":
        """Create a default instruction tuning data configuration."""
        return cls(
            datasets=[
                DatasetConfig(
                    name="llava-instruct-150k",
                    path="liuhaotian/LLaVA-Instruct-150K",
                    image_folder="",
                    annotation_file="llava_instruct_150k.json",
                    image_key="image",
                    question_key="question",
                    answer_key="answer"
                )
            ]
        )


@dataclass
class EvaluationDataConfig:
    """Configuration for evaluation data."""
    datasets: List[DatasetConfig] = field(default_factory=list)
    image_size: int = 336
    max_length: int = 2048
    pad_to_max_length: bool = False
    preprocessing_num_workers: int = 4
    
    @classmethod
    def default(cls) -> "EvaluationDataConfig":
        """Create a default evaluation data configuration."""
        return cls(
            datasets=[
                DatasetConfig(
                    name="vqav2",
                    path="",
                    image_folder="coco/val2014",
                    annotation_file="vqav2_val.json",
                    split="val",
                    question_key="question",
                    answer_key="answer"
                ),
                DatasetConfig(
                    name="gqa",
                    path="",
                    image_folder="gqa/images",
                    annotation_file="gqa_testdev_balanced.json",
                    split="testdev",
                    question_key="question",
                    answer_key="answer"
                ),
                DatasetConfig(
                    name="textvqa",
                    path="",
                    image_folder="textvqa/train_images",
                    annotation_file="textvqa_val.json",
                    split="val",
                    question_key="question",
                    answer_key="answer"
                )
            ]
        )


@dataclass
class DataConfig:
    """Configuration for all data processing."""
    pretraining: PretrainingDataConfig = field(default_factory=PretrainingDataConfig.default)
    instruction_tuning: InstructionTuningDataConfig = field(default_factory=InstructionTuningDataConfig.default)
    evaluation: EvaluationDataConfig = field(default_factory=EvaluationDataConfig.default)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataConfig":
        """Create a configuration from a dictionary."""
        # Process pretraining config
        pretraining_dict = config_dict.get("pretraining", {})
        pretraining_datasets = []
        for dataset_dict in pretraining_dict.get("datasets", []):
            pretraining_datasets.append(DatasetConfig(**dataset_dict))
        
        pretraining_config = PretrainingDataConfig(
            datasets=pretraining_datasets,
            **{k: v for k, v in pretraining_dict.items() if k != "datasets"}
        )
        
        # Process instruction tuning config
        instruction_dict = config_dict.get("instruction_tuning", {})
        instruction_datasets = []
        for dataset_dict in instruction_dict.get("datasets", []):
            instruction_datasets.append(DatasetConfig(**dataset_dict))
        
        instruction_config = InstructionTuningDataConfig(
            datasets=instruction_datasets,
            **{k: v for k, v in instruction_dict.items() if k != "datasets"}
        )
        
        # Process evaluation config
        evaluation_dict = config_dict.get("evaluation", {})
        evaluation_datasets = []
        for dataset_dict in evaluation_dict.get("datasets", []):
            evaluation_datasets.append(DatasetConfig(**dataset_dict))
        
        evaluation_config = EvaluationDataConfig(
            datasets=evaluation_datasets,
            **{k: v for k, v in evaluation_dict.items() if k != "datasets"}
        )
        
        # Create and return the configuration
        return cls(
            pretraining=pretraining_config,
            instruction_tuning=instruction_config,
            evaluation=evaluation_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        # Process pretraining datasets
        pretraining_datasets = []
        for dataset in self.pretraining.datasets:
            pretraining_datasets.append({
                "name": dataset.name,
                "path": dataset.path,
                "image_folder": dataset.image_folder,
                "annotation_file": dataset.annotation_file,
                "split": dataset.split,
                "max_samples": dataset.max_samples,
                "image_key": dataset.image_key,
                "caption_key": dataset.caption_key,
                "question_key": dataset.question_key,
                "answer_key": dataset.answer_key
            })
        
        # Process instruction tuning datasets
        instruction_datasets = []
        for dataset in self.instruction_tuning.datasets:
            instruction_datasets.append({
                "name": dataset.name,
                "path": dataset.path,
                "image_folder": dataset.image_folder,
                "annotation_file": dataset.annotation_file,
                "split": dataset.split,
                "max_samples": dataset.max_samples,
                "image_key": dataset.image_key,
                "caption_key": dataset.caption_key,
                "question_key": dataset.question_key,
                "answer_key": dataset.answer_key
            })
        
        # Process evaluation datasets
        evaluation_datasets = []
        for dataset in self.evaluation.datasets:
            evaluation_datasets.append({
                "name": dataset.name,
                "path": dataset.path,
                "image_folder": dataset.image_folder,
                "annotation_file": dataset.annotation_file,
                "split": dataset.split,
                "max_samples": dataset.max_samples,
                "image_key": dataset.image_key,
                "caption_key": dataset.caption_key,
                "question_key": dataset.question_key,
                "answer_key": dataset.answer_key
            })
        
        # Create the configuration dictionary
        config_dict = {
            "pretraining": {
                "datasets": pretraining_datasets,
                "image_size": self.pretraining.image_size,
                "max_length": self.pretraining.max_length,
                "pad_to_max_length": self.pretraining.pad_to_max_length,
                "preprocessing_num_workers": self.pretraining.preprocessing_num_workers,
                "overwrite_cache": self.pretraining.overwrite_cache
            },
            "instruction_tuning": {
                "datasets": instruction_datasets,
                "image_size": self.instruction_tuning.image_size,
                "max_length": self.instruction_tuning.max_length,
                "pad_to_max_length": self.instruction_tuning.pad_to_max_length,
                "preprocessing_num_workers": self.instruction_tuning.preprocessing_num_workers,
                "overwrite_cache": self.instruction_tuning.overwrite_cache
            },
            "evaluation": {
                "datasets": evaluation_datasets,
                "image_size": self.evaluation.image_size,
                "max_length": self.evaluation.max_length,
                "pad_to_max_length": self.evaluation.pad_to_max_length,
                "preprocessing_num_workers": self.evaluation.preprocessing_num_workers
            }
        }
        
        return config_dict