"""
HuggingFace Hub uploader for publishing trained models.

This module handles:
- Merging LoRA weights before upload (optional)
- Generating model cards with training metadata
- Uploading models, tokenizers, and configs to HF Hub
- Setting appropriate tags and licenses
"""

import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config
from src.utils.io import ensure_dir


class HubUploader:
    """
    Upload trained models to the HuggingFace Hub.

    Args:
        repo_id: HuggingFace Hub repository ID (e.g., "username/model-name")
        token: HuggingFace API token. If None, uses HF_TOKEN env var or cached token.
        private: Whether to create a private repository
        merge_lora: Whether to merge LoRA weights before uploading
        commit_message: Custom commit message
        tags: List of tags to attach to the model card
        license: License identifier for the model
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        merge_lora: bool = True,
        commit_message: str = "Upload distilled student model",
        tags: Optional[List[str]] = None,
        license: str = "mit",
    ):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self.private = private
        self.merge_lora = merge_lora
        self.commit_message = commit_message
        self.tags = tags or []
        self.license = license

        self.api = HfApi(token=self.token)

    def create_repository(self, exist_ok: bool = True) -> str:
        """
        Create the HF Hub repository if it doesn't exist.

        Args:
            exist_ok: If True, don't raise error if repo already exists

        Returns:
            Repository URL
        """
        repo_url = create_repo(
            repo_id=self.repo_id,
            token=self.token,
            private=self.private,
            exist_ok=exist_ok,
        )
        print(f"Repository ready: {repo_url}")
        return repo_url

    def generate_model_card(
        self,
        model_path: Union[str, Path],
        config: Optional[Config] = None,
        base_model: Optional[str] = None,
        teacher_model: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> str:
        """
        Generate a model card markdown string.

        Args:
            model_path: Path to the saved model
            config: Training configuration object
            base_model: Base model name (student base)
            teacher_model: Teacher model name
            dataset: Dataset used for training

        Returns:
            Model card markdown string
        """
        model_path = Path(model_path)

        # Extract info from config if provided
        if config:
            base_model = base_model or config.models.get("student", {}).name
            teacher_model = teacher_model or config.models.get("teacher", {}).name
            dataset = dataset or config.dataset.name
            temperature = config.distillation.temperature
            alpha = config.distillation.alpha
            beta = config.distillation.beta
            lora_r = config.lora.r
            lora_alpha = config.lora.lora_alpha
            epochs = config.training.num_train_epochs
            lr = config.training.learning_rate
        else:
            temperature = alpha = beta = lora_r = lora_alpha = epochs = lr = "N/A"

        tags_str = ", ".join([f'"{t}"' for t in self.tags]) if self.tags else '"distillation", "lora"'

        card = f"""---
license: {self.license}
base_model: {base_model or "unknown"}
tags:
  - distillation
  - knowledge-distillation
  - lora
  - {tags_str}
---

# Distilled Student Model

This model was created using **Knowledge Distillation** from a larger teacher model to a smaller student model, achieving comparable performance with significantly reduced computational requirements.

## Model Details

- **Base Model:** {base_model or "Unknown"}
- **Teacher Model:** {teacher_model or "Unknown"}
- **Training Dataset:** {dataset or "Unknown"}
- **Distillation Temperature:** {temperature}
- **Loss Weights:** α (CE) = {alpha}, β (KD) = {beta}

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | {epochs} |
| Learning Rate | {lr} |
| LoRA Rank (r) | {lora_r} |
| LoRA Alpha | {lora_alpha} |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")

inputs = tokenizer("Explain machine learning in simple terms:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

## Distillation Method

This model was trained with a combined loss function:

```
loss = α × CE(student_logits, labels) + β × T² × KL(student_soft, teacher_soft)
```

Where:
- **α** = weight for cross-entropy with ground truth
- **β** = weight for KL divergence with teacher soft targets
- **T** = temperature for softening probability distributions

## Citation

```bibtex
@article{{hinton2015distilling,
  title={{Distilling the knowledge in a neural network}},
  author={{Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff}},
  journal={{arXiv preprint arXiv:1503.02531}},
  year={{2015}}
}}
```
"""
        return card

    def upload(
        self,
        model_path: Union[str, Path],
        config: Optional[Config] = None,
        base_model: Optional[str] = None,
        teacher_model: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> str:
        """
        Upload a model folder to HuggingFace Hub.

        This loads the model, optionally merges LoRA weights, generates a model card,
        and uploads everything to the specified repository.

        Args:
            model_path: Local path to the saved model directory
            config: Training configuration for model card generation
            base_model: Base model name
            teacher_model: Teacher model name
            dataset: Dataset name

        Returns:
            URL of the uploaded model on HF Hub
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Ensure repo exists
        self.create_repository(exist_ok=True)

        # Determine if model has LoRA adapters
        adapter_config_path = model_path / "adapter_config.json"
        has_lora = adapter_config_path.exists()

        # Prepare upload directory
        upload_dir = Path("./artifacts/hub_upload_temp")
        ensure_dir(upload_dir)

        if has_lora and self.merge_lora:
            print("LoRA adapters detected. Merging weights before upload...")

            # Load and merge
            model = AutoModelForCausalLM.from_pretrained(
                base_model or config.models.get("student", {}).name if config else base_model,
                torch_dtype="auto",
                device_map="cpu",  # Load on CPU to save VRAM during upload
                trust_remote_code=True,
            )

            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(model_path))
            model = model.merge_and_unload()

            tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            # Save merged model
            merged_path = upload_dir / "merged_model"
            ensure_dir(merged_path)
            model.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)

            upload_source = merged_path
        else:
            # Upload as-is (either no LoRA or merge_lora=False)
            upload_source = model_path

        # Generate and save model card
        card_content = self.generate_model_card(
            model_path=model_path,
            config=config,
            base_model=base_model,
            teacher_model=teacher_model,
            dataset=dataset,
        )

        readme_path = upload_source / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_content)

        # Upload to Hub
        print(f"Uploading to {self.repo_id}...")
        self.api.upload_folder(
            folder_path=str(upload_source),
            repo_id=self.repo_id,
            token=self.token,
            commit_message=self.commit_message,
        )

        url = f"https://huggingface.co/{self.repo_id}"
        print(f"Upload complete! Model available at: {url}")

        # Cleanup temp directory
        if (upload_dir / "merged_model").exists():
            import shutil
            shutil.rmtree(upload_dir)

        return url


def push_model_to_hub(
    model_path: Union[str, Path],
    repo_id: str,
    config: Optional[Config] = None,
    token: Optional[str] = None,
    private: bool = False,
    merge_lora: bool = True,
    commit_message: str = "Upload distilled student model",
    tags: Optional[List[str]] = None,
    license: str = "mit",
    base_model: Optional[str] = None,
    teacher_model: Optional[str] = None,
    dataset: Optional[str] = None,
) -> str:
    """
    Convenience function to push a model to HuggingFace Hub.

    Args:
        model_path: Local path to the saved model directory
        repo_id: HF Hub repository ID (e.g., "username/model-name")
        config: Training configuration object
        token: HF API token (or uses HF_TOKEN env var)
        private: Whether repo should be private
        merge_lora: Whether to merge LoRA weights before upload
        commit_message: Git commit message
        tags: Model tags
        license: License identifier
        base_model: Base model name
        teacher_model: Teacher model name
        dataset: Dataset name

    Returns:
        URL of the uploaded model
    """
    uploader = HubUploader(
        repo_id=repo_id,
        token=token,
        private=private,
        merge_lora=merge_lora,
        commit_message=commit_message,
        tags=tags,
        license=license,
    )

    return uploader.upload(
        model_path=model_path,
        config=config,
        base_model=base_model,
        teacher_model=teacher_model,
        dataset=dataset,
    )
