"""
Custom trainer for knowledge distillation.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction

from src.models.losses import compute_kd_loss


class KnowledgeDistillationTrainer(Trainer):
    """
    Custom trainer for knowledge distillation training.
    
    This trainer extends the Hugging Face Trainer to support:
    - Teacher model for generating soft targets
    - Combined CE + KD loss
    - Temperature scaling
    - Proper handling of padding tokens
    
    Args:
        teacher_model: Frozen teacher model for generating soft targets
        temperature: Temperature for softening distributions
        alpha: Weight for cross-entropy loss
        beta: Weight for KL divergence loss
        **kwargs: Additional arguments passed to Trainer
    """
    
    def __init__(
        self,
        teacher_model: Optional[torch.nn.Module] = None,
        temperature: float = 2.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        # Validate loss weights
        assert abs(alpha + beta - 1.0) < 1e-6, f"alpha + beta should equal 1.0, got {alpha + beta}"
        
        # Move teacher to same device as model
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
    
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute the distillation loss.
        
        Args:
            model: Student model
            inputs: Input batch with input_ids, attention_mask, labels
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for gradient accumulation)
        
        Returns:
            Loss value or (loss, outputs) tuple
        """
        # Extract labels
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Labels are required for distillation training")
        
        # Forward pass through student
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        student_logits = outputs.logits
        
        # Get teacher logits if teacher is available
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"].to(self.teacher_model.device),
                    attention_mask=inputs["attention_mask"].to(self.teacher_model.device),
                )
                teacher_logits = teacher_outputs.logits.to(student_logits.device)
            
            # Compute distillation loss
            loss = compute_kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=self.temperature,
                alpha=self.alpha,
                beta=self.beta,
                ignore_index=-100,
            )
        else:
            # Standard cross-entropy if no teacher
            import torch.nn.functional as F
            loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        
        # Compute and store metrics for logging during training
        if model.training:
            with torch.no_grad():
                preds = torch.argmax(student_logits, dim=-1)
                mask = labels != -100
                if mask.any():
                    self._last_accuracy = (preds[mask] == labels[mask]).float().mean().item()
                else:
                    self._last_accuracy = 0.0
                
                if self.teacher_model is not None:
                    kl = compute_kl_divergence(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        temperature=self.temperature,
                    )
                    self._last_kl_loss = kl.item()
                else:
                    self._last_kl_loss = 0.0
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step with custom metric logging.
        
        Logs accuracy and KL divergence alongside the standard loss.
        """
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Log custom metrics that were computed during compute_loss
        if model.training:
            if hasattr(self, "_last_accuracy"):
                self.log("accuracy", self._last_accuracy)
            if hasattr(self, "_last_kl_loss"):
                self.log("kd_loss", self._last_kl_loss)
        
        return loss
    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform a prediction step.
        
        Args:
            model: Model to evaluate
            inputs: Input batch
            prediction_loss_only: Whether to return only loss
            ignore_keys: Keys to ignore in model output
        
        Returns:
            Tuple of (loss, logits, labels)
        """
        # Move inputs to device
        inputs = self._prepare_inputs(inputs)
        
        # Compute loss
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Get logits for evaluation
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            logits = outputs.logits
        
        labels = inputs.get("labels")
        
        return (loss, logits, labels)


class DistillationTrainingArguments:
    """
    Training arguments specific to knowledge distillation.
    
    This class extends standard training arguments with
    distillation-specific parameters.
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        use_kd: bool = True,
        **training_args
    ):
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.use_kd = use_kd
        self.training_args = training_args
    
    def to_training_arguments(self):
        """Convert to Hugging Face TrainingArguments."""
        from transformers import TrainingArguments
        return TrainingArguments(**self.training_args)


def create_distillation_trainer(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    data_collator: Any,
    output_dir: str,
    temperature: float = 2.0,
    alpha: float = 0.3,
    beta: float = 0.7,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    eval_steps: int = 100,
    save_steps: int = 100,
    bf16: bool = True,
    fp16: bool = False,
    **kwargs
) -> KnowledgeDistillationTrainer:
    """
    Create a KnowledgeDistillationTrainer with common settings.
    
    Args:
        student_model: Student model to train
        teacher_model: Teacher model for distillation
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
        data_collator: Data collator
        output_dir: Output directory
        temperature: Distillation temperature
        alpha: CE loss weight
        beta: KD loss weight
        **kwargs: Additional training arguments
    
    Returns:
        Configured KnowledgeDistillationTrainer
    """
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=fp16,
        report_to="none",
        remove_unused_columns=False,
        **kwargs
    )
    
    trainer = KnowledgeDistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        temperature=temperature,
        alpha=alpha,
        beta=beta,
    )
    
    return trainer
