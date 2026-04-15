"""
Loss functions for knowledge distillation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    
    Combines:
    1. Cross-entropy loss with ground truth labels
    2. KL divergence loss between teacher and student distributions
    
    The final loss is: alpha * CE + beta * KD
    
    Args:
        temperature: Temperature for softening distributions
        alpha: Weight for cross-entropy loss
        beta: Weight for KL divergence loss
        ignore_index: Index to ignore in loss computation
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        
        # Validate weights
        assert abs(alpha + beta - 1.0) < 1e-6, "alpha + beta should equal 1.0"
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the combined distillation loss.
        
        Args:
            student_logits: Logits from student model [batch_size, seq_len, vocab_size]
            teacher_logits: Logits from teacher model [batch_size, seq_len, vocab_size]
            labels: Ground truth labels [batch_size, seq_len]
        
        Returns:
            Combined loss value
        """
        return compute_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            temperature=self.temperature,
            alpha=self.alpha,
            beta=self.beta,
            ignore_index=self.ignore_index,
        )
    
    def __repr__(self) -> str:
        return (
            f"DistillationLoss("
            f"temperature={self.temperature}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"ignore_index={self.ignore_index})"
        )


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.3,
    beta: float = 0.7,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute knowledge distillation loss.
    
    The loss combines:
    1. Cross-entropy with ground truth labels (hard targets)
    2. KL divergence between teacher and student softened distributions (soft targets)
    
    Formula: loss = alpha * CE(student_logits, labels) + beta * T^2 * KL(student_soft, teacher_soft)
    
    The T^2 factor compensates for the gradients being scaled by 1/T^2 due to softmax temperature.
    
    Args:
        student_logits: Logits from student model [batch_size, seq_len, vocab_size]
        teacher_logits: Logits from teacher model [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len]
        temperature: Temperature for softening distributions
        alpha: Weight for cross-entropy loss
        beta: Weight for KL divergence loss
        ignore_index: Index to ignore in loss computation
    
    Returns:
        Combined loss value
    """
    # Flatten for loss computation
    batch_size, seq_len, vocab_size = student_logits.shape
    
    # Compute cross-entropy loss with ground truth
    ce_loss = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=ignore_index,
    )
    
    # Compute KL divergence loss with teacher
    # Soften distributions with temperature
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    
    with torch.no_grad():
        # Teacher probabilities (no gradients needed)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL divergence: KL(teacher || student) = sum(teacher * log(teacher / student))
    # Using F.kl_div with log_target=False: KL(target || input)
    # where input is student_log_probs and target is teacher_probs
    kl_loss = F.kl_div(
        student_log_probs.view(-1, vocab_size),
        teacher_probs.view(-1, vocab_size),
        reduction="batchmean",
        log_target=False,
    )
    
    # Scale by T^2 to compensate for gradient scaling
    kl_loss = kl_loss * (temperature ** 2)
    
    # Combine losses
    total_loss = alpha * ce_loss + beta * kl_loss
    
    return total_loss


def compute_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute standard cross-entropy loss.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len]
        ignore_index: Index to ignore
    
    Returns:
        Cross-entropy loss
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )


def compute_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Compute KL divergence between student and teacher.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        temperature: Temperature for softening
    
    Returns:
        KL divergence loss
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    kl_loss = F.kl_div(
        student_log_probs.view(-1, student_logits.size(-1)),
        teacher_probs.view(-1, teacher_logits.size(-1)),
        reduction="batchmean",
        log_target=False,
    )
    
    return kl_loss * (temperature ** 2)


def compute_mse_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute MSE loss between student and teacher logits.
    
    This is an alternative to KL divergence for distillation.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
    
    Returns:
        MSE loss
    """
    return F.mse_loss(student_logits, teacher_logits)


def compute_cosine_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity loss between student and teacher.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
    
    Returns:
        Cosine embedding loss
    """
    # Flatten and normalize
    student_flat = student_logits.view(-1, student_logits.size(-1))
    teacher_flat = teacher_logits.view(-1, teacher_logits.size(-1))
    
    # Cosine similarity loss: 1 - cos(x, y)
    return 1 - F.cosine_similarity(student_flat, teacher_flat, dim=-1).mean()


def get_loss_function(
    loss_type: str = "distillation",
    **kwargs
) -> nn.Module:
    """
    Get a loss function by type.
    
    Args:
        loss_type: Type of loss function
        **kwargs: Arguments for the loss function
    
    Returns:
        Loss function module
    """
    if loss_type == "distillation":
        return DistillationLoss(**kwargs)
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == "mse":
        return nn.MSELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def explain_distillation_loss() -> str:
    """
    Return an explanation of how knowledge distillation loss works.
    
    Returns:
        Explanation string
    """
    explanation = """
    Knowledge Distillation Loss Explanation
    =======================================
    
    The distillation loss combines two objectives:
    
    1. Cross-Entropy Loss (Hard Targets):
       - Standard loss between student predictions and ground truth labels
       - Ensures the student learns the correct answers
       - Weight: alpha (typically 0.2-0.4)
    
    2. KL Divergence Loss (Soft Targets):
       - Measures difference between student and teacher probability distributions
       - Transfers "dark knowledge" from teacher to student
       - Uses temperature scaling to soften distributions
       - Weight: beta (typically 0.6-0.8)
    
    Temperature Scaling:
       - Higher temperature (>1) produces softer probability distributions
       - Reveals relationships between classes that one-hot labels don't capture
       - Teacher's confidence patterns are transferred to student
       - T^2 scaling compensates for gradient magnitude reduction
    
    Final Loss:
       loss = alpha * CE(student_logits, labels) + beta * T^2 * KL(student_soft, teacher_soft)
    
    Benefits:
       - Student learns from teacher's reasoning patterns
       - Often achieves better performance than training from scratch
       - Can match larger models with smaller, faster models
    """
    return explanation
