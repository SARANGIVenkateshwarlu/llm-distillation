"""
Tests for loss functions.
"""

import pytest
import torch

from src.models.losses import compute_kd_loss, DistillationLoss


def test_kd_loss_basic():
    """Test basic KD loss computation."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = compute_kd_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=2.0,
        alpha=0.3,
        beta=0.7,
    )
    
    assert loss.shape == torch.Size([])
    assert loss.item() > 0


def test_kd_loss_module():
    """Test DistillationLoss module."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    
    loss_fn = DistillationLoss(temperature=2.0, alpha=0.3, beta=0.7)
    
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = loss_fn(student_logits, teacher_logits, labels)
    
    assert loss.shape == torch.Size([])
    assert loss.item() > 0


def test_alpha_beta_sum():
    """Test that alpha + beta = 1."""
    with pytest.raises(AssertionError):
        DistillationLoss(temperature=2.0, alpha=0.5, beta=0.6)


def test_kd_loss_with_ignore_index():
    """Test KD loss with ignored indices."""
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[0, 0] = -100  # Ignore index
    
    loss = compute_kd_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=2.0,
        alpha=0.3,
        beta=0.7,
        ignore_index=-100,
    )
    
    assert loss.shape == torch.Size([])
    assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
