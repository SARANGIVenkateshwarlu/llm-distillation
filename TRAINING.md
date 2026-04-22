# Training Procedure & Process

A simplified guide to training the student model with knowledge distillation.

---

## 1. Setup

```bash
# Install dependencies
make install
# Or manually: pip install -r requirements.txt && pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your HF token and any custom settings
```

---

## 2. Quick Train (Default Config)

```bash
make train
```

This runs:
```bash
python scripts/train.py --config configs/default.yaml
```

---

## 3. Custom Training

### Via CLI flags
```bash
python scripts/train.py \
    --teacher Qwen/Qwen2.5-7B-Instruct \
    --student Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 3 \
    --batch-size 1 \
    --learning-rate 2e-4 \
    --bf16
```

### Via custom config
Copy `configs/default.yaml`, edit values, then:
```bash
python scripts/train.py --config configs/my_config.yaml
```

---

## 4. What Happens During Training

1. **Load Dataset** → `databricks/databricks-dolly-15k` (default)
2. **Load Teacher** → Large model (e.g., 7B) with 4-bit quantization
3. **Load Student** → Small model (e.g., 1.5B) + LoRA adapters
4. **Tokenize** → Format instructions + context + response
5. **Distill** → Combined loss:
   ```
   loss = α × CE(student_logits, labels) + β × T² × KL(student_soft, teacher_soft)
   ```
6. **Save** → Best model to `artifacts/best_model/final/`
7. **Plot** → Training curves to `artifacts/plots/`

---

## 5. Hyperparameter Optimization (Optional)

```bash
# Round 1: Quick search (~1 hour, 20 trials)
make optimize-round1

# Round 2: Fine-tune best region (~8 hours, 75 trials)
make optimize-round2
```

---

## 6. Evaluate

```bash
make evaluate
# Or:
python scripts/evaluate.py --model-path artifacts/best_model/final --generate
```

---

## 7. Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `teacher` | Qwen2.5-7B-Instruct | Large model to learn from |
| `student` | Qwen2.5-1.5B-Instruct | Small model to train |
| `epochs` | 3 | Training epochs |
| `batch-size` | 1 | Per-device batch size |
| `learning-rate` | 2e-4 | Optimizer LR |
| `temperature` | 2.0 | Softmax smoothing for KD |
| `alpha` | 0.3 | Weight for ground-truth CE loss |
| `beta` | 0.7 | Weight for teacher KL loss |
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling factor |

---

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| Out of Memory | Reduce `batch-size` to 1, increase `gradient_accumulation_steps` to 16, or use `8bit` teacher |
| Slow Training | Enable `gradient_checkpointing`, use `bf16` |
| Poor Results | Increase `temperature` (3-5), adjust `alpha/beta` (try 0.2/0.8), increase `lora_r` |

---

## 9. Output Artifacts

```
artifacts/
├── best_model/final/     # Trained student model + tokenizer
├── checkpoints/          # Intermediate checkpoints
├── plots/                # Training curves
└── logs/                 # Training logs
```
