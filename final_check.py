import json
import ast
import sys

notebooks = [
    "notebooks/01_setup_and_train.ipynb",
    "notebooks/02_optimize_round1.ipynb",
    "notebooks/03_optimize_round2.ipynb",
    "notebooks/04_check_progress.ipynb",
]

print("=" * 60)
print("FINAL VALIDATION: Notebooks + Requirements + Source Code")
print("=" * 60)

all_ok = True

# 1. Validate notebooks
for nb_path in notebooks:
    print(f"\n📓 {nb_path}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"   Cells: {len(nb['cells'])} total, {len(code_cells)} code")
    
    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"   ❌ Cell {i+1}: SyntaxError - {e}")
            all_ok = False
    
    if all_ok:
        print(f"   ✅ All {len(code_cells)} code cells valid")

# 2. Validate source files
src_files = [
    "src/models/distillation.py",
    "src/config.py",
    "src/utils/env.py",
    "scripts/train.py",
    "scripts/optimize_round1.py",
    "scripts/optimize_round2.py",
]

print(f"\n🐍 Source files:")
for sf in src_files:
    with open(sf, "r", encoding="utf-8") as f:
        src = f.read()
    try:
        ast.parse(src)
        print(f"   ✅ {sf}")
    except SyntaxError as e:
        print(f"   ❌ {sf}: {e}")
        all_ok = False

# 3. Check distillation import
with open("src/models/distillation.py", "r", encoding="utf-8") as f:
    dist_src = f.read()
if "compute_kl_divergence" in dist_src and "from src.models.losses import compute_kd_loss, compute_kl_divergence" in dist_src:
    print(f"\n✅ compute_kl_divergence import present")
else:
    print(f"\n❌ compute_kl_divergence import MISSING")
    all_ok = False

# 4. Check requirements-cloud.txt exists and has key deps
with open("requirements-cloud.txt", "r", encoding="utf-8") as f:
    req = f.read()
key_deps = ["torch", "transformers", "datasets", "optuna", "jupyter", "bitsandbytes", "accelerate"]
missing = [d for d in key_deps if d not in req]
if missing:
    print(f"\n❌ Missing from requirements-cloud.txt: {missing}")
    all_ok = False
else:
    print(f"\n✅ requirements-cloud.txt has all key dependencies")

# 5. Check notebooks reference correct paths
for nb_path in notebooks:
    with open(nb_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "configs/default.yaml" in content:
        print(f"   ✅ {nb_path} references config correctly")
    else:
        print(f"   ⚠️ {nb_path} may have wrong config path")

print("\n" + "=" * 60)
if all_ok:
    print("ALL CHECKS PASSED ✅")
else:
    print("SOME CHECKS FAILED ❌")
print("=" * 60)
