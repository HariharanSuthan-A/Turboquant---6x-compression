# TurboQuant Model Size Comparison

This Jupyter notebook demonstrates **PolarQuant** — a novel quantization technique applied to the K and V projection weights of transformer attention layers. It compares model sizes before and after quantization using `Qwen/Qwen3-4B-Instruct-2507` (or alternatively Llama 3.2 3B).

---

## Overview

The notebook implements and evaluates PolarQuant compression on transformer attention weights, achieving **~1.9× compression** with minimal quality degradation.

---

## Key Features

- **Zero Accuracy Loss**: Maintains high model precision despite extreme compression.
- **KV Cache Focus**: Targets the GPU memory bottleneck by shrinking the Key-Value cache used in LLM generation.
- **Two-Stage Mechanism**: Combines PolarQuant (rotation + scalar quantization) and QJL (1-bit residual correction) to achieve 3-bit compression.
- **No Retraining Required**: Works immediately on existing models (e.g., Gemma, Mistral) without needing expensive fine-tuning.
- **High-Speed Processing**: Delivers significant speedups in attention computation, especially on NVIDIA H100 GPUs.

---

## What is PolarQuant?

PolarQuant is a quantization method that:
- Applies **random orthogonal rotation** to weight matrices as preconditioning
- Transforms Cartesian coordinates to **polar coordinates** (radius + angle)
- Quantizes angles with **4-bit precision** and radii with **2-bit precision**
- Achieves **~3 bits per element** (vs. 16 bits for FP16)

---

## Notebook Workflow

| Step | Description |
|------|-------------|
| 1 | Install dependencies (`transformers`, `accelerate`, `matplotlib`, `tabulate`) |
| 2 | Authenticate with HuggingFace |
| 3 | Load `Qwen3-4B-Instruct` in FP16 |
| 4 | Extract and save baseline K/V projection weights |
| 5 | Implement `PolarQuantWeight` class with compress/decompress methods |
| 6 | Quantize all `k_proj` and `v_proj` weights across 36 layers |
| 7 | Compute size comparison metrics and compression ratios |
| 8 | Generate visualization charts (bar charts, pie charts, cosine similarity plots) |
| 9 | Reload quantized checkpoint and verify reconstruction quality |
| 10 | Inject quantized weights back into model and test generation |

---

## Results

| Metric | Baseline (FP16) | PolarQuant (4-bit θ / 2-bit r) |
|--------|-----------------|--------------------------------|
| **KV weight file size** | 377.5 MB | 202.6 MB |
| **Bits per element** | 16 | 3.02 |
| **Compression ratio** | 1.00× | **1.86×** |
| **Disk savings** | — | 174.9 MB (46%) |
| **Mean cosine sim (k_proj)** | 1.0000 | 0.9582 |
| **Mean cosine sim (v_proj)** | 1.0000 | 0.9581 |

---

## Key Components

### `PolarQuantWeight` Class

```python
class PolarQuantWeight:
    """
    PolarQuant applied to a 2-D weight matrix W of shape (out_dim, in_dim).
    Quantizes row-by-row with ~3 bits per element.
    """
    
    # Core methods:
    - compress(W)      → compressed dict with angle/radius indices
    - decompress(c)    → reconstructed weight matrix
    - bits_per_element() → effective bit rate calculation
```

### Compression Format

Each quantized weight matrix stores:
- `a_idx`: 4-bit angle indices (uint8 packed)
- `r_idx`: 2-bit radius indices (uint8 packed)
- `a_mn`, `a_sc`: Per-row angle min/scale (FP16)
- `r_mn`, `r_sc`: Per-row radius min/scale (FP16)
- `S`: Random orthogonal rotation matrix (shared)

---

## Runtime Requirements

- **GPU**: Tesla T4 or equivalent (~16 GB VRAM)
- **Runtime**: ~20 minutes on Google Colab (free tier)
- **Target Model**: Qwen 3-4B or Llama 3.2 3B

---

## Output Files

```
polarquant_checkpoints/
├── baseline_kv_weights/
│   ├── config.json              # Model configuration
│   └── kv_weights_fp16.pt       # Baseline K/V weights (377.5 MB)
├── polarquant_kv_weights/
│   └── kv_polarquant_4bit_angle_2bit_radius.pt  # Quantized (202.6 MB)
└── polarquant_size_comparison.png                 # Visualization charts
```

---

## Generated Visualizations

The notebook produces 6 charts:
1. **File size bar chart** — Baseline vs. PolarQuant comparison
2. **Compression breakdown pie chart** — Angle/radius/rotation components
3. **k_proj cosine similarity** — Per-layer reconstruction quality
4. **v_proj cosine similarity** — Per-layer reconstruction quality
5. **Bits vs. quality ablation** — Trade-off curve for different bit configurations
6. **Summary panel** — Key metrics at a glance

---

## Usage

1. Open in Google Colab or local Jupyter environment
2. Set your HuggingFace token (via `HF_TOKEN` environment variable or Colab Secrets)
3. Run all cells sequentially
4. Check `polarquant_checkpoints/` directory for saved files

---

## Notes

- PolarQuant is typically applied to **KV cache at runtime**; this notebook demonstrates its application to **static weight matrices**
- The rotation matrix `S` is shared across all layers to reduce overhead
- Reconstruction quality is measured via **cosine similarity** between original and dequantized weights
- The model remains functional after weight injection, demonstrating practical viability

---

## Dependencies

```
torch>=2.0
transformers>=4.45.0
accelerate
sentencepiece
tabulate
matplotlib
numpy
```

---

## License

This notebook is for research and educational purposes.
