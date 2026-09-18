# Confusion Adjacency Graph Correction

Official implementation of **Test-Time Classification Refinement via Confusion Adjacency Graphs**.

CAGC is a training-free decision-level correction method. A directed confusion graph is built once on a validation split. At test time, low-confidence predictions whose runner-up direction is inconsistent with historical confusion are iteratively re-ranked. The method requires no test labels, gradients, or parameter updates.

## Implemented functionality

- Unified probability-score interface for three classifier families:
  - Family I: zero-shot text-aligned classifiers such as CLIP.
  - Family II: linear probes trained on frozen visual features.
  - Family III: support-set prototype classifiers.
- Directed confusion graph construction from validation predictions.
- Filter-then-rank out-neighbor selection with `theta` followed by Top-`k`.
- Exact CAGC inference rule with confidence threshold, margin threshold, and maximum correction steps.
- Optional topology-aware gating from the paper analysis.
- Operational topology analysis for symmetric adjacency, hub absorption, asymmetric transmission, well-separated classes, and residual mixed classes.
- ImageNet-C evaluation grouped by noise, blur, weather, and digital corruptions.
- Multi-episode few-shot prototype evaluation.
- Baseline and CAGC accuracy, paired bootstrap testing, graph overlap, and correction-behavior reporting.
- Score-file adapters for cascading CAGC after external TTA methods including TPT, GS-Bias, DPE, TDA, SCA, PTA, BATCLIP, and Tent.

## Repository layout

```text
cagc/                    Core library
configs/                 Experiment configurations
data/                    Dataset manifest templates
scripts/                 Feature, graph, training, and evaluation entry points
examples/                Minimal API example
checkpoints/             Checkpoint placeholders
outputs/                 Default experiment outputs
```

## Installation

```bash
git clone <repository-url>
cd CAGC
python -m pip install -r requirements.txt
```

## Checkpoints

Checkpoint paths are intentionally empty in the released configuration files. Trained weights and processed artifacts will be released after the paper is accepted. Add local checkpoint paths only inside private configuration copies and do not commit proprietary or unreleased weights.

## Data format

ImageNet should use the standard `torchvision.datasets.ImageFolder` layout. ImageNet-C and fine-grained datasets use a CSV manifest with the following columns:

```csv
path,label
relative/image/path.jpg,0
```

For ImageNet-C, place `manifest.csv` under:

```text
<root>/<corruption>/<severity>/manifest.csv
```

The manifest paths are interpreted relative to that directory.

## Default CAGC configuration

```yaml
cagc:
  gamma: 0.6
  tau: 0.15
  max_steps: 3

graph:
  edge_threshold: 0.05
  top_k: 5

temperature: 0.01
```

The graph operator always filters by `theta` first and then applies Top-`k`.

## Workflow

### 1. Extract frozen visual features

```bash
python scripts/extract_features.py \
  --config configs/default.yaml \
  --dataset imagenet_validation
```

### 2. Produce validation scores

Family I:

```bash
python scripts/export_text_embeddings.py \
  --config configs/default.yaml \
  --class-names data/class_names.txt \
  --output outputs/text_embeddings.npy

python scripts/predict_scores.py \
  --config configs/default.yaml \
  --family I \
  --features outputs/features/imagenet_validation.npz \
  --text-embeddings outputs/text_embeddings.npy \
  --output outputs/scores/imagenet_validation.npz
```

Family II:

```bash
python scripts/train_linear_probe.py \
  --config configs/default.yaml \
  --train-features outputs/features/imagenet_train.npz \
  --output checkpoints/linear_probe.pt

python scripts/predict_scores.py \
  --config configs/default.yaml \
  --family II \
  --features outputs/features/imagenet_validation.npz \
  --probe-checkpoint checkpoints/linear_probe.pt \
  --output outputs/scores/imagenet_validation.npz
```

Family III:

```bash
python scripts/predict_scores.py \
  --config configs/default.yaml \
  --family III \
  --features outputs/features/imagenet_validation.npz \
  --support-features outputs/features/support.npz \
  --output outputs/scores/imagenet_validation.npz
```

### 3. Build the confusion adjacency graph

```bash
python scripts/build_confusion_graph.py \
  --config configs/default.yaml \
  --scores outputs/scores/imagenet_validation.npz \
  --targets outputs/features/imagenet_validation.npz \
  --output outputs/graphs/imagenet_graph.json
```

### 4. Evaluate baseline and CAGC

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --scores outputs/scores/imagenet_test.npz \
  --targets outputs/features/imagenet_test.npz \
  --graph outputs/graphs/imagenet_graph.json \
  --output outputs/reports/imagenet_cagc.json
```

### 5. Cascade CAGC after an external TTA method

Save the external method's adapted probabilities as an NPZ file containing `scores`, or logits as `logits`, then run:

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --scores outputs/scores/imagenet_test.npz \
  --targets outputs/features/imagenet_test.npz \
  --graph outputs/graphs/imagenet_graph.json \
  --adapter-method DPE \
  --adapter-scores outputs/scores/dpe_imagenet_test.npz \
  --output outputs/reports/dpe_cagc.json
```

Supported adapter names are `TPT`, `GS-Bias`, `DPE`, `TDA`, `SCA`, `PTA`, `BATCLIP`, and `Tent`.

### 6. Run multi-episode prototype evaluation

```bash
python scripts/run_few_shot.py \
  --config configs/default.yaml \
  --support-features outputs/features/support_pool.npz \
  --query-features outputs/features/query.npz \
  --graph outputs/graphs/support_graph.json \
  --output outputs/reports/few_shot.json
```

### 7. Evaluate all ImageNet-C corruptions

```bash
python scripts/run_imagenet_c.py \
  --config configs/default.yaml \
  --scores-dir outputs/scores/imagenet_c \
  --targets outputs/features/imagenet_c_targets.npz \
  --graph outputs/graphs/imagenet_graph.json \
  --output outputs/reports/imagenet_c.json
```

The score directory should contain one file per corruption, such as `gaussian_noise.npz` and `jpeg_compression.npz`.

### 8. Analyze graph topology and corrections

```bash
python scripts/analyze_topology.py \
  --graph outputs/graphs/imagenet_graph.json \
  --output outputs/reports/topology.json

python scripts/analyze_corrections.py \
  --config configs/default.yaml \
  --scores outputs/scores/imagenet_test.npz \
  --targets outputs/features/imagenet_test.npz \
  --graph outputs/graphs/imagenet_graph.json \
  --output outputs/reports/correction_behavior.json
```

## Minimal API example

```python
from cagc import CAGCCorrector, ConfusionAdjacencyGraph

graph = ConfusionAdjacencyGraph.from_predictions(
    validation_predictions,
    validation_targets,
    num_classes=num_classes,
)
corrector = CAGCCorrector(graph.out_neighbors)
corrected_predictions = corrector.correct(test_scores)
```

See `examples/quickstart.py` for a complete synthetic example.

## Scope of baseline integrations

CAGC is orthogonal to existing test-time adaptation methods and consumes their adapted score matrices. This repository provides a stable score-file interface for those methods rather than vendoring their official implementations. This keeps baseline licensing, training procedures, and checkpoint provenance separate from the CAGC implementation.

## License

This project is released under the MIT License.
