
# LVAD Prediction

This repository contains deep learning models and evaluation scripts for LVAD prediction. It supports comparisons across multiple model architectures and configurable frequency ranges and mel-frequency resolutions.

## Supported Models

Use the following identifiers with the `--model` argument:

| Model | Identifier |
|---|---|
| CNN | `cnn_only` |
| CNN–RNN | `cnn_rnn` |
| Vision Transformer (ViT) | `vit` |
| ResNet-18 | `resnet18` |
| DenseNet-121 | `densenet121` |
| EfficientNet-B0 | `efficientnet_b0` |

## Holdout Evaluation

Use `holdout_eval.py` to run holdout evaluation for a selected model and input configuration.

For example, to evaluate ResNet-18 using the **1–200 Hz** frequency range and **8 mel bins**:

```bash
python holdout_eval.py \
    --model resnet18 \
    --n-mels 8 \
    --min-freq 1 \
    --max-freq 200 \
    --no-show
```

### Arguments

| Argument | Description |
|---|---|
| `--model` | Model identifier from the table above. |
| `--n-mels` | Number of mel-frequency bins. |
| `--min-freq` | Lower frequency bound in Hz. |
| `--max-freq` | Upper frequency bound in Hz. |
| `--no-show` | Disable interactive plot display. |

To evaluate another architecture, replace the model identifier. For example, to evaluate ViT with the same frequency configuration:

```bash
python holdout_eval.py \
    --model vit \
    --n-mels 8 \
    --min-freq 1 \
    --max-freq 200 \
    --no-show
```

The frequency range can also be adjusted. For example, to evaluate CNN using **200–400 Hz** and **8 mel bins**:

```bash
python holdout_eval.py \
    --model cnn_only \
    --n-mels 8 \
    --min-freq 200 \
    --max-freq 400 \
    --no-show
```

## Reliability Diagrams

Use `plot_reliability.py` to generate reliability diagrams from holdout evaluation results:

```bash
python plot_reliability.py results_holdout_*
```

The wildcard `results_holdout_*` selects matching result paths in the current directory. Ensure that the intended evaluation results are available before running this command.

Reliability diagrams help assess calibration by comparing predicted probabilities with observed outcome frequencies.

## DeLong Test

Run the DeLong analysis using:

```bash
python delong_test.py
```

Before running the script, check that its input paths and model comparisons correspond to the intended holdout evaluation results.

Paired DeLong tests compare the areas under correlated ROC curves, such as those obtained from models evaluated on the same observations. Predictions must be aligned so that each model is evaluated against the same outcome labels in the same order.

Conventional paired DeLong tests do not account for within-patient clustering when multiple segments or recordings are obtained from the same patient. In this setting, the resulting p-values should be interpreted as nominal, exploratory comparisons rather than definitive evidence of model superiority.

## Suggested Evaluation Workflow

1. Select the model architecture, frequency range, and number of mel bins.
2. Run holdout evaluation using `holdout_eval.py`.
3. Generate reliability diagrams from the corresponding evaluation results.
4. Run `delong_test.py` for the intended AuROC comparisons.

For consistent comparisons, use the same data split, preprocessing configuration, and evaluation unit across models.

## Reproducibility

When reporting an experiment, record the following information:

- Model architecture and checkpoint.
- Training, validation, and test splits.
- Frequency range and number of mel bins.
- Random seed and software environment.
- Evaluation level, such as segment or recording level.
- Method used to estimate uncertainty and account for repeated observations from the same patient.

To reproduce a specific experiment from the paper, use its corresponding configuration and data split.
