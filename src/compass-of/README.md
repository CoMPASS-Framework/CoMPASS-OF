# Level-1

This folder contains the Python implementation of the **Level-1 model**. The workflow converts ordered position coordinates into movement datastreams and fits a hidden Markov model (HMM) using:

- **Step length** modeled with a Gamma distribution
- **Turning angle** modeled with a von Mises distribution

The default analysis fits **two behavioral states** and reorders them after fitting so that:

1. **State 1 — Surveillance:** lower step length and greater turning variability
2. **State 2 — Ambulatory:** higher step length and lower turning variability

## Folder contents

```text
level-1/
├── datastreams.py
├── model.py
└── README.md
```

### `datastreams.py`

Creates the movement datastreams used by the Level-1 model.

Public functions:

- `prep_data(...)`
- `plot_step_and_angle_distributions(...)`

### `model.py`

Contains the Gamma–von Mises HMM, model fitting, model selection, state reordering, posterior probabilities, Viterbi decoding, and summary functions.

Main public objects and functions:

- `GammaVonMisesHMM`
- `fit_best_hmm_momentu_style(...)`
- `reorder_lowstep_highturn_first(...)`
- `summarize_hmm(...)`
- `print_hmm_summary(...)`
- `compute_sequence_state_probabilities(...)`
- `BestResult`

