# Repository Modernization Plan

## Objective
Bring this repository from a solid educational implementation to a senior-level engineering artifact that is:

- reproducible
- testable
- maintainable
- benchmarked
- easy for others to run and evaluate

The current codebase already shows good intent: the transformer stack is split across `layers/`, `blocks/`, `models/`, and `utils/`, and there is a working end-to-end translation example. The main gap is not the core idea. It is the engineering surface around it.

## Current Assessment

### What is already working
- The repo has a clear transformer implementation structure.
- There is a runnable training example for EN-FR translation.
- Supporting utilities exist for tokenisation, logging, training, and BLEU scoring.
- The project has a basic README and dependency file.

### What currently reads as a student project
- The main workflow is driven by a script with hard-coded paths and hyperparameters in `examples/train_fr_en.py`.
- There is no packaging metadata such as `pyproject.toml`.
- There is no automated test suite.
- Reproducibility controls are minimal: no config system, no seed management, no experiment registry, no versioned artifacts.
- Evaluation is incomplete in the README and not presented as a benchmarkable result.
- Data and model artifacts are stored directly in the repository.
- Some implementation details need another pass for production-quality robustness, especially data loading, inference flow, and training orchestration.

## North Star
The repo should present as:

"A clean, reproducible PyTorch reference implementation of a transformer for sequence-to-sequence translation, with proper configs, tests, experiment tracking, and documented evaluation."

## Plan

## Status Update

### Completed
- Added `pyproject.toml` and migrated dependency management to `uv`.
- Removed `requirements.txt`.
- Upgraded and constrained the core runtime dependencies.
- Documented the supported Python version in project metadata.
- Added a `.gitignore` for local environment files and generated artifacts.
- Updated the README install and run instructions to use `uv`.

## Phase 1: Stabilize the Foundation
Goal: make the repo safe to run, easy to install, and predictable to maintain.

### Tasks
- [x] Add `pyproject.toml` and move dependency management away from a minimal `requirements.txt`-only setup.
- [ ] Introduce a proper package layout or confirm the existing layout with explicit import boundaries.
- [x] Add a `.gitignore` for IDE files, OS files, checkpoints, logs, and generated data artifacts.
- [x] Remove committed local-development noise such as `.DS_Store` and `.idea/` from version control policy.
- [x] Pin or constrain core dependencies more deliberately and document the supported Python version.
- [ ] Add a `Makefile` or simple task runner targets for `install`, `test`, `lint`, `format`, and `train`.

### Deliverable
- A new contributor should be able to clone the repo and run one documented setup command without guessing.

## Phase 2: Rework the User Interface of the Project
Goal: replace ad hoc scripts with a disciplined training and evaluation entrypoint.

### Tasks
- Convert `examples/train_fr_en.py` from a hard-coded script into a CLI entrypoint.
- Move training hyperparameters, model hyperparameters, and dataset paths into config files.
- Separate concerns:
  - dataset preparation
  - tokenizer training/loading
  - model construction
  - training
  - evaluation
  - inference
- Add a dedicated inference path instead of relying on `model(phrase_tokens)` after training.
- Replace `os.system("python data_prep.py")` with an explicit Python API or CLI subcommand.

### Deliverable
- Users can run commands like:
  - `python -m ... train --config configs/fr_en_base.yaml`
  - `python -m ... eval --checkpoint ...`
  - `python -m ... translate --text "..." --checkpoint ...`

## Phase 3: Improve Reproducibility and Experiment Management
Goal: make results repeatable and inspectable.

### Tasks
- Add global seed control for Python and PyTorch.
- Log exact run configuration, dependency versions, device info, and dataset metadata.
- Save checkpoints with structured metadata rather than bare `state_dict` only.
- Record train/validation metrics per epoch in machine-readable format such as JSON or CSV.
- Add deterministic or partially deterministic execution options where feasible.
- Version datasets and tokenizers by config, not by ad hoc filenames like `*_50_epochs.pkl`.

### Deliverable
- A run from six months later should be reproducible from config plus checkpoint.

## Phase 4: Raise Code Quality to Senior Level
Goal: make the implementation easier to trust, review, and extend.

### Tasks
- Add type hints consistently across the codebase.
- Tighten docstrings so they explain invariants and tensor shapes, not just generic descriptions.
- Review naming consistency:
  - `tokeniser` vs `tokenizer`
  - `optimiser` vs `optimizer`
- Refactor the training loop to support:
  - scheduler stepping
  - gradient clipping
  - mixed precision
  - checkpoint resume
  - separate train/eval/inference logic
- Review correctness of masking, padding, and loss indexing.
- Revisit the dataset implementation to avoid long-lived open file handles in object state.
- Replace print-based error handling with explicit exceptions and structured logging.

### Deliverable
- The codebase should read like a maintained reference implementation, not a one-off experiment.

## Phase 5: Add Tests That Prove Correctness
Goal: show engineering discipline through targeted automated verification.

### Tasks
- Add `pytest`.
- Write unit tests for:
  - tokenizer encode/decode round-trips
  - positional encoding shapes
  - attention masking behavior
  - multi-head attention output shapes
  - transformer forward-pass shapes
  - data collation and padding behavior
- Add small integration tests for:
  - one training step
  - checkpoint save/load
  - deterministic inference on a tiny toy batch
- Add CI to run tests and lint checks on every push.

### Deliverable
- The repo should fail fast when a refactor breaks tensor contracts or data assumptions.

## Phase 6: Make Evaluation Real
Goal: replace "TBC" results with credible experimental evidence.

### Tasks
- Define one or two benchmark configurations for the Europarl EN-FR task.
- Track:
  - validation loss
  - BLEU
  - training time
  - parameter count
- Add a reproducible evaluation script.
- Document what preprocessing and tokenization were used for each result.
- Include a short discussion of tradeoffs versus the original paper and modern transformer practice.

### Deliverable
- The README should contain a compact results table with exact config references and caveats.

## Phase 7: Modernize the README and Project Narrative
Goal: make the repo communicate seniority before anyone reads the source.

### Tasks
- Rewrite the README around:
  - project purpose
  - architecture overview
  - setup
  - training
  - evaluation
  - inference
  - results
  - limitations
  - roadmap
- Add an architecture diagram or a concise module interaction diagram.
- Explain design choices:
  - post-norm vs pre-norm
  - custom BPE tokenizer
  - iterable dataset design
  - translation task choice
- Add a "Lessons learned / future improvements" section that reflects your current senior perspective.

### Deliverable
- The README should position this as a thoughtful reference project, not just coursework.

## Phase 8: Decide the Strategic Direction
Goal: choose whether this remains an educational implementation or becomes a stronger research/engineering showcase.

### Option A: Keep it as a clean educational reference
- Focus on clarity, correctness, tests, and documentation.
- Keep dependencies light.
- Emphasize readable implementation over feature breadth.

### Option B: Turn it into a stronger engineering portfolio piece
- Add structured configs and experiment tracking.
- Support multiple datasets and tokenizer backends.
- Add distributed or mixed-precision training support.
- Compare against library baselines.

### Recommendation
Take a hybrid approach:
- keep the core implementation readable
- add professional tooling around it
- avoid turning it into a framework

That balance best demonstrates seniority.

## Suggested Execution Order
1. Repository hygiene and packaging
2. CLI plus config refactor
3. Reproducibility and checkpointing
4. Tests and CI
5. Evaluation pipeline
6. README rewrite and results publication

## Suggested First Milestone
Define a "v1 modernization" scope that includes only:

- `pyproject.toml`
- `.gitignore`
- `pytest` with core tensor-shape tests
- config-driven training entrypoint
- checkpoint save/load cleanup
- README rewrite

This is the smallest scope that will materially change how the repo is perceived.

## Concrete Signals of Seniority
If you want the repo to reflect your current level, optimize for these visible signals:

- clear boundaries between research code and reusable code
- reproducible experiments
- thoughtful failure handling
- tests that target invariants, not just happy paths
- honest documentation of limitations
- benchmarked results with exact configs
- clean setup and one-command workflows

## Anti-Goals
Do not spend early effort on:

- excessive abstraction
- adding many model variants before the base repo is stable
- polishing visuals before reproducibility exists
- premature optimization without benchmark baselines

## Proposed Outcome
After this plan, the repository should look like a deliberate reference implementation by an experienced engineer:

- easy to install
- easy to verify
- easy to extend
- backed by tests and results
- credible as both a learning resource and a portfolio artifact
