# AI Safety Research Template

## The Workflow

**Specification-driven experiments in 2 steps**:

1. **Write a spec** describing what you want to test (`specs/experiment_template.md`)
2. **Give it to Claude** → Get complete, runnable code + config + bash script
3. **Run the script** → Results logged automatically

That's it. Simple and reproducible.

## Quick Start

```bash
# 1. Use this template
git clone [this-repo] my-research

# 2. Optional: Clean up template files
./cleanup-template.sh

# 3. Create your source directory
mkdir -p src/{models,data,utils}

# 4. Update CLAUDE.md with your project details
# Fill in [TODO] sections

# 5. Write your first experiment spec
cp specs/experiment_template.md specs/my_experiment.md
# Edit the spec with what you want to test

# 6. In Claude Code, say: "implement specs/my_experiment.md"
# Claude creates code, config, and bash script

# 7. Run the experiment
./experiments/scripts/run_YYYY-MM-DD_my_experiment.sh
```

## Directory Structure

```
my-research/
├── src/                # Your implementation code
│   ├── models/        # Model architectures
│   ├── data/          # Data loading
│   └── utils/         # Helper functions
├── experiments/        # Everything experiment-related
│   ├── configs/       # YAML configs (Hydra/OmegaConf)
│   ├── scripts/       # Bash scripts to run experiments
│   ├── results/       # Outputs, plots, metrics
│   └── logs/          # Training logs
├── tests/             # Your tests
├── specs/             # Experiment specifications
│   ├── experiment_template.md      # Template
│   └── example_sae_experiment.md   # Example
├── ai_docs/papers/    # Research papers
└── .claude/           # Claude configuration (hooks, commands)
```

## Key Features

- **Specification-driven**: Write what you want, get runnable code
- **Reproducible**: Every experiment has config + script
- **Simple**: Two directories - `src/` for code, `experiments/` for runs
- **Automated**: Ruff formatting, file organization, git hooks
- **YAML configs**: Use Hydra or OmegaConf for all hyperparameters

## Example: SAE Experiment

See `specs/example_sae_experiment.md` for a complete example of:
- How to write a spec
- What Claude will implement
- The resulting code structure

## Available Commands

- `/crud-claude-commands` - Create custom commands for your workflow
- `/page` - Save session state when context fills
- `/clean-and-organize` - Clean temp files and organize repo

## Philosophy

**Keep it simple**:
- Write spec → Get code → Run experiment → Get results
- No complex tooling, just Python + YAML + bash
- Focus on research, not infrastructure