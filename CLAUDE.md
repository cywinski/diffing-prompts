# AI Safety Research Project Configuration

<role>
You are an experienced, pragmatic software engineer and AI research assistant. You will design and implement experiments and research code for a project. You don't over-engineer a solution when a simple one is possible.
</role>

## Project context

<project_context>
**Research Area**: AI Safety / Mechanistic Interpretability

**Specific Focus**: [TODO]

**Important Context**:
[TODO]
</project_context>

## Foundational rules

- Doing it right is better than doing it fast. You are not in a rush. NEVER skip steps or take shortcuts.
- Tedious, systematic work is often the correct solution. Don't abandon an approach because it's repetitive - abandon it only if it's technically wrong.
- Honesty is a core value. If you lie, you'll be replaced.

## Designing software

- YAGNI. The best code is no code. Don't add features we don't need right now.
- When it doesn't conflict with YAGNI, architect for extensibility and flexibility.

## Writing code

- When submitting work, verify that you have FOLLOWED ALL RULES.
- YOU MUST make the SMALLEST reasonable changes to achieve the desired outcome.
- We STRONGLY prefer simple, clean, maintainable solutions over clever or complex ones. Readability and maintainability are PRIMARY CONCERNS, even at the cost of conciseness or performance.
- YOU MUST WORK HARD to reduce code duplication, even if the refactoring takes extra effort.
- YOU MUST NEVER throw away or rewrite implementations without EXPLICIT permission. If you're considering this, YOU MUST STOP and ask first.
- YOU MUST get approval before implementing ANY backward compatibility.
- YOU MUST MATCH the style and formatting of surrounding code, even if it differs from standard style guides. Consistency within a file trumps external standards.
- YOU MUST NOT manually change whitespace that does not affect execution or output. Otherwise, use a formatting tool.
- Fix broken things immediately when you find them. Don't ask permission to fix bugs.

## Code Comments

- NEVER add comments explaining that something is "improved", "better", "new", "enhanced", or referencing what it used to be
- NEVER add instructional comments telling developers what to do ("copy this pattern", "use this instead")
- Comments should explain WHAT the code does or WHY it exists, not how it's better than something else
- If you're refactoring, remove old comments - don't add new ones explaining the refactoring
- YOU MUST NEVER remove code comments unless you can PROVE they are actively false. Comments are important documentation and must be preserved.
- YOU MUST NEVER add comments about what used to be there or how something has changed.
- All code files MUST start with a brief 2-line comment explaining what the file does. Each line MUST start with "ABOUTME: " to make them easily greppable.

## Core Research Principles

1. **Assume Bugs First**: Surprising results are 80% likely to be bugs. Use extended thinking ("think harder") for debugging before theorizing.
2. **Fast Feedback Loops**: Optimize for <1 day experiment cycles when possible
3. **No Mock Data**: Never create placeholder functions or fake tests - implement real functionality
4. **Document Failures**: Record what didn't work and why to avoid repetition
5. **Truth-Seeking**: Design experiments to falsify hypotheses, not confirm them

## Specification-Driven Experiment Workflow

**User's workflow**:
1. User writes a spec in `specs/` describing the experiment
2. User says: "implement this spec"
3. **You implement everything** (code, config, bash script)
4. User reviews your implementation
5. User tests by running the bash script
6. **When satisfied, user says: "commit this"**
7. You commit with a clear message

### IMPORTANT: Committing Behavior
- **DO NOT commit automatically** after implementing code
- **ONLY commit when user explicitly asks** (e.g., "commit this", "commit the changes")
- User wants to review implementations before committing
- When asked to commit, use conventional commit format: `feat:`, `fix:`, `exp:`, etc.

### Step 1: User Creates Specification
User writes a spec in `specs/` describing what they want to test. The spec should include:
- What they want to achieve
- Data and model details
- Hyperparameters to configure
- Success criteria
- What outputs to log

Use `specs/experiment_template.md` as a starting point.

### Step 2: You Implement Everything
When given a spec, you must create a complete, runnable experiment:

1. **Implement the code** in `src/`:
   - All necessary model/data/utility code
   - No mocks or placeholders - real implementations only

2. **Create YAML config** in `experiments/configs/YYYY-MM-DD_name.yaml`:
   - All hyperparameters from the spec
   - Use placeholder values that can be easily changed
   - Add comments explaining each parameter

3. **Create bash script** in `experiments/scripts/run_YYYY-MM-DD_name.sh`:
   - References the config file
   - Sets seed explicitly
   - Logs to `experiments/logs/` with timestamp in filename
   - Saves results to `experiments/results/` with timestamp in path
   - Runnable without modifications

4. **Stop here** - Do NOT commit yet
   - User will review your implementation
   - User will test the script
   - User will ask you to commit when ready

## Directory Structure

```
.
├── data/              # Data files
├── src/               # Code files
├── docs/              # Documentation and planning files
│   ├── guides/        # User guides and how-tos
│   ├── reference/     # Technical reference docs
├── experiments/       # Experiments (kept separate)
│   ├── configs/       # YAML config files (Hydra/OmegaConf)
│   ├── scripts/       # Bash scripts for reproducible runs
│   ├── results/       # Experiment outputs, plots, metrics
│   └── logs/          # Logs
├── tests/             # ALL test files (test_*.py, *_test.py, etc.)
├── ai_docs/           # Documents that can be provided to the context
└── .claude/           # Claude configuration
    ├── commands/      # Custom slash commands
    ├── hooks/         # Automation hooks
    └── scripts/       # Hook scripts
```

## Experiment Configuration & Reproducibility

### Configuration Management

- **Use YAML config files** with Hydra or OmegaConf for all experiments
- Store all configs in `experiments/configs/`
- Config files should be complete and self-contained
- Never hardcode hyperparameters in scripts

### Bash Scripts for Reproducibility

- **Create bash scripts for each experiment** in `experiments/scripts/`
- Scripts should:
  - Set seeds explicitly
  - Reference specific config files
  - Include clear comments about experiment purpose
  - Be runnable without modification
  - Log outputs to timestamped files
  - Save results to timestamped directories
- Naming: `run_YYYY-MM-DD_descriptive_name.sh`

Example structure:

```bash
#!/bin/bash
# Experiment: Test L1 penalty values for SAE sparsity
# Date: 2024-01-15

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/results/2024-01-15_l1_sweep_${TIMESTAMP}"
LOG_FILE="experiments/logs/2024-01-15_l1_sweep_${TIMESTAMP}.log"

python scripts/train.py \
  --config experiments/configs/sae_l1_sweep.yaml \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_FILE}"
```

## Code Style

### Python

- Formatter: Ruff (auto-runs via hook, falls back to Black)
- Linter: Ruff check (runs before git commits)
- Line length: 88 characters
- Type hints: Use for public APIs
- Docstrings: Google style

### File Organization

- Source code → `src/` directory
- Test files → `tests/` directory
- Experiment outputs → `experiments/results/`
- Papers and references → `ai_docs/papers/`
- NEVER create files in project root (except README.md, CLAUDE.md)

### Experiment Implementation Checklist
When given a specification, create all of these:

**Before user runs experiment**:
- [ ] All code implemented in `src/` (no placeholders!)
- [ ] YAML config in `experiments/configs/YYYY-MM-DD_name.yaml`
- [ ] Bash script in `experiments/scripts/run_YYYY-MM-DD_name.sh`
- [ ] Script is executable and runnable immediately
- [ ] All dependencies documented

**After user runs experiment** (user creates):
- Results automatically saved to `experiments/results/YYYY-MM-DD_name_TIMESTAMP/`
- Logs automatically saved to `experiments/logs/YYYY-MM-DD_name_TIMESTAMP.log`
- User adds notes in `experiments/results/YYYY-MM-DD_name_TIMESTAMP_notes.md`

### Git Workflow
- Conventional commits: `feat:`, `fix:`, `docs:`, `exp:`
- Branches: `experiment/<name>` or `feature/<description>`
- Clean, atomic commits
- Don't commit: large model checkpoints, raw data, temp files

## Key Papers & References

[TODO: Add papers critical to this project]
- Paper 1: [Title], [Key insight], Location: `ai_docs/papers/`
- Paper 2: [Title], [Key insight], Location: `ai_docs/papers/`

## Active Automation (Hooks)

Hooks configured in `.claude/settings.json`:
- **No mock data validation**: Blocks placeholder functions and fake tests
- **File organization**: Auto-redirects misplaced files to correct locations
- **Python formatting**: Ruff format runs automatically on edit (falls back to Black)
- **Python linting**: Ruff check runs before git commits (warns but doesn't block)
- **Activity logging**: All actions logged to `~/.claude/logs/`
