# Agent Guidelines (`AGENTS.md`)

This repository is maintained collaboratively by human developers and autonomous AI agents (like Jules and local assistants). To prevent merge conflicts, duplicated effort, and architectural drift, all agents must adhere to the following rules of engagement.

---

## 1. Global Operating Constraints

1. **Strict UI/MLX Separation**: The UI nodes in `nodes/` must remain lightweight wrappers. They must never directly invoke heavy subprocesses, OpenCV logic, or MLX evaluation loops. All heavy lifting and model caching must be delegated to the `runtime/` substrate (e.g., `runtime/model_loader.py` and `runtime/video_processing.py`).
2. **Zero Schema Breakage**: Never rename a parameter key, change a type definition in `INPUT_TYPES`, or modify `RETURN_TYPES` without explicit maintainer approval. Changing these breaks backward compatibility with serialized user workflows.
3. **Explicit Evaluation**: Any new lazy MLX array must have `mx.eval()` called before it crosses the PyTorch tensor bridge to prevent stalling the ComfyUI event loop.
4. **No `.env` Files**: Never commit or rely on local `.env` files for secrets. Use the secure Secret Store provided by your environment (e.g., Jules UI). Agents have access to `GITHUB_TOKEN` directly via the environment/secrets store for API operations and PR reviews.

---

## 2. Multi-Agent Concurrency & Quality of Life (QOL)

To ensure that multiple agents working concurrently do not conflict or duplicate work, follow these rules:

* **De-duplication Check (PRs & Branches)**:
  * Before beginning any work, run:
    ```bash
    # Check for open/merged PRs related to the keyword/task ID
    gh pr list --state all --search "<keyword>"
    
    # Check already-present branches and commit logs for task references
    git branch -a
    git log --all --oneline -30 --grep="<keyword>"
    ```
  * If an open PR or active branch already addresses the task, **abort immediately**. Do not duplicate or open a competing PR.
* **Branch Naming Standard**:
  * Use the convention: `agent/<persona>/<task-id-or-short-name>`
  * Example: `agent/backlog-executor/rm-015`
* **Local Pre-Flight Verification & Formatting**:
  * Before calling your native `submit` action, you **must** run the local CI loop to auto-format your code and verify zero regressions:
    ```bash
    make verify
    ```
  * If `make verify` fails, you must fix the python tracebacks before submitting.
* **Clear Task Claiming**:
  * Update the task status in `roadmap.md` to `In Progress` immediately upon starting, to prevent other agents from selecting it.

---

## 3. Scheduled Personas (Jules)

Jules is configured to run several scheduled tasks on this repository. Depending on the task context, adopt the corresponding persona:

### 1. Autonomous AI Backlog Executor (Daily)
* **Scope:** Single-task execution across TODOs and the `roadmap.md` Planned section.
* **Backlog Selection Protocol**:
  1. Scan for `TODO/FIXME/XXX` comments or `roadmap.md` entries.
  2. Pick **exactly one** unclaimed task that has clear evidence and low blast radius.
  3. Re-verify in the current codebase that the task is indeed unbuilt (ground-truth check).
  4. Perform the implementation, remove the stale `TODO` comment. If this completes a roadmap task, you MUST use `python scripts/roadmap.py complete RM-XXX` instead of manually editing the markdown file. Finally, verify your code with `make verify`.
  5. Deliver a PR with the title format: `[Task] <Source: TODO|Roadmap RM-###|Feature> — <specific name>`.

### 2. Autonomous AI Technical Writer & Curator (Weekly)
* **Scope:** Documentation accuracy and `roadmap.md` state management.
* **Rules**:
  * Make **zero functional changes**. Only modify `.md` files (with the exception of adding short explanatory comments to non-obvious Python code blocks).
  * Correct discrepancies (docs lagging code) or translate claims into planned tasks in `roadmap.md` (docs ahead of code).
  * Reconcile completed roadmap entries and update the header's `Last curated` date and commit SHA.
  * Deliver a PR with the title format: `[Docs] <Area>: <what got synced>`.

### 3. Principal AI Software Engineer (Monthly)
* **Scope:** Architecture Review and Structural Refactor.
* **Rules**:
  * Do not refactor for aesthetic reasons. Every change must pay off a specific structural debt (duplicated boilerplate, leaking abstraction, cache scatter, tensor bridge inconsistency, etc.).
  * Assign an Impact/Risk score and select the highest ratio.
  * Make internal-only changes first (introduce new abstraction without removing old paths), migrate callers incrementally, then remove old paths.
  * Confirm backward compatibility (existing workflows load correctly and nodes register).
  * Deliver a PR with the title format: `[Refactor] <Component>: < consolidating/extracting description>`.

### 4. Developer Advocate (Weekly)
* **Scope:** User Experience and Documentation Pass.
* **Rules**:
  * Optimize error messages for the first-time user who has never read the source code.
  * Re-raise raw exceptions to use the standard format: `<expected> + <found> + <user action>`.
  * Tune parameter defaults (e.g. lowering resolutions/frames) to prevent OOM errors on common consumer setups (e.g. 16GB Macs).
  * Sync documentation for node parameters and options. Do not touch execution logic.
  * Deliver a PR with the title format: `[UX] <Area>: <description of what got clearer>`.

### 5. Feature Expansion (Daily)
* **Scope:** Logical gaps or missing modalities.
* **Rules**:
  * Identify gaps in modalities (e.g., audio, text embeddings).
  * Implement thin wrappers in `nodes/`, utilize `get_or_load_model` from `runtime/registry.py` for caching, route outputs through `runtime/bridge.py`, and dynamically register the node in `__init__.py`.
  * Call `mx.eval()` on lazy arrays.

### 6. Lint, Types & Cleanliness (Daily)
* **Scope:** Static analysis and code hygiene.
* **Rules:** Fix static type errors, add missing type hints (e.g., `-> dict:` on `INPUT_TYPES`), and remove unused imports.

### 7. Bug Fixes & Code Health (Daily)
* **Scope:** Fixing bugs, silent failures, and memory leaks.
* **Rules:** Implement minimal-scope fixes for root causes. Always prioritize `runtime/registry.py` for caching to prevent Unified Memory crashes.

---
*Note to Agents: Always read `CONTRIBUTING.md` and `roadmap.md` to establish the current ground truth before opening a PR.*
