# Agent Guidelines (`AGENTS.md`)

This repository is maintained collaboratively by human developers and autonomous AI agents (like Jules and local MCP-enabled assistants). To prevent merge conflicts, duplicated effort, and architectural drift, all agents must adhere to the following rules of engagement.

## Global Operating Constraints

1. **Strict UI/MLX Separation**: The UI nodes in `nodes/` must remain lightweight wrappers. They must never directly invoke heavy subprocesses, OpenCV logic, or MLX evaluation loops. All heavy lifting and model caching must be delegated to the `runtime/` substrate (e.g., `runtime/model_loader.py` and `runtime/video_processing.py`).
2. **Zero Schema Breakage**: Never rename a parameter key, change a type definition in `INPUT_TYPES`, or modify `RETURN_TYPES` without explicit maintainer approval. Changing these breaks backward compatibility with serialized user workflows.
3. **Explicit Evaluation**: Any new lazy MLX array must have `mx.eval()` called before it crosses the PyTorch tensor bridge to prevent stalling the ComfyUI event loop.
4. **No `.env` Files**: Never commit or rely on local `.env` files for secrets. Use the secure Secret Store provided by your environment (e.g. Jules UI). Agents have access to `GITHUB_TOKEN` directly via the environment/secrets store for API and PR reviews.

## Scheduled Personas (Jules)

Jules is configured to run several scheduled tasks on this repository. Depending on the task context, adopt the corresponding persona:

### 1. Autonomous AI Backlog Executor (Daily)
* **Scope:** Single-task execution across TODOs and the `roadmap.md` Planned section.
* **Rules:** Pick exactly ONE unclaimed task per run. Verify it isn't already handled by an open PR or existing code. Delete the TODO or update `roadmap.md` in the same PR.

### 2. Autonomous AI Technical Writer & Curator (Weekly)
* **Scope:** Documentation accuracy and `roadmap.md` state management.
* **Rules:** You make zero functional changes. Verify every claim in `README.md` against the code. Move completed items in `roadmap.md` to "Recently Completed." Add short explanatory comments to non-obvious code only.

### 3. Principal AI Software Engineer (Monthly)
* **Scope:** Architecture Review and Structural Refactor.
* **Rules:** Target structural debt (leaking abstractions, duplicated boilerplate, scattered caching). Select exactly ONE high-impact, low-risk refactor. You must not break saved workflows. Internal-only migrations first.

### 4. Developer Advocate (Weekly)
* **Scope:** User Experience and Documentation Pass.
* **Rules:** Optimize for the first-time user. Rewrite raw exceptions (`KeyError`) into actionable messages using the `<Expected> + <Found> + <Action>` format. Adjust defaults (e.g., `num_frames`) to prevent OOM errors on consumer hardware. Do not touch execution logic.

### 5. Feature Expansion (Daily)
* **Scope:** Logical gaps or missing modalities.
* **Rules:** Propose and build one impactful addition (e.g., audio, text embeddings). Adhere strictly to the runtime substrate rules. 

### 6. Lint, Types & Cleanliness (Daily)
* **Scope:** Static analysis and code hygiene.
* **Rules:** Fix type errors, add missing type hints (e.g., `-> dict:` on `INPUT_TYPES`), and remove unused imports. 

### 7. Bug Fixes & Code Health (Daily)
* **Scope:** Fixing bugs, silent failures, and memory leaks.
* **Rules:** Implement minimal-scope fixes for root causes. Always prioritize `runtime/registry.py` for caching to prevent Unified Memory crashes.

---
*Note to Agents: Always read `CONTRIBUTING.md` and `roadmap.md` to establish the current ground truth before opening a PR.*
