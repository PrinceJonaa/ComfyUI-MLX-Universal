#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from datetime import datetime

ROADMAP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "roadmap.md")


def get_current_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def read_roadmap():
    if not os.path.exists(ROADMAP_PATH):
        print(f"Error: Could not find {ROADMAP_PATH}")
        sys.exit(1)
    with open(ROADMAP_PATH, "r") as f:
        return f.read()


def write_roadmap(content):
    with open(ROADMAP_PATH, "w") as f:
        f.write(content)


def complete_task(task_id):
    content = read_roadmap()

    # 1. Update timestamp
    today = datetime.now().strftime("%Y-%m-%d")
    commit = get_current_commit()
    content = re.sub(
        r"> Last curated: \d{4}-\d{2}-\d{2} at commit [a-z0-9]+",
        f"> Last curated: {today} at commit {commit}",
        content,
    )

    # 2. Extract the task block
    # A task block looks like:
    # ### [RM-XXX] Title
    # - Status: Planned
    # - Evidence: ...
    # - Why it matters: ...
    # (ends at double newline or next heading)

    pattern = rf"(### \[\s*{task_id}\s*\].*?)(?=\n### |\n## |\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"Error: Could not find task {task_id} in {ROADMAP_PATH}")
        sys.exit(1)

    task_block = match.group(1).strip()

    # Remove it from wherever it currently is
    content = content.replace(match.group(1), "")

    # Clean up multiple blank lines left behind
    content = re.sub(r"\n{3,}", "\n\n", content)

    # 3. Update the status in the extracted block
    if "- Status: Planned" in task_block:
        task_block = task_block.replace("- Status: Planned", "- Status: Completed")
    else:
        # If it doesn't have a status line, append completed to the title
        lines = task_block.split("\n")
        lines[0] = f"{lines[0]} — completed {today}"
        task_block = "\n".join(lines)

    # 4. Insert into "Recently Completed"
    # Find the "## Recently Completed" header and insert right after it
    rc_pattern = r"(## Recently Completed\n)"
    if not re.search(rc_pattern, content):
        print("Error: Could not find '## Recently Completed' section in roadmap.md")
        sys.exit(1)

    insertion = f"## Recently Completed\n\n{task_block}\n"
    content = re.sub(rc_pattern, insertion, content, count=1)

    write_roadmap(content)
    print(f"Successfully completed {task_id} and moved to 'Recently Completed'.")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "complete":
        print("Usage: python scripts/roadmap.py complete RM-XXX")
        sys.exit(1)

    task_id = sys.argv[2]
    complete_task(task_id)
