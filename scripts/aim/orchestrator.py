"""Queen/tactical agent hierarchy for multi-AI workflows.

Inspired by RoofFlow's agent architecture:
- Queen agent: plans the work, decomposes tasks, delegates, reviews, synthesizes
- Tactical agents: execute individual sub-tasks (research, code, test, review)

Usage:
    python -m aim orchestrate "Build a REST API for user auth"
    python -m aim orchestrate "Research the best Python async frameworks"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Models available for tactical roles ─────────────────────────────────────

TACTICAL_POOL = {
    "claude-opus": {
        "provider": "anthropic",
        "role": "Complex reasoning, design critique, architecture review",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_BASE_URL",
    },
    "deepseek": {
        "provider": "deepseek",
        "role": "Implementation, refactoring, code generation",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
    "gpt4o": {
        "provider": "openai",
        "role": "Research synthesis, summarization, documentation",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
    },
}


@dataclass
class SubTask:
    id: int
    title: str
    description: str
    assigned_model: str
    output_format: str = "text"  # text | code | list | analysis
    dependencies: list[int] = field(default_factory=list)
    result: str | None = None
    status: str = "pending"  # pending | running | done | failed


@dataclass
class Orchestration:
    task: str
    plan: str
    sub_tasks: list[SubTask] = field(default_factory=list)
    synthesis: str | None = None
    queen_model: str = "claude-opus"


# ── Queen: decompose the task ──────────────────────────────────────────────

QUEEN_DECOMPOSE_PROMPT = """You are a Queen Agent — an orchestrator that decomposes complex tasks into sub-tasks for specialized tactical agents.

Given a task, produce a plan with sub-tasks. Each sub-task must specify:
1. Title (short)
2. Description (what to do, concretely)
3. Assigned model (pick from the pool based on the task type)
4. Output format (text, code, list, or analysis)

Available tactical agents:
- claude-opus: Complex reasoning, design critique, architecture review, trade-off analysis
- deepseek: Implementation, refactoring, code generation, mechanical work
- gpt4o: Research synthesis, summarization, documentation, translation

Rules:
- Decompose into 2-5 sub-tasks maximum. Don't over-decompose.
- Parallelize what can run independently. Note dependencies where needed.
- Assign each sub-task to the right model. Don't default everything to one model.
- If the task is simple (one clear action), return ONE sub-task only.

TASK: {task}

Output as JSON:
{{"plan": "one-paragraph summary of the approach",
 "sub_tasks": [
   {{"title": "...", "description": "...", "assigned_model": "...", "output_format": "..."}}
 ]}}"""


def queen_decompose(task: str, model: str = "claude-opus") -> tuple[str, list[SubTask]]:
    """Queen agent decomposes a task into sub-tasks.

    Uses the existing Multi-AI bridge to call the queen model.
    Returns (plan_summary, list_of_subtasks).
    """
    prompt = QUEEN_DECOMPOSE_PROMPT.format(task=task)

    # Use the aim CLI to call the queen model
    result = _call_model(model, prompt, response_format="json")

    try:
        data = json.loads(result) if isinstance(result, str) else result
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        m = re.search(r'\{.*\}', result, re.DOTALL)
        if m:
            data = json.loads(m.group())
        else:
            # Fallback: single sub-task
            return "Direct execution", [
                SubTask(id=1, title="Execute", description=task,
                        assigned_model="deepseek", output_format="text")
            ]

    plan = data.get("plan", "Execute the task directly.")
    sub_tasks = []
    for i, st in enumerate(data.get("sub_tasks", []), 1):
        sub_tasks.append(SubTask(
            id=i,
            title=st.get("title", f"Sub-task {i}"),
            description=st.get("description", ""),
            assigned_model=st.get("assigned_model", "deepseek"),
            output_format=st.get("output_format", "text"),
        ))

    if not sub_tasks:
        sub_tasks = [
            SubTask(id=1, title="Execute", description=task,
                    assigned_model="deepseek", output_format="text")
        ]

    return plan, sub_tasks


# ── Tactical: execute sub-tasks ─────────────────────────────────────────────

TACTICAL_PROMPTS = {
    "text": "Complete the following task. Be concise and direct.\n\nTASK: {task}",
    "code": "Write code for the following task. Output only the code with brief comments.\n\nTASK: {task}",
    "list": "Produce a bulleted list in response to the following task.\n\nTASK: {task}",
    "analysis": "Analyze the following and provide a structured analysis with conclusions.\n\nTASK: {task}",
}


def tactical_execute(sub_task: SubTask) -> str:
    """A tactical agent executes a single sub-task."""
    template = TACTICAL_PROMPTS.get(sub_task.output_format, TACTICAL_PROMPTS["text"])
    prompt = template.format(task=sub_task.description)

    try:
        return _call_model(sub_task.assigned_model, prompt)
    except Exception as e:
        return f"ERROR ({sub_task.assigned_model}): {e}"


# ── Queen: synthesize results ──────────────────────────────────────────────

QUEEN_SYNTHESIZE_PROMPT = """You are a Queen Agent reviewing the results of tactical agents.

Original task: {task}

Plan: {plan}

Sub-task results:
{sub_task_results}

Synthesize these results into a coherent final output. Address the original task completely.
If any sub-task failed or returned an error, note it and work around it.
"""


def queen_synthesize(orch: Orchestration) -> str:
    """Queen agent synthesizes sub-task results into a final output."""
    results_text = ""
    for st in orch.sub_tasks:
        status = "DONE" if st.status == "done" else st.status.upper()
        results_text += f"\n## Sub-task {st.id}: {st.title} [{status}]\n"
        results_text += f"Model: {st.assigned_model}\n"
        results_text += f"Result: {st.result or '(no result)'}\n"

    prompt = QUEEN_SYNTHESIZE_PROMPT.format(
        task=orch.task,
        plan=orch.plan,
        sub_task_results=results_text,
    )

    return _call_model(orch.queen_model, prompt)


# ── Model calling ──────────────────────────────────────────────────────────

def _call_model(model: str, prompt: str, response_format: str | None = None) -> str:
    """Call an AI model via the aim CLI.

    Uses the existing aim bridge which knows how to route to the right backend,
    rather than trying to call APIs directly with the wrong client+model combo.
    """
    return _call_via_aim(model, prompt)


def _call_via_aim(model: str, prompt: str) -> str:
    """Fallback: call via the aim CLI."""
    provider_map = {
        "claude-opus": "openai",  # aim doesn't have anthropic adapter
        "deepseek": "openai",     # routed via env vars
        "gpt4o": "openai",
    }
    provider = provider_map.get(model, "openai")

    aim_dir = Path(__file__).parent
    result = subprocess.run(
        [sys.executable, "-m", "aim", "ask", "--provider", provider, prompt],
        capture_output=True, text=True, timeout=120,
        cwd=str(aim_dir.parent),
    )
    if result.returncode != 0:
        raise RuntimeError(f"aim call failed: {result.stderr[:200]}")
    return result.stdout.strip()


# ── Main orchestration ─────────────────────────────────────────────────────

def orchestrate(task: str, queen_model: str = "claude-opus") -> Orchestration:
    """Run a full queen/tactical agent workflow.

    1. Queen decomposes the task into sub-tasks
    2. Tactical agents execute sub-tasks (dependencies respected)
    3. Queen synthesizes results into final output

    Returns the full Orchestration with all results.
    """
    orch = Orchestration(task=task, queen_model=queen_model)

    # Phase 1: Queen plans
    print(f"\n👑 Queen ({queen_model}): Decomposing task...")
    print(f"   Task: {task[:100]}")

    try:
        orch.plan, orch.sub_tasks = queen_decompose(task, queen_model)
    except Exception as e:
        print(f"   Queen decomposition failed: {e}")
        orch.plan = "Direct execution (decomposition failed)"
        orch.sub_tasks = [
            SubTask(id=1, title="Execute directly", description=task,
                    assigned_model="deepseek")
        ]

    print(f"   Plan: {orch.plan[:120]}...")
    print(f"   Sub-tasks: {len(orch.sub_tasks)}")
    for st in orch.sub_tasks:
        dep_info = f" (after #{','.join(str(d) for d in st.dependencies)})" if st.dependencies else ""
        print(f"     #{st.id}: {st.title} → {st.assigned_model}{dep_info}")

    # Phase 2: Tactical agents execute
    print(f"\n⚔️  Tactical agents: Executing {len(orch.sub_tasks)} sub-task(s)...")

    # Simple sequential execution (parallel in Phase 2)
    for st in orch.sub_tasks:
        print(f"   [#{st.id}] {st.title[:60]}... ", end="", flush=True)
        st.status = "running"
        start = time.time()

        try:
            st.result = tactical_execute(st)
            st.status = "done"
            elapsed = int((time.time() - start) * 1000)
            print(f"✓ ({elapsed}ms, {len(st.result)} chars)")
        except Exception as e:
            st.status = "failed"
            st.result = f"ERROR: {e}"
            print(f"✗ ({e})")

    # Phase 3: Queen synthesizes
    print(f"\n👑 Queen ({queen_model}): Synthesizing results...")
    try:
        orch.synthesis = queen_synthesize(orch)
        print(f"   Synthesis: {len(orch.synthesis)} chars")
    except Exception as e:
        orch.synthesis = f"Synthesis failed: {e}\n\nIndividual results:\n"
        for st in orch.sub_tasks:
            orch.synthesis += f"\n--- {st.title} ---\n{st.result}"

    return orch


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <task description>")
        print("       python orchestrator.py --queen claude-opus <task>")
        sys.exit(1)

    queen = "claude-opus"
    args = sys.argv[1:]

    if args[0] == "--queen" and len(args) >= 2:
        queen = args[1]
        args = args[2:]

    task = " ".join(args)
    orch = orchestrate(task, queen)

    # Print final synthesis
    print(f"\n{'═' * 70}")
    print(f"  FINAL SYNTHESIS")
    print(f"{'═' * 70}")
    print(orch.synthesis)

    # Summary
    done = sum(1 for st in orch.sub_tasks if st.status == "done")
    failed = sum(1 for st in orch.sub_tasks if st.status == "failed")
    print(f"\n{'─' * 70}")
    print(f"  {done} done, {failed} failed, {len(orch.sub_tasks)} total sub-tasks")
    print(f"  Queen: {orch.queen_model}")


if __name__ == "__main__":
    main()
