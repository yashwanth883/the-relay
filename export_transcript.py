import os
import json

conv_id = 'cfce2f47-0ee3-426d-be13-2509a49883c0'
transcript_path = os.path.expanduser(f'~/.gemini/antigravity/brain/{conv_id}/.system_generated/logs/transcript.jsonl')
output_path = 'conversation_history.md'

with open(transcript_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

formatted_steps = []
formatted_steps.append("# Complete Conversation History with the Assistant\n")
formatted_steps.append("This log captures the actual prompts, tool calls, and agent interactions throughout the assignment.\n\n---\n")

for line in lines:
    if not line.strip():
        continue
    try:
        obj = json.loads(line)
        step_idx = obj.get("step_index", "")
        source = obj.get("source", "")
        step_type = obj.get("type", "")
        content = obj.get("content", "")
        created_at = obj.get("created_at", "")
        
        if step_type == "USER_INPUT":
            formatted_steps.append(f"### [Step {step_idx}] User Prompt ({created_at})\n\n```text\n{content}\n```\n\n---\n")
        elif step_type == "PLANNER_RESPONSE" and content:
            # Shortened summary if massive
            preview = content[:1500] + ("\n... [truncated for readability]" if len(content) > 1500 else "")
            formatted_steps.append(f"### [Step {step_idx}] Assistant Response ({created_at})\n\n{preview}\n\n---\n")
        elif "tool_calls" in obj and obj["tool_calls"]:
            tools_called = [tc.get("tool_name", "") for tc in obj["tool_calls"]]
            formatted_steps.append(f"**Step {step_idx} - Tool Execution:** `{', '.join(tools_called)}`\n\n")
    except Exception:
        pass

with open(output_path, 'w', encoding='utf-8') as f:
    f.writelines(formatted_steps)

print(f"Exported {len(lines)} transcript steps to {output_path}")
