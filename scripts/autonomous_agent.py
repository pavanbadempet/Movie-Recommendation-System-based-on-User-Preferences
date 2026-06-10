import os
import sys
import json
import subprocess
import requests

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def ask_llm(prompt, hf_token):
    url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}

    # Simple prompt formatting for Qwen
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    data = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 1500,
            "return_full_text": False,
            "temperature": 0.1
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"Error from HF API: {response.text}")
        return ""

    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", "")
    return ""

def main():
    issue_number = os.environ.get("ISSUE_NUMBER")
    hf_token = os.environ.get("HF_TOKEN")

    if not issue_number:
        print("No ISSUE_NUMBER provided.")
        sys.exit(1)

    if not hf_token:
        print("No HF_TOKEN provided.")
        sys.exit(1)

    print(f"Fetching issue #{issue_number}...")
    stdout, stderr, rc = run_cmd(f"gh issue view {issue_number} --json title,body")
    if rc != 0:
        print(f"Failed to fetch issue: {stderr}")
        sys.exit(1)

    issue_data = json.loads(stdout)
    issue_title = issue_data.get("title", "")
    issue_body = issue_data.get("body", "")

    print(f"Issue Title: {issue_title}")

    log_content = f"## Autonomous Agent Execution Log\n\n**Issue**: #{issue_number} - {issue_title}\n\n"

    # Step 1: Analysis & Searching
    print("Step 1: Analyzing the issue and finding relevant files...")
    search_prompt = f"""
    You are an autonomous debugging agent. An issue has been reported in the system:
    Title: {issue_title}
    Body: {issue_body}

    Based on the error, suggest exactly 2 or 3 bash commands (like grep or find) to search the repository for the files related to this issue.
    Return ONLY the bash commands, one per line. Do not include markdown blocks or any explanation.
    """

    search_cmds_text = ask_llm(search_prompt, hf_token)
    search_cmds = [cmd.strip() for cmd in search_cmds_text.strip().split('\n') if cmd.strip() and not cmd.startswith('```')]

    log_content += "### Step 1: Investigation\n"
    log_content += "The agent executed the following commands to investigate:\n"

    investigation_results = ""
    for cmd in search_cmds[:3]:  # Limit to 3 cmds for safety
        log_content += f"- `{cmd}`\n"
        print(f"Running: {cmd}")
        stdout, stderr, _ = run_cmd(cmd)
        out = stdout[:1000] # truncate output
        investigation_results += f"Command: {cmd}\nOutput:\n{out}\n\n"

    # Step 2: Proposing a Fix
    print("Step 2: Proposing a fix...")
    fix_prompt = f"""
    You are an autonomous coding agent.
    Issue: {issue_title}
    {issue_body}

    Investigation Results:
    {investigation_results}

    Propose a bash script that will fix this issue. The script should use tools like `sed`, `awk`, or `cat << 'EOF' > file.py` to modify or create files.
    Return ONLY the raw bash script. Do not include ```bash wrappers or any explanation.
    Ensure the script is safe and directly addresses the issue.
    """

    fix_script = ask_llm(fix_prompt, hf_token)

    # Strip markdown blocks if the LLM included them
    if fix_script.startswith("```bash"):
        fix_script = fix_script[7:]
    if fix_script.startswith("```"):
        fix_script = fix_script[3:]
    if fix_script.endswith("```"):
        fix_script = fix_script[:-3]

    fix_script = fix_script.strip()

    log_content += "\n### Step 2: Applied Fix\n"
    log_content += "The agent applied the following script:\n```bash\n" + fix_script + "\n```\n"

    with open("apply_fix.sh", "w") as f:
        f.write(fix_script)

    stdout, stderr, rc = run_cmd("bash apply_fix.sh")
    log_content += f"\n**Execution Output**:\n```\n{stdout}\n{stderr}\n```\n"
    print("Fix script executed.")

    # Check if files were actually changed
    stdout, _, _ = run_cmd("git status --porcelain")
    if not stdout.strip():
        log_content += "\n**Note**: The fix script did not result in any changed files.\n"
        print("No files changed.")
    else:
        log_content += "\n**Modified Files**:\n```\n" + stdout + "\n```\n"
        print(f"Files modified:\n{stdout}")

    with open("agent_log.md", "w") as f:
        f.write(log_content)

    print("Agent process complete.")

if __name__ == "__main__":
    main()
