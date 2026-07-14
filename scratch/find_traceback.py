from pathlib import Path

log_path = Path(r"C:\Users\pavan\.gemini\antigravity\brain\255ccf08-cfe9-417c-871a-91da2ca5ecb5\.system_generated\tasks\task-8734.log")
content = log_path.read_text(encoding="utf-8")

target = "test_v1_search_sanitizes_nan_optional_fields"
idx = 0
while True:
    idx = content.find(target, idx)
    if idx == -1:
        break
    print(f"Found occurrence at {idx}:")
    print(content[max(0, idx - 200):min(len(content), idx + 2000)])
    print("-" * 80)
    idx += len(target)
