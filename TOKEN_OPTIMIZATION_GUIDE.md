# AI Coding Assistant Token Optimization Guide

This guide helps you reduce token usage when using AI coding assistants like Cursor, Claude Code, Codex, and Windsurf.

## Quick Start

### 1. Install Token Optimizer Plugin (Claude Code)
```bash
/plugin marketplace add alexgreensh/token-optimizer
/plugin install token-optimizer@alexgreensh-token-optimizer
```

Then enable auto-update:
```
/plugin → Marketplaces tab → select alexgreensh-token-optimizer → Enable auto-update
```

### 2. Use Token-Saving Commands
- `/token-optimizer` - Run optimization audit
- `/token-coach` - Get coaching on token usage
- `/token-optimizer quick` - Quick optimization
- `/compact` - Compress conversation history
- `/reset` - Clear conversation context

## Core Principles

### Be Direct and Specific
❌ "I think there might be a performance issue in the user authentication module, could you take a look and see if you can find any problems and suggest improvements?"

✅ "Fix performance issue in backend/auth.py line 45-50"

### Use File References
❌ "Here's the content of my main.py file: [pastes 200 lines]"

✅ "@backend/main.py - fix the bug in the user login function"

### Request Minimal Changes
❌ "Refactor this entire module to be more maintainable"

✅ "Extract the validation logic from process_user() into a separate function"

### Avoid Explanations
❌ "Can you explain how this works and then fix it?"

✅ "Fix the bug in the error handling"

## Conversation Management

### Start Fresh for New Tasks
- Clear context when switching between unrelated features
- Use `/reset` when starting a new major task
- Avoid long conversations spanning multiple features

### Keep Sessions Focused
- Break large tasks into smaller, focused sessions
- One session = one feature or bug fix
- Clear context between sessions

### Monitor Context Size
- Use `/token-optimizer` to check context usage
- Compact when context exceeds 70% capacity
- Remove unnecessary files from context

## File Management

### Use .cursorignore
The `.cursorignore` file in this repo excludes unnecessary files from AI context:
- Dependencies (.venv, node_modules)
- Build artifacts (dist, build)
- Data files (data/, *.csv)
- Logs (logs/, *.log)
- Cache (.cache/, *.cache)

### Reference Files Smartly
- Use `@filename` instead of pasting content
- Only include relevant files
- Remove files from context when done

### Limit File Scope
- Focus on specific files: "Fix bug in backend/auth.py"
- Avoid: "Fix bug in the authentication system"
- Use line numbers when possible

## Prompt Patterns

### Bug Fixes
❌ "There's a bug somewhere in the code, can you find it?"
✅ "Fix the TypeError in backend/main.py line 123"

### Feature Additions
❌ "Add a new feature for user preferences"
✅ "Add a save_preference() function to backend/user.py that accepts user_id and preference_data"

### Refactoring
❌ "Improve the code quality"
✅ "Extract the duplicate validation logic into a validate_input() helper function"

### Testing
❌ "Write comprehensive tests"
✅ "Add unit tests for the calculate_score() function covering edge cases: empty input, negative numbers, and null values"

## Tool-Specific Tips

### Cursor
- Use `.cursorrules` file for project-specific rules
- Leverage @file references
- Use Cmd+K for quick inline edits
- Clear context with Cmd+Shift+P → "Cursor: Clear Context"

### Claude Code
- Install token-optimizer plugin
- Use `/compact` to compress history
- Enable prompt caching in settings
- Use `/token-coach` for usage insights

### Windsurf
- Use `.windsurfrules` for project rules
- Leverage /compact, /focus, /clear commands
- Start new conversations for different tasks
- Use @file references

### Codex
- Keep prompts under 2000 characters when possible
- Use specific function names
- Provide only necessary context
- Request code-only outputs

## Advanced Techniques

### Prompt Caching
- Keep system prompts consistent
- Reuse common prompt patterns
- Cache frequently used code snippets
- Use prompt templates for repeated tasks

### Context Compression
- Remove redundant information
- Summarize long conversations
- Focus on current task only
- Remove resolved issues from context

### Model Selection
- Use faster models for simple tasks
- Reserve powerful models for complex reasoning
- Consider task complexity before model selection
- Use model routing when available

## Monitoring Token Usage

### Check Usage Stats
```bash
# Claude Code with token-optimizer
/token-optimizer

# View dashboard
/token-coach
```

### Track Patterns
- Monitor which tasks use most tokens
- Identify high-waste patterns
- Adjust prompting strategy based on data
- Set token budgets per task

## Common Token Wastes

### Structural Waste
- Bloated configuration files
- Unused skills/plugins
- Duplicate system prompts
- Stale memory/context

### Runtime Waste
- Verbose command output
- Oversized file reads
- Repeated file reads
- Long error messages

### Behavioral Waste
- Letting cache expire
- Compacting too late
- Looping on failing approaches
- Switching models mid-session

## Expected Savings

Following these practices typically reduces token usage by:
- **Direct prompts**: 40-60% reduction
- **File references**: 50-70% reduction  
- **Focused sessions**: 60-80% reduction
- **Plugin optimization**: 70-90% reduction
- **Combined**: 70-85% total reduction

## Resources

- [token-optimizer GitHub](https://github.com/alexgreensh/token-optimizer)
- [Cursor Documentation](https://cursor.sh/docs)
- [Claude Code Documentation](https://docs.anthropic.com/claude/code)
- [Windsurf Documentation](https://windsurf.ai/docs)

## Project-Specific Files

This repository includes:
- `.cursorrules` - Cursor-specific optimization rules
- `.windsurfrules` - Windsurf-specific optimization rules  
- `.claudecoderules` - Claude Code-specific optimization rules
- `.cursorignore` - Files to exclude from AI context
- `TOKEN_OPTIMIZATION_GUIDE.md` - This guide

Use these files as templates and customize for your workflow.
