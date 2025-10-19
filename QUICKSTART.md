# Hacker News MCP Server - Quick Start Guide

Get the Hacker News MCP server running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Windows 10/11 with PowerShell or Git Bash

## Quick Installation

### Option 1: Automated Installation (Recommended)

**PowerShell:**
```powershell
cd C:\Users\simon\Downloads\projects\claude_code_skills\skill_create_hn_mcp_server
.\install.ps1
```

**Git Bash:**
```bash
cd /c/Users/simon/Downloads/projects/claude_code_skills/skill_create_hn_mcp_server
bash install.sh
```

The script will:
1. Check Python installation
2. Create virtual environment
3. Install dependencies
4. Test the installation
5. Display configuration instructions

### Option 2: Manual Installation

```bash
# Navigate to project directory
cd C:\Users\simon\Downloads\projects\claude_code_skills\skill_create_hn_mcp_server

# Create virtual environment
python -m venv venv

# Activate virtual environment
# PowerShell:
.\venv\Scripts\Activate.ps1
# Git Bash:
source venv/Scripts/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test installation
python -c "from mcp.server.fastmcp import FastMCP; print('Success!')"
```

## Configure Claude Desktop

### Step 1: Find Configuration File

Open File Explorer and navigate to:
```
%APPDATA%\Claude\claude_desktop_config.json
```

Or in PowerShell:
```powershell
notepad $env:APPDATA\Claude\claude_desktop_config.json
```

### Step 2: Get Full Paths

**PowerShell:**
```powershell
# Get Python path
(Get-Item venv\Scripts\python.exe).FullName

# Get server path
(Get-Item hackernews_mcp.py).FullName
```

**Git Bash:**
```bash
# Get Python path
cygpath -w "$(pwd)/venv/Scripts/python.exe"

# Get server path
cygpath -w "$(pwd)/hackernews_mcp.py"
```

### Step 3: Edit Configuration

Add this to your `claude_desktop_config.json` (use the paths from Step 2):

```json
{
  "mcpServers": {
    "hackernews": {
      "command": "C:\\Users\\simon\\Downloads\\projects\\claude_code_skills\\skill_create_hn_mcp_server\\venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\simon\\Downloads\\projects\\claude_code_skills\\skill_create_hn_mcp_server\\hackernews_mcp.py"
      ]
    }
  }
}
```

**Important:**
- Use double backslashes (`\\`) in Windows paths
- Use your actual full paths from Step 2
- If you have other MCP servers, add this as another entry

### Step 4: Restart Claude Desktop

Close and restart Claude Desktop completely.

## Verify Installation

Ask Claude in Claude Desktop:

```
What Hacker News tools do you have available?
```

Claude should list 9 tools:
- `hn_get_item`
- `hn_get_top_stories`
- `hn_get_new_stories`
- `hn_get_best_stories`
- `hn_get_ask_stories`
- `hn_get_show_stories`
- `hn_get_job_stories`
- `hn_get_user`
- `hn_get_max_item_id`

## Try It Out!

Ask Claude:

```
Show me the top 5 stories on Hacker News right now
```

```
Get details about the famous Dropbox story with ID 8863
```

```
What's the profile for HN user 'pg'?
```

```
What are the latest Ask HN posts?
```

## Troubleshooting

### "Command not found" or "Python not recognized"

Python is not in your PATH. Reinstall Python and check "Add Python to PATH".

### "No module named 'mcp'"

The virtual environment isn't activated or dependencies aren't installed:

```bash
# Activate venv
source venv/Scripts/activate  # Git Bash
.\venv\Scripts\Activate.ps1   # PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Claude Desktop doesn't show tools

1. Verify JSON syntax is correct (use [jsonlint.com](https://jsonlint.com))
2. Check you used double backslashes in paths
3. Verify paths are absolute (full paths, not relative)
4. Restart Claude Desktop
5. Check Claude Desktop logs at `%APPDATA%\Claude\logs`

### Server appears to hang

This is **normal**! MCP servers are long-running processes. They wait for requests and don't exit on their own. Press `Ctrl+C` to stop if running manually.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out [Tool Reference](README.md#tool-reference) for all available tools
- Try the [evaluation questions](evaluation.xml) to test the server

## Need Help?

1. Check the [Troubleshooting section](README.md#troubleshooting) in README.md
2. Verify your Python version: `python --version` (should be 3.8+)
3. Test imports: `python test_server.py`
4. Review Claude Desktop logs for detailed errors

## What's Next?

Once the server is running, you can:

- Browse Hacker News stories through Claude
- Research user profiles and karma
- Analyze comment threads
- Find job postings
- Track trending topics
- Explore historical HN data

Enjoy exploring Hacker News with Claude!
