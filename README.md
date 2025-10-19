# Hacker News MCP Server

A Model Context Protocol (MCP) server that provides access to the Hacker News API. This server allows AI assistants like Claude to fetch stories, comments, user profiles, and other content from Hacker News.

## Features

- **9 Comprehensive Tools** for accessing Hacker News data:
  - `hn_get_item` - Get any item (story, comment, job, poll) by ID
  - `hn_get_top_stories` - Get current front page stories
  - `hn_get_new_stories` - Get newest stories
  - `hn_get_best_stories` - Get best stories of all time
  - `hn_get_ask_stories` - Get Ask HN posts
  - `hn_get_show_stories` - Get Show HN posts
  - `hn_get_job_stories` - Get job postings
  - `hn_get_user` - Get user profile information
  - `hn_get_max_item_id` - Get the latest item ID

- **Flexible Output Formats**: Both Markdown (human-readable) and JSON (machine-readable)
- **Pagination Support**: Efficiently browse large result sets
- **Detail Levels**: Choose between concise summaries or detailed information
- **Rate Limit Handling**: Graceful error handling with helpful messages
- **Character Limits**: Automatic truncation for large responses
- **Parallel Fetching**: Efficient batch requests for multiple items

## Requirements

- **Python 3.8+** (tested on Python 3.10+)
- **Windows 10/11** (compatible with PowerShell and Git Bash)

## Installation

### Step 1: Install Python

If you don't have Python installed:

1. Download Python from [python.org](https://www.python.org/downloads/)
2. **Important**: During installation, check "Add Python to PATH"
3. Verify installation:
   ```bash
   python --version
   ```

### Step 2: Create a Virtual Environment (Recommended)

Using a virtual environment keeps dependencies isolated:

**PowerShell:**
```powershell
cd C:\Users\simon\Downloads\projects\claude_code_skills\skill_create_hn_mcp_server
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Git Bash:**
```bash
cd /c/Users/simon/Downloads/projects/claude_code_skills/skill_create_hn_mcp_server
python -m venv venv
source venv/Scripts/activate
```

You should see `(venv)` in your terminal prompt after activation.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install mcp httpx pydantic
```

Verify installation:
```bash
python -c "from mcp.server.fastmcp import FastMCP; print('MCP installed successfully')"
```

### Step 4: Test the Server

Verify the server can run:

```bash
python hackernews_mcp.py --help
```

You should see FastMCP help output. The server will appear to hang - this is normal! Press `Ctrl+C` to stop it.

## Configuration

### For Claude Desktop (Windows)

1. Locate your Claude Desktop config file:
   ```
   %APPDATA%\Claude\claude_desktop_config.json
   ```

2. Open it in a text editor and add the Hacker News MCP server:

**PowerShell users:**
```json
{
  "mcpServers": {
    "hackernews": {
      "command": "/path/to/python.exe",
      "args": [
        "/path/to/hackernews_mcp.py"
      ]
    }
  }
}
```

**Git Bash users:**
```json
{
  "mcpServers": {
    "hackernews": {
      "command": "/path/to/python.exe",
      "args": [
        "/path/to/hackernews_mcp.py"
      ]
    }
  }
}
```

**Important Notes:**
- Use double backslashes (`\\`) in the JSON file paths
- Use the full absolute paths to both `python.exe` and `hackernews_mcp.py`
- If you have other MCP servers, add this as another entry in `mcpServers`

3. Restart Claude Desktop

4. Verify the connection:
   - Open Claude Desktop
   - Look for a "hammer" or "tools" icon indicating MCP servers are connected
   - Ask Claude: "What Hacker News tools do you have available?"

### For Other MCP Clients

The server uses stdio transport by default. Configure your MCP client to run:

```bash
python /path/to/hackernews_mcp.py
```

## Usage Examples

Once configured, you can ask Claude to use the Hacker News tools:

### Get Top Stories
```
Show me the top 10 stories on Hacker News right now
```

### Get Story Details
```
Get details about HN story with ID 8863
```

### Search for User
```
What's the profile for HN user 'pg'?
```

### Get Ask HN Posts
```
Show me the latest Ask HN questions with at least 50 comments
```

### Browse Job Postings
```
What are the latest 5 job postings on HN?
```

## Tool Reference

### hn_get_item
Get complete details for any HN item (story, comment, job, poll).

**Parameters:**
- `item_id` (required): The item ID number
- `response_format`: "markdown" (default) or "json"
- `detail_level`: "detailed" (default) or "concise"

**Example:**
```json
{
  "item_id": 8863,
  "response_format": "markdown",
  "detail_level": "detailed"
}
```

### hn_get_top_stories
Get current front page stories.

**Parameters:**
- `limit`: Max stories to return (1-100, default: 30)
- `offset`: Skip this many stories (default: 0)
- `response_format`: "markdown" (default) or "json"
- `detail_level`: "concise" (default) or "detailed"

### hn_get_new_stories
Get newest stories in chronological order.

**Parameters:** Same as `hn_get_top_stories`

### hn_get_best_stories
Get best stories by HN's algorithm.

**Parameters:** Same as `hn_get_top_stories`

### hn_get_ask_stories
Get latest Ask HN posts.

**Parameters:** Same as `hn_get_top_stories`

### hn_get_show_stories
Get latest Show HN posts.

**Parameters:** Same as `hn_get_top_stories`

### hn_get_job_stories
Get latest job postings.

**Parameters:** Same as `hn_get_top_stories`

### hn_get_user
Get user profile information.

**Parameters:**
- `username` (required): Case-sensitive username
- `response_format`: "markdown" (default) or "json"
- `include_submissions`: Include full submission list (default: false)

**Example:**
```json
{
  "username": "pg",
  "response_format": "markdown",
  "include_submissions": false
}
```

### hn_get_max_item_id
Get the current maximum item ID.

**Parameters:** None

## Troubleshooting

### "python: command not found"

**Solution:** Python is not in your PATH.

**PowerShell:**
```powershell
# Find Python installation
Get-Command python

# If not found, reinstall Python with "Add to PATH" checked
```

**Git Bash:**
```bash
# Check if Python is installed
which python

# Add Python to PATH temporarily
export PATH="/c/Users/YourUsername/AppData/Local/Programs/Python/Python311:$PATH"
```

### "No module named 'mcp'"

**Solution:** Install dependencies in the virtual environment:
```bash
# Make sure virtual environment is activated (you should see (venv) in prompt)
source venv/Scripts/activate  # Git Bash
# or
.\venv\Scripts\Activate.ps1   # PowerShell

# Install dependencies
pip install mcp httpx pydantic
```

### "Permission denied" when activating virtual environment (PowerShell)

**Solution:** Enable script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Server appears to hang when running

This is **normal behavior**! MCP servers are long-running processes that wait for requests over stdio. They don't exit immediately. To test if it's working:

1. Run `python hackernews_mcp.py --help` - should show help
2. Configure it in Claude Desktop and use it through Claude
3. Use `Ctrl+C` to stop the server if running it manually

### Claude Desktop doesn't show HN tools

**Checklist:**
1. Verify JSON syntax in `claude_desktop_config.json` (use a JSON validator)
2. Use double backslashes in Windows paths: `C:\\Users\\...`
3. Use full absolute paths, not relative paths
4. Restart Claude Desktop after config changes
5. Check Claude Desktop logs (usually in `%APPDATA%\Claude\logs`)

### "Rate limit exceeded" errors

The Hacker News API has no official rate limit, but if you're making too many requests:

1. Use `limit` parameter to fetch fewer items
2. Use `detail_level: "concise"` to reduce API calls
3. Add delays between requests if needed

## API Documentation

This server uses the official Hacker News Firebase API:
- Base URL: `https://hacker-news.firebaseio.com/v0/`
- Documentation: See `docs/README.md` in this repository
- Source: [Hacker News API on GitHub](https://github.com/HackerNews/API)

## Architecture

- **Framework**: FastMCP (official MCP Python SDK)
- **HTTP Client**: httpx (async)
- **Validation**: Pydantic v2
- **Transport**: stdio (standard input/output)

### Key Features:
- Async/await for all I/O operations
- Parallel batch fetching for efficiency
- Comprehensive error handling
- Character limit enforcement (25,000 chars)
- Pagination support
- Input validation with detailed constraints

## Development

### Running Tests

```bash
# Activate virtual environment first
source venv/Scripts/activate  # Git Bash
# or
.\venv\Scripts\Activate.ps1   # PowerShell

# Test imports
python -c "from hackernews_mcp import mcp; print('Import successful')"

# Test server help
python hackernews_mcp.py --help
```

### Code Structure

```
hackernews_mcp.py
├── Constants (API_BASE_URL, CHARACTER_LIMIT, etc.)
├── Enums (ResponseFormat, DetailLevel)
├── Pydantic Models (ItemInput, StoryListInput, UserInput)
├── Shared Utilities
│   ├── _fetch_from_api() - HTTP requests
│   ├── _fetch_items_batch() - Parallel fetching
│   ├── _format_timestamp() - Time formatting
│   ├── _format_item_markdown() - Markdown output
│   ├── _format_item_json() - JSON output
│   ├── _handle_api_error() - Error handling
│   └── _check_truncation() - Character limit enforcement
└── Tools (9 @mcp.tool decorated functions)
```

## License

This MCP server is provided as-is for use with the Hacker News API. The Hacker News API is provided by Y Combinator.

## Credits

- Hacker News API: Y Combinator
- MCP Protocol: Anthropic
- Server Implementation: Created with Claude Code

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review the API documentation in `docs/README.md`
3. Verify your Python and dependency versions
4. Check Claude Desktop logs for detailed error messages

## Version

Version: 1.0.0
Last Updated: 2025-10-18
