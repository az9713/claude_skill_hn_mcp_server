#!/bin/bash
# Hacker News MCP Server - Bash Installation Script (for Git Bash on Windows)

echo -e "\033[0;32mInstalling Hacker News MCP Server...\033[0m"
echo ""

# Check if Python is installed
echo -e "\033[0;33mChecking Python installation...\033[0m"
if ! command -v python &> /dev/null; then
    echo -e "\033[0;31mERROR: Python is not installed or not in PATH\033[0m"
    echo -e "\033[0;31mPlease install Python from https://www.python.org/downloads/\033[0m"
    echo -e "\033[0;31mMake sure to check 'Add Python to PATH' during installation\033[0m"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
echo -e "\033[0;32mFound: $PYTHON_VERSION\033[0m"

# Create virtual environment
echo ""
echo -e "\033[0;33mCreating virtual environment...\033[0m"
python -m venv venv

if [ ! -f "venv/Scripts/python.exe" ]; then
    echo -e "\033[0;31mERROR: Failed to create virtual environment\033[0m"
    exit 1
fi

echo -e "\033[0;32mVirtual environment created successfully\033[0m"

# Activate virtual environment and install dependencies
echo ""
echo -e "\033[0;33mInstalling dependencies...\033[0m"

venv/Scripts/python.exe -m pip install --upgrade pip --quiet
venv/Scripts/python.exe -m pip install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo -e "\033[0;31mERROR: Failed to install dependencies\033[0m"
    exit 1
fi

echo -e "\033[0;32mDependencies installed successfully\033[0m"

# Test the installation
echo ""
echo -e "\033[0;33mTesting installation...\033[0m"

TEST_RESULT=$(venv/Scripts/python.exe -c "from mcp.server.fastmcp import FastMCP; print('OK')" 2>&1)

if [[ $TEST_RESULT == *"OK"* ]]; then
    echo -e "\033[0;32mInstallation test passed!\033[0m"
else
    echo -e "\033[0;31mERROR: Installation test failed\033[0m"
    echo -e "\033[0;31m$TEST_RESULT\033[0m"
    exit 1
fi

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH="$SCRIPT_DIR/venv/Scripts/python.exe"
SERVER_PATH="$SCRIPT_DIR/hackernews_mcp.py"

# Convert to Windows paths
PYTHON_PATH_WIN=$(cygpath -w "$PYTHON_PATH" 2>/dev/null || echo "$PYTHON_PATH" | sed 's|/c/|C:/|' | sed 's|/|\\|g')
SERVER_PATH_WIN=$(cygpath -w "$SERVER_PATH" 2>/dev/null || echo "$SERVER_PATH" | sed 's|/c/|C:/|' | sed 's|/|\\|g')

# Display next steps
echo ""
echo -e "\033[0;36m============================================\033[0m"
echo -e "\033[0;32mInstallation Complete!\033[0m"
echo -e "\033[0;36m============================================\033[0m"
echo ""
echo -e "\033[0;33mNext steps:\033[0m"
echo ""
echo -e "\033[1;37m1. Python path (for Claude Desktop config):\033[0m"
echo -e "   \033[0;36m$PYTHON_PATH_WIN\033[0m"
echo ""
echo -e "\033[1;37m2. Server path (for Claude Desktop config):\033[0m"
echo -e "   \033[0;36m$SERVER_PATH_WIN\033[0m"
echo ""
echo -e "\033[1;37m3. Add this server to Claude Desktop config:\033[0m"
echo -e "   Config file location: %APPDATA%\\Claude\\claude_desktop_config.json"
echo ""
echo -e "   Add this entry (use paths from above):"
echo -e "   \033[0;36m{"
echo -e "     \"mcpServers\": {"
echo -e "       \"hackernews\": {"
echo -e "         \"command\": \"$PYTHON_PATH_WIN\","
echo -e "         \"args\": [\"$SERVER_PATH_WIN\"]"
echo -e "       }"
echo -e "     }"
echo -e "   }\033[0m"
echo ""
echo -e "\033[1;37m4. Restart Claude Desktop\033[0m"
echo ""
echo -e "\033[0;33mFor detailed instructions, see README.md\033[0m"
echo ""
