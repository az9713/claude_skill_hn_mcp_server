# Hacker News MCP Server - PowerShell Installation Script

Write-Host "Installing Hacker News MCP Server..." -ForegroundColor Green
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host "Virtual environment created successfully" -ForegroundColor Green

# Activate virtual environment and install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow

& "venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
& "venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "Dependencies installed successfully" -ForegroundColor Green

# Test the installation
Write-Host ""
Write-Host "Testing installation..." -ForegroundColor Yellow

$testResult = & "venv\Scripts\python.exe" -c "from mcp.server.fastmcp import FastMCP; print('OK')" 2>&1

if ($testResult -like "*OK*") {
    Write-Host "Installation test passed!" -ForegroundColor Green
} else {
    Write-Host "ERROR: Installation test failed" -ForegroundColor Red
    Write-Host $testResult -ForegroundColor Red
    exit 1
}

# Display next steps
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Get the full path to python.exe:" -ForegroundColor White
Write-Host "   " -NoNewline
Write-Host "(Get-Item venv\Scripts\python.exe).FullName" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Get the full path to the server:" -ForegroundColor White
Write-Host "   " -NoNewline
Write-Host "(Get-Item hackernews_mcp.py).FullName" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Add this server to Claude Desktop config:" -ForegroundColor White
Write-Host "   Config file location: %APPDATA%\Claude\claude_desktop_config.json" -ForegroundColor Cyan
Write-Host ""
Write-Host "   Add this entry (use paths from steps 1 and 2):" -ForegroundColor White
Write-Host '   {' -ForegroundColor Cyan
Write-Host '     "mcpServers": {' -ForegroundColor Cyan
Write-Host '       "hackernews": {' -ForegroundColor Cyan
Write-Host '         "command": "C:\\path\\to\\venv\\Scripts\\python.exe",' -ForegroundColor Cyan
Write-Host '         "args": ["C:\\path\\to\\hackernews_mcp.py"]' -ForegroundColor Cyan
Write-Host '       }' -ForegroundColor Cyan
Write-Host '     }' -ForegroundColor Cyan
Write-Host '   }' -ForegroundColor Cyan
Write-Host ""
Write-Host "4. Restart Claude Desktop" -ForegroundColor White
Write-Host ""
Write-Host "For detailed instructions, see README.md" -ForegroundColor Yellow
Write-Host ""
