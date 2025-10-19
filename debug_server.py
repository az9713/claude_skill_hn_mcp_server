#!/usr/bin/env python3
"""
Debug script to test MCP server startup and communication.
"""
import sys
import json
import subprocess

def test_server_startup():
    """Test if the server can start and respond to basic MCP messages."""
    print("Testing MCP server startup...")

    # Path to the server
    python_exe = r"/path/to/python.exe"
    server_script = r"/path/to/hackernews_mcp.py"

    # Start the server process
    try:
        process = subprocess.Popen(
            [python_exe, server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print(f"Server process started with PID: {process.pid}")

        # Send an initialize request (MCP protocol)
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }

        print("Sending initialize request...")
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()

        # Wait for response (with timeout)
        import time
        time.sleep(2)

        # Check if process is still running
        if process.poll() is None:
            print("[OK] Server is still running (good sign)")

            # Try to read response
            stdout_data = process.stdout.readline()
            stderr_data = process.stderr.read()

            if stdout_data:
                print(f"Server response: {stdout_data}")
            else:
                print("No response from server yet")

            if stderr_data:
                print(f"Server stderr: {stderr_data}")

        else:
            print(f"[ERROR] Server exited with code: {process.returncode}")
            stdout_data = process.stdout.read()
            stderr_data = process.stderr.read()

            if stdout_data:
                print(f"Server stdout: {stdout_data}")
            if stderr_data:
                print(f"Server stderr: {stderr_data}")

        # Cleanup
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)

    except Exception as e:
        print(f"Error testing server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_server_startup()
