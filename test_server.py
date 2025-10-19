#!/usr/bin/env python3
"""
Quick test script to verify the Hacker News MCP server functionality.
This script tests imports and basic functionality without running the full server.
"""

import sys
import asyncio

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        from mcp.server.fastmcp import FastMCP
        print("  ✓ FastMCP imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import FastMCP: {e}")
        print("    Run: pip install mcp")
        return False

    try:
        import httpx
        print("  ✓ httpx imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import httpx: {e}")
        print("    Run: pip install httpx")
        return False

    try:
        from pydantic import BaseModel, Field
        print("  ✓ Pydantic imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import Pydantic: {e}")
        print("    Run: pip install pydantic")
        return False

    return True

def test_server_import():
    """Test that the server file can be imported."""
    print("\nTesting server import...")
    try:
        import hackernews_mcp
        print("  ✓ hackernews_mcp.py imported successfully")

        # Check that the mcp object exists
        if hasattr(hackernews_mcp, 'mcp'):
            print("  ✓ MCP server object found")
        else:
            print("  ✗ MCP server object not found")
            return False

        return True
    except Exception as e:
        print(f"  ✗ Failed to import hackernews_mcp: {e}")
        return False

async def test_api_access():
    """Test that we can access the Hacker News API."""
    print("\nTesting Hacker News API access...")
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            # Test getting max item ID
            response = await client.get(
                "https://hacker-news.firebaseio.com/v0/maxitem.json",
                timeout=10.0
            )
            response.raise_for_status()
            max_id = response.json()
            print(f"  ✓ API accessible (current max item ID: {max_id})")

            # Test getting a specific item
            response = await client.get(
                "https://hacker-news.firebaseio.com/v0/item/8863.json",
                timeout=10.0
            )
            response.raise_for_status()
            item = response.json()
            print(f"  ✓ Item fetch works (got item: {item.get('title', 'untitled')[:50]}...)")

        return True
    except Exception as e:
        print(f"  ✗ API test failed: {e}")
        return False

def test_pydantic_models():
    """Test that Pydantic models are properly defined."""
    print("\nTesting Pydantic models...")
    try:
        from hackernews_mcp import ItemInput, StoryListInput, UserInput

        # Test ItemInput
        item_input = ItemInput(item_id=8863)
        print(f"  ✓ ItemInput model works (item_id={item_input.item_id})")

        # Test StoryListInput
        story_input = StoryListInput(limit=10)
        print(f"  ✓ StoryListInput model works (limit={story_input.limit})")

        # Test UserInput
        user_input = UserInput(username="pg")
        print(f"  ✓ UserInput model works (username={user_input.username})")

        return True
    except Exception as e:
        print(f"  ✗ Pydantic model test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Hacker News MCP Server - Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: Server import
    results.append(("Server Import", test_server_import()))

    # Test 3: Pydantic models
    results.append(("Pydantic Models", test_pydantic_models()))

    # Test 4: API access (async)
    print("\nRunning async tests...")
    api_result = asyncio.run(test_api_access())
    results.append(("API Access", api_result))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Server is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
