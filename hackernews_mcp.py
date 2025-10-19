#!/usr/bin/env python3
'''
MCP Server for Hacker News API.

This server provides tools to interact with the Hacker News API (via Firebase),
including fetching stories, comments, user profiles, and various story categories.
'''

from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import json
import asyncio
import httpx
from pydantic import BaseModel, Field, field_validator, ConfigDict
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("hackernews_mcp")

# Constants
API_BASE_URL = "https://hacker-news.firebaseio.com/v0"
CHARACTER_LIMIT = 25000  # Maximum response size in characters
MAX_CONCURRENT_REQUESTS = 10  # Max parallel item fetches

# Enums
class ResponseFormat(str, Enum):
    '''Output format for tool responses.'''
    MARKDOWN = "markdown"
    JSON = "json"

class DetailLevel(str, Enum):
    '''Level of detail for responses.'''
    CONCISE = "concise"
    DETAILED = "detailed"

# Pydantic Models for Input Validation
class ItemInput(BaseModel):
    '''Input model for fetching a single item.'''
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    item_id: int = Field(..., description="The item ID to fetch (e.g., 8863, 2921983)", ge=1)
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )
    detail_level: DetailLevel = Field(
        default=DetailLevel.DETAILED,
        description="Level of detail: 'concise' for summary or 'detailed' for full information"
    )

class StoryListInput(BaseModel):
    '''Input model for fetching story lists.'''
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    limit: Optional[int] = Field(
        default=30,
        description="Maximum number of stories to return (1-100)",
        ge=1,
        le=100
    )
    offset: Optional[int] = Field(
        default=0,
        description="Number of stories to skip for pagination",
        ge=0
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )
    detail_level: DetailLevel = Field(
        default=DetailLevel.CONCISE,
        description="Level of detail: 'concise' for IDs/titles or 'detailed' for full story data"
    )

class UserInput(BaseModel):
    '''Input model for fetching user profiles.'''
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )

    username: str = Field(
        ...,
        description="The user's unique username (case-sensitive, e.g., 'jl', 'pg', 'dhouston')",
        min_length=1,
        max_length=100
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )
    include_submissions: bool = Field(
        default=False,
        description="Whether to include the list of user submissions (can be very long)"
    )

# Shared utility functions
async def _fetch_from_api(endpoint: str) -> Any:
    '''Fetch data from Hacker News API.

    Args:
        endpoint: API endpoint path (e.g., 'item/8863.json', 'topstories.json')

    Returns:
        Parsed JSON response from the API

    Raises:
        httpx.HTTPStatusError: If the request fails
    '''
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/{endpoint}",
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

async def _fetch_items_batch(item_ids: List[int], max_concurrent: int = MAX_CONCURRENT_REQUESTS) -> List[Dict[str, Any]]:
    '''Fetch multiple items in parallel with concurrency control.

    Args:
        item_ids: List of item IDs to fetch
        max_concurrent: Maximum number of concurrent requests

    Returns:
        List of item dictionaries (None for failed fetches)
    '''
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(item_id: int) -> Optional[Dict[str, Any]]:
        async with semaphore:
            try:
                return await _fetch_from_api(f"item/{item_id}.json")
            except Exception:
                return None

    return await asyncio.gather(*[fetch_one(id) for id in item_ids])

def _format_timestamp(unix_time: int) -> str:
    '''Convert Unix timestamp to human-readable format.

    Args:
        unix_time: Unix timestamp in seconds

    Returns:
        Formatted datetime string (e.g., "2024-01-15 10:30:00 UTC")
    '''
    dt = datetime.utcfromtimestamp(unix_time)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

def _format_item_markdown(item: Dict[str, Any], detail_level: DetailLevel = DetailLevel.DETAILED) -> str:
    '''Format item data as Markdown.

    Args:
        item: Item dictionary from API
        detail_level: Level of detail to include

    Returns:
        Markdown-formatted string
    '''
    if not item:
        return "Item not found or has been deleted."

    lines = []
    item_type = item.get('type', 'unknown')

    # Title/header
    if item.get('title'):
        lines.append(f"# {item['title']}")
    else:
        lines.append(f"# Item {item['id']} ({item_type})")
    lines.append("")

    # Basic info
    lines.append(f"**Type**: {item_type}")
    lines.append(f"**ID**: {item['id']}")

    if item.get('by'):
        lines.append(f"**Author**: {item['by']}")

    if item.get('time'):
        lines.append(f"**Posted**: {_format_timestamp(item['time'])}")

    if item.get('score') is not None:
        lines.append(f"**Score**: {item['score']}")

    if item.get('descendants') is not None:
        lines.append(f"**Comments**: {item['descendants']}")

    lines.append("")

    # URL for stories
    if item.get('url'):
        lines.append(f"**URL**: {item['url']}")
        lines.append("")

    # Text content
    if item.get('text') and detail_level == DetailLevel.DETAILED:
        lines.append("## Content")
        lines.append("")
        lines.append(item['text'])
        lines.append("")

    # Kids (comments/replies)
    if item.get('kids') and detail_level == DetailLevel.DETAILED:
        kids_count = len(item['kids'])
        lines.append(f"## Comments/Replies ({kids_count})")
        lines.append("")
        lines.append(f"Comment IDs: {', '.join(map(str, item['kids'][:20]))}")
        if kids_count > 20:
            lines.append(f"... and {kids_count - 20} more")
        lines.append("")

    # Parent reference
    if item.get('parent'):
        lines.append(f"**Parent ID**: {item['parent']}")
        lines.append("")

    # Poll parts
    if item.get('parts'):
        lines.append(f"**Poll Options**: {len(item['parts'])} options")
        lines.append(f"Option IDs: {', '.join(map(str, item['parts']))}")
        lines.append("")

    # Status flags
    if item.get('deleted'):
        lines.append("*This item has been deleted.*")
    if item.get('dead'):
        lines.append("*This item is dead.*")

    return "\n".join(lines)

def _format_item_json(item: Dict[str, Any], detail_level: DetailLevel = DetailLevel.DETAILED) -> Dict[str, Any]:
    '''Format item data as JSON with optional detail reduction.

    Args:
        item: Item dictionary from API
        detail_level: Level of detail to include

    Returns:
        Formatted dictionary
    '''
    if not item:
        return {"error": "Item not found or has been deleted"}

    if detail_level == DetailLevel.CONCISE:
        # Return only essential fields
        concise = {
            "id": item.get("id"),
            "type": item.get("type"),
            "by": item.get("by"),
            "time": item.get("time"),
        }
        if item.get("title"):
            concise["title"] = item["title"]
        if item.get("url"):
            concise["url"] = item["url"]
        if item.get("score") is not None:
            concise["score"] = item["score"]
        if item.get("descendants") is not None:
            concise["descendants"] = item["descendants"]
        return concise

    return item

def _handle_api_error(e: Exception) -> str:
    '''Consistent error formatting across all tools.

    Args:
        e: Exception that occurred

    Returns:
        User-friendly error message
    '''
    if isinstance(e, httpx.HTTPStatusError):
        if e.response.status_code == 404:
            return "Error: Item not found. The ID may be invalid or the item may have been deleted."
        elif e.response.status_code == 429:
            return "Error: Rate limit exceeded. Please wait a moment before making more requests."
        elif e.response.status_code >= 500:
            return "Error: Hacker News API is experiencing issues. Please try again later."
        return f"Error: API request failed with status {e.response.status_code}"
    elif isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out. The API may be slow or unavailable. Please try again."
    elif isinstance(e, httpx.ConnectError):
        return "Error: Could not connect to Hacker News API. Please check your internet connection."
    return f"Error: Unexpected error occurred: {type(e).__name__} - {str(e)}"

def _check_truncation(content: str, data_count: int, data_type: str) -> str:
    '''Check if content exceeds character limit and add truncation message.

    Args:
        content: The formatted content string
        data_count: Number of items in the response
        data_type: Type of data (e.g., "stories", "comments")

    Returns:
        Content with truncation message if needed
    '''
    if len(content) > CHARACTER_LIMIT:
        truncated = content[:CHARACTER_LIMIT]
        truncated += f"\n\n--- TRUNCATED ---\n"
        truncated += f"Response exceeded {CHARACTER_LIMIT} character limit. "
        truncated += f"Showing partial results. Use offset/limit parameters or filters to see more {data_type}."
        return truncated
    return content

# Tool definitions
@mcp.tool(
    name="hn_get_item",
    annotations={
        "title": "Get Hacker News Item",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def hn_get_item(params: ItemInput) -> str:
    '''Get details for a specific Hacker News item (story, comment, job, poll, or pollopt).

    This tool fetches complete information about any HN item by its unique ID. Items include
    stories, comments, job postings, Ask HN posts, polls, and poll options.

    Args:
        params (ItemInput): Validated input parameters containing:
            - item_id (int): The unique item ID (e.g., 8863 for Dropbox story, 2921983 for a comment)
            - response_format (ResponseFormat): 'markdown' (default) or 'json'
            - detail_level (DetailLevel): 'detailed' (default, includes full text and comments) or 'concise' (summary only)

    Returns:
        str: Formatted item details

        Markdown format includes:
        - Title, author, timestamp, score, comment count
        - URL (for stories)
        - Full text content (for detailed level)
        - Comment IDs (for detailed level)
        - Parent/poll references where applicable

        JSON format returns the raw API response with all fields, or a concise subset.

    Examples:
        - Use when: "Show me details about HN item 8863" -> Get the famous Dropbox YC application story
        - Use when: "What did user norvig say in comment 2921983?" -> Get specific comment details
        - Use when: "Get the job posting with ID 192327" -> Fetch job listing details
        - Don't use when: You want to search for stories by topic (HN API doesn't support search)
        - Don't use when: You need the latest stories (use hn_get_top_stories or hn_get_new_stories instead)

    Error Handling:
        - Returns "Error: Item not found" if the ID is invalid or item was deleted (404)
        - Returns "Error: Rate limit exceeded" if making too many requests (429)
        - Returns "Error: Request timed out" if API is slow
        - Deleted items show in results with a "deleted" flag
    '''
    try:
        item = await _fetch_from_api(f"item/{params.item_id}.json")

        if params.response_format == ResponseFormat.MARKDOWN:
            return _format_item_markdown(item, params.detail_level)
        else:
            result = _format_item_json(item, params.detail_level)
            return json.dumps(result, indent=2)

    except Exception as e:
        return _handle_api_error(e)

@mcp.tool(
    name="hn_get_top_stories",
    annotations={
        "title": "Get Top Hacker News Stories",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def hn_get_top_stories(params: StoryListInput) -> str:
    '''Get the current top stories from Hacker News front page.

    This tool fetches up to 500 top-ranked stories (including jobs). Stories are ranked
    by HN's algorithm considering score, time, and other factors.

    Args:
        params (StoryListInput): Validated input parameters containing:
            - limit (Optional[int]): Max stories to return, 1-100 (default: 30)
            - offset (Optional[int]): Number of stories to skip for pagination (default: 0)
            - response_format (ResponseFormat): 'markdown' (default) or 'json'
            - detail_level (DetailLevel): 'concise' (default, IDs and titles only) or 'detailed' (full story data)

    Returns:
        str: List of top stories with pagination info

        Concise format shows:
        - Story ID, title, author, score, comment count

        Detailed format includes:
        - All concise fields plus URL, full text, timestamps
        - Pagination metadata (total, count, offset, has_more)

    Examples:
        - Use when: "What are the top stories on HN right now?" -> Get current front page
        - Use when: "Show me the top 10 HN stories" -> Set limit=10
        - Use when: "Get stories 20-40 from the front page" -> Set offset=20, limit=20
        - Don't use when: You want the newest stories (use hn_get_new_stories)
        - Don't use when: You want best of all time (use hn_get_best_stories)

    Error Handling:
        - Returns "Error: Rate limit exceeded" if too many requests (429)
        - Returns "Error: Request timed out" if API is slow
        - Includes truncation warning if response exceeds 25,000 characters
        - Gracefully handles deleted stories (shows as null in results)
    '''
    try:
        story_ids = await _fetch_from_api("topstories.json")

        # Apply pagination
        total_count = len(story_ids)
        paginated_ids = story_ids[params.offset:params.offset + params.limit]

        if not paginated_ids:
            return f"No stories found at offset {params.offset}. Total available: {total_count}"

        # For concise, just return IDs; for detailed, fetch full story data
        if params.detail_level == DetailLevel.DETAILED:
            stories = await _fetch_items_batch(paginated_ids)
        else:
            stories = None

        # Format response
        if params.response_format == ResponseFormat.MARKDOWN:
            lines = [f"# Top Hacker News Stories", ""]
            lines.append(f"Showing {len(paginated_ids)} of {total_count} stories (offset: {params.offset})")
            lines.append("")

            if params.detail_level == DetailLevel.CONCISE:
                for idx, story_id in enumerate(paginated_ids, start=params.offset + 1):
                    lines.append(f"{idx}. Story ID: {story_id}")
                lines.append("")
                lines.append(f"Use hn_get_item with these IDs to see full details.")
            else:
                for idx, story in enumerate(stories, start=params.offset + 1):
                    if story:
                        title = story.get('title', f"Story {story['id']}")
                        score = story.get('score', 0)
                        by = story.get('by', 'unknown')
                        comments = story.get('descendants', 0)
                        lines.append(f"## {idx}. {title}")
                        lines.append(f"- **ID**: {story['id']} | **Author**: {by} | **Score**: {score} | **Comments**: {comments}")
                        if story.get('url'):
                            lines.append(f"- **URL**: {story['url']}")
                        lines.append("")

            # Pagination info
            has_more = (params.offset + len(paginated_ids)) < total_count
            if has_more:
                next_offset = params.offset + len(paginated_ids)
                lines.append(f"---")
                lines.append(f"More stories available. Use offset={next_offset} to see next page.")

            content = "\n".join(lines)
            return _check_truncation(content, len(paginated_ids), "stories")

        else:
            # JSON format
            result = {
                "total": total_count,
                "count": len(paginated_ids),
                "offset": params.offset,
                "has_more": (params.offset + len(paginated_ids)) < total_count,
                "next_offset": params.offset + len(paginated_ids) if (params.offset + len(paginated_ids)) < total_count else None
            }

            if params.detail_level == DetailLevel.CONCISE:
                result["story_ids"] = paginated_ids
            else:
                result["stories"] = [_format_item_json(s, params.detail_level) for s in stories]

            content = json.dumps(result, indent=2)
            return _check_truncation(content, len(paginated_ids), "stories")

    except Exception as e:
        return _handle_api_error(e)

@mcp.tool(
    name="hn_get_new_stories",
    annotations={
        "title": "Get New Hacker News Stories",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def hn_get_new_stories(params: StoryListInput) -> str:
    '''Get the newest stories from Hacker News.

    This tool fetches up to 500 newest stories in chronological order. These are
    stories that were recently submitted, regardless of score or ranking.

    Args:
        params (StoryListInput): Validated input parameters containing:
            - limit (Optional[int]): Max stories to return, 1-100 (default: 30)
            - offset (Optional[int]): Number of stories to skip for pagination (default: 0)
            - response_format (ResponseFormat): 'markdown' (default) or 'json'
            - detail_level (DetailLevel): 'concise' (default, IDs only) or 'detailed' (full story data)

    Returns:
        str: List of newest stories with pagination info

    Examples:
        - Use when: "What are the latest stories posted to HN?" -> Get most recent submissions
        - Use when: "Show me new HN submissions from the last hour" -> Get newest stories
        - Don't use when: You want the front page (use hn_get_top_stories)
        - Don't use when: You want Ask HN posts (use hn_get_ask_stories)

    Error Handling:
        - Returns "Error: Rate limit exceeded" if too many requests (429)
        - Includes truncation warning if response exceeds character limit
    '''
    try:
        story_ids = await _fetch_from_api("newstories.json")

        # Apply pagination
        total_count = len(story_ids)
        paginated_ids = story_ids[params.offset:params.offset + params.limit]

        if not paginated_ids:
            return f"No stories found at offset {params.offset}. Total available: {total_count}"

        # Fetch full data if detailed
        if params.detail_level == DetailLevel.DETAILED:
            stories = await _fetch_items_batch(paginated_ids)
        else:
            stories = None

        # Format response
        if params.response_format == ResponseFormat.MARKDOWN:
            lines = [f"# New Hacker News Stories", ""]
            lines.append(f"Showing {len(paginated_ids)} of {total_count} stories (offset: {params.offset})")
            lines.append("")

            if params.detail_level == DetailLevel.CONCISE:
                for idx, story_id in enumerate(paginated_ids, start=params.offset + 1):
                    lines.append(f"{idx}. Story ID: {story_id}")
                lines.append("")
                lines.append(f"Use hn_get_item with these IDs to see full details.")
            else:
                for idx, story in enumerate(stories, start=params.offset + 1):
                    if story:
                        title = story.get('title', f"Story {story['id']}")
                        by = story.get('by', 'unknown')
                        time = _format_timestamp(story.get('time', 0)) if story.get('time') else 'unknown'
                        lines.append(f"## {idx}. {title}")
                        lines.append(f"- **ID**: {story['id']} | **Author**: {by} | **Posted**: {time}")
                        if story.get('url'):
                            lines.append(f"- **URL**: {story['url']}")
                        lines.append("")

            # Pagination info
            has_more = (params.offset + len(paginated_ids)) < total_count
            if has_more:
                next_offset = params.offset + len(paginated_ids)
                lines.append(f"---")
                lines.append(f"More stories available. Use offset={next_offset} to see next page.")

            content = "\n".join(lines)
            return _check_truncation(content, len(paginated_ids), "stories")

        else:
            # JSON format
            result = {
                "total": total_count,
                "count": len(paginated_ids),
                "offset": params.offset,
                "has_more": (params.offset + len(paginated_ids)) < total_count,
                "next_offset": params.offset + len(paginated_ids) if (params.offset + len(paginated_ids)) < total_count else None
            }

            if params.detail_level == DetailLevel.CONCISE:
                result["story_ids"] = paginated_ids
            else:
                result["stories"] = [_format_item_json(s, params.detail_level) for s in stories]

            content = json.dumps(result, indent=2)
            return _check_truncation(content, len(paginated_ids), "stories")

    except Exception as e:
        return _handle_api_error(e)

@mcp.tool(
    name="hn_get_best_stories",
    annotations={
        "title": "Get Best Hacker News Stories",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def hn_get_best_stories(params: StoryListInput) -> str:
    '''Get the best stories from Hacker News.

    This tool fetches up to 500 "best" stories as determined by HN's algorithm.
    These are high-quality stories that performed well historically.

    Args:
        params (StoryListInput): Validated input parameters

    Returns:
        str: List of best stories with pagination info

    Examples:
        - Use when: "What are the best stories on HN?" -> Get highest quality stories
        - Use when: "Show me the most acclaimed HN posts" -> Get best stories
    '''
    try:
        story_ids = await _fetch_from_api("beststories.json")

        total_count = len(story_ids)
        paginated_ids = story_ids[params.offset:params.offset + params.limit]

        if not paginated_ids:
            return f"No stories found at offset {params.offset}. Total available: {total_count}"

        if params.detail_level == DetailLevel.DETAILED:
            stories = await _fetch_items_batch(paginated_ids)
        else:
            stories = None

        if params.response_format == ResponseFormat.MARKDOWN:
            lines = [f"# Best Hacker News Stories", ""]
            lines.append(f"Showing {len(paginated_ids)} of {total_count} stories (offset: {params.offset})")
            lines.append("")

            if params.detail_level == DetailLevel.CONCISE:
                for idx, story_id in enumerate(paginated_ids, start=params.offset + 1):
                    lines.append(f"{idx}. Story ID: {story_id}")
                lines.append("")
                lines.append(f"Use hn_get_item with these IDs to see full details.")
            else:
                for idx, story in enumerate(stories, start=params.offset + 1):
                    if story:
                        title = story.get('title', f"Story {story['id']}")
                        score = story.get('score', 0)
                        by = story.get('by', 'unknown')
                        comments = story.get('descendants', 0)
                        lines.append(f"## {idx}. {title}")
                        lines.append(f"- **ID**: {story['id']} | **Author**: {by} | **Score**: {score} | **Comments**: {comments}")
                        if story.get('url'):
                            lines.append(f"- **URL**: {story['url']}")
                        lines.append("")

            has_more = (params.offset + len(paginated_ids)) < total_count
            if has_more:
                next_offset = params.offset + len(paginated_ids)
                lines.append(f"---")
                lines.append(f"More stories available. Use offset={next_offset} to see next page.")

            content = "\n".join(lines)
            return _check_truncation(content, len(paginated_ids), "stories")

        else:
            result = {
                "total": total_count,
                "count": len(paginated_ids),
                "offset": params.offset,
                "has_more": (params.offset + len(paginated_ids)) < total_count,
                "next_offset": params.offset + len(paginated_ids) if (params.offset + len(paginated_ids)) < total_count else None
            }

            if params.detail_level == DetailLevel.CONCISE:
                result["story_ids"] = paginated_ids
            else:
                result["stories"] = [_format_item_json(s, params.detail_level) for s in stories]

            content = json.dumps(result, indent=2)
            return _check_truncation(content, len(paginated_ids), "stories")

    except Exception as e:
        return _handle_api_error(e)

@mcp.tool(
    name="hn_get_ask_stories",
    annotations={
        "title": "Get Ask HN Stories",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def hn_get_ask_stories(params: StoryListInput) -> str:
    '''Get the latest Ask HN stories.

    This tool fetches up to 200 of the latest "Ask HN" posts where users ask
    the community questions or request feedback.

    Args:
        params (StoryListInput): Validated input parameters

    Returns:
        str: List of Ask HN stories with pagination info

    Examples:
        - Use when: "What are people asking on HN today?" -> Get latest Ask HN posts
        - Use when: "Show me Ask HN questions" -> Get Ask stories
    '''
    try:
        story_ids = await _fetch_from_api("askstories.json")

        total_count = len(story_ids)
        paginated_ids = story_ids[params.offset:params.offset + params.limit]

        if not paginated_ids:
            return f"No stories found at offset {params.offset}. Total available: {total_count}"

        if params.detail_level == DetailLevel.DETAILED:
            stories = await _fetch_items_batch(paginated_ids)
        else:
            stories = None

        if params.response_format == ResponseFormat.MARKDOWN:
            lines = [f"# Ask HN Stories", ""]
            lines.append(f"Showing {len(paginated_ids)} of {total_count} stories (offset: {params.offset})")
            lines.append("")

            if params.detail_level == DetailLevel.CONCISE:
                for idx, story_id in enumerate(paginated_ids, start=params.offset + 1):
                    lines.append(f"{idx}. Story ID: {story_id}")
                lines.append("")
                lines.append(f"Use hn_get_item with these IDs to see full details.")
            else:
                for idx, story in enumerate(stories, start=params.offset + 1):
                    if story:
                        title = story.get('title', f"Story {story['id']}")
                        score = story.get('score', 0)
                        by = story.get('by', 'unknown')
                        comments = story.get('descendants', 0)
                        lines.append(f"## {idx}. {title}")
                        lines.append(f"- **ID**: {story['id']} | **Author**: {by} | **Score**: {score} | **Comments**: {comments}")
                        lines.append("")

            has_more = (params.offset + len(paginated_ids)) < total_count
            if has_more:
                next_offset = params.offset + len(paginated_ids)
                lines.append(f"---")
                lines.append(f"More stories available. Use offset={next_offset} to see next page.")

            content = "\n".join(lines)
            return _check_truncation(content, len(paginated_ids), "stories")

        else:
            result = {
                "total": total_count,
                "count": len(paginated_ids),
                "offset": params.offset,
                "has_more": (params.offset + len(paginated_ids)) < total_count,
                "next_offset": params.offset + len(paginated_ids) if (params.offset + len(paginated_ids)) < total_count else None
            }

            if params.detail_level == DetailLevel.CONCISE:
                result["story_ids"] = paginated_ids
            else:
                result["stories"] = [_format_item_json(s, params.detail_level) for s in stories]

            content = json.dumps(result, indent=2)
            return _check_truncation(content, len(paginated_ids), "stories")

    except Exception as e:
        return _handle_api_error(e)

@mcp.tool(
    name="hn_get_show_stories",
    annotations={
        "title": "Get Show HN Stories",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def hn_get_show_stories(params: StoryListInput) -> str:
    '''Get the latest Show HN stories.

    This tool fetches up to 200 of the latest "Show HN" posts where users
    share their projects, products, or creations with the community.

    Args:
        params (StoryListInput): Validated input parameters

    Returns:
        str: List of Show HN stories with pagination info

    Examples:
        - Use when: "What projects are people showing on HN?" -> Get latest Show HN posts
        - Use when: "Show me Show HN submissions" -> Get Show stories
    '''
    try:
        story_ids = await _fetch_from_api("showstories.json")

        total_count = len(story_ids)
        paginated_ids = story_ids[params.offset:params.offset + params.limit]

        if not paginated_ids:
            return f"No stories found at offset {params.offset}. Total available: {total_count}"

        if params.detail_level == DetailLevel.DETAILED:
            stories = await _fetch_items_batch(paginated_ids)
        else:
            stories = None

        if params.response_format == ResponseFormat.MARKDOWN:
            lines = [f"# Show HN Stories", ""]
            lines.append(f"Showing {len(paginated_ids)} of {total_count} stories (offset: {params.offset})")
            lines.append("")

            if params.detail_level == DetailLevel.CONCISE:
                for idx, story_id in enumerate(paginated_ids, start=params.offset + 1):
                    lines.append(f"{idx}. Story ID: {story_id}")
                lines.append("")
                lines.append(f"Use hn_get_item with these IDs to see full details.")
            else:
                for idx, story in enumerate(stories, start=params.offset + 1):
                    if story:
                        title = story.get('title', f"Story {story['id']}")
                        score = story.get('score', 0)
                        by = story.get('by', 'unknown')
                        comments = story.get('descendants', 0)
                        lines.append(f"## {idx}. {title}")
                        lines.append(f"- **ID**: {story['id']} | **Author**: {by} | **Score**: {score} | **Comments**: {comments}")
                        if story.get('url'):
                            lines.append(f"- **URL**: {story['url']}")
                        lines.append("")

            has_more = (params.offset + len(paginated_ids)) < total_count
            if has_more:
                next_offset = params.offset + len(paginated_ids)
                lines.append(f"---")
                lines.append(f"More stories available. Use offset={next_offset} to see next page.")

            content = "\n".join(lines)
            return _check_truncation(content, len(paginated_ids), "stories")

        else:
            result = {
                "total": total_count,
                "count": len(paginated_ids),
                "offset": params.offset,
                "has_more": (params.offset + len(paginated_ids)) < total_count,
                "next_offset": params.offset + len(paginated_ids) if (params.offset + len(paginated_ids)) < total_count else None
            }

            if params.detail_level == DetailLevel.CONCISE:
                result["story_ids"] = paginated_ids
            else:
                result["stories"] = [_format_item_json(s, params.detail_level) for s in stories]

            content = json.dumps(result, indent=2)
            return _check_truncation(content, len(paginated_ids), "stories")

    except Exception as e:
        return _handle_api_error(e)

@mcp.tool(
    name="hn_get_job_stories",
    annotations={
        "title": "Get HN Job Postings",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def hn_get_job_stories(params: StoryListInput) -> str:
    '''Get the latest job postings from Hacker News.

    This tool fetches up to 200 of the latest job postings on HN.

    Args:
        params (StoryListInput): Validated input parameters

    Returns:
        str: List of job postings with pagination info

    Examples:
        - Use when: "What jobs are posted on HN?" -> Get latest job postings
        - Use when: "Show me HN job listings" -> Get job stories
    '''
    try:
        story_ids = await _fetch_from_api("jobstories.json")

        total_count = len(story_ids)
        paginated_ids = story_ids[params.offset:params.offset + params.limit]

        if not paginated_ids:
            return f"No jobs found at offset {params.offset}. Total available: {total_count}"

        if params.detail_level == DetailLevel.DETAILED:
            stories = await _fetch_items_batch(paginated_ids)
        else:
            stories = None

        if params.response_format == ResponseFormat.MARKDOWN:
            lines = [f"# Hacker News Job Postings", ""]
            lines.append(f"Showing {len(paginated_ids)} of {total_count} jobs (offset: {params.offset})")
            lines.append("")

            if params.detail_level == DetailLevel.CONCISE:
                for idx, story_id in enumerate(paginated_ids, start=params.offset + 1):
                    lines.append(f"{idx}. Job ID: {story_id}")
                lines.append("")
                lines.append(f"Use hn_get_item with these IDs to see full details.")
            else:
                for idx, story in enumerate(stories, start=params.offset + 1):
                    if story:
                        title = story.get('title', f"Job {story['id']}")
                        by = story.get('by', 'unknown')
                        time = _format_timestamp(story.get('time', 0)) if story.get('time') else 'unknown'
                        lines.append(f"## {idx}. {title}")
                        lines.append(f"- **ID**: {story['id']} | **Posted by**: {by} | **Date**: {time}")
                        if story.get('url'):
                            lines.append(f"- **URL**: {story['url']}")
                        lines.append("")

            has_more = (params.offset + len(paginated_ids)) < total_count
            if has_more:
                next_offset = params.offset + len(paginated_ids)
                lines.append(f"---")
                lines.append(f"More jobs available. Use offset={next_offset} to see next page.")

            content = "\n".join(lines)
            return _check_truncation(content, len(paginated_ids), "jobs")

        else:
            result = {
                "total": total_count,
                "count": len(paginated_ids),
                "offset": params.offset,
                "has_more": (params.offset + len(paginated_ids)) < total_count,
                "next_offset": params.offset + len(paginated_ids) if (params.offset + len(paginated_ids)) < total_count else None
            }

            if params.detail_level == DetailLevel.CONCISE:
                result["job_ids"] = paginated_ids
            else:
                result["jobs"] = [_format_item_json(s, params.detail_level) for s in stories]

            content = json.dumps(result, indent=2)
            return _check_truncation(content, len(paginated_ids), "jobs")

    except Exception as e:
        return _handle_api_error(e)

@mcp.tool(
    name="hn_get_user",
    annotations={
        "title": "Get HN User Profile",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def hn_get_user(params: UserInput) -> str:
    '''Get user profile information from Hacker News.

    This tool fetches public profile data for any HN user who has activity
    (comments or story submissions) on the site. Usernames are case-sensitive.

    Args:
        params (UserInput): Validated input parameters containing:
            - username (str): The case-sensitive username (e.g., 'jl', 'pg', 'dhouston')
            - response_format (ResponseFormat): 'markdown' (default) or 'json'
            - include_submissions (bool): Whether to include full submission list (default: False, can be very long)

    Returns:
        str: User profile information

        Markdown format includes:
        - Username, karma, account creation date
        - About section (user bio)
        - Submission count and IDs (if include_submissions=True)

        JSON format returns raw API response with all fields.

    Examples:
        - Use when: "Who is user 'pg' on HN?" -> Get Paul Graham's profile
        - Use when: "What's the karma for user 'jl'?" -> Get user karma
        - Use when: "Show me all submissions by user 'dhouston'" -> Set include_submissions=True
        - Don't use when: Username is unknown (will return 404 error)

    Error Handling:
        - Returns "Error: User not found" if username doesn't exist or has no public activity (404)
        - Usernames are case-sensitive - 'JL' and 'jl' are different users
        - Submission list can be very long (thousands of IDs) - use include_submissions=False by default
    '''
    try:
        user = await _fetch_from_api(f"user/{params.username}.json")

        if not user:
            return f"User '{params.username}' not found. Note: usernames are case-sensitive."

        if params.response_format == ResponseFormat.MARKDOWN:
            lines = [f"# Hacker News User: {user['id']}", ""]

            lines.append(f"**Karma**: {user.get('karma', 0)}")

            if user.get('created'):
                created_date = _format_timestamp(user['created'])
                lines.append(f"**Account Created**: {created_date}")

            lines.append("")

            if user.get('about'):
                lines.append("## About")
                lines.append("")
                lines.append(user['about'])
                lines.append("")

            if user.get('submitted'):
                submission_count = len(user['submitted'])
                lines.append(f"## Submissions ({submission_count} total)")
                lines.append("")

                if params.include_submissions:
                    # Show all submission IDs
                    lines.append(f"Submission IDs: {', '.join(map(str, user['submitted'][:100]))}")
                    if submission_count > 100:
                        lines.append(f"... and {submission_count - 100} more")
                else:
                    # Just show count and sample
                    lines.append(f"This user has {submission_count} submissions.")
                    lines.append(f"Recent submission IDs: {', '.join(map(str, user['submitted'][:10]))}")
                    lines.append("")
                    lines.append(f"Use include_submissions=True to see all {submission_count} submission IDs.")

            return "\n".join(lines)

        else:
            # JSON format
            result = user.copy()

            # Truncate submissions if not requested
            if not params.include_submissions and result.get('submitted'):
                submission_count = len(result['submitted'])
                result['submitted'] = result['submitted'][:10]
                result['submission_count'] = submission_count
                result['submissions_truncated'] = True

            return json.dumps(result, indent=2)

    except Exception as e:
        return _handle_api_error(e)

@mcp.tool(
    name="hn_get_max_item_id",
    annotations={
        "title": "Get Max HN Item ID",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def hn_get_max_item_id() -> str:
    '''Get the current maximum item ID on Hacker News.

    This tool returns the largest item ID currently on HN. Item IDs are assigned
    sequentially, so this represents the most recently created item. You can walk
    backward from this ID to discover all items.

    Args:
        None

    Returns:
        str: The maximum item ID as a string

    Examples:
        - Use when: "What's the newest item ID on HN?" -> Get max ID
        - Use when: "I want to iterate through recent HN items" -> Start from max ID and go backward
        - Don't use when: You want the newest stories (use hn_get_new_stories instead)

    Error Handling:
        - Returns "Error: Request timed out" if API is slow
    '''
    try:
        max_id = await _fetch_from_api("maxitem.json")
        return f"Current maximum item ID: {max_id}\n\nYou can use hn_get_item to fetch items from ID 1 up to {max_id}."

    except Exception as e:
        return _handle_api_error(e)

if __name__ == "__main__":
    mcp.run()
