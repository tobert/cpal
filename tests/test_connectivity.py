#!/usr/bin/env python3
"""
Manual test: Verify API connectivity and basic response.

Run with: uv run python tests/test_connectivity.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpal.server import get_client, get_model_aliases

# Skip if no API key available
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)


def test_connectivity():
    """Test basic API connectivity with each model tier."""
    client = get_client()

    print("Testing Claude API connectivity...\n")

    for alias, model in get_model_aliases().items():
        print(f"Testing {alias} ({model})...")
        try:
            response = client.messages.create(
                model=model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            )
            text = response.content[0].text if response.content else "(no response)"
            print(f"  ✓ Response: {text.strip()}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\nDone!")


if __name__ == "__main__":
    test_connectivity()
