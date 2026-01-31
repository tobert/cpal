#!/usr/bin/env python3
"""
Manual test: Verify Claude can use tools autonomously.

Run with: uv run python tests/test_agentic.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpal.server import _consult

# Skip if no API key available
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)


def test_agentic_exploration():
    """Test that Claude can autonomously explore the codebase."""
    print("Testing agentic exploration with Haiku...\n")

    response = _consult(
        query="List the files in the current directory and tell me what kind of project this is.",
        session_id="test-agentic",
        model_alias="haiku",
    )

    print("Response:")
    print("-" * 60)
    print(response)
    print("-" * 60)
    print()


def test_file_reading():
    """Test that Claude can read files autonomously."""
    print("Testing file reading with Sonnet...\n")

    response = _consult(
        query="Read the pyproject.toml and summarize what this project does in one sentence.",
        session_id="test-reading",
        model_alias="sonnet",
    )

    print("Response:")
    print("-" * 60)
    print(response)
    print("-" * 60)
    print()


def test_extended_thinking():
    """Test extended thinking mode."""
    print("Testing extended thinking with Sonnet...\n")

    response = _consult(
        query="What is the most important design decision in this codebase? Think carefully.",
        session_id="test-thinking",
        model_alias="sonnet",
        extended_thinking=True,
        thinking_budget=5000,
    )

    print("Response:")
    print("-" * 60)
    print(response)
    print("-" * 60)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--explore", action="store_true", help="Run exploration test")
    parser.add_argument("--read", action="store_true", help="Run file reading test")
    parser.add_argument("--think", action="store_true", help="Run extended thinking test")
    args = parser.parse_args()

    if args.all or args.explore:
        test_agentic_exploration()

    if args.all or args.read:
        test_file_reading()

    if args.all or args.think:
        test_extended_thinking()

    if not any([args.all, args.explore, args.read, args.think]):
        print("Usage: python test_agentic.py [--all|--explore|--read|--think]")
