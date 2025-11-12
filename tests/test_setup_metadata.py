#!/usr/bin/env python3
"""
Test to verify setup.py metadata is correctly configured.

This test ensures that placeholder values have been replaced with real
author information and repository URLs.
"""

import unittest
import sys
import os
from setuptools import setup

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_setup_py_metadata():
    """Test that setup.py contains correct metadata, not placeholder values."""

    # Read setup.py content
    setup_py_path = os.path.join(os.path.dirname(__file__), '..', 'setup.py')
    with open(setup_py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Test that placeholder values are NOT present
    assert "Your Name" not in content, "setup.py still contains placeholder 'Your Name'"
    assert "your-email@example.com" not in content, "setup.py still contains placeholder 'your-email@example.com'"
    assert "your-username" not in content, "setup.py still contains placeholder 'your-username'"

    # Test that real values ARE present
    assert "Erin Green" in content, "setup.py missing real author name 'Erin Green'"
    assert "erin@democratize.technology" in content, "setup.py missing real email"
    assert "democratize-technology" in content, "setup.py missing real repository URL"

    # Verify the specific URL format
    assert "https://github.com/democratize-technology/chain-of-thought-tool" in content, \
        "setup.py missing correct repository URL"

    # Test that the package name is correct
    assert 'name="chain-of-thought-tool"' in content, "setup.py missing correct package name"

    # Test that version is present
    assert "version=" in content, "setup.py missing version information"

    # Test that description is present
    assert "description=" in content, "setup.py missing description"

    print("✅ All setup.py metadata tests passed!")

def test_setup_py_syntax():
    """Test that setup.py has valid Python syntax."""

    setup_py_path = os.path.join(os.path.dirname(__file__), '..', 'setup.py')

    try:
        with open(setup_py_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try to compile the setup.py to check syntax
        compile(content, setup_py_path, 'exec')
        print("✅ setup.py syntax is valid!")

    except SyntaxError as e:
        raise AssertionError(f"setup.py has syntax error: {e}")
    except Exception as e:
        raise AssertionError(f"setup.py cannot be processed: {e}")

def test_setup_py_required_fields():
    """Test that setup.py contains all required fields."""

    setup_py_path = os.path.join(os.path.dirname(__file__), '..', 'setup.py')
    with open(setup_py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    required_fields = [
        'name=',
        'version=',
        'author=',
        'author_email=',
        'description=',
        'packages=find_packages()',
        'python_requires=',
        'project_urls=',
        'classifiers='
    ]

    for field in required_fields:
        assert field in content, f"setup.py missing required field: {field}"

    print("✅ All required setup.py fields are present!")

if __name__ == "__main__":
    """Run all setup.py metadata tests."""

    print("🧪 Testing setup.py metadata configuration...")

    try:
        test_setup_py_syntax()
        test_setup_py_required_fields()
        test_setup_py_metadata()

        print("\n🎉 All setup.py tests passed successfully!")
        print("   ✓ No placeholder values found")
        print("   ✓ Real author information present")
        print("   ✓ Correct repository URLs")
        print("   ✓ All required fields present")
        print("   ✓ Valid Python syntax")

    except AssertionError as e:
        print(f"\n❌ Setup.py test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup.py testing: {e}")
        sys.exit(1)