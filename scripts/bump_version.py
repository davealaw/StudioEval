#!/usr/bin/env python3
"""Script to bump version numbers."""

import re
import sys
from pathlib import Path


def bump_version(current_version: str, bump_type: str) -> str:
    """Bump version based on type (major, minor, patch)."""
    major, minor, patch = map(int, current_version.split("."))

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def update_version_file(new_version: str):
    """Update the __version__.py file."""
    version_file = Path(__file__).parent.parent / "__version__.py"
    content = version_file.read_text()

    # Update __version__ line
    content = re.sub(
        r'__version__ = "[^"]*"', f'__version__ = "{new_version}"', content
    )

    version_file.write_text(content)
    print(f"Updated __version__.py to {new_version}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ["major", "minor", "patch"]:
        print("Usage: python bump_version.py [major|minor|patch]")
        sys.exit(1)

    bump_type = sys.argv[1]

    # Read current version from file
    version_file = Path(__file__).parent.parent / "__version__.py"
    content = version_file.read_text()

    # Extract version using regex
    match = re.search(r'__version__ = "([^"]+)"', content)
    if not match:
        print("Could not find version in __version__.py")
        sys.exit(1)

    current_version = match.group(1)

    # Calculate new version
    new_version = bump_version(current_version, bump_type)

    print(f"Bumping version from {current_version} to {new_version}")

    # Update version file
    update_version_file(new_version)

    print(f"Version bumped to {new_version}")
    print("Don't forget to commit and tag the release!")


if __name__ == "__main__":
    main()
