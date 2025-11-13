#!/usr/bin/env python3
"""Script to prepare a release - bump version and update changelog."""

import re
import sys
from datetime import datetime
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
    print(f"✅ Updated __version__.py to {new_version}")


def update_changelog(new_version: str):
    """Update changelog - move Unreleased to versioned release."""
    changelog_file = Path(__file__).parent.parent / "CHANGELOG.md"

    if not changelog_file.exists():
        print("⚠️  CHANGELOG.md not found, skipping changelog update")
        return

    content = changelog_file.read_text()
    today = datetime.now().strftime("%Y-%m-%d")

    # Check if there are unreleased changes
    if "## [Unreleased]" not in content:
        print("⚠️  No [Unreleased] section found in CHANGELOG.md")
        return

    # Find the unreleased section
    unreleased_match = re.search(
        r"## \[Unreleased\]\s*\n(.*?)(?=\n## |\n\[|\Z)", content, re.DOTALL
    )
    if not unreleased_match:
        print("⚠️  Could not parse [Unreleased] section")
        return

    unreleased_content = unreleased_match.group(1).strip()

    if not unreleased_content or unreleased_content.isspace():
        print("⚠️  No changes found in [Unreleased] section")
        print("   Please add your changes to CHANGELOG.md before releasing")
        return False

    # Replace [Unreleased] with the new version
    new_content = content.replace(
        "## [Unreleased]", f"## [Unreleased]\n\n## [{new_version}] - {today}"
    )

    # Update the links at the bottom
    # First, add the new version link
    if f"[{new_version}]:" not in new_content:
        # Find existing version links pattern
        link_pattern = r"(\[Unreleased\]: .+?\.\.\.v[\d.]+\.\.\.HEAD\n)"
        if re.search(link_pattern, new_content):
            # Update Unreleased link to point to new version
            new_content = re.sub(
                r"\[Unreleased\]: (.+?)\.\.\.v[\d.]+\.\.\.HEAD",
                f"[Unreleased]: \\1...v{new_version}...HEAD",
                new_content,
            )

            # Add the new version link before the last link
            version_link = f"[{new_version}]: https://github.com/your-org/studioeval/releases/tag/v{new_version}"
            new_content = re.sub(
                r"(\[[\d.]+\]: https://github\.com/.+?\n)$",
                f"{version_link}\n\\1",
                new_content,
                flags=re.MULTILINE,
            )

    changelog_file.write_text(new_content)
    print(f"✅ Updated CHANGELOG.md with version {new_version}")
    return True


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ["major", "minor", "patch"]:
        print("Usage: python release.py [major|minor|patch]")
        print("\nThis script will:")
        print("  1. Bump the version number")
        print("  2. Update CHANGELOG.md (move [Unreleased] to new version)")
        print("  3. Remind you to commit and tag")
        sys.exit(1)

    bump_type = sys.argv[1]

    # Read current version from file
    version_file = Path(__file__).parent.parent / "__version__.py"
    content = version_file.read_text()

    # Extract version using regex
    match = re.search(r'__version__ = "([^"]+)"', content)
    if not match:
        print("❌ Could not find version in __version__.py")
        sys.exit(1)

    current_version = match.group(1)

    # Calculate new version
    new_version = bump_version(current_version, bump_type)

    print(f"🚀 Preparing release {current_version} → {new_version}")
    print()

    # Update version file
    update_version_file(new_version)

    # Update changelog
    changelog_updated = update_changelog(new_version)

    print()
    print(f"🎉 Release {new_version} prepared!")
    print()
    print("Next steps:")
    print("  1. Review the changes:")
    print("     git diff")
    print("  2. Commit the release:")
    print(f"     git add . && git commit -m 'Release v{new_version}'")
    print("  3. Tag the release:")
    print(f"     git tag v{new_version}")
    print("  4. Push to remote:")
    print("     git push && git push --tags")

    if not changelog_updated:
        print()
        print("⚠️  Note: Changelog was not automatically updated.")
        print("   Please manually update CHANGELOG.md before committing.")


if __name__ == "__main__":
    main()
