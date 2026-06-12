# Run with: just <task-name>
# Install just on mac: brew install just
# Install just if not present on linux (Debian based)
# curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
# To format justfile: just --fmt --unstable

# Variables
src_dir := "src/langchain_tool_args_validation_middleware"

# Default recipe to display available commands
default:
    @just --list

# Sync dependencies (including dev extras)
sync:
    uv sync --extra dev

# Auto-format code
format:
    uv run --extra dev ruff check --select I --fix .
    uv run --extra dev ruff format .

# Lint and format check
lint:
    uv run --extra dev ruff format --check --diff .
    uv run --extra dev ruff check --show-fixes .
    uv run --extra dev mypy {{ src_dir }}

# Run unit tests (optional path to narrow scope, e.g. just test tests/test_validation.py)
test path="tests":
    uv run --extra dev pytest --cov={{ src_dir }} {{ path }}

# Run format, lint, and tests to check everything before committing
pre-commit: format lint test

# Build sdist and wheel into dist/
build:
    rm -rf dist
    uv build

# Publish to TestPyPI (local: uses UV_PUBLISH_TOKEN / ~/.pypirc; CI uses trusted publishing)
publish-test: build
    uv publish --publish-url https://test.pypi.org/legacy/

# Publish to PyPI
publish: build
    uv publish

# Bump version (just bump patch|minor|major|rc), commit, and tag
bump level="patch":
    uv version --bump {{ level }}
    git commit -am "release: v$(uv version --short)"
    git tag "v$(uv version --short)"
    @echo "Pushed nothing yet. Run: git push && git push --tags"

# Clean up cache files
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + && \
    find . -type f -name "*.pyc" -delete && \
    rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache

# Install just if not present on mac
install-just-on-mac:
    @command -v just >/dev/null 2>&1 || { echo "Installing just..."; brew install just; }

# Install uv if not present
install-uv:
    @command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }

install-tools:
    uv tool install ruff==0.15.11 --force
    uv tool install mypy==1.18.2 --force
