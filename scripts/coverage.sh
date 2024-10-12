# cd to scripts parent directory
cd "$(dirname "$0")/.."

# Run tests
echo "Running tests..."
poetry run coverage run --source ./yada -m pytest
poetry run coverage report --show-missing --skip-empty