# cd to scripts parent directory
cd "$(dirname "$0")/.."

# Run tests
echo "Running tests..."
poetry run pytest