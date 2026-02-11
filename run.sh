#!/bin/bash
# CSJ-ID Experiment Runner
# ICML 2026 Submission

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="$SCRIPT_DIR/src"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  CSJ-ID Experiment Runner${NC}"
echo -e "${GREEN}  ICML 2026 Submission${NC}"
echo -e "${GREEN}========================================${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Python version:${NC}"
python3 --version

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Not running in a virtual environment${NC}"
fi

# Parse arguments
QUICK_MODE=false
STAGE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quick) QUICK_MODE=true ;;
        --stage) STAGE="$2"; shift ;;
        -h|--help)
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick       Run quick test (fewer epochs)"
            echo "  --stage NAME  Run specific stage (data, semantic, lightgcn, rqvae, genrec, eval)"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Build command
CMD="python3 $SRC_DIR/run_experiments.py"

if [ "$QUICK_MODE" = true ]; then
    CMD="$CMD --quick"
    echo -e "${YELLOW}Running in QUICK mode (fewer epochs)${NC}"
fi

if [ -n "$STAGE" ]; then
    CMD="$CMD --stage $STAGE"
    echo -e "${YELLOW}Running stage: $STAGE${NC}"
fi

echo -e "${GREEN}Running: $CMD${NC}"
echo ""

# Run experiment
$CMD

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Experiment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Results saved to: $SCRIPT_DIR/outputs/"
