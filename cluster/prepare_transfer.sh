#!/bin/bash
# ============================================
# CSJ-ID: Prepare files for cluster transfer
# Destination: Generative_Recommendation
# ============================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration - UPDATE THIS WITH YOUR NEU USERNAME
NEU_USERNAME="your_neu_username"  # <-- CHANGE THIS!

# Paths
PROJECT_DIR="/Users/ritik/Desktop/Research/ICMLFinal"
TRANSFER_DIR="/Users/ritik/Desktop/Research/Generative_Recommendation_transfer"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  CSJ-ID Cluster Transfer Preparation${NC}"
echo -e "${GREEN}  Destination: Generative_Recommendation${NC}"
echo -e "${GREEN}============================================${NC}"

# Check if username is set
if [ "$NEU_USERNAME" == "your_neu_username" ]; then
    echo -e "${YELLOW}⚠️  Please edit this script and set your NEU_USERNAME first!${NC}"
    echo "   Open: $PROJECT_DIR/cluster/prepare_transfer.sh"
    echo "   Change line 14: NEU_USERNAME=\"your_actual_username\""
    exit 1
fi

# Create clean transfer directory
echo -e "\n${BLUE}Creating transfer directory...${NC}"
rm -rf "$TRANSFER_DIR"
mkdir -p "$TRANSFER_DIR"

# Copy essential files only
echo -e "${BLUE}Copying essential files...${NC}"

# Source code
cp -r "$PROJECT_DIR/src" "$TRANSFER_DIR/"
rm -rf "$TRANSFER_DIR/src/__pycache__"  # Remove cache

# Cluster scripts
cp -r "$PROJECT_DIR/cluster" "$TRANSFER_DIR/"

# Datasets
echo -e "${BLUE}Copying datasets (this may take a moment)...${NC}"
cp "$PROJECT_DIR/Beauty.json.gz" "$TRANSFER_DIR/"
cp "$PROJECT_DIR/Sports.json.gz" "$TRANSFER_DIR/"

# Config files
cp "$PROJECT_DIR/requirements.txt" "$TRANSFER_DIR/"
cp "$PROJECT_DIR/run.sh" "$TRANSFER_DIR/"
cp "$PROJECT_DIR/README.md" "$TRANSFER_DIR/"

# Create necessary directories
mkdir -p "$TRANSFER_DIR/logs"
mkdir -p "$TRANSFER_DIR/outputs_beauty"
mkdir -p "$TRANSFER_DIR/outputs_sports"

# Calculate size
TRANSFER_SIZE=$(du -sh "$TRANSFER_DIR" | cut -f1)

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}  Transfer package ready!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "📁 Location: ${BLUE}$TRANSFER_DIR${NC}"
echo -e "📦 Size: ${BLUE}$TRANSFER_SIZE${NC}"
echo ""
echo -e "${YELLOW}Contents:${NC}"
ls -la "$TRANSFER_DIR"
echo ""

# Print transfer instructions
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Transfer Instructions${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${YELLOW}Option 1: Using scp (recommended)${NC}"
echo -e "Run this command in your terminal:"
echo ""
echo -e "${BLUE}scp -r $TRANSFER_DIR ${NEU_USERNAME}@xfer.discovery.neu.edu:/scratch/${NEU_USERNAME}/research/Generative_Recommendation${NC}"
echo ""
echo -e "${YELLOW}Option 2: Using rsync (for resumable transfer)${NC}"
echo ""
echo -e "${BLUE}rsync -avz --progress $TRANSFER_DIR/ ${NEU_USERNAME}@xfer.discovery.neu.edu:/scratch/${NEU_USERNAME}/research/Generative_Recommendation/${NC}"
echo ""
echo -e "${YELLOW}Option 3: Using OOD Web Interface${NC}"
echo "1. Go to https://ood.explorer.northeastern.edu/"
echo "2. Click 'Files' → 'Home Directory'"
echo "3. Navigate to /scratch/${NEU_USERNAME}/research/"
echo "4. Create folder 'Generative_Recommendation'"
echo "5. Upload files from: $TRANSFER_DIR"
echo ""
echo -e "${GREEN}After transfer, SSH to cluster and run:${NC}"
echo -e "${BLUE}ssh ${NEU_USERNAME}@login.discovery.neu.edu${NC}"
echo -e "${BLUE}cd /scratch/\$USER/research/Generative_Recommendation${NC}"
echo -e "${BLUE}bash cluster/setup_cluster.sh${NC}"
