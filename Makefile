# CSJ-ID Makefile
# ICML 2026 Submission

.PHONY: install test run quick clean figures report help

# Default target
help:
	@echo "CSJ-ID: Collaborative-Semantic Joint IDs"
	@echo "========================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run setup tests"
	@echo "  make run        - Run full experiment"
	@echo "  make quick      - Run quick experiment (fewer epochs)"
	@echo "  make figures    - Generate publication figures"
	@echo "  make report     - Generate experiment report"
	@echo "  make clean      - Clean output files"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	python src/test_setup.py

# Run full experiment
run:
	python src/run_experiments.py

# Run quick experiment
quick:
	python src/run_experiments.py --quick

# Generate figures
figures:
	python src/visualize.py --output_dir outputs

# Generate report
report:
	python src/report.py --output_dir outputs --save

# Clean outputs (keep models)
clean:
	rm -f outputs/*.txt
	rm -f outputs/*.png
	rm -f outputs/figures/*.png
	@echo "Cleaned output files (kept model checkpoints)"

# Clean everything
clean-all:
	rm -rf outputs/*
	@echo "Cleaned all output files"
