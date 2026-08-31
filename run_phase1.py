"""
Entry point for Phase 1.
Usage: python run_phase1.py
"""

from phase1.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(config_path="config/config.yaml")