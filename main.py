"""
Main Entry Point for NEC ML Pipeline
Single-command execution
"""

import argparse
from src.final_pipeline import NECPipeline


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='NEC ML Pipeline - Smart Plant Selection'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: Use reduced parameter grid for faster execution'
    )
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Skip hyperparameter tuning (baseline only)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = NECPipeline(verbose=not args.quiet)
    
    results = pipeline.run_complete_pipeline(
        tune=not args.no_tune,
        quick=args.quick
    )
    
    print("\n Pipeline execution complete!")
    print(" Check results/ directory for outputs")


if __name__ == "__main__":
    main()