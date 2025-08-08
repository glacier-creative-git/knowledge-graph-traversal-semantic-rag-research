#!/usr/bin/env python3
"""
Semantic RAG Pipeline - Main Entry Point
========================================

Main executable for running the semantic graph traversal RAG system.
Supports various execution modes and configuration overrides.
"""

import argparse
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from pipeline import SemanticRAGPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Semantic RAG Pipeline - Graph traversal based retrieval system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run full pipeline with default config
  python main.py --config custom.yaml     # Use custom config file
  python main.py --mode data_only          # Only run data acquisition phase
  python main.py --device cuda            # Force CUDA device
  python main.py --no-viz                 # Disable visualizations
        """
    )

    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    # Execution mode override
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['full_pipeline', 'data_only', 'embedding_only', 'evaluation_only', 'visualization_only'],
        help='Override execution mode from config'
    )

    # Device override
    parser.add_argument(
        '--device', '-d',
        type=str,
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Override device selection from config'
    )

    # Quick toggles
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable all visualizations'
    )

    parser.add_argument(
        '--force-recompute',
        type=str,
        nargs='+',
        choices=['data', 'embeddings', 'similarities', 'datasets', 'all'],
        help='Force recomputation of specific components'
    )

    parser.add_argument(
        '--skip-phases',
        type=str,
        nargs='+',
        help='Skip specific pipeline phases'
    )

    # Logging control
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Override logging level from config'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal console output'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose console output'
    )

    return parser.parse_args()


def apply_config_overrides(pipeline, args):
    """Apply command line argument overrides to pipeline config."""

    # Override execution mode
    if args.mode:
        pipeline.config['execution']['mode'] = args.mode
        print(f"🔄 Execution mode override: {args.mode}")

    # Override device
    if args.device:
        pipeline.config['system']['device'] = args.device
        print(f"🔄 Device override: {args.device}")

    # Override visualizations
    if args.no_viz:
        pipeline.config['visualization']['enabled'] = False
        print(f"🔄 Visualizations disabled")

    # Override force recompute
    if args.force_recompute:
        if 'all' in args.force_recompute:
            pipeline.config['execution']['force_recompute'] = ['data', 'embeddings', 'similarities', 'datasets']
        else:
            pipeline.config['execution']['force_recompute'] = args.force_recompute
        print(f"🔄 Force recompute: {args.force_recompute}")

    # Override skip phases
    if args.skip_phases:
        pipeline.config['execution']['skip_phases'] = args.skip_phases
        print(f"🔄 Skip phases: {args.skip_phases}")

    # Override logging level
    if args.log_level:
        pipeline.config['logging']['level'] = args.log_level
        print(f"🔄 Log level override: {args.log_level}")

    # Adjust logging for quiet/verbose
    if args.quiet:
        pipeline.config['logging']['log_to_console'] = False
        pipeline.config['logging']['level'] = 'WARNING'
    elif args.verbose:
        pipeline.config['logging']['level'] = 'DEBUG'


def main():
    """Main entry point."""
    print("🧠 Semantic RAG Pipeline")
    print("=" * 50)

    try:
        # Parse arguments
        args = parse_arguments()

        # Check if config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            print(f"💡 Create {config_path} or specify a different config with --config")
            sys.exit(1)

        # Initialize pipeline
        print(f"📋 Loading config from: {config_path}")
        pipeline = SemanticRAGPipeline(str(config_path))

        # Load config first to apply overrides
        pipeline._load_config()

        # Apply command line overrides
        apply_config_overrides(pipeline, args)

        # Show execution plan
        print(f"\n📋 Execution Plan:")
        print(f"   Mode: {pipeline.config['execution']['mode']}")
        print(f"   Device: {pipeline.config['system']['device']}")
        print(f"   Visualizations: {pipeline.config['visualization']['enabled']}")

        if pipeline.config['execution']['skip_phases']:
            print(f"   Skip phases: {pipeline.config['execution']['skip_phases']}")
        if pipeline.config['execution']['force_recompute']:
            print(f"   Force recompute: {pipeline.config['execution']['force_recompute']}")

        print()

        # Run pipeline
        results = pipeline.pipe()

        # Success message
        print("\n" + "=" * 50)
        print("🎉 Pipeline completed successfully!")
        print(f"📋 Experiment ID: {results['experiment_id']}")
        print(f"⏱️  Execution time: {results['execution_time']}")
        print(f"📁 Results saved in experiments directory")

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()