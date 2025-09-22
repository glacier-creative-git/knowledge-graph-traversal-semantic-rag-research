#!/usr/bin/env python3
"""
CSV Dataset Conversion Script
============================

Converts synthetic_dataset.json to CSV format for manual upload to DeepEval dashboard.
This allows dataset integration on the free tier by bypassing API upload limitations.

Usage:
    python conversion.py
    python conversion.py --input data/synthetic_dataset.json --output data/synthetic_dataset.csv
    python conversion.py --help
"""

import json
import csv
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any


def setup_logging() -> logging.Logger:
    """Configure logging for the conversion script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('DatasetConverter')


def load_json_dataset(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load the synthetic dataset from JSON file.

    Args:
        input_path: Path to the input JSON file

    Returns:
        List of dataset items as dictionaries

    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both single dict and list formats
    if isinstance(data, dict):
        data = [data]

    return data


def convert_context_to_string(context: List[str], delimiter: str = " | ") -> str:
    """
    Convert context list to delimited string for CSV storage.

    Args:
        context: List of context strings
        delimiter: String to separate context items

    Returns:
        Delimited string representation of context
    """
    if not context:
        return ""

    # Clean and escape any delimiter characters in context items
    cleaned_context = []
    for item in context:
        if isinstance(item, str):
            # Replace delimiter in content to avoid parsing issues
            cleaned_item = item.replace(delimiter, " ; ")
            cleaned_context.append(cleaned_item)

    return delimiter.join(cleaned_context)


def convert_to_csv_format(dataset: List[Dict[str, Any]], context_delimiter: str = " | ") -> List[Dict[str, str]]:
    """
    Convert dataset to CSV-compatible format.

    Args:
        dataset: List of dataset items
        context_delimiter: Delimiter for context array serialization

    Returns:
        List of dictionaries ready for CSV writing
    """
    csv_data = []

    for item in dataset:
        csv_row = {
            'input': item.get('input', ''),
            'expected_output': item.get('expected_output', ''),
            'actual_output': item.get('actual_output', ''),  # Usually empty for goldens
            'context': '',
            'retrieval_context': ''
        }

        # Handle context field (list of strings)
        if 'context' in item and item['context']:
            csv_row['context'] = convert_context_to_string(item['context'], context_delimiter)

        # Handle retrieval_context field (if present)
        if 'retrieval_context' in item and item['retrieval_context']:
            csv_row['retrieval_context'] = convert_context_to_string(item['retrieval_context'], context_delimiter)

        csv_data.append(csv_row)

    return csv_data


def write_csv_dataset(csv_data: List[Dict[str, str]], output_path: Path) -> None:
    """
    Write CSV data to file with proper formatting.

    Args:
        csv_data: List of dictionaries to write as CSV
        output_path: Path where to save the CSV file
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define field order for CSV
    fieldnames = ['input', 'expected_output', 'actual_output', 'context', 'retrieval_context']

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(csv_data)


def main():
    """Main conversion script entry point."""
    parser = argparse.ArgumentParser(
        description='Convert DeepEval synthetic dataset JSON to CSV for manual upload',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path('data/synthetic_dataset.json'),
        help='Input JSON dataset file path'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output CSV file path (default: same directory as input with .csv extension)'
    )

    parser.add_argument(
        '--context-delimiter',
        default=' | ',
        help='Delimiter for joining context arrays into strings'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Determine output path
    if args.output is None:
        args.output = args.input.with_suffix('.csv')

    try:
        logger.info(f"🔄 Converting dataset: {args.input} → {args.output}")

        # Load JSON dataset
        logger.info("📂 Loading JSON dataset...")
        dataset = load_json_dataset(args.input)
        logger.info(f"✅ Loaded {len(dataset)} items from JSON")

        # Convert to CSV format
        logger.info("🔄 Converting to CSV format...")
        csv_data = convert_to_csv_format(dataset, args.context_delimiter)
        logger.info(f"✅ Converted {len(csv_data)} items to CSV format")

        # Write CSV file
        logger.info("💾 Writing CSV file...")
        write_csv_dataset(csv_data, args.output)
        logger.info(f"✅ CSV file saved: {args.output}")

        # Print summary
        logger.info("📊 Conversion Summary:")
        logger.info(f"   Input file: {args.input}")
        logger.info(f"   Output file: {args.output}")
        logger.info(f"   Items converted: {len(csv_data)}")
        logger.info(f"   Context delimiter: '{args.context_delimiter}'")

        # Print upload instructions
        logger.info("")
        logger.info("🌐 Next Steps for Dashboard Upload:")
        logger.info("   1. Go to your DeepEval dashboard")
        logger.info("   2. Navigate to Datasets section")
        logger.info("   3. Click 'Upload Goldens' button")
        logger.info(f"   4. Upload the file: {args.output}")
        logger.info("   5. Map columns during import:")
        logger.info("      - input → input")
        logger.info("      - expected_output → expected_output")
        logger.info("      - context → context")
        logger.info(f"   6. Set context delimiter to: '{args.context_delimiter}'")
        logger.info("   7. Name your dataset to match your evaluation alias")
        logger.info("")
        logger.info("✅ Conversion completed successfully!")

    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())