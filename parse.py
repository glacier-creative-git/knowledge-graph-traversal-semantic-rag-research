#!/usr/bin/env python3
"""
Natural Questions Dataset Inspector
==================================

Simple script to examine the Natural Questions dataset structure
and test filtering/sampling approaches before integrating into the main pipeline.
"""

import os
from datasets import load_dataset
from itertools import islice
import json


def examine_natural_questions_structure():
    """Examine the structure of Natural Questions samples"""
    print("🔍 NATURAL QUESTIONS DATASET INSPECTOR")
    print("=" * 60)

    # Set up environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        print("📚 Loading Natural Questions streaming dataset...")
        streaming_dataset = load_dataset("natural_questions", split="validation", streaming=True)
        print("✅ Streaming dataset loaded")

        print("\n🔍 EXAMINING FIRST 5 SAMPLES:")
        print("-" * 40)

        # Examine first few samples
        for i, sample in enumerate(islice(streaming_dataset, 5)):
            print(f"\n📋 SAMPLE {i + 1}:")
            print(f"   All keys: {list(sample.keys())}")

            # Question structure
            if 'question' in sample:
                question = sample['question']
                print(f"   Question type: {type(question)}")
                if isinstance(question, dict):
                    print(f"   Question keys: {list(question.keys())}")
                    if 'text' in question:
                        print(f"   Question text: {question['text']}")
                else:
                    print(f"   Question value: {question}")

            # Document structure
            if 'document' in sample:
                doc = sample['document']
                print(f"   Document keys: {list(doc.keys())}")
                if 'html' in doc:
                    html_len = len(doc['html'])
                    print(f"   HTML length: {html_len} chars")
                    print(f"   HTML preview: {doc['html'][:200]}...")
                if 'title' in doc:
                    print(f"   Document title: {doc['title']}")

            # Annotations structure
            if 'annotations' in sample:
                annotations = sample['annotations']
                print(f"   Annotations type: {type(annotations)}")
                print(f"   Annotations length: {len(annotations) if isinstance(annotations, list) else 'not_list'}")
                if isinstance(annotations, list) and len(annotations) > 0:
                    first_annotation = annotations[0]
                    print(f"   First annotation type: {type(first_annotation)}")
                    if isinstance(first_annotation, dict):
                        print(f"   First annotation keys: {list(first_annotation.keys())}")
                else:
                    print(f"   Annotations value: {annotations}")

            # Long answer candidates
            if 'long_answer_candidates' in sample:
                candidates = sample['long_answer_candidates']
                print(
                    f"   Long answer candidates length: {len(candidates) if isinstance(candidates, list) else 'not_list'}")
                if isinstance(candidates, list) and len(candidates) > 0:
                    first_candidate = candidates[0]
                    print(
                        f"   First candidate keys: {list(first_candidate.keys()) if isinstance(first_candidate, dict) else 'not_dict'}")

            print("-" * 40)

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"❌ Error examining dataset: {e}")
        import traceback
        traceback.print_exc()


def test_filtering_approaches():
    """Test different filtering approaches for topics"""
    print("\n🧪 TESTING FILTERING APPROACHES")
    print("=" * 60)

    # Test keywords
    test_keywords = ["neuroscience", "brain", "science", "technology"]

    try:
        streaming_dataset = load_dataset("natural_questions", split="validation", streaming=True)

        print(f"🔍 Searching for samples containing: {test_keywords}")

        found_samples = []
        checked_count = 0
        max_to_check = 500  # Reasonable limit for testing

        for sample in streaming_dataset:
            checked_count += 1

            # Get question text
            question_text = ""
            if 'question' in sample and isinstance(sample['question'], dict):
                question_text = sample['question'].get('text', '').lower()

            # Get document info
            doc_title = ""
            doc_html = ""
            if 'document' in sample:
                doc = sample['document']
                doc_title = doc.get('title', '').lower()
                doc_html = doc.get('html', '')

            # Check for keyword matches
            matches = []
            for keyword in test_keywords:
                if keyword.lower() in question_text:
                    matches.append(f"question:{keyword}")
                if keyword.lower() in doc_title:
                    matches.append(f"title:{keyword}")
                # For HTML, just check if keyword appears (simplified)
                if keyword.lower() in doc_html.lower():
                    matches.append(f"content:{keyword}")

            if matches:
                found_samples.append({
                    'question': question_text,
                    'title': doc_title,
                    'html_length': len(doc_html),
                    'matches': matches,
                    'sample_number': checked_count
                })

                print(f"   ✅ Found #{len(found_samples)}: {question_text[:80]}...")
                print(f"      Matches: {matches}")
                print(f"      Title: {doc_title}")
                print(f"      HTML length: {len(doc_html)}")

                # Stop after finding a few examples
                if len(found_samples) >= 3:
                    print(f"   🎯 Found {len(found_samples)} examples after checking {checked_count} samples")
                    break

            # Progress updates
            if checked_count % 100 == 0:
                print(f"   📊 Checked {checked_count} samples, found {len(found_samples)} matches...")

            # Safety limit
            if checked_count >= max_to_check:
                print(f"   ⚠️ Checked maximum {max_to_check} samples")
                break

        print(f"\n📊 FILTERING RESULTS:")
        print(f"   Samples checked: {checked_count}")
        print(f"   Matches found: {len(found_samples)}")
        print(f"   Success rate: {len(found_samples) / checked_count * 100:.1f}%")

    except Exception as e:
        print(f"❌ Error testing filtering: {e}")
        import traceback
        traceback.print_exc()


def test_simple_sampling():
    """Test simple random sampling with islice"""
    print("\n🎲 TESTING SIMPLE RANDOM SAMPLING")
    print("=" * 60)

    try:
        streaming_dataset = load_dataset("natural_questions", split="validation", streaming=True)

        # Skip some random samples and take a few
        import random
        skip_count = random.randint(10, 100)
        take_count = 3

        print(f"🎲 Randomly skipping {skip_count} samples, then taking {take_count}")

        # Skip samples
        skipped_iter = islice(streaming_dataset, skip_count, None)

        # Take samples
        selected_samples = list(islice(skipped_iter, take_count))

        print(f"✅ Successfully collected {len(selected_samples)} samples")

        for i, sample in enumerate(selected_samples):
            question_text = sample.get('question', {}).get('text', 'No question')
            doc_title = sample.get('document', {}).get('title', 'No title')
            html_len = len(sample.get('document', {}).get('html', ''))

            print(f"\n📄 Sample {i + 1}:")
            print(f"   Question: {question_text}")
            print(f"   Title: {doc_title}")
            print(f"   HTML length: {html_len}")

            # Check annotations structure for this sample
            if 'annotations' in sample:
                annotations = sample['annotations']
                print(
                    f"   Annotations: {type(annotations)} with length {len(annotations) if isinstance(annotations, list) else 'not_list'}")
                if isinstance(annotations, list) and len(annotations) > 0:
                    print(
                        f"   First annotation keys: {list(annotations[0].keys()) if isinstance(annotations[0], dict) else 'not_dict'}")
                else:
                    print(f"   Annotations empty or unusual: {annotations}")

    except Exception as e:
        print(f"❌ Error testing sampling: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all inspection tests"""
    examine_natural_questions_structure()
    test_simple_sampling()
    test_filtering_approaches()

    print("\n🎯 NEXT STEPS:")
    print("   1. Look at the annotations structure output above")
    print("   2. Decide how to handle empty annotations")
    print("   3. Choose filtering strategy based on success rates")
    print("   4. Update data_loader.py with findings")


if __name__ == "__main__":
    main()