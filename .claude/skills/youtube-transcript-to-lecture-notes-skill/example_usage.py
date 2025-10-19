#!/usr/bin/env python3
"""
Example Usage Script for YouTube Transcript to Lecture Notes Converter
======================================================================
This script demonstrates how to use the converter with various configurations.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lecture_notes_converter import LectureNotesConverter, DEFAULT_CONFIG
import json

def example_basic_conversion():
    """Basic conversion with default settings."""
    print("=" * 60)
    print("Example 1: Basic Conversion")
    print("=" * 60)
    
    # Create converter with default config
    converter = LectureNotesConverter()
    
    # Convert transcript
    result = converter.convert(
        transcript_file='sample_transcript.txt',
        output_dir='output',
        title='Introduction to Machine Learning'
    )
    
    if result['success']:
        print(f"✓ Generated PDF: {result['pdf_path']}")
        print(f"✓ Generated HTML: {result['html_path']}")
    else:
        print(f"✗ Error: {result['error']}")

def example_custom_config():
    """Conversion with custom configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    # Custom configuration
    custom_config = {
        'cleaning': {
            'remove_fillers': True,
            'filler_threshold': 0.6,  # More aggressive cleaning
            'remove_timestamps': True,
            'normalize_whitespace': True
        },
        'structuring': {
            'window_size': 7,  # Larger context window
            'similarity_threshold': 0.25,  # More sections
            'min_section_sentences': 5,
            'max_section_sentences': 40
        },
        'enhancement': {
            'extract_concepts': True,
            'max_concepts_per_section': 7,  # More concepts
            'generate_examples': True,
            'examples_per_concept': 3
        },
        'output': {
            'pdf_page_size': 'A4',
            'html_theme': 'modern',
            'include_toc': True,
            'include_metadata': True
        }
    }
    
    # Create converter with custom config
    converter = LectureNotesConverter(custom_config)
    
    # Convert with verbose output
    result = converter.convert(
        transcript_file='sample_transcript.txt',
        output_dir='output',
        title='Advanced Deep Learning Architectures',
        verbose=True
    )
    
    if result['success']:
        print(f"\n✓ Conversion successful!")
        print(f"  Statistics:")
        stats = result['statistics']
        print(f"  • Original size: {stats['original_length']:,} chars")
        print(f"  • Cleaned size: {stats['cleaned_length']:,} chars")
        print(f"  • Sections created: {stats['sections']}")
        print(f"  • Concepts extracted: {stats['total_concepts']}")

def example_batch_conversion():
    """Convert multiple transcripts in batch."""
    print("\n" + "=" * 60)
    print("Example 3: Batch Conversion")
    print("=" * 60)
    
    # Setup converter
    converter = LectureNotesConverter()
    
    # List of transcripts to convert
    transcripts = [
        ('lecture1.txt', 'Introduction to Python'),
        ('lecture2.txt', 'Data Structures and Algorithms'),
        ('lecture3.txt', 'Web Development Fundamentals'),
    ]
    
    # Process each transcript
    results = []
    for transcript_file, title in transcripts:
        print(f"\nProcessing: {transcript_file}")
        
        # Check if file exists (for demo)
        if not Path(transcript_file).exists():
            print(f"  ⚠ File not found, skipping")
            continue
        
        result = converter.convert(
            transcript_file=transcript_file,
            output_dir='batch_output',
            title=title
        )
        
        results.append({
            'file': transcript_file,
            'title': title,
            'success': result['success'],
            'pdf': result.get('pdf_path', None),
            'html': result.get('html_path', None)
        })
    
    # Summary
    print("\n" + "-" * 40)
    print("Batch Conversion Summary:")
    successful = sum(1 for r in results if r['success'])
    print(f"✓ Successful: {successful}/{len(results)}")
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['title']}")

def create_sample_transcript():
    """Create a sample transcript for testing."""
    sample_text = """
[00:00:00] Um, hello everyone, and welcome to today's lecture on machine learning fundamentals.
[00:00:05] So, uh, today we're going to cover, you know, the basic concepts that, um, underpin all of machine learning.

[00:00:15] First, let's talk about supervised learning. This is, like, the most common type of machine learning.
In supervised learning, we have labeled data - that means we know the correct answers for our training examples.
The algorithm learns from these examples and then can make predictions on new, unseen data.

[00:00:45] There are two main types of supervised learning problems: classification and regression.
Classification is when we're trying to predict discrete categories, like is this email spam or not spam.
Regression is when we're predicting continuous values, like what will the temperature be tomorrow.

[00:01:15] Um, let's look at a specific example - decision trees. These are, you know, really intuitive models.
A decision tree makes predictions by asking a series of questions about the input features.
Each internal node represents a question, and each leaf node represents a prediction.

[00:01:45] Now, uh, moving on to unsupervised learning. This is where things get really interesting.
In unsupervised learning, we don't have labeled data. The algorithm has to find patterns on its own.
Common techniques include clustering, where we group similar data points together.

[00:02:15] K-means clustering is probably the most well-known clustering algorithm.
It works by iteratively assigning data points to clusters and updating cluster centers.
The algorithm continues until the cluster assignments stop changing.

[00:02:45] Another important concept is dimensionality reduction. This is, like, super useful when dealing with high-dimensional data.
Principal Component Analysis, or PCA, is a popular technique for this.
PCA finds the directions of maximum variance in the data and projects onto those directions.

[00:03:15] Finally, let's briefly discuss reinforcement learning. This is, um, quite different from the other two.
In reinforcement learning, an agent learns by interacting with an environment.
The agent takes actions and receives rewards or penalties based on those actions.

[00:03:45] The goal is to learn a policy that maximizes the expected cumulative reward over time.
This type of learning is used in game playing, robotics, and many other applications.

[00:04:00] So, to summarize: we've covered supervised learning with classification and regression,
unsupervised learning with clustering and dimensionality reduction,
and reinforcement learning with agents and environments.

[00:04:15] These are the fundamental paradigms of machine learning that you'll encounter throughout this course.
In our next lecture, we'll dive deeper into specific algorithms and their mathematical foundations.

[00:04:30] Um, any questions? Feel free to, you know, reach out during office hours. Thanks everyone!
"""
    
    # Save sample transcript
    with open('sample_transcript.txt', 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    print("✓ Created sample_transcript.txt")

def example_with_sample():
    """Complete example using the sample transcript."""
    print("\n" + "=" * 60)
    print("Example 4: Complete Demo with Sample Transcript")
    print("=" * 60)
    
    # Create sample transcript
    print("\n1. Creating sample transcript...")
    create_sample_transcript()
    
    # Configure converter
    print("\n2. Configuring converter...")
    config = {
        'cleaning': {
            'remove_fillers': True,
            'filler_threshold': 0.7,
            'remove_timestamps': True,
            'normalize_whitespace': True
        },
        'structuring': {
            'window_size': 5,
            'similarity_threshold': 0.3,
            'min_section_sentences': 3,
            'max_section_sentences': 50
        },
        'enhancement': {
            'extract_concepts': True,
            'max_concepts_per_section': 5,
            'generate_examples': True,
            'examples_per_concept': 2
        },
        'output': {
            'pdf_page_size': 'letter',
            'html_theme': 'academic',
            'include_toc': True,
            'include_metadata': True
        }
    }
    
    # Create output directory
    output_dir = Path('demo_output')
    output_dir.mkdir(exist_ok=True)
    
    # Convert
    print("\n3. Converting transcript to lecture notes...")
    converter = LectureNotesConverter(config)
    
    result = converter.convert(
        transcript_file='sample_transcript.txt',
        output_dir=str(output_dir),
        title='Machine Learning Fundamentals',
        verbose=True
    )
    
    if result['success']:
        print("\n" + "=" * 60)
        print("✅ CONVERSION SUCCESSFUL!")
        print("=" * 60)
        print(f"\n📄 PDF Output: {result['pdf_path']}")
        print(f"🌐 HTML Output: {result['html_path']}")
        
        stats = result['statistics']
        print(f"\n📊 Statistics:")
        print(f"  • Original transcript: {stats['original_length']:,} characters")
        print(f"  • After cleaning: {stats['cleaned_length']:,} characters")
        print(f"  • Reduction: {(1 - stats['cleaned_length']/stats['original_length'])*100:.1f}%")
        print(f"  • Sections created: {stats['sections']}")
        print(f"  • Key concepts extracted: {stats['total_concepts']}")
        
        print("\n💡 Next Steps:")
        print("  1. Open the HTML file in a browser to see the interactive version")
        print("  2. Open the PDF file to see the printable version")
        print("  3. Both files contain identical content with different formatting")
    else:
        print(f"\n❌ Conversion failed: {result['error']}")

def main():
    """Run all examples."""
    print("\n" + "🎓" * 30)
    print(" YouTube Transcript to Lecture Notes Converter")
    print(" Example Usage Demonstrations")
    print("🎓" * 30)
    
    # Run the complete demo with sample
    example_with_sample()
    
    # Uncomment to run other examples:
    # example_basic_conversion()
    # example_custom_config()
    # example_batch_conversion()
    
    print("\n" + "🎓" * 30)
    print(" Examples Complete!")
    print("🎓" * 30)

if __name__ == '__main__':
    main()
