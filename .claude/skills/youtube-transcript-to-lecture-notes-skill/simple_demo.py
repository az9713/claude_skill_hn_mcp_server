#!/usr/bin/env python3
"""
Simple Demonstration of Lecture Notes Converter
================================================
This script demonstrates the core functionality without external dependencies.
"""

import re
from pathlib import Path
from datetime import datetime

def simple_clean_transcript(text):
    """Simple transcript cleaning without NLTK."""
    # Remove timestamps
    text = re.sub(r'\[?\d{1,2}:\d{2}(:\d{2})?\]?', '', text)
    
    # Remove common filler words
    fillers = ['um', 'uh', 'ah', 'er', 'hmm', 'like', 'you know', 'I mean']
    for filler in fillers:
        text = re.sub(r'\b' + filler + r'\b', '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix punctuation spacing
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
    
    return text.strip()

def simple_structure_content(text):
    """Simple content structuring."""
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Group into sections (every 5 sentences)
    sections = []
    section_size = 5
    
    for i in range(0, len(sentences), section_size):
        section_sentences = sentences[i:i+section_size]
        if section_sentences:
            sections.append({
                'title': f"Section {len(sections) + 1}",
                'content': '. '.join(section_sentences) + '.'
            })
    
    return sections

def generate_html_output(title, sections):
    """Generate simple HTML output."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2c3e50;
            font-size: 32px;
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 3px solid #3498db;
        }}
        .metadata {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            font-size: 24px;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-left: 10px;
            border-left: 4px solid #3498db;
        }}
        p {{
            text-align: justify;
            margin-bottom: 15px;
            line-height: 1.8;
        }}
        .toc {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .toc h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc li {{
            margin-bottom: 10px;
        }}
        .toc a {{
            color: #3498db;
            text-decoration: none;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .concept-box {{
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
        }}
        .print-only {{
            display: none;
        }}
        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
            .toc {{
                page-break-after: always;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="metadata">
            Generated on {datetime.now().strftime('%B %d, %Y')} | {len(sections)} sections
        </div>
        
        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>
"""
    
    # Add TOC entries
    for i, section in enumerate(sections, 1):
        html += f'                <li><a href="#section-{i}">{section["title"]}</a></li>\n'
    
    html += """            </ul>
        </div>
        
        <div class="content">
"""
    
    # Add sections
    for i, section in enumerate(sections, 1):
        html += f"""
        <div class="section" id="section-{i}">
            <h2>{section['title']}</h2>
            <p>{section['content']}</p>
        </div>
"""
    
    html += """        </div>
    </div>
</body>
</html>"""
    
    return html

def generate_pdf_content(title, sections):
    """Generate simple PDF-like text output."""
    content = []
    content.append("=" * 60)
    content.append(title.center(60))
    content.append("=" * 60)
    content.append(f"Generated on {datetime.now().strftime('%B %d, %Y')}")
    content.append(f"Total Sections: {len(sections)}")
    content.append("")
    content.append("TABLE OF CONTENTS")
    content.append("-" * 40)
    
    for i, section in enumerate(sections, 1):
        content.append(f"{i}. {section['title']}")
    
    content.append("")
    content.append("=" * 60)
    content.append("")
    
    for i, section in enumerate(sections, 1):
        content.append(f"{i}. {section['title']}")
        content.append("-" * 40)
        content.append(section['content'])
        content.append("")
    
    return "\n".join(content)

def extract_key_concepts(text):
    """Simple key concept extraction."""
    # Find capitalized phrases (likely to be important terms)
    concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    
    # Count occurrences
    concept_counts = {}
    for concept in concepts:
        if len(concept) > 3:  # Skip short words
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
    
    # Get top concepts
    top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    return [concept for concept, count in top_concepts if count > 1]

def process_transcript(input_file, output_dir="output", title=None):
    """Main processing function."""
    # Read input
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    print(f"📖 Processing transcript: {input_file}")
    print(f"   Original length: {len(raw_text):,} characters")
    
    # Clean
    print("🧹 Cleaning transcript...")
    cleaned_text = simple_clean_transcript(raw_text)
    print(f"   Cleaned length: {len(cleaned_text):,} characters")
    print(f"   Reduction: {(1 - len(cleaned_text)/len(raw_text))*100:.1f}%")
    
    # Structure
    print("📐 Structuring content...")
    sections = simple_structure_content(cleaned_text)
    print(f"   Created {len(sections)} sections")
    
    # Extract concepts
    print("💡 Extracting key concepts...")
    concepts = extract_key_concepts(cleaned_text)
    if concepts:
        print(f"   Key concepts: {', '.join(concepts)}")
    
    # Add concepts to first section
    if sections and concepts:
        sections[0]['concepts'] = concepts
    
    # Generate outputs
    print("📄 Generating outputs...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate title if not provided
    if not title:
        title = "Lecture Notes"
        if concepts:
            title = f"Lecture on {concepts[0]}"
    
    # Generate HTML
    html_content = generate_html_output(title, sections)
    html_file = output_path / "lecture_notes.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"   ✓ HTML saved to: {html_file}")
    
    # Generate PDF-like text
    pdf_content = generate_pdf_content(title, sections)
    pdf_file = output_path / "lecture_notes.txt"
    with open(pdf_file, 'w', encoding='utf-8') as f:
        f.write(pdf_content)
    print(f"   ✓ Text saved to: {pdf_file}")
    
    return {
        'html_file': str(html_file),
        'pdf_file': str(pdf_file),
        'sections': len(sections),
        'concepts': concepts
    }

def create_sample_transcript():
    """Create a sample transcript for testing."""
    sample = """
[00:00:00] Um, hello everyone, and welcome to today's lecture on Machine Learning fundamentals.
[00:00:05] So, uh, today we're going to cover, you know, the basic concepts that underpin all of Machine Learning.

[00:00:15] First, let's talk about Supervised Learning. This is, like, the most common type of Machine Learning.
In Supervised Learning, we have labeled data - that means we know the correct answers for our training examples.
The algorithm learns from these examples and then can make predictions on new, unseen data.

[00:00:45] There are two main types of Supervised Learning problems: Classification and Regression.
Classification is when we're trying to predict discrete categories, like is this email spam or not spam.
Regression is when we're predicting continuous values, like what will the temperature be tomorrow.

[00:01:15] Um, let's look at a specific example - Decision Trees. These are, you know, really intuitive models.
A Decision Tree makes predictions by asking a series of questions about the input features.
Each internal node represents a question, and each leaf node represents a prediction.

[00:01:45] Now, uh, moving on to Unsupervised Learning. This is where things get really interesting.
In Unsupervised Learning, we don't have labeled data. The algorithm has to find patterns on its own.
Common techniques include Clustering, where we group similar data points together.

[00:02:15] K-means Clustering is probably the most well-known clustering algorithm.
It works by iteratively assigning data points to clusters and updating cluster centers.
The algorithm continues until the cluster assignments stop changing.

[00:02:45] Another important concept is Dimensionality Reduction. This is, like, super useful when dealing with high-dimensional data.
Principal Component Analysis, or PCA, is a popular technique for this.
PCA finds the directions of maximum variance in the data and projects onto those directions.

[00:03:15] Finally, let's briefly discuss Reinforcement Learning. This is, um, quite different from the other two.
In Reinforcement Learning, an agent learns by interacting with an environment.
The agent takes actions and receives rewards or penalties based on those actions.

[00:03:45] The goal is to learn a policy that maximizes the expected cumulative reward over time.
This type of learning is used in game playing, robotics, and many other applications.

[00:04:00] So, to summarize: we've covered Supervised Learning with Classification and Regression,
Unsupervised Learning with Clustering and Dimensionality Reduction,
and Reinforcement Learning with agents and environments.

[00:04:15] These are the fundamental paradigms of Machine Learning that you'll encounter throughout this course.
In our next lecture, we'll dive deeper into specific algorithms and their mathematical foundations.

[00:04:30] Um, any questions? Feel free to, you know, reach out during office hours. Thanks everyone!
"""
    
    with open('sample_transcript.txt', 'w', encoding='utf-8') as f:
        f.write(sample)
    
    return 'sample_transcript.txt'

def main():
    """Main demonstration."""
    print("\n" + "="*60)
    print(" YouTube Transcript to Lecture Notes Converter")
    print(" Simple Demonstration (No External Dependencies)")
    print("="*60 + "\n")
    
    # Create sample transcript
    print("📝 Creating sample transcript...")
    sample_file = create_sample_transcript()
    print(f"   ✓ Created: {sample_file}\n")
    
    # Process transcript
    result = process_transcript(
        sample_file,
        output_dir="simple_output",
        title="Machine Learning Fundamentals"
    )
    
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print("="*60)
    print("\n📊 Summary:")
    print(f"   • Sections created: {result['sections']}")
    print(f"   • Key concepts: {', '.join(result['concepts']) if result['concepts'] else 'None'}")
    print(f"\n📁 Output Files:")
    print(f"   • HTML: {result['html_file']}")
    print(f"   • Text: {result['pdf_file']}")
    print("\n💡 Next Steps:")
    print("   1. Open the HTML file in a browser to see the formatted notes")
    print("   2. The text file contains a printable version")
    print("\n" + "="*60)

if __name__ == '__main__':
    main()
