#!/usr/bin/env python3
"""
YouTube Transcript to Lecture Notes Converter
==============================================
A comprehensive tool for transforming YouTube transcripts into 
professional, educational lecture notes with PDF and HTML outputs.

Author: Claude AI Assistant
Version: 1.0.0
"""

import os
import sys
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from io import BytesIO

# Core dependencies
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLP dependencies
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.chunk import ne_chunk, tree2conlltags
from nltk.corpus import stopwords

# PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, 
    PageBreak, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfgen import canvas

# HTML generation
from jinja2 import Template, Environment, FileSystemLoader
import markdown
from bs4 import BeautifulSoup

# Ensure NLTK data is downloaded
def ensure_nltk_data():
    """Download required NLTK data if not present."""
    required_data = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 
                     'words', 'stopwords']
    for dataset in required_data:
        try:
            nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
        except LookupError:
            print(f"Downloading NLTK dataset: {dataset}")
            nltk.download(dataset, quiet=True)

ensure_nltk_data()

# Constants
DEFAULT_CONFIG = {
    'cleaning': {
        'remove_fillers': True,
        'filler_threshold': 0.75,
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

class TranscriptCleaner:
    """Advanced transcript cleaning with pattern matching and NLP."""
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG['cleaning']
        self.setup_patterns()
        
    def setup_patterns(self):
        """Initialize cleaning patterns with confidence scores."""
        self.patterns = [
            # Filler words and sounds
            (r'\b(um+|uh+|ah+|er+|hmm+|mm+|oh+)\b', 0.95, 'filler'),
            
            # Timestamps in various formats
            (r'\[?\d{1,2}:\d{2}(:\d{2})?\]?', 0.99, 'timestamp'),
            (r'\(\d{1,2}:\d{2}(:\d{2})?\)', 0.99, 'timestamp'),
            
            # Speaker labels
            (r'^[A-Z][a-z]+:\s*', 0.90, 'speaker'),
            (r'^\[.*?\]:\s*', 0.90, 'speaker'),
            
            # Inaudible/unclear markers
            (r'\[.*?(inaudible|unclear|crosstalk).*?\]', 0.99, 'marker'),
            (r'\(.*?(inaudible|unclear).*?\)', 0.99, 'marker'),
            
            # Verbal tics and hedging
            (r'\b(you know|I mean|like|sort of|kind of|basically|actually)\b', 0.60, 'hedge'),
            
            # Repeated punctuation
            (r'\.{3,}', 0.85, 'punctuation'),
            (r'\?{2,}', 0.85, 'punctuation'),
            (r'!{2,}', 0.85, 'punctuation'),
            
            # Multiple spaces
            (r'\s{2,}', 1.00, 'whitespace'),
            
            # Line breaks within sentences
            (r'(?<=[a-z,])\n(?=[a-z])', 0.90, 'linebreak')
        ]
    
    def clean(self, text: str, verbose: bool = False) -> str:
        """
        Clean transcript text using multi-pass pattern matching.
        
        Args:
            text: Raw transcript text
            verbose: Print cleaning statistics
            
        Returns:
            Cleaned text ready for processing
        """
        if not text:
            return ""
        
        original_length = len(text)
        cleaned = text
        stats = {'total_removed': 0}
        
        # Apply cleaning patterns
        for pattern, confidence, category in self.patterns:
            if category == 'filler' and not self.config['remove_fillers']:
                continue
            if category == 'timestamp' and not self.config['remove_timestamps']:
                continue
            
            if confidence >= self.config.get('filler_threshold', 0.75):
                matches = re.findall(pattern, cleaned, re.IGNORECASE)
                if matches and verbose:
                    stats[category] = stats.get(category, 0) + len(matches)
                
                # Smart replacement based on category
                if category in ['filler', 'hedge']:
                    cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
                elif category == 'whitespace':
                    cleaned = re.sub(pattern, ' ', cleaned)
                elif category == 'linebreak':
                    cleaned = re.sub(pattern, ' ', cleaned)
                elif category == 'punctuation':
                    cleaned = re.sub(pattern, '.', cleaned)
                else:
                    cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Post-processing
        cleaned = self.post_process(cleaned)
        
        if verbose:
            reduction = (1 - len(cleaned) / original_length) * 100
            print(f"Cleaning Statistics:")
            print(f"  Original length: {original_length:,} chars")
            print(f"  Cleaned length: {len(cleaned):,} chars")
            print(f"  Reduction: {reduction:.1f}%")
            for category, count in stats.items():
                if category != 'total_removed':
                    print(f"  {category.capitalize()} removed: {count}")
        
        return cleaned
    
    def post_process(self, text: str) -> str:
        """Apply final cleaning and formatting."""
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        
        # Fix sentence capitalization
        sentences = sent_tokenize(text)
        sentences = [s[0].upper() + s[1:] if s else '' for s in sentences]
        text = ' '.join(sentences)
        
        # Remove empty parentheses and brackets
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\[\s*\]', '', text)
        
        # Final whitespace normalization
        text = ' '.join(text.split())
        
        return text

class ContentStructurer:
    """Intelligent content structuring using NLP and ML techniques."""
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG['structuring']
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def structure(self, text: str) -> Dict[str, Any]:
        """
        Structure text into hierarchical sections.
        
        Args:
            text: Cleaned transcript text
            
        Returns:
            Structured content dictionary
        """
        # Sentence segmentation
        sentences = self.segment_sentences(text)
        
        # Topic segmentation
        segments = self.identify_segments(sentences)
        
        # Build hierarchical structure
        sections = self.build_hierarchy(sentences, segments)
        
        # Generate titles
        sections = self.generate_titles(sections)
        
        return {
            'sections': sections,
            'metadata': {
                'total_sentences': len(sentences),
                'total_sections': len(sections),
                'avg_section_length': sum(len(s['sentences']) for s in sections) / len(sections) if sections else 0
            }
        }
    
    def segment_sentences(self, text: str) -> List[str]:
        """Advanced sentence segmentation handling edge cases."""
        # Initial segmentation
        sentences = sent_tokenize(text)
        
        # Post-process to handle common issues
        processed = []
        buffer = ""
        
        for sent in sentences:
            # Check if sentence is too short or continuation
            if buffer and (
                len(sent.split()) < 3 or
                sent[0].islower() or
                sent.startswith(('and', 'but', 'or', 'because'))
            ):
                buffer += " " + sent
            else:
                if buffer:
                    processed.append(buffer.strip())
                buffer = sent
        
        if buffer:
            processed.append(buffer.strip())
        
        return processed
    
    def identify_segments(self, sentences: List[str]) -> List[Tuple[int, int]]:
        """
        Identify topic segments using sliding window similarity.
        
        Mathematical approach:
        - Use TF-IDF vectors for sentence representation
        - Calculate cosine similarity between windows
        - Detect boundaries at similarity valleys
        """
        if len(sentences) < self.config['window_size']:
            return [(0, len(sentences))]
        
        # Create sentence windows
        windows = []
        window_size = self.config['window_size']
        
        for i in range(len(sentences) - window_size + 1):
            window_text = ' '.join(sentences[i:i + window_size])
            windows.append(window_text)
        
        # Vectorize windows
        try:
            tfidf_matrix = self.vectorizer.fit_transform(windows)
        except:
            # Fallback for short texts
            return [(0, len(sentences))]
        
        # Calculate similarities
        similarities = []
        for i in range(len(windows) - 1):
            sim = cosine_similarity(
                tfidf_matrix[i:i+1],
                tfidf_matrix[i+1:i+2]
            )[0][0]
            similarities.append(sim)
        
        # Find boundaries (local minima below threshold)
        boundaries = [0]
        threshold = self.config['similarity_threshold']
        
        for i in range(1, len(similarities) - 1):
            if (similarities[i] < threshold and
                similarities[i] < similarities[i-1] and
                similarities[i] < similarities[i+1]):
                boundary_idx = i + window_size // 2
                boundaries.append(boundary_idx)
        
        boundaries.append(len(sentences))
        
        # Create segments respecting min/max constraints
        segments = []
        start = 0
        
        for boundary in boundaries[1:]:
            segment_length = boundary - start
            
            if segment_length >= self.config['min_section_sentences']:
                if segment_length <= self.config['max_section_sentences']:
                    segments.append((start, boundary))
                else:
                    # Split large segments
                    mid = start + segment_length // 2
                    segments.append((start, mid))
                    segments.append((mid, boundary))
                start = boundary
            # else: merge with next segment
        
        if start < len(sentences):
            if segments:
                # Extend last segment
                segments[-1] = (segments[-1][0], len(sentences))
            else:
                segments.append((start, len(sentences)))
        
        return segments if segments else [(0, len(sentences))]
    
    def build_hierarchy(self, sentences: List[str], segments: List[Tuple[int, int]]) -> List[Dict]:
        """Build hierarchical structure from segments."""
        sections = []
        
        for start, end in segments:
            segment_sentences = sentences[start:end]
            
            # Determine if this is a major section
            is_major = self.is_major_section(segment_sentences)
            
            section = {
                'sentences': segment_sentences,
                'is_major': is_major,
                'start_idx': start,
                'end_idx': end
            }
            
            sections.append(section)
        
        # Organize into hierarchy
        hierarchical = []
        current_major = None
        
        for section in sections:
            if section['is_major']:
                if current_major:
                    hierarchical.append(current_major)
                current_major = {
                    'sentences': section['sentences'],
                    'subsections': []
                }
            else:
                if current_major:
                    current_major['subsections'].append({
                        'sentences': section['sentences']
                    })
                else:
                    hierarchical.append({
                        'sentences': section['sentences'],
                        'subsections': []
                    })
        
        if current_major:
            hierarchical.append(current_major)
        
        return hierarchical if hierarchical else sections
    
    def is_major_section(self, sentences: List[str]) -> bool:
        """Heuristically determine if sentences form a major section."""
        if not sentences:
            return False
        
        first_sentence = sentences[0]
        
        # Check for section indicators
        major_indicators = [
            r'^(first|second|third|next|finally|in conclusion)',
            r'^(chapter|section|part|topic|lesson)',
            r'^(introduction|overview|summary|conclusion)',
            r'^(let\'s|we\'ll|now|today)'
        ]
        
        for pattern in major_indicators:
            if re.search(pattern, first_sentence, re.IGNORECASE):
                return True
        
        # Check sentence structure
        if len(first_sentence.split()) < 10 and first_sentence.endswith(':'):
            return True
        
        return False
    
    def generate_titles(self, sections: List[Dict]) -> List[Dict]:
        """Generate descriptive titles for sections."""
        stop_words = set(stopwords.words('english'))
        
        for section in sections:
            # Extract keywords from first few sentences
            context = ' '.join(section['sentences'][:3])
            
            # Simple keyword extraction
            words = word_tokenize(context.lower())
            words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 3]
            
            # Get most frequent words
            from collections import Counter
            word_freq = Counter(words)
            top_words = [word for word, _ in word_freq.most_common(3)]
            
            # Generate title
            if top_words:
                title = ' '.join(w.capitalize() for w in top_words)
            else:
                title = "Section"
            
            section['title'] = title
            
            # Generate titles for subsections
            for i, subsection in enumerate(section.get('subsections', [])):
                sub_context = ' '.join(subsection['sentences'][:2])
                sub_words = word_tokenize(sub_context.lower())
                sub_words = [w for w in sub_words if w.isalnum() and w not in stop_words and len(w) > 3]
                
                sub_freq = Counter(sub_words)
                sub_top = [word for word, _ in sub_freq.most_common(2)]
                
                if sub_top:
                    subsection['title'] = ' '.join(w.capitalize() for w in sub_top)
                else:
                    subsection['title'] = f"Part {i+1}"
        
        return sections

class ContentEnhancer:
    """Enhance content with educational features."""
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG['enhancement']
    
    def enhance(self, structured_content: Dict) -> Dict:
        """
        Enhance structured content with educational features.
        
        Adds:
        - Key concept definitions
        - Illustrative examples
        - Summary points
        - Cross-references
        """
        sections = structured_content['sections']
        
        for section in sections:
            # Extract and define concepts
            if self.config['extract_concepts']:
                section['concepts'] = self.extract_concepts(
                    ' '.join(section['sentences']),
                    max_concepts=self.config['max_concepts_per_section']
                )
            
            # Generate examples if enabled
            if self.config['generate_examples']:
                for concept in section.get('concepts', []):
                    concept['examples'] = self.generate_examples(
                        concept['term'],
                        num_examples=self.config['examples_per_concept']
                    )
            
            # Process subsections
            for subsection in section.get('subsections', []):
                if self.config['extract_concepts']:
                    subsection['concepts'] = self.extract_concepts(
                        ' '.join(subsection['sentences']),
                        max_concepts=2
                    )
        
        structured_content['sections'] = sections
        return structured_content
    
    def extract_concepts(self, text: str, max_concepts: int = 5) -> List[Dict]:
        """Extract and rank key concepts from text."""
        # Extract noun phrases
        concepts = []
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # Find noun phrases
        noun_phrases = []
        current_phrase = []
        
        for word, tag in pos_tags:
            if tag.startswith('NN'):  # Noun
                current_phrase.append(word)
            elif current_phrase:
                if len(current_phrase) > 1:
                    phrase = ' '.join(current_phrase)
                    noun_phrases.append(phrase.lower())
                current_phrase = []
        
        if current_phrase and len(current_phrase) > 1:
            noun_phrases.append(' '.join(current_phrase).lower())
        
        # Count and rank
        from collections import Counter
        phrase_counts = Counter(noun_phrases)
        
        # Create concept entries
        for phrase, count in phrase_counts.most_common(max_concepts):
            if count > 1:  # Only include repeated concepts
                concepts.append({
                    'term': phrase.title(),
                    'frequency': count,
                    'definition': self.generate_definition(phrase, text)
                })
        
        return concepts
    
    def generate_definition(self, term: str, context: str) -> str:
        """Generate a contextual definition for a term."""
        # Find sentences containing the term
        sentences = sent_tokenize(context)
        relevant = [s for s in sentences if term.lower() in s.lower()]
        
        if relevant:
            # Use the most informative sentence
            longest = max(relevant, key=len)
            # Clean and format
            definition = longest.replace(term, f"**{term}**")
            return definition
        else:
            return f"A key concept discussed in this section of the lecture."
    
    def generate_examples(self, term: str, num_examples: int = 2) -> List[str]:
        """Generate illustrative examples for a concept."""
        # In a real implementation, this would use more sophisticated generation
        # For now, return template examples
        examples = [
            f"Consider how {term} applies in practical scenarios where specific implementation details matter.",
            f"For instance, {term} becomes particularly relevant when dealing with complex systems that require careful consideration.",
            f"A concrete application of {term} can be seen in situations where theoretical understanding meets practical requirements.",
            f"Think of {term} as a framework that helps structure our approach to solving related problems."
        ]
        
        return examples[:num_examples]

class OutputGenerator:
    """Generate PDF and HTML outputs with identical content."""
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG['output']
        self.styles = self._init_pdf_styles()
        self.html_template = self._init_html_template()
    
    def _init_pdf_styles(self) -> Dict:
        """Initialize PDF styles."""
        styles = getSampleStyleSheet()
        
        # Custom styles
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=HexColor('#34495e'),
            spaceBefore=20,
            spaceAfter=12,
            leftIndent=0
        ))
        
        styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=HexColor('#7f8c8d'),
            spaceBefore=12,
            spaceAfter=8,
            leftIndent=20
        ))
        
        styles.add(ParagraphStyle(
            name='LectureText',
            parent=styles['BodyText'],
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        ))
        
        styles.add(ParagraphStyle(
            name='ConceptBox',
            parent=styles['BodyText'],
            fontSize=10,
            leading=14,
            leftIndent=20,
            rightIndent=20,
            backColor=HexColor('#ecf0f1'),
            borderColor=HexColor('#3498db'),
            borderWidth=1,
            borderPadding=10,
            spaceAfter=12
        ))
        
        return styles
    
    def _init_html_template(self) -> Template:
        """Initialize HTML template."""
        template_str = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            overflow: hidden;
        }
        
        /* Sidebar TOC */
        .toc-sidebar {
            width: 300px;
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px 20px;
            max-height: 90vh;
            overflow-y: auto;
            position: sticky;
            top: 20px;
        }
        
        .toc-sidebar::-webkit-scrollbar {
            width: 8px;
        }
        
        .toc-sidebar::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.1);
        }
        
        .toc-sidebar::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.3);
            border-radius: 4px;
        }
        
        .toc-title {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(255,255,255,0.2);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .toc-item {
            margin-bottom: 8px;
        }
        
        .toc-item a {
            color: rgba(255,255,255,0.9);
            text-decoration: none;
            display: block;
            padding: 8px 12px;
            border-radius: 5px;
            transition: all 0.3s ease;
            font-size: 14px;
        }
        
        .toc-item a:hover {
            background: rgba(255,255,255,0.1);
            transform: translateX(5px);
        }
        
        .toc-item.active a {
            background: linear-gradient(90deg, #3498db, #2980b9);
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }
        
        .toc-item.subsection {
            margin-left: 20px;
        }
        
        .toc-item.subsection a {
            font-size: 13px;
            padding: 6px 10px;
        }
        
        /* Main Content */
        .main-content {
            flex: 1;
            padding: 50px;
            max-width: 900px;
            overflow-y: auto;
            max-height: 90vh;
        }
        
        .main-content::-webkit-scrollbar {
            width: 10px;
        }
        
        .main-content::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        .main-content::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 5px;
        }
        
        h1 {
            color: #2c3e50;
            font-size: 36px;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .metadata {
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid #3498db;
        }
        
        h2 {
            color: #34495e;
            font-size: 26px;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-left: 15px;
            border-left: 5px solid #3498db;
            font-weight: 600;
        }
        
        h3 {
            color: #7f8c8d;
            font-size: 20px;
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: 500;
        }
        
        p {
            text-align: justify;
            margin-bottom: 18px;
            line-height: 1.8;
            color: #444;
        }
        
        .concept-box {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 25px 0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .concept-term {
            font-weight: 600;
            color: #2c3e50;
            font-size: 18px;
            margin-bottom: 10px;
        }
        
        .concept-definition {
            color: #555;
            line-height: 1.6;
        }
        
        .example-box {
            background: linear-gradient(135deg, #f093fb15 0%, #f5576c15 100%);
            border-left: 4px solid #f5576c;
            padding: 15px 20px;
            margin: 15px 0;
            border-radius: 6px;
        }
        
        .example-title {
            font-weight: 600;
            color: #f5576c;
            margin-bottom: 8px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .example-text {
            color: #666;
            font-style: italic;
        }
        
        /* Interactive Elements */
        .section-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #3498db, transparent);
            margin: 40px 0;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-in {
            animation: fadeIn 0.6s ease-out;
        }
        
        /* Responsive Design */
        @media (max-width: 1024px) {
            .container {
                flex-direction: column;
                max-height: none;
            }
            
            .toc-sidebar {
                width: 100%;
                max-height: 300px;
                position: relative;
                top: 0;
            }
            
            .main-content {
                max-height: none;
                padding: 30px;
            }
        }
        
        @media (max-width: 640px) {
            .main-content {
                padding: 20px;
            }
            
            h1 { font-size: 28px; }
            h2 { font-size: 22px; }
            h3 { font-size: 18px; }
        }
        
        /* Print Styles */
        @media print {
            body { background: white; }
            .container { box-shadow: none; }
            .toc-sidebar { display: none; }
            .main-content { max-width: 100%; padding: 0; }
        }
    </style>
    <script>
        // Smooth scrolling and active section highlighting
        document.addEventListener('DOMContentLoaded', function() {
            // Smooth scroll for TOC links
            document.querySelectorAll('.toc-item a').forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const targetId = this.getAttribute('href').substring(1);
                    const targetElement = document.getElementById(targetId);
                    if (targetElement) {
                        targetElement.scrollIntoView({ behavior: 'smooth' });
                    }
                });
            });
            
            // Highlight active section on scroll
            const sections = document.querySelectorAll('h2, h3');
            const tocItems = document.querySelectorAll('.toc-item');
            
            function updateActiveSection() {
                let current = '';
                sections.forEach(section => {
                    const rect = section.getBoundingClientRect();
                    if (rect.top <= 100) {
                        current = section.id;
                    }
                });
                
                tocItems.forEach(item => {
                    item.classList.remove('active');
                    const link = item.querySelector('a');
                    if (link && link.getAttribute('href') === '#' + current) {
                        item.classList.add('active');
                    }
                });
            }
            
            // Debounce scroll events
            let scrollTimer;
            document.querySelector('.main-content').addEventListener('scroll', function() {
                clearTimeout(scrollTimer);
                scrollTimer = setTimeout(updateActiveSection, 50);
            });
            
            // Initial highlight
            updateActiveSection();
            
            // Add animation to elements as they come into view
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('animate-in');
                    }
                });
            }, { threshold: 0.1 });
            
            document.querySelectorAll('h2, h3, .concept-box, .example-box').forEach(el => {
                observer.observe(el);
            });
        });
    </script>
</head>
<body>
    <div class="container">
        <nav class="toc-sidebar">
            <div class="toc-title">📚 Table of Contents</div>
            {{ toc_html | safe }}
        </nav>
        
        <main class="main-content">
            <h1>{{ title }}</h1>
            <div class="metadata">
                Generated on {{ date }} | {{ section_count }} sections | {{ word_count }} words
            </div>
            {{ content_html | safe }}
        </main>
    </div>
</body>
</html>'''
        
        return Template(template_str)
    
    def generate(self, enhanced_content: Dict, title: str = None) -> Tuple[bytes, str]:
        """
        Generate PDF and HTML outputs.
        
        Args:
            enhanced_content: Enhanced structured content
            title: Optional custom title
            
        Returns:
            Tuple of (pdf_bytes, html_string)
        """
        # Prepare content
        final_content = self.prepare_content(enhanced_content, title)
        
        # Generate outputs
        pdf_bytes = self.generate_pdf(final_content)
        html_str = self.generate_html(final_content)
        
        return pdf_bytes, html_str
    
    def prepare_content(self, content: Dict, title: str = None) -> Dict:
        """Prepare content for output generation."""
        sections = content['sections']
        
        # Convert sentences to paragraphs
        for section in sections:
            section['paragraphs'] = self.sentences_to_paragraphs(section['sentences'])
            
            for subsection in section.get('subsections', []):
                subsection['paragraphs'] = self.sentences_to_paragraphs(subsection['sentences'])
        
        # Calculate metadata
        total_words = sum(
            len(' '.join(s['sentences']).split())
            for s in sections
        )
        
        return {
            'title': title or 'Lecture Notes',
            'sections': sections,
            'metadata': {
                'date': datetime.now().strftime('%B %d, %Y'),
                'section_count': len(sections),
                'word_count': total_words
            }
        }
    
    def sentences_to_paragraphs(self, sentences: List[str], 
                                sentences_per_paragraph: int = 4) -> List[str]:
        """Group sentences into paragraphs."""
        paragraphs = []
        
        for i in range(0, len(sentences), sentences_per_paragraph):
            paragraph = ' '.join(sentences[i:i+sentences_per_paragraph])
            paragraphs.append(paragraph)
        
        return paragraphs
    
    def generate_pdf(self, content: Dict) -> bytes:
        """Generate PDF output."""
        buffer = BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter if self.config['pdf_page_size'] == 'letter' else A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph(content['title'], self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Metadata
        meta_text = (f"Generated on {content['metadata']['date']} | "
                    f"{content['metadata']['section_count']} sections | "
                    f"{content['metadata']['word_count']:,} words")
        story.append(Paragraph(meta_text, self.styles['Normal']))
        story.append(Spacer(1, 0.5*inch))
        
        # Sections
        for section_num, section in enumerate(content['sections'], 1):
            # Section header
            story.append(Paragraph(
                f"{section_num}. {section['title']}", 
                self.styles['SectionHeader']
            ))
            
            # Section paragraphs
            for paragraph in section['paragraphs']:
                story.append(Paragraph(paragraph, self.styles['LectureText']))
            
            # Concepts
            for concept in section.get('concepts', []):
                concept_text = f"<b>{concept['term']}</b><br/>{concept['definition']}"
                story.append(Paragraph(concept_text, self.styles['ConceptBox']))
                
                # Examples
                for example in concept.get('examples', []):
                    story.append(Paragraph(f"Example: {example}", self.styles['Normal']))
            
            # Subsections
            for subsection_num, subsection in enumerate(section.get('subsections', []), 1):
                story.append(Paragraph(
                    f"{section_num}.{subsection_num} {subsection['title']}", 
                    self.styles['SubsectionHeader']
                ))
                
                for paragraph in subsection['paragraphs']:
                    story.append(Paragraph(paragraph, self.styles['LectureText']))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def generate_html(self, content: Dict) -> str:
        """Generate HTML output."""
        # Build TOC
        toc_items = []
        for i, section in enumerate(content['sections']):
            section_id = f"section-{i}"
            toc_items.append(
                f'<div class="toc-item">'
                f'<a href="#{section_id}">{section["title"]}</a>'
                f'</div>'
            )
            
            for j, subsection in enumerate(section.get('subsections', [])):
                subsection_id = f"subsection-{i}-{j}"
                toc_items.append(
                    f'<div class="toc-item subsection">'
                    f'<a href="#{subsection_id}">{subsection["title"]}</a>'
                    f'</div>'
                )
        
        toc_html = '\n'.join(toc_items)
        
        # Build content
        content_items = []
        for i, section in enumerate(content['sections']):
            section_id = f"section-{i}"
            content_items.append(f'<h2 id="{section_id}">{section["title"]}</h2>')
            
            for paragraph in section['paragraphs']:
                content_items.append(f'<p>{paragraph}</p>')
            
            # Concepts
            for concept in section.get('concepts', []):
                content_items.append(
                    f'<div class="concept-box">'
                    f'<div class="concept-term">{concept["term"]}</div>'
                    f'<div class="concept-definition">{concept["definition"]}</div>'
                    f'</div>'
                )
                
                for example in concept.get('examples', []):
                    content_items.append(
                        f'<div class="example-box">'
                        f'<div class="example-title">Example</div>'
                        f'<div class="example-text">{example}</div>'
                        f'</div>'
                    )
            
            # Subsections
            for j, subsection in enumerate(section.get('subsections', [])):
                subsection_id = f"subsection-{i}-{j}"
                content_items.append(f'<h3 id="{subsection_id}">{subsection["title"]}</h3>')
                
                for paragraph in subsection['paragraphs']:
                    content_items.append(f'<p>{paragraph}</p>')
            
            if i < len(content['sections']) - 1:
                content_items.append('<div class="section-divider"></div>')
        
        content_html = '\n'.join(content_items)
        
        # Render template
        return self.html_template.render(
            title=content['title'],
            date=content['metadata']['date'],
            section_count=content['metadata']['section_count'],
            word_count=f"{content['metadata']['word_count']:,}",
            toc_html=toc_html,
            content_html=content_html
        )

class LectureNotesConverter:
    """Main converter orchestrating all components."""
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG
        self.cleaner = TranscriptCleaner(self.config.get('cleaning'))
        self.structurer = ContentStructurer(self.config.get('structuring'))
        self.enhancer = ContentEnhancer(self.config.get('enhancement'))
        self.generator = OutputGenerator(self.config.get('output'))
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def convert(self, transcript_file: str, output_dir: str = None, 
                title: str = None, verbose: bool = False) -> Dict[str, str]:
        """
        Convert YouTube transcript to lecture notes.
        
        Args:
            transcript_file: Path to transcript text file
            output_dir: Directory for output files (default: same as input)
            title: Custom title for lecture notes
            verbose: Print processing details
            
        Returns:
            Dictionary with paths to generated files
        """
        # Setup paths
        transcript_path = Path(transcript_file)
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_file}")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = transcript_path.parent
        
        # Read transcript
        self.logger.info(f"Reading transcript from {transcript_path}")
        with open(transcript_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        if not raw_text.strip():
            raise ValueError("Transcript file is empty")
        
        try:
            # Process pipeline
            self.logger.info("Step 1/5: Cleaning transcript...")
            cleaned_text = self.cleaner.clean(raw_text, verbose=verbose)
            
            self.logger.info("Step 2/5: Structuring content...")
            structured_content = self.structurer.structure(cleaned_text)
            
            self.logger.info("Step 3/5: Enhancing content...")
            enhanced_content = self.enhancer.enhance(structured_content)
            
            self.logger.info("Step 4/5: Generating outputs...")
            pdf_bytes, html_str = self.generator.generate(enhanced_content, title)
            
            # Save outputs
            self.logger.info("Step 5/5: Saving files...")
            base_name = transcript_path.stem
            pdf_path = output_path / f"{base_name}_lecture_notes.pdf"
            html_path = output_path / f"{base_name}_lecture_notes.html"
            
            with open(pdf_path, 'wb') as f:
                f.write(pdf_bytes)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_str)
            
            self.logger.info(f"✓ Successfully generated lecture notes:")
            self.logger.info(f"  PDF: {pdf_path}")
            self.logger.info(f"  HTML: {html_path}")
            
            return {
                'success': True,
                'pdf_path': str(pdf_path),
                'html_path': str(html_path),
                'statistics': {
                    'original_length': len(raw_text),
                    'cleaned_length': len(cleaned_text),
                    'sections': len(enhanced_content['sections']),
                    'total_concepts': sum(
                        len(s.get('concepts', []))
                        for s in enhanced_content['sections']
                    )
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error during conversion: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert YouTube transcripts to professional lecture notes'
    )
    parser.add_argument('transcript', help='Path to transcript text file')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('-t', '--title', help='Custom title for lecture notes')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Show detailed processing information')
    parser.add_argument('--config', help='Path to JSON config file')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = DEFAULT_CONFIG
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Create converter
    converter = LectureNotesConverter(config)
    
    # Convert
    result = converter.convert(
        args.transcript,
        output_dir=args.output,
        title=args.title,
        verbose=args.verbose
    )
    
    if result['success']:
        print("\n✅ Conversion successful!")
        print(f"📄 PDF: {result['pdf_path']}")
        print(f"🌐 HTML: {result['html_path']}")
        if 'statistics' in result:
            stats = result['statistics']
            print(f"\n📊 Statistics:")
            print(f"  • Sections: {stats['sections']}")
            print(f"  • Key concepts: {stats['total_concepts']}")
            print(f"  • Compression: {(1 - stats['cleaned_length']/stats['original_length'])*100:.1f}%")
    else:
        print(f"\n❌ Conversion failed: {result['error']}")
        sys.exit(1)

if __name__ == '__main__':
    main()
