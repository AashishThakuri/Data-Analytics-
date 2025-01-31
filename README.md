# DataMaster Analytics

A powerful data analysis application that supports numeric, image, and text analysis with a modern and intuitive user interface.

## Features

### Numeric Data Analysis
- Upload and analyze CSV/Excel files
- Automatic statistical summaries
- Interactive visualizations (scatter plots, box plots, histograms, line plots)
- Correlation analysis
- Missing value detection

### Image Analysis
- Support for common image formats (PNG, JPG, JPEG)
- Basic image properties analysis
- Edge detection
- Color analysis
- Object detection using state-of-the-art models

### Text Analysis
- Basic text statistics
- Named Entity Recognition (NER)
- Sentiment Analysis
- Part of Speech (POS) analysis
- Word frequency analysis

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

Run the application using Streamlit:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## Requirements

- Python 3.8+
- See requirements.txt for all Python package dependencies

## License

MIT License
