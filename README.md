# ESG-Asset-Analyzer

A tool for analyzing and matching ESG assets to companies using NLP and LLMs.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)


## Installation
1. Clone the respository
```bash
   git clone 
```

3. Navigate to project directory

4. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
1. Run main script:
 ```bash
   python main.py
```

## Features
- This module analyzes all assets in asset_data.csv and determines relevant matches to companies in isin_companies.csv
- Uses fuzzy matching to generate shortlist of potentially relevant companies in isin_companies.csv
- Passes the shortlist to an LLM to get a final match
