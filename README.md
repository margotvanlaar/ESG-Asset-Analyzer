# ESG-Asset-Analyzer

A tool for analyzing and matching ESG assets to companies using NLP and LLMs.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)


## Installation
1. Clone the respository
```bash
   git clone https://github.com/margotvanlaar/ESG-Asset-Analyzer.git
```

2. Navigate to project directory
```bash
cd <<project directory>>
```

3. (Optional) Create virtual environment
```bash
python -m venv venv
venv/Scripts/activate
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
1. Create config.json file
Create config.json file in root directory, containing valid openAI API key:
```bash
{
"OPENAI_KEY": "<<insert key>>"
}
```

2. Run main script:
 ```bash
   python main.py
```

## Features
- This module analyzes all assets in asset_data.csv and determines relevant matches to companies in isin_companies.csv
- Uses fuzzy matching to generate shortlist of potentially relevant companies in isin_companies.csv
- Passes the shortlist to an LLM to get a final match
- Results are saved in 'assets_with_matches.csv' in data folder
