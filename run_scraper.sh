#!/bin/bash
# Wrapper to run the scraper using the virtual environment

# Ensure venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    venv/bin/pip install requests pandas html5lib lxml beautifulsoup4
fi

# Run the scraper
# Pass all arguments to the python script
venv/bin/python scrape_10_year_history.py "$@"
