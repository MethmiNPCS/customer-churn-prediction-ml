import json
import re

with open('decision_tree_model.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Split the content by the step markers
# This regex keeps the delimiter as part of the split so we can zipper it back
parts = re.split(r'(# ── STEP(?:.*?)─+)', content)

cells = []

# First part is the header / imports
if parts[0].strip():
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in parts[0].split('\n')[:-1]]
    })

# The rest are the steps
for i in range(1, len(parts), 2):
    header = parts[i]
    body = parts[i+1]
    
    # Create a markdown cell for the header
    header_clean = header.replace('# ──', '').replace('─', '').strip()
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"### {header_clean}\n"]
    })
    
    # Create a code cell for the body
    source_lines = [line + '\n' for line in body.split('\n')]
    # Remove last newline if it's empty
    if source_lines and source_lines[-1] == '\n':
        source_lines = source_lines[:-1]
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    })

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('decision_tree_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Successfully generated decision_tree_model.ipynb")
