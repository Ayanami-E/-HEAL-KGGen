import json
import os
from typing import List, Dict, Any

def load_medqa_data(file_path):
    """Load medical QA data from a JSON or JSONL file"""
    data = []
    
    # Get the file extension
    _, extension = os.path.splitext(file_path)
    
    try:
        if extension.lower() == '.jsonl':
            # JSONL format: each line is a JSON object
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        data.append(sample)
                    except json.JSONDecodeError:
                        print(f"Error parsing line: {line}")
        else:
            # Standard JSON format: the whole file is a JSON array or object
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
                # Check if the loaded data is a list
                if isinstance(json_data, list):
                    data = json_data
                # If it's a dictionary, might need to extract a specific key
                elif isinstance(json_data, dict) and "questions" in json_data:
                    data = json_data["questions"]
                elif isinstance(json_data, dict) and "data" in json_data:
                    data = json_data["data"]
                else:
                    # If it's another structure, more specific handling may be needed
                    print(f"Warning: Unexpected JSON structure in {file_path}")
                    # Add the dictionary as a single sample
                    if isinstance(json_data, dict):
                        data = [json_data]
        
        print(f"Loaded {len(data)} samples from {file_path}")
    except Exception as e:
        print(f"Error while loading data: {e}")
    
    return data

def show_sample(data, index=0):
    """Display detailed information of a sample entry"""
    if data and len(data) > index:
        print(f"\nSample data entry (index {index}):")
        print(json.dumps(data[index], indent=2))
