"""
Export parsed documents to JSON and CSV outputs.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_to_json(documents: List[Any], output_dir: str = "data/output/json"):
    """
    Save documents to individual JSON files.

    Args:
        documents: List of document objects
        output_dir: Directory to save JSON files
    """
    output_path = (Path(__file__).resolve().parent / output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for doc in documents:
        # Convert Pydantic model to dict
        if hasattr(doc, 'dict'):
            doc_dict = doc.dict()
        else:
            doc_dict = doc

        # Create filename from document type and ID
        doc_id = doc_dict.get('document_id', 'unknown')
        doc_type = doc_dict.get('document_type', 'unknown')
        filename = f"{doc_type}_{doc_id[:8]}.json"

        file_path = output_path / filename

        with open(file_path, 'w') as f:
            json.dump(doc_dict, f, indent=2, default=str)

        logger.info(f"Saved: {filename}")

    logger.info(f"Saved {len(documents)} documents to {output_dir}")


def save_to_csv(documents: List[Any], output_file: str = "data/output/master_data.csv"):
    """
    Save documents to a single CSV file.

    Args:
        documents: List of document objects
        output_file: Path to output CSV file
    """
    output_path = (Path(__file__).resolve().parent / output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert documents to list of dicts
    records = []
    for doc in documents:
        if hasattr(doc, 'dict'):
            doc_dict = doc.dict()
        else:
            doc_dict = doc

        # Flatten nested structures for CSV
        flattened = flatten_dict(doc_dict)
        records.append(flattened)

    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved {len(documents)} documents to {output_file}")
    return df


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to JSON strings for CSV compatibility
            items.append((new_key, json.dumps(v)))
        else:
            items.append((new_key, v))

    return dict(items)
