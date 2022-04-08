from pathlib import Path
import json

import numpy as np
from PIL import Image


FORMAT_CHARS = [
    '<b>',
    '</b>',
    '<i>',
    '</i>',
    '<sup>',
    '</sup>',
    '<sub>',
    '</sub>',
]


def clean_text(cell):
    """
    Removes HTML tags from text
    """
    text = ''.join(cell["tokens"]).strip()
    for fmt in FORMAT_CHARS:
        text = text.replace(fmt, '')
    return text

def html_iterator(y):
    """
    Converts HTML structure tokens to an iterator of
    cell properties
    """
    html = y['html']['structure']['tokens']
    
    row = 0
    column = 0
    is_head = False
    is_body = False
    row_start=None
    row_end=None
    col_start=0
    col_end=0
    raw = ''
    
    for t in html:
        raw += t
        if '<td' in t:
            col_start = column
        
        elif t == '</td>':
            col_end = column
            row_end = row
            column += 1
            yield {
                'col_start':col_start,
                'col_end':col_end,
                'row_start':row_start,
                'row_end':row_end,
                'is_head':is_head,
                'is_body':is_body,
                'raw': raw,
            }
            raw = ''
            
        elif t == '<thead>':
            row_start = row
            is_head = True
            
        elif t == '</thead>':
            is_head = False
            
        elif t == '<tr>':
            row_start = row
            
        elif t == '</tr>':
            column = 0
            row += 1
            
        elif "colspan" in t:
            # extract the int from the string and increment the row counter
            column += int(t.split('"')[1]) - 1
        
        elif t == '<tbody>':
            is_body = True
            
        elif t == '</tbody>':
            is_body = False
            
        elif t in ['>']:
            pass
        
        else:
            raise ValueError(t)

def json_2_labels(data):
    struct = html_iterator(data)
    
    cells = []
    for cell in data['html']['cells']:
        cell_data = next(struct)
        if 'bbox' in cell:
            cell_data['text'] = clean_text(cell),
            cell_data['bbox'] = cell["bbox"]
            cells.append(cell_data)
    return cells

class PubTabNet:
    def __init__(self, images_dir, jsonl_path=None):
        images_dir = Path(images_dir)
        
        
        self.filenames = list(images_dir.glob("*.png"))
        if jsonl_path is None:
            self.no_target = True
        else:
            self.no_target = False
            jsonl_path = Path(jsonl_path)
            with open(jsonl_path, "r", encoding="utf8") as f:
                self.json_list = list(f)
            self.fname_index_map = {}

            for i, json_str in enumerate(self.json_list):
                result = json.loads(json_str)
                self.fname_index_map[result["filename"]] = i
            
    def __getitem__(self, index):
        fname = self.filenames[index]
        if self.no_target:
            return Image.open(fname), None
        target = json.loads(
            self.json_list[
                self.fname_index_map[
                    fname.parts[-1]
                ]
            ]
        )
        
        image = Image.open(fname)
        labels = json_2_labels(target)
        return image, labels
