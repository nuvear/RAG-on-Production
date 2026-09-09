"""Build the guided Colab edition from the workbook prose and existing lab code.

No API calls. Student/RAG_2h_Student.ipynb remains the code source of truth.
"""
from pathlib import Path
import copy
import json

ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / 'Student'
guide = (STUDENT / 'Hands-On-Workbook.md').read_text(encoding='utf-8')
source = json.loads((STUDENT / 'RAG_2h_Student.ipynb').read_text(encoding='utf-8'))
workbook = copy.deepcopy(source)

def section(start, end):
    return guide.split(start, 1)[1].split(end, 1)[0].strip() + '\n'

introductions = {
    'cell-005': section('<!-- LAB1_START -->', '<!-- LAB1_END -->'),
    'cell-013': section('<!-- LAB2_START -->', '<!-- LAB2_END -->'),
    'cell-019': section('<!-- LAB3_START -->', '<!-- LAB3_END -->'),
    'cell-023': section('<!-- LAB4_START -->', '<!-- LAB4_END -->'),
}
for cell in workbook['cells']:
    if cell['id'] in introductions:
        # Retain the prior technical explanations as reference material after
        # the student procedure. No executable cells are changed.
        prior = ''.join(cell['source']).split('\n', 1)[1].strip()
        cell['source'] = introductions[cell['id']] + '\n### Implementation reference\n\n' + prior + '\n'
    elif cell['id'] == 'cell-000':
        cell['source'] = guide.split('<!-- LAB1_START -->', 1)[0].replace(
            'This workbook accompanies the executable Colab notebook.',
            'This is the executable Colab workbook. The code cells follow each lab procedure.')
    elif cell['id'] == 'cell-030':
        cell['source'] = section('<!-- LAB4_END -->', '## Quick troubleshooting reference')

workbook['cells'].append({
    'cell_type':'markdown', 'id':'workbook-troubleshooting', 'metadata':{},
    'source':'## Quick troubleshooting reference\n' + guide.split('## Quick troubleshooting reference',1)[1]
})
for cell in workbook['cells']:
    if cell['cell_type'] == 'markdown':
        cell['source'] = ''.join(cell['source']).replace(
            '(Python-Code-Notes.md)',
            '(https://github.com/nuvear/RAG-on-Production/blob/main/Student/Python-Code-Notes.md)'
        ).replace('(../REFERENCES.md)',
                  '(https://github.com/nuvear/RAG-on-Production/blob/main/REFERENCES.md)')

original_code = [''.join(c['source']) for c in source['cells'] if c['cell_type']=='code']
guided_code = [''.join(c['source']) for c in workbook['cells'] if c['cell_type']=='code']
assert original_code == guided_code, 'Guided edition must preserve the tested code exactly.'
name = 'RAG_2h_Hands_On_Workbook.ipynb'
workbook['metadata']['colab']['name'] = name
(STUDENT / name).write_text(json.dumps(workbook,indent=1),encoding='utf-8')
print(f'Built {name}: {len(workbook["cells"])} cells; all {len(guided_code)} code cells unchanged.')
