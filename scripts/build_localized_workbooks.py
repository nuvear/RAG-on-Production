"""Build localized prose around byte-identical, previously tested lab code.

Edit Student/{language}/Hands-On-Workbook-*.md and scripts/localization/*.json.
This builder makes no API calls and never reads credentials.
"""
from pathlib import Path
import copy
import json
import re

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / 'Student/RAG_2h_Student.ipynb').read_text())
source_code = [''.join(c['source']) for c in source['cells'] if c['cell_type'] == 'code']

for language, suffix, notebook_suffix, title in [
    ('ja', 'JA', 'JA', 'Python解説：根拠に基づくRAGの実装'),
    ('zh-CN', 'ZH-CN', 'ZH_CN', 'Python代码说明：基于证据的RAG实现'),
]:
    folder = ROOT / 'Student' / language
    prose = (folder / f'Hands-On-Workbook-{suffix}.md').read_text()
    notes = json.loads((ROOT / 'scripts/localization' / f'{language}.json').read_text())

    def section(name):
        return prose.split(f'<!-- {name}_START -->', 1)[1].split(f'<!-- {name}_END -->', 1)[0].strip()

    replacements = {
        'cell-000': prose.split('<!-- LAB1_START -->', 1)[0].strip(),
        'cell-005': section('LAB1'),
        'cell-013': section('LAB2') + '\n\n' + notes['retrieval_reference'],
        'cell-018': section('BREAK'),
        'cell-019': section('LAB3') + '\n\n' + notes['generation_reference'],
        'cell-023': section('LAB4'),
        'cell-030': section('SUBMISSION'),
        'cell-033': section('NEXT'),
    }
    replacements.update({key: value for key, value in notes.items() if key.startswith('cell-')})
    nb = copy.deepcopy(source)
    code_notes = [f'# {title}\n']
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            assert cell['id'] in replacements, f'Missing translation: {language}/{cell["id"]}'
            value = replacements[cell['id']]
            value = value.replace(f'(Python-Code-Notes-{suffix}.md)',
                f'(https://github.com/nuvear/RAG-on-Production/blob/main/Student/{language}/Python-Code-Notes-{suffix}.md)')
            value = value.replace('(../../REFERENCES.md)',
                '(https://github.com/nuvear/RAG-on-Production/blob/main/REFERENCES.md)')
            cell['source'] = value + '\n'
            if cell['id'] in notes:
                code_notes.append(notes[cell['id']])
            elif cell['id'] in ('cell-005', 'cell-013', 'cell-019', 'cell-023'):
                code_notes.append(value.split('\n', 1)[0])
                for ref_cell, ref in [('cell-013','retrieval_reference'), ('cell-019','generation_reference')]:
                    if cell['id'] == ref_cell: code_notes.append(notes[ref])
        else:
            assert cell['execution_count'] is None and not cell['outputs']
            code_notes.append('```python\n' + ''.join(cell['source']).rstrip() + '\n```')
    nb['cells'].append({'cell_type': 'markdown', 'id': 'localized-troubleshooting',
                        'metadata': {}, 'source': section('TROUBLESHOOTING') + '\n'})
    assert source_code == [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    name = f'RAG_2h_Hands_On_Workbook_{notebook_suffix}.ipynb'
    nb['metadata']['colab']['name'] = name
    (folder / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    (folder / f'Python-Code-Notes-{suffix}.md').write_text('\n\n'.join(code_notes) + '\n')
    print(f'{language}: {len(nb["cells"])} cells; {len(source_code)} executable cells unchanged.')
