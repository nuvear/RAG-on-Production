"""Build all three guided notebooks and full Python notes from shared code.

No credentials, package installation, or API calls are used by this builder.
"""
from pathlib import Path
import json, re, ast

ROOT = Path(__file__).resolve().parents[1]
code = {p.stem: p.read_text() for p in sorted((ROOT/'code').glob('*.py'))}
for name, source in code.items(): ast.parse(source, filename=name)
for language, note_title in [('en','Detailed Python notes and executable reference'),
                             ('ja','Python解説と実装リファレンス'),
                             ('zh-CN','Python详解与完整参考代码')]:
    folder = ROOT/'Student'/language
    guide = (folder/'Hands-On-Workbook.md').read_text()
    parts = re.split(r'<!-- CELL:([a-z0-9_]+) -->', guide)
    cells = [{'cell_type':'markdown','metadata':{},'id':'intro','source':parts[0].strip()+'\n'}]
    notes = ['# '+note_title, parts[0].strip()]
    used = []
    for i in range(1,len(parts),2):
        name, prose = parts[i], parts[i+1]
        tail = None
        if '<!-- END -->' in prose: prose, tail = prose.split('<!-- END -->',1)
        assert name in code, name
        used.append(name)
        cells.extend([
            {'cell_type':'markdown','metadata':{},'id':'notes-'+name,'source':prose.strip()+'\n'},
            {'cell_type':'code','metadata':{},'id':'code-'+name,'source':code[name],
             'execution_count':None,'outputs':[]}])
        notes += [prose.strip(), '```python\n'+code[name].rstrip()+'\n```']
        if tail:
            cells.append({'cell_type':'markdown','metadata':{},'id':'recovery','source':tail.strip()+'\n'})
            notes.append(tail.strip())
    assert used == list(code), f'Missing or reordered cells: {language}'
    for cell in cells:
        if cell['cell_type']=='markdown':
            cell['source'] = cell['source'].replace('(Python-Code-Notes.md)',
              f'(https://github.com/nuvear/RAG-on-Production/blob/main/Session-2/Student/{language}/Python-Code-Notes.md)')
            cell['source'] = cell['source'].replace('(../../REFERENCES.md)',
              '(https://github.com/nuvear/RAG-on-Production/blob/main/Session-2/REFERENCES.md)')
    nb = {'nbformat':4,'nbformat_minor':5,'metadata':{
        'colab':{'name':f'RAG_Session2_{language}.ipynb','provenance':[]},
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.11'}},'cells':cells}
    (folder/'RAG_Session2.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    (folder/'Python-Code-Notes.md').write_text('\n\n'.join(notes)+'\n')
    print(language, len(cells),'cells;',len(code),'shared executable cells')
