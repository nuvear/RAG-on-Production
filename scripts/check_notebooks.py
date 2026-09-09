"""Offline checks of the actual notebook functions. Never makes an API call."""
from pathlib import Path
import ast
import copy
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
notebooks = [ROOT/'Student/RAG_2h_Student.ipynb', ROOT/'Instructor/RAG_2h_Instructor.ipynb']
functions = {}
for path in notebooks:
    nb = json.loads(path.read_text())
    assert nb['nbformat'] == 4
    assert len({cell['id'] for cell in nb['cells']}) == len(nb['cells'])
    for cell in nb['cells']:
        if cell['cell_type'] != 'code': continue
        source = ''.join(cell['source'])
        tree = ast.parse(source)
        assert cell['execution_count'] is None and not cell['outputs'], 'Release must omit run outputs.'
        for node in tree.body:
            if isinstance(node, ast.FunctionDef): functions[node.name] = node
    print(path.name, 'syntax and clean-output checks passed')

scope = {'np':np}
for name in ['chunk_documents','validate_question','validate_output','retrieval_metrics','search']:
    exec(compile(ast.Module(body=[functions[name]],type_ignores=[]),name,'exec'),scope)

def rejects(function, *args, **kwargs):
    try: function(*args, **kwargs)
    except ValueError: return
    raise AssertionError('Expected rejection')

documents = json.loads((ROOT/'Student/library_corpus.json').read_text())
chunk = scope['chunk_documents']
assert len(chunk(documents,20,0)) == 26
assert len(chunk(documents,20,5)) == 30
assert len(chunk(documents,40,8)) == 16
rejects(chunk, documents,20,20)
rejects(chunk, documents,0,0)
assert all(c['doc_id'] and c['status'] for c in chunk(documents))

question = scope['validate_question']
assert question('  renew a book? ') == 'renew a book?'
for bad in ['', '   ', 'x'*501, None, 123]: rejects(question,bad)

validate = scope['validate_output']
hits = [{'doc_id':'D02','text':'Renewal lasts 14 days.'}]
valid = {'answer':'14 days','abstain':False,'citations':[{'source_id':'D02','quote':'14 days'}]}
assert validate(valid,hits) == valid
for bad in [
    {'answer':'14 days','abstain':False,'citations':[]},
    {'answer':'14 days','abstain':False,'citations':[{'source_id':'D999','quote':'14 days'}]},
    {'answer':'99 days','abstain':False,'citations':[{'source_id':'D02','quote':'99 days'}]},
    {'answer':'14 days','abstain':False,'citations':[{'source_id':[],'quote':'14 days'}]},
    {'answer':'14 days','abstain':False,'citations':[{'source_id':'D02','quote':''}]},
    {'answer':'unknown','abstain':True,'citations':[{'source_id':'D02','quote':'14 days'}]},
]: rejects(validate,bad,hits)
# Deliberate limitation: correct reference/quote does not prove answer entailment.
misleading = copy.deepcopy(valid); misleading['answer']='99 days'
assert validate(misleading,hits)['answer'] == '99 days'
assert validate({'answer':'Unsupported invented text','abstain':True,'citations':[]},hits)['answer'] == 'I do not know from the supplied evidence.'

metric=scope['retrieval_metrics']
assert metric(['A','C','B'],['A','B'],2) == {'precision':.5,'recall':.5,'rr':1.}
assert metric(['C','A'],['A'],2) == {'precision':.5,'recall':1.,'rr':.5}
assert metric(['C','D'],['A'],2) == {'precision':0.,'recall':0.,'rr':0.}
rejects(metric,['A','A'],['A'],2)
rejects(metric,['A'],[],1)

# Synthetic vectors for an offline ranking contract test, not real embeddings.
scope.update(chunks=[
    {'doc_id':'ARCH','status':'archived','text':'old'},
    {'doc_id':'A','status':'current','text':'current best'},
    {'doc_id':'A','status':'current','text':'current duplicate'},
    {'doc_id':'B','status':'current','text':'second source'}],
    dense_matrix=np.array([[1.,0.],[.9,.1],[.8,.2],[.7,.3]]),
    embed=lambda texts:np.array([[1.,0.]]))
search=scope['search']
assert [h['doc_id'] for h in search('q',k=2)] == ['A','B']
assert [h['doc_id'] for h in search('q',k=2,current_only=False)] == ['ARCH','A']
print('Chunking, metadata eligibility, deduplication, guardrail boundaries and metric checks passed. No API calls.')
