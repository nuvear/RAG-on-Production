"""Offline tests of released cells. No key or API access needed."""
from pathlib import Path
import ast, json, copy, re, hashlib
ROOT = Path(__file__).resolve().parents[1]
code = {p.stem:p.read_text() for p in sorted((ROOT/'code').glob('*.py'))}
scope = {'json':json,'re':re,'time':__import__('time'),'MODEL':'offline'}
for name, source in code.items():
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=functions,type_ignores=[]),name,'exec'),scope)
    # Only load literal teaching constants; never run initialization or API requests.
    for n in tree.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {
            'POLICIES','SLOTS','TOOLS','EDGES','DEVICES','ANSWER_SCHEMA'}:
            scope[n.targets[0].id]=ast.literal_eval(n.value)

def rejects(fn,*args):
    try: fn(*args)
    except (ValueError,TypeError): return
    raise AssertionError('Invalid input accepted')

dispatch=scope['dispatch']; paths=scope['eligible_paths']; check=scope['check_answer']
assert all(p['status']=='current' for p in dispatch('search_policy',{'query':'booking hours'}))
assert dispatch('get_slots',{'room':'LabA','day':'Tuesday'})[0]['id']=='SLOTS-LabA'
for name,args in [('delete_booking',{}),('get_slots',{'room':'LabC','day':'Tuesday'}),
                  ('get_slots',{'room':'LabA','day':'Friday'}),('search_policy',{'query':'x','extra':True}),
                  ('search_policy',{'query':'x'*501})]: rejects(dispatch,name,args)
edges=scope['EDGES']; baseline=copy.deepcopy(edges)
assert paths('Team Orion',edges,1)==[]
assert [[e['id'] for e in p] for p in paths('Team Orion',edges,2)]==[['E01','E02']]
changed=copy.deepcopy(edges);changed[0]['status']='revoked';assert paths('Team Orion',changed)==[]
assert paths('Unknown Team',edges)==[] and edges==baseline
rejects(paths,'Team Orion',edges,3)
valid={'answer':'two hours','abstain':False,'sources':['P01']}
assert check(valid,['P01'])==valid
rejects(check,{'answer':'four','abstain':False,'sources':['FAKE']},['P01'])
rejects(check,{'answer':'unknown','abstain':True,'sources':['P01']},['P01'])
assert check({'answer':'fabrication','abstain':True,'sources':[]},[])['answer'].startswith('I do not know')
# Explicit limitation: an allowed citation does not validate semantic entailment.
assert check({'answer':'four hours','abstain':False,'sources':['P01']},['P01'])['answer']=='four hours'

def fake_tool(name='get_slots',arguments='{"room":"LabA","day":"Tuesday"}'):
    return lambda _: {'status':'completed','output':[{'type':'function_call','name':name,
                                                     'arguments':arguments,'call_id':'offline'}]}
run=scope['run_agent']
assert run('test',max_tool_calls=0,send=fake_tool())['status']=='tool_budget_exhausted'
assert run('test',max_rounds=1,send=fake_tool())['status']=='round_budget_exhausted'
assert run('test',send=fake_tool('delete_booking','{}'))['status']=='tool_rejected'
assert run('test',send=fake_tool(arguments='not JSON'))['status']=='tool_rejected'
scope['api']=lambda *args,**kwargs: {'status':'incomplete','output':[]}
assert scope['grounded']('test',[])['guardrail_status']=='incomplete'
scope['api']=lambda *args,**kwargs: {'status':'completed','output':[{'type':'message','content':[
    {'type':'output_text','text':'{"answer":"unknown","abstain":true,"sources":["P01"]}'}]}]}
assert scope['grounded']('test',[{'id':'P01','text':'text'}])['guardrail_status']=='output_rejected'

# Cleanup touches the ledger's resources only and remains retryable after a partial failure.
deleted=[]
scope['ledger']={'store_id':'vs_own','files':{'P01':'file_own'},'attached':['P01']}
scope['save_ledger']=lambda:None
scope['api']=lambda method,path: deleted.append((method,path)) or {'deleted':True}
assert all(r['deleted'] for r in scope['cleanup_resources']())
assert deleted==[('DELETE','/vector_stores/vs_own'),('DELETE','/files/file_own')]
assert scope['cleanup_resources']()==[]
scope['ledger']={'store_id':'vs_partial','files':{'P01':'file_partial'},'attached':['P01']}
def fail_file_delete(method,path):
    if path.startswith('/files/'): raise RuntimeError('Synthetic network interruption')
    return {'deleted':True}
scope['api']=fail_file_delete
try:
    scope['cleanup_resources']()
    raise AssertionError('Expected partial cleanup interruption')
except RuntimeError:
    pass
assert scope['ledger']['store_id'] is None and scope['ledger']['files']=={'P01':'file_partial'}
resumed=[]
scope['api']=lambda method,path: resumed.append(path) or {'deleted':True}
scope['cleanup_resources']()
assert resumed==['/files/file_partial'] and not scope['ledger']['files']

for language in ('en','ja','zh-CN'):
    folder=ROOT/'Student'/language
    nb=json.loads((folder/'RAG_Session2.ipynb').read_text())
    actual=[c['source'] for c in nb['cells'] if c['cell_type']=='code']
    assert actual==list(code.values()),language
    assert len({c['id'] for c in nb['cells']})==len(nb['cells'])
    for c in nb['cells']:
        if c['cell_type']=='code': assert c['outputs']==[] and c['execution_count'] is None
    guide=(folder/'Hands-On-Workbook.md').read_text()
    for prefix,limit in [(0,3),(5,4),(6,4),(7,4),(8,4),(9,3)]:
        for step in range(1,limit+1): assert f'{prefix}.{step}' in guide
    for link in re.findall(r'\]\(([^)]+)\)',guide):
        if not link.startswith('http'): assert (folder/link).exists(),link
    print(language,'shared-code, clean-output, all 22 steps and local-link checks passed')
for record in json.loads((ROOT/'data/image_manifest.json').read_text()):
    assert hashlib.sha256((ROOT/'data'/record['filename']).read_bytes()).hexdigest()==record['sha256']
print('Agent dispatch/budgets, graph eligibility, answer contracts, cleanup and asset checks passed. No API calls.')
