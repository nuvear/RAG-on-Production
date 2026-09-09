# Directed, typed graph edges, each with its own provenance and trust state.
EDGES = [
 {'id':'E01','head':'Team Orion','relation':'HAS_CERT','tail':'Safety101','status':'current','verified':True},
 {'id':'E02','head':'Safety101','relation':'QUALIFIES_FOR','tail':'Projector X','status':'current','verified':True},
 {'id':'E03','head':'Team Orion','relation':'HAS_CERT','tail':'Advanced202','status':'revoked','verified':True},
 {'id':'E04','head':'Advanced202','relation':'QUALIFIES_FOR','tail':'Projector Y','status':'current','verified':True},
 {'id':'E05','head':'Team Orion','relation':'HAS_CERT','tail':'Advanced202','status':'current','verified':False}]
DEVICES = [
 {'id':'D-X','entity':'Projector X','text':'Projector X is a portable classroom projection device.'},
 {'id':'D-Y','entity':'Projector Y','text':'Projector Y is a high-resolution classroom projection device.'}]

def eligible_paths(team, edges, max_hops=2):
    if max_hops not in (1,2): raise ValueError('This exercise supports one or two hops.')
    trusted = [e for e in edges if e['status']=='current' and e['verified'] is True]
    first = [e for e in trusted if e['head']==team and e['relation']=='HAS_CERT']
    if max_hops == 1: return []  # One hop reaches a certificate, not a device.
    return [[a,b] for a in first for b in trusted
            if b['head']==a['tail'] and b['relation']=='QUALIFIES_FOR']

def graph_candidates(query):
    vectors = embed([d['text'] for d in DEVICES] + [query])
    return sorted([{**d,'score':sum(a*b for a,b in zip(v,vectors[-1]))}
                   for d,v in zip(DEVICES,vectors[:-1])], key=lambda d:d['score'], reverse=True)

GRAPH_QUESTION = 'Which classroom projector may Team Orion use under the equipment access policy?'
one_hop = eligible_paths('Team Orion', EDGES, max_hops=1)
two_hops = eligible_paths('Team Orion', EDGES, max_hops=2)
candidates = graph_candidates(GRAPH_QUESTION)
eligible = {path[-1]['tail'] for path in two_hops}
hybrid = [d for d in candidates if d['entity'] in eligible]
assert eligible == {'Projector X'} and not one_hop
assert all(e['status']=='current' and e['verified'] for path in two_hops for e in path)
graph_evidence = [{'id': e['id'], 'text': f"{e['head']} --{e['relation']}--> {e['tail']}"}
                  for path in two_hops for e in path]
graph_evidence += [{'id': p['id'], 'text': p['text']} for p in POLICIES if p['id']=='P03']
graph_evidence += [{'id': d['id'], 'text': d['text']} for d in hybrid]
graph_answer = grounded(GRAPH_QUESTION, graph_evidence)
observations['lab8'] = {'one_hop': one_hop, 'two_hops': two_hops, 'vector_candidates': candidates,
                        'hybrid_ids': [d['id'] for d in hybrid], 'answer': graph_answer}
print(json.dumps(observations['lab8'], indent=2))
