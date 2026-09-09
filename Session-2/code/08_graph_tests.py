import copy
revoked = copy.deepcopy(EDGES)
revoked[0]['status'] = 'revoked'
assert eligible_paths('Team Orion', revoked) == []
assert eligible_paths('Unknown Team', EDGES) == []
changed_answer = grounded(GRAPH_QUESTION, [])
observations['graph_controls'] = {'after_revocation': [], 'unknown_team': [], 'no_evidence_answer': changed_answer}
print(json.dumps(observations['graph_controls'], indent=2))
