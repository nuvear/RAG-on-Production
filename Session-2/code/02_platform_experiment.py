def platform_search(query, current_only=True):
    if platform_state['status'] != 'ready': raise RuntimeError('Run the indexing cell until ready.')
    body = {'query': query, 'max_num_results': 4}
    if current_only: body['filters'] = {'type': 'eq', 'key': 'status', 'value': 'current'}
    result = api('POST', f"/vector_stores/{ledger['store_id']}/search", body)
    return [{'id': row['attributes']['source_id'], 'status': row['attributes']['status'],
             'score': row['score'], 'text': '\n'.join(c['text'] for c in row['content'] if c['type']=='text')}
            for row in result['data']]

PLATFORM_QUESTION = 'Can a student book a makerspace room for four hours per day?'
unfiltered = platform_search(PLATFORM_QUESTION, current_only=False)
filtered = platform_search(PLATFORM_QUESTION, current_only=True)
assert all(hit['status'] == 'current' for hit in filtered)
platform_answer = grounded(PLATFORM_QUESTION, filtered)
observations['lab5'] = {'unfiltered': unfiltered, 'filtered': filtered, 'answer': platform_answer}
print(json.dumps(observations['lab5'], indent=2))
