def cleanup_resources():
    results = []
    if ledger['store_id']:
        result = api('DELETE', '/vector_stores/' + ledger['store_id'])
        if result.get('deleted'):
            ledger['store_id'] = None; ledger['attached'] = []; save_ledger()
        results.append({'kind':'vector_store','deleted':bool(result.get('deleted'))})
    for pid, fid in list(ledger['files'].items()):
        result = api('DELETE', '/files/' + fid)
        if result.get('deleted'):
            del ledger['files'][pid]; save_ledger()
        results.append({'kind':'file','source_id':pid,'deleted':bool(result.get('deleted'))})
    return results

cleanup_result = cleanup_resources()
observations['cleanup'] = {'results': cleanup_result,
    'complete': ledger['store_id'] is None and not ledger['files']}
print(observations['cleanup'])
print('Keep this output. Vector-store expiry alone does not remove the original Files objects.')
