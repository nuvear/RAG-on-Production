# Original synthetic university makerspace policies; not real institutional advice.
POLICIES = [
    {'id': 'P01', 'status': 'current', 'text': 'Current makerspace booking policy: a student may book '
     'a room for at most two hours per day. An available slot is a proposal, not a confirmed booking.'},
    {'id': 'P02', 'status': 'archived', 'text': 'Archived makerspace booking policy: students could '
     'book a room for at most four hours per day. This policy is no longer valid.'},
    {'id': 'P03', 'status': 'current', 'text': 'Equipment access policy: a team needs a current, '
     'verified certificate that qualifies it for the specific device. Room availability does not '
     'grant equipment permission.'},
    {'id': 'P04', 'status': 'current', 'text': 'Support policy: maintenance questions go to the '
     'makerspace service desk. The service desk does not disclose private student records.'}]

def upload_policy(policy):
    boundary = 'rag2_' + uuid.uuid4().hex
    filename = f"rag2_{ledger['run_id']}_{policy['id']}.txt"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nassistants\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f'Content-Type: text/plain\r\n\r\n{policy["id"]}: {policy["text"]}\r\n'
            f'--{boundary}--\r\n').encode()
    return api('POST', '/files', raw=body, content_type='multipart/form-data; boundary=' + boundary)['id']

def prepare_platform():
    if not ledger['store_id']:
        store = api('POST', '/vector_stores', {'name': 'rag2-classroom-' + ledger['run_id'],
                    'expires_after': {'anchor': 'last_active_at', 'days': 1}})
        ledger['store_id'] = store['id']; save_ledger()
    for policy in POLICIES:
        pid = policy['id']
        if pid not in ledger['files']:
            ledger['files'][pid] = upload_policy(policy); save_ledger()
        if pid not in ledger['attached']:
            api('POST', f"/vector_stores/{ledger['store_id']}/files", {
                'file_id': ledger['files'][pid],
                'attributes': {'source_id': pid, 'status': policy['status']}})
            ledger['attached'].append(pid); save_ledger()
    # One cell waits at most ~60 seconds. Rerun safely to poll again if still indexing.
    for _ in range(12):
        store = api('GET', f"/vector_stores/{ledger['store_id']}")
        counts = store['file_counts']
        if counts.get('failed', 0): raise RuntimeError('Indexing failed; inspect the project vector store.')
        if counts.get('completed') == len(POLICIES):
            return {'status': 'ready', 'file_counts': counts}
        time.sleep(3)
    return {'status': 'indexing_pending', 'file_counts': counts}

platform_state = prepare_platform()
observations['platform_state'] = platform_state
print(platform_state)
print('Download session2_resources.json now as a recovery record. It contains resource IDs, not a key.')
if IN_COLAB:
    from google.colab import files
    files.download(str(LEDGER_PATH))
