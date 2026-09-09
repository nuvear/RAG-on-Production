before = len(api_log)
for name, args in [('delete_booking', {'id': 'B1'}), ('get_slots', {'room': 'LabA', 'day': 'Friday'})]:
    try:
        dispatch(name, args)
        raise AssertionError('Invalid call was accepted.')
    except ValueError:
        print('Expected local rejection:', name)

def scripted_tool_response(body):
    # A deterministic test double, not a live model result.
    return {'status': 'completed', 'output': [{'type': 'function_call', 'name': 'get_slots',
        'arguments': '{"room":"LabA","day":"Tuesday"}', 'call_id': 'test_call'}]}

budget_test = run_agent('Find a slot.', max_tool_calls=0, send=scripted_tool_response)
assert budget_test['status'] == 'tool_budget_exhausted' and budget_test['trace'] == []
assert len(api_log) == before
observations['agent_controls'] = {'forbidden_tool': 'rejected', 'invalid_day': 'rejected',
                                'zero_tool_budget': budget_test['status'], 'api_calls_added': 0}
print(observations['agent_controls'])
