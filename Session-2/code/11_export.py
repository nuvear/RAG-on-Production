complete = all(isinstance(v, str) and v.strip() for v in reflections.values()) and all(
    isinstance(v, str) and v.strip() for v in human_review.values())
report = {'session': 2, 'models': [MODEL, EMBED_MODEL], 'complete': complete,
          'reflections': reflections, 'human_review': human_review,
          'observations': observations, 'api_usage': api_log}
report_path = Path('rag_session2_report.json')
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print('reflections complete:', complete, 'cleanup complete:', observations.get('cleanup',{}).get('complete',False))
if IN_COLAB:
    from google.colab import files
    files.download(str(report_path))
