def image_search(question):
    vectors = embed([item['caption'] for item in IMAGE_RECORDS] + [question])
    query = vectors[-1]
    scores = [sum(a*b for a,b in zip(v, query)) for v in vectors[:-1]]
    return sorted([{**item, 'score': score} for item, score in zip(IMAGE_RECORDS, scores)],
                  key=lambda item: item['score'], reverse=True)

IMAGE_QUESTION = 'How many occupied seats were recorded on Tuesday?'
image_hits = image_search(IMAGE_QUESTION)
selected = image_hits[0]
evidence = [{'id': selected['id'], 'text': selected['caption']}]
text_only = grounded(IMAGE_QUESTION, evidence)
image_uri = 'data:image/png;base64,' + base64.b64encode(image_bytes[selected['id']]).decode()
with_image = grounded(IMAGE_QUESTION, evidence, image=image_uri)
observations['lab7'] = {'ranked_ids': [h['id'] for h in image_hits], 'selected_id': selected['id'],
                        'text_only': text_only, 'with_image': with_image}
print(json.dumps(observations['lab7'], indent=2))
from IPython.display import Image, display
display(Image(data=image_bytes[selected['id']]))
