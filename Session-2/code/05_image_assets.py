# Captions support retrieval but deliberately omit numerical answers.
IMAGE_RECORDS = [{'id': 'IMG-SEATS', 'filename': 'seats.png', 'caption': 'A bar chart of occupied seats in the makerspace on Monday, Tuesday and Wednesday.', 'sha256': 'a5c7665d4571066a01a82168747ca75b1b89a841b7d8c2543d00a31ebfe2c18a'}, {'id': 'IMG-PRINTS', 'filename': 'prints.png', 'caption': 'A bar chart of completed print jobs in the makerspace on Monday, Tuesday and Wednesday.', 'sha256': '77964c609172efc6349bc87d54348f34226a14f5a01555cb4bde901a2682ee97'}]
ASSET_BASE = 'https://raw.githubusercontent.com/nuvear/RAG-on-Production/main/Session-2/data/'
image_bytes = {}
for item in IMAGE_RECORDS:
    local_dir = os.environ.get('SESSION2_DATA_DIR')
    if local_dir:
        blob = (Path(local_dir) / item['filename']).read_bytes()
    else:
        with urlopen(ASSET_BASE + item['filename'], timeout=30) as response:
            blob = response.read()
    if hashlib.sha256(blob).hexdigest() != item['sha256']:
        raise ValueError('Image hash mismatch. Use the released workshop assets.')
    image_bytes[item['id']] = blob
print('Verified image assets:', list(image_bytes))
