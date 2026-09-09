"""Create original synthetic chart evidence for Lab 7, not model-generated imagery.

Authoring only: requires matplotlib==3.10.8. Colab downloads the finished PNGs.
"""
from pathlib import Path
import json, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
records = []
for image_id, filename, title, unit, values, caption in [
    ('IMG-SEATS','seats.png','Makerspace occupancy','Occupied seats',[18,42,27],
     'A bar chart of occupied seats in the makerspace on Monday, Tuesday and Wednesday.'),
    ('IMG-PRINTS','prints.png','Makerspace printing','Completed print jobs',[31,16,25],
     'A bar chart of completed print jobs in the makerspace on Monday, Tuesday and Wednesday.')]:
    fig, ax = plt.subplots(figsize=(10,6), dpi=120)
    fig.patch.set_facecolor('#f7f5ef'); ax.set_facecolor('#f7f5ef')
    bars=ax.bar(['Monday','Tuesday','Wednesday'],values,color=['#779aaa','#087e83','#779aaa'],width=.55)
    ax.bar_label(bars,padding=7,fontsize=22)
    ax.set_title(title,fontsize=26,pad=24);ax.set_ylabel(unit,fontsize=17)
    ax.set_ylim(0,60);ax.tick_params(labelsize=16)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.02,'Original synthetic classroom data • source '+image_id,ha='center',fontsize=12)
    fig.tight_layout(rect=(.02,.05,.98,.98))
    path=ROOT/'data'/filename;fig.savefig(path);plt.close(fig)
    records.append({'id':image_id,'filename':filename,'caption':caption,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(ROOT/'data/image_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
code='''# Captions support retrieval but deliberately omit numerical answers.
IMAGE_RECORDS = '''+repr(records)+'''
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
'''
(ROOT/'code/05_image_assets.py').write_text(code)
print('Created two chart images, manifest and asset-loading cell.')
