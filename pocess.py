import json
import codecs

file_name = 'dataset/WebNLG/data/train.json'

with codecs.open(file_name, 'r', encoding='utf-8') as ftr:
    file = json.load(ftr)

with codecs.open(file_name, 'w', encoding='utf-8') as f:
    json.dump(file, f, ensure_ascii=False)
