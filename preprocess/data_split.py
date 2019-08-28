import json
import sys
import numpy as np
np.random.seed(233)

# input(original fewrel files): fewrel_val.json / fewrel_train.json
# output(original fewrel files): rel2id.json / fewrel80_train.json / fewrel80_test_test.json / fewrel80_test_train.json
# about rel2id.json: 2-65 for trainset, 66-81 for testset

def format_transform(old_line,rid,rel2id):
    new_line = {}
    new_line['relation'] = rid
    new_line['relid'] = rel2id[rid]
    new_line['sentence'] = old_line['tokens']

    new_line['head'] = {}
    new_line['head']['word'] = old_line['h'][0]
    new_line['head']['type'] = old_line['h'][1]
    new_line['head']['e1_begin'] = old_line['h'][2][0][0]
    new_line['head']['e1_end'] = old_line['h'][2][0][-1]

    new_line['tail'] = {}
    new_line['tail']['word'] = old_line['t'][0]
    new_line['tail']['type'] = old_line['t'][1]
    new_line['tail']['e2_begin'] = old_line['t'][2][0][0]
    new_line['tail']['e2_end'] = old_line['t'][2][0][-1]
    return new_line

with open('fewrel_train.json','r') as r:
    train_data = json.load(r)

id_count = 2
rel2id = {}

new_lines_train = []
for key in train_data:
    rel2id[key] = id_count
    id_count += 1

for key in train_data:
    for item in train_data[key]:
        new_lines_train.append(format_transform(item,key,rel2id))


with open('fewrel_val.json','r') as r:
    val_data = json.load(r)

new_lines_test = []
for key in val_data:
    rel2id[key] = id_count
    id_count += 1

for key in val_data:
    for item in val_data[key]:
        new_lines_test.append(format_transform(item,key,rel2id))

new_lines = new_lines_train + new_lines_test

with open('rel2id.json','w') as w:
    json.dump(rel2id,w,indent=4)
with open('fewrel80_train.json','w') as w:
    json.dump(new_lines_train,w,indent=4)
# with open('fewrel80_test.json','w') as w:
#     json.dump(new_lines_test,w,indent=4)
# with open('fewrel80.json','w') as w:
#     json.dump(new_lines,w,indent=4)

test_idx = np.random.choice(np.arange(len(new_lines_test)),1600,replace=False)
new_lines_test_test = []
new_lines_test_train = []
for idx,item in enumerate(new_lines_test):
    if idx in test_idx:
        new_lines_test_test.append(item)
    else:
        new_lines_test_train.append(item)

with open('fewrel80_test_test.json','w') as w:
    json.dump(new_lines_test_test,w,indent=4)
with open('fewrel80_test_train.json','w') as w:
    json.dump(new_lines_test_train,w,indent=4)

    