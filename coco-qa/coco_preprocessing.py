import json
import os

#dir with annotations
dir_ann = '/home/angeliki/sata/DATA/VQA/questions/'

#load annotations
train = json.load(open(dir_ann+'train2014/OpenEnded_mscoco_train2014_questions.json','r'))
val = json.load(open(dir_ann+'val2014/OpenEnded_mscoco_val2014_questions.json','r'))


# combine all qs and annotations together
qs = val['questions'] + train['questions']

# for efficiency lets group annotations by image
itoa = {}
for q in train['questions']:
    imgid = str(q['image_id'])
    if not imgid in itoa: itoa[imgid] = []
    q['split'] = 'train'
    itoa[imgid].append(q)
    
for q in val['questions']:
    imgid = str(q['image_id'])
    if not imgid in itoa: itoa[imgid] = []
    q['split'] = 'val'
    itoa[imgid].append(q)


# create the json blob
out = []
for i,q in enumerate(qs):
    imgid = str(q['image_id'])
    
    # coco specific here, they store train/val images separately
    loc = 'train2014' if q['split']=='train' else 'val2014'
    
    jimg = {}
                                           #COCO_train2014_00000000014.jpg
    jimg['file_path'] = os.path.join(loc, 'COCO_'+loc+'_'+'0' * (12-len(imgid)) +imgid+'.jpg')
    jimg['id'] = imgid
    
    questions = []
    annotsi = itoa[imgid]
    for a in annotsi:
        questions.append(a['question'])
    #just for convenience we will name them captions
    jimg['captions'] = questions
    out.append(jimg)
    
json.dump(out, open('coco_qa_raw.json', 'w'))

print out[0]
