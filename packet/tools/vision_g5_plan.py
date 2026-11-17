"""Prospective seven-case G5 plan; four originals plus multi-image/order/native-boundary cases."""
import copy,json

def cases(original,prefix_repeats=1900):
 if len(original)!=4:raise ValueError('four original known-label cases required')
 images=copy.deepcopy(original);images[0]['image']='red1024.png'
 def multi(name,group,kind):
  questions=[x['question'] for x in group]
  return dict(id=name,kind=kind,images=[x['image'] for x in group],expected=[x['expected'] for x in group],
   question='Ignore filler x and z. Answer these questions for the images in order: '+json.dumps(questions)+' Return ONLY a JSON array of four answers, no explanation.',
   prefix_repeats=prefix_repeats if kind=='native-boundary' else 0,suffix_repeats=4600 if kind=='native-boundary' else 0)
 return copy.deepcopy(original)+[multi('four-distinct',images,'multi'),multi('four-reversed',list(reversed(images)),'multi'),multi('four-native-boundary',images,'native-boundary')]

def messages(case,image_part):
 if 'images' not in case:return [dict(role='user',content=[image_part(case['image']),dict(type='text',text=case['question'])])]
 content=[]
 if case.get('prefix_repeats'):content.append(dict(type='text',text='G5 independent native-splice geometry. Ignore filler.'+' x'*case['prefix_repeats']))
 content.extend(image_part(name) for name in case['images'])
 content.append(dict(type='text',text=' z'*case.get('suffix_repeats',0)+'\n'+case['question']))
 return [dict(role='user',content=content)]
