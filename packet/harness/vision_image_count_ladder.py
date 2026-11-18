#!/usr/bin/env python3
"""One-at-a-time known-number image ladder; never raises the engine's eight-image safety cap."""
import argparse,base64,hashlib,http.client,json,re,struct,sys,time,traceback,uuid,zlib
from pathlib import Path
from urllib.parse import urlsplit
STEPS=(4,8,12,16,20,24,32)
FONT={'0':['111','101','101','101','111'],'1':['010','110','010','010','111'],'2':['111','001','111','100','111'],'3':['111','001','111','001','111'],'4':['101','101','111','001','001'],'5':['111','100','111','001','111'],'6':['111','100','111','101','111'],'7':['111','001','010','010','010'],'8':['111','101','111','101','111'],'9':['111','101','111','001','111']}
def png(number):
 text=str(number);width=height=128;pixels=bytearray(b'\xff'*(width*height*3));scale=12;left=(width-(len(text)*4-1)*scale)//2
 for k,digit in enumerate(text):
  for y,row in enumerate(FONT[digit]):
   for x,on in enumerate(row):
    if on=='1':
     for dy in range(scale):
      for dx in range(scale):
       at=((34+y*scale+dy)*width+left+(k*4+x)*scale+dx)*3;pixels[at:at+3]=b'\0\0\0'
 raw=b''.join(b'\0'+pixels[y*width*3:(y+1)*width*3] for y in range(height))
 def chunk(kind,data):return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
 return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!IIBBBBB',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
def memory(log,request_id,response_id):
 candidates={request_id,response_id}
 if isinstance(response_id,str):candidates.update(re.findall(r'(?<![A-Za-z0-9_-])'+re.escape(response_id)+r'-[0-9a-f]{8}(?![A-Za-z0-9_-])',log))
 hashes={hashlib.sha256(x.encode()).hexdigest() for x in candidates if isinstance(x,str)};rows=[]
 for line in log.splitlines():
  if 'flash-s5 native-mm-boundary ' not in line:continue
  row=json.loads(line.split('flash-s5 native-mm-boundary ',1)[1]);
  if row.get('request_sha256') in hashes:rows.append(row)
 result={}
 for rank in range(4):
  found=[r for r in rows if r.get('worker_rank')==rank]
  if not found or len({r['request_sha256'] for r in found})!=1:raise ValueError('MM-boundary-rank-'+str(rank)+'-missing-or-ambiguous')
  result[str(rank)]=dict(minimum_free_bytes=min(r['free_bytes'] for r in found),max_cache_bytes=max(r['native_mm']['cache_storage_bytes'] for r in found),boundary_count=len(found))
 return result

def run(url,key,engine_log,out,steps=STEPS):
 u=urlsplit(url);assert u.scheme=='http' and not u.username and not u.password
 result=dict(schema='vision-image-ladder/1',largest_clean=0,limiting_factor=None,steps=[],scope='single request, synthetic 128px numbered images; allocator peaks are encoder GPU metrics, not physical-card peak')
 out.mkdir(parents=True,exist_ok=False)
 for n in steps:
  row=dict(images=n,status='HOLD');result['steps'].append(row)
  try:
   offset=engine_log.stat().st_size
   uid='w28-ladder-'+uuid.uuid4().hex
   body=dict(model='qwen38-w8a8-tp4',request_id=uid,messages=[dict(role='user',content=[dict(type='text',text='Ignore filler x and z. '+('x '*1500))]+[dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(png(i)).decode())) for i in range(1,n+1)]+[dict(type='text',text=('z '*4600)+'Read the number printed in each image. Reply only with a JSON array of integers in image order, without explanation.')])],max_tokens=256,temperature=0,chat_template_kwargs=dict(enable_thinking=False))
   conn=http.client.HTTPConnection(u.hostname,u.port or 80,timeout=180);started=time.monotonic()
   try:
    conn.request('POST',u.path.rstrip('/')+'/v1/chat/completions',json.dumps(body),{'Content-Type':'application/json','Authorization':'Bearer '+key})
    reply=conn.getresponse();raw=reply.read();row.update(http_status=reply.status,wall_s=time.monotonic()-started)
    if reply.status!=200:
     value=json.loads(raw);raise ValueError('http-'+str(reply.status)+':'+str(value.get('error',{}).get('code','backend-error')))
    value=json.loads(raw);row['response_id']=value['id']
    row['encoder_wall_ms']=float(reply.getheader('x-flashnext-encoder-wall-ms'))
    row['encoder_peak_allocated_bytes']=int(reply.getheader('x-flashnext-encoder-peak-allocated-bytes'))
    row['encoder_peak_reserved_bytes']=int(reply.getheader('x-flashnext-encoder-peak-reserved-bytes'))
   finally:conn.close()
   content=value['choices'][0]['message']['content'];row['answer']=content
   try:labels=json.loads(content.strip().removeprefix('```json').removesuffix('```').strip())
   except ValueError:raise ValueError('labels-not-JSON') from None
   if labels!=list(range(1,n+1)):raise ValueError('wrong-label-or-order')
   for attempt in range(10):
    with engine_log.open(errors='replace') as f:f.seek(offset);log=f.read()
    try:row['ranks']=memory(log,uid,value['id']);break
    except ValueError:
     if attempt==9:raise
     time.sleep(.2)
   if any(r['max_cache_bytes']>40*2**20 for r in row['ranks'].values()):raise ValueError('MM-cache-over40MiB')
   if any(r['minimum_free_bytes']<40*2**20 for r in row['ranks'].values()):raise ValueError('rank-physical-free-below40MiB')
   if row['encoder_peak_reserved_bytes']<=0:raise ValueError('encoder-peak-missing')
   row['status']='PASS';result['largest_clean']=n
  except (OSError,ValueError,KeyError,TypeError,http.client.HTTPException) as exc:
   row['reason']=str(exc);result['limiting_factor']=row['reason']
  (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
  print('VISION_IMAGE_LADDER_STEP '+str(n)+' '+row['status']+' '+row.get('reason','labels-ordered-cache-floor-PASS'),flush=True)
  if row['status']!='PASS':break
 result['complete']=True
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print('VISION_IMAGE_LADDER largest_clean='+str(result['largest_clean'])+' limit='+str(result['limiting_factor'] or 'requested-sweep-complete'))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--key-file',type=Path,required=True);p.add_argument('--engine-log',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 try:
  from vision_answer_probe import key
  run(a.url,key(a.key_file,'VLLM_API_KEY'),a.engine_log,a.out)
 except Exception as exc:
  f=traceback.extract_tb(exc.__traceback__)[-1];print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {f.filename}:{f.lineno}',file=sys.stderr);raise SystemExit(2)
