#!/usr/bin/env python3
"""Owner-only fixture export during a cleared encoder interval; no GPU/engine control."""
import argparse,base64,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from vision_answer_probe import key,post,validate_embedding
from vision_encoder_backend import MODEL_MANIFEST_SHA
from vision_encoder_service import error_info


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--encoder-url',default='http://<encoder-host>:9041')
    p.add_argument('--encoder-key-file',type=Path,required=True);p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    if a.out.exists():raise FileExistsError('fixture export already exists; fresh name required')
    token=key(a.encoder_key_file,'VISION_ENCODER_API_KEY');cases=json.loads(a.cases.read_text())
    if not isinstance(cases,list) or not 1<=len(cases)<=8:raise ValueError('bounded fixture list required')
    value=dict(schema='vision-embedding-cache/1',manifest_sha256=MODEL_MANIFEST_SHA,features={})
    for case in cases:
        image=(a.cases.parent/case['image']).resolve()
        if not image.is_relative_to(a.cases.parent.resolve()):raise ValueError('fixture image outside packet')
        raw=image.read_bytes()
        if not 0<len(raw)<=8*2**20:raise ValueError('fixture image byte bound')
        digest=hashlib.sha256(raw).hexdigest()
        if digest not in value['features']:
            value['features'][digest]=validate_embedding(post(a.encoder_url+'/v1/encode',token,
                dict(image_base64=base64.b64encode(raw).decode())))
    data=(json.dumps(value)+'\n').encode()
    if len(data)>64*2**20:raise ValueError('fixture cache exceeds64MiB')
    with os.fdopen(os.open(a.out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as out:out.write(data)
    print('VISION_EMBEDDINGS_EXPORT PASS sha256='+hashlib.sha256(data).hexdigest()+' images='+str(len(value['features']))+' path='+str(a.out))
    return 0

if __name__=='__main__':
    try:code=main()
    except (Exception,SystemExit) as exc:
        if not isinstance(exc,SystemExit) or exc.code:print('PACKET_CLIENT_ERROR '+json.dumps(error_info(exc)),file=sys.stderr)
        if isinstance(exc,SystemExit):raise
        print('VISION_EMBEDDINGS_EXPORT HOLD '+type(exc).__name__)
        raise SystemExit(2)
    raise SystemExit(code)
