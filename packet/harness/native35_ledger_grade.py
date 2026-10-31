"""Native35 startup ledger: per-rank native-private builder storage, allocated once.

Reads the retained startup grade (queried construction inventory per rank) and
requires the native_attention_builder_storage record on all four ranks: exact
cloned bytes, every builder field classified, no GDN/linear alignment alias of
stock storage, no metadata generation in flight. Bytes are reported against the
queried free device memory; no headroom verdict beyond "fits" is claimed.
"""
import argparse
import json
from pathlib import Path

CLASSES={'private-tensor','private-container','shared-read-only'}


def grade(startup):
    failures,missing,ledger=[],[],{}
    workers=startup.get('workers') if type(startup) is dict else None
    if type(workers) is not dict:
        return dict(schema='native35-ledger/1',verdict='INCONCLUSIVE',failures=[],missing=['startup-workers'],ledger={})
    for rank in range(4):
        prefix='rank%d:'%rank;row=workers.get(str(rank)) or workers.get(rank)
        if type(row) is not dict:missing.append(prefix+'startup-row');continue
        inv=row.get('native_inventory') or {};storage=inv.get('native_attention_builder_storage')
        if type(storage) is not dict:missing.append(prefix+'native-attention-builder-storage');continue
        nbytes=storage.get('native_builder_bytes');count=storage.get('native_builder_tensor_count');fields=storage.get('native_builder_fields')
        ok=type(nbytes) is int and nbytes>=0 and type(count) is int and count>=1 and type(fields) is list and bool(fields)
        if not ok:failures.append(prefix+'builder-storage-shape');continue
        classified=0;private_tensor_fields=0;field_bytes=0;bad=[]
        for builder in fields:
            for field in (builder.get('fields') or []) if type(builder) is dict else []:
                classified+=1;c=field.get('classification')
                if c not in CLASSES or type(field.get('tensor_bytes')) is not int or field['tensor_bytes']<0:bad.append(field.get('name'))
                if c=='private-tensor':private_tensor_fields+=1
                field_bytes+=field.get('tensor_bytes') if type(field.get('tensor_bytes')) is int else 0
        if bad:failures.append(prefix+'unclassified-builder-fields:'+','.join(str(b) for b in bad[:8]))
        if classified==0 or private_tensor_fields==0:failures.append(prefix+'no-private-builder-tensor')
        if field_bytes!=nbytes:failures.append(prefix+'field-bytes-differ-from-ledger')
        if count<private_tensor_fields:failures.append(prefix+'tensor-count-below-private-fields')
        if storage.get('metadata_generation_in_flight') is not False:failures.append(prefix+'metadata-generation-in-flight-at-startup')
        if storage.get('metadata_generation')!=0 or storage.get('chunks')!=0:failures.append(prefix+'chunks-before-first-admission')
        gdn=storage.get('gdn_linear_alignment')
        if type(gdn) is not dict:missing.append(prefix+'gdn-linear-alignment-inventory')
        elif gdn.get('stock_aliases')!=0:failures.append(prefix+'gdn-context-aliases-stock-storage')
        memory=row.get('memory') or {};free=memory.get('free_bytes');total=memory.get('total_bytes')
        if type(free) is not int or type(total) is not int:missing.append(prefix+'queried-memory')
        elif nbytes>=free:failures.append(prefix+'private-builder-bytes-exceed-free-memory')
        ledger[str(rank)]=dict(native_builder_bytes=nbytes,native_builder_tensor_count=count,builders=len(fields),
            classified_fields=classified,private_tensor_fields=private_tensor_fields,
            gdn_gpu_bytes=(gdn or {}).get('gpu_bytes') if type(gdn) is dict else None,
            gdn_host_bytes=(gdn or {}).get('host_bytes') if type(gdn) is dict else None,
            lane_tables_bytes=storage.get('lane_tables_bytes'),free_bytes=free,total_bytes=total)
    return dict(schema='native35-ledger/1',verdict='FAIL' if failures else 'INCONCLUSIVE' if missing else 'PASS',
        failures=failures,missing=missing,ledger=ledger,
        scope='queried startup inventory only; one-time allocation bytes per rank; fits-in-free-memory check, no loaded-headroom claim')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--startup',required=True);p.add_argument('--out',required=True);a=p.parse_args(argv)
    try:startup=json.loads(Path(a.startup).read_text())
    except (OSError,ValueError) as exc:startup={'unreadable':type(exc).__name__}
    result=grade(startup)
    with Path(a.out).open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print('NATIVE35_LEDGER '+result['verdict'])
    return {'PASS':0,'FAIL':1}.get(result['verdict'],2)


if __name__=='__main__':raise SystemExit(main())
