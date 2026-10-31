#!/usr/bin/env python3
"""Cross-arm proof: the R, O and L bundles differ only in native30-provenance.json."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from build_native30_held import sha,verify,dump


def check(builds,out):
    rows=[(json.loads(b.read_text()),) for b in builds];manifests=[];tables=[]
    for (b,) in rows:
        bundle=Path(b['bundle']);assert sha(bundle/'SHA256SUMS')==b['manifest'];manifests.append(b['manifest']);tables.append(verify(bundle))
    base=tables[0];changed=set()
    for table in tables[1:]:
        assert set(table)==set(base),set(table)^set(base)
        changed|={name for name in base if table[name]!=base[name]}
    assert changed=={'native30-provenance.json'},changed
    overlays={sha(Path(b['bundle'])/'native30_overlay/SHA256SUMS') for (b,) in rows};assert len(overlays)==1
    result=dict(verdict='PASS',arms=[b['arm'] for (b,) in rows],manifests=manifests,overlay_manifest=overlays.pop(),changed=sorted(changed),engine_and_plugins_byte_identical=True,gpu_execution=False)
    assert not out.exists();dump(out,result);print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,action='append',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();check(a.build,a.out)
