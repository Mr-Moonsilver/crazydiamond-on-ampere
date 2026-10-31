#!/usr/bin/env python3
"""Native35 freeze proof: compare an R/O/L held bundle with its exact native34 I4 parent.

Allowed deltas are derived from the git diff between the parent's source commit
and the bundle's source commit (harness, tools, patches/s5_engine), mapped onto
bundle paths, plus the regenerated metadata. The engine subset must equal the
declared native35 fix set; compiled libraries and every other parent plugin
stay byte-identical. The runner may differ from the I4 runner only by names,
pins, the self-run gate and (L only) the lifetime window import.
"""
import argparse,ast,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from build_native30_held import sha,verify,replace,dump
import build_native35 as b35

METADATA={'S5-OVERLAP-GENERATION.json','e3/native-implementation-closure.json','e3/native-overlap-readiness.json'}


def source_delta(parent_source,source):
    names=subprocess.check_output(['git','diff','--name-only',parent_source,source,'--','harness','tools','patches/s5_engine'],cwd=ROOT,text=True).split()
    status=subprocess.check_output(['git','diff','--name-status',parent_source,source,'--','harness','tools','patches/s5_engine'],cwd=ROOT,text=True).splitlines()
    added={line.split('\t',1)[1] for line in status if line.startswith('A')}
    assert not any(line.startswith(('D','R')) for line in status),status
    return set(names),added


def bundle_paths(name):
    """Every bundle path a tracked source path lands on."""
    if name.startswith('harness/'):
        rel=name[len('harness/'):];paths={rel}
        if rel in {'native30_host.py','native30_policy.py',*b35.PLUGINS}:paths.add('native30_overlay/e3/native30_plugins/'+rel)
        return paths
    if name.startswith('patches/s5_engine/'):
        rel=name[len('patches/s5_engine/'):];paths={'e3_builder/'+rel}
        if rel.endswith('.py') and rel!='native_candidate_review.json':paths.add('e3/'+rel);paths.add('native30_overlay/e3/'+rel)
        return paths
    if name.startswith('tools/'):
        return {d+'/'+name[len('tools/'):] for d in ('native30_tools','native32_tools','native33_tools','native34_tools','native35_tools')}
    raise AssertionError(name)


def check(build,out):
    b=json.loads(build.read_text());arm=b['arm'];new=Path(b['bundle']);old=b35.PARENT
    assert sha(old/'SHA256SUMS')==b35.PARENT_MANIFEST and sha(new/'SHA256SUMS')==b['manifest']
    before=verify(old);after=verify(new);prev=verify(old/'native30_overlay');ov=verify(new/'native30_overlay')
    assert set(prev)==set(ov);changed_overlay={p for p in prev if prev[p]!=ov[p]}
    engine={p for p in changed_overlay if p.endswith(('.py','.so'))}
    assert engine==b35.EXPECTED_ENGINE_CHANGED|b35.EXPECTED_PLUGIN_CHANGED,engine
    assert not any(p.endswith('.so') for p in changed_overlay) and changed_overlay-engine<=METADATA,changed_overlay
    for plugin in (old/'native30_overlay/e3/native30_plugins').rglob('*'):
        rel='e3/native30_plugins/'+plugin.relative_to(old/'native30_overlay/e3/native30_plugins').as_posix()
        if plugin.is_file() and rel not in b35.EXPECTED_PLUGIN_CHANGED:assert prev[rel]==ov[rel],rel
    for p in ('e3/lane_chunk.py','e3/lane_dispatch.py','e3/native_startup.py','e3/s5_startup.py'):
        assert sha(new/'native30_overlay'/p)==sha(new/'e3_builder'/p[3:]),p
    for p in b35.EXPECTED_PLUGIN_CHANGED:
        assert sha(new/'native30_overlay'/p)==sha(new/Path(p).name)==sha(ROOT/'harness'/Path(p).name),p
    parent_source=json.loads((old/'native30-provenance.json').read_text())['source']
    delta,added_src=source_delta(parent_source,b['source'])
    allowed=set();allowed_added=set()
    for name in delta:
        for p in bundle_paths(name):
            (allowed_added if name in added_src else allowed).add(p)
    allowed|={'native30-provenance.json','native30-review.json','e3_builder/native_candidate_review.json','native30_overlay/SHA256SUMS'}|{'native30_overlay/'+p for p in METADATA}
    changed={p for p in before if p in after and before[p]!=after[p]}
    assert changed<=allowed,changed-allowed
    added=set(after)-set(before)
    expected_added={p for p in allowed_added if not p.startswith(('native30_tools/','native32_tools/','native33_tools/','native34_tools/'))}
    expected_added|={'native35_tools/'+p.name for p in (old/'native34_tools').glob('*.py')}
    assert added==expected_added,added^expected_added
    assert not set(before)-set(after)
    for p in ('native34_gate.py','native30_gate.py','e3_native34_causal.py','native34_control.py','native34_indexer.py','native34_storage.py','native34_diag.py','native34_grade.py'):
        assert sha(old/p)==sha(new/p),p
    prior=b35.PARENT_RUNNER.read_text();actual=Path(b['runner']).read_text()
    stem=b['name'].replace('-','_')
    norm=actual.replace(stem,'q18a_native34_i4').replace(b['name'],'q18a-native34-i4').replace('from native35_gate import lifetime_window_ok as window_ok','from native34_gate import window_ok').replace(b35.new_gate(arm),b35.OLD_GATE)
    assignments={n.targets[0].id:ast.get_source_segment(prior,n.value) for n in ast.parse(prior).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    assert replace(norm,{k:assignments[k] for k in ('H','BUNDLE','REVIEW','EXPECTED_SOURCES','LAUNCHER')})==prior
    if arm=='L':assert 'from native35_gate import lifetime_window_ok as window_ok' in actual
    else:assert 'from native34_gate import window_ok' in actual
    drafts={'runs/q18a_native35_'+arm.lower()+s for s in ('.draft.sh','_declare.sh','_settings.sh')}
    assert drafts<=added
    result=dict(verdict='PASS',classification=None,arm=arm,parent_manifest=b35.PARENT_MANIFEST,parent_source=parent_source,source=b['source'],manifest=b['manifest'],
        engine_changed=sorted(engine&b35.EXPECTED_ENGINE_CHANGED),plugins_changed=sorted(engine&b35.EXPECTED_PLUGIN_CHANGED),compiled_libraries_byte_identical=True,
        other_parent_plugins_byte_identical=True,gate_row4_and_native34_grade_byte_identical=True,
        new_device_behavior='engine: native-only metadata-builder view with private storage and a one-generation reuse lease; plugins: arm-O evidence readback without drain or fence',
        runner_only_names_pins_count_gate_and_l_window=True,changed=sorted(changed),added=sorted(added),removed=[],overlay_changed=sorted(changed_overlay),source_delta=sorted(delta),gpu_execution=False,box_action=False)
    assert not out.exists();dump(out,result);print(json.dumps(dict(verdict='PASS',arm=arm,out=str(out),sha256=sha(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();check(a.build,a.out)
