#!/usr/bin/env python3
"""Compose the exploratory q2024p-j2 decode-merge probe packets (dmp-off, dmp-offsmall) from the measured j2 bundle.

Reuses tools/build_q2024p_j2_hgb.py (same parent f63cff5d1e19, same frozen regenerator and runner derivation) with:
  * the reviewed source delta 06d2beee..commit (the host-gap batch at FLASH_S5_SMALL_PIECE_BATCH=0, plus this branch:
    decode_merge 'off-small', DecodeLiveness, the submit-cpu ride-along), each modified file proved identical to the
    parent bundle's copy first;
  * bundle n3j2_packet.py: marked tail lines, j2's configure then n3dmp_packet (sp | n3 X X n3 on seed 1);
  * bundle n3j2_gate.py: the nominal budget for five replay arms (marked line), every per-phase cap j2's;
  * bundle n3j2_box.py: the host-gap status line, plus the liveness guard bound and its status-row line (marked);
  * bundle tools/q2024p_j2_queue_probe.py (run by the frozen CPU tool): the queue loop under N3DMP_VARIANT with the probe
    selection native like n3, then the capture probe back on j2's table (marked lines);
  * the queue triple under the new name: settings + FLASH_S5_SMALL_PIECE_BATCH=0 FLASH_S5_DECODE_LIVENESS=1
    FLASH_S5_SUBMIT_CPU=1 (asserted by declaration and draft), the arm loop sp n3 X X n3, the budget header;
  * the runner, frozen CPU tool and queue adapter under the new name, binding the moved engine modules as well.
No box command, GPU, VM, deletion or push.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_q2024p_j2_hgb_base', ROOT / 'tools/build_q2024p_j2_hgb.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

PROBES = {'off': ('nm', 'off'), 'offsmall': ('ns', 'off-small')}
SETTINGS_ADD = ('FLASH_S5_SMALL_PIECE_BATCH=0', 'FLASH_S5_DECODE_LIVENESS=1', 'FLASH_S5_SUBMIT_CPU=1')
MODULES = base.HGB_MODULES + ('stage_overlap', 'native_overlap_control')
SUITES = base.SUITES + ('test_decode_merge_probe', 'test_dmp_reader')
J2_LOOP = 'core:sp.g2-live-0 core:n3.g2-live-0 core:n3.g2-live-1 core:sp.g2-live-1'
J2_HEADER = '# n3j2: operator00:24 budget exception,54min nominal/81min hard; relative limits only.\n'
# Five replay arms: n3j2_gate.estimate() is 3757 s at scale 1 (j2: 3236 s); 64 min nominal, x1.5 = 96 min hard.
NOMINAL_MIN = 64
HARD_MIN = 96
GATE_OLD = 'NOMINAL_S = 54 * 60\n'
GATE_NEW = 'NOMINAL_S = %d * 60  # decode-merge probe: five replay arms (sp n3 X X n3); j2: 54\n' % NOMINAL_MIN
PACKET_TAIL = ('\n# decode-merge probe (q2024p-j2-dmp, exploratory): j2\'s configure above; the probe table only where the\n'
               '# environment names it (N3DMP_VARIANT, exported by the dmp draft), j2\'s table everywhere else.\n'
               'import n3dmp_packet as _n3dmp\n'
               'configure = _n3dmp.wrap_env(configure)\n')
DRAFT_ANCHOR = 'set -uo pipefail\numask 077\n'
DRAFT_EXPORT = ('export N3DMP_VARIANT=%s  # decode-merge probe: n3j2_packet.configure applies the dmp table in every python '
                'started below\n')
J2_SHAPE = '# sp0,n30,n31,sp1 at480s per window are retained.'
BOX_IMPORT_NEW = (base.BOX_IMPORT_NEW +
                  'import decode_liveness_guard  # decode-merge probe: the arm liveness guard\n'
                  'decode_liveness_guard.install(e3_compose5_box)  # decode-merge probe\n')
BOX_ROW_NEW = base.BOX_ROW_NEW.replace(
    '    persist_small_piece_batch(row, reply)\n',
    '    persist_small_piece_batch(row, reply)\n'
    '    # decode-merge probe (observational): keep the engine liveness block; no key without one.\n'
    '    decode_liveness_guard.persist(row, reply)\n')
TOOLS = ('tools/q2024p_j2_hgb_reader.py', 'tools/build_q2024p_j2_hgb.py', 'tools/build_q2024p_j2_dmp.py',
         'tools/q2024p_j2_dmp_reader.py')


def name_of(variant):
    assert variant in PROBES, variant
    return base.PARENT_NAME + '-dmp-' + variant


def once(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new, 1)


def queues(variant):
    """j2's queue triple under the new name; the settings, their assertions, the arm loop and the budget header."""
    name = name_of(variant)
    sel, mode = PROBES[variant]
    result = {}
    for suffix in ('.draft.sh', '_declare.sh', '_settings.sh'):
        text = (base.PARENT / 'runs' / (base.under(base.PARENT_NAME) + suffix)).read_text()
        text = text.replace(base.under(base.PARENT_NAME), base.under(name)).replace(base.PARENT_NAME, name)
        if suffix == '_settings.sh':
            text = once(text, 'FLASH_S5_NEXT_LAYER_PREFETCH=1 ', 'FLASH_S5_NEXT_LAYER_PREFETCH=1 ' + ' '.join(SETTINGS_ADD) + ' ')
        else:
            anchor = "assert env.get('FLASH_S5_NEXT_LAYER_PREFETCH') == '1', 'prefetch differs'\n"
            added = ''.join("assert env.get(%r) == %r, 'decode-merge probe setting differs: %s'\n" % (k, v, k)
                            for k, v in (s.split('=') for s in SETTINGS_ADD))
            text = once(text, anchor, anchor + added)
        if suffix == '.draft.sh':
            text = once(text, J2_LOOP, 'core:sp.g2-live-0 core:n3.g2-live-0 core:%s.g2-live-0 core:%s.g2-live-1 core:n3.g2-live-1' % (sel, sel))
            text = once(text, J2_SHAPE, '# sp0,n30,%s0,%s1,n31 at 480s on seed 1 (dmp) replace j2\'s sp0,n30,n31,sp1.' % (sel, sel))
            head, sep, rest = text.partition(DRAFT_ANCHOR)
            code = [line for line in head.splitlines() if not line.lstrip().startswith('#')]
            assert sep and not any(m in line for line in code for m in ('n3j2_packet', 'n3j2_gate', 'n3j2_box', 'n3j2_arms', 'python')), \
                'table use before the export'
            text = once(text, DRAFT_ANCHOR, DRAFT_ANCHOR + DRAFT_EXPORT % variant)
            text = once(text, J2_HEADER, '# dmp-%s (exploratory decode-merge probe, decode_merge %s): %dmin nominal/%dmin hard from n3j2_gate '
                                         '(five replay arms; j2 54/81 by the operator\'s 00:24 exception); relative limits only.\n'
                                         % (variant, mode, NOMINAL_MIN, HARD_MIN))
        result['runs/' + base.under(name) + suffix] = text
    return result


PROBE_EDITS = (
    ("    table=configure();fixture=parent.ArmLoop(",
     "    import n3dmp_packet as _n3dmp  # decode-merge probe: the queue loop on the probe table, as the draft exports it\n"
     "    os.environ[_n3dmp.ENV] = %(variant)r  # decode-merge probe\n"
     "    table=configure();fixture=parent.ArmLoop("),
    ("enabled=label.startswith('n3.'))", "enabled=label.startswith(('n3.', %(sel)r)))"),   # decode-merge probe: X is native
    ("if label.startswith(('sp.','n3.')))", "if label.startswith(('sp.','n3.', %(sel)r)))"),
    ("    complete=capture_probe(reader)\n",
     "    os.environ.pop(_n3dmp.ENV)  # decode-merge probe: the capture probe on j2's table (its rule fixture is j2's)\n"
     "    configure()  # decode-merge probe: ARM_TABLE back to j2's\n"
     "    complete=capture_probe(reader)\n"),
)


def bundle_edits(bundle, variant):
    sel, _ = PROBES[variant]
    probe = bundle / 'tools/q2024p_j2_queue_probe.py'
    text = probe.read_text()
    for old, new in PROBE_EDITS:
        text = once(text, old, new % dict(variant=variant, sel=sel + '.'))
    probe.write_text(text)
    packet = bundle / 'n3j2_packet.py'
    packet.write_text(packet.read_text() + PACKET_TAIL)
    gate = bundle / 'n3j2_gate.py'
    gate.write_text(once(gate.read_text(), GATE_OLD, GATE_NEW))


def install(variant):
    """Point the hgb tool at this packet: names, modules, suites, marked lines, the delta's extra files."""
    base.VARIANTS = {v: '0' for v in PROBES}            # the batch setting of every dmp packet (recorded as small_piece_batch)
    base.name_of = name_of
    base.queues = queues
    base.HGB_MODULES = MODULES
    base.ENGINE_MOVED = {'e3/' + n + '.py' for n in MODULES}
    base.SUITES = SUITES
    base.BOX_IMPORT_NEW = BOX_IMPORT_NEW
    base.BOX_ROW_NEW = BOX_ROW_NEW
    hgb_copy_delta = base.copy_delta

    def copy_delta(b, commit, bundle):
        copied = hgb_copy_delta(b, commit, bundle)
        for name in ('harness/decode_liveness_guard.py', 'harness/n3dmp_packet.py', 'harness/test_decode_merge_probe.py', 'harness/test_dmp_reader.py',
                     'patches/s5_engine/stage_overlap.py', 'patches/s5_engine/native_overlap_control.py'):
            assert name in copied, 'delta lacks ' + name
        for tool in TOOLS:
            target = bundle / 'tools' / Path(tool).name
            target.write_bytes(b.git(ROOT, 'show', commit + ':' + tool))
            copied[tool] = dict(status='A', bundle=target.relative_to(bundle).as_posix(), sha256=b.sha(target))
        bundle_edits(bundle, variant)
        return copied
    base.copy_delta = copy_delta
    hgb_frozen_cpu_tool = base.frozen_cpu_tool

    def frozen_cpu_tool(bundle):
        hgb_frozen_cpu_tool(bundle)
        path = bundle / 'tools/n3j2_frozen_cpu.py'
        path.write_text(once(path.read_text(), "'q2024p-j2-hgb-frozen-cpu-'", "'q2024p-j2-dmp-frozen-cpu-'"))
    base.frozen_cpu_tool = frozen_cpu_tool


def prepare(commit, variant, out):
    install(variant)
    record = base.prepare(commit, variant, out)
    sel, mode = PROBES[variant]
    record.update(schema='q2024p-j2-dmp-build/1', probe_selection=sel, decode_merge=mode, settings_added=SETTINGS_ADD,
                  nominal_minutes=NOMINAL_MIN, hard_minutes=HARD_MIN, dmp_modules=MODULES)
    path = out / 'build.json'
    path.chmod(0o644)
    base_record = json.loads(path.read_text())
    base_record.update(record)
    path.write_text(json.dumps(base_record, indent=2, sort_keys=True) + '\n')
    return base_record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--variant', choices=sorted(PROBES), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source_commit, args.variant, args.out.resolve())
    print(json.dumps({k: v for k, v in result.items() if k != 'source_changes'}, sort_keys=True))
    sys.exit(0)
