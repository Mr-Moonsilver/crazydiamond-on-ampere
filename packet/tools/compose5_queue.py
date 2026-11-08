#!/usr/bin/env python3
"""Derive the q18a-parity-o3f box scripts (lane-o3-fix: q18a-parity-na2's derivation with this table and name) from q18a-compose-4's committed texts. Nothing a past queue ran is edited. Discovery mode.

Declared deltas over compose-4's three texts, each by exact count:
  names     q18a-compose-4 -> q18a-compose-throughput-p2 and every compose-4 name of the queue, records, box, gate and table -> compose5. The box imports compose-4's box, table and replay, so those three keep their names;
  body      this table's core list; the arm and grade texts that name the arms;
  modules   compose-4's box and table added to the preflight's and the declaration's lists: compose-5's box binds its table over them;
  chunk     FLASH_S5_CHUNK_TOKENS=2048 in EXTRA_ENV for the whole run, as dp8d's launch; both preflights assert it, where compose-4's asserted it unset.
  yield     coverage-throughput: FLASH_S5_APPEND_YIELD_MAX_NEW is the table's (8192, a1's 2048) in EXTRA_ENV; both preflights assert it.
  ahead     coverage-throughput build-2: FLASH_S5_PREPARE_AHEAD=1 (compose5_arms.PREPARE_AHEAD_ENV) in EXTRA_ENV after the chunk; both preflights assert it.
  logits    build-2b24c64: VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 (compose5_arms.LOGITS_ENV) in EXTRA_ENV after prepare-ahead, for the whole run; both preflights assert it.
  o3        lane-o3: FLASH_S5_O3=1 (compose5_arms.O3_ENV) in EXTRA_ENV after the logits cap, for the whole run (installed; each arm selects it by o3_split); both preflights assert it.
  nosync    lane-o3-na: FLASH_S5_AHEAD_NOSYNC=1 (compose5_arms.AHEAD_NOSYNC_ENV) right after it, for the whole run (each arm selects it by ahead_nosync); both preflights assert it.
  split     parity-n3s: the decode SM count is the table's (compose5_arms.DECODE_SMS, legal by compose5_arms.legal_decode_sms) where compose-4 declared 44: the settings'
            default, guard and EXTRA_ENV, the draft's guard, the declaration's assert; the draft grades startup with e3_split_startup_grade.py (the native grade's
            two literal 44s bound to the count), listed beside e3_native_startup_grade.py in both interface lists.
The other switches, the yield's other three bounds and the ingress check stay compose-4's. The arm list comes from harness/compose5_arms.py; --check verifies the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import compose5_arms as table  # noqa: E402

OLD, NEW = 'q18a_compose_4', 'q18a_parity_n3s24'
OLD_CORE = 'core:dec1.g2-live-0 core:cmp.g2-live-0 core:ctl.g2-live-0'
CORE = ' '.join('%s:%s' % (r['part'], r['label']) for r in table.arms(table.GRADED) if r['label'] != table.SMOKE_ARM)
ARM_TEXT = ('dec1 one standing stream and one decode window, cmp and ctl eight replay agents for 18 minutes', 'dec1 one standing stream and one decode window, pf po pg two decoders and three back-to-back natives (O3 off, on, off), sp n3 eight replay agents for %d s, sp n3 n3 sp' % table.REPLAY['DURATION'])
HEADER_ANCHOR = '# compose-4: the four lever tracks on one engine under the closed-loop agent replay'
HEADER_NOTE = ("# parity-n3s: q18a-parity-n3u's engine, launch, table, gate, grade and stop rule with the decode SM count moved: FLASH_S5_DECODE_SMS=%d of 82 (n3u 44), the only knob (compose5_arms.DECODE_SMS); the startup grade binds it exactly (tmp/decode-phase/READY-parity-n3s.md). " % table.DECODE_SMS
               + "# parity-n3u: q18a-parity-o3f's engine and launch on Codex's legacy UVA retirement (926f17f1: private native UVA descriptor slots, retired before reuse; native_uva.py in the package): the pair arms pf po pg (the stop rule unchanged), the replay sp n3 | n3 sp, n3 against this run's stock sp (tmp/decode-phase/READY-parity-n3u.md). The reader is parity-n3u/scratch/parity_n3u_read.py on the hub. "
               "# lane-o3-fix: q18a-parity-na2's engine and launch with the fixed O3 and K2's settle: the pair arms pf po pg back (the stop rule unchanged), the replay na nk n3 | n3 nk na, nk being na with k2_settle on native_stage_local (acknowledged: FAIL otherwise), n3 nk with O3; the grade INC on an O3 arm that kept runs and armed none and on nk/n3 with K2 never settled, FAIL on K2 settling on another arm (tmp/decode-phase/READY-parity-o3f.md). The reader is parity-b/scratch/o3f/parity_o3f_read.py on the hub. "
               '# parity-na2: q18a-parity-o3\'s engine and launch (FLASH_S5_O3=1 installed, o3_split False on every arm), the lean repeat of na against nt and sp: '
               'the replay na nt sp | sp nt na, no pair arms, no \'no\'; the ahead-split receipt through the receipt writer\'s allow-list (n1 lost them all) '
               '(tmp/decode-phase/READY-parity-na2.md). '
               '# lane-o3: parity-2\'s engine, launch and table with O3, stay split across runs (DESIGN-multi-prompt-lane O3): FLASH_S5_O3=1 for the whole run, '
               'selected per arm by o3_split on native_stage_local (acknowledged: FAIL otherwise); the pair arms pf po pg (two decoders, three '
               'back-to-back native requests, greedy with top-20 logprobs; O3 off, on, off) carry the stop rule over the first-token top-20 sets '
               '(FAIL when po falls outside the pf|pg envelope); the replay no na nt sp | sp nt na no, no being nt with O3, na nt with the ahead nosync (FLASH_S5_AHEAD_NOSYNC=1 for the whole run, selected per arm by ahead_nosync; parity-ahead 9daa2ac3), sp the lane off (parity-base) (tmp/decode-phase/READY-parity-o3.md). '
               'The reader is parity-b/scratch/o3/parity_o3_read.py on the hub. '
               '# parity-2 build-1: build-2b24c64\'s engine, launch and ABBA with fix A on nt alone: the terminal successor (terminal_successor on '
               'native_stage_local, needs the exact successor; ny and the smoke send False; the grade FAILs an unacknowledged value, INC on nt never '
               'engaged, FAIL on ny engaged) (tmp/decode-phase/READY-parity-2.md). The reader is parity-2/scratch/parity2_read.py on the hub. '
               '# coverage-throughput build-2b24c64: build-2b\'s engine and full ABBA with nt\'s chunk_bound b24 [97152, 1024] (ny b13) and the indexer logits cap '
               'VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 for the whole run (both preflights assert it); the grade FAILs any allocator retry over 64 MiB on any rank. Build-2: build-1\'s table, its candidate nt (the hold, the residue and the append yield) against ny on dnr\'s closed-loop turn mix, '
               "ABBA nt ny ny nt, 480 s of traffic an arm, discovery mode, over an engine with (a) prepare-ahead, the safe variant (a later chunk's PREPARE "
               "slices 1-2 staged after the live chunk's state_finalize; FLASH_S5_PREPARE_AHEAD=1 for the whole run, used by native chunks only), (b) K2 "
               "(the next run selected at the scheduled handback, no cooldown) and (c) the geometry tail at the prompt's last prefix unit. Unchanged from "
               "build-1: the grade (the geometry-tail class, per Worker_PP RSS failing nt, any other stock prefill row and a request still open after the "
               "drain failing nt) and chunk geometry (but nt's bound) (harness/compose5_arms.py, tmp/decode-phase/READY-coverage-throughput-b2b24c64.md). "
               "The reader with the parity word is coverage-throughput/scratch/throughput2b24c64_read.py on the hub.\n")
NAMES = ((OLD, NEW), ('q18a-compose-4', 'q18a-parity-n3s24'), ('compose-4', 'compose-5'), ('compose4', 'compose5'))
# compose-5's box imports compose-4's box, table and replay by their names; the rename above must not reach them.
SWITCHES = table.CONNECTIONS_ENV + ' ' + table.PLE_ENV + ' ' + table.CHUNK_ENV
OLD_LAST_SWITCH = table.PLE_ENV + ' '
DRAFT_FILES = 'e3_compose5_box.py compose5_arms.py e3_attn_meta_box.py attn_meta_arms.py e3_inwait_box.py inwait_arms.py e3_native_refusal_client.py; do'
DECLARE_FILES = 'e3_compose5_box.py compose5_arms.py compose5_gate.py inwait_gate.py e3_inwait_box.py inwait_arms.py e3_native_refusal_client.py'
MODULES_AFTER = ' e3_a1_trunk_box.py a1_trunk_arms.py compose5_replay.py agent_replay.py'
MODULES = ' e3_a1_trunk_box.py a1_trunk_arms.py compose4_replay.py agent_replay.py e3_compose4_box.py compose4_arms.py'


def once(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new)


def renames(text):
    for old, new in NAMES:
        text = text.replace(old, new)
    return text


def source(suffix):
    return (ROOT / 'harness/runs' / (OLD + suffix)).read_text()


def chunk_assert(text, form):
    return once(text, "assert 'FLASH_S5_CHUNK_TOKENS' not in env, 'the chunk stays the default 1024'", "assert env.get('FLASH_S5_CHUNK_TOKENS')%s'%d', 'native chunk rows installed'" % (form, table.CHUNK_TOKENS))


def yield_bound(text, form):
    """coverage-throughput: the yield's new-row bound is this table's (compose5_arms.YIELD_MAX_NEW), installed and asserted."""
    return once(text, "assert env.get('FLASH_S5_APPEND_YIELD_MAX_NEW')%s'2048', 'the yield bound installed'" % form,
                "assert env.get('FLASH_S5_APPEND_YIELD_MAX_NEW')%s'%d', 'the yield bound installed'" % (form, table.LAUNCH['max_new']))


def ahead_assert(text, form):
    """coverage-throughput build-2: prepare-ahead installed for the whole run, asserted right after the chunk."""
    line = "assert env.get('FLASH_S5_CHUNK_TOKENS')%s'%d', 'native chunk rows installed'" % (form, table.CHUNK_TOKENS)
    name, value = table.PREPARE_AHEAD_ENV.split('=')
    return once(text, line, line + "\nassert env.get('%s')%s'%s', 'build-2 prepare-ahead installed'" % (name, form, value))


def logits_assert(text, form):
    """build-2b24c64: the indexer logits cap installed for the whole run, asserted right after prepare-ahead."""
    name, value = table.PREPARE_AHEAD_ENV.split('=')
    line = "assert env.get('%s')%s'%s', 'build-2 prepare-ahead installed'" % (name, form, value)
    name, value = table.LOGITS_ENV.split('=')
    return once(text, line, line + "\nassert env.get('%s')%s'%s', 'build-2b24c64 indexer logits cap installed'" % (name, form, value))


def o3_assert(text, form):
    """lane-o3: O3 installed for the whole run, asserted right after the logits cap."""
    name, value = table.LOGITS_ENV.split('=')
    line = "assert env.get('%s')%s'%s', 'build-2b24c64 indexer logits cap installed'" % (name, form, value)
    name, value = table.O3_ENV.split('=')
    return once(text, line, line + "\nassert env.get('%s')%s'%s', 'lane-o3 O3 installed'" % (name, form, value))


def nosync_assert(text, form):
    """lane-o3-na: the ahead nosync installed for the whole run (parity-ahead 9daa2ac3), asserted right after O3."""
    name, value = table.O3_ENV.split('=')
    line = "assert env.get('%s')%s'%s', 'lane-o3 O3 installed'" % (name, form, value)
    name, value = table.AHEAD_NOSYNC_ENV.split('=')
    return once(text, line, line + "\nassert env.get('%s')%s'%s', 'parity-ahead ahead nosync installed'" % (name, form, value))


def split(text, old, new):
    """parity-n3s: compose-4's literal 44 at one site, the table's legal count instead."""
    count = int(table.legal_decode_sms(table.DECODE_SMS))
    return once(text, old % ((44,) * old.count('%d')), new % ((count,) * new.count('%d')))


def draft(text):
    text = split(text, '[[ $S5_REQUESTED_DECODE_SMS == %d ]] || fail_early decode-sms-must-be-%d', '[[ $S5_REQUESTED_DECODE_SMS == %d ]] || fail_early decode-sms-must-be-%d')
    text = once(text, 'python3 "$H/e3_native_startup_grade.py"', 'python3 "$H/e3_split_startup_grade.py"')
    text = once(text, 'for _interface in e3_native_startup_grade.py ', 'for _interface in e3_native_startup_grade.py e3_split_startup_grade.py ')
    text = yield_bound(chunk_assert(text, ' == '), ' == ')
    text = nosync_assert(o3_assert(logits_assert(ahead_assert(text, ' == '), ' == '), ' == '), ' == ')
    # dnr: a replay arm doesn't read the workload, and index 2-9 would have named band 2's; one file for every index, the one sm has.
    text = once(text, 'if [[ $_pair == [01] ]]; then _workload=', 'if [[ $_pair == [0-9] ]]; then _workload=')
    text = once(text, OLD_CORE, CORE)
    text = once(text, ARM_TEXT[0], ARM_TEXT[1])
    text = once(text, HEADER_ANCHOR, HEADER_NOTE + HEADER_ANCHOR)
    text = renames(text)
    return once(text, MODULES_AFTER, MODULES)


def settings(text):
    text = split(text, 'S5_REQUESTED_DECODE_SMS=${S5_REQUESTED_DECODE_SMS:-%d}', 'S5_REQUESTED_DECODE_SMS=${S5_REQUESTED_DECODE_SMS:-%d}')
    text = split(text, "[[ $S5_REQUESTED_DECODE_SMS == %d ]] || {\n  echo 'decode split remains declared %d' >&2; return 2; }",
                 "[[ $S5_REQUESTED_DECODE_SMS == %d ]] || {\n  echo 'decode split remains declared %d' >&2; return 2; }")
    text = split(text, 'FLASH_S5_DECODE_SMS=%d ', 'FLASH_S5_DECODE_SMS=%d ')
    text = once(text, OLD_LAST_SWITCH, OLD_LAST_SWITCH + table.CHUNK_ENV + ' ' + table.PREPARE_AHEAD_ENV + ' ' + table.LOGITS_ENV + ' ' + table.O3_ENV + ' ' + table.AHEAD_NOSYNC_ENV + ' ')
    text = once(text, 'FLASH_S5_APPEND_YIELD_MAX_NEW=2048 ', 'FLASH_S5_APPEND_YIELD_MAX_NEW=%d ' % table.LAUNCH['max_new'])
    return renames(text)


def declare(text):
    text = split(text, "assert env.get('FLASH_S5_DECODE_SMS')=='%d', 'decode split differs'", "assert env.get('FLASH_S5_DECODE_SMS')=='%d', 'decode split differs'")
    text = once(text, '  for name in e3_native_startup_grade.py ', '  for name in e3_native_startup_grade.py e3_split_startup_grade.py ')
    text = yield_bound(chunk_assert(text, '=='), '==')
    text = nosync_assert(o3_assert(logits_assert(ahead_assert(text, '=='), '=='), '=='), '==')
    text = renames(text)
    return once(text, MODULES_AFTER, MODULES)


def derive():
    out = {}
    for suffix, edit in (('.draft.sh', draft), ('_settings.sh', settings), ('_declare.sh', declare)):
        out['harness/runs/' + NEW + suffix] = edit(source(suffix))
    return out


def final_body():
    """The arm loop through the read, as it stands in the draft."""
    text = derive()['harness/runs/' + NEW + '.draft.sh']
    start = text.index('alpha_next_arm() {')
    stop = text.index('\n', text.index('read_step compose5-read', start))
    return text[start:stop + 1]


def table_text():
    import compose5_gate as gate
    lines = ['q18a-parity-n3s24 arm table (harness/compose5_arms.py); entry limits relative to the run start, never a clock',
             '%-3s %-16s %-10s %-6s %-5s %s' % ('#', 'label', 'kind', 'yield', 'ple', 'enters while SECONDS <=')]
    for k, row in enumerate(gate.rows(), 1):
        lines.append('%-3d %-16s %-10s %-6s %-5s %s %d s' % (k, row['label'], row.get('kind', 'decode'), {True: 'on', False: 'off'}.get(row.get('yield_on'), '-'), {True: 'on', False: 'off'}.get(row.get('ple_private'), '-'), row['entry_limit'], gate.ENTRY_LIMIT_S[row['entry_limit']]))
    exp = gate.EXPECTED_RUNTIME_S
    e, slow = exp['nominal'], exp['slower']
    lines.append('box budget %d s, cleanup included; worst case of the caps %d s' % (gate.BOX_BUDGET_S, gate.WORST_CASE_SUM_S))
    lines.append('expected: %d arms entering, last arm ends +%d s, about %.1f min with startup and the end; smallest entry margin %d s' % (e['arms_entering'], e['last_arm_ends_s'], e['total_min'], e['min_margin_s']))
    lines.append('at %.1fx the walls: %d arms entering, about %.1f min; not run: %s' % (gate.SLOWER, slow['arms_entering'], slow['total_min'], ', '.join(slow['not_run']) or 'none'))
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--check', action='store_true')
    p.add_argument('--table', action='store_true', help='print the arm table with its entry limits; write nothing')
    a = p.parse_args(argv)
    if a.table:
        print(table_text())
        return
    for name, text in derive().items():
        path = ROOT / name
        if a.check:
            assert path.read_text() == text, name
        else:
            path.write_text(text)
            path.chmod(0o664)
    print('parity-n3u queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
