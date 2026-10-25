#!/usr/bin/env python3
"""Grade a real-route canary run (harness/flash_canary.sh) strictly.

OUTDIR holds expected.txt ("round session code note workdir" per line) and r<round>-s<session>.out (the session's
whole stdout/stderr plus a final "rc=N" line). With TRANSCRIPTS (the Claude Code projects dir) each session's
transcript is read for its tool calls and per-request cache usage.

Per session, first match wins:
  LEAK      any other session's code or note appears ANYWHERE in the output or the transcript's assistant text
  MISSING   no .out file, or no rc= line
  NO_TRANSCRIPT  TRANSCRIPTS given but this session has no transcript written during this run (newer than
            expected.txt) that contains its own code
  RC_FAIL   rc != 0 (e.g. 124 = timeout)
  WRONG     the final CODE= line lacks the own code or note
  TOOL_SKIP correct final line but the requested tool chain (Read + 2 Bash) did not run (needs TRANSCRIPTS)
  OK
Run verdict: FAIL on any LEAK / MISSING / NO_TRANSCRIPT / RC_FAIL, duplicate identifiers, or expected != graded count; WRONG and
TOOL_SKIP are model/tool-compliance failures: reported, counted, and make the verdict INCONCLUSIVE (never PASS).
usage: canary_grade.py OUTDIR [TRANSCRIPTS] [OUT.json]   exit 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE
"""
import glob, json, os, re, sys


def transcript_facts(transcripts, workdir, not_before=0.0):
    proj = os.path.join(transcripts, re.sub(r'[/.]', '-', workdir))
    files = sorted((f for f in glob.glob(os.path.join(proj, '*.jsonl')) if os.path.getmtime(f) >= not_before),
                   key=os.path.getmtime)
    if not files:
        return None
    tools, cached, texts, seen = [], [], [], set()
    for line in open(files[-1]):
        d = json.loads(line)
        m = d.get('message') or {}
        if d.get('type') != 'assistant':
            continue
        content = m.get('content')
        for c in content if isinstance(content, list) else []:
            if c.get('type') == 'tool_use':
                tools.append(c.get('name'))
            elif c.get('type') == 'text':
                texts.append(c.get('text', ''))
        u = m.get('usage')
        if u and m.get('id') not in seen:
            seen.add(m.get('id'))
            cached.append(u.get('cache_read_input_tokens') or 0)
    return {'tools': tools, 'cached': cached, 'requests': len(cached), 'request_ids': sorted(seen), 'text': '\n'.join(texts)}


def grade(outdir, transcripts=None):
    exp = os.path.join(outdir, 'expected.txt')
    rows = [l.split() for l in open(exp) if l.strip()]
    marker = os.path.join(outdir, "run-start.json")
    run_start = json.load(open(marker))["started_wall"] if os.path.exists(marker) else os.path.getmtime(exp)
    fails, sessions = [], []
    if not rows:
        fails.append("no expected sessions")
    if len({(r[0], r[1]) for r in rows}) != len(rows):
        fails.append("duplicate round/session")
    codes = [r[2] for r in rows]
    notes = [r[3] for r in rows]
    if len(set(codes)) != len(codes) or len(set(notes)) != len(notes):
        fails.append('duplicate session codes or notes in expected.txt')
    for rd, i, code, note, wd in rows:
        rec: dict = {'round': rd, 'session': i, 'code': code, 'note': note}
        path = os.path.join(outdir, f'r{rd}-s{i}.out')
        out = open(path, errors='replace').read() if os.path.exists(path) else None
        facts = transcript_facts(transcripts, wd, run_start) if transcripts else None
        if facts:
            rec.update(tools=facts['tools'], cached=facts['cached'], request_ids=facts['request_ids'])
        hay = (out or '') + '\n' + (facts['text'] if facts else '')
        foreign = [x for x in codes + notes if x not in (code, note) and x in hay]
        rcm = re.findall(r'^rc=(\d+)$', out or '', re.M)
        final = (re.findall(r'^.*CODE=.*$', out or '', re.M) or [''])[-1]
        if foreign:
            rec['verdict'] = 'LEAK'; rec['foreign'] = foreign
        elif out is None or not rcm:
            rec['verdict'] = 'MISSING'
        elif transcripts and (facts is None or code not in facts['text']):
            rec['verdict'] = 'NO_TRANSCRIPT'
        elif rcm[-1] != '0':
            rec['verdict'] = 'RC_FAIL'; rec['rc'] = int(rcm[-1])
        elif code not in final or note not in final:
            rec['verdict'] = 'WRONG'; rec['final'] = final
        elif facts is not None and (facts['tools'].count('Read') < 1 or facts['tools'].count('Bash') < 2):
            rec['verdict'] = 'TOOL_SKIP'
        else:
            rec['verdict'] = 'OK'
        sessions.append(rec)
    counts = {}
    for s in sessions:
        counts[s['verdict']] = counts.get(s['verdict'], 0) + 1
    if len(sessions) != len(rows):
        fails.append(f'graded {len(sessions)} of {len(rows)}')
    hard = sum(counts.get(k, 0) for k in ('LEAK', 'MISSING', 'NO_TRANSCRIPT', 'RC_FAIL'))
    soft = sum(counts.get(k, 0) for k in ('WRONG', 'TOOL_SKIP'))
    verdict = 'FAIL' if hard or fails else ('INCONCLUSIVE' if soft else 'PASS')
    partial = sum(1 for s in sessions for c in s.get('cached', []) if c and c % 4048)
    return {'verdict': verdict, 'counts': counts, 'expected': len(rows), 'fails': fails,
            'partial_hits': partial if transcripts else None, 'sessions': sessions}


if __name__ == '__main__':
    res = grade(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None)
    if len(sys.argv) > 3:
        json.dump(res, open(sys.argv[3], 'w'), indent=1)
    for s in res['sessions']:
        if s['verdict'] != 'OK':
            print(s['verdict'], {k: v for k, v in s.items() if k != 'verdict'})
    print(f"canary {res['verdict']}: {res['counts']} of {res['expected']}; partial hits {res['partial_hits']}; {res['fails']}")
    sys.exit({'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[res['verdict']])
