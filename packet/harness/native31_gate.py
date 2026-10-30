"""Native30-c host budget and operator window; no engine/device operations."""
import datetime
from pathlib import Path
import time

SNAPSHOT_CAP = 512 * 1024**2
MEASURED_SNAPSHOT_MAX = 52852723
TRANSIENT_MARGIN = 5 * 1024**3
NOT_BEFORE = datetime.datetime(2026,10,30,17,30,tzinfo=datetime.timezone.utc).timestamp()
HARD_END = datetime.datetime(2026,10,30,23,tzinfo=datetime.timezone.utc).timestamp()
ACTIVE_AND_CLEANUP = 2100 + 420


def window_ok(now):
    return NOT_BEFORE <= now and now + ACTIVE_AND_CLEANUP <= HARD_END


def evaluate(cap, mem_available, cgroup):
    if type(cap) is not int or cap != SNAPSHOT_CAP:
        raise ValueError('native30-c-snapshot-cap-must-be-512MiB')
    if type(mem_available) is not int or mem_available < 0:
        raise ValueError('invalid-MemAvailable')
    available = mem_available
    if all(str(cgroup.get(n,'')).isdigit() for n in ('memory.current','memory.max')):
        available = min(available,int(cgroup['memory.max'])-int(cgroup['memory.current']))
    required = 4*cap + TRANSIENT_MARGIN
    return dict(schema=2,mem_available_bytes=mem_available,available_bytes=available,
        cgroup=cgroup,snapshot_cap_per_worker=cap,required_bytes=required,
        measured_snapshot_max_per_worker=MEASURED_SNAPSHOT_MAX,
        snapshot_margin_per_worker=cap-MEASURED_SNAPSHOT_MAX,
        observer_correlation_report_transient_margin_bytes=TRANSIENT_MARGIN,
        verdict='PASS' if available>=required else 'INCONCLUSIVE',
        scope='point sample; measured historical snapshots plus bounded growth allowance; unchanged transient margin; not a host peak guarantee')


def host_receipt(cap):
    info={line.split(':')[0]:int(line.split()[1])*1024
          for line in Path('/proc/meminfo').read_text().splitlines()
          if len(line.split())>=2 and line.split()[1].isdigit()}
    cgroup={}
    for name in ('memory.current','memory.max'):
        path=Path('/sys/fs/cgroup')/name
        if path.exists():cgroup[name]=path.read_text().strip()
    return dict(evaluate(cap,info['MemAvailable'],cgroup),wall_s=time.time())
