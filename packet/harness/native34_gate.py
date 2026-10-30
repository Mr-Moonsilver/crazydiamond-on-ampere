"""Fresh weekend window; inherited native30 host-budget gate stays intact."""
import datetime

from native30_gate import (ACTIVE_AND_CLEANUP,SNAPSHOT_CAP,MEASURED_SNAPSHOT_MAX,
                           TRANSIENT_MARGIN,evaluate,host_receipt)

NOT_BEFORE=datetime.datetime(2026,10,30,22,1,tzinfo=datetime.timezone.utc).timestamp()
HARD_END=datetime.datetime(2026,11,2,6,tzinfo=datetime.timezone.utc).timestamp()


def window_ok(now):
    return NOT_BEFORE<=now and now+ACTIVE_AND_CLEANUP<=HARD_END
