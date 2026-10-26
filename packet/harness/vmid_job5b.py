#!/usr/bin/env python3
"""Fresh queue identity for immutable progress snapshots; workload unchanged."""
import argparse
from pathlib import Path
import sys
import vmid_job5 as job
import vmid_support as support


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('probe','monitor','finalize'),required=True)
    p.add_argument('--modules',type=Path);p.add_argument('--native',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--process-rc',type=int,default=2)
    a=p.parse_args()
    if a.mode=='monitor':return support.monitor(a.out,'flashnext-job5b')
    if a.mode=='finalize':return support.finalize(a.out,a.process_rc,job.grade,'job5b')
    return job.probe(a.modules,a.native,a.out)

if __name__=='__main__':sys.exit(main())
