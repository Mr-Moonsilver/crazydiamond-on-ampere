"""Lossless observer archive keeps the primary VM probe small; no device calls."""
import hashlib
import json
import lzma
from pathlib import Path


def archive_observer(record,out):
    section=record.get('sections',{}).get('local_overlap',{})
    value=section.get('observer_end')
    if value is None:return
    path=Path(out)/'native31-observer-end.json.xz'
    with lzma.open(path,'xt',preset=3) as stream:json.dump(value,stream,sort_keys=True,default=str)
    grade=value.get('composed_placement') or {}
    summary={key:value[key] for key in ('rank','route','route_generation','observer_stopped','receipt','ack') if key in value}
    summary['composed_placement']={key:grade.get(key) for key in ('schema','verdict','row4','raw','raw_unchanged','missing','notes')}
    summary['composed_placement']['failure_count']=len(grade.get('failures') or [])
    summary['composed_placement']['failures']=(grade.get('failures') or [])[:32]
    baseline=grade.get('baseline') or {}
    summary['composed_placement']['baseline']={key:baseline.get(key) for key in ('verdict','conservation_proved','missing')}
    summary['full_evidence_archive']=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),compressed_bytes=path.stat().st_size,lossless=True,scope='complete unchanged observer_end; primary probe preserves original placement verdict and first 32 failures; full detail in archive')
    section['observer_end']=summary
