"""Prospective smaller discriminator, derived from frozen fit-b; never edits the parent."""
import ast
from build_vision_serve import once


def span(text, start, end, replacement=''):
    if text.count(start)!=1 or text.count(end)!=1:raise ValueError('headroom protocol anchor differs')
    low,high=text.index(start),text.index(end)
    if low>=high:raise ValueError('headroom protocol anchor order')
    return text[:low]+replacement+text[high:]


def client(text):
    text=once(text,"NAME='q18a-vision-fit-8x200k-b'", "NAME='q18a-g4-headroom-a'")
    text=once(text,'PREFIXES=(60000,120000,160000,190320)','PREFIXES=(190320,)')
    text=span(text,"        first=prepare(args,feature,4096",'        import n3j2_packet\n')
    text=span(text,'        cancel=prepare(args,feature,60000','        previous=None\n')
    text=span(text,"        record['tail'],tail_streams=cohort(","        record.update(complete=True,verdict='FIT'",'')
    text=text.replace("reason='fixed-image-growth-tail-and-cold-under-seven-complete'", "reason='headroom-discriminator-one-eight-member-stage-complete'")
    text=span(text,"    tail=record.get('tail') or {}\n", "    if record.get('calibration_lease'", '''    for stage in record.get('stages',[]):
        if type(stage.get('native_admissions')) is not int or stage['native_admissions']<=0:reasons.append('no-native-execution')
        for member in stage.get('conversations',[]):
            if (member.get('prompt_verified') is not True or member.get('image_rows')!=1024
                    or member.get('error') or member.get('output_tokens')!=member.get('output_cap')):reasons.append('MM-stream-unproven')
''')
    text=span(text,"    cold=record.get('cold') or {}\n", "    mm=record.get('memory',{}).get('mm_peaks',{})\n")
    text=once(text,"    record=dict(schema='vision-fit-8x200k-client/1',role='e6',plan=plan(),complete=False,stages=[],cold=None,",
        "    record=dict(schema='vision-fit-8x200k-client/1',scope='headroom-discriminator-only-not-full-G4',role='e6',plan=plan(),complete=False,stages=[],cold=None,")
    tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='plan')
    old=ast.get_source_segment(text,node)
    new='''def plan():
    return dict(schema='g4-headroom-plan/1',name=NAME,scope='headroom discriminator, not full G4 or adoption',
        agents=8,max_model_len=200000,prefixes=[190320],output_cap=8192,
        warmup='unchanged serve 302 + 70903; FULL semantic lease retained',
        floor_mib=40,target_headroom_mib=250,kv_bytes=2150000000,rowcache=[8,6144],
        native_workload_cap_s=2700,outer_cap_s=6000,owned_cleanup_grace_s=330,
        omitted=['reuse','cancel','staged-growth','tail-image-stage','cold-beside-seven'],
        unchanged=['cache binding','keyed401','native coverage and registered exceptions','end identity','owned stop','death scan'])'''
    text=once(text,old,new)
    text=once(text,"result['semantic']=record.get('semantic');result['tail']=tail","result['semantic']=record.get('semantic');result['tail']=None")
    ast.parse(text)
    return text


def base_reader(text):
    text=span(text,'    cold = record.get(\'cold\') or {}\n',"    if record.get('errors400')",'')
    text=once(text,"        if (any(type(s.get('native_admissions')) is not int or s['native_admissions']<=0 for s in stages)\n                or type(cold.get('native_admissions')) is not int or cold['native_admissions']<=0):", "        if any(type(s.get('native_admissions')) is not int or s['native_admissions']<=0 for s in stages):")
    ast.parse(text)
    return text
