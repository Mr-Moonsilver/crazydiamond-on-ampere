"""Non-gating full/split launch configuration comparison; no payload or handles emitted."""
FIELDS=['cacheConfig.both', 'sharedMemoryConfig', 'registersPerThread', 'partitionedGlobalCacheRequested', 'partitionedGlobalCacheExecuted', 'gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ', 'staticSharedMemory', 'dynamicSharedMemory', 'localMemoryPerThread', 'localMemoryTotal', 'launchType', 'isSharedMemoryCarveoutRequested', 'sharedMemoryCarveoutRequested', 'sharedMemoryExecuted', 'shmemLimitConfig', 'channelID', 'channelType', 'clusterX', 'clusterY', 'clusterZ', 'clusterSchedulingPolicy', 'localMemoryTotal_v2', 'maxPotentialClusterSize', 'maxActiveClusters', 'isDeviceLaunched', 'accessPolicyWindow_present', 'accessPolicyWindow_base_private', 'accessPolicyWindow_num_bytes', 'accessPolicyWindow_hitRatio_bits', 'accessPolicyWindow_hitProp', 'accessPolicyWindow_missProp']
import hashlib
import json

EXCLUDED=('kind (fixed concurrent-kernel record type)','start/end/completed/queued/submitted timestamps',
          'deviceId/contextId/streamId/correlationId/gridId/graphId/graphNodeId provenance',
          'reserved0/padding/padding3 undefined storage')

def inventory(session,probe,full):
    from graph_router import descriptor_name
    bank=probe.bank;manager=bank.stock if full else bank.manager
    result={}
    for descriptor,local in bank.pairs:
        graph=manager.graphs[descriptor]
        result[descriptor_name(local)]=session._report_call(session.binding._lib.s5ro_graph_ids,
                                                           (graph.raw_cuda_graph(),))
    return result


def _normalise(report,graphs):
    if report.get('complete') is not True:raise ValueError('native activity loss or outstanding buffer')
    graph_parent={};node_parent={}
    for g,og,n,on in report['aliases']:
        for parents,child,parent in ((graph_parent,g,og),(node_parent,n,on)):
            if not child or not parent or child==parent:continue
            if child in parents and parents[child]!=parent:raise ValueError('ambiguous native alias')
            parents[child]=parent
    def root(value,parents):
        seen=set()
        while value in parents:
            if value in seen:raise ValueError('native alias cycle')
            seen.add(value);value=parents[value]
        return value
    expected={};counts={}
    for label,graph in graphs.items():
        ids=sorted(graph['nodes'])
        if not ids or len(set(ids))!=len(ids):raise ValueError('static kernel inventory absent')
        gid=root(graph['graph'],graph_parent)
        for i,node in enumerate(ids):
            key=(gid,root(node,node_parent))
            if key in expected:raise ValueError('static node alias')
            expected[key]=(label,i)
        counts[label]=len(ids)
    result={}
    for row in report['nodes']:
        key=(root(row['graph'],graph_parent),root(row['node'],node_parent))
        location=expected.get(key)
        if location is None:continue  # non-bank sampler/eager work, never bank proof
        if (row.get('complete') is not True or type(row.get('name')) is not str or not row['name']
                or type(row.get('fields')) is not list or len(row['fields'])!=len(FIELDS)
                or any(type(x) is not int for x in row['fields'])):
            raise ValueError('launch fields or interval incomplete')
        if location in result:raise ValueError('duplicate native alias records')
        result[location]=row
    labels={label for label,_ in result}
    if not labels:raise ValueError('no exercised bank graph')
    for label in labels:
        rows=[row for (l,_),row in result.items() if l==label]
        if len(rows)!=counts[label] or len({r['records'] for r in rows})!=1 or rows[0]['records']<=0:
            raise ValueError('per-node launch coverage incomplete')
    return result


def compare(full,split):
    result={'schema':1,'verdict':'INCONCLUSIVE','gates_w1':False,'fields':FIELDS,
            'excluded_fields':EXCLUDED,'kernels':[],'missing':[],
            'scope':'exercised same-descriptor graphs; node ordinal by native capture ID; private pointer values compared but never emitted'}
    if full is None or split is None:
        result['missing']=['full and split launch samples required'];return result
    try:
        a=_normalise(*full);b=_normalise(*split)
        if set(a)!=set(b):raise ValueError('full/split exercised node inventories differ')
        differs=False
        for key in sorted(a):
            x,y=a[key],b[key]
            changed=[field for i,field in enumerate(FIELDS) if x['fields'][i]!=y['fields'][i]]
            if x['name']!=y['name']:changed.append('kernel_symbol')
            if x['stable'] is not True or y['stable'] is not True:changed.append('within_route_launch_variation')
            differs|=bool(changed)
            digest=lambda row:hashlib.sha256(json.dumps([row['name'],row['fields']],separators=(',',':')).encode()).hexdigest()
            xf=list(x['fields']);yf=list(y['fields']);private=FIELDS.index('accessPolicyWindow_base_private')
            xf[private]='private pointer';yf[private]='same pointer' if x['fields'][private]==y['fields'][private] else 'different pointer'
            result['kernels'].append({'full_fields':xf,'split_fields':yf,'descriptor':key[0],'node_ordinal':key[1],
                'full_kernel':x['name'],'split_kernel':y['name'],'differing_fields':changed,
                'full_config_sha256':digest(x),'split_config_sha256':digest(y)})
        result['verdict']='DIFFER' if differs else 'PASS'
    except (ValueError,KeyError,TypeError) as exc:
        result['missing']=[str(exc) if isinstance(exc,ValueError) else 'invalid launch sample']
    return result
