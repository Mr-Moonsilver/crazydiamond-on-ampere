"""Host-only bounded counterfactual state machine; no numerical repair."""
import copy


class ProtocolRefused(RuntimeError):pass


class Arm:
    def __init__(self,name,epoch,ranks=(0,1,2,3)):
        if name not in ('S','O','I') or type(epoch) is not int or epoch<1 or tuple(ranks)!=(0,1,2,3):
            raise ProtocolRefused('native34-arm-identity')
        self.name=name;self.epoch=epoch;self.ranks=tuple(ranks)
        self.stage='ARMED';self.checkpoint=None;self.sequence=None
        self.stock_ticks=0;self.prefix=None;self.records=[]

    def admission(self,admission):
        tx=admission.transaction
        if (tx.engine_epoch!=self.epoch or tx.txn_seq!=100
                or tx.chunk_start!=0 or tx.chunk_end!=1024):
            raise ProtocolRefused('native34-failing-admission-state-mismatch')
        self.prefix=dict(transaction=tx.txn_seq,chunk_start=tx.chunk_start,
                         chunk_end=tx.chunk_end,layer_end=4)
        self.stage='BEFORE_DRAIN' if self.name=='S' else 'PREFIX'

    def accepted_prefix(self,command,accepted):
        d=command.descriptor
        if self.stage!='PREFIX' or getattr(d.kind,'value',d.kind)!='layer' or d.stage!=0 or d.layer_start!=3:
            return
        targets=[row for row in accepted if row.worker_rank in (0,1)]
        if (len(targets)!=2 or any(not row.accepted or row.pending for row in targets)):
            return
        self.prefix.update(op_seq=command.op_seq,layer_start=d.layer_start)
        self.stage='AFTER_DRAIN' if self.name=='S' else 'POST_DECODE'

    def needed_checkpoint(self):
        if self.sequence is not None:return None
        return {'BEFORE_DRAIN':'before-prefix','AFTER_DRAIN':'after-prefix',
                'POST_DRAIN':'post-decode'}.get(self.stage)

    def submitted(self,sequence,phase):
        if self.sequence is not None or phase!=self.needed_checkpoint():
            raise ProtocolRefused('native34-checkpoint-submission-order')
        self.sequence=sequence;self.checkpoint=phase

    def completed(self,sequence,rows):
        if sequence!=self.sequence or sequence is None:
            raise ProtocolRefused('native34-checkpoint-reply-identity')
        if len(rows)!=4 or {row.get('worker_rank') for row in rows}!=set(self.ranks):
            raise ProtocolRefused('native34-checkpoint-rank-coverage')
        values=[]
        for row in rows:
            proof=(row.get('qualification') or {}).get('native34') or {}
            if (proof.get('phase')!=self.checkpoint or proof.get('arm')!=self.name
                    or proof.get('engine_epoch')!=self.epoch
                    or proof.get('drain_complete') is not True):
                raise ProtocolRefused('native34-checkpoint-drain-unproved')
            if (self.checkpoint!='before-prefix' and row['worker_rank'] in (0,1)
                    and proof.get('prefix_complete') is not True):
                raise ProtocolRefused('native34-native-prefix-completion-unproved')
            if self.checkpoint=='post-decode' and proof.get('both_post_prefix_cohorts_complete') is not True:
                raise ProtocolRefused('native34-exact-decode-cohort-completion-unproved')
            values.append(copy.deepcopy(proof))
        self.records.append(dict(phase=self.checkpoint,workers=values))
        self.stage={'before-prefix':'PREFIX','after-prefix':'POST_DECODE','post-decode':'DONE'}[self.checkpoint]
        self.sequence=None;self.checkpoint=None

    def ordinary_tick(self):
        if self.stage=='POST_DECODE':
            self.stock_ticks+=1
            if self.stock_ticks>=2:self.stage='POST_DRAIN'

    def blocks_native(self):return self.stage not in ('ARMED','PREFIX')
    def blocks_stock(self):return self.stage not in ('ARMED','PREFIX','POST_DECODE') or (self.name=='S' and self.stage=='PREFIX')

    def record(self):
        return dict(arm=self.name,engine_epoch=self.epoch,stage=self.stage,
            prefix=copy.deepcopy(self.prefix),checkpoints=copy.deepcopy(self.records),
            stock_ticks_after_prefix=self.stock_ticks,terminal=self.stage=='DONE',
            scope='bounded causal diagnostic; no stability/performance qualification')
