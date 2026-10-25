"""Offline declaration tests: real reviewed launcher, inert Docker, fixture mounts."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import engine_identity as ident
import identity_declare as dec

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
TMP=ROOT.parent/'tmp'
IMAGE='sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
DUMMY='fixture-private-value'


class KeepBuffer(io.StringIO):
    def close(self): self.saved=self.getvalue();super().close()


class DeclareTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='declare-',dir=TMP));self.task=self.root/'task';self.task.mkdir()
        self.ovl=self.task/'prod/overlay-v16g';self.ovl.mkdir(parents=True)
        for folder,names in ((self.ovl,['qsa.py','ops_qsa.py','model_runner_v2.py']),
            (self.ovl/'core',['block_pool.py','single_type_kv_cache_manager.py','scheduler.py','kv_cache_utils.py','kv_cache_manager.py'])):
            folder.mkdir(exist_ok=True)
            for n in names: (folder/n).write_text('# fixture '+n+'\n')
            (folder/'SHA256SUMS').write_text(''.join(dec.sha(folder/n)+'  '+n+'\n' for n in names))
        model=self.task/'models/intel-w4a16';model.mkdir(parents=True);(model/'config.json').write_text('{}\n')
        (model/'weights.fixture').write_bytes(b'not real model weights')
        (self.task/'cache').mkdir();(self.task/'logs').mkdir()
        (self.task/'cache/mutable').write_text('1');(self.task/'logs/reconfig.log').write_text('mutable')
        self.defaults=self.root/'defaults'
        self.defaults.write_text('CORE_OVERLAY=1\nPP_LAYER_PARTITION=24,24\nMAX_SEQS=10\nGPU_KV_BYTES=2150000000\n'
                                 'HOST_KV_GIB_PER_RANK=33\nHF_TOKEN="'+DUMMY+'"\n')
        self.image={'Id':IMAGE,'RepoDigests':['repo/model@'+IMAGE],
                    'Config':{'Entrypoint':['vllm','serve'],'Cmd':None,'Volumes':None,
                              'Env':['PATH=/image/bin','NCCL_DEBUG=INFO','DEBIAN_FRONTEND=noninteractive']}}
        self.a=argparse.Namespace(defaults=str(self.defaults),setcfg=str(ROOT/'deploy/setcfg.py'),
           launcher=str(ROOT/'deploy/run-flash-prod.sh'),source_mirror=str(self.task),
           timeout=60,max_bytes=2**24,max_files=1000)
        self.policy=json.loads((HERE/'pp_release_mount_policy.json').read_text())

    def declare(self,verify=False):
        o=dict(dec.PP_SETTINGS)
        if verify:o['QSA_COW_VERIFY']='1'
        return dec.declare(self.a,o,self.image,self.policy,self.root/('verify' if verify else 'on'))[0]

    def test_real_launcher_renders_sources_and_entrypoint(self):
        before=self.defaults.read_bytes();d=self.declare()['identity']
        self.assertEqual(self.defaults.read_bytes(),before)
        self.assertEqual(d['command']['path'],'vllm');self.assertEqual(d['command']['args'][:2],['serve','/model'])
        self.assertEqual(d['env']['NCCL_DEBUG'],'WARN')
        self.assertEqual(d['env']['PATH'],'/image/bin')
        self.assertEqual(d['env']['VLLM_PP_LAYER_PARTITION'],'24,24')
        self.assertEqual(d['env']['QSA_ROWCACHE_REQS'],'10')
        self.assertEqual(d['defaults']['HF_TOKEN'],{'redacted_length':len(DUMMY)})
        self.assertNotIn(DUMMY,json.dumps(d))
        self.assertTrue(any(m['destination'].endswith('/core/kv_cache_utils.py') for m in d['mounts']))
        self.assertTrue(all(m['source'].startswith(dec.TASK) for m in d['mounts']))
        self.assertFalse((self.root/'on/run-argv.private.json').exists())

    def test_pair_only_verifier_diff(self):
        on,verify=self.declare(),self.declare(True);dec.assert_pair(on,verify)
        changed=copy.deepcopy(verify);changed['identity']['env']['QSA_ROWCACHE_REQS']='9'
        with self.assertRaises(ident.EvidenceError):dec.assert_pair(on,changed)
        changed=copy.deepcopy(on);changed['identity']['mounts']=[m for m in changed['identity']['mounts'] if not m['destination'].endswith('kv_cache_utils.py')]
        wrong=copy.deepcopy(changed)
        for sec in ('env','defaults'):wrong['identity'][sec]['QSA_COW_VERIFY']='1'
        with self.assertRaises(ident.EvidenceError):dec.assert_pair(changed,wrong)

    def test_planned_launcher_swap_is_hashed_not_ignored(self):
        d=self.declare()['identity']
        work=next(m for m in d['mounts'] if m['destination']=='/work')
        entry=dict(path='prod/run-flash-prod.sh',kind='file',bytes=1,sha256=dec.FOUND_SHA)
        work['content']['entries'].append(entry)
        dec.project_launcher(d,self.a.launcher)
        self.assertEqual(entry['sha256'],dec.LAUNCHER_SHA)
        self.assertEqual(entry['bytes'],Path(self.a.launcher).stat().st_size)
        entry['sha256']='0'*64
        with self.assertRaises(ident.EvidenceError):dec.project_launcher(d,self.a.launcher)

    def test_removal_commands_are_inert(self):
        flag=self.task/'rowcache.off';flag.write_text('keep this file')
        on=self.declare();verify=self.declare(True)
        self.assertEqual(flag.read_text(),'keep this file')
        with self.assertRaises(ident.EvidenceError):dec.assert_pair(on,verify)

    def test_setcfg_matches_actual_source_with_no_real_open(self):
        text='# untouched\nA=old\nQSA_COW_VERIFY=1\nEXTRA_ARGS="old"\n'
        over={'A':'quote" slash\\ dollar$ backtick`','QSA_COW_VERIFY':'-','EXTRA_ARGS':'--flag value','NEW':'v'}
        writer=KeepBuffer()
        def fake_open(path,mode='r'):
            if path=='/etc/default/vllm-flash' and mode=='r':return io.StringIO(text)
            if path=='/etc/default/vllm-flash.next' and mode=='w':return writer
            raise AssertionError('unexpected path')
        with patch('builtins.open',fake_open),patch.object(sys,'argv',['setcfg.py']+[k+'='+v for k,v in over.items()]),contextlib.redirect_stdout(io.StringIO()):
            exec(compile((ROOT/'deploy/setcfg.py').read_text(),'setcfg.py','exec'),{})
        self.assertEqual(dec.apply_setcfg(text,over,ROOT/'deploy/setcfg.py'),writer.getvalue())
        vals=dec.raw_defaults(dec.apply_setcfg(text,over,ROOT/'deploy/setcfg.py'))
        self.assertEqual(vals['A'],over['A']);self.assertNotIn('QSA_COW_VERIFY',vals)

    def test_unreviewed_source_and_shell_injection_rejected(self):
        source=self.root/'modified-launcher';source.write_text((ROOT/'deploy/run-flash-prod.sh').read_text()+'\ntrue\n')
        self.a.launcher=str(source)
        with self.assertRaises(ident.EvidenceError):self.declare()
        self.a.launcher=str(ROOT/'deploy/run-flash-prod.sh')
        self.defaults.write_text('BASH_ENV=/not-allowed\n')
        with self.assertRaises(ident.EvidenceError):self.declare()

    def test_settings_and_policy_path_escape_rejected(self):
        for words in (['A=1','A=2'],['bad-key=x'],['missing']):
            with self.assertRaises(ident.EvidenceError):dec.settings(words)
        self.policy['/work']['watch']=['../escape']
        with self.assertRaises(ident.EvidenceError):self.declare()

    def test_host_content_hash_not_metadata_only(self):
        d=self.declare()['identity'];model=next(m for m in d['mounts'] if m['destination']=='/model')
        entry=next(x for x in model['content']['entries'] if x['path']=='weights.fixture')
        self.assertEqual(entry['sha256'],hashlib.sha256(b'not real model weights').hexdigest())
        self.a.max_bytes=1
        with self.assertRaises(ident.EvidenceError):self.declare(True)

    def test_image_metadata_can_be_supplied_offline(self):
        path=self.root/'image.json';path.write_text(json.dumps([self.image]))
        idfile=self.root/'image.txt';idfile.write_text(IMAGE)
        args=argparse.Namespace(image_file=str(idfile),image_inspect=str(path),timeout=1)
        with patch('identity_declare.identity.run',side_effect=AssertionError('no Docker allowed')):
            self.assertEqual(dec.load_image(args)['Id'],IMAGE)
        args.image_inspect=None
        with patch('identity_declare.identity.run',return_value=[self.image]) as run:
            self.assertEqual(dec.load_image(args)['Id'],IMAGE)
            run.assert_called_once_with(['docker','image','inspect',IMAGE],1)

    def test_unknown_environment_stays_fail_closed(self):
        self.image['Config']['Env'].append('UNREVIEWED_TUNING=value')
        with self.assertRaises(ident.EvidenceError):self.declare()

    def test_cli_stores_no_raw_secrets_and_never_calls_docker(self):
        path=self.root/'image.json';path.write_text(json.dumps([self.image]))
        idfile=self.root/'image.txt';idfile.write_text(IMAGE)
        policy=self.root/'policy.json';policy.write_text(json.dumps(self.policy))
        out=self.root/'requested.json'
        cmd=[sys.executable,str(HERE/'engine_identity.py'),'declare',str(out),'--defaults',str(self.defaults),
             '--launcher',self.a.launcher,'--setcfg',self.a.setcfg,'--image-file',str(idfile),
             '--image-inspect',str(path),'--mount-policy',str(policy),'--source-mirror',str(self.task),
             '--settings',*[k+'='+v for k,v in dec.PP_SETTINGS.items()]]
        p=subprocess.run(cmd,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertNotIn(DUMMY,p.stdout+p.stderr+out.read_text())
        self.assertEqual(out.stat().st_mode & 0o777,0o600)

if __name__=='__main__':unittest.main()
