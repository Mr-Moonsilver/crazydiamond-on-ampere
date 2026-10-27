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

    def test_stage_timer_zero_is_one_explicit_launcher_assignment(self):
        over=dict(dec.PP_SETTINGS,EXTRA_ENV='QSA_STAGE_TIMING=0 FLASH_PP_GROUPING=1 FLASH_POOL_STATS=200')
        result,_=dec.declare(self.a,over,self.image,self.policy,self.root/'timer-off')
        self.assertEqual(result['identity']['env']['QSA_STAGE_TIMING'],'0')
        argv=dec.render(self.a.launcher,dec.raw_defaults(dec.apply_setcfg(self.defaults.read_text(),over,self.a.setcfg)),
                        IMAGE,self.root/'timer-argv',self.task)
        self.assertEqual([s for s in argv if s.startswith('QSA_STAGE_TIMING=')],['QSA_STAGE_TIMING=0'])

    def test_duplicate_launcher_env_remains_inconclusive(self):
        # q16b's dense declaration failed here before any engine launch. Keep
        # strictness for identical as well as conflicting duplicate keys; do
        # not silently implement Docker last-wins in the declaration parser.
        path=self.root/'duplicate-image.json';path.write_text(json.dumps([self.image]))
        idfile=self.root/'duplicate-image.txt';idfile.write_text(IMAGE)
        policy=self.root/'duplicate-policy.json';policy.write_text(json.dumps(self.policy))
        for value in ('0','1'):
            overrides=dict(dec.PP_SETTINGS,EXTRA_ENV='VLLM_QSA_KVO_SPARSE_PREFILL='+value)
            argv=dec.render(self.a.launcher,dec.raw_defaults(dec.apply_setcfg(
                self.defaults.read_text(),overrides,self.a.setcfg)),IMAGE,
                self.root/('duplicate-render-'+value),self.task)
            self.assertEqual([v for v in argv if v.startswith('VLLM_QSA_KVO_SPARSE_PREFILL=')],
                             ['VLLM_QSA_KVO_SPARSE_PREFILL=1','VLLM_QSA_KVO_SPARSE_PREFILL='+value])
            with self.assertRaises(ident.EvidenceError):dec.docker_spec(argv,self.image,self.task)
            out=self.root/('duplicate-request-'+value+'.json')
            command=[sys.executable,str(HERE/'engine_identity.py'),'declare',str(out),
                '--defaults',str(self.defaults),'--launcher',self.a.launcher,'--setcfg',self.a.setcfg,
                '--image-file',str(idfile),'--image-inspect',str(path),'--mount-policy',str(policy),
                '--source-mirror',str(self.task),'--settings',*[k+'='+v for k,v in overrides.items()]]
            p=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(p.returncode,2,p.stdout+p.stderr)
            self.assertIn('identity declaration INCONCLUSIVE',p.stdout)
            self.assertFalse(out.exists())
            self.assertNotIn(DUMMY,p.stdout+p.stderr)

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

    def test_e2_pinned_launcher_mounts_and_projected_swap_without_host_write(self):
        sys.path.insert(0,str(ROOT/'patches/s5_engine'))
        import make_e2_arena as g
        launch=self.root/'e2-launcher';launch.write_text(g.launcher((ROOT/'deploy/run-flash-prod.sh').read_text()))
        self.assertEqual(dec.sha(launch),dec.E2_LAUNCHER_SHA)
        found=self.task/'prod/run-flash-prod.sh';found.write_bytes((ROOT/'deploy/run-flash-prod.sh').read_bytes())
        found_before=found.read_bytes()
        names=['e2/__init__.py',*[f'e2/{n}.py' for n in g.RUNTIME],
               'e2_sources/scheduler_output.py','e2_sources/block_table.py']
        with (self.ovl/'SHA256SUMS').open('a') as manifest:
            for name in names:
                p=self.ovl/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
                manifest.write(dec.sha(p)+'  '+name+'\n')
        self.a.launcher=str(launch)
        policy=json.loads((HERE/'e2_mount_policy.json').read_text())
        over=dict(dec.PP_SETTINGS,EXTRA_ENV='FLASH_E2_ARENA=1 FLASH_E2_DMA=1 FLASH_E2_VERIFY=0')
        d,_=dec.declare(self.a,over,self.image,policy,self.root/'e2-render',install_launcher=True)
        work=next(m for m in d['identity']['mounts'] if m['destination']=='/work')
        entry=next(x for x in work['content']['entries'] if x['path']=='prod/run-flash-prod.sh')
        self.assertEqual(entry['sha256'],dec.E2_LAUNCHER_SHA)
        self.assertEqual(found.read_bytes(),found_before)
        self.assertEqual({w['path']:w['kind'] for w in work['watch']}['e2-arena.off'],'absent')
        for suffix in ('/flash_e2','/core/sched/output.py','/worker/gpu/block_table.py'):
            self.assertTrue(any(m['destination'].endswith(suffix) for m in d['identity']['mounts']))
        self.assertEqual(d['identity']['env']['FLASH_E2_DMA'],'1')
        self.assertNotIn(DUMMY,json.dumps(d))

    def test_s5_pinned_launcher_declares_selectors_and_projected_swap(self):
        sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
        import make_e2_arena as e2g
        import make_s5_overlap as s5g
        launch = self.root / 's5-launcher'
        launch.write_text(s5g._s5_launcher(
            e2g.launcher((ROOT / 'deploy/run-flash-prod.sh').read_text())))
        self.assertEqual(dec.sha(launch), dec.S5_LAUNCHER_SHA)
        found = self.task / 'prod/run-flash-prod.sh'
        found.write_bytes((ROOT / 'deploy/run-flash-prod.sh').read_bytes())
        found_before = found.read_bytes()
        names = ['e3/__init__.py', 'e3/decode-probe-evidence.json',
                 *[f'e3/{name}.py' for name in s5g.E3_MODULES],
                 *[f'e3/{name}' for name in s5g.NATIVE_FILES],
                 'e3/native/build_green_helper.sh',
                 'e3_sources/executor.py', 'e3_sources/core.py',
                 'e3_sources/worker.py', 'e3_sources/cudagraph_utils.py',
                 'e3_sources/sampler.py', 'e3_sources/async_utils.py',
                 'e3_sources/model.py', 'e3_sources/api_routers.py']
        with (self.ovl / 'SHA256SUMS').open('a') as manifest:
            for name in names:
                p = self.ovl / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('# fixture ' + name + '\n')
                manifest.write(dec.sha(p) + '  ' + name + '\n')
        self.a.launcher = str(launch)
        policy = json.loads((HERE / 'e3_startup_mount_policy.json').read_text())
        from e3_private_auth import prepare
        private = self.root / 'auth'; private.mkdir(mode=0o700)
        envfile = private / 'api.env'; prepare(envfile, {'VLLM_API_KEY': DUMMY})
        over = dict(dec.PP_SETTINGS, S5='1', CORE_OVERLAY='1', FLASH_S5_AUTH_ENV_FILE=str(envfile),
                    EXTRA_ENV=('QSA_STAGE_TIMING=1 FLASH_PP_GROUPING=1 FLASH_POOL_STATS=200 '
                               'FLASH_S5_OVERLAP=0 FLASH_S5_DECODE_ONLY_PROBE=1 '
                               'FLASH_S5_DECODE_SMS=44 FLASH_S5_GREEN_LIB_SHA256=' + 'a' * 64))
        d, _ = dec.declare(self.a, over, self.image, policy,
                           self.root / 's5-render', install_launcher=True)
        identity = d['identity']
        work = next(m for m in identity['mounts'] if m['destination'] == '/work')
        entry = next(x for x in work['content']['entries']
                     if x['path'] == 'prod/run-flash-prod.sh')
        self.assertEqual(entry['sha256'], dec.S5_LAUNCHER_SHA)
        self.assertEqual(found.read_bytes(), found_before)
        self.assertEqual(identity['defaults']['S5'], '1')
        self.assertEqual(identity['env']['FLASH_S5_DECODE_ONLY_PROBE'], '1')
        self.assertEqual(identity['env']['FLASH_S5_DECODE_SMS'], '44')
        self.assertEqual(identity['env']['VLLM_API_KEY'], {'redacted_length': len(DUMMY)})
        raw_argv=dec.render(launch, dec.raw_defaults(dec.apply_setcfg(self.defaults.read_text(),over,self.a.setcfg)), IMAGE,self.root/'s5-argv',self.task)
        self.assertIn('--env-file',raw_argv)
        self.assertEqual(raw_argv[raw_argv.index('--env-file')+1],str(envfile))
        self.assertNotIn(DUMMY,json.dumps(raw_argv))
        self.assertEqual(identity['env']['FLASH_S5_GREEN_LIB_SHA256'], 'a' * 64)
        # The exact reviewed generated launcher mounts the split-bank package
        # and the source transforms; no second E2 or S5 mount is fabricated.
        destinations = {m['destination'] for m in identity['mounts']}
        self.assertTrue(any(p.endswith('/vllm/v1/flash_e3') for p in destinations))
        for suffix in ('/vllm/v1/engine/core.py', '/vllm/v1/worker/gpu_worker.py',
                       '/vllm/v1/worker/gpu/cudagraph_utils.py',
                       '/vllm/v1/executor/multiproc_executor.py',
                       '/vllm/entrypoints/launchers/api_server/routers.py'):
            self.assertTrue(any(p.endswith(suffix) for p in destinations), suffix)
        self.assertNotIn(DUMMY, json.dumps(d))
        # q18a-native1 stopped here in production before any GPU work: its
        # public target label was absent from identity's exact vocabulary.
        # Source the ACTUAL old and repaired settings, render the trusted
        # launcher, hash fixture mounts and run no_unknown/validate_identity.
        # No identity executable/success result is stubbed in this regression.
        for attempt in ('q18a_native1', 'q18a_native2'):
            script = ('S5_OVERLAY=$2 S5_AUTH_ENV_FILE=$3 '
                      'S5_GREEN_LIB_SHA256=$4 S5_REPLAY_LIB_SHA256=$5 '
                      'Q18A_TARGET_LABEL=s5-overlap-80-80; '
                      'source "$1"; printf "%s\\0" "${S5_SETTINGS[@]}"')
            rendered = subprocess.run(
                ['bash', '-uc', script, 'fixture',
                 str(HERE/'runs'/(attempt+'_settings.sh')),
                 dec.TASK+'/prod/overlay-v16g', str(envfile), 'a'*64, 'b'*64],
                capture_output=True, check=True)
            overrides = dec.settings(rendered.stdout.decode().rstrip('\0').split('\0'))
            actual, _ = dec.declare(self.a, overrides, self.image, policy,
                                    self.root/(attempt+'-actual-settings'), install_launcher=True)
            self.assertEqual(actual['identity']['env']['Q18A_TARGET_LABEL'], 's5-overlap-80-80')
            self.assertEqual(actual['identity']['defaults']['EXTRA_ENV']['Q18A_TARGET_LABEL'],
                             's5-overlap-80-80')
            self.assertTrue(dec.no_unknown(actual['identity']))
            self.assertEqual(actual['identity']['env']['VLLM_API_KEY'],
                             {'redacted_length': len(DUMMY)})
            self.assertNotIn(DUMMY, json.dumps(actual))
        # Unknown names still reach the unchanged strict no_unknown refusal.
        overrides['EXTRA_ENV'] += ' Q18A_UNREVIEWED=1'
        with self.assertRaises(ident.EvidenceError):
            dec.declare(self.a, overrides, self.image, policy,
                        self.root/'q18a-unknown-refused', install_launcher=True)
        values = dec.raw_defaults(dec.apply_setcfg(
            self.defaults.read_text(), over, self.a.setcfg))
        for index, private_override in enumerate((
                dict(values, EXTRA_ARGS='--api-key ' + DUMMY),
                dict(values, EXTRA_ARGS='--api-key=' + DUMMY),
                dict(values, EXTRA_ENV=values['EXTRA_ENV'] + ' VLLM_API_KEY=' + DUMMY))):
            with self.subTest(auth_override=index), self.assertRaises(ident.EvidenceError):
                dec.render(launch, private_override, IMAGE,
                           self.root/('s5-private-refused-' + str(index)), self.task)
        bad = self.root / 's5-launcher-modified'
        bad.write_text(launch.read_text() + '\ntrue\n')
        with self.assertRaises(ident.EvidenceError):
            dec.render(bad, values, IMAGE, self.root / 's5-modified')

    def test_target_policy_enum_is_exact_and_has_no_prefix_allowance(self):
        import e3_overlap_grade
        self.assertEqual(set(ident.PUBLIC_TARGET_LABELS), set(e3_overlap_grade.TARGET_POLICIES))
        for label in ident.PUBLIC_TARGET_LABELS:
            self.assertEqual(ident.clean_value('Q18A_TARGET_LABEL', label), label)
        for label in ('unknown', 's5-overlap-80-80\n', DUMMY, 'https://private.invalid'):
            with self.subTest(label_length=len(label)), self.assertRaises(ident.EvidenceError):
                ident.clean_value('Q18A_TARGET_LABEL', label)
        unknown = ident.clean_env({'Q18A_UNREVIEWED': '1'})
        self.assertFalse(dec.no_unknown(unknown))
        self.assertEqual(unknown['Q18A_UNREVIEWED'], {'redacted_length': 1, 'unclassified': True})

    def test_declaration_cli_failure_retains_sites_without_exception_values(self):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['engine_identity.py', 'declare', '/unused']), \
                patch.object(dec, 'run', side_effect=ValueError(DUMMY+'\nprivate parser input')), \
                contextlib.redirect_stdout(output):
            self.assertEqual(ident.main(), 2)
        text = output.getvalue()
        self.assertNotIn(DUMMY, text)
        self.assertNotIn('private parser input', text)
        rows = text.splitlines()
        self.assertEqual(len(rows), 2)
        prefix = 'identity declaration diagnostic '
        diagnostic = json.loads(rows[1].removeprefix(prefix))
        self.assertEqual(diagnostic['error_type'], 'ValueError')
        self.assertTrue(0 < len(diagnostic['error_stack']) <= 8)
        self.assertTrue(all(set(row) == {'file', 'function', 'line'}
                            for row in diagnostic['error_stack']))

    def test_q16_dense_and_other_configs_render_one_sparse_assignment(self):
        sys.path.insert(0,str(ROOT/'patches/s5_engine'))
        import make_e2_arena as g
        launch=self.root/'q16-launcher'
        launch.write_text(g.launcher((ROOT/'deploy/run-flash-prod.sh').read_text()))
        self.assertEqual(dec.sha(launch),dec.E2_LAUNCHER_SHA)
        old=self.root/'q16b-launcher'
        old.write_text(launch.read_text().replace(
            '-e VLLM_QSA_KVO_SPARSE_PREFILL="${SPARSE_PREFILL:-1}"',
            '-e VLLM_QSA_KVO_SPARSE_PREFILL=1'))
        old_sha='d0a5967e58336e8694161b887fcada6103ed4ad270984875c6201ef3b4507491'
        self.assertEqual(dec.sha(old),old_sha)
        names=['e2/__init__.py',*[f'e2/{n}.py' for n in g.RUNTIME],
               'e2_sources/scheduler_output.py','e2_sources/block_table.py']
        with (self.ovl/'SHA256SUMS').open('a') as manifest:
            for name in names:
                p=self.ovl/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
                manifest.write(dec.sha(p)+'  '+name+'\n')
        self.a.launcher=str(launch)
        policy=json.loads((HERE/'e2_mount_policy.json').read_text())
        configs={}
        for cfg in ('DENSE','CONTROL','VERIFY','SHIP'):
            # Source the actual shared settings arrays, not a handwritten copy.
            script='unset E2_KV_BYTES; QNAME=q16-regression; E2_OVERLAY=$3; source "$1"; declare -n words="$2"; printf "%s\\0" "${words[@]}"'
            p=subprocess.run(['bash','-uc',script,'fixture',str(HERE/'runs/q16_settings.sh'),
                              'E2_'+cfg,dec.TASK+'/prod/overlay-v16g'],capture_output=True,check=True)
            words=p.stdout.decode().rstrip('\0').split('\0')
            configs[cfg]=dec.settings(words)
        original=self.defaults.read_text()
        dense_values=dec.raw_defaults(dec.apply_setcfg(original,configs['DENSE'],self.a.setcfg))
        self.assertEqual(dense_values['SPARSE_PREFILL'],'0')
        for cfg,overrides in configs.items():
            with self.subTest(config=cfg):
                expected='0' if cfg=='DENSE' else '1'
                self.assertEqual(overrides['SPARSE_PREFILL'],'0' if cfg=='DENSE' else '-')
                self.assertEqual(overrides['GPU_KV_BYTES'],'1500000000')
                self.assertNotIn('VLLM_QSA_KVO_SPARSE_PREFILL=',overrides['EXTRA_ENV'])
                # Model the current defaults after a previous dense run: each
                # normal configuration MUST explicitly clear the persisted 0.
                self.defaults.write_text(dec.apply_setcfg(original,configs['DENSE'],self.a.setcfg))
                future=dec.apply_setcfg(self.defaults.read_text(),overrides,self.a.setcfg)
                values=dec.raw_defaults(future)
                if cfg!='DENSE':self.assertNotIn('SPARSE_PREFILL',values)
                argv=dec.render(launch,values,IMAGE,self.root/('q16-argv-'+cfg),self.task)
                self.assertEqual(argv.count('--kv-cache-memory-bytes'),1)
                self.assertEqual(argv[argv.index('--kv-cache-memory-bytes')+1],'1500000000')
                assignments=[v for v in argv if v.startswith('VLLM_QSA_KVO_SPARSE_PREFILL=')]
                self.assertEqual(assignments,['VLLM_QSA_KVO_SPARSE_PREFILL='+expected])
                declared,_=dec.declare(self.a,overrides,self.image,policy,self.root/('q16-declare-'+cfg))
                self.assertEqual(declared['identity']['env']['VLLM_QSA_KVO_SPARSE_PREFILL'],expected)
                if cfg!='DENSE':
                    # Compare complete effective command/env/mount structure
                    # to the prior pinned launcher; only source identity changes.
                    with patch.object(dec,'E2_LAUNCHER_SHA',old_sha):
                        old_argv=dec.render(old,values,IMAGE,self.root/('q16b-argv-'+cfg),self.task)
                    self.assertEqual(dec.docker_spec(argv,self.image,self.task),
                                     dec.docker_spec(old_argv,self.image,self.task))
        self.assertNotIn(DUMMY,json.dumps(declared))

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

    def test_setcfg_persistence_unset_and_duplicate_word_semantics(self):
        source=(ROOT/'deploy/setcfg.py').read_text()
        def actual(raw,words):
            writer=KeepBuffer()
            def fake_open(path,mode='r'):
                if path=='/etc/default/vllm-flash' and mode=='r':return io.StringIO(raw)
                if path=='/etc/default/vllm-flash.next' and mode=='w':return writer
                raise AssertionError('unexpected path')
            with patch('builtins.open',fake_open),patch.object(sys,'argv',['setcfg.py']+words),contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source,'setcfg.py','exec'),{})
            return dec.raw_defaults(writer.getvalue())
        raw='SPARSE_PREFILL=0\nCORE_OVERLAY=1\n'
        self.assertEqual(actual(raw,['CORE_OVERLAY=1'])['SPARSE_PREFILL'],'0')
        self.assertNotIn('SPARSE_PREFILL',actual(raw,['SPARSE_PREFILL=-']))
        self.assertEqual(actual(raw,['SPARSE_PREFILL=0','SPARSE_PREFILL=1'])['SPARSE_PREFILL'],'1')
        # The declaration interface is intentionally stricter than direct setcfg.
        with self.assertRaises(ident.EvidenceError):dec.settings(['SPARSE_PREFILL=0','SPARSE_PREFILL=1'])

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

    def test_declared_exclusions_match_capture_and_active_weights_stay_hashed(self):
        self.defaults.write_text(dec.apply_setcfg(self.defaults.read_text(),dec.PP_SETTINGS,self.a.setcfg))
        declared=self.declare(); spec=declared['identity']
        work=next(m for m in spec['mounts'] if m['destination']=='/work')
        self.assertFalse(any(e['path']=='models' or e['path'].startswith('models/') for e in work['content']['entries']))
        model=next(m for m in spec['mounts'] if m['destination']=='/model')
        self.assertTrue(any(e['path']=='weights.fixture' for e in model['content']['entries']))
        argv=dec.render(self.a.launcher,dec.raw_defaults(self.defaults.read_text()),IMAGE,self.root/'capture-render',self.task)
        before={'Id':'c'*64,'Image':IMAGE,'Path':spec['command']['path'],'Args':['serve']+argv[argv.index(IMAGE)+1:],
                'Config':{'Env':[k+'='+v for k,v in spec['env'].items()]},'RestartCount':0,
                'State':{'StartedAt':'fixture','Pid':123},'Mounts':[
                {'Type':m['type'],'Source':m['source'],'Destination':m['destination'],'RW':m['rw'],'Propagation':m['propagation']}
                for m in spec['mounts']]}
        expected={m['destination']:m for m in spec['mounts']}
        def run(argv,timeout,data=None):
            if argv[:3]==['docker','image','inspect']:return [self.image]
            query=json.loads(data);destinations=[]
            for m in query['mounts']:
                destinations.append(m['destination'])
                self.assertEqual(m['exclusions'],expected[m['destination']]['content'].get('exclusions',[]))
                m['destination']=str(dec.host_path(expected[m['destination']]['source'],self.task))
            p=subprocess.run([sys.executable,str(HERE/'identity_probe.py')],input=json.dumps(query),capture_output=True,text=True)
            value=json.loads(p.stdout)
            for m,d in zip(value['mounts'],destinations):m['destination']=d
            return value
        a=argparse.Namespace(container='fixture',defaults=str(self.defaults),timeout=1800,max_bytes=2**24,max_files=1000)
        with patch('engine_identity.inspect',return_value=before),patch('engine_identity.run',side_effect=run):
            captured=ident.capture(a,declared)
            self.assertEqual(captured['identity'],spec)
            self.assertGreaterEqual(captured['io']['wall_seconds'],0)
            # An inactive checkpoint is excluded, but the active /model bytes
            # still change the captured identity and fail the declared comparison.
            inactive=self.task/'models/inactive';inactive.mkdir();(inactive/'weights').write_text('irrelevant')
            self.assertEqual(ident.capture(a,declared)['identity'],spec)
            (self.task/'models/intel-w4a16/weights.fixture').write_text('changed weights')
            self.assertNotEqual(ident.capture(a,declared)['identity'],spec)

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
