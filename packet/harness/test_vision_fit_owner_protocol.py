"""Owner shell + every generated client through its whole protocol; IO leaves only."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import test_vision_fit_8x200k as client
import test_vision_fit_owner as identity

class OwnerProtocol(client.Protocol):
    def owner_fixture(self):
        fixture=identity.Identity();fixture.setUp()
        packet=self.root/'packet';shutil.copytree(client.PACKET,packet)
        for name in ('vision_fit_8x200k.py','fit_8x200k.py','packet_client_contract.py','fit_death.py','vision_answer_probe.py'):
            shutil.copy2(client.ROOT/'harness'/name,packet/'harness'/name)
        for name in ('vision_fit_identity.py','vision_fit_finish.py'):shutil.copy2(client.ROOT/'tools'/name,packet/'tools'/name)
        owner_source=client.ROOT/'owner.sh'
        if not owner_source.exists():owner_source=client.ROOT/'deploy/vision-fit/owner.sh'
        shutil.copy2(owner_source,packet/'owner.sh')
        shutil.copy2(client.BINDING,packet/'vision-fit-inputs.json')
        # The real identity parser/capture/probe/check run over the declared fixture mounts.
        for name in ('engine_identity.py','identity_probe.py','identity_declare.py'):
            shutil.copy2(client.ROOT/'harness'/name,packet/'harness'/name)
        R=fixture.runroot;run=R/'data/fit-owner';(run/'private').mkdir(parents=True,mode=0o700);run.chmod(0o700)
        for name in ('semantic','eight-unique'):
            dest=run/'private'/(name+'.json');shutil.copy2(client.CACHES/dest.name,dest);dest.chmod(0o600)
        (packet/'serve-params.env').write_text('SERVE_TAG=cpu-fit-owner\n')
        auth=R/'root/.flashnext-serve-cpu-fit-owner';auth.mkdir(parents=True,mode=0o700)
        (auth/'vllm-api.env').write_text('VLLM_API_KEY='+'a'*64+'\n');(auth/'vllm-api.env').chmod(0o600)
        # Only engine control is stubbed: it must create the run and log valid lifecycle bytes.
        (packet/'serve-launch.sh').write_text('#!/bin/bash\nset -eu\n[[ $1 == engine ]]\nmkdir -m 700 -p "$OUT/private"\ntouch "$OUT/private/launcher.found" "$OUT/private/defaults.found"\necho SERVE_UP\n')
        (packet/'serve-stop.sh').write_text('#!/bin/bash\nset -eu\n[[ -d $1/private ]]\necho "SERVE_STOP_STATE READY"\n')
        bindir=self.root/'bin';bindir.mkdir()
        (bindir/'docker').write_text('#!/bin/bash\nif [[ $1 == logs ]]; then echo "2026-11-16T17:00:00Z INFO 127.0.0.1 GET /health HTTP/1.1 200 OK";exit 0;fi\nexec "'+str(fixture.fixture.bin/'docker')+'" "$@"\n');(bindir/'docker').chmod(0o700)
        # Declarations are source-IO fixtures. Captures/checks use the REAL CLI and probe.
        shim=self.root/'owner-shim';shim.mkdir()
        (shim/'sitecustomize.py').write_text((self.shim/'sitecustomize.py').read_text()+'\n'+(Path(fixture.env['PYTHONPATH'])/'sitecustomize.py').read_text())
        env=dict(fixture.env,PATH=str(bindir)+os.pathsep+fixture.env['PATH'],PYTHONPATH=str(shim),SERVE_OWNER_UID=str(os.getuid()),SERVE_PORT=str(self.http.server_port),IDENTITY_H=str(packet/'harness'))
        names=sorted(p.relative_to(packet).as_posix() for p in packet.iterdir() if p.is_file() and p.name!='PACKET-SHA256SUMS')
        (packet/'PACKET-SHA256SUMS').write_text(''.join(hashlib.sha256((packet/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in names))
        argv=['bash',str(packet/'owner.sh'),str(run)]
        (self.root/'owner.argv.json').write_text(json.dumps(argv))
        return packet,run,env,argv

    def test_complete_owner_to_post_stop_verdict(self):
        packet,run,env,argv=self.owner_fixture()
        result=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=180)
        (self.root/'owner.stdout').write_text(result.stdout);(self.root/'owner.stderr').write_text(result.stderr)
        detail=(run/'final-read.stderr.log').read_text() if (run/'final-read.stderr.log').exists() else ''
        self.assertEqual(result.returncode,0,result.stdout+result.stderr+detail+' retained '+str(run))
        self.assertIn('VISION_FIT_OWNER FIT',result.stdout)
        value=json.loads((run/'arm-result.json').read_text());self.assertEqual(value['verdict'],'FIT')
        self.assertTrue((run/'host-stop-intent.json').exists());self.assertTrue(self.finished)
        self.assertEqual(json.loads((run/'input-preflight.json').read_text())['cache_shas'],client.vf.CACHE_SHAS)

    def test_refused_launch_never_stops_a_foreign_serve(self):
        packet,run,env,argv=self.owner_fixture()
        (packet/'serve-launch.sh').write_text('#!/bin/bash\nmkdir -m 700 -p "$OUT/private"\nexit 1\n')
        (packet/'serve-stop.sh').write_text('#!/bin/bash\ntouch "$1/FOREIGN-STOP-CALLED"\necho SERVE_STOP_STATE READY\n')
        manifest=packet/'PACKET-SHA256SUMS'
        lines=[]
        for line in manifest.read_text().splitlines():
            _,name=line.split('  ',1);lines.append(hashlib.sha256((packet/name).read_bytes()).hexdigest()+'  '+name+'\n')
        manifest.write_text(''.join(lines))
        result=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        self.assertFalse((run/'serve-engine/FOREIGN-STOP-CALLED').exists())
        self.assertIn('startup-refused-without-owned-serve',result.stdout)
