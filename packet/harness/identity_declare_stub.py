#!/usr/bin/env python3
"""Private-PATH launcher renderer. No Docker, sudo or deletion is ever executed."""
import json
import os
from pathlib import Path
import sys


def main():
    name=Path(sys.argv[0]).name; args=sys.argv[1:]
    if name=='sudo':
        if args and args[0]=='-n': args=args[1:]
        if not args or args[0] not in ('docker','rm'): raise ValueError('unsupported sudo command')
        name,args=args[0],args[1:]
    root=os.environ['DECLARE_TASK_DIR']; image=os.environ['DECLARE_IMAGE']
    if name=='rm':
        if args!=['-f',root+'/rowcache.off',root+'/rowcache.report',root+'/retain.off']:
            raise ValueError('unexpected removal request')
        return 0  # render only; nothing is removed
    if name!='docker': raise ValueError('unsupported renderer command')
    if args==['image','inspect',image,'--format','{{.Id}}']:
        print(image);return 0
    if args==['rm','-f','qwen-flash-prod']: return 0  # not sent to Docker
    if args[:1]!=['run']: raise ValueError('unsupported Docker command')
    # Parent captures this pipe in RAM; it never echoes raw argv or writes a log.
    print(json.dumps(args))
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception:
        print('launcher render refused',file=sys.stderr);raise SystemExit(2)
