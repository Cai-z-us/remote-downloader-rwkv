"""Inventory-aware two-host allocation. Stop ALL writers before generating a plan.

Verified replicas take precedence over partials; partials stay where they are.
Only untouched files are scheduled by projected completion time, not file count.
"""
import argparse
import concurrent.futures
import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path
import shlex
import subprocess
import time

from .repo import atomic_json, read_manifest
from .service import ssh_command


def audit_inventory(destination, rows):
    # Self-contained so identical code can execute over SSH without an installation.
    import hashlib
    from pathlib import Path
    dest = Path(destination).expanduser().resolve()
    result = {}
    trusted = {}
    for state_path in dest.glob('.hf-download-state*.json'):
        try:
            state = __import__('json').loads(state_path.read_text())
            for name, row in state.items():
                if name != '_meta' and row.get('status') == 'done' and row.get('verified') is True:
                    trusted[name] = row
        except (OSError, ValueError):
            pass
    for item in rows:
        name = item['path']
        paths = {}
        for suffix in ('', '.uploading'):
            p = dest / (name + suffix)
            if p.is_symlink() or not p.resolve().is_relative_to(dest):
                raise ValueError('unsafe inventory path: ' + str(p))
            if not p.exists():
                continue
            before = p.stat()
            valid = False
            digest = None
            if before.st_size == item['size']:
                known = trusted.get(name)
                if known and known.get('fingerprint') == [before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino]:
                    digest = known.get('digest')
                    valid = True
                else:
                    h = hashlib.sha256() if item.get('sha256') else hashlib.sha1()
                    if not item.get('sha256'):
                        h.update(f'blob {item["size"]}\0'.encode())
                    with p.open('rb') as f:
                        for block in iter(lambda: f.read(1024 * 1024), b''):
                            h.update(block)
                    digest = h.hexdigest()
                    valid = digest == (item.get('sha256') or item.get('git_oid'))
            after = p.stat()
            signature = lambda s: [s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino]
            if signature(before) != signature(after):
                raise RuntimeError('inventory changed during audit; stop writers: ' + name)
            paths[suffix] = {'bytes': after.st_size, 'valid': valid, 'digest': digest,
                             'fingerprint': signature(after)}
        result[name] = paths
    return result


def inventory(host, destination, files):
    rows = [asdict(f) for f in files]
    if host == 'local':
        return audit_inventory(destination, rows)
    code = ('import json,sys\n' + inspect.getsource(audit_inventory)
            + '\np=json.load(sys.stdin); print(json.dumps(audit_inventory(p["dest"],p["rows"])))')
    proc = subprocess.run(ssh_command(host, 'python3 -c ' + shlex.quote(code)),
                          input=json.dumps({'dest': destination, 'rows': rows}), text=True,
                          capture_output=True, timeout=1800, check=True)
    return json.loads(proc.stdout)


def allocate(files, inventories, speeds):
    if len(inventories) != 2 or len(speeds) != 2 or min(speeds) <= 0:
        raise ValueError('two positive worker speeds are required')
    owners, reasons, credit = {}, {}, {}
    remaining = [0, 0]
    untouched = []
    for f in files:
        copies = [inv.get(f.path, {}) for inv in inventories]
        # Prefer the final destination (remote=1) when both replicas are complete.
        complete = [i for i in (1, 0) if any(v.get('valid') for v in copies[i].values())]
        if complete:
            owner = complete[0]
            amount, reason = f.size, 'verified-existing'
        else:
            partials = [c.get('.uploading', {}).get('bytes', 0) for c in copies]
            partials = [b if 0 < b < f.size else 0 for b in partials]
            if not any(partials):
                untouched.append(f)
                continue
            owner = max((1, 0), key=lambda i: partials[i])
            amount, reason = partials[owner], 'resume-largest-prefix'
        owners[f.path], reasons[f.path], credit[f.path] = owner, reason, amount
        remaining[owner] += f.size - amount
    # Longest jobs first; minimize projected makespan on unequal-speed machines.
    for f in sorted(untouched, key=lambda f: (-f.size, f.path)):
        def finish(i):
            loads = remaining.copy(); loads[i] += f.size
            return max(loads[j] / speeds[j] for j in (0, 1)), loads[i] / speeds[i], i
        owner = min((0, 1), key=finish)
        owners[f.path], reasons[f.path], credit[f.path] = owner, 'weighted-unstarted', 0
        remaining[owner] += f.size
    if len(owners) != len(files):
        raise ValueError('duplicate output paths in input')
    return owners, reasons, credit, remaining


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--local-dest', required=True)
    p.add_argument('--remote-host', required=True)
    p.add_argument('--remote-dest', required=True)
    p.add_argument('--speeds', type=float, nargs=2, required=True,
                   metavar=('LOCAL_MIB', 'REMOTE_MIB'),
                   help='measured download speeds for local and remote workers')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('output must be a new directory; plans are immutable')
    files = read_manifest(args.manifest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(inventory, 'local', args.local_dest, files)
        b = pool.submit(inventory, args.remote_host, args.remote_dest, files)
        inventories = [a.result(), b.result()]
    speeds = [v * 1024**2 for v in args.speeds]
    owners, reasons, credit, remaining = allocate(files, inventories, speeds)
    assignments = [{**asdict(f), 'owner': owners[f.path], 'reason': reasons[f.path],
                    'reused_bytes': credit[f.path]} for f in files]
    plan_id = hashlib.sha256(json.dumps(assignments, sort_keys=True).encode()).hexdigest()[:16]
    args.output.mkdir(parents=True)
    for i, label in enumerate(('local', 'remote')):
        subset = [asdict(f) for f in files if owners[f.path] == i]
        atomic_json(args.output / f'{label}.json', {'plan_id': plan_id, 'files': subset})
        atomic_json(args.output / f'{label}.inventory.json', inventories[i])
    atomic_json(args.output / 'full.json', {'plan_id': plan_id, 'files': [asdict(f) for f in files]})
    report = {'plan_id': plan_id, 'created': time.time(), 'speeds_mib': args.speeds,
              'destinations': [args.local_dest, args.remote_host + ':' + args.remote_dest],
              'remaining_bytes': remaining, 'estimated_download_seconds': max(remaining[i] / speeds[i] for i in (0, 1)),
              'assignments': assignments}
    atomic_json(args.output / 'plan.json', report)
    for i, label in enumerate(('local', 'remote')):
        print(f'{label}: {sum(v == i for v in owners.values())} files; remaining '
              f'{remaining[i]/1024**3:.2f} GiB; ETA {remaining[i]/speeds[i]/3600:.2f} h')
    print('Plan:', args.output, 'id:', plan_id)


if __name__ == '__main__':
    main()
