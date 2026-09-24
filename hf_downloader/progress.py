"""Multi-host, multi-shard progress with recursive paths and smoothed speed/ETA."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .config import add_proxy_arguments, apply_legacy_proxy_env, normalize_proxies
from .repo import API_ROOT, ProxyPool, discover_with_pool, read_manifest, validate_endpoint
from .service import ssh_command

# Run the same snapshot code locally and remotely; no GNU find/macOS differences.
SNAPSHOT = '''
import json, os
from pathlib import Path
p=Path(destination).expanduser()
files={}
errors=[]
if not p.is_dir(): raise FileNotFoundError(str(p))
for root, dirs, names in os.walk(p):
    dirs[:]=[d for d in dirs if d != '.hf-locks']
    for name in names:
        f=Path(root)/name
        if name.startswith('.hf-') or name.endswith('.corrupt'): continue
        try: files[f.relative_to(p).as_posix()]=f.stat().st_size
        except FileNotFoundError: pass
states=[]
for f in p.glob('.hf-download-state*.json'):
    try: states.append(json.loads(f.read_text()))
    except (ValueError,OSError) as e: errors.append(f.name+': '+str(e))
expected={}
for f in p.glob('.hf-catalog.*.json'):
    try:
        for item in json.loads(f.read_text()).get('files',[]): expected[item['path']]=item['size']
    except (ValueError,OSError) as e: errors.append(f.name+': '+str(e))
result={'files':files,'states':states,'expected':expected,'errors':errors}
'''


def snapshot(host, dest):
    if host in ('local', None):
        scope = {'destination': str(dest)}
        exec(SNAPSHOT, scope)
        return scope['result']
    code = 'destination=' + repr(str(dest)) + '\n' + SNAPSHOT + '\nprint(json.dumps(result))'
    r = subprocess.run(ssh_command(host, 'python3 -c ' + shlex.quote(code)),
                       capture_output=True, text=True, timeout=30)
    if r.returncode:
        raise RuntimeError(r.stderr.strip() or 'SSH snapshot failed')
    return json.loads(r.stdout)


def remote_files(host, dest):
    return snapshot(host, dest)['files']


def fmt_bytes(value):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.2f} {unit}'
        value /= 1024


def fmt_duration(value):
    if value is None:
        return '--'
    value = max(0, int(value))
    days, value = divmod(value, 86400)
    hours, value = divmod(value, 3600)
    minutes, seconds = divmod(value, 60)
    return f'{days}d {hours:02}h {minutes:02}m' if days else f'{hours:02}h {minutes:02}m {seconds:02}s'


class SpeedWindow:
    def __init__(self, samples=5):
        self.samples = deque(maxlen=samples + 1)

    def sample(self, current, stamp=None):
        stamp = time.monotonic() if stamp is None else stamp
        if self.samples and current < self.samples[-1][1]:
            self.samples.clear()  # quarantine/restart must not fabricate negative throughput
        self.samples.append((stamp, current))
        if len(self.samples) < 2:
            return None
        first = self.samples[0]
        return max(0, current - first[1]) / max(.001, stamp - first[0])


def merged_states(states):
    rows = {}
    for state in states:
        for name, value in state.items():
            if name == '_meta':
                continue
            if value.get('updated', 0) >= rows.get(name, {}).get('updated', 0):
                rows[name] = value
    return rows


def summarize(expected, snapshots):
    files, states = {}, []
    for snap in snapshots:
        for name, size in snap['files'].items():
            files[name] = max(files.get(name, 0), size)  # duplicate replicas don't inflate progress
        states.extend(snap['states'])
    rows = merged_states(states)
    current = sum(min(size, max(files.get(n, 0), files.get(n + '.uploading', 0)))
                  for n, size in expected.items())
    complete = sum(files.get(n) == size for n, size in expected.items())
    verified = sum(files.get(n) == size and rows.get(n, {}).get('status') == 'done'
                   and rows.get(n, {}).get('verified') is True for n, size in expected.items())
    failures = [(n, v) for n, v in rows.items() if n in expected and v.get('status') == 'failed']
    return current, complete, verified, failures, files, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('repository_url')
    p.add_argument('destination', nargs='?')
    p.add_argument('--host', action='append', help='repeat for aggregate; local selects this machine')
    p.add_argument('--target', action='append', default=[], help='additional HOST=DEST (local=DEST allowed)')
    p.add_argument('--endpoint', default=os.getenv('HF_ENDPOINT', API_ROOT))
    add_proxy_arguments(p)
    p.add_argument('--history', action='store_true')
    p.add_argument('--manifest', type=Path, help='optional legacy catalog import')
    p.add_argument('--state', type=Path, help='optional legacy state on this machine')
    p.add_argument('--once', action='store_true')
    p.add_argument('--interval', type=float, default=10)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--no-color', action='store_true')
    args = p.parse_args(argv)
    if args.interval <= 0 or args.samples < 1:
        p.error('interval and sample count must be positive')
    try:
        args.endpoint = validate_endpoint(args.endpoint)
        apply_legacy_proxy_env(args)
        args.proxy = normalize_proxies(args.proxy)
    except ValueError as exc:
        p.error(str(exc))
    if args.proxy and args.no_proxy:
        p.error('--proxy and --no-proxy cannot be used together')
    if args.destination is None:
        # Backward-compatible --status DEST: expected files come from the local/remote catalog.
        args.destination, args.repository_url = args.repository_url, 'download status'
    hosts = args.host or [os.getenv('HF_REMOTE_HOST') or 'local']
    targets = [(h, args.destination) for h in hosts]
    for value in args.target:
        if '=' not in value:
            p.error('--target must be HOST=DEST')
        targets.append(tuple(value.split('=', 1)))
    expected = ({f.path: f.size for f in read_manifest(args.manifest, args.endpoint)}
                if args.manifest else {})
    discovery_pool = ProxyPool(
        args.proxy,
        args.endpoint,
        use_environment=not args.no_proxy and args.proxy is None,
    )
    speed = SpeedWindow(args.samples)
    host_speeds = [SpeedWindow(args.samples) for _ in targets]
    previous = {}
    try:
        while True:
            snaps, errors, lines = [], [], []
            for i, (host, dest) in enumerate(targets):
                try:
                    snap = snapshot(host, dest)
                    expected.update(snap['expected'])
                    for state in snap['states']:
                        expected.update(state.get('_meta', {}).get('expected', {}))
                    snaps.append(snap)
                    errors.extend(f'{host}: {e}' for e in snap['errors'])
                    local_expected = {}
                    for state in snap['states']:
                        local_expected.update(state.get('_meta', {}).get('expected', {}))
                    local_expected = local_expected or expected
                    amount, count, _, _, _, _ = summarize(local_expected, [snap])
                    rate = host_speeds[i].sample(amount)
                    lines.append(f'{host}: {count}/{len(local_expected)} files, {fmt_bytes(amount)}, '
                                 + ('sampling' if rate is None else fmt_bytes(rate) + '/s'))
                except Exception as exc:
                    errors.append(f'{host}: {exc}')
            if args.state and snaps:
                snaps[0]['states'].append(json.loads(args.state.expanduser().read_text()))
            if not expected:
                expected = {
                    f.path: f.size
                    for f in discover_with_pool(
                        args.repository_url,
                        os.getenv('HF_TOKEN'),
                        discovery_pool,
                        endpoint=args.endpoint,
                        history=args.history,
                    )
                }
            current, complete, verified, failed, files, rows = summarize(expected, snaps)
            if errors:
                speed.samples.clear(); rate = None
            else:
                rate = speed.sample(current)
            total = sum(expected.values())
            ratio = current / total if total else 1
            eta = (total - current) / rate if rate and rate > 0 else None
            if not args.no_color and sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(f'{args.repository_url}: [{"#" * int(ratio * 32):.<32}] {ratio:.2%}')
            print(f'{fmt_bytes(current)} / {fmt_bytes(total)} | files {complete}/{len(expected)} '
                  f'| verified {verified} | failed {len(failed)}')
            print('Speed: ' + ('sampling/unavailable' if rate is None else fmt_bytes(rate) + '/s')
                  + f' (last {args.samples} intervals) | ETA: {fmt_duration(eta)}')
            for line in lines:
                print('  ' + line)
            pending = [(n, files[n + '.uploading']) for n in expected if n + '.uploading' in files]
            print('Partial files (presence does NOT mean active):')
            for name, size in sorted(pending, key=lambda x: x[0])[:12]:
                status = rows.get(name, {}).get('status', 'pending')
                old = previous.get(name)
                change = 'sampling' if old is None else ('growing' if size > old else 'no growth')
                print(f'  {name}: {fmt_bytes(size)}/{fmt_bytes(expected[name])} [{status}; {change}]')
            previous = dict(pending)
            for name, row in sorted(failed, key=lambda x: x[1].get('updated', 0), reverse=True)[:8]:
                print(f'FAILED {name} ({row.get("failures", 0)} attempts): {row.get("error", "unknown")}')
            for error in errors:
                print('UNAVAILABLE ' + error)
            sys.stdout.flush()
            if args.once:
                return 1 if errors else 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f'progress error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
