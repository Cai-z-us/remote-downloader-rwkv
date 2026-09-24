"""Persistent systemd/launchd workers and SSH deployment (never an SSH data relay)."""
import base64
import hashlib
import json
import os
import platform
import plistlib
import shlex
import subprocess
import sys
from pathlib import Path

from .config import apply_legacy_proxy_env, normalize_proxies
from .repo import atomic_json


# systemd --user and launchd do not reliably inherit the installing shell's
# proxy environment.  Store only network settings in the private job config.
PROXY_ENV_KEYS = (
    'HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy',
    'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy',
    'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE',
)


def ssh_command(host, command):
    if not host or host.startswith('-'):
        raise ValueError('invalid SSH host')
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', host, command]


def service_name(args):
    identity = (
        f'{args.repository_url}\0{args.endpoint}\0'
        f'{Path(args.destination).expanduser().resolve()}\0{args.shard}\0{args.history}'
    )
    return 'hf-download-' + hashlib.sha256(identity.encode()).hexdigest()[:16]


def systemd_quote(value):
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%').replace('$', '$$') + '"'


def unit_text(command, cwd):
    return ('[Unit]\nDescription=Verified Hugging Face downloader\n'
            'StartLimitIntervalSec=0\n\n[Service]\nType=simple\n'
            f'WorkingDirectory={cwd}\n'
            f'ExecStart={" ".join(systemd_quote(x) for x in command)}\n'
            'Restart=always\nRestartSec=30\nKillMode=control-group\nTimeoutStopSec=30\n'
            'UMask=0077\nEnvironment=PYTHONUNBUFFERED=1\n'
            '\n[Install]\nWantedBy=default.target\n')


def launchd_config(name, command, cwd, log):
    return {'Label': name, 'ProgramArguments': command, 'WorkingDirectory': str(cwd),
            'RunAtLoad': True, 'KeepAlive': {'SuccessfulExit': False}, 'ThrottleInterval': 30,
            'ProcessType': 'Background', 'StandardOutPath': str(log), 'StandardErrorPath': str(log),
            'EnvironmentVariables': {'PYTHONUNBUFFERED': '1'}}


def install_service(args, start=False):
    system = platform.system()
    if system not in ('Linux', 'Darwin'):
        raise RuntimeError('service installation supports Linux and macOS only')
    name = service_name(args)
    root = Path.home() / '.local/share/hf-remote-downloader/jobs'
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    config = root / f'{name}.json'
    data = vars(args).copy()
    data.update(host=None, supervise=True, install_service=False, background=False, status=False)
    data['destination'] = str(Path(args.destination).expanduser().resolve())
    data['manifest'] = str(args.manifest.expanduser().resolve()) if args.manifest else None
    if args.manifest:
        # Service never relies on /tmp or the lifetime of the controller's catalog.
        catalog = root / f'{name}.catalog.json'
        atomic_json(catalog, json.loads(Path(data['manifest']).read_text(encoding='utf-8')))
        data['manifest'] = str(catalog)
    cwd = Path(__file__).resolve().parent.parent
    old_umask = os.umask(0o077)
    try:
        proxy_env = ({key: os.environ[key] for key in PROXY_ENV_KEYS if key in os.environ}
                     if getattr(args, 'proxy', None) is None
                     and not getattr(args, 'no_proxy', False) else {})
        atomic_json(config, {'args': data, 'token': os.getenv('HF_TOKEN'),
                             'proxy_env': proxy_env})
        config.chmod(0o600)
    finally:
        os.umask(old_umask)
    command = [sys.executable, '-m', 'hf_downloader.service', '--run', str(config)]
    if system == 'Linux':
        units = Path.home() / '.config/systemd/user'
        units.mkdir(parents=True, exist_ok=True)
        unit = units / f'{name}.service'
        unit.write_text(unit_text(command, cwd))
        subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True, timeout=30)
        print(f'Installed {unit}\nEnable lingering if needed: loginctl enable-linger {shlex.quote(os.getenv("USER", ""))}')
        print(f'systemctl --user enable --now {name}.service')
        print(f'journalctl --user -u {name}.service -f', flush=True)
        if start:
            linger = subprocess.run(['loginctl', 'show-user', str(os.getuid()), '-p', 'Linger', '--value'],
                                    capture_output=True, text=True, timeout=15)
            if linger.returncode or linger.stdout.strip() != 'yes':
                raise RuntimeError('Linger is not enabled; enable it explicitly before starting an unattended worker')
            subprocess.run(['systemctl', '--user', 'enable', '--now', name + '.service'], check=True, timeout=30)
    else:
        agents = Path.home() / 'Library/LaunchAgents'
        agents.mkdir(parents=True, exist_ok=True)
        plist = agents / f'{name}.plist'
        plist.write_bytes(plistlib.dumps(launchd_config(name, command, cwd, root / f'{name}.log')))
        print(f'Installed {plist}\nlaunchctl bootstrap gui/{os.getuid()} {shlex.quote(str(plist))}')
        print('LaunchAgents require a logged-in GUI user; sleep/offline stops LOCAL transfers.', flush=True)
        if start:
            subprocess.run(['launchctl', 'bootout', f'gui/{os.getuid()}/{name}'], capture_output=True, timeout=30)
            subprocess.run(['launchctl', 'bootstrap', f'gui/{os.getuid()}', str(plist)], check=True, timeout=30)
    return 0


BOOTSTRAP = r'''
import base64, json, os, pathlib, subprocess, sys

payload = json.load(sys.stdin)
root = pathlib.Path.home() / '.local/share/hf-remote-downloader'
release = root / 'releases' / payload['release']
release.mkdir(parents=True, exist_ok=True)
for name, content in payload['files'].items():
    path = release / 'hf_downloader' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(content))

venv = root / 'venv'
python = venv / 'bin/python'
if not python.exists():
    subprocess.run(['python3', '-m', 'venv', '--system-site-packages', str(venv)], check=True)
if subprocess.run(
    [str(python), '-c', 'import requests'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
).returncode:
    pip_env = os.environ.copy()
    proxy = payload.get('proxy')
    proxy_keys = (
        'HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy',
        'ALL_PROXY', 'all_proxy', 'HF_PROXY',
    )
    if payload.get('no_proxy'):
        for key in proxy_keys + ('HF_PROXIES',):
            pip_env.pop(key, None)
    elif not proxy:
        for key in proxy_keys:
            if pip_env.get(key):
                proxy = pip_env[key]
                break
        if not proxy and pip_env.get('HF_PROXIES'):
            proxy = pip_env['HF_PROXIES'].split(',', 1)[0].strip()
    cmd = [str(python), '-m', 'pip', 'install', 'requests>=2.28']
    if proxy:
        cmd += ['--proxy', proxy]
    subprocess.run(cmd, check=True, timeout=240, env=pip_env)

args = payload['argv']
if payload.get('manifest') is not None:
    manifest = release / 'imported-catalog.json'
    manifest.write_text(json.dumps(payload['manifest']))
    args += ['--manifest', str(manifest)]

env = os.environ.copy()
# The remote worker must not interpret its own SSH host as another deployment,
# but it may use proxy variables configured on the remote machine.
env.pop('HF_REMOTE_HOST', None)
env.pop('HF_TOKEN', None)
if payload.get('no_proxy'):
    for key in (
        'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy',
        'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy', 'HF_PROXY', 'HF_PROXIES',
    ):
        env.pop(key, None)
if payload.get('token'):
    env['HF_TOKEN'] = payload['token']
subprocess.run([str(python), '-m', 'hf_downloader.repo'] + args,
               cwd=release, env=env, check=True)
'''


def deploy_remote(args):
    package = Path(__file__).parent
    files = {p.name: base64.b64encode(p.read_bytes()).decode() for p in package.glob('*.py')}
    release = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    argv = [args.repository_url, args.destination, '--shard', '/'.join(map(str, args.shard)),
            '--order', args.order, '--endpoint', args.endpoint, '--workers', str(args.workers),
            '--retries', str(args.retries), '--retry-delay', str(args.retry_delay),
            '--install-service' if args.install_service and not args.background else '--background']
    if args.history:
        argv += ['--history']
    for proxy in args.proxy or []:
        argv += ['--proxy', proxy]
    if getattr(args, 'no_proxy', False):
        argv += ['--no-proxy']
    payload = {
        'release': release,
        'files': files,
        'argv': argv,
        'proxy': next(iter(args.proxy or []), None),
        'no_proxy': getattr(args, 'no_proxy', False),
        'token': os.getenv('HF_TOKEN'),
        'manifest': json.loads(args.manifest.read_text()) if args.manifest else None,
    }
    # Package and credentials travel in encrypted SSH stdin, not command-line arguments.
    subprocess.run(ssh_command(args.host, 'python3 -c ' + shlex.quote(BOOTSTRAP)),
                   input=json.dumps(payload), text=True, check=True, timeout=360)
    print(f'Remote worker installed on {args.host}; no data traffic or control loop depends on this terminal.')
    return 0


def main():
    import argparse
    from .repo import run
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, type=Path)
    config = json.loads(p.parse_args().run.read_text(encoding='utf-8'))
    os.environ.pop('HF_REMOTE_HOST', None)
    for key in (
        'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy',
        'ALL_PROXY', 'all_proxy', 'NO_PROXY', 'no_proxy', 'HF_PROXY', 'HF_PROXIES',
    ):
        os.environ.pop(key, None)
    for key, value in config.get('proxy_env', {}).items():
        if key in PROXY_ENV_KEYS:
            os.environ[key] = value
    if config.get('token'):
        os.environ['HF_TOKEN'] = config['token']
    else:
        os.environ.pop('HF_TOKEN', None)
    args = argparse.Namespace(**config['args'])
    args.shard = tuple(args.shard)
    try:
        apply_legacy_proxy_env(args)
        args.proxy = normalize_proxies(getattr(args, 'proxy', None))
        return run(args)
    except Exception as exc:
        from .repo import error_text
        print(error_text(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
