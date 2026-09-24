"""Resumable, verified Hugging Face downloads on the machine running the worker."""
import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import platform
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import add_proxy_arguments, apply_legacy_proxy_env, normalize_proxies

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None

API_ROOT = 'https://huggingface.co'
STATE_NAME = '.hf-download-state.json'
BLOCK = 1024 * 1024


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    url: str
    sha256: str | None = None
    git_oid: str | None = None


def safe_path(value):
    p = PurePosixPath(value)
    if (p.is_absolute() or '..' in p.parts or not value or value == '.'
            or '\\' in value or any(ord(c) < 32 for c in value)):
        raise ValueError(f'unsafe repository path: {value!r}')
    return str(p)


def validate_endpoint(value):
    parsed = urlparse(value)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError('endpoint must be an HTTPS URL without embedded credentials')
    return value.rstrip('/')


def parse_repo_url(value, endpoint=API_ROOT):
    endpoint = validate_endpoint(endpoint)
    parsed = urlparse(value)
    allowed_hosts = {urlparse(API_ROOT).netloc, urlparse(endpoint).netloc}
    if parsed.scheme != 'https' or parsed.netloc not in allowed_hosts:
        if len(allowed_hosts) == 1:
            hint = f'https://{next(iter(allowed_hosts))}'
        else:
            hint = 'the configured HTTPS endpoint'
        raise ValueError(f'repository URL must use {hint}')
    parts = parsed.path.strip('/').split('/')
    kind = 'models'
    if parts and parts[0] in ('datasets', 'models', 'spaces'):
        kind = parts.pop(0)
    if len(parts) == 4 and parts[2] == 'tree' and parts[3] == 'main':
        parts = parts[:2]
    if len(parts) != 2 or any(x in ('', '.', '..') for x in parts):
        raise ValueError('repository URL must identify owner/repository (optionally /tree/main)')
    return kind, '/'.join(parts)


def path_hash(path):
    return int.from_bytes(hashlib.sha256(path.encode('utf-8')).digest(), 'big')


def parse_shard(value):
    try:
        index, count = map(int, value.split('/'))
        if count < 1 or not 0 <= index < count:
            raise ValueError()
        return index, count
    except ValueError:
        raise argparse.ArgumentTypeError('shard must be N/M with 0 <= N < M') from None


def select_files(files, shard=(0, 1), order='hash'):
    index, count = shard
    selected = [f for f in files if path_hash(f.path) % count == index]
    return sorted(selected, key=(lambda f: (f.size, f.path)) if order == 'size' else
                  (lambda f: (path_hash(f.path), f.path)))


def session(proxy=None, token=None, *, trust_env=None):
    """Create an HTTP session with explicit or standard environment routing.

    An explicit proxy is fail-closed: environment variables cannot silently
    replace it.  With no explicit proxy, requests' normal HTTP(S)_PROXY,
    ALL_PROXY and NO_PROXY handling is retained unless ``trust_env`` is false.
    """
    s = requests.Session()
    if trust_env is None:
        trust_env = proxy is None
    s.trust_env = trust_env
    if proxy:
        s.proxies = {'http': proxy, 'https': proxy}
    if token:
        s.headers['Authorization'] = f'Bearer {token}'
    s.headers['Accept-Encoding'] = 'identity'
    s.mount('https://', HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))))
    return s


class ProxyPool:
    """Choose explicit failover proxies without probing a specific endpoint."""
    def __init__(self, values=None, endpoint=API_ROOT, *, use_environment=None):
        self.values = [v for v in (values or []) if v]
        if use_environment is None:
            # ``None`` means no explicit configuration and therefore follows
            # requests' standard environment behavior.  An explicit empty
            # list retains direct-connection behavior for library users.
            use_environment = values is None
        self.use_environment = bool(use_environment) and not self.values
        self.endpoint = endpoint
        self.current = None
        self.lock = threading.Lock()

    def choose(self):
        with self.lock:
            if not self.values:
                return None
            if self.current is None:
                self.current = self.values[0]
            return self.current

    def invalidate(self):
        with self.lock:
            if self.current in self.values:
                self.values.append(self.values.pop(self.values.index(self.current)))
            self.current = None


def pages(s, url, params=None):
    origin = urlparse(url).netloc
    while url:
        r = s.get(url, params=params, timeout=(15, 60))
        r.raise_for_status()
        yield from r.json()
        linked = r.links.get('next', {}).get('url')
        cursor = r.headers.get('x-next-cursor')
        if cursor:
            params = dict(params or {})
            params['cursor'] = cursor
        else:
            url, params = linked, None
        if url and urlparse(url).netloc != origin:
            raise ValueError('pagination changed API host')


def discover(url, token=None, proxy=None, *, endpoint=API_ROOT, history=False,
             use_environment=None):
    endpoint = validate_endpoint(endpoint)
    kind, repo = parse_repo_url(url, endpoint)
    prefix = '' if kind == 'models' else kind + '/'
    api = f'{endpoint}/api/{kind}/{repo}'
    base = f'{endpoint}/{prefix}{repo}/resolve'
    unique = {}
    with session(proxy, token, trust_env=use_environment) as s:
        if history:
            revisions = [c['id'] for c in pages(s, api + '/commits/main', {'limit': 100})]
        else:
            r = s.get(api, timeout=(15, 60)); r.raise_for_status()
            revisions = [r.json()['sha']]
        # Newest first, preserving same-name/different-content versions separately.
        for revision in revisions:
            for x in pages(s, api + '/tree/' + revision, {'recursive': 'true', 'expand': 'true'}):
                if x.get('type') != 'file' or (history and not x['path'].endswith('.pth')):
                    continue
                path = safe_path(x['path'])
                digest = (x.get('lfs') or {}).get('oid')
                oid = x.get('oid') if not digest else None
                key = (path, digest or oid)
                if key in unique:
                    continue
                output = path
                if any(k[0] == path for k in unique):
                    output = f'.versions/{digest or oid}/{path}'
                unique[key] = RemoteFile(output, int(x['size']),
                    f'{base}/{revision}/{quote(path, safe="/")}', digest, oid)
    return list(unique.values())


def discover_with_pool(url, token, pool, *, endpoint=API_ROOT, history=False):
    """Discover metadata, rotating an explicit proxy list on request errors."""
    attempts = max(1, len(pool.values))
    last_error = None
    for _ in range(attempts):
        try:
            return discover(
                url,
                token,
                pool.choose(),
                endpoint=endpoint,
                history=history,
                use_environment=pool.use_environment,
            )
        except requests.RequestException as exc:
            last_error = exc
            pool.invalidate()
    if last_error is not None:
        raise last_error
    raise RuntimeError('repository discovery failed')


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def state_path(dest, shard=(0, 1)):
    i, n = shard
    return Path(dest) / (STATE_NAME if n == 1 else f'.hf-download-state.{i}-of-{n}.json')


class StateStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.data = (json.loads(self.path.read_text(encoding='utf-8'))
                     if self.path.exists() else {})

    def get(self, key):
        with self.lock:
            return dict(self.data.get(key, {}))

    def update(self, key, **values):
        with self.lock:
            self.data.setdefault(key, {}).update(values, updated=time.time())
            atomic_json(self.path, self.data)


@contextlib.contextmanager
def file_lock(path):
    """Acquire a non-blocking cross-process lock on Unix or Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return

        # Keep the core downloader usable on Windows even though its optional
        # service installer is intentionally limited to Linux/macOS.
        import msvcrt  # pragma: no cover - exercised on Windows

        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write('0')
            f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def fingerprint(path):
    st = path.stat()
    return [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino]


def verify(path, item, state):
    state.update(item.path, status='verifying', bytes=path.stat().st_size,
                 size=item.size, verified=False)
    if path.stat().st_size != item.size:
        return False
    if not item.sha256 and not item.git_oid:
        raise ValueError('missing trusted content hash; refusing size-only verification')
    h = hashlib.sha256() if item.sha256 else hashlib.sha1()
    if item.git_oid:
        h.update(f'blob {item.size}\0'.encode())
    with path.open('rb') as f:
        for b in iter(lambda: f.read(BLOCK), b''):
            h.update(b)
    return h.hexdigest() == (item.sha256 or item.git_oid)


def output_path(dest, path):
    final = dest / safe_path(path)
    if not final.resolve().is_relative_to(dest.resolve()):
        raise ValueError('output path escapes destination through a symlink')
    return final


def error_text(exc):
    # requests exceptions can contain signed URLs or private query parameters.
    text = re.sub(r'https?://\S+', '<url>', str(exc))
    return f'{type(exc).__name__}: {text}'[:500]


def _download(item, dest, pool, token, state, retries, retry_delay):
    final = output_path(dest, item.path)
    temp = Path(str(final) + '.uploading')
    if temp.is_symlink():
        raise ValueError('partial file is a symlink')
    final.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = item.sha256 or item.git_oid
    saved = state.get(item.path)
    if final.exists():
        if (saved.get('verified') and saved.get('digest') == expected_hash
                and saved.get('fingerprint') == fingerprint(final)) or verify(final, item, state):
            state.update(item.path, status='done', bytes=item.size, size=item.size,
                         verified=True, digest=expected_hash, fingerprint=fingerprint(final), error=None)
            return True
        # Preserve the rejected file for diagnosis rather than deleting user's data.
        final.replace(Path(str(final) + '.corrupt'))
    for attempt in range(retries):
        try:
            offset = temp.stat().st_size if temp.exists() else 0
            if offset > item.size:
                temp.replace(Path(str(temp) + '.corrupt')); offset = 0
            if offset < item.size:
                state.update(item.path, status='downloading', bytes=offset, size=item.size,
                             verified=False, error=None)
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                proxy = pool.choose()
                with session(proxy, token, trust_env=pool.use_environment) as s:
                    with s.get(item.url, headers=headers, stream=True, timeout=(15, 30)) as r:
                        r.raise_for_status()
                        if r.headers.get('Content-Encoding', 'identity') != 'identity':
                            raise ValueError('encoded response cannot be resumed safely')
                        if r.status_code == 206:
                            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', r.headers.get('Content-Range', ''))
                            if (not match or int(match[1]) != offset or int(match[3]) != item.size
                                    or int(match[2]) != item.size - 1):
                                raise ValueError('invalid Content-Range; partial file left untouched')
                        elif r.status_code == 200:
                            # A server ignoring Range must never be appended to a prefix.
                            offset = 0
                        else:
                            raise ValueError(f'unexpected HTTP status {r.status_code}')
                        total, stamp, window, window_bytes = offset, time.monotonic(), time.monotonic(), 0
                        with temp.open('ab' if offset else 'wb') as f:
                            for b in r.iter_content(64 * 1024):
                                if not b:
                                    continue
                                if total + len(b) > item.size:
                                    raise ValueError('response exceeds expected file size')
                                f.write(b); total += len(b); window_bytes += len(b)
                                now = time.monotonic()
                                if now - stamp >= 2:
                                    f.flush()
                                    state.update(item.path, status='downloading', bytes=total, size=item.size)
                                    stamp = now
                                if now - window >= 60:
                                    if window_bytes / (now - window) < 1024:
                                        raise TimeoutError('transfer below 1 KiB/s for 60 seconds')
                                    window, window_bytes = now, 0
                            f.flush(); os.fsync(f.fileno())
                        if total != item.size:
                            raise IOError(f'incomplete transfer: {total}/{item.size}')
            if not temp.exists():
                temp.touch()  # legitimate zero-byte file
            if not verify(temp, item, state):
                temp.replace(Path(str(temp) + '.corrupt'))
                raise ValueError('content hash mismatch; corrupt partial quarantined')
            temp.replace(final)
            state.update(item.path, status='done', bytes=item.size, size=item.size,
                         verified=True, digest=expected_hash, fingerprint=fingerprint(final), error=None)
            return True
        except Exception as exc:
            old = state.get(item.path)
            state.update(item.path, status='failed', verified=False,
                         bytes=temp.stat().st_size if temp.exists() else 0,
                         size=item.size, failures=old.get('failures', 0) + 1, error=error_text(exc))
            pool.invalidate()
            if attempt + 1 < retries:
                time.sleep(retry_delay)
    return False


def download_one(item, dest, pool, token, state, retries=3, retry_delay=5):
    """A single worker never raises ordinary exceptions into the batch collector."""
    try:
        lock = Path(dest) / '.hf-locks' / (hashlib.sha256(item.path.encode()).hexdigest() + '.lock')
        with file_lock(lock):
            return _download(item, Path(dest), pool, token, state, retries, retry_delay)
    except Exception as exc:
        try:
            state.update(item.path, status='failed', size=item.size, verified=False,
                         failures=state.get(item.path).get('failures', 0) + 1, error=error_text(exc))
        except Exception:
            pass
        print(f'failed {item.path}: {error_text(exc)}', flush=True)
        return False


def read_manifest(path, endpoint=API_ROOT):
    endpoint = validate_endpoint(endpoint)
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = raw.get('files', raw.get('rows', []))
    repo = raw.get('repo', '').strip('/')
    out = []
    for r in rows:
        url = r.get('url')
        if not url:
            if not repo or not r.get('revision'):
                raise ValueError('manifest row needs url or repo and revision')
            url = f'{endpoint}/{repo}/resolve/{r["revision"]}/{quote(r["path"], safe="/")}'
        parsed = urlparse(url)
        if parsed.scheme != 'https' or not parsed.netloc:
            raise ValueError('manifest download URL must be HTTPS')
        if parsed.netloc in {urlparse(API_ROOT).netloc, urlparse(endpoint).netloc}:
            url = endpoint + parsed.path
        out.append(RemoteFile(safe_path(r['path']), int(r['size']), url,
                              r.get('sha256', r.get('oid')), r.get('git_oid')))
    if len({f.path for f in out}) != len(out):
        raise ValueError('manifest has conflicting output paths')
    return out


def run(args):
    dest = Path(args.destination).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    i, n = args.shard
    state = StateStore(state_path(dest, args.shard))
    # Separate process lock for each shard plus per-output locks across shards/jobs.
    with file_lock(dest / '.hf-locks' / f'job-{i}-of-{n}.lock'):
        proxy_values = getattr(args, 'proxy', None)
        pool = ProxyPool(
            proxy_values,
            args.endpoint,
            use_environment=(
                not getattr(args, 'no_proxy', False) and proxy_values is None
            ),
        )
        metadata = {
            'url': args.repository_url,
            'history': args.history,
            'shard': [i, n],
            'endpoint': args.endpoint,
        }
        catalog = dest / f'.hf-catalog.{i}-of-{n}.json'
        if args.manifest:
            files = read_manifest(args.manifest, args.endpoint)
        elif catalog.exists() and json.loads(catalog.read_text()).get('source') == metadata:
            files = read_manifest(catalog, args.endpoint)
        else:
            files = discover_with_pool(
                args.repository_url,
                os.getenv('HF_TOKEN'),
                pool,
                endpoint=args.endpoint,
                history=args.history,
            )
        atomic_json(catalog, {'source': metadata, 'files': [asdict(f) for f in files]})
        selected = select_files(files, args.shard, args.order)
        state.update('_meta', **metadata, files=len(selected), total_bytes=sum(f.size for f in selected),
                     host=platform.node(), status='running', pid=os.getpid(),
                     expected={f.path: f.size for f in selected}, started=time.time())
        pending = selected
        while True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(download_one, f, dest, pool, os.getenv('HF_TOKEN'), state,
                                     args.retries, args.retry_delay): f for f in pending}
                failed = []
                for future in concurrent.futures.as_completed(futures):
                    f = futures[future]
                    if not future.result():
                        failed.append(f)
                    print(f'{state.get(f.path).get("status")} {f.path}', flush=True)
            state.update('_meta', status='retrying' if failed else 'complete')
            if not args.supervise:
                return 1 if failed else 0
            if not failed:
                # Stay alive under Restart=always without rescanning/re-hashing every 30s.
                while True:
                    state.update('_meta', status='complete', heartbeat=time.time())
                    time.sleep(900)
            pending = select_files(failed, (0, 1), args.order)
            time.sleep(args.retry_delay)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('repository_url', nargs='?')
    p.add_argument('destination', nargs='?')
    p.add_argument('--host', default=os.getenv('HF_REMOTE_HOST'))
    p.add_argument('--shard', type=parse_shard, default=parse_shard(
        f'{os.getenv("HF_SHARD_INDEX", "0")}/{os.getenv("HF_SHARD_COUNT", "1")}'))
    p.add_argument('--order', choices=['hash', 'size'], default='hash')
    p.add_argument('--endpoint', default=os.getenv('HF_ENDPOINT', API_ROOT))
    add_proxy_arguments(p)
    p.add_argument('--history', action='store_true', help='all historical .pth blobs, pinned to commits')
    p.add_argument('--manifest', type=Path, help='optional import of a previously pinned catalog')
    p.add_argument('--workers', type=int, default=int(os.getenv('HF_DOWNLOAD_WORKERS', '4')))
    p.add_argument('--retries', type=int, default=int(os.getenv('HF_DOWNLOAD_RETRIES', '3')))
    p.add_argument('--retry-delay', type=float, default=30)
    p.add_argument('--supervise', action='store_true', help='retry failed files until verified')
    p.add_argument('--install-service', action='store_true', help='install user service; print activation commands')
    p.add_argument('--background', action='store_true', help='install AND start supervised user service')
    p.add_argument('--status', action='store_true', help='same as hf-download-progress --once')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.workers, args.retries) < 1 or args.retry_delay < 0:
        p.error('workers/retries must be positive; retry delay must be nonnegative')
    try:
        args.endpoint = validate_endpoint(args.endpoint)
        args.proxy = normalize_proxies(args.proxy)
    except ValueError as exc:
        p.error(str(exc))
    if args.proxy and args.no_proxy:
        p.error('--proxy and --no-proxy cannot be used together')
    if args.status:
        from .progress import main as progress
        if not args.repository_url:
            p.error('--status requires a destination, optionally preceded by repository URL')
        options = [args.repository_url] + ([args.destination] if args.destination else []) + ['--once']
        if args.host:
            options += ['--host', args.host]
        if args.history:
            options += ['--history']
        options += ['--endpoint', args.endpoint]
        if args.manifest:
            options += ['--manifest', str(args.manifest)]
        for proxy in args.proxy or []:
            options += ['--proxy', proxy]
        if args.no_proxy:
            options += ['--no-proxy']
        return progress(options)
    if not args.repository_url or not args.destination:
        p.error('repository URL and destination are required')
    try:
        parse_repo_url(args.repository_url, args.endpoint)
    except ValueError as exc:
        p.error(str(exc))
    # A controller's legacy HF_PROXY setting must never be copied implicitly
    # to a different SSH host.  That host can use its own standard environment.
    if not args.host:
        apply_legacy_proxy_env(args)
        try:
            args.proxy = normalize_proxies(args.proxy)
        except ValueError as exc:
            p.error(str(exc))
    if args.host:
        from .service import deploy_remote
        return deploy_remote(args)
    if args.background or args.install_service:
        from .service import install_service
        return install_service(args, start=args.background)
    try:
        return run(args)
    except Exception as exc:
        print(error_text(exc), file=sys.stderr)
        return 1  # systemd/launchd restart (including discovery failure)


if __name__ == '__main__':
    sys.exit(main())
