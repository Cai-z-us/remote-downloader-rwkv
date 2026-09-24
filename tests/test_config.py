"""Offline regression tests for proxy and endpoint configuration."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

from hf_downloader.config import apply_legacy_proxy_env, normalize_proxies
from hf_downloader.repo import (
    API_ROOT, ProxyPool, parse_repo_url, parser, read_manifest, session,
)


class ConfigTests(unittest.TestCase):
    def test_default_uses_standard_proxy_environment(self):
        with patch.dict(os.environ, {'HTTPS_PROXY': 'http://proxy.example:8080'}, clear=True):
            args = parser().parse_args([API_ROOT + '/owner/repo', '/tmp/repo'])
            apply_legacy_proxy_env(args)
            self.assertIsNone(args.proxy)
            pool = ProxyPool(args.proxy)
            self.assertIsNone(pool.choose())
            self.assertTrue(pool.use_environment)
            with session(pool.choose(), trust_env=pool.use_environment) as client:
                self.assertTrue(client.trust_env)
                settings = client.merge_environment_settings(
                    'https://huggingface.co/', {}, None, None, None,
                )
                self.assertEqual(settings['proxies']['https'], 'http://proxy.example:8080')

    def test_no_proxy_and_explicit_routes_ignore_environment(self):
        with patch.dict(os.environ, {'HTTPS_PROXY': 'http://proxy.example:8080'}, clear=True):
            args = parser().parse_args([
                API_ROOT + '/owner/repo', '/tmp/repo', '--no-proxy',
            ])
            apply_legacy_proxy_env(args)
            self.assertIsNone(args.proxy)
            pool = ProxyPool(args.proxy, use_environment=not args.no_proxy)
            with session(pool.choose(), trust_env=pool.use_environment) as client:
                self.assertFalse(client.trust_env)
                self.assertFalse(client.proxies)
            with session('http://explicit.example:8080') as client:
                self.assertFalse(client.trust_env)
                self.assertEqual(client.proxies['https'], 'http://explicit.example:8080')

    def test_explicit_failover_does_not_probe_api_with_head(self):
        pool = ProxyPool(['http://first.example:8080', 'http://second.example:8080'])
        self.assertFalse(pool.use_environment)
        self.assertEqual(pool.choose(), 'http://first.example:8080')
        pool.invalidate()
        self.assertEqual(pool.choose(), 'http://second.example:8080')
        pool.invalidate()
        self.assertEqual(pool.choose(), 'http://first.example:8080')

    def test_legacy_values_only_when_no_explicit_proxy(self):
        with patch.dict(os.environ, {
            'HF_PROXIES': ' http://one.example:80, http://two.example:80 ',
        }, clear=True):
            args = argparse.Namespace(proxy=None, no_proxy=False)
            apply_legacy_proxy_env(args)
            self.assertEqual(normalize_proxies(args.proxy), [
                'http://one.example:80', 'http://two.example:80',
            ])
            args.no_proxy, args.proxy = True, None
            apply_legacy_proxy_env(args)
            self.assertIsNone(args.proxy)
            args.no_proxy, args.proxy = False, ['http://chosen.example:80']
            apply_legacy_proxy_env(args)
            self.assertEqual(args.proxy, ['http://chosen.example:80'])

    def test_discovery_rotates_explicit_proxies(self):
        from hf_downloader import repo

        pool = ProxyPool(['http://first.example:80', 'http://second.example:80'])
        with patch.object(repo, 'discover', side_effect=[
            requests.ConnectionError('first proxy is down'),
            ['discovered'],
        ]) as discover:
            result = repo.discover_with_pool('https://huggingface.co/owner/repo', None, pool)
        self.assertEqual(result, ['discovered'])
        self.assertEqual(discover.call_args_list[0].args[2], 'http://first.example:80')
        self.assertEqual(discover.call_args_list[1].args[2], 'http://second.example:80')

    def test_invalid_proxy_and_custom_endpoint(self):
        with self.assertRaises(ValueError):
            normalize_proxies(['not-a-url'])
        self.assertEqual(
            parse_repo_url('https://mirror.example/datasets/team/data',
                           'https://mirror.example'),
            ('datasets', 'team/data'),
        )
        with self.assertRaises(ValueError):
            parse_repo_url('https://unrelated.example/team/data',
                           'https://mirror.example')

    def test_manifest_urls_with_custom_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps({'files': [
                {'path': 'first.bin', 'size': 0, 'url': API_ROOT + '/u/r/resolve/x/first.bin',
                 'sha256': '0' * 64},
                {'path': 'second.bin', 'size': 0,
                 'url': 'https://cdn.example/u/r/second.bin', 'sha256': '0' * 64},
            ]}))
            files = read_manifest(path, endpoint='https://mirror.example')
            self.assertEqual(files[0].url, 'https://mirror.example/u/r/resolve/x/first.bin')
            self.assertEqual(files[1].url, 'https://cdn.example/u/r/second.bin')


if __name__ == '__main__':
    unittest.main()
