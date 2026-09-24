"""Shared command-line and network configuration helpers.

The downloader deliberately prefers the standard proxy environment variables
used by ``requests``.  ``HF_PROXY`` and ``HF_PROXIES`` are kept as a small
backwards-compatible convenience for installations that already use them.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse


_PROXY_SCHEMES = {"http", "https", "socks4", "socks4a", "socks5", "socks5h"}


def split_proxy_values(value: str | None) -> list[str]:
    """Split a comma-separated proxy setting, ignoring empty entries."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_proxies(values: list[str] | tuple[str, ...] | None) -> list[str] | None:
    """Normalize and validate explicit proxy values.

    ``argparse`` supplies one item per ``--proxy`` occurrence, while the
    legacy environment variable may contain a comma-separated list.  Accept
    both forms so configuration can move between the command line and the
    environment without changing meaning.
    """
    if values is None:
        return None

    result: list[str] = []
    for value in values:
        for proxy in split_proxy_values(value):
            parsed = urlparse(proxy)
            if parsed.scheme.lower() not in _PROXY_SCHEMES or not parsed.hostname:
                raise ValueError(
                    f"invalid proxy URL {proxy!r}; use http(s):// or install "
                    "requests[socks] for a SOCKS proxy"
                )
            result.append(proxy)
    return result


def legacy_proxy_values() -> list[str]:
    """Read the project's legacy proxy variables, if present."""
    raw = os.getenv("HF_PROXIES")
    if raw is None:
        raw = os.getenv("HF_PROXY")
    return split_proxy_values(raw)


def apply_legacy_proxy_env(args):
    """Fill an unset proxy option from legacy ``HF_*`` variables.

    This is intentionally separate from standard proxy handling: when no
    explicit value is returned, ``requests`` reads HTTP(S)_PROXY, ALL_PROXY
    and NO_PROXY itself.  Callers deploying to another host should not call
    this function with the controller's environment.
    """
    if getattr(args, "proxy", None) is None and not getattr(args, "no_proxy", False):
        values = legacy_proxy_values()
        if values:
            args.proxy = values
    return args


def add_proxy_arguments(parser):
    """Add the common proxy options to an argparse parser."""
    parser.add_argument(
        "--proxy",
        action="append",
        metavar="URL",
        help=(
            "proxy for the execution host; repeat for failover. If omitted, "
            "requests uses HTTP(S)_PROXY/ALL_PROXY"
        ),
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="connect directly and ignore all proxy environment variables",
    )
    return parser
