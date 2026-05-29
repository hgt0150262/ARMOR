#!/usr/bin/env python3
"""
PyPI Registry Proxy - forwards pip requests to an upstream PyPI mirror.
Used to install Python packages on offline servers via SSH reverse tunnel.

Usage:
    python3 pypi-registry-proxy.py --listen 127.0.0.1 --port 7891 --registry https://pypi.tuna.tsinghua.edu.cn

Then on the remote server (after setting up SSH reverse tunnel):
    pip install --index-url http://127.0.0.1:7891/simple/ --trusted-host 127.0.0.1 <package>
"""
import argparse
import contextlib
import http.server
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

REQUEST_HEADERS = {
    "accept",
    "accept-encoding",
    "accept-language",
    "cache-control",
    "if-none-match",
    "if-modified-since",
    "user-agent",
}

RESPONSE_HEADERS = {
    "cache-control",
    "content-encoding",
    "content-length",
    "content-type",
    "etag",
    "expires",
    "last-modified",
    "location",
}


def normalize_registry(registry: str) -> str:
    parsed = urllib.parse.urlparse(registry)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("registry must be an HTTP(S) URL")
    return registry.rstrip("/")


def iter_header_items(headers) -> Iterable[tuple[str, str]]:
    for key in headers.keys():
        value = headers.get(key)
        if value is not None:
            yield key, value


class PyPIProxyHandler(http.server.BaseHTTPRequestHandler):
    server_version = "PyPIRegistryProxy/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self.forward_request(send_body=True)

    def do_HEAD(self) -> None:
        self.forward_request(send_body=False)

    def do_POST(self) -> None:
        self.send_error(405, "Method Not Allowed")

    def do_PUT(self) -> None:
        self.send_error(405, "Method Not Allowed")

    def do_DELETE(self) -> None:
        self.send_error(405, "Method Not Allowed")

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    @property
    def registry(self) -> str:
        return self.server.registry

    @property
    def timeout(self) -> float:
        return self.server.upstream_timeout

    def forward_request(self, *, send_body: bool) -> None:
        upstream_url = self.build_upstream_url()
        headers = self.build_upstream_headers()
        request = urllib.request.Request(upstream_url, headers=headers, method=self.command)

        self.log_message("-> %s %s", self.command, upstream_url)

        # Create SSL context that doesn't verify (some mirrors have cert issues)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=ctx) as response:
                self.send_upstream_response(response, send_body=send_body)
        except urllib.error.HTTPError as exc:
            self.send_upstream_response(exc, send_body=send_body)
        except Exception as exc:
            self.log_error("upstream error for %s: %s", upstream_url, exc)
            body = f"upstream request failed: {exc}\n".encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if send_body:
                with contextlib.suppress(BrokenPipeError, ConnectionResetError, socket.timeout):
                    self.wfile.write(body)

    def build_upstream_url(self) -> str:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path or "/"
        query = f"?{parsed.query}" if parsed.query else ""

        # If the path already contains a full URL (some pip versions do this)
        if path.startswith("/http://") or path.startswith("/https://"):
            return path.lstrip("/") + query

        # Rewrite /simple/ and /packages/ to upstream
        return f"{self.registry}{path}{query}"

    def build_upstream_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        for key, value in iter_header_items(self.headers):
            lower_key = key.lower()
            if lower_key in REQUEST_HEADERS and lower_key not in HOP_BY_HOP_HEADERS:
                headers[key] = value
        # pip expects application/vnd.pypi.simple.v1+json or text/html
        if "Accept" not in headers:
            headers["Accept"] = "text/html, application/vnd.pypi.simple.v1+json"
        return headers

    def send_upstream_response(self, response, *, send_body: bool) -> None:
        status = response.status if hasattr(response, "status") else response.code

        # Handle redirects: rewrite Location to go through proxy
        location = None
        for key, value in iter_header_items(response.headers):
            if key.lower() == "location":
                location = value
                break

        if status in (301, 302, 303, 307, 308) and location:
            # Follow redirect internally rather than passing it to pip
            # because pip may not be able to reach the redirect target
            redirect_req = urllib.request.Request(location, method=self.command)
            for key, value in self.build_upstream_headers().items():
                redirect_req.add_header(key, value)

            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            try:
                with urllib.request.urlopen(redirect_req, timeout=self.timeout, context=ctx) as redir_resp:
                    self.send_upstream_response(redir_resp, send_body=send_body)
                return
            except urllib.error.HTTPError as exc:
                self.send_upstream_response(exc, send_body=send_body)
                return
            except Exception as exc:
                self.log_error("redirect follow error for %s: %s", location, exc)
                body = f"redirect follow failed: {exc}\n".encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if send_body:
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError, socket.timeout):
                        self.wfile.write(body)
                return

        self.send_response(status)
        content_length = None
        for key, value in iter_header_items(response.headers):
            lower_key = key.lower()
            if lower_key in RESPONSE_HEADERS and lower_key not in HOP_BY_HOP_HEADERS:
                if lower_key != "location":  # Don't forward location
                    self.send_header(key, value)
                if lower_key == "content-length":
                    content_length = value
        self.end_headers()

        if not send_body:
            with contextlib.suppress(Exception):
                response.close()
            return

        bytes_sent = 0
        while True:
            chunk = response.read(1024 * 128)
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
                bytes_sent += len(chunk)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                break


class ThreadingPyPIProxy(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class, *, registry: str, upstream_timeout: float):
        super().__init__(server_address, handler_class)
        self.registry = registry
        self.upstream_timeout = upstream_timeout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward pip/PyPI requests to an upstream PyPI registry."
    )
    parser.add_argument("--listen", default="127.0.0.1", help="listen address, default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=7891, help="listen port, default: 7891")
    parser.add_argument(
        "--registry",
        default="https://pypi.tuna.tsinghua.edu.cn",
        help="upstream PyPI registry, default: https://pypi.tuna.tsinghua.edu.cn",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="upstream timeout seconds, default: 120")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        registry = normalize_registry(args.registry)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    server = ThreadingPyPIProxy(
        (args.listen, args.port),
        PyPIProxyHandler,
        registry=registry,
        upstream_timeout=args.timeout,
    )
    print(f"PyPI registry proxy listening on http://{args.listen}:{args.port}")
    print(f"upstream registry: {registry}")
    print(f"")
    print(f"On remote server, install packages with:")
    print(f"  pip install --index-url http://127.0.0.1:{args.port}/simple/ --trusted-host 127.0.0.1 <package>")
    print(f"")
    print("press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping PyPI registry proxy")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
