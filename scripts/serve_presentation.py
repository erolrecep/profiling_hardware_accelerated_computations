#!/usr/bin/env python3
"""
Presentation Local Development & Live Server
Author: Recep Erol, PhD (Rice University Center for Research Computing)

Provides an instant local HTTP development server for the Reveal.js presentation with:
- Cache-busting headers (never serve stale CSS/JS during editing)
- Optional live-reload watch mode (auto-refresh browser on file save)
- Automatic free port detection
- Automatic browser opening
"""

import argparse
import http.server
import os
import re
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PRESENTATION_DIR = REPO_ROOT / "presentation"

# Global state for live reload
_FILE_VERSION = 0
_FILE_VERSION_LOCK = threading.Lock()


def get_tree_mtime(directory: Path) -> float:
    """Calculate the maximum modification time across all files in directory."""
    max_mtime = 0.0
    for root, _, files in os.walk(directory):
        for f in files:
            # Skip hidden files
            if f.startswith("."):
                continue
            fp = os.path.join(root, f)
            try:
                mt = os.path.getmtime(fp)
                if mt > max_mtime:
                    max_mtime = mt
            except OSError:
                pass
    return max_mtime


def file_watcher_thread(directory: Path, poll_interval: float = 0.5):
    """Background watcher that increments _FILE_VERSION when any file changes."""
    global _FILE_VERSION
    last_mtime = get_tree_mtime(directory)
    while True:
        time.sleep(poll_interval)
        current_mtime = get_tree_mtime(directory)
        if current_mtime > last_mtime:
            last_mtime = current_mtime
            with _FILE_VERSION_LOCK:
                _FILE_VERSION += 1
            print(f"\n⚡ [LiveReload] Detected change in presentation/ (version {_FILE_VERSION}) -> Reloading browser...")


LIVE_RELOAD_SCRIPT = b"""
<script>
/* Reveal.js Instant Live-Reload Client */
(function() {
  let currentVersion = null;
  async function pollForUpdates() {
    try {
      const resp = await fetch('/__livereload_poll__?t=' + Date.now());
      if (resp.ok) {
        const ver = await resp.text();
        if (currentVersion !== null && ver !== currentVersion) {
          console.log('[LiveReload] Change detected, reloading slides...');
          window.location.reload();
          return;
        }
        currentVersion = ver;
      }
    } catch(err) {}
    setTimeout(pollForUpdates, 600);
  }
  pollForUpdates();
})();
</script>
</body>
"""


class PresentationHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP Request Handler tailored for Reveal.js presentation development."""

    def __init__(self, *args, directory=None, live_reload=False, **kwargs):
        self.live_reload = live_reload
        super().__init__(*args, directory=str(directory), **kwargs)

    def end_headers(self):
        """Append strict cache-busting headers so CSS/HTML edits apply instantly."""
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self):
        # Live reload polling endpoint
        if self.live_reload and self.path.startswith("/__livereload_poll__"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            with _FILE_VERSION_LOCK:
                v = str(_FILE_VERSION).encode("utf-8")
            self.wfile.write(v)
            return

        # If live reload enabled and requesting HTML, inject the client reload snippet
        if self.live_reload and (self.path == "/" or self.path.startswith("/index.html") or self.path.endswith(".html")):
            file_path = self.translate_path(self.path)
            if os.path.isdir(file_path):
                file_path = os.path.join(file_path, "index.html")

            if os.path.exists(file_path) and os.path.isfile(file_path):
                try:
                    with open(file_path, "rb") as f:
                        content = f.read()
                    if b"</body>" in content:
                        content = content.replace(b"</body>", LIVE_RELOAD_SCRIPT)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return
                except Exception as e:
                    # Fallback to standard handler if read fails
                    pass

        # Standard file serving
        return super().do_GET()

    def log_message(self, format, *args):
        # Suppress noisy polling logs in console
        if args and any("/__livereload_poll__" in str(arg) for arg in args):
            return
        # Print concise request log
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def is_port_free(host: str, port: int) -> bool:
    """Check if a host:port is free to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_available_port(host: str, start_port: int, max_attempts: int = 20) -> int:
    """Find the first available port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        if is_port_free(host, p):
            return p
    raise RuntimeError(f"Could not find an open port in range {start_port}-{start_port + max_attempts}")


def main():
    parser = argparse.ArgumentParser(
        description="Local Development & Live Server for Reveal.js Presentation"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port to serve presentation on (default: 8000)"
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--live", "-l", "--watch",
        action="store_true",
        help="Enable live-reload watch mode (auto-refreshes browser on file save)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically launch web browser"
    )
    parser.add_argument(
        "--dir", "-d",
        type=Path,
        default=DEFAULT_PRESENTATION_DIR,
        help=f"Directory to serve (default: {DEFAULT_PRESENTATION_DIR})"
    )

    args = parser.parse_args()

    presentation_dir = args.dir.resolve()
    if not presentation_dir.exists():
        sys.exit(f"Error: Presentation directory '{presentation_dir}' does not exist.")

    # Find free port
    requested_port = args.port
    actual_port = find_available_port(args.host, requested_port)
    if actual_port != requested_port:
        print(f"⚠️  Port {requested_port} is currently busy. Automatically assigned port {actual_port}.")

    url = f"http://{args.host}:{actual_port}/"

    # Start watcher thread if live mode is requested
    if args.live:
        watcher = threading.Thread(
            target=file_watcher_thread,
            args=(presentation_dir,),
            daemon=True
        )
        watcher.start()

    # Create handler factory with directory and live_reload setting
    handler_class = lambda *h_args, **h_kwargs: PresentationHTTPRequestHandler(
        *h_args,
        directory=presentation_dir,
        live_reload=args.live,
        **h_kwargs
    )

    # Initialize ThreadingHTTPServer
    server = http.server.ThreadingHTTPServer((args.host, actual_port), handler_class)

    print("\n" + "=" * 74)
    print("  🚀 PROFILING WORKSHOP — LOCAL PRESENTATION SERVER")
    print("=" * 74)
    print(f"  👉 URL:                {url}")
    print(f"  📂 Serving Directory:  {presentation_dir}")
    print(f"  ⚡ Live-Reload:        {'ENABLED (watching HTML/CSS/images)' if args.live else 'DISABLED (run with --live for auto-reload)'}")
    print(f"  🛡️ Cache Control:      no-cache / no-store (instant CSS updates)")
    print("-" * 74)
    print("  ⌨️  Reveal.js Keynote Shortcuts:")
    print("     [SPACE] / [→]     : Next slide")
    print("     [←]               : Previous slide")
    print("     [S]               : Speaker View (Notes, timer, next slide)")
    print("     [F]               : Toggle Fullscreen")
    print("     [O] / [ESC]       : Slide Overview Grid")
    print("     [B] / [.]         : Blank / Pause Screen")
    print("     [?]               : Reveal.js Built-in Help")
    print("-" * 74)
    print("  Press [Ctrl+C] to stop the server.\n")

    # Auto open browser
    if not args.no_browser:
        def open_browser():
            time.sleep(0.4)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n>> Shutting down presentation server. Bye!")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()

