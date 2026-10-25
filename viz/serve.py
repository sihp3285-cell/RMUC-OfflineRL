"""Serve the replay viewer locally.

``replay.html`` loads its match through ``fetch``, and browsers refuse that over
``file://``, so opening the page by double-clicking it silently shows an empty
field.  This starts a plain static server rooted at ``viz/`` and opens the page.

It also writes ``replays/index.json`` — the list of exported matches — which is
what populates the viewer's match selector.

    python -m viz.serve            # http://localhost:8000/replay.html
    python -m viz.serve --port 8080 --no-open
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import sys
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))


def write_index():
    d = os.path.join(HERE, "replays")
    os.makedirs(d, exist_ok=True)
    names = sorted(f for f in os.listdir(d)
                   if f.endswith(".json") and f != "index.json")
    out = []
    for n in names:
        label = n[:-5]
        try:
            with open(os.path.join(d, n), "r", encoding="utf-8") as f:
                m = json.load(f)["meta"]
            label = (f"{m['red_school']} vs {m['blue_school']} · "
                     f"{m['winner']}方胜 · {m['T']}s")
        except Exception:
            pass                       # a half-written export should not break the list
        out.append(dict(file=n, label=label))
    with open(os.path.join(d, "index.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    idx = write_index()
    if not idx:
        print("no replays yet — run viz.export_replay first", file=sys.stderr)
    else:
        print(f"[serve] {len(idx)} replay(s):")
        for r in idx:
            print(f"   {r['file']}  {r['label']}")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=HERE)
    # Threading matters here: the viewer holds a keep-alive connection open, and
    # a single-threaded server then refuses to answer anything else — the page
    # looks alive while every other request (a second tab, a reload, curl)
    # hangs forever.
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    http.server.ThreadingHTTPServer.daemon_threads = True
    url = f"http://localhost:{args.port}/replay.html"
    with http.server.ThreadingHTTPServer(("", args.port), handler) as httpd:
        print(f"[serve] {url}   (ctrl-c to stop)")
        if not args.no_open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[serve] stopped")


if __name__ == "__main__":
    main()
