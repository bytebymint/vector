"""Automated browser-assisted playUrl resolver for Pattaya public livestreams.

Solves the Cloudflare Turnstile bot challenge offscreen via Chrome CDP
and obtains an authorized playUrl without logging or persisting credentials.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
from urllib.parse import urljoin
import aiohttp
import requests

LOG = logging.getLogger(__name__)
BASE_URL = 'https://livestream.pattaya.go.th/'

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_browser_executable() -> str | None:
    for path in CHROME_CANDIDATES:
        if os.path.isfile(path):
            return path
    # Try finding in PATH
    for name in ('chrome', 'msedge'):
        p = shutil.which(name)
        if p and os.path.isfile(p):
            return p
    return None


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


async def resolve_pattaya_play_url(
    stream_id: str,
    reported_ip: str | None = '83.110.153.183',
    timeout: float = 25.0
) -> str | None:
    """Launch an offscreen browser session to solve Turnstile and fetch playUrl."""
    browser_exe = find_browser_executable()
    if not browser_exe:
        LOG.warning("No Chrome/Edge browser found for automated stream resolution")
        return None

    port = find_free_port()
    temp_dir = tempfile.mkdtemp(prefix='pattaya_cdp_')
    cmd = [
        browser_exe,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={temp_dir}",
        "--window-position=-2500,-2500",
        "--window-size=800,600",
        "--no-first-run",
        "--no-default-browser-check",
        BASE_URL
    ]

    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        LOG.info("Launched browser CDP helper on port %d (PID: %d)", port, proc.pid)

        # Wait for CDP endpoint to become ready
        cdp_ready = False
        start_wait = time.time()
        while time.time() - start_wait < 8.0:
            await asyncio.sleep(0.4)
            try:
                r = requests.get(f"http://127.0.0.1:{port}/json", timeout=1)
                if r.status_code == 200:
                    cdp_ready = True
                    break
            except Exception:
                pass

        if not cdp_ready:
            LOG.warning("Browser failed to expose CDP on port %d within timeout", port)
            return None

        # Fetch page list and attach to target tab
        r = requests.get(f"http://127.0.0.1:{port}/json", timeout=2)
        pages = [p for p in r.json() if p.get("type") == "page"]
        if not pages:
            LOG.warning("No browser page tab found")
            return None

        ws_url = pages[0]["webSocketDebuggerUrl"]

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                msg_id = 0
                async def cdp_call(method, params=None):
                    nonlocal msg_id
                    msg_id += 1
                    cur_id = msg_id
                    payload = {"id": cur_id, "method": method}
                    if params:
                        payload["params"] = params
                    await ws.send_json(payload)
                    while True:
                        msg = await ws.receive()
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            d = json.loads(msg.data)
                            if d.get("id") == cur_id:
                                return d

                await cdp_call("Runtime.enable")
                await cdp_call("Page.enable")

                # Allow page and scripts to settle
                await asyncio.sleep(4.0)

                ip_param = f"?reportedIp={reported_ip}" if reported_ip else ""
                js_script = f"""
                (async () => {{
                    try {{
                        const siteKey = window.__NUXT__?.config?.public?.turnstileSiteKey || "0x4AAAAAADmq2_dfAP_YeJ2A";
                        if (!window.turnstile) {{
                            await new Promise((resolve, reject) => {{
                                const s = document.createElement("script");
                                s.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
                                s.onload = resolve;
                                s.onerror = reject;
                                document.head.appendChild(s);
                            }});
                        }}
                        const token = await new Promise((resolve, reject) => {{
                            const container = document.createElement("div");
                            document.body.appendChild(container);
                            window.turnstile.render(container, {{
                                sitekey: siteKey,
                                callback: (t) => resolve(t),
                                "error-callback": (err) => reject("Turnstile error: " + err),
                                "timeout-callback": () => reject("Turnstile timeout")
                            }});
                        }});
                        
                        const resp = await fetch("/api/public-stream/{stream_id}{ip_param}", {{
                            headers: {{ "X-Turnstile-Token": token }},
                            credentials: "include"
                        }});
                        const data = await resp.json();
                        return {{ status: resp.status, data: data }};
                    }} catch (e) {{
                        return {{ error: e.toString() }};
                    }}
                }})()
                """

                # Retry up to 3 times in case upstream media server had a transient 500
                for attempt in range(3):
                    res = await cdp_call("Runtime.evaluate", {
                        "expression": js_script,
                        "awaitPromise": True,
                        "returnByValue": True
                    })
                    val = res.get("result", {}).get("result", {}).get("value", {})
                    if val.get("status") == 200:
                        play_url = val.get("data", {}).get("details", {}).get("playUrl")
                        if play_url:
                            full_url = urljoin(BASE_URL, play_url.strip())
                            LOG.info("Successfully acquired playUrl via browser helper for %s", stream_id)
                            return full_url
                    LOG.info("Browser helper attempt %d got status %s: %s", attempt + 1, val.get("status"), val.get("data") or val.get("error"))
                    if attempt < 2:
                        await asyncio.sleep(2.0)

        return None

    except Exception as exc:
        LOG.warning("Browser helper failed to resolve playUrl: %s", exc)
        return None
    finally:
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        # Best effort cleanup of temp dir
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
