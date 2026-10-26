"""Exercise real frontend assets with Firefox and an isolated in-memory HTTP fixture.

No business API, database, Redis, model service, or real account is used.
Usage: python tests/browser_ui_check.py --geckodriver /path/to/geckodriver --output-dir /tmp/ui-check
"""

import argparse
import base64
import json
import io
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
from urllib.request import ProxyHandler, Request, build_opener
from urllib.error import HTTPError

WEB = Path(__file__).resolve().parents[1] / "src/pharma_agent/web"
TOKEN = "synthetic-browser-fixture-token"
ANSWER = '**这是浏览器功能检查的合成回答。**\n\n- 第一项\n- 第二项\n\n| 项目 | 状态 |\n| --- | --- |\n| 流式重连 | 正常 |\n\n<img src=x onerror="window.__injected=true">'
HISTORY_ANSWER = "**回答偏好已集中放进设置。**\n\n你可以在左下角打开设置，分别调整回答语言、回答格式与引用样式。\n\n## 更适合阅读的正文\n\n正文默认采用 16px 现代无衬线字体。字体与字号可以在“阅读与显示”中调整，历史对话与上传入口也保留在侧栏中。\n\n| 功能 | 位置 |\n| --- | --- |\n| 回答偏好 | 设置 |\n| 文件暂存 | 附件 |\n| 队列演示 | 开发工具 |"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geckodriver", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "conversations": [
            {
                "id": "conv_history",
                "title": '设置与阅读方式，"选项": []',
                "created_at": "2026-10-22T00:00:00Z",
                "active_run_id": None,
            }
        ],
        "memories": [],
        "jobs": [],
        "runs": {},
        "messages": {
            "conv_history": [
                {"role": "user", "content": "回答偏好和字体在哪里修改？", "run_id": "run_history"},
                {"role": "assistant", "content": HISTORY_ANSWER, "run_id": "run_history"},
            ]
        },
        "requests": [],
        "streams": [],
        "assets": [],
        "imaging_jobs": [],
        "model_offline": False,
        "segmentation_models": [],
        "segmentation_service_status": None,
        "segmentation_service_message": None,
        "reject_next_run": False,
        "hold_jobs_cid": None,
        "jobs_waiting": threading.Event(),
        "release_jobs": threading.Event(),
        "account": {"tenant_id": "tenant_synthetic", "tenant_code": "guke", "tenant_name": "合成测试团队",
                    "user_id": "user_synthetic", "email": "reader@example.test", "role": "owner",
                    "department_id": "dept_bone", "department_name": "骨科",
                    "execution_mode": "ask_for_approval", "scopes": ["tasks:draft", "reviews:submit", "organization:manage"]},
        "departments": [{"id": "dept_bone", "name": "骨科"}],
        "members": [{"id": "user_synthetic", "email": "reader@example.test", "display_name": "合成管理员",
                     "role": "owner", "department_id": "dept_bone", "status": "active"}],
        "reviews": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def send(self, data, code=200, mime="application/json"):
            body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            if urlsplit(self.path).path == "/":
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                    "img-src 'self' blob: data:; frame-ancestors 'none'; base-uri 'self'",
                )
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.send((WEB / "index.html").read_bytes(), mime="text/html; charset=utf-8")
                return
            if path == "/brand-review":
                review = args.output_dir / "brand-preview.html"
                self.send(
                    review.read_text().replace("../../agent_app/src/pharma_agent/web/", "/static/").encode(),
                    mime="text/html; charset=utf-8",
                )
                return
            if path.startswith("/static/") and path.rsplit("/", 1)[1] in {
                "style.css",
                "app.js",
                "ui_helpers.js",
                "imaging.js",
                "hospital.js",
                "surface_viewer.js",
                "logo.svg",
                "favicon.ico",
            } | {p.name for p in WEB.glob("*.svg")} | {p.name for p in WEB.glob("favicon-*.png")}:
                filename = path.rsplit("/", 1)[1]
                self.send(
                    (WEB / filename).read_bytes(),
                    mime="image/svg+xml"
                    if filename.endswith(".svg")
                    else "image/png"
                    if filename.endswith(".png")
                    else "image/x-icon"
                    if filename.endswith(".ico")
                    else "text/css"
                    if filename.endswith(".css")
                    else "application/javascript",
                )
                return
            if path == "/health/ready":
                self.send({"model_mode": "fake", "retrieval_mode": "hash"})
                return
            self.route("GET", path, None)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            data = (
                {"volume": b'.nii.gz"' in raw, "invalid": b"broken.nii.gz" in raw}
                if "multipart/form-data" in self.headers.get("Content-Type", "")
                else json.loads(raw or b"{}")
            )
            self.route("POST", urlsplit(self.path).path, data)

        def do_DELETE(self):
            self.route("DELETE", urlsplit(self.path).path, None)

        def do_PATCH(self):
            self.route("PATCH", urlsplit(self.path).path,
                       json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}"))

        def route(self, method, path, data):
            state["requests"].append({"method": method, "path": path, "data": data})
            if path.startswith("/api/v1/auth/"):
                if path.endswith("/register"):
                    state["account"].update(tenant_code=data.get("tenant_code", "auto-fixture"),
                                            tenant_name=data.get("tenant_name", "合成测试团队"))
                self.send(
                    {"access_token": TOKEN, **state["account"]}
                )
                return
            if path == "/api/v1/users/me/execution-mode":
                state["account"]["execution_mode"] = data["execution_mode"]
                self.send({"execution_mode": data["execution_mode"]})
                return
            if path == "/api/v1/organization":
                if method == "PATCH":
                    state["account"].update(tenant_name=data["name"], tenant_code=data["login_code"])
                    self.send(state["account"])
                else:
                    self.send({"account": state["account"], "departments": state["departments"], "members": state["members"]})
                return
            if path == "/api/v1/users/me":
                self.send(state["account"])
                return
            if path == "/api/v1/organization/departments":
                row = {"id": "dept_" + str(len(state["departments"])), "name": data["name"]}
                state["departments"].append(row)
                self.send(row, 201)
                return
            if path == "/api/v1/organization/members":
                row = {"id": "member_" + str(len(state["members"])), **{k: v for k, v in data.items() if k != "password"}, "status": "active"}
                state["members"].append(row)
                self.send(row, 201)
                return
            if path == "/api/v1/reviews":
                self.send(state["reviews"])
                return
            if path.startswith("/api/v1/jobs/") and path.endswith("/review"):
                job = next(row for row in state["imaging_jobs"] if row["id"] == path.split("/")[-2])
                review = {"id": "review_fixture", "job_id": job["id"], "submitter_id": "user_synthetic",
                          "source_asset_id": job["result"]["source_asset_id"],
                          "result_asset_id": job["result"]["asset_id"], "status": "pending", "comment": "",
                          "created_at": "2026-10-25T00:00:00Z"}
                state["reviews"] = [review]
                self.send(review, 201)
                return
            if path.startswith("/api/v1/reviews/") and path.endswith("/decision"):
                state["reviews"][0].update(status=data["decision"], comment=data["comment"])
                self.send(state["reviews"][0])
                return
            if path.startswith("/api/v1/reviews/") and "/assets/" in path:
                path = "/api/v1/assets/" + path.split("/assets/", 1)[1]
            if self.headers.get("Authorization") != "Bearer " + TOKEN:
                self.send({"error": {"code": "AUTH_REQUIRED", "message": "fixture auth required"}}, 401)
                return
            if path == "/api/v1/imaging/capabilities":
                self.send(
                    {
                        "max_upload_bytes": 64 * 1048576,
                        "retention_days": 30,
                        "segmentation": {
                            "status": "configured" if state["segmentation_models"] else "unconfigured",
                            "models": state["segmentation_models"],
                            "service_status": state["segmentation_service_status"] or ("ready" if state["segmentation_models"] else "unconfigured"),
                            "service_message": state["segmentation_service_message"],
                        },
                    }
                )
                return
            if path.endswith("/surface"):
                # Synthetic ellipsoid phantom geometry, never a patient or a real model result.
                faces = []

                def point(u, v):
                    return [24 * math.cos(v) * math.cos(u), 19 * math.cos(v) * math.sin(u), 31 * math.sin(v)]

                for i in range(30):
                    for j in range(20):
                        u = i * 2 * math.pi / 30
                        v = -math.pi / 2 + j * math.pi / 20
                        faces.append(
                            [
                                n
                                for p in (
                                    point(u, v),
                                    point(u + 2 * math.pi / 30, v),
                                    point(u + 2 * math.pi / 30, v + math.pi / 20),
                                    point(u, v + math.pi / 20),
                                )
                                for n in p
                            ]
                        )
                self.send(
                    {
                        "segments": [
                            {
                                "id": 1,
                                "name": "fixture_label",
                                "color": [65, 190, 163],
                                "quads": [n for f in faces for n in f],
                            }
                        ],
                        "face_count": len(faces),
                        "stride": 1,
                    }
                )
                return
            if path == "/api/v1/model-info":
                self.send(
                    {
                        "chat_models": [
                            {"alias": "fixture-chat", "name": "PharmaLLM-1.0"},
                            {"alias": "fixture-alternative", "name": "Synthetic alternative"},
                        ],
                        "default_alias": "fixture-chat",
                        "model_id": "fixture-only",
                        "revision": "synthetic",
                        "graph_version": "fixture",
                        "status": "unavailable" if state["model_offline"] else "demo",
                    }
                )
                return
            if path.endswith("/imaging-jobs"):
                cid = path.split("/")[-2]
                if state["hold_jobs_cid"] == cid:
                    state["jobs_waiting"].set()
                    state["release_jobs"].wait(5)
                self.send(
                    [
                        row
                        for row in state["imaging_jobs"]
                        if row["input"].get("conversation_id", "conv_history") == cid
                    ]
                )
                return
            if path == "/api/v1/imaging/uploads":
                if data.get("invalid"):
                    self.send(
                        {"error": {"code": "INVALID_IMAGE", "message": "Synthetic upload failure"}}, 422
                    )
                    return
                row = {
                    "asset_id": "asset_" + str(len(state["assets"])),
                    "name": "phantom.nii.gz" if data.get("volume") else "notes.txt",
                    "kind": "volume" if data.get("volume") else "attachment",
                    "status": "ready",
                    "mime": "application/gzip" if data.get("volume") else "text/plain",
                    "size_bytes": 1024,
                    "format": "NIfTI" if data.get("volume") else "TXT",
                    "dimensions": [64, 64, 32],
                    "spacing": [1, 1, 2],
                    "units": "mm",
                    "display_dimensions": [64, 64, 32],
                    "display_spacing": [1, 1, 2],
                    "display_orientation": ["R", "A", "S"],
                    "intensity_range": [0, 100],
                    "channels": 1,
                    "modality": "T1",
                }
                state["assets"].append(row)
                self.send(row, 201)
                return
            if path.startswith("/api/v1/assets/") and path.endswith("/metadata"):
                self.send(next(r for r in state["assets"] if r["asset_id"] == path.split("/")[-2]))
                return
            if path.startswith("/api/v1/assets/") and path.endswith("/slice"):
                from PIL import Image, ImageDraw

                picture = Image.new("RGB", (64, 64), "#111820")
                draw = ImageDraw.Draw(picture)
                draw.ellipse((8, 5, 56, 59), fill="#666e76")
                draw.ellipse((18, 13, 46, 52), fill="#9da6ae")
                draw.ellipse((25, 22, 39, 43), fill="#384650")
                if "mask_id=" in self.path and "1" in parse_qs(urlsplit(self.path).query).get("labels", [""])[
                    0
                ].split(","):
                    draw.ellipse((18, 13, 46, 52), outline="#41bea3", width=3)
                output = io.BytesIO()
                picture.save(output, format="PNG")
                self.send(output.getvalue(), mime="image/png")
                return
            if path.startswith("/api/v1/conversations/") and method == "DELETE":
                cid = path.split("/")[-1]
                state["conversations"] = [r for r in state["conversations"] if r["id"] != cid]
                self.send({"deleted": True})
                return
            if path == "/api/v1/conversations":
                if method == "POST":
                    row = {
                        "id": "conv_" + str(len(state["conversations"])),
                        "title": data["title"],
                        "created_at": "2026-10-22T02:00:00Z",
                        "active_run_id": None,
                    }
                    state["conversations"].insert(0, row)
                    state["messages"][row["id"]] = []
                    self.send(row, 201)
                else:
                    self.send(state["conversations"])
            elif path.endswith("/messages"):
                self.send(state["messages"].get(path.split("/")[-2], []))
            elif path.endswith("/runs") and method == "POST":
                if state["reject_next_run"]:
                    state["reject_next_run"] = False
                    self.send(
                        {"error": {"code": "SYNTHETIC_REJECTION", "message": "Synthetic send rejection"}}, 503
                    )
                    return
                cid, rid = path.split("/")[-2], "run_" + str(len(state["runs"]))
                run = {
                    "run_id": rid,
                    "conversation_id": cid,
                    "status": "running",
                    "answer": None,
                    "error_code": None,
                    "citations": [],
                    "calls": 0,
                    "cancel_test": "停止测试" in data["message"],
                }
                state["runs"][rid] = run
                state["messages"][cid].append(
                    {
                        "role": "user",
                        "content": data["message"],
                        "run_id": rid,
                        "asset_ids": data.get("asset_ids", []),
                    }
                )
                next(row for row in state["conversations"] if row["id"] == cid)["active_run_id"] = rid
                self.send({"run_id": rid}, 202)
            elif path == "/api/v1/runs/run_history":
                self.send(
                    {"status": "completed", "answer": HISTORY_ANSWER, "citations": [], "error_code": None}
                )
            elif path.endswith("/cancel"):
                state["runs"][path.split("/")[-2]]["status"] = "cancelled"
                self.send({"status": "cancelled"})
            elif path.endswith("/events"):
                rid = path.split("/")[-2]
                run = state["runs"][rid]
                cursor = self.headers.get("Last-Event-ID", "0")
                state["streams"].append(cursor)

                def frame(number, event, body):
                    return f"id: {number}\nevent: {event}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"

                if run["status"] == "cancelled":
                    stream = frame(3, "error", {"code": "RUN_CANCELLED", "message": "运行已取消"})
                elif run["calls"] == 0:
                    stream = frame(1, "status", {"phase": "generating"}) + frame(
                        2, "delta", {"text": ANSWER[:18]}
                    )
                elif run["cancel_test"]:
                    stream = ": heartbeat\n\n"
                else:
                    stream = (
                        frame(2, "delta", {"text": ANSWER[:18]})
                        + frame(3, "delta", {"text": ANSWER[18:]})
                        + frame(
                            4,
                            "completed",
                            {
                                "citations": [],
                                "warning_code": "OUTPUT_TOKEN_LIMIT",
                                "trace_id": "synthetic-trace",
                            },
                        )
                    )
                    run.update(status="completed", answer=ANSWER, error_code="OUTPUT_TOKEN_LIMIT")
                    state["messages"][run["conversation_id"]].append(
                        {"role": "assistant", "content": ANSWER, "run_id": rid}
                    )
                if run["status"] in {"completed", "cancelled"}:
                    next(row for row in state["conversations"] if row["id"] == run["conversation_id"])[
                        "active_run_id"
                    ] = None
                run["calls"] += 1
                self.send(stream.encode(), mime="text/event-stream")
            elif path.startswith("/api/v1/runs/"):
                self.send(state["runs"][path.split("/")[-1]])
            elif path == "/api/v1/memories":
                if method == "POST":
                    assert data["consent"] is True
                    row = next((row for row in state["memories"] if row["kind"] == data["kind"]), None)
                    if row:
                        row["value"] = data["value"]
                    else:
                        row = {
                            "id": "mem_" + str(len(state["memories"])),
                            "kind": data["kind"],
                            "value": data["value"],
                        }
                        state["memories"].append(row)
                    self.send(row)
                else:
                    self.send(state["memories"])
            elif path.startswith("/api/v1/memories/") and method == "DELETE":
                state["memories"] = [row for row in state["memories"] if row["id"] != path.split("/")[-1]]
                self.send({"deleted": True})
            elif path == "/api/v1/jobs/drafts":
                row = {"id": "job_synthetic", "status": "draft", "result": {}, "error_code": None}
                state["jobs"].append(row)
                self.send(row, 201)
            elif path == "/api/v1/jobs":
                self.send(state["jobs"])
            elif path.endswith("/confirm"):
                assert data["confirmed"] is True
                state["jobs"][0]["status"] = "succeeded"
                self.send(state["jobs"][0])
            elif path == "/api/v1/assets":
                self.send(state["assets"])
            else:
                self.send({"error": {"message": "Unknown fixture route: " + path}}, 404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        driver_port = sock.getsockname()[1]
    opener = build_opener(ProxyHandler({}))

    def request(method, path, data=None):
        body = json.dumps(data).encode() if data is not None else None
        req = Request(
            f"http://127.0.0.1:{driver_port}" + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=30) as response:
            result = json.load(response)
        value = result.get("value")
        if isinstance(value, dict) and value.get("error"):
            raise RuntimeError(value)
        return value

    log = (args.output_dir / "browser-driver.log").open("w")
    process = subprocess.Popen(
        [str(args.geckodriver), "--host", "127.0.0.1", "--port", str(driver_port)],
        stdout=log,
        stderr=log,
        start_new_session=True,
        env={**os.environ, "LIBGL_ALWAYS_SOFTWARE": "1", "MOZ_WEBRENDER": "0"},
    )
    session = None
    checks = []
    try:
        for _ in range(100):
            try:
                request("GET", "/status")
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError("Browser driver stopped; see browser-driver.log")
                time.sleep(0.1)
        capabilities = {
            "capabilities": {
                "alwaysMatch": {
                    "browserName": "firefox",
                    "moz:firefoxOptions": {
                        "args": ["-headless"],
                        "prefs": {
                            "network.proxy.type": 0,
                            "gfx.webrender.software": True,
                            "layers.acceleration.disabled": True,
                            "media.hardware-video-decoding.enabled": False,
                        },
                    },
                }
            }
        }
        session = request("POST", "/session", capabilities)["sessionId"]
        prefix = "/session/" + session

        def execute(script):
            return request("POST", prefix + "/execute/sync", {"script": script, "args": []})

        def until(script, description, seconds=10):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                try:
                    ready = execute("return Boolean(" + script + ")")
                except HTTPError as exc:
                    detail = exc.read().decode()
                    if not any(
                        text in detail
                        for text in (
                            "document was unloaded",
                            "browsing context",
                            "document.querySelector(...) is null",
                        )
                    ):
                        raise RuntimeError(detail) from exc
                    ready = False
                if ready:
                    checks.append(description)
                    return
                time.sleep(0.1)
            raise AssertionError(
                description
                + ": "
                + str(execute("return document.getElementById('error-message').textContent"))
            )

        def screenshot(name):
            # Capture settled UI, after transient toasts and short color transitions finish.
            for _ in range(45):
                if execute(
                    "return !document.getElementById('toast') || document.getElementById('toast').hidden"
                ):
                    break
                time.sleep(0.1)
            time.sleep(0.2)
            (args.output_dir / name).write_bytes(base64.b64decode(request("GET", prefix + "/screenshot")))

        request("POST", prefix + "/window/rect", {"width": 1600, "height": 1100})
        request("POST", prefix + "/url", {"url": f"http://127.0.0.1:{server.server_port}/"})
        until(
            "document.getElementById('health').textContent === '演示模式'",
            "service status loads without exposing engineering configuration",
        )
        screenshot("01-login.png")
        execute("document.getElementById('tenant').value='20_26';document.getElementById('tenant-name').value='合成注册团队';document.getElementById('email').value='registration@example.test';document.getElementById('password').value='TestPass123';document.getElementById('register').click()")
        until("!document.getElementById('auth-field-error').hidden && document.getElementById('auth-field-error').textContent.includes('团队登录代号') && document.activeElement.id==='tenant'",
              "invalid registration code is explained beside the form and focused before sending")
        assert not any(row["path"].endswith("/auth/register") for row in state["requests"])
        execute("document.getElementById('tenant').value='2026';document.getElementById('tenant').dispatchEvent(new Event('input'));document.getElementById('password').value='12345';document.getElementById('register').click()")
        until("!document.getElementById('auth-field-error').hidden && document.getElementById('auth-field-error').textContent.includes('6–12') && document.activeElement.id==='password'", "registration rejects a five-character password with explicit 6-to-12 guidance")
        execute("document.getElementById('password').value='1234567890123';document.getElementById('register').click()")
        until("!document.getElementById('auth-field-error').hidden && document.getElementById('auth-field-error').textContent.includes('6–12')", "registration rejects a thirteen-character password before sending")
        assert not any(row["path"].endswith("/auth/register") for row in state["requests"])
        execute("document.getElementById('password').value='123456';document.getElementById('password').dispatchEvent(new Event('input'));document.getElementById('register').click()")
        until("!document.getElementById('workspace').hidden && document.getElementById('settings-dialog').open && document.getElementById('identity').textContent==='2026'",
              "pure numeric code 2026 registers and appears in the new account")
        submitted = next(row["data"] for row in state["requests"] if row["path"].endswith("/auth/register"))
        assert submitted["tenant_code"] == "2026" and submitted["tenant_name"] == "合成注册团队"
        assert submitted["password"] == "123456"
        checks.append("six-digit numeric password is accepted without extra composition requirements")
        checks.append("registration sends the numeric code as a string without adding prefixes")
        screenshot("21-numeric-registration.png")
        state["account"].update(tenant_code="guke", tenant_name="合成测试团队")
        execute("document.getElementById('logout').click()")
        request("POST", prefix + "/url", {"url": f"http://127.0.0.1:{server.server_port}/"})
        until("!document.getElementById('auth').hidden", "registration validation does not prevent returning to legacy-ID login")
        execute(
            "document.getElementById('tenant').value='tenant_synthetic'; document.getElementById('email').value='reader@example.test'; document.getElementById('password').value='TestPass123'; document.getElementById('auth-form').requestSubmit();"
        )
        until(
            "!document.getElementById('workspace').hidden && document.querySelector('.history-item')",
            "login and history load",
        )
        execute("document.querySelector('.history-item').click()")
        until(
            "document.querySelector('.answer-body strong') && document.querySelector('.answer-table')",
            "history renders modern text, emphasis and tables",
        )
        until(
            "!document.getElementById('run-info').textContent.includes('trace_id')",
            "run diagnostics stay out of the main chat",
        )
        until(
            "document.getElementById('chat-model').options.length === 2",
            "configured chat model choices are available beside upload",
        )
        assert execute(
            "return document.querySelector('.topbar #health') === null && document.querySelector('.topbar #model-name') === null"
        )
        assert execute(
            "return !document.querySelector('.brand').textContent.includes('V1.1') && !!document.querySelector('.brand-logo')"
        )
        checks.append("brand logo replaces version text and technical model tooltip is removed")
        screenshot("02-chat.png")
        execute("document.getElementById('execution-mode').value='full_access'; document.getElementById('execution-mode').dispatchEvent(new Event('change'))")
        until("document.getElementById('execution-mode-settings').value==='full_access' && !document.getElementById('execution-mode').disabled",
              "Full access selection is saved and mirrored in settings")
        assert state["account"]["execution_mode"] == "full_access"
        execute("document.getElementById('execution-mode').value='ask_for_approval'; document.getElementById('execution-mode').dispatchEvent(new Event('change'))")
        until("document.getElementById('execution-mode-settings').value==='ask_for_approval' && !document.getElementById('execution-mode').disabled",
              "Ask for approval can be restored for future requests")
        execute("document.querySelector('[data-open=settings]').click();document.querySelector('[data-tab=organization]').click()")
        until("document.querySelectorAll('.hospital-member').length===1", "administrator can open department and member management")
        execute("document.getElementById('team-code').value='bone-lab';document.getElementById('team-form').requestSubmit()")
        until("document.getElementById('identity').textContent==='bone-lab'", "readable team login code replaces the internal ID in account settings")
        execute("document.getElementById('department-name').value='影像科';document.getElementById('department-form').requestSubmit()")
        until("document.getElementById('member-department').options.length===2", "creating a department updates the member assignment choices")
        execute("document.getElementById('member-name').value='合成审核员';document.getElementById('member-email').value='reviewer@example.test';document.getElementById('member-password').value='TestPass123';document.getElementById('member-role').value='reviewer';document.getElementById('member-form').requestSubmit()")
        until("document.querySelectorAll('.hospital-member').length===2 && !document.getElementById('member-password').value", "member creation refreshes the list and clears the password field")
        screenshot("18-organization.png")
        execute("document.getElementById('settings-dialog').close()")
        until(
            "getComputedStyle(document.querySelector('.answer-body')).fontFamily === getComputedStyle(document.getElementById('prompt')).fontFamily && getComputedStyle(document.querySelector('.sidebar')).fontFamily === getComputedStyle(document.getElementById('prompt')).fontFamily",
            "sidebar, composer and answers share one font family",
        )
        execute(
            "document.getElementById('theme').value='light';document.getElementById('theme').dispatchEvent(new Event('change'))"
        )
        until(
            "document.documentElement.dataset.theme === 'light' && getComputedStyle(document.querySelector('.brand-logo')).backgroundColor === 'rgb(23, 129, 91)' && getComputedStyle(document.getElementById('new-conversation')).backgroundColor === 'rgb(255, 255, 255)' && document.getElementById('favicon').href.endsWith('favicon-light.svg')",
            "light theme uses the exact requested logo color and matching favicon",
        )
        screenshot("11-light-chat.png")
        execute("document.getElementById('toggle-sidebar').click()")
        until(
            "document.querySelector('.topbar-brand').getBoundingClientRect().width === 28 && getComputedStyle(document.querySelector('.topbar-brand')).display !== 'none'",
            "collapsed sidebar retains the independent brand icon in the header",
        )
        screenshot("12-collapsed-light.png")
        execute(
            "document.getElementById('theme').value='dark';document.getElementById('theme').dispatchEvent(new Event('change'))"
        )
        until(
            "document.documentElement.dataset.theme === 'dark' && getComputedStyle(document.querySelector('.topbar-brand')).backgroundColor === 'rgb(107, 221, 173)' && document.getElementById('favicon').href.endsWith('favicon-dark.svg')",
            "dark theme uses the exact requested logo color and matching favicon",
        )
        screenshot("13-collapsed-dark.png")
        execute("document.getElementById('toggle-sidebar').click()")
        execute(
            "document.querySelector('[data-open=settings]').click(); document.querySelector('[data-tab=appearance]').click(); document.getElementById('font-size').value='22'; document.getElementById('font-size').dispatchEvent(new Event('change'));"
        )
        until(
            "getComputedStyle(document.documentElement).getPropertyValue('--body-size').trim() === '22px'",
            "font size changes in the browser",
        )
        screenshot("09-font-settings.png")
        execute(
            "document.querySelector('[data-tab=preferences]').click(); document.getElementById('language').value='zh'; document.getElementById('answer-format').value='bullets'; document.getElementById('remember').checked=true; document.getElementById('memory-form').requestSubmit();"
        )
        until(
            "document.querySelectorAll('.saved-preference').length === 2",
            "preferences persist through the existing consent API",
        )
        screenshot("03-settings.png")
        assert all(
            row["data"]["consent"] is True
            for row in state["requests"]
            if row["method"] == "POST" and row["path"] == "/api/v1/memories"
        )
        execute("document.getElementById('clear-preferences').click()")
        until(
            "document.querySelectorAll('.saved-preference').length === 0", "saved preferences can be cleared"
        )
        execute(
            "document.querySelector('[data-tab=developer]').click(); document.querySelector('.developer-details').open=true; document.getElementById('draft-job').click()"
        )
        until(
            "document.querySelector('.job-item button')", "demo draft remains available under developer tools"
        )
        assert state["jobs"][0]["status"] == "draft"
        execute("document.querySelector('.job-item button').click()")
        until(
            "document.querySelector('.job-item').textContent.includes('已完成')",
            "demo task requires a separate confirmation",
        )
        execute(
            "document.getElementById('settings-dialog').close(); document.querySelector('[data-open=files]').click(); const file=new File(['synthetic'], 'notes.txt', {type:'text/plain'}); const transfer=new DataTransfer(); transfer.items.add(file); document.getElementById('asset-file').files=transfer.files; document.getElementById('upload-form').requestSubmit();"
        )
        until(
            "document.querySelector('.asset-item') && document.getElementById('asset-result').textContent.includes('已解析')",
            "attachment upload publishes real parsed-state UI",
        )
        screenshot("04-attachments.png")
        execute(
            "document.getElementById('files-dialog').close(); document.querySelector('.attachment-chip button').click()"
        )
        until(
            "document.querySelectorAll('.attachment-chip').length === 0",
            "attachment removal only detaches the composer reference",
        )
        assert len(state["assets"]) == 1
        execute(
            "document.getElementById('font-size').value='16';document.getElementById('font-size').dispatchEvent(new Event('change'))"
        )
        execute(
            "const f=new File(['synthetic'], 'phantom.nii.gz'); const dt=new DataTransfer(); dt.items.add(f); document.getElementById('composer-file').files=dt.files; document.getElementById('composer-file').dispatchEvent(new Event('change'))"
        )
        until(
            "document.querySelector('.attachment-chip')?.textContent.includes('可使用') && document.getElementById('image-workspace').hidden && document.getElementById('upload-modality').value === 'T1'",
            "NIfTI upload defaults to editable T1 and stays attached without opening the workspace",
        )
        screenshot("14-upload-attached.png")
        execute("document.querySelector('.attachment-chip button').click()")
        until(
            "!document.getElementById('image-workspace').hidden && document.querySelectorAll('.viewport').length === 4",
            "explicit View action opens the four-pane workspace",
        )
        until(
            "document.querySelectorAll('.viewport canvas').length === 3 && document.querySelector('.viewport canvas').width > 0",
            "authenticated slice images render on canvas",
        )
        until(
            "document.getElementById('seg-status').textContent.includes('未配置') && document.getElementById('ask-segmentation').disabled",
            "unconfigured segmentation stays explicit and disabled",
        )
        assert (
            execute(
                "return [...document.querySelectorAll('.viewport:first-child .direction')].map(x=>x.textContent).join('')"
            )
            == "APLR"
        )
        execute(
            "const s=document.querySelector('.viewport input[type=range]'); s.value='3'; s.dispatchEvent(new Event('input'))"
        )
        until(
            "document.querySelector('.viewport-heading').textContent.includes('4 / 32')",
            "slice sliders update actual slice requests",
        )
        state["segmentation_models"] = [
            {
                "id": "fixture-segmentation",
                "name": "股骨头与髋臼",
                "modalities": ["T1"],
                "channels": 1,
                "task_types": ["femoral_head_segmentation"],
                "labels": {"0": "background", "1": "fixture_label"},
            }
        ]
        execute("document.dispatchEvent(new Event('visibilitychange'))")
        until(
            "!document.getElementById('seg-model') && !document.getElementById('ask-segmentation').disabled",
            "segmentation target dropdown is removed while supported image requests remain enabled",
        )
        state["segmentation_service_status"] = "loading"
        state["segmentation_service_message"] = "分割模型正在加载，请等待终端 C 显示就绪"
        execute("document.dispatchEvent(new Event('visibilitychange'))")
        until("document.getElementById('seg-status').textContent.includes('正在加载') && document.getElementById('ask-segmentation').disabled", "resident model loading is distinct from a ready service")
        state["segmentation_service_status"] = "offline"
        state["segmentation_service_message"] = "无法连接分割服务，请在终端 C 使用 --seg-gpu 指定显卡启动"
        execute("document.dispatchEvent(new Event('visibilitychange'))")
        until("document.getElementById('seg-status').textContent.includes('--seg-gpu') && document.getElementById('ask-segmentation').disabled", "offline segmentation shows an actionable startup message")
        state["segmentation_service_status"] = None
        state["segmentation_service_message"] = None
        execute("document.dispatchEvent(new Event('visibilitychange'))")
        until("!document.getElementById('ask-segmentation').disabled", "segmentation controls recover after the resident service becomes ready")
        execute("document.getElementById('ask-segmentation').click()")
        assert execute("return document.getElementById('prompt').value === '请分割这份影像中的'")
        checks.append(
            "user supplies the segmentation target in natural language without a forced model choice"
        )
        screenshot("07-imaging.png")
        execute(
            "document.getElementById('font-size').value='16';document.getElementById('font-size').dispatchEvent(new Event('change'))"
        )
        source = state["assets"][-1]
        mask = {
            **source,
            "asset_id": "asset_fixture_mask",
            "kind": "mask",
            "name": "synthetic-mask.nii.gz",
            "source_asset_id": source["asset_id"],
            "labels": {"0": "background", "1": "fixture_label"},
            "measurements": [
                {"label": 1, "name": "fixture_label", "volume_mm3": 1000, "basis": "synthetic fixture only"}
            ],
        }
        state["assets"].append(mask)
        job = {
            "id": "job_imaging_fixture",
            "status": "failed",
            "input": {
                "file_id": source["asset_id"],
                "file_name": "Synthetic test phantom",
                "model_id": "fixture-segmentation",
            },
            "elapsed_seconds": 4,
            "error_code": "SEGMENTATION_UNAVAILABLE",
            "result": {"phase": "failed", "error_message": "Synthetic service offline"},
        }
        state["imaging_jobs"] = [job]
        execute("window.ImagingUI.refreshJobs()")
        until(
            "document.querySelector('.imaging-job').textContent.includes('Synthetic service offline')",
            "segmentation failures display actionable state without fake completion",
        )
        assert not execute(
            "return [...document.querySelectorAll('.imaging-job button')].some(b=>b.textContent.includes('取消'))"
        )
        checks.append("no segmentation cancellation button is offered without backend support")
        job.update(
            status="succeeded",
            error_code=None,
            result={
                "phase": "completed",
                "asset_id": mask["asset_id"],
                "source_asset_id": source["asset_id"],
                "labels": mask["labels"],
                "model_id": "fixture-segmentation",
            },
        )
        execute("window.ImagingUI.refreshJobs()")
        until(
            "document.querySelector('.imaging-job').textContent.includes('打开结果')",
            "persisted result jobs expose reopen and NIfTI download actions",
        )
        until(
            "!document.getElementById('image-workspace').hidden && !document.getElementById('download-mask').hidden && document.querySelector('.mask-legend input')",
            "completed segmentation automatically updates the workspace overlay without clicking Open result",
        )
        until(
            "document.querySelector('.viewport:first-child .viewport-heading').textContent.includes('4 / 32')",
            "automatic overlay refresh keeps the current slice instead of resetting the viewer",
        )
        until(
            "Number(document.querySelector('.surface-host canvas')?.dataset.renderedFaces)>0",
            "automatic completion refresh also renders the actual supplied 3D result mesh",
        )
        screenshot("17-auto-result.png")
        execute("document.getElementById('close-image').click();window.ImagingUI.refreshJobs()")
        time.sleep(0.3)
        assert execute("return document.getElementById('image-workspace').hidden")
        checks.append("repeated polling respects an explicitly closed result workspace")
        execute("document.querySelector('.imaging-job button').click()")
        until(
            "document.querySelector('.mask-legend input') && !document.getElementById('download-mask').hidden",
            "reopened masks expose configured labels and download",
        )
        execute(
            "document.getElementById('mask-opacity').value='0.2';document.getElementById('mask-opacity').dispatchEvent(new Event('input'));document.querySelector('.mask-legend input').click()"
        )
        until(
            "!document.querySelector('.mask-legend input').checked",
            "mask opacity and label switches control overlay requests",
        )
        until(
            "document.querySelector('.viewport:first-child .viewport-heading').textContent.includes('16 / 32')",
            "result viewer reloads actual source slices",
        )
        until(
            "document.querySelector('.viewport canvas')?.dataset.loadedLabels === ''",
            "overlay switches finish rendering the requested label selection",
        )
        execute(
            "document.querySelector('.mask-legend input').click();document.getElementById('mask-opacity').value='0.45';document.getElementById('mask-opacity').dispatchEvent(new Event('input'))"
        )
        until(
            "document.querySelector('.viewport canvas')?.dataset.loadedLabels === '1'",
            "overlay can be restored after hiding a label",
        )
        until(
            "Number(document.querySelector('.surface-host canvas')?.dataset.renderedFaces)>0",
            "real 3D renderer displays the supplied result mesh",
        )
        execute(
            "document.querySelector('.surface-host canvas').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true,cancelable:true}))"
        )
        assert execute(
            "return getComputedStyle(document.getElementById('viewports')).gridTemplateAreas.includes('axial surface')"
        )
        checks.append(
            "four-pane positions keep axial upper-left, surface upper-right, coronal lower-left and sagittal lower-right"
        )
        screenshot("08-result.png")
        execute(
            "document.getElementById('font-size').value='22';document.getElementById('font-size').dispatchEvent(new Event('change'))"
        )
        request("POST", prefix + "/window/rect", {"width": 390, "height": 844})
        until(
            "getComputedStyle(document.querySelector('.main')).display === 'none' && !document.getElementById('image-workspace').hidden",
            "mobile imaging uses a full-width workspace",
        )
        screenshot("10-mobile-imaging.png")
        execute("document.getElementById('mobile-image-tab').click()")
        until(
            "getComputedStyle(document.querySelector('.main')).display !== 'none' && getComputedStyle(document.getElementById('image-workspace')).display === 'none'",
            "mobile image tab returns to the conversation",
        )
        request("POST", prefix + "/window/rect", {"width": 1600, "height": 1100})
        execute("document.getElementById('image-workspace').scrollTop=10000")
        assert execute(
            "const h=document.querySelector('.image-heading').getBoundingClientRect(),p=document.getElementById('image-workspace').getBoundingClientRect();return h.top >= p.top-1 && h.bottom <= p.bottom"
        )
        checks.append("workspace close button remains visible while scrolling")
        execute("document.getElementById('close-image-from-chat').click()")
        until(
            "document.getElementById('image-workspace').hidden",
            "image workspace closes without losing the conversation",
        )
        state["model_offline"] = True
        execute("document.dispatchEvent(new Event('visibilitychange'))")
        until(
            "document.getElementById('health').textContent.includes('离线')",
            "model availability follows server state rather than model text",
        )
        state["model_offline"] = False
        execute("document.dispatchEvent(new Event('visibilitychange'))")
        until(
            "document.getElementById('health').textContent.includes('演示模式')",
            "model availability can recover",
        )
        execute(
            "const f=new File(['broken'], 'broken.nii.gz');const dt=new DataTransfer();dt.items.add(f);document.getElementById('composer-file').files=dt.files;document.getElementById('composer-file').dispatchEvent(new Event('change'))"
        )
        until(
            "document.querySelector('.attachment-chip.failed') && document.getElementById('error-message').textContent.includes('Synthetic upload failure')",
            "upload errors remain failed attachments and are never marked ready",
        )
        execute(
            "document.getElementById('dismiss-error').click();document.getElementById('new-conversation').click();document.getElementById('prompt').value='中文输入法测试';const e=new KeyboardEvent('keydown',{key:'Enter',isComposing:true,bubbles:true,cancelable:true});document.getElementById('prompt').dispatchEvent(e)"
        )
        assert len(state["runs"]) == 0
        checks.append("IME composition Enter does not send a request")
        execute("document.getElementById('chat-model').value='fixture-alternative'")
        execute(
            "document.getElementById('files-dialog').close(); document.getElementById('new-conversation').click(); document.getElementById('prompt').value='测试流式断线恢复'; document.getElementById('chat-form').requestSubmit();"
        )
        until(
            "document.getElementById('run-info').textContent.includes('长度上限') && !document.getElementById('send').disabled",
            "stream reconnect finishes and preserves output-limit warning",
            15,
        )
        assert (
            next(
                row
                for row in reversed(state["requests"])
                if row["method"] == "POST" and row["path"].endswith("/runs")
            )["data"]["model_alias"]
            == "fixture-alternative"
        )
        checks.append("selected chat model is passed as an allowlisted alias in the run request")
        text = execute("return document.querySelector('.answer-body').textContent")
        assert text.count("这是浏览器功能检查的合成回答") == 1
        assert "2" in state["streams"]
        until(
            "document.querySelectorAll('.answer-body img, .answer-body a').length === 0 && !window.__injected",
            "model HTML is inert after Markdown rendering",
        )
        stored = execute("return JSON.parse(localStorage.getItem('pharma-reading-v3'))")
        assert stored == {"font": "modern", "size": 22, "density": "comfortable"}
        assert TOKEN not in execute("return JSON.stringify(localStorage)")
        checks.append("browser persistence contains display settings only, never tokens")
        execute("document.querySelector('.answer-actions button:nth-child(2)').click()")
        until(
            "document.querySelectorAll('.answer').length === 2 && !document.getElementById('send').disabled",
            "regenerate submits a fresh run and preserves previous answers",
            15,
        )
        assert len(state["runs"]) == 2
        execute(
            "document.getElementById('new-conversation').click(); document.getElementById('prompt').value='停止测试'; document.getElementById('chat-form').requestSubmit();"
        )
        until("!document.getElementById('cancel').hidden", "stop control appears for active runs")
        execute("document.getElementById('cancel').click()")
        until(
            "document.getElementById('run-info').textContent.includes('已停止回答') && !document.getElementById('send').disabled",
            "stop requests cancel the run and restore the composer",
            15,
        )
        execute(
            "const dt=new DataTransfer();dt.items.add(new File(['synthetic'],'followup.nii.gz'));document.getElementById('composer-file').files=dt.files;document.getElementById('composer-file').dispatchEvent(new Event('change'))"
        )
        until(
            "document.querySelector('.attachment-chip')?.textContent.includes('可使用')",
            "new attachment is ready for a send-and-retry check",
        )
        state["reject_next_run"] = True
        execute(
            "document.getElementById('prompt').value='测试附件发送重试';document.getElementById('chat-form').requestSubmit()"
        )
        until(
            "document.getElementById('error-message').textContent.includes('Synthetic send rejection') && !document.getElementById('send').disabled",
            "rejected message submission restores the composer",
        )
        assert execute("return document.querySelectorAll('.attachment-chip').length === 1")
        checks.append("rejected submissions preserve unsent attachments for retry")
        execute(
            "document.getElementById('dismiss-error').click();document.getElementById('chat-form').requestSubmit()"
        )
        until(
            "document.querySelectorAll('.attachment-chip').length === 0 && document.querySelector('.user-message .message-attachments button')",
            "accepted send clears the composer attachment while keeping its message reference",
        )
        until("!document.getElementById('send').disabled", "attachment retry finishes streaming")
        sent = next(
            row
            for row in reversed(state["requests"])
            if row["method"] == "POST" and row["path"].endswith("/runs")
        )
        assert len(sent["data"]["asset_ids"]) == 1 and "segmentation_model_id" not in sent["data"]
        screenshot("16-sent-attachment.png")
        execute(
            "document.getElementById('prompt').value='这个股骨头的体积有多少';document.getElementById('chat-form').requestSubmit()"
        )
        until(
            "!document.getElementById('send').disabled && document.getElementById('prompt').value === ''",
            "follow-up question sends without reattaching the previous file",
        )
        sent = next(
            row
            for row in reversed(state["requests"])
            if row["method"] == "POST" and row["path"].endswith("/runs")
        )
        assert sent["data"]["asset_ids"] == []
        checks.append("follow-ups rely on server conversation context rather than pinned composer files")
        stale_mask = {**mask, "asset_id": "asset_stale_fixture"}
        state["assets"].append(stale_mask)
        job["result"]["asset_id"] = stale_mask["asset_id"]
        state["hold_jobs_cid"] = "conv_history"
        execute(
            "[...document.querySelectorAll('.history-item')].find(el=>el.textContent.includes('设置与阅读方式')).click()"
        )
        assert state["jobs_waiting"].wait(3)
        execute("document.getElementById('new-conversation').click()")
        state["hold_jobs_cid"] = None
        state["release_jobs"].set()
        time.sleep(0.3)
        assert execute(
            "return document.getElementById('image-workspace').hidden && document.getElementById('imaging-jobs').children.length === 0"
        )
        assert not any(stale_mask["asset_id"] in row["path"] for row in state["requests"])
        checks.append("a delayed completion response cannot open a result after switching conversations")
        job["result"]["asset_id"] = mask["asset_id"]
        execute("document.getElementById('execution-mode').value='full_access';document.getElementById('execution-mode').dispatchEvent(new Event('change'))")
        until("document.getElementById('execution-mode-settings').value==='full_access' && !document.getElementById('execution-mode').disabled", "automatic-workflow screenshot uses the selected Full access policy")
        auto_job = {"id": "job_auto_fixture", "status": "queued", "tool_name": "segment_medical_volume",
                    "elapsed_seconds": 0, "error_code": None, "result": {"phase": "queued"},
                    "input": {"conversation_id": "conv_history", "file_id": source["asset_id"],
                              "file_name": "synthetic-auto.nii.gz", "execution_mode": "full_access"}}
        state["imaging_jobs"].insert(0, auto_job)
        execute("[...document.querySelectorAll('.history-item')].find(el=>el.textContent.includes('设置与阅读方式')).click()")
        until("!document.getElementById('image-workspace').hidden && document.getElementById('download-mask').hidden",
              "Full access queued task opens the source image without a View click")
        assert execute("return document.getElementById('image-workspace').getBoundingClientRect().left > document.querySelector('.main').getBoundingClientRect().left")
        checks.append("desktop imaging workspace is on the right of the conversation")
        auto_job.update(status="succeeded", result={"phase": "completed", "asset_id": mask["asset_id"], "source_asset_id": source["asset_id"]})
        execute("window.ImagingUI.refreshJobs()")
        until("!document.getElementById('download-mask').hidden && Number(document.querySelector('.surface-host canvas')?.dataset.renderedFaces)>0",
              "Full access completion updates the mask and 3D view without confirmation or Open result clicks")
        screenshot("19-full-access.png")
        execute("[...document.querySelectorAll('.imaging-job button')].find(button=>button.textContent==='提交科室复核').click()")
        until("document.getElementById('settings-dialog').open && document.querySelector('.hospital-review')", "completed segmentation can be submitted to the review inbox")
        execute("[...document.querySelectorAll('.hospital-review button')].find(button=>button.textContent==='查看影像与结果').click()")
        until("!document.getElementById('settings-dialog').open && Number(document.querySelector('.surface-host canvas')?.dataset.renderedFaces)>0", "review results open in the existing four-pane workspace")
        assert any("/reviews/review_fixture/assets/" in row["path"] for row in state["requests"])
        state["account"].update(user_id="reviewer_fixture", role="reviewer", execution_mode="ask_for_approval", scopes=["reviews:decide"])
        execute("document.getElementById('logout').click()")
        request("POST", prefix + "/url", {"url": f"http://127.0.0.1:{server.server_port}/"})
        until("!document.getElementById('auth').hidden", "fixture reviewer starts with a separate login session")
        execute("document.getElementById('tenant').value='bone-lab';document.getElementById('email').value='reviewer@example.test';document.getElementById('password').value='TestPass123';document.getElementById('auth-form').requestSubmit()")
        until("!document.getElementById('workspace').hidden && document.getElementById('account-role').textContent.includes('审核员')", "reviewer identity is loaded from login response")
        execute("document.querySelector('[data-open=settings]').click();document.querySelector('[data-tab=reviews]').click()")
        until("document.querySelector('.hospital-review textarea')", "reviewer receives decision controls for a submitted result")
        assert execute("return document.getElementById('tab-organization').hidden && document.getElementById('execution-mode').disabled")
        execute("[...document.querySelectorAll('.hospital-review button')].find(button=>button.textContent==='查看影像与结果').click()")
        until("!document.getElementById('settings-dialog').open && document.getElementById('image-modality').disabled", "reviewer opens a read-only result workspace")
        execute("document.querySelector('[data-open=settings]').click();document.querySelector('[data-tab=reviews]').click()")
        until("document.querySelector('.hospital-review textarea')", "reviewer can return to the pending review")
        execute("document.querySelector('.hospital-review textarea').value='合成复核意见';[...document.querySelectorAll('.hospital-review button')].find(button=>button.textContent==='通过').click()")
        until("document.querySelector('.hospital-review').textContent.includes('已通过')", "review decision and comment remain visible after refresh")
        screenshot("20-review.png")
        state["account"].update(user_id="user_synthetic", role="owner", scopes=["tasks:draft", "reviews:submit", "organization:manage"])
        execute("document.getElementById('logout').click()")
        request("POST", prefix + "/url", {"url": f"http://127.0.0.1:{server.server_port}/"})
        until("!document.getElementById('auth').hidden", "fixture administrator starts with a fresh session")
        execute("document.getElementById('tenant').value='bone-lab';document.getElementById('email').value='reader@example.test';document.getElementById('password').value='TestPass123';document.getElementById('auth-form').requestSubmit()")
        until("!document.getElementById('workspace').hidden && document.querySelector('.history-item')", "administrator returns after reviewer checks")
        state["imaging_jobs"].remove(auto_job)
        execute(
            "document.getElementById('search').value='不存在的对话'; document.getElementById('search').dispatchEvent(new Event('input'))"
        )
        until(
            "document.getElementById('conversations').textContent.includes('没有找到')",
            "conversation search filters history",
        )
        execute(
            "document.getElementById('search').value=''; document.getElementById('search').dispatchEvent(new Event('input')); document.querySelectorAll('.history-item')[document.querySelectorAll('.history-item').length-1].click()"
        )
        until(
            "document.querySelector('.answer-table')",
            "history remains accessible after streaming and cancellation",
        )
        before_delete_count = len(state["conversations"])
        delete_id = state["conversations"][0]["id"]
        execute("window.confirm=()=>true;document.querySelector('.history-delete').click()")
        until(
            "document.querySelectorAll('.history-row').length === " + str(before_delete_count - 1),
            "conversation deletion updates the history list",
        )
        assert any(r["method"] == "DELETE" and r["path"].endswith(delete_id) for r in state["requests"])
        request("POST", prefix + "/window/rect", {"width": 390, "height": 844})
        execute("document.getElementById('toggle-sidebar').click()")
        until(
            "document.getElementById('workspace').classList.contains('sidebar-open')",
            "mobile navigation opens",
        )
        execute(
            "document.getElementById('sidebar-backdrop').click(); document.querySelector('.mobile-settings').click()"
        )
        until("document.getElementById('settings-dialog').open", "mobile settings are reachable")
        assert execute("return document.documentElement.scrollWidth <= innerWidth + 1")
        screenshot("05-mobile-settings.png")
        execute("document.getElementById('settings-dialog').close()")
        screenshot("06-mobile-chat.png")
        execute(
            "document.querySelector('.mobile-settings').click(); document.querySelector('[data-tab=account]').click(); document.getElementById('logout').click()"
        )
        until("!document.getElementById('auth').hidden", "logout clears the authenticated workspace")
        until(
            "getComputedStyle(document.documentElement).getPropertyValue('--body-size').trim() === '22px'",
            "reading settings survive logout and page reload",
        )
        if (args.output_dir / "brand-preview.html").is_file():
            request("POST", prefix + "/window/rect", {"width": 1440, "height": 1160})
            request("POST", prefix + "/url", {"url": f"http://127.0.0.1:{server.server_port}/brand-review"})
            until(
                "document.querySelectorAll('img').length > 6 && [...document.images].every(im => im.complete && im.naturalWidth > 0)",
                "path-only logo assets render at 16, 24 and 32 px in both theme colors",
            )
            screenshot("15-brand-specimen.png")
        result = {
            "status": "passed",
            "checks": checks,
            "check_count": len(checks),
            "requests_to_business_services": 0,
            "model_generations": 0,
            "fixture_requests": len(state["requests"]),
        }
        (args.output_dir / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False))
    finally:
        if session:
            try:
                request("DELETE", "/session/" + session)
            except Exception:
                pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        log.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
