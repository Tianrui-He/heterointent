from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "showcase"


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HeteroIntent 推荐诊断台</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #637083;
      --line: #dfe5ee;
      --blue: #2f6fed;
      --green: #1f9d75;
      --red: #d94f70;
      --purple: #7d5fff;
      --amber: #b7791f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      height: 72px;
      padding: 14px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 22px; font-weight: 700; }
    .subtitle { margin-top: 4px; color: var(--muted); font-size: 13px; }
    .shell {
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      min-height: calc(100vh - 72px);
    }
    nav {
      padding: 18px;
      border-right: 1px solid var(--line);
      background: #fbfcff;
    }
    .tab {
      width: 100%;
      border: 0;
      text-align: left;
      padding: 12px 14px;
      margin-bottom: 8px;
      border-radius: 8px;
      background: transparent;
      color: var(--text);
      cursor: pointer;
      font-size: 15px;
    }
    .tab.active {
      background: #eaf1ff;
      color: var(--blue);
      font-weight: 700;
    }
    main { padding: 22px; min-width: 0; }
    section { display: none; }
    section.active { display: block; }
    .grid { display: grid; gap: 14px; }
    .grid.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .grid.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .grid.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }
    .metric-label { color: var(--muted); font-size: 13px; }
    .metric-value { font-size: 25px; font-weight: 750; margin-top: 6px; }
    .section-title {
      margin: 0 0 12px;
      font-size: 18px;
      font-weight: 750;
    }
    .toolbar {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }
    input, select {
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 12px;
      background: #fff;
      color: var(--text);
      min-width: 180px;
    }
    button.primary {
      height: 38px;
      padding: 0 14px;
      border-radius: 8px;
      border: 0;
      background: var(--blue);
      color: #fff;
      cursor: pointer;
      font-weight: 700;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }
    th:first-child, td:first-child { text-align: left; }
    tr.item-row { cursor: pointer; }
    tr.item-row:hover { background: #f3f7ff; }
    .pill {
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      background: #eef2f7;
      color: var(--muted);
      font-size: 12px;
      margin-right: 5px;
    }
    .pill.collect { background: #ffecef; color: var(--red); }
    .pill.share { background: #f1edff; color: var(--purple); }
    .pill.hard { background: #fff7e6; color: var(--amber); }
    .pill.intent { background: #e8fff6; color: var(--green); }
    .item-cell {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 150px;
    }
    .thumb {
      width: 44px;
      height: 44px;
      border-radius: 6px;
      border: 1px solid var(--line);
      object-fit: cover;
      background: #eef2f7;
      flex: 0 0 auto;
    }
    .thumb.empty {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 11px;
    }
    .detail-thumb {
      width: 120px;
      height: 120px;
      border-radius: 8px;
      border: 1px solid var(--line);
      object-fit: cover;
      background: #eef2f7;
      margin-bottom: 12px;
    }
    .bar {
      height: 9px;
      background: #eef2f7;
      border-radius: 999px;
      overflow: hidden;
      min-width: 90px;
    }
    .bar > span {
      display: block;
      height: 100%;
      width: 0;
      background: var(--blue);
    }
    .bars .bar-line {
      display: grid;
      grid-template-columns: 120px minmax(100px, 1fr) 70px;
      gap: 10px;
      align-items: center;
      margin: 9px 0;
      font-size: 13px;
    }
    .chart-img {
      width: 100%;
      max-height: 340px;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .case-list {
      display: grid;
      gap: 10px;
      max-height: 620px;
      overflow: auto;
    }
    .case-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      cursor: pointer;
    }
    .case-item:hover { border-color: var(--blue); }
    .case-title { font-weight: 750; margin-bottom: 7px; }
    .muted { color: var(--muted); }
    .table-wrap { overflow: auto; max-height: 580px; }
    .detail-empty { color: var(--muted); padding: 18px; }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      nav { border-right: 0; border-bottom: 1px solid var(--line); display: flex; gap: 8px; overflow-x: auto; }
      .tab { width: auto; white-space: nowrap; }
      .grid.cols-4, .grid.cols-3, .grid.cols-2 { grid-template-columns: 1fr; }
      main { padding: 14px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>HeteroIntent 推荐诊断台</h1>
      <div class="subtitle">本机离线展示：多目标排序、稀疏行为识别、模态贡献与用户意图解释</div>
    </div>
    <div class="muted" id="runInfo">loading</div>
  </header>
  <div class="shell">
    <nav>
      <button class="tab active" data-tab="overview">Overview</button>
      <button class="tab" data-tab="explorer">Request Explorer</button>
      <button class="tab" data-tab="why">Why This Rank</button>
      <button class="tab" data-tab="cases">Showcase Cases</button>
    </nav>
    <main>
      <section id="overview" class="active">
        <div class="grid cols-4" id="metricCards"></div>
        <div class="grid cols-2" style="margin-top:14px">
          <div class="card"><h2 class="section-title">样本不均衡</h2><img class="chart-img" src="/charts/imbalance.png" alt="imbalance"></div>
          <div class="card"><h2 class="section-title">核心解释指标</h2><img class="chart-img" src="/charts/metrics.png" alt="metrics"></div>
          <div class="card"><h2 class="section-title">Top-20 模态贡献</h2><img class="chart-img" src="/charts/modality_gate.png" alt="gates"></div>
          <div class="card"><h2 class="section-title">Top-20 标签命中</h2><img class="chart-img" src="/charts/top20_labels.png" alt="labels"></div>
        </div>
      </section>
      <section id="explorer">
        <div class="toolbar">
          <select id="caseSelect"></select>
          <input id="requestInput" placeholder="输入 valid request_id">
          <button class="primary" id="loadRequest">加载请求</button>
          <span class="muted" id="requestHint"></span>
        </div>
        <div class="grid cols-2">
          <div class="card">
            <h2 class="section-title">Valid Top-20 推荐列表</h2>
            <div class="table-wrap"><table id="rankTable"></table></div>
          </div>
          <div class="card">
            <h2 class="section-title">选中 item 解释</h2>
            <div id="itemDetail" class="detail-empty">点击左侧任意 item 查看分数来源。</div>
          </div>
        </div>
      </section>
      <section id="why">
        <div class="grid cols-2">
          <div class="card">
            <h2 class="section-title">分数构成</h2>
            <div id="scoreBars" class="bars detail-empty">请先在 Request Explorer 中选择一个 item。</div>
          </div>
          <div class="card">
            <h2 class="section-title">模态 gate</h2>
            <div id="gateBars" class="bars detail-empty">请先选择一个 item。</div>
          </div>
          <div class="card">
            <h2 class="section-title">用户意图信号</h2>
            <div id="intentPanel" class="detail-empty">请先选择一个 item。</div>
          </div>
          <div class="card">
            <h2 class="section-title">讲解提示</h2>
            <div id="talkTrack" class="detail-empty">选择 item 后生成现场讲解口径。</div>
          </div>
        </div>
      </section>
      <section id="cases">
        <div class="grid cols-2">
          <div class="card">
            <h2 class="section-title">精选案例</h2>
            <div class="case-list" id="caseList"></div>
          </div>
          <div class="card">
            <h2 class="section-title">案例说明</h2>
            <div id="caseDetail" class="detail-empty">点击左侧案例即可跳转到 Request Explorer。</div>
          </div>
        </div>
      </section>
    </main>
  </div>
  <script>
    const state = { overview: null, cases: [], currentRows: [], selectedItem: null };
    const fmt = (v, n=4) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "-" : Number(v).toFixed(n);
    const pct = v => `${Math.max(0, Math.min(100, Number(v || 0) * 100)).toFixed(1)}%`;
    const mediaUrl = path => `/media?path=${encodeURIComponent(path || "")}`;
    function thumbHtml(row, compact=false) {
      if (!row.thumbnail_path) return `<span class="thumb empty">no img</span>`;
      return `<img class="${compact ? "detail-thumb" : "thumb"}" src="${mediaUrl(row.thumbnail_path)}" alt="thumbnail" loading="lazy">`;
    }
    function tagsHtml(tags) {
      return String(tags || "").split(",").filter(Boolean).map(t => `<span class="pill ${t}">${t}</span>`).join("");
    }
    function setTab(name) {
      document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
      document.querySelectorAll("main section").forEach(s => s.classList.toggle("active", s.id === name));
    }
    document.querySelectorAll(".tab").forEach(b => b.addEventListener("click", () => setTab(b.dataset.tab)));
    async function fetchJson(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }
    function renderOverview() {
      const m = state.overview.metrics.valid_best || {};
      const d = state.overview.dataset.train || {};
      document.getElementById("runInfo").textContent = `${state.overview.run_dir}`;
      const cards = [
        ["valid WeightedHit@20", m["weighted_hit@20"]],
        ["valid NDCG@20", m["ndcg@20"]],
        ["collect AUC", m["request_auc_collect"]],
        ["share AUC", m["request_auc_share"]],
        ["train rows", d.rows, 0],
        ["train requests", d.requests, 0],
        ["collect row rate", d.row_positive_rate?.collect],
        ["share row rate", d.row_positive_rate?.share],
      ];
      document.getElementById("metricCards").innerHTML = cards.map(([label, value, digits]) => `
        <div class="card"><div class="metric-label">${label}</div><div class="metric-value">${fmt(value, digits ?? 4)}</div></div>
      `).join("");
    }
    function renderCases() {
      const select = document.getElementById("caseSelect");
      select.innerHTML = state.cases.map(c => `<option value="${c.request_id}">#${c.request_id} ${c.tags}</option>`).join("");
      const list = document.getElementById("caseList");
      list.innerHTML = state.cases.map(c => `
        <div class="case-item" data-id="${c.request_id}">
          <div class="case-title">request ${c.request_id}</div>
          <div>${tagsHtml(c.tags)}</div>
          <div class="muted" style="margin-top:6px">${c.explanation}</div>
        </div>
      `).join("");
      list.querySelectorAll(".case-item").forEach(el => el.addEventListener("click", () => loadRequest(el.dataset.id, true)));
      select.addEventListener("change", () => loadRequest(select.value, false));
    }
    function renderTable(rows) {
      const table = document.getElementById("rankTable");
      if (!rows.length) {
        table.innerHTML = "<tr><td>没有找到该 request。</td></tr>";
        return;
      }
      table.innerHTML = `
        <thead><tr><th>item</th><th>rank</th><th>score</th><th>click</th><th>collect</th><th>share</th><th>p_click</th><th>p_collect</th><th>p_share</th></tr></thead>
        <tbody>${rows.map(r => `
          <tr class="item-row" data-item="${r.item_id}">
            <td><div class="item-cell">${thumbHtml(r)}<span>${r.item_id}</span></div></td><td>${r.rank}</td><td>${fmt(r.final_score ?? r.score)}</td>
            <td>${r.click ?? 0}</td><td>${r.collect ?? 0}</td><td>${r.share ?? 0}</td>
            <td>${fmt(r.p_click)}</td><td>${fmt(r.p_collect)}</td><td>${fmt(r.p_share)}</td>
          </tr>`).join("")}</tbody>`;
      table.querySelectorAll(".item-row").forEach(row => row.addEventListener("click", () => {
        const itemId = Number(row.dataset.item);
        const item = state.currentRows.find(r => Number(r.item_id) === itemId);
        selectItem(item);
      }));
      selectItem(rows[0]);
    }
    async function loadRequest(id, jump) {
      if (!id) return;
      const data = await fetchJson(`/api/request?request_id=${encodeURIComponent(id)}`);
      state.currentRows = data.rows || [];
      document.getElementById("requestInput").value = id;
      document.getElementById("requestHint").textContent = `${state.currentRows.length} rows`;
      const c = state.cases.find(x => String(x.request_id) === String(id));
      document.getElementById("caseDetail").innerHTML = c ? `<p>${tagsHtml(c.tags)}</p><p>${c.explanation}</p>` : "自定义 request";
      renderTable(state.currentRows);
      if (jump) setTab("explorer");
    }
    function barLine(label, value, color="var(--blue)") {
      return `<div class="bar-line"><div>${label}</div><div class="bar"><span style="width:${pct(value)};background:${color}"></span></div><div>${fmt(value)}</div></div>`;
    }
    function selectItem(item) {
      state.selectedItem = item;
      if (!item) return;
      document.getElementById("itemDetail").innerHTML = `
        ${thumbHtml(item, true)}
        <div><span class="pill">item ${item.item_id}</span><span class="pill">rank ${item.rank}</span><span class="pill">type ${item.item_type ?? "-"}</span><span class="pill">tax ${item.taxonomy_id ?? "-"}</span></div>
        <div class="muted" style="margin-top:6px">thumbnail: ${item.thumbnail_source || "none"}</div>
        <div class="bars" style="margin-top:12px">
          ${barLine("p_click", item.p_click, "var(--blue)")}
          ${barLine("p_collect", item.p_collect, "var(--red)")}
          ${barLine("p_share", item.p_share, "var(--purple)")}
          ${barLine("final_score", item.final_score ?? item.score, "var(--green)")}
        </div>`;
      renderWhy(item);
    }
    function renderWhy(item) {
      document.getElementById("scoreBars").innerHTML = [
        barLine("weighted_prob", item.weighted_prob_score, "var(--blue)"),
        barLine("final_score", item.final_score ?? item.score, "var(--purple)"),
        barLine("p_click", item.p_click, "var(--blue)"),
        barLine("p_collect", item.p_collect, "var(--red)"),
        barLine("p_share", item.p_share, "var(--purple)")
      ].join("");
      const gates = ["item_id","item_type","taxonomy","position","text","image","video","dense","graph"];
      document.getElementById("gateBars").innerHTML = gates.map(g => barLine(g, item[`gate_${g}`], g === "graph" ? "var(--green)" : "var(--blue)")).join("");
      document.getElementById("intentPanel").innerHTML = `
        <p><span class="pill">target type ${item.target_item_type ?? "-"}</span><span class="pill">hist type ${item.hist_dominant_item_type ?? "-"}</span></p>
        ${barLine("attention type mass", item.attention_type_target_mass, "var(--green)")}
        ${barLine("attention taxonomy mass", item.attention_taxonomy_target_mass, "var(--green)")}
        <p class="muted">type shift=${item.is_type_shift ?? "-"} taxonomy shift=${item.is_taxonomy_shift ?? "-"}</p>`;
      const topGate = gates.map(g => [g, Number(item[`gate_${g}`] || 0)]).sort((a,b)=>b[1]-a[1])[0];
      document.getElementById("talkTrack").innerHTML = `
        <p>这个 item 排在第 ${item.rank} 位，最终分数为 ${fmt(item.final_score ?? item.score)}。</p>
        <p>三任务概率分别为 click=${fmt(item.p_click)}、collect=${fmt(item.p_collect)}、share=${fmt(item.p_share)}，可用于解释多目标推荐权衡。</p>
        <p>当前贡献最高的模态是 <b>${topGate[0]}</b>，gate=${fmt(topGate[1])}；这可以说明模型不仅依赖单一 ID，而是在融合内容、图关系和统计特征。</p>`;
    }
    document.getElementById("loadRequest").addEventListener("click", () => loadRequest(document.getElementById("requestInput").value, false));
    async function init() {
      state.overview = await fetchJson("/api/overview");
      state.cases = await fetchJson("/api/cases");
      renderOverview();
      renderCases();
      if (state.cases.length) loadRequest(state.cases[0].request_id, false);
    }
    init().catch(err => { document.body.innerHTML = `<pre>${err.stack || err}</pre>`; });
  </script>
</body>
</html>
"""


class ShowcaseState:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        overview_path = data_dir / "overview.json"
        cases_path = data_dir / "showcase_cases.parquet"
        valid_path = data_dir / "valid_top20.parquet"
        test_path = data_dir / "test_top20.parquet"
        missing = [path for path in [overview_path, cases_path, valid_path, test_path] if not path.exists()]
        if missing:
            names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing showcase artifacts: {names}. Run scripts/export_showcase_data.py first.")
        self.overview = json.loads(overview_path.read_text(encoding="utf-8"))
        self.cases = pd.read_parquet(cases_path)
        self.valid = pd.read_parquet(valid_path)
        self.test = pd.read_parquet(test_path)
        self.thumbnail_paths = self._collect_thumbnail_paths()

    def _collect_thumbnail_paths(self) -> set[str]:
        paths: set[str] = set()
        for df in [self.valid, self.test]:
            if "thumbnail_path" not in df:
                continue
            for value in df["thumbnail_path"].dropna().astype(str).unique():
                if value:
                    paths.add(value)
        return paths

    def request_rows(self, request_id: str, split: str = "valid") -> list[dict]:
        df = self.valid if split == "valid" else self.test
        try:
            request_value = int(request_id)
        except ValueError:
            return []
        rows = df[df["request_id"].astype(int).eq(request_value)].sort_values("rank")
        return _records(rows)


def _records(df: pd.DataFrame) -> list[dict]:
    clean = df.replace({np.nan: None}) if "np" in globals() else df.where(pd.notna(df), None)
    return clean.to_dict("records")


try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


def make_handler(state: ShowcaseState):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload, status: int = 200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            query = parse_qs(parsed.query)
            try:
                if path in {"/", "/index.html"}:
                    self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/api/overview":
                    self._send_json(state.overview)
                elif path == "/api/cases":
                    self._send_json(_records(state.cases))
                elif path == "/api/request":
                    self._send_json({"rows": state.request_rows(query.get("request_id", [""])[0], "valid")})
                elif path == "/api/test_request":
                    self._send_json({"rows": state.request_rows(query.get("request_id", [""])[0], "test")})
                elif path == "/media":
                    requested = unquote(query.get("path", [""])[0])
                    if requested not in state.thumbnail_paths:
                        self._send_json({"error": "media not allowed"}, status=404)
                        return
                    media_path = Path(requested)
                    if not media_path.is_file():
                        self._send_json({"error": "media not found"}, status=404)
                        return
                    content_type = mimetypes.guess_type(str(media_path))[0] or "application/octet-stream"
                    self._send_bytes(media_path.read_bytes(), content_type)
                elif path.startswith("/charts/"):
                    chart_path = (state.data_dir / path.lstrip("/")).resolve()
                    if not str(chart_path).startswith(str(state.data_dir.resolve())) or not chart_path.exists():
                        self._send_json({"error": "chart not found"}, status=404)
                        return
                    content_type = mimetypes.guess_type(str(chart_path))[0] or "application/octet-stream"
                    self._send_bytes(chart_path.read_bytes(), content_type)
                else:
                    self._send_json({"error": "not found"}, status=404)
            except Exception as exc:  # pragma: no cover
                self._send_json({"error": str(exc)}, status=500)

        def log_message(self, fmt, *args):  # noqa: A003
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    return Handler


def smoke_test(data_dir: Path) -> None:
    state = ShowcaseState(data_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.1)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        root = conn.getresponse()
        body = root.read()
        assert root.status == 200 and b"HeteroIntent" in body
        conn.request("GET", "/api/overview")
        overview = conn.getresponse()
        payload = overview.read()
        assert overview.status == 200 and b"score_weights" in payload
        conn.request("GET", "/api/cases")
        cases = conn.getresponse()
        assert cases.status == 200 and len(json.loads(cases.read().decode("utf-8"))) > 0
    finally:
        server.shutdown()
        server.server_close()
    print("showcase app smoke ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the offline HeteroIntent showcase app.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--smoke", action="store_true", help="Start a temporary server, verify core endpoints, and exit.")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if args.smoke:
        smoke_test(data_dir)
        return
    state = ShowcaseState(data_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}"
    print(f"HeteroIntent showcase app running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
