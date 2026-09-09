/* Fastexp Web Studio — frontend logic (Phase 2 playground). */
(function () {
  "use strict";

  var state = {
    editor: null,
    operators: [],
    fields: [],
    templates: [],
    currentJob: null,
    connected: false,
  };

  var $ = function (id) { return document.getElementById(id); };

  function appendLine(level, text) {
    var con = $("console");
    if (!text && level === "PING") return;
    var span = document.createElement("span");
    span.className = level;
    span.textContent = text + "\n";
    con.appendChild(span);
    con.scrollTop = con.scrollHeight;
  }

  function classify(text, baseLevel) {
    if (baseLevel === "SYSTEM") return "SYSTEM";
    if (baseLevel === "STDERR") return "STDERR";
    if (/DECISION\s*:\s*PASS/i.test(text)) return "GATE_PASS";
    if (/DECISION\s*:\s*(FAIL|BLOCKED)/i.test(text)) return "GATE_FAIL";
    if (/\|\s*PASS\b/i.test(text)) return "GATE_PASS";
    if (/\|\s*FAIL\b|\|\s*BLOCKED\b/i.test(text)) return "GATE_FAIL";
    if (/Traceback|ERROR/i.test(text)) return "STDERR";
    return "INFO";
  }

  function streamLogs(jobId) {
    var proto = location.protocol === "https:" ? "wss" : "ws";
    var ws = new WebSocket(proto + "://" + location.host + "/api/v1/ws/logs/" + jobId);
    ws.onmessage = function (ev) {
      var msg = JSON.parse(ev.data);
      if (msg.done) {
        appendLine("SYSTEM", "— complete — status=" + msg.status +
          (msg.run_ids && msg.run_ids.length ? " run_ids=" + msg.run_ids.join(", ") : ""));
        refreshRuns();
        loadJobs();
        setButtons(false);
        if (msg.run_ids && msg.run_ids.length) {
          appendReportLinks(msg.run_ids);
        }
        state.currentJob = null;
        return;
      }
      appendLine(classify(msg.text, msg.level), msg.text);
    };
    ws.onerror = function () { appendLine("STDERR", "[websocket error]"); };
  }

  function appendReportLinks(runIds) {
    var con = $("console");
    var div = document.createElement("div");
    div.style.marginTop = "6px";
    runIds.forEach(function (rid) {
      var a = document.createElement("a");
      a.href = "/reports/" + rid + "/dashboard.html";
      a.target = "_blank";
      a.textContent = "[ Open Interactive Report: " + rid + " ]  ";
      a.style.color = "#93C5FD";
      div.appendChild(a);
    });
    con.appendChild(div);
    con.scrollTop = con.scrollHeight;
  }

  function setButtons(running) {
    $("btn-run").disabled = running;
    $("btn-batch").disabled = running;
    $("btn-stop").disabled = !running;
  }

  function params() {
    return {
      dataset: $("p-dataset").value,
      mode: $("p-mode").value,
      seed: parseInt($("p-seed").value, 10) || 11,
      universe: $("p-universe").value || null,
      neutralization: $("p-neutralization").value || null,
      truncation: parseFloat($("p-truncation").value),
      decay: parseInt($("p-decay").value, 10) || 0,
      include_private: $("p-private").checked,
    };
  }

  async function postJSON(url, body) {
    var res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    var data = await res.json().catch(function () { return {}; });
    if (!res.ok) throw new Error(data.error || ("HTTP " + res.status));
    return data;
  }

  function startJob(url, body, kindLabel) {
    appendLine("SYSTEM", "—— " + kindLabel + " submitted ——");
    setButtons(true);
    postJSON(url, body).then(function (data) {
      state.currentJob = data.job_id;
      appendLine("SYSTEM", "job_id=" + data.job_id + " status=" + data.status);
      streamLogs(data.job_id);
    }).catch(function (e) {
      appendLine("STDERR", "submit failed: " + e.message);
      setButtons(false);
    });
  }

  function validateExpression(expr) {
    var lint = $("lint");
    if (!expr.trim()) { lint.textContent = ""; lint.className = "lint"; return true; }
    var depth = 0;
    for (var i = 0; i < expr.length; i++) {
      if (expr[i] === "(") depth++;
      else if (expr[i] === ")") { depth--; if (depth < 0) break; }
    }
    if (depth !== 0) {
      lint.textContent = "unbalanced parentheses (depth " + depth + ")";
      lint.className = "lint error";
      return false;
    }
    var m = expr.match(/\b([a-z_][a-z0-9_]*)\s*\(/g) || [];
    var known = {};
    state.operators.forEach(function (o) { known[o.name] = true; });
    state.fields.forEach(function (f) { known[f] = true; });
    known.if = known.then = known.else = known.and = known.or = known.not = true;
    var unknown = [];
    m.forEach(function (tok) {
      var name = tok.replace(/[\s(]/g, "");
      if (!known[name] && name !== "") unknown.push(name);
    });
    if (unknown.length) {
      lint.textContent = "unknown operator/field: " + Array.from(new Set(unknown)).join(", ");
      lint.className = "lint error";
      return false;
    }
    lint.textContent = "✓ syntax looks valid";
    lint.className = "lint ok";
    return true;
  }

  function fmtElapsed(t) {
    var s = Math.max(0, Math.floor((Date.now() / 1000) - t));
    var m = Math.floor(s / 60); s = s % 60;
    return (m ? m + "m " : "") + s + "s";
  }

  function stopJob(jobId) {
    postJSON("/api/v1/runs/stop", { job_id: jobId }).then(function (d) {
      appendLine("SYSTEM", "stop requested: " + d.status);
      loadJobs();
    });
  }

  function renderJobs(jobs) {
    var box = $("jobs");
    box.innerHTML = "";
    if (!jobs.length) { box.textContent = "No jobs yet."; return; }
    jobs.forEach(function (j) {
      var el = document.createElement("div");
      el.className = "job";
      var jid = document.createElement("span");
      jid.className = "jid"; jid.textContent = j.job_id;
      var kind = document.createElement("span");
      kind.className = "kind"; kind.textContent = j.kind;
      var badge = document.createElement("span");
      badge.className = "badge " + j.status; badge.textContent = j.status;
      var elapsed = document.createElement("span");
      elapsed.className = "elapsed";
      elapsed.textContent = (j.status === "RUNNING" || j.status === "QUEUED") ? fmtElapsed(j.created_at) : "";
      var actions = document.createElement("span");
      if (j.status === "RUNNING" || j.status === "QUEUED") {
        var stop = document.createElement("button");
        stop.className = "btn danger small"; stop.textContent = "Stop";
        stop.addEventListener("click", function () { stopJob(j.job_id); });
        actions.appendChild(stop);
      }
      if (j.run_ids && j.run_ids.length) {
        var a = document.createElement("a");
        a.href = "/reports/" + j.run_ids[0] + "/dashboard.html";
        a.target = "_blank"; a.textContent = "Open Report";
        actions.appendChild(a);
      }
      el.appendChild(jid); el.appendChild(kind); el.appendChild(badge);
      el.appendChild(elapsed); el.appendChild(actions);
      box.appendChild(el);
    });
  }

  function loadJobs() {
    fetch("/api/v1/jobs").then(function (r) { return r.json(); }).then(function (d) {
      renderJobs(d.jobs || []);
    });
  }

  function handleFile(file) {
    var reader = new FileReader();
    reader.onload = function () {
      var text = reader.result;
      var name = file.name || "";
      if (name.toLowerCase().endsWith(".jsonl")) {
        var exprs = [];
        text.split("\n").forEach(function (line) {
          line = line.trim(); if (!line) return;
          try {
            var rec = JSON.parse(line);
            var e = rec.expression || rec.expr;
            if (e) exprs.push(e);
          } catch (err) {}
        });
        if (!exprs.length) { appendLine("STDERR", "no expressions parsed from " + name); return; }
        appendLine("SYSTEM", "parsed " + exprs.length + " expressions from " + name);
        startJob("/api/v1/batch", Object.assign({ expressions: exprs }, params()), "batch upload (" + exprs.length + ")");
      } else {
        var lines = text.split("\n").map(function (s) { return s.trim(); }).filter(Boolean);
        if (lines.length === 1 && state.editor) {
          state.editor.setValue(lines[0]);
          appendLine("SYSTEM", "loaded single expression into editor");
        } else {
          $("batch").value = lines.join("\n");
          appendLine("SYSTEM", "loaded " + lines.length + " expressions into batch editor");
        }
      }
    };
    reader.readAsText(file);
  }

  function setupDropzone() {
    var dz = $("dropzone");
    var fi = $("file-input");
    dz.addEventListener("click", function () { fi.click(); });
    dz.addEventListener("dragover", function (e) { e.preventDefault(); dz.classList.add("drag"); });
    dz.addEventListener("dragleave", function () { dz.classList.remove("drag"); });
    dz.addEventListener("drop", function (e) {
      e.preventDefault(); dz.classList.remove("drag");
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fi.addEventListener("change", function () { if (fi.files.length) handleFile(fi.files[0]); });
  }

  function refreshRuns() {
    fetch("/api/v1/runs").then(function (r) { return r.json(); }).then(function (data) {
      var box = $("runs");
      box.innerHTML = "";
      if (!data.runs.length) { box.textContent = "No runs yet."; return; }
      data.runs.forEach(function (r) {
        var el = document.createElement("div");
        el.className = "run";
        var id = document.createElement("span");
        id.className = "id";
        id.textContent = r.run_id;
        var meta = document.createElement("span");
        meta.className = "meta";
        meta.textContent = (r.dataset_id || "").slice(0, 8) + " · seed " + r.seed + " · " + r.mode;
        var spacer = document.createElement("span");
        spacer.className = "spacer";
        var badge = document.createElement("span");
        badge.className = "badge " + r.status;
        badge.textContent = r.status;
        var dash = document.createElement("a");
        dash.href = "/reports/" + r.run_id + "/dashboard.html";
        dash.target = "_blank";
        dash.textContent = "Dashboard";
        var rep = document.createElement("a");
        rep.href = "/reports/" + r.run_id + "/report.html";
        rep.target = "_blank";
        rep.textContent = "Report";
        el.appendChild(id); el.appendChild(meta); el.appendChild(spacer);
        el.appendChild(badge); el.appendChild(dash); el.appendChild(rep);
        box.appendChild(el);
      });
    });
  }

  function initEditor() {
    state.editor = monaco.editor.create($("editor"), {
      value: "rank(ts_zscore(close, 120))",
      language: "python",
      theme: "vs-dark",
      fontSize: 13,
      fontFamily: "JetBrains Mono, Fira Code, monospace",
      minimap: { enabled: false },
      automaticLayout: true,
      scrollBeyondLastLine: false,
    });

    monaco.languages.registerCompletionItemProvider("python", {
      provideCompletionItems: function (model, position) {
        var word = model.getWordUntilPosition(position);
        var range = {
          startLineNumber: position.lineNumber, endLineNumber: position.lineNumber,
          startColumn: word.startColumn, endColumn: word.endColumn,
        };
        var sug = state.operators.map(function (op) {
          return {
            label: op.name, kind: monaco.languages.CompletionItemKind.Function,
            detail: op.sig, documentation: op.doc,
            insertText: op.name + "(", range: range,
          };
        });
        var fld = state.fields.map(function (f) {
          return {
            label: f, kind: monaco.languages.CompletionItemKind.Variable,
            insertText: f, range: range,
          };
        });
        return { suggestions: sug.concat(fld) };
      },
    });

    state.editor.onDidChangeModelContent(function () {
      var v = state.editor.getValue();
      $("expr-status").textContent = v.length + " chars";
      validateExpression(v);
    });
    $("expr-status").textContent = state.editor.getValue().length + " chars";
  }

  function renderOps() {
    var q = ($("op-search").value || "").toLowerCase();
    var box = $("op-list");
    box.innerHTML = "";
    var items = state.operators.map(function (o) {
      return { sig: o.sig, cat: o.category, doc: o.doc, hay: (o.name + o.sig + o.category + o.doc).toLowerCase() };
    });
    var fields = state.fields.map(function (f) {
      return { sig: f, cat: "Field", doc: "Data field available in the panel.", hay: f.toLowerCase() };
    });
    items = items.concat(fields);
    items.filter(function (i) { return !q || i.hay.indexOf(q) >= 0; })
      .forEach(function (i) {
        var el = document.createElement("div");
        el.className = "op";
        el.innerHTML = '<div class="cat">' + i.cat + '</div><div class="sig">' + i.sig + '</div><div class="doc">' + i.doc + "</div>";
        box.appendChild(el);
      });
  }

  function loadMeta() {
    fetch("/api/v1/operators").then(function (r) { return r.json(); }).then(function (d) {
      state.operators = d.operators || [];
      state.fields = d.fields || [];
      renderOps();
    });
    fetch("/api/v1/templates").then(function (r) { return r.json(); }).then(function (d) {
      state.templates = d.templates || [];
      var sel = $("template");
      state.templates.forEach(function (t) {
        var o = document.createElement("option");
        o.value = t.expr; o.textContent = t.name;
        sel.appendChild(o);
      });
    });
  }

  function wire() {
    $("btn-run").addEventListener("click", function () {
      var expr = state.editor ? state.editor.getValue() : "";
      if (!validateExpression(expr)) return;
      startJob("/api/v1/simulate", Object.assign({ expression: expr }, params()), "simulate");
    });
    $("btn-batch").addEventListener("click", function () {
      var lines = $("batch").value.split("\n").map(function (s) { return s.trim(); }).filter(Boolean);
      if (!lines.length) { appendLine("STDERR", "batch is empty"); return; }
      startJob("/api/v1/batch", Object.assign({ expressions: lines }, params()), "batch (" + lines.length + " exprs)");
    });
    $("btn-stop").addEventListener("click", function () {
      if (!state.currentJob) return;
      postJSON("/api/v1/runs/stop", { job_id: state.currentJob }).then(function (d) {
        appendLine("SYSTEM", "stop requested: " + d.status);
      });
    });
    $("btn-refresh").addEventListener("click", refreshRuns);
    $("btn-jobs-refresh").addEventListener("click", loadJobs);
    $("btn-clear").addEventListener("click", function () { $("console").innerHTML = ""; });
    $("btn-console-toggle").addEventListener("click", function () {
      var card = document.querySelector(".console-card");
      var collapsed = card.classList.toggle("collapsed");
      this.textContent = collapsed ? "Expand" : "Collapse";
    });
    $("btn-validate").addEventListener("click", function () {
      validateExpression(state.editor ? state.editor.getValue() : "");
    });
    $("template").addEventListener("change", function () {
      if (this.value && state.editor) state.editor.setValue(this.value);
    });
    $("p-truncation").addEventListener("input", function () { $("trunc-val").textContent = this.value; });
    $("p-private").addEventListener("change", function () {
      $("private-warn").classList.toggle("hidden", !this.checked);
    });
    $("btn-docs").addEventListener("click", function () {
      $("drawer").classList.remove("hidden");
      $("scrim").classList.remove("hidden");
    });
    $("drawer-close").addEventListener("click", closeDrawer);
    $("scrim").addEventListener("click", closeDrawer);
    $("op-search").addEventListener("input", renderOps);
  }

  function closeDrawer() {
    $("drawer").classList.add("hidden");
    $("scrim").classList.add("hidden");
  }

  function ready() {
    fetch("/api/v1/runs").then(function (r) {
      state.connected = r.ok;
      $("conn").textContent = r.ok ? "backend connected" : "backend unreachable";
      $("conn").className = "pill " + (r.ok ? "ok" : "warn");
      return r.json();
    }).catch(function () {
      $("conn").textContent = "backend unreachable";
      $("conn").className = "pill warn";
    }).then(refreshRuns);
    initEditor();
    wire();
    setupDropzone();
    loadJobs();
    setInterval(loadJobs, 2500);
  }

  function waitForMonaco() {
    if (window.monaco) { loadMeta(); ready(); return; }
    var t = 0;
    var iv = setInterval(function () {
      t += 100;
      if (window.monaco) { clearInterval(iv); loadMeta(); ready(); }
      else if (t > 15000) { clearInterval(iv); appendLine("STDERR", "Monaco failed to load (no network?)."); }
    }, 100);
  }

  waitForMonaco();
})();
