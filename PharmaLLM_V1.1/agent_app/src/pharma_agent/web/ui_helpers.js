"use strict";
(function (root, factory) {
  const helpers = factory();
  if (typeof module === "object" && module.exports) module.exports = helpers;
  else root.PharmaUI = helpers;
})(typeof globalThis === "object" ? globalThis : this, function () {
  function title(raw) {
    let text = String(raw || "").trim();
    if (text.startsWith("{")) {
      try {
        const row = JSON.parse(text);
        text = String(row.problem_text || row.question || row.title || text);
      } catch { /* A truncated JSON title is handled as plain text. */ }
    }
    text = text.split(/["']?(?:选项|choices|options)["']?\s*[:：]/i)[0];
    text = text.replace(/^[\s{"']+|[\s,"'，]+$/g, "").replace(/\s+/g, " ").trim();
    if (!text || text === "新会话") return "新对话";
    const points = Array.from(text);
    return points.length > 40 ? points.slice(0, 40).join("") + "…" : text;
  }
  function reading(value) {
    const parsed = value && typeof value === "object" ? value : {};
    return {
      font: ["modern", "sans", "serif"].includes(parsed.font) ? parsed.font : "modern",
      size: [15, 16, 18, 20, 22].includes(Number(parsed.size)) ? Number(parsed.size) : 16,
      density: parsed.density === "compact" ? "compact" : "comfortable",
    };
  }
  function teamCodeError(value, optional = false) {
    const code = String(value || "").trim().toLowerCase();
    if (!code && optional) return "";
    return /^[a-z0-9][a-z0-9-]{2,31}$/.test(code) ? "" : "团队登录代号需为 3–32 位英文字母、数字或连字符，可用纯数字，首位不能是连字符";
  }
  function authIssue(values, register = false) {
    const code = String(values.tenant || "").trim();
    const codeError = register ? teamCodeError(code, true) : (!code || code.length > 64 ? "请填写团队登录代号或原团队 ID" : "");
    if (codeError) return { field: "tenant", message: codeError };
    if (register && Array.from(String(values.tenantName || "").trim()).length > 120) return { field: "tenant-name", message: "团队名称最多 120 个字符" };
    const email = String(values.email || "").trim();
    if (Array.from(email).length > 254 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return { field: "email", message: "请填写有效的邮箱地址" };
    const length = Array.from(String(values.password || "").trim()).length;
    if (length < 6 || length > (register ? 12 : 256)) return { field: "password", message: register ? "新密码需为 6–12 个字符" : "密码需为 6–256 个字符（原较长密码可继续登录）" };
    return null;
  }
  function eventBuffer(cursor = 0) {
    let buffer = "";
    return {
      push(chunk) {
        buffer += chunk;
        const events = [];
        let boundary;
        while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
          const frame = buffer.slice(0, boundary.index);
          buffer = buffer.slice(boundary.index + boundary[0].length);
          let event = "message", id = null;
          const lines = [];
          for (const line of frame.split(/\r?\n/)) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            if (line.startsWith("id:")) {
              const value = line.slice(3).trim();
              if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value))) throw new Error("事件编号无效，请重新连接。");
              id = Number(value);
            }
            if (line.startsWith("data:")) lines.push(line.slice(5).replace(/^ /, ""));
          }
          if (!lines.length || (id !== null && id <= cursor)) continue;
          let data;
          try { data = JSON.parse(lines.join("\n")); }
          catch { throw new Error("事件数据不完整，请重新连接。"); }
          if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error("事件格式无效，请重新连接。");
          if (id !== null) cursor = id;
          events.push({ event, id, data });
        }
        return events;
      },
    };
  }
  // Text-only Markdown blocks. The renderer never assigns model text to innerHTML.
  function markdown(text) {
    const lines = String(text).replace(/\r\n/g, "\n").split("\n");
    const blocks = [];
    let index = 0;
    const cells = line => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(cell => cell.trim());
    const divider = line => /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
    const special = line => /^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|```)/.test(line);
    while (index < lines.length) {
      if (!lines[index].trim()) { index += 1; continue; }
      const line = lines[index];
      if (line.startsWith("```")) {
        const content = [];
        index += 1;
        while (index < lines.length && !lines[index].startsWith("```")) content.push(lines[index++]);
        if (index < lines.length) index += 1;
        blocks.push({ type: "code", text: content.join("\n") });
      } else if (/^#{1,6}\s/.test(line)) {
        const match = /^(#{1,6})\s+(.*)$/.exec(line);
        blocks.push({ type: "heading", level: Math.min(match[1].length + 1, 6), text: match[2] });
        index += 1;
      } else if (index + 1 < lines.length && line.includes("|") && divider(lines[index + 1])) {
        const headers = cells(line), rows = [];
        index += 2;
        while (index < lines.length && lines[index].includes("|") && lines[index].trim()) rows.push(cells(lines[index++]));
        blocks.push({ type: "table", headers, rows });
      } else if (/^(?:[-*+]\s|\d+[.)]\s)/.test(line)) {
        const ordered = /^\d/.test(line), items = [];
        const pattern = ordered ? /^\d+[.)]\s+(.*)$/ : /^[-*+]\s+(.*)$/;
        while (index < lines.length && pattern.test(lines[index])) items.push(pattern.exec(lines[index++])[1]);
        blocks.push({ type: "list", ordered, items });
      } else {
        const content = [lines[index++]];
        while (index < lines.length && lines[index].trim() && !special(lines[index]) && !(index + 1 < lines.length && lines[index].includes("|") && divider(lines[index + 1]))) content.push(lines[index++]);
        blocks.push({ type: "paragraph", text: content.join("\n") });
      }
    }
    return blocks;
  }
  function inline(text) {
    const result = [], pattern = /(\*\*([^*\n]+)\*\*|`([^`\n]+)`)/g;
    let start = 0, match;
    while ((match = pattern.exec(text))) {
      if (match.index > start) result.push({ type: "text", text: text.slice(start, match.index) });
      result.push({ type: match[2] ? "strong" : "code", text: match[2] || match[3] });
      start = pattern.lastIndex;
    }
    if (start < text.length) result.push({ type: "text", text: text.slice(start) });
    return result;
  }
  return { title, reading, teamCodeError, authIssue, eventBuffer, markdown, inline };
});
