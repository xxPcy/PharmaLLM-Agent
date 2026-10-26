"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const H = require("../src/pharma_agent/web/ui_helpers.js");

test("team codes accept numeric and mixed codes while retaining length and character checks", () => {
  for (const code of ["2026", "002026", " 2026-LAB ", "guke"]) assert.equal(H.teamCodeError(code), "");
  for (const code of ["26", "-2026", "20_26", "团队2026"]) assert.match(H.teamCodeError(code), /团队登录代号/);
  assert.equal(H.teamCodeError("", true), "");
});
test("registration checks do not reject legacy identifiers during login", () => {
  const values = { tenant: "tenant_" + "a".repeat(32), email: "synthetic@example.org", password: "SyntheticPassword_123" };
  assert.equal(H.authIssue(values), null);
  assert.equal(H.authIssue(values, true).field, "tenant");
  assert.equal(H.authIssue({ ...values, tenant: "2026", password: "TestPass123" }, true), null);
  assert.equal(H.authIssue({ ...values, tenant: "", password: "TestPass123" }, true), null);
});
test("auth guidance identifies email or password without containing rejected values", () => {
  const values = { tenant: "2026", email: "synthetic@example.org", password: "hid1" };
  const issue = H.authIssue(values, true);
  assert.equal(issue.field, "password"); assert.match(issue.message, /12/);
  assert.equal(JSON.stringify(issue).includes(values.password), false);
  assert.equal(H.authIssue({ ...values, email: "bad-address" }, true).field, "email");
  assert.equal(H.authIssue({ ...values, password: "字".repeat(12) }, true), null);
});
test("new passwords accept 6 and 12 characters and reject 5 and 13 while old login remains compatible", () => {
  const values = { tenant: "2026", email: "synthetic@example.org" };
  for (const length of [6, 12]) assert.equal(H.authIssue({ ...values, password: "1".repeat(length) }, true), null);
  for (const length of [5, 13]) assert.equal(H.authIssue({ ...values, password: "1".repeat(length) }, true).field, "password");
  assert.equal(H.authIssue({ ...values, password: "SyntheticPassword_123" }, false), null);
});

test("conversation labels retain the question and omit serialized options", () => {
  assert.equal(H.title('原发性高血压病（ ） 。", "选项": ["甲", "乙"]'), "原发性高血压病（ ） 。");
  assert.equal(H.title('{"problem_text":"哪项是正确答案？","choices":["甲","乙"]}'), "哪项是正确答案？");
  assert.equal(H.title("新会话"), "新对话");
  assert.equal(H.title("😀".repeat(45)), "😀".repeat(40) + "…");
});
test("reading settings reject unrecognized values", () => {
  assert.deepEqual(H.reading({ font: "url(remote)", size: 1000, density: "invalid" }), { font: "modern", size: 16, density: "comfortable" });
  assert.deepEqual(H.reading({ font: "sans", size: "22", density: "compact" }), { font: "sans", size: 22, density: "compact" });
  assert.deepEqual(H.reading(null), { font: "modern", size: 16, density: "comfortable" });
});
test("SSE keeps partial CRLF frames and deduplicates replayed IDs", () => {
  const buffer = H.eventBuffer(1);
  assert.deepEqual(buffer.push('id: 2\r\nevent: delta\r\ndata: {"text":"中'), []);
  assert.deepEqual(buffer.push('文"}\r\n\r'), []);
  assert.deepEqual(buffer.push('\nid: 2\nevent: delta\ndata: {"text":"重复"}\n\n'), [
    { id: 2, event: "delta", data: { text: "中文" } },
  ]);
  assert.deepEqual(buffer.push(': heartbeat\n\nid: 3\nevent: completed\ndata: {}\n\n'), [{ id: 3, event: "completed", data: {} }]);
});
test("SSE supports multiline data and transient errors without IDs", () => {
  const buffer = H.eventBuffer(12);
  assert.deepEqual(buffer.push('event: error\ndata: {\ndata: "code":"EVENT_STREAM_UNAVAILABLE"}\n\n'), [
    { id: null, event: "error", data: { code: "EVENT_STREAM_UNAVAILABLE" } },
  ]);
});
test("malformed event payloads fail without echoing upstream data", () => {
  assert.throws(() => H.eventBuffer().push('id: 1\ndata: PRIVATE_UPSTREAM_TEXT\n\n'), error => !error.message.includes("PRIVATE_UPSTREAM_TEXT"));
  assert.throws(() => H.eventBuffer().push('id: invalid\ndata: {}\n\n'));
  assert.throws(() => H.eventBuffer().push('id: 2\ndata: []\n\n'));
});
test("Markdown preserves hostile HTML and links as inert text", () => {
  const text = '<img src=x onerror="steal()"> [打开](javascript:steal())';
  assert.deepEqual(H.markdown(text), [{ type: "paragraph", text }]);
  assert.deepEqual(H.inline("**<script>unsafe</script>**"), [{ type: "strong", text: "<script>unsafe</script>" }]);
});
test("Markdown recognizes readable headings, paragraphs, lists and tables", () => {
  const blocks = H.markdown("## 结论\n\n一段正文\n第二行\n\n- 甲\n- 乙\n\n| 用途 | 入口 |\n| --- | --- |\n| 文件 | 附件 |");
  assert.deepEqual(blocks.map(block => block.type), ["heading", "paragraph", "list", "table"]);
  assert.equal(blocks[1].text, "一段正文\n第二行");
  assert.deepEqual(blocks[2].items, ["甲", "乙"]);
  assert.deepEqual(blocks[3].rows, [["文件", "附件"]]);
});
test("an unfinished code fence remains inert code", () => {
  assert.deepEqual(H.markdown("```html\n<script>unsafe()</script>"), [{ type: "code", text: "<script>unsafe()</script>" }]);
});
