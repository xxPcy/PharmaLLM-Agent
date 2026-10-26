"use strict";
// Account policies are persisted server-side, never inferred from model output.
(() => {
  const roleNames = { owner: "管理员", doctor: "医生", member: "医生", reviewer: "审核员" };
  const reviewNames = { pending: "待复核", approved: "已通过", changes_requested: "已退回" };
  let epoch = 0, changing = false;
  function button(text, action) {
    const node = element("button", text, "secondary"); node.type = "button";
    node.onclick = async () => { node.disabled = true; try { await action(); } catch (error) { fail(error); } finally { node.disabled = false; } };
    return node;
  }
  function options(select, rows, value) {
    select.replaceChildren();
    for (const row of rows) { const option = element("option", row.name); option.value = row.id; select.append(option); }
    select.value = value ?? rows[0]?.id ?? "";
  }
  function applyAccount(value) {
    const canRun = value.scopes?.includes("tasks:draft") ?? value.role !== "reviewer";
    const canManage = value.role === "owner";
    $("identity").textContent = value.tenant_code || value.tenant_id;
    $("account-email").textContent = value.email || "";
    $("account-role").textContent = `${value.tenant_name || "团队"} · ${value.department_name || "未分配科室"} · ${roleNames[value.role] || "成员"}`;
    $("account-label").textContent = `${roleNames[value.role] || "成员"}${value.department_name ? ` · ${value.department_name}` : ""}`;
    $("tab-organization").hidden = !canManage;
    for (const id of ["execution-mode", "execution-mode-settings"]) {
      $(id).value = value.execution_mode || "ask_for_approval";
      $(id).disabled = !canRun || isBusy;
    }
    $("execution-mode-status").textContent = canRun ? "执行模式保存在当前账户，用于之后的消息。" : "审核员可以复核已提交结果，不能启动分割任务。";
  }
  async function changeMode(event) {
    if (changing || !account) return;
    const value = event.target.value, version = epoch;
    changing = true;
    $("send").disabled = true;
    for (const id of ["execution-mode", "execution-mode-settings"]) $(id).disabled = true;
    try {
      const result = await api("/api/v1/users/me/execution-mode", { method: "PATCH", body: JSON.stringify({ execution_mode: value }) });
      if (version !== epoch || !account) return;
      account.execution_mode = result.execution_mode;
      notify(value === "full_access" ? "之后的明确分割需求将自动执行。" : "之后的分割任务将先请求确认。");
    } catch (error) { fail(error); }
    finally { changing = false; if (version === epoch && account) { applyAccount(account); busy(isBusy); } }
  }
  $("execution-mode").onchange = $("execution-mode-settings").onchange = changeMode;

  async function organization() {
    const version = epoch, data = await api("/api/v1/organization");
    if (version !== epoch) return;
    $("team-name").value = data.account.tenant_name; $("team-code").value = data.account.tenant_code;
    $("department-list").textContent = data.departments.length ? data.departments.map(row => row.name).join(" · ") : "先创建科室，再添加医生或审核员。";
    options($("member-department"), data.departments);
    $("organization-members").replaceChildren();
    for (const member of data.members) {
      const row = element("form", undefined, "hospital-member");
      const name = element("input"), role = element("select"), department = element("select"), status = element("select");
      name.value = member.display_name || member.email; name.required = true; name.maxLength = 80; name.setAttribute("aria-label", "姓名");
      options(role, Object.entries(roleNames).filter(([id]) => id !== "member").map(([id, name]) => ({ id, name })), member.role === "member" ? "doctor" : member.role);
      options(department, [{ id: "", name: "未分配科室（仅管理员）" }, ...data.departments], member.department_id);
      options(status, [{ id: "active", name: "启用" }, { id: "disabled", name: "停用" }], member.status);
      role.setAttribute("aria-label", "角色"); department.setAttribute("aria-label", "科室"); status.setAttribute("aria-label", "账号状态");
      const save = element("button", "保存", "secondary"); save.type = "submit";
      row.append(element("strong", member.email), name, department, role, status, save);
      row.onsubmit = async event => {
        event.preventDefault(); save.disabled = true;
        try {
          await api(`/api/v1/organization/members/${encodeURIComponent(member.id)}`, { method: "PATCH", body: JSON.stringify({ display_name: name.value, department_id: department.value || null, role: role.value, status: status.value }) });
          const me = await api("/api/v1/users/me"); if (version !== epoch) return;
          account = me; applyAccount(account);
          if (account.role === "owner") await organization(); else setTab("account");
          notify("成员信息已保存。");
        } catch (error) { fail(error); } finally { save.disabled = false; }
      };
      $("organization-members").append(row);
    }
  }
  function formHandler(id, action) {
    $(id).onsubmit = async event => {
      event.preventDefault(); const submit = event.target.querySelector('button[type="submit"]'); submit.disabled = true;
      try { await action(); } catch (error) { fail(error); } finally { submit.disabled = false; }
    };
  }
  $("team-code").oninput = () => $("team-code").setCustomValidity("");
  $("team-code").onchange = () => { $("team-code").value = $("team-code").value.trim().toLowerCase(); };
  $("team-code").oninvalid = () => $("team-code").setCustomValidity(H.teamCodeError($("team-code").value));
  formHandler("team-form", async () => {
    const result = await api("/api/v1/organization", { method: "PATCH", body: JSON.stringify({ name: $("team-name").value.trim(), login_code: $("team-code").value.trim().toLowerCase() }) });
    account = result; applyAccount(account); $("tenant").value = account.tenant_code; notify("团队信息已保存，成员之后使用新的代号登录。");
  });
  formHandler("department-form", async () => {
    await api("/api/v1/organization/departments", { method: "POST", body: JSON.stringify({ name: $("department-name").value.trim() }) });
    $("department-form").reset(); await organization();
  });
  formHandler("member-form", async () => {
    await api("/api/v1/organization/members", { method: "POST", body: JSON.stringify({ display_name: $("member-name").value.trim(), email: $("member-email").value.trim(), password: $("member-password").value, department_id: $("member-department").value, role: $("member-role").value }) });
    $("member-form").reset(); await organization(); notify("账号已创建。成员使用本团队代号、邮箱和初始密码登录。");
  });
  async function reviews() {
    const version = epoch, rows = await api("/api/v1/reviews");
    if (version !== epoch) return;
    $("review-list").replaceChildren();
    if (!rows.length) $("review-list").append(element("p", "还没有提交的分割结果。", "settings-hint"));
    for (const review of rows) {
      const card = element("article", undefined, "hospital-review");
      card.append(element("strong", `分割结果 · ${reviewNames[review.status] || review.status}`), element("p", new Date(review.created_at).toLocaleString("zh-CN"), "settings-hint"));
      const actions = element("div", undefined, "button-row");
      actions.append(button("查看影像与结果", async () => { await window.ImagingUI.openReview(review); $("settings-dialog").close(); }));
      if (review.comment) card.append(element("p", `复核意见：${review.comment}`));
      if (review.status === "pending" && review.submitter_id !== account?.user_id && ["owner", "reviewer"].includes(account?.role)) {
        const comment = element("textarea"); comment.maxLength = 2000; comment.placeholder = "填写复核意见（退回必填）"; comment.setAttribute("aria-label", "复核意见"); card.append(comment);
        for (const [decision, text] of [["approved", "通过"], ["changes_requested", "退回修改"]]) actions.append(button(text, async () => {
          await api(`/api/v1/reviews/${encodeURIComponent(review.id)}/decision`, { method: "POST", body: JSON.stringify({ decision, comment: comment.value.trim() }) }); await reviews();
        }));
      }
      card.append(actions); $("review-list").append(card);
    }
  }
  $("refresh-reviews").onclick = () => reviews().catch(fail);
  window.HospitalUI = {
    applyAccount,
    changing() { return changing; },
    refresh(name) { (name === "organization" ? organization() : reviews()).catch(fail); },
    reset() { epoch += 1; for (const id of ["review-list", "organization-members", "department-list"]) $(id).replaceChildren(); for (const id of ["member-form", "team-form", "department-form"]) $(id).reset(); $("account-role").textContent = ""; },
    async submitReview(jobId) { await api(`/api/v1/jobs/${encodeURIComponent(jobId)}/review`, { method: "POST" }); openPanel("settings", "reviews"); },
  };
})();
