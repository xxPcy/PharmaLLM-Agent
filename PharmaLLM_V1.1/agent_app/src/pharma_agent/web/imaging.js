"use strict";
// Extends the existing authenticated chat; inference decisions remain server-side.
(() => {
  let capabilities = null, selected = [], uploads = 0, generation = 0, jobTimer = null;
  let current = null, currentMask = null, views = [], labels = new Set(), opacity = .45;
  let openSequence = 0;
  let surface = null, surfaceController = null;
  let jobRefresh = 0, fileCache = new Map();
  const shownResults = new Map();
  const shownStarts = new Set(), reviewAccess = new Map(), dismissedJobs = new Set();
  let activeAutoJob = null;
  function assetPath(id, action) {
    const reviewId = reviewAccess.get(id);
    return reviewId ? `/api/v1/reviews/${encodeURIComponent(reviewId)}/assets/${encodeURIComponent(id)}/${action}` : `/api/v1/assets/${encodeURIComponent(id)}/${action}`;
  }
  const colors = ["#41bea3", "#f1b258", "#739bf0", "#dc779a", "#aa91e2"];
  const stages = { draft: "等待确认", queued: "排队中", preprocessing: "预处理中", inference: "推理中", postprocessing: "后处理中", completed: "已完成", failed: "失败", running: "处理中" };
  const opposite = { R: "L", L: "R", A: "P", P: "A", S: "I", I: "S" };
  const specs = [{ key: "axial", name: "轴位", axis: 2, h: 0, v: 1 }, { key: "coronal", name: "冠状位", axis: 1, h: 0, v: 2 }, { key: "sagittal", name: "矢状位", axis: 0, h: 1, v: 2 }];
  const number = n => Number(n).toLocaleString("zh-CN", { maximumFractionDigits: 2 });
  function action(text, callback, className = "secondary") {
    const button = element("button", text, className); button.type = "button";
    button.onclick = async () => { button.disabled = true; try { await callback(); } catch (e) { fail(e); } finally { button.disabled = false; } };
    return button;
  }
  async function metadata(id) {
    if (!fileCache.has(id)) fileCache.set(id, await api(assetPath(id, "metadata")));
    return fileCache.get(id);
  }
  function renderSelected() {
    $("composer-assets").replaceChildren();
    for (const file of selected) {
      const row = element("div", undefined, `attachment-chip${file.status === "failed" ? " failed" : ""}`);
      row.append(element("span", file.name || file.asset_id), element("small", file.status === "uploading" ? "上传中" : file.status === "parsing" ? "校验与解析中" : file.status === "failed" ? "失败" : "可使用"));
      if (file.asset_id && ["image", "volume", "dicom_slice"].includes(file.kind)) row.append(action("查看", () => openImage(file.asset_id), "text-button"));
      const remove = action("×", () => { selected = selected.filter(r => r !== file); renderSelected(); }, "text-button");
      remove.setAttribute("aria-label", `移除附件 ${file.name || ""}`); remove.disabled = ["uploading", "parsing"].includes(file.status); row.append(remove);
      $("composer-assets").append(row);
    }
  }
  function addSelected(file) {
    if (selected.some(r => r.asset_id === file.asset_id)) return;
    if (selected.filter(r => r.asset_id).length >= 8) throw new Error("每轮最多引用 8 个附件，请先移除部分附件。");
    selected.push(file); renderSelected();
  }
  async function uploadFiles(files) {
    if (!token || !files.length) return;
    if (uploads) throw new Error("请等待当前上传完成。");
    const multipleDicom = files.length > 1 && files.every(f => !/\.(png|jpe?g|nii(?:\.gz)?|txt|json|zip)$/i.test(f.name));
    const groups = multipleDicom ? [files] : files.map(f => [f]);
    const total = files.reduce((n, f) => n + f.size, 0);
    if (capabilities && ((multipleDicom && total > capabilities.max_upload_bytes) || files.some(f => f.size > capabilities.max_upload_bytes))) throw new Error(`上传超过上限（${number(capabilities.max_upload_bytes / 1048576)} MB）。`);
    for (const group of groups) {
      if (selected.length >= 8) { notify("已达到 8 个附件，请移除部分附件后继续。"); break; }
      const pending = { name: group.length > 1 ? `DICOM 序列（${group.length} 层）` : group[0].name, status: "uploading" };
      const version = generation, requestToken = token;
      selected.push(pending); uploads += 1; renderSelected();
      $("asset-result").textContent = "正在上传…";
      try {
        const body = new FormData(); group.forEach(f => body.append("file", f));
        body.append("modality", $("upload-modality").value);
        if (conversationId) body.append("conversation_id", conversationId);
        const result = await new Promise((resolve, reject) => {
          const xhr = new XMLHttpRequest(); xhr.open("POST", "/api/v1/imaging/uploads"); xhr.setRequestHeader("Authorization", `Bearer ${requestToken}`);
          xhr.upload.onload = () => { if (version === generation) { pending.status = "parsing"; renderSelected(); $("asset-result").textContent = "上传完成，服务器正在校验与解析…"; } };
          xhr.onload = () => {
            let data; try { data = JSON.parse(xhr.responseText); } catch { return reject(new Error("服务器未返回有效的解析结果。")); }
            if (xhr.status === 401 && requestToken === token) expireSession();
            if (xhr.status < 200 || xhr.status >= 300) return reject(new Error(data.error?.message || `上传失败（${xhr.status}）`));
            resolve(data);
          };
          xhr.onerror = () => reject(new Error("上传连接失败，请检查网络后重试。")); xhr.onabort = () => reject(new Error("上传已中断。")); xhr.timeout = 180000; xhr.ontimeout = () => reject(new Error("上传或解析超时，请重新打开文件列表检查后重试。")); xhr.send(body);
        });
        if (version !== generation || token !== requestToken) return;
        Object.assign(pending, result); fileCache.set(result.asset_id, result);
        uploadedAssets.unshift(result); renderFileList(); renderSelected();
        $("asset-result").textContent = `已解析：${pending.name}。文件编号 ${result.asset_id}`;
        // Upload attaches the file; opening the image is an explicit user action.
      } catch (e) {
        if (version === generation) { pending.status = "failed"; renderSelected(); $("asset-result").textContent = e.message; fail(e); }
      } finally { uploads = Math.max(0, uploads - 1); }
    }
  }
  function describe(file) {
    const values = [file.format || file.mime, file.size_bytes < 1048576 ? `${number(file.size_bytes / 1024)} KB` : `${number(file.size_bytes / 1048576)} MB`];
    if (file.dimensions) values.push(file.dimensions.join(" × "));
    if (file.spacing) values.push(`间距 ${file.spacing.map(number).join(" × ")} ${file.units === "mm" ? "mm" : file.units || "单位未知"}`);
    if (file.slice_count) values.push(`${file.slice_count} 层 / ${file.series_count} 序列`);
    if (file.modality && file.modality !== "unknown") values.push(file.modality);
    return values.filter(Boolean).join(" · ");
  }
  function renderFileList() {
    $("uploaded-assets").replaceChildren();
    if (!uploadedAssets.length) $("uploaded-assets").append(element("p", "还没有文件。", "settings-hint"));
    for (const file of uploadedAssets) {
      const row = element("div", undefined, "asset-item");
      row.append(element("strong", file.name || file.asset_id), element("p", describe(file), "settings-hint"));
      const actions = element("div", undefined, "button-row");
      if (["volume", "image", "dicom_slice", "mask"].includes(file.kind)) actions.append(action("打开影像", async () => { $("files-dialog").close(); await openImage(file.asset_id); }));
      if (file.kind !== "mask") actions.append(action("附加到对话", () => { addSelected(file); $("files-dialog").close(); notify("已附加到输入框。"); }, "text-button"));
      actions.append(action("下载", () => downloadAsset(file.asset_id, file.name), "text-button"));
      actions.append(action("删除", async () => {
        if (!confirm("删除此文件？服务器内容和派生预览将被移除。")) return;
        await api(`/api/v1/assets/${encodeURIComponent(file.asset_id)}`, { method: "DELETE" });
        uploadedAssets = uploadedAssets.filter(r => r.asset_id !== file.asset_id); selected = selected.filter(r => r.asset_id !== file.asset_id); fileCache.delete(file.asset_id);
        if (current?.asset_id === file.asset_id || currentMask?.asset_id === file.asset_id) closeImage(); renderFileList(); renderSelected();
      }, "text-button")); row.append(actions); $("uploaded-assets").append(row);
    }
  }
  async function refreshFiles() { uploadedAssets = await api("/api/v1/assets"); uploadedAssets.forEach(f => fileCache.set(f.asset_id, f)); renderFileList(); }
  $("files-dialog").addEventListener("toggle", () => { if ($("files-dialog").open && token) refreshFiles().catch(fail); });
  // toggle is not supported by all older dialog implementations; explicit open buttons work there too.
  $$('[data-open="files"]').forEach(b => b.addEventListener("click", () => refreshFiles().catch(fail)));
  $("upload-form").onsubmit = event => { event.preventDefault(); uploadFiles([...$("asset-file").files]).catch(fail); };
  $("dicom-folder").onchange = () => uploadFiles([...$("dicom-folder").files]).catch(fail);
  $("composer-file").onchange = () => { uploadFiles([...$("composer-file").files]).catch(fail); $("composer-file").value = ""; };
  $("attach-file").onclick = () => $("composer-file").click();
  $("example-upload").onclick = () => openPanel("files");
  $$('[data-example]').forEach(b => b.onclick = () => { $("prompt").value = b.dataset.example; $("prompt").focus(); });
  for (const name of ["dragenter", "dragover"]) $("chat-form").addEventListener(name, e => { e.preventDefault(); $("chat-form").classList.add("drag-over"); });
  $("chat-form").addEventListener("dragleave", e => { if (!$("chat-form").contains(e.relatedTarget)) $("chat-form").classList.remove("drag-over"); });
  $("chat-form").addEventListener("drop", e => { e.preventDefault(); $("chat-form").classList.remove("drag-over"); uploadFiles([...e.dataTransfer.files]).catch(fail); });
  $("reading-area").addEventListener("scroll", () => { const a = $("reading-area"); $("jump-latest").hidden = a.scrollHeight - a.scrollTop - a.clientHeight < 180; });
  $("jump-latest").onclick = () => { scrollToBottom(true); $("jump-latest").hidden = true; };

  async function fetchBitmap(path, signal) {
    const requestToken = token;
    const response = await fetch(path, { headers: { Authorization: `Bearer ${requestToken}` }, signal });
    if (!response.ok) { if (response.status === 401 && requestToken === token) expireSession(); const data = await response.json().catch(() => ({})); throw new Error(data.error?.message || "切片读取失败。"); }
    if (requestToken !== token) throw new Error("登录状态已变更。");
    return createImageBitmap(await response.blob());
  }
  function disposeViews() { surfaceController?.abort(); surface?.destroy(); surface = null; for (const v of views) { v.controller?.abort(); v.bitmap?.close(); v.observer?.disconnect(); } views = []; }
  function closeImage() { openSequence += 1; activeAutoJob = null; disposeViews(); current = null; currentMask = null; $("image-workspace").hidden = true; $("image-divider").hidden = true; $("mobile-image-tab").hidden = true; $("close-image-from-chat").hidden = true; $("workspace").classList.remove("has-image", "chat-active"); }
  function dismissImage() { if (activeAutoJob) dismissedJobs.add(activeAutoJob); closeImage(); }
  $("close-image").onclick = $("close-image-from-chat").onclick = dismissImage;
  document.addEventListener("keydown", e => { if (e.key === "Escape" && current && !document.querySelector("dialog[open]")) dismissImage(); });
  $("mobile-image-tab").onclick = () => { const chat = $("workspace").classList.toggle("chat-active"); $("mobile-image-tab").textContent = chat ? "查看影像" : "返回对话"; views.forEach(draw); };
  function draw(view) {
    if (!view.bitmap) return;
    const box = view.canvas.parentElement, ratio = devicePixelRatio || 1;
    const width = box.clientWidth, height = box.clientHeight;
    if (!width || !height) return;
    view.canvas.width = width * ratio; view.canvas.height = height * ratio;
    const ctx = view.canvas.getContext("2d"); ctx.scale(ratio, ratio); ctx.fillStyle = "#0c0f12"; ctx.fillRect(0, 0, width, height);
    const aspect = view.spec && current?.display_spacing ? current.display_spacing[view.spec.v] / current.display_spacing[view.spec.h] : 1;
    const iw = view.bitmap.width, ih = view.bitmap.height * aspect;
    const scale = Math.min((width - 28) / iw, (height - 28) / ih) * view.zoom;
    ctx.imageSmoothingEnabled = true; ctx.drawImage(view.bitmap, (width - iw * scale) / 2 + view.panX, (height - ih * scale) / 2 + view.panY, iw * scale, ih * scale);
  }
  async function loadSlice(view) {
    view.controller?.abort(); const controller = new AbortController(); view.controller = controller;
    const file = current, mask = currentMask, version = generation;
    let path = assetPath(file.asset_id, "preview");
    if (view.spec) {
      const query = new URLSearchParams({ plane: view.spec.key, index: view.index, window: $("window-width").value, level: $("window-level").value });
      if (mask) { query.set("mask_id", mask.asset_id); query.set("opacity", opacity); query.set("labels", [...labels].join(",")); }
      path = `${assetPath(file.asset_id, "slice")}?${query}`;
      view.counter.textContent = `${view.index + 1} / ${view.slider.max * 1 + 1}`;
    }
    try {
      const bitmap = await fetchBitmap(path, controller.signal);
      if (controller.signal.aborted || current !== file || version !== generation) { bitmap.close(); return; }
      view.bitmap?.close(); view.bitmap = bitmap; view.canvas.dataset.loadedLabels = mask ? [...labels].join(",") : ""; draw(view); $("viewer-error").textContent = "";
    } catch (e) { if (e.name !== "AbortError" && current === file) $("viewer-error").textContent = e.message; }
  }
  function reloadSlices() { views.forEach(v => loadSlice(v)); }
  function makeView(spec) {
    const root = element("div", undefined, `viewport${spec ? " plane-" + spec.key : " preview-image"}`), head = element("div", undefined, "viewport-heading");
    const counter = element("span"); head.append(element("span", spec?.name || "图像预览"), counter);
    const wrap = element("div", undefined, "canvas-wrap"), canvas = element("canvas"); canvas.setAttribute("aria-label", spec ? `${spec.name}切片` : "二维图像预览"); wrap.append(canvas);
    if (spec) {
      const codes = current.display_orientation;
      const directions = { top: codes[spec.v], bottom: opposite[codes[spec.v]], left: opposite[codes[spec.h]], right: codes[spec.h] };
      for (const [side, label] of Object.entries(directions)) wrap.append(element("span", label || "?", `direction ${side}`));
    }
    root.append(head, wrap); $("viewports").append(root);
    const view = { spec, root, canvas, counter, zoom: 1, panX: 0, panY: 0, index: 0 };
    if (spec) {
      const slider = element("input"); slider.type = "range"; slider.min = 0; slider.max = current.display_dimensions[spec.axis] - 1;
      view.index = Math.floor(Number(slider.max) / 2); slider.value = view.index; slider.setAttribute("aria-label", `${spec.name}切片编号`); view.slider = slider;
      let timer; slider.oninput = () => { view.index = Number(slider.value); clearTimeout(timer); timer = setTimeout(() => loadSlice(view), 60); }; root.append(slider);
      view.observer = new ResizeObserver(() => draw(view)); view.observer.observe(wrap);
    } else { view.observer = new ResizeObserver(() => draw(view)); view.observer.observe(wrap); }
    let pointer = null;
    wrap.onpointerdown = e => { if (e.button !== 0) return; pointer = { x: e.clientX, y: e.clientY, px: view.panX, py: view.panY }; wrap.setPointerCapture(e.pointerId); };
    wrap.onpointermove = e => { if (!pointer) return; view.panX = pointer.px + e.clientX - pointer.x; view.panY = pointer.py + e.clientY - pointer.y; draw(view); };
    wrap.onpointerup = wrap.onpointercancel = () => { pointer = null; };
    wrap.addEventListener("wheel", e => { e.preventDefault(); if (e.ctrlKey || !spec) { view.zoom = Math.max(.25, Math.min(8, view.zoom * (e.deltaY < 0 ? 1.1 : .9))); draw(view); } else { view.index = Math.max(0, Math.min(Number(view.slider.max), view.index + (e.deltaY > 0 ? 1 : -1))); view.slider.value = view.index; loadSlice(view); } }, { passive: false });
    views.push(view); loadSlice(view);
  }
  function makeSurfaceView(mask) {
    const pane = element("section", undefined, "viewport surface-pane"), head = element("div", undefined, "viewport-heading");
    head.append(element("span", "三维分割")); const reset = action("重置", () => surface?.reset(), "text-button"); reset.disabled = !mask; head.append(reset);
    const host = element("div", undefined, "surface-host"), hint = element("p", mask ? "正在构建结果表面…" : "分割完成后，三维结果将在这里显示", "surface-hint");
    pane.append(head, host, hint); $("viewports").append(pane);
    if (!mask) { host.append(element("div", "暂无分割结果", "surface-empty")); return; }
    const expected = mask.asset_id; surfaceController = new AbortController(); const loading = surfaceController;
    fetch(assetPath(expected, "surface"), {headers:{Authorization:`Bearer ${token}`},signal:surfaceController.signal}).then(async response => {
      if (!response.ok) { const data = await response.json().catch(()=>({})); throw new Error(data.error?.message || "三维表面读取失败"); }
      const mesh = await response.json(); if (loading.signal.aborted || currentMask?.asset_id !== expected || !token || !host.isConnected) return;
      surface = new SurfaceViewer(host); surface.setMesh(mesh); surface.setLabels(labels);
      hint.textContent = mesh.face_count ? `体素边界表面${mesh.stride > 1 ? ` · ${mesh.sampling?.join("×") || mesh.stride} 倍采样预览` : ""} · 拖动旋转，滚轮缩放` : "当前标签没有非零分割体素";
    }).catch(e => { if (e.name !== "AbortError" && currentMask?.asset_id === expected) hint.textContent = e.message; });
  }
  async function refreshCapabilities() {
    if (!token) return;
    capabilities = await api("/api/v1/imaging/capabilities");
    $("upload-limits").textContent = ` 上传上限 ${number(capabilities.max_upload_bytes / 1048576)} MB；保留 ${capabilities.retention_days} 天，到期拒绝访问并由维护命令清理。`;
    updateSegControls();
  }
  async function openImage(id, maskId = null, preserveView = false) {
    const version = generation, cid = conversationId, opening = ++openSequence; const file = await metadata(id), mask = maskId ? await metadata(maskId) : null;
    if (version !== generation || cid !== conversationId || opening !== openSequence) return;
    if (file.kind === "mask") return openImage(file.source_asset_id, id);
    if (!["volume", "image", "dicom_slice"].includes(file.kind)) throw new Error("此附件没有影像预览。");
    const previous = preserveView && current?.asset_id === file.asset_id ? {
      width: $("window-width").value, level: $("window-level").value,
      views: views.map(view => ({key: view.spec?.key, index: view.index, zoom: view.zoom, panX: view.panX, panY: view.panY})),
    } : null;
    disposeViews(); current = file; currentMask = mask; labels = new Set(Object.keys(mask?.labels || {}).map(Number).filter(v => v !== 0));
    $("image-title").textContent = file.name; $("image-metadata").textContent = describe(file);
    $("image-workspace").hidden = false; $("image-divider").hidden = false; $("mobile-image-tab").hidden = false; $("close-image-from-chat").hidden = false; $("mobile-image-tab").textContent = "返回对话";
    $("workspace").classList.add("has-image"); $("workspace").classList.remove("chat-active");
    $("viewports").replaceChildren(); $("viewer-error").textContent = ""; $("volume-controls").hidden = file.kind !== "volume";
    $("download-mask").hidden = !mask; $("opacity-control").hidden = !mask;
    if (file.kind === "volume") {
      const [low, high] = file.intensity_range; $("window-width").value = Math.max(high - low, .001); $("window-level").value = (low + high) / 2; specs.forEach(makeView); makeSurfaceView(mask);
    } else makeView(null);
    if (previous) {
      $("window-width").value = previous.width; $("window-level").value = previous.level;
      for (const view of views) {
        const saved = previous.views.find(item => item.key === view.spec?.key); if (!saved) continue;
        for (const key of ["index", "zoom", "panX", "panY"]) view[key] = saved[key];
        if (view.slider) view.slider.value = view.index;
      }
      reloadSlices();
    }
    $("mask-legend").replaceChildren(); $("image-measurements").replaceChildren();
    for (const [value, name] of Object.entries(mask?.labels || {})) {
      if (Number(value) === 0) continue;
      const label = element("label"), input = element("input"), dot = element("span", undefined, "legend-dot"); input.type = "checkbox"; input.checked = true; dot.style.backgroundColor = colors[(Number(value) - 1) % colors.length];
      input.onchange = () => { input.checked ? labels.add(Number(value)) : labels.delete(Number(value)); surface?.setLabels(labels); reloadSlices(); };
      label.append(input, dot, document.createTextNode(name)); $("mask-legend").append(label);
    }
    for (const row of mask?.measurements || []) {
      if (row.volume_mm3 !== undefined) { const text = element("p", `${row.name}：${number(row.volume_mm3)} mm³`); text.title = row.basis; $("image-measurements").append(text); }
    }
    const modality = $("image-modality"); if (modality) { modality.value = file.modality || "unknown"; modality.disabled = file.kind !== "volume" || reviewAccess.has(file.asset_id); }
    $("modality-save-status").textContent = "";
    updateSegControls(); refreshCapabilities().catch(fail);
  }
  function updateSegControls() {
    const configured = capabilities?.segmentation;
    const models = configured?.models || [];
    const compatible = models.filter(model => current?.kind === "volume" && model.modalities.includes(current.modality) && model.channels === current.channels);
    const connected = !configured?.service_status || configured.service_status === "ready";
    const enabled = configured?.status === "configured" && connected && current?.kind === "volume" && current.modality !== "unknown" && compatible.length > 0;
    $("ask-segmentation").disabled = !enabled;
    $("seg-status").textContent = configured?.status !== "configured" ? "分割服务未配置，仍可查看影像。" : !connected ? (configured?.service_message || "分割服务暂未连接，请稍后重试。") : current?.kind !== "volume" ? "分割需要三维影像，请上传完整体数据。" : !enabled ? "当前序列或通道不支持此分割任务，请核对影像序列。" : "可以开始分割。";
  }
  $("image-modality").onchange = async () => {
    if (!current || current.kind !== "volume") return;
    const file = current, body = new FormData(); body.append("modality", $("image-modality").value);
    $("image-modality").disabled = true;
    try {
      const result = await api(`/api/v1/assets/${encodeURIComponent(file.asset_id)}/modality`, { method: "POST", body });
      Object.assign(file, result); fileCache.set(file.asset_id, file);
      $("modality-save-status").textContent = "已保存影像序列。"; updateSegControls();
    } catch (e) { $("image-modality").value = file.modality || "unknown"; fail(e); }
    finally { $("image-modality").disabled = current?.kind !== "volume"; }
  };
  $("window-width").onchange = $("window-level").onchange = () => { if (Number($("window-width").value) > 0 && Number.isFinite(Number($("window-level").value))) reloadSlices(); };
  $("mask-opacity").oninput = () => { opacity = Number($("mask-opacity").value); reloadSlices(); };
  $("reset-views").onclick = () => { surface?.reset(); for (const view of views) { view.zoom = 1; view.panX = view.panY = 0; if (view.slider) { view.index = Math.floor(view.slider.max / 2); view.slider.value = view.index; } } if (current?.intensity_range) { const [a, b] = current.intensity_range; $("window-width").value = Math.max(.001, b - a); $("window-level").value = (a + b) / 2; } reloadSlices(); };
  $("download-mask").onclick = () => downloadAsset(currentMask.asset_id, "segmentation.nii.gz", assetPath(currentMask.asset_id, "download")).catch(fail);
  $("image-screenshot").onclick = () => {
    const view = views[0]; if (!view?.bitmap) { notify("请等待切片载入。"); return; }
    const image = document.createElement("canvas"); image.width = view.canvas.width; image.height = view.canvas.height;
    const ctx = image.getContext("2d"); ctx.drawImage(view.canvas, 0, 0); ctx.fillStyle = "#ddd"; ctx.font = "14px sans-serif";
    if (view.spec && current.display_orientation) {
      const h = current.display_orientation[view.spec.h], v = current.display_orientation[view.spec.v];
      ctx.fillText(opposite[h], 8, image.height / 2); ctx.fillText(h, image.width - 20, image.height / 2);
      ctx.fillText(v, image.width / 2, 18); ctx.fillText(opposite[v], image.width / 2, image.height - 8);
    }
    ctx.fillText(`${view.spec?.name || "图像"} ${view.index + 1}${currentMask ? " · segmentation overlay" : ""}`, 12, 22);
    image.toBlob(blob => { const url = URL.createObjectURL(blob), a = element("a"); a.href = url; a.download = "pharma-view.png"; a.click(); setTimeout(() => URL.revokeObjectURL(url), 2000); });
    if (reviewAccess.has(current?.asset_id) || account?.role === "reviewer") $("ask-segmentation").disabled = true;
  };
  $("ask-segmentation").onclick = () => {
    if (!current) return;
    addSelected(current);
    $("prompt").value = "请分割这份影像中的";
    $("workspace").classList.add("chat-active"); $("mobile-image-tab").textContent = "查看影像"; $("prompt").focus();
  };
  const divider = $("image-divider"); let resizing = false;
  function setWidth(value) { $("workspace").style.setProperty("--image-width", `${Math.max(32, Math.min(65, value))}%`); }
  divider.onpointerdown = e => { resizing = true; divider.setPointerCapture(e.pointerId); };
  divider.onpointermove = e => { if (resizing) { const box = $("workspace").getBoundingClientRect(); setWidth((box.right - e.clientX) / box.width * 100); } };
  divider.onpointerup = divider.onpointercancel = () => { resizing = false; };
  divider.onkeydown = e => { if (["ArrowLeft", "ArrowRight"].includes(e.key)) { e.preventDefault(); const value = parseFloat(getComputedStyle($("workspace")).getPropertyValue("--image-width")); setWidth(value + (e.key === "ArrowLeft" ? 2 : -2)); } };

  async function refreshJobs() {
    clearTimeout(jobTimer); const cid = conversationId, version = ++jobRefresh, viewAtRequest = openSequence;
    if (!token || !cid) { $("imaging-jobs").replaceChildren(); return; }
    const rows = await api(`/api/v1/conversations/${encodeURIComponent(cid)}/imaging-jobs`);
    if (cid !== conversationId || version !== jobRefresh) return;
    $("imaging-jobs").replaceChildren();
    for (const row of rows) {
      const card = element("article", undefined, "imaging-job"), header = element("header");
      const stage = row.status === "succeeded" ? "completed" : row.status === "failed" ? "failed" : row.result?.phase || row.status;
      header.append(element("strong", "影像分割"), element("span", `${stages[stage] || stage}${row.status !== "draft" ? ` · ${row.elapsed_seconds || 0}s` : ""}`));
      card.append(header, element("p", row.input?.file_name || "影像"));
      if (row.error_code) card.append(element("p", row.result?.error_message || `处理失败：${row.error_code}`, "warning"));
      const actions = element("div", undefined, "button-row");
      if (["draft", "failed"].includes(row.status)) actions.append(action(row.status === "draft" ? "确认并开始分割" : "重试分割", async () => { shownStarts.delete(row.id); dismissedJobs.delete(row.id); await api(`/api/v1/jobs/${encodeURIComponent(row.id)}/confirm`, { method: "POST", body: JSON.stringify({ confirmed: true }) }); await refreshJobs(); }));
      if (row.result?.asset_id) {
        actions.append(action("打开结果", () => openImage(row.result.source_asset_id, row.result.asset_id)));
        actions.append(action("下载 NIfTI", () => downloadAsset(row.result.asset_id, "segmentation.nii.gz"), "text-button"));
        actions.append(action("解释结果", () => { $("prompt").value = `请查询任务 ${row.id}，仅依据工具返回的标签与量化信息解释结果，说明单位和限制。`; $("prompt").focus(); }, "text-button"));
        if (account?.scopes?.includes("reviews:submit")) actions.append(action("提交科室复核", () => window.HospitalUI.submitReview(row.id), "text-button"));
      }
      const log = element("details"), summary = element("summary", "任务详情"); log.append(summary, element("pre", JSON.stringify({ job_id: row.id, phase: stage, error_code: row.error_code }, null, 2)));
      card.append(actions, log); $("imaging-jobs").append(card);
    }
    let shown = shownResults.get(cid);
    if (!shown) { shown = new Set(); shownResults.set(cid, shown); }
    const starting = rows.find(row => ["queued", "running"].includes(row.status) && row.input?.execution_mode === "full_access" && row.input?.file_id && !shownStarts.has(row.id));
    if (starting && viewAtRequest === openSequence) {
      for (const row of rows.filter(row => ["queued", "running"].includes(row.status))) shownStarts.add(row.id);
      try {
        await openImage(starting.input.file_id, null, true);
        if (cid === conversationId && version === jobRefresh && current?.asset_id === starting.input.file_id) activeAutoJob = starting.id;
      } catch (error) { if (cid === conversationId && version === jobRefresh) fail(error); }
    }
    if (cid !== conversationId || version !== jobRefresh) return;
    const completed = rows.filter(row => row.status === "succeeded" && row.result?.asset_id && row.result?.source_asset_id);
    const next = completed.find(row => !shown.has(`${row.id}:${row.result.asset_id}`));
    for (const row of completed) shown.add(`${row.id}:${row.result.asset_id}`);
    if (next && !dismissedJobs.has(next.id) && viewAtRequest === openSequence) {
      if (!current || current.asset_id === next.result.source_asset_id || next.input?.execution_mode === "full_access") {
        try { await openImage(next.result.source_asset_id, next.result.asset_id, true); }
        catch (error) { if (cid === conversationId && version === jobRefresh) fail(error); }
      } else notify(`${next.input?.file_name || "影像"}分割已完成，可在任务卡打开结果。`);
    }
    if (cid !== conversationId || version !== jobRefresh) return;
    if (rows.some(r => ["queued", "running"].includes(r.status))) jobTimer = setTimeout(() => refreshJobs().catch(fail), 2500);
  }
  async function modelInfo() {
    if (!token) return;
    $("health").textContent = "连接中";
    try {
      const info = await api("/api/v1/model-info"); $("model-name").textContent = info.model_id; $("model-name").removeAttribute("title");
      const select = $("chat-model"), previous = select.value; select.replaceChildren();
      for (const row of info.chat_models || [{alias:info.default_alias || "",name:"PharmaLLM-1.0"}]) { const option = element("option", row.name); option.value = row.alias; select.append(option); }
      select.value = [...select.options].some(o=>o.value===previous) ? previous : info.default_alias || select.options[0]?.value || "";
      $("health").textContent = { ready: "就绪", demo: "演示模式", unavailable: "模型离线" }[info.status] || "状态未知";
      $("health").classList.toggle("warning", info.status === "unavailable");
    } catch (e) { $("health").textContent = "未连接"; $("health").classList.add("warning"); }
  }
  window.ImagingUI = {
    async start() { generation += 1; try { await refreshCapabilities(); } catch (e) { fail(e); } await modelInfo(); await refreshFiles().catch(fail); },
    reset() { generation += 1; selected = []; fileCache.clear(); shownResults.clear(); shownStarts.clear(); dismissedJobs.clear(); reviewAccess.clear(); clearTimeout(jobTimer); jobRefresh += 1; closeImage(); renderSelected(); $("imaging-jobs").replaceChildren(); },
    conversationChanged() { generation += 1; selected = []; closeImage(); renderSelected(); refreshJobs().catch(fail); },
    refreshJobs() { refreshJobs().catch(fail); },
    attachmentIds() { return selected.filter(f => f.asset_id && f.status === "ready").map(f => f.asset_id); },
    consumeAttachments(ids) { const sent = new Set(ids); selected = selected.filter(file => !sent.has(file.asset_id)); renderSelected(); },
    uploading() { return uploads > 0 || selected.some(f => f.status === "failed"); },
    async useAttachments(ids) { selected = await Promise.all(ids.map(metadata)); renderSelected(); },
    messageAttachments(box, ids) { if (!ids.length) return; const row = element("div", undefined, "message-attachments"); for (const id of ids) { const button = action("查看附件", async () => { const file = await metadata(id); if (["volume", "image", "dicom_slice"].includes(file.kind)) await openImage(id); else await downloadAsset(id, file.name); }, "text-button"); metadata(id).then(f => { button.textContent = f.name || "附件"; }).catch(() => { button.textContent = "附件已不可用"; }); row.append(button); } box.append(row); },
    async openReview(review) {
      for (const id of [review.source_asset_id, review.result_asset_id]) { reviewAccess.set(id, review.id); fileCache.delete(id); }
      await openImage(review.source_asset_id, review.result_asset_id);
    },
    openImage, refreshFiles,
  };
  document.addEventListener("visibilitychange", () => { if (!document.hidden && token) { modelInfo(); refreshCapabilities().catch(fail); refreshJobs().catch(fail); } });
})();
