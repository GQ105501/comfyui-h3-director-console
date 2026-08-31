import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TAB_ID = "h3-director-console";
const NODE_TYPE = "H3DirectorConsole";
const STORE_KEY = "h3DirectorConsole.activeProject";
const STEP_OPTIONS = [4, 6, 8, 12, 20];
const DEFAULT_INFERENCE = { steps: 20, sampler: "res_multistep", scheduler: "beta" };

let rootElement = null;
let config = null;
let project = null;
let stopRequested = false;
let saveTimer = null;
let renderTimer = null;
const runtime = new Map();

const uid = (prefix = "shot") => `${prefix}-${crypto.randomUUID().slice(0, 8)}`;
const html = (tag, className, text) => {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
};

function defaultShot(index = 1) {
  return {
    id: uid("shot"),
    title: `镜头 ${String(index).padStart(3, "0")}`,
    prompt: "",
    enabled: true,
    assets: { images: [], videos: [], audios: [] },
  };
}

function defaultProject() {
  return {
    version: 4,
    id: `director-${new Date().toISOString().slice(0, 10)}`,
    name: "MiniMax H3 导演台项目",
    settings: {
      mode: "ref2v",
      aspect_ratio: "16_9",
      megapixels: "0_6",
      duration_seconds: 10,
      fps: 24,
      acceleration_enabled: false,
      acceleration_lora: "ref2v_v0_1",
      steps: 20,
      sampler: "res_multistep",
      scheduler: "beta",
      lora_strength: 1,
      seed: 2026082601,
      seed_mode: "increment",
      sequence_mode: "continuous",
      continuity: true,
      continuity_strategy: "masked_latent",
      context_frames: 39,
      video_feather_tokens: 0,
      audio_feather_ticks: 8,
      continuity_prompt_lock: true,
      second_pass: false,
      output_quality: "720p",
      refine_steps: 2,
    },
    shots: [defaultShot()],
  };
}

async function request(path, options = {}) {
  const response = await api.fetchApi(path, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = { ok: false, error: `服务器返回了非 JSON 响应（${response.status}）` };
  }
  if (!response.ok || payload.ok === false) throw new Error(payload.error || `请求失败（${response.status}）`);
  return payload;
}

async function initialize() {
  config = await request("/director_console/config");
  const activeId = localStorage.getItem(STORE_KEY);
  const selected = activeId || config.projects?.[0]?.id;
  if (selected) {
    try {
      project = (await request(`/director_console/project/${encodeURIComponent(selected)}`)).project;
    } catch {
      project = defaultProject();
    }
  } else {
    project = defaultProject();
  }
  localStorage.setItem(STORE_KEY, project.id);
  await refreshAllOutputs();
  render();
}

function setNotice(message, tone = "info") {
  if (!rootElement) return;
  const region = rootElement.querySelector("[data-live]");
  if (!region) return;
  region.textContent = message;
  region.dataset.tone = tone;
}

async function saveProject({ quiet = false } = {}) {
  if (!project) return;
  const result = await request("/director_console/project", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(project),
  });
  // Do not replace `project` here. Rendered controls hold references to its
  // settings and shots; replacing the object after an autosave would make all
  // subsequent handlers mutate stale state. Import explicitly adopts the
  // normalized response before it renders a new control tree.
  localStorage.setItem(STORE_KEY, result.project.id);
  if (!quiet) setNotice("项目已保存", "success");
  return result.project;
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => saveProject({ quiet: true }).catch((error) => setNotice(error.message, "error")), 700);
}

// Rebuilding the sidebar inside a checkbox's change event can replay label
// activation against the replacement control. A short macrotask delay lets
// both the browser and assistive automation finish the action before teardown.
function scheduleRender() {
  if (renderTimer !== null) clearTimeout(renderTimer);
  renderTimer = setTimeout(() => {
    renderTimer = null;
    render();
  }, 120);
}

function field(labelText, control, hint = "") {
  const label = html("label", "h3dc-field");
  label.append(html("span", "h3dc-label", labelText), control);
  if (hint) label.append(html("small", "h3dc-hint", hint));
  return label;
}

function selectControl(value, options, onChange) {
  const select = html("select", "h3dc-select");
  for (const option of options) {
    const item = html("option", "", option.label ?? String(option.value));
    item.value = option.value;
    item.selected = String(option.value) === String(value);
    item.disabled = Boolean(option.disabled);
    if (option.title) item.title = option.title;
    select.append(item);
  }
  select.addEventListener("change", () => onChange(select.value));
  return select;
}

function numberControl(value, min, max, step, onChange) {
  const input = html("input", "h3dc-input");
  Object.assign(input, { type: "number", value, min, max, step });
  input.addEventListener("change", () => onChange(Number(input.value)));
  return input;
}

function checkboxControl(checked, onChange, text) {
  const wrapper = html("div", "h3dc-check");
  const input = html("input");
  const id = uid("switch");
  input.type = "checkbox";
  input.id = id;
  input.checked = Boolean(checked);
  input.addEventListener("change", () => onChange(input.checked));
  const label = html("label", "", text);
  label.htmlFor = id;
  wrapper.append(input, label);
  return wrapper;
}

function button(text, action, kind = "ghost", title = "") {
  const control = html("button", `h3dc-btn h3dc-btn--${kind}`, text);
  control.type = "button";
  control.title = title || text;
  control.addEventListener("click", action);
  return control;
}

function modeDependencies(mode) {
  const dependencies = config?.dependencies;
  if (!dependencies) return { ready: false, missing: ["依赖状态未知"] };
  const requiredModels = ["clip", "video_vae", "audio_vae"];
  requiredModels.push(config.modes[mode]?.model || "ref_model");
  if (project.settings.acceleration_enabled) {
    const lora = config.loras[project.settings.acceleration_lora];
    if (!lora) return { ready: false, missing: ["请选择加速 LoRA"] };
    if (!lora.compatible_modes.includes(mode)) {
      return { ready: false, missing: [`${lora.label} 与 ${config.modes[mode]?.label || mode} 不兼容`] };
    }
    requiredModels.push(lora.model);
  }
  if (project.settings.second_pass) requiredModels.push("upscaler");
  const missing = requiredModels
    .filter((key) => !dependencies.models[key]?.available)
    .map((key) => dependencies.models[key]?.filename || key);
  const requiredNodes = ["MiniMaxH3SigmaShift", "PathchSageAttentionKJ", "CreateVideo", "SaveVideo"];
  if (project.settings.acceleration_enabled) requiredNodes.push("LoraLoaderModelOnly");
  requiredNodes.push(mode === "ref2v" ? "MiniMaxH3ReferenceToVideo" : "JZL_MiniMaxH3ImageToVideoDual");
  if (mode === "ref2v") requiredNodes.push("VHS_LoadVideo");
  if (project.settings.continuity) {
    requiredNodes.push("H3DirectorMotionCondition", "H3DirectorTrimAV", "H3DirectorSaveAVLatent", "H3DirectorLoadAVLatent");
    if (project.settings.continuity_strategy !== "motion_context") requiredNodes.push("H3DirectorGeneratedAVMaskedContext");
  }
  if (project.settings.second_pass) {
    requiredNodes.push("MinimaxH3LatentUpscaler3D", "LTXVSeparateAVLatent", "LTXVConcatAVLatent");
    if (mode === "ref2v") requiredNodes.push("JZL_MiniMaxH3CondSync");
  }
  const missingNodes = requiredNodes.filter((name) => !dependencies.nodes[name]);
  return { ready: missing.length === 0 && missingNodes.length === 0, missing: [...missing, ...missingNodes] };
}

function renderDependencyBanner(container) {
  const report = modeDependencies(project.settings.mode);
  const banner = html("div", `h3dc-deps ${report.ready ? "is-ready" : "is-missing"}`);
  banner.append(html("strong", "", report.ready ? "运行环境已就绪" : "当前模式缺少依赖"));
  banner.append(html("span", "", report.ready ? "模型和节点均已检测到" : report.missing.join("、")));
  container.append(banner);
}

function refreshDependencyBanner() {
  const current = rootElement?.querySelector(".h3dc-deps");
  if (!current) return;
  const holder = html("div");
  renderDependencyBanner(holder);
  current.replaceWith(holder.firstElementChild);
}

function frameSummary(seconds, fps = 24) {
  let frames = Math.max(5, Math.round(Number(seconds) * fps));
  while (frames % 17 !== 5) frames += 1;
  return `${frames} 帧，实际约 ${(frames / fps).toFixed(2)} 秒`;
}

function alignDimension(value) {
  return Math.max(32, Math.min(4096, Math.round(Number(value) / 32) * 32));
}

function baseDimensions(settings) {
  const aspect = config.aspect_ratios[settings.aspect_ratio];
  const megapixels = config.megapixels[settings.megapixels]?.value;
  if (!aspect || !megapixels) return { width: settings.width || 1024, height: settings.height || 576 };
  const ratio = aspect.width / aspect.height;
  const pixels = megapixels * 1024 * 1024;
  const height = Math.sqrt(pixels / ratio);
  return { width: alignDimension(height * ratio), height: alignDimension(height) };
}

function outputDimensions(settings, qualityId = settings.output_quality) {
  const aspect = config.aspect_ratios[settings.aspect_ratio];
  const quality = config.output_qualities[qualityId];
  if (!aspect || !quality) return { width: 1280, height: 736 };
  const ratio = aspect.width / aspect.height;
  const shortEdge = quality.short_edge;
  let width = ratio >= 1 ? shortEdge * ratio : shortEdge;
  let height = ratio >= 1 ? shortEdge : shortEdge / ratio;
  if (Math.max(width, height) > 4096) {
    const scale = 4096 / Math.max(width, height);
    width *= scale;
    height *= scale;
  }
  return { width: alignDimension(width), height: alignDimension(height) };
}

function sizeLabel({ width, height }) {
  return `${width}×${height}`;
}

function outputQualityOptions(settings) {
  const base = baseDimensions(settings);
  const baseArea = base.width * base.height;
  return Object.entries(config.output_qualities).map(([value, item]) => {
    const size = outputDimensions(settings, value);
    const disabled = size.width * size.height <= baseArea;
    return {
      value,
      label: `${item.label} · ${sizeLabel(size)}`,
      disabled,
      title: disabled ? "该档位不高于基础生成尺寸，不能用于二次采样" : "",
    };
  });
}

function ensureOutputQuality(settings) {
  const options = outputQualityOptions(settings);
  if (options.some((item) => item.value === settings.output_quality && !item.disabled)) return;
  settings.output_quality = options.find((item) => !item.disabled)?.value || options.at(-1)?.value || "2160p";
}

function compatibleLoraOptions(mode) {
  return Object.entries(config.loras).map(([value, item]) => ({
    value,
    label: item.experimental_modes?.includes(mode) ? `${item.label}（跨模式实验）` : item.label,
    disabled: !item.compatible_modes.includes(mode),
    title: item.experimental_modes?.includes(mode)
      ? "可以加载运行，但官方尚未将此版本列为 Ref2VA 专用 LoRA"
      : (item.compatible_modes.includes(mode) ? "" : "与当前生成模式不兼容"),
  }));
}

function recommendedLora(mode) {
  return Object.entries(config.loras).find(([, item]) => (
    item.compatible_modes.includes(mode) && !item.experimental_modes?.includes(mode)
  )) || Object.entries(config.loras).find(([, item]) => item.compatible_modes.includes(mode));
}

function applyAccelerationPreset(loraId) {
  const preset = config.loras[loraId];
  if (!preset) return;
  Object.assign(project.settings, {
    acceleration_lora: loraId,
    steps: preset.recommended_steps,
    sampler: preset.sampler,
    scheduler: preset.scheduler,
  });
}

function renderSettings(container) {
  const settings = project.settings;
  const panel = html("details", "h3dc-settings");
  panel.open = true;
  panel.append(html("summary", "", "生成设置"));
  const grid = html("div", "h3dc-settings-grid");
  const modeOptions = Object.entries(config.modes).map(([value, item]) => ({ value, label: item.label }));
  grid.append(field("生成模式", selectControl(settings.mode, modeOptions, (value) => {
    settings.mode = value;
    if (settings.acceleration_enabled) {
      const current = config.loras[settings.acceleration_lora];
      if (!current?.compatible_modes.includes(value)) {
        const compatible = recommendedLora(value);
        if (compatible) applyAccelerationPreset(compatible[0]);
      }
    }
    scheduleSave();
    scheduleRender();
  }), config.modes[settings.mode]?.description || ""));
  grid.append(field("画面宽高比", selectControl(settings.aspect_ratio, Object.entries(config.aspect_ratios).map(([value, item]) => ({ value, label: item.label })), (value) => {
    settings.aspect_ratio = value; ensureOutputQuality(settings); scheduleSave(); scheduleRender();
  }), "二次采样会自动继承，不需要重复选择"));
  grid.append(field("基础像素量", selectControl(settings.megapixels, Object.entries(config.megapixels).map(([value, item]) => ({ value, label: item.label })), (value) => {
    settings.megapixels = value; ensureOutputQuality(settings); scheduleSave(); scheduleRender();
  }), `H3 对齐后的实际尺寸：${sizeLabel(baseDimensions(settings))}`));
  grid.append(field("时长（秒）", numberControl(settings.duration_seconds, 1, 15, 0.5, (value) => {
    settings.duration_seconds = value; scheduleSave(); scheduleRender();
  }), frameSummary(settings.duration_seconds, settings.fps || 24)));
  grid.append(field("基础种子", numberControl(settings.seed, 0, 4294967295, 1, (value) => { settings.seed = value; scheduleSave(); })));
  const sequenceModes = config.sequence_modes || {
    continuous: { label: "连续长镜头", description: "锁定机位和运动，把所有片段视为同一次拍摄。" },
    shots: { label: "独立分镜", description: "允许切换景别和机位，后期按正常剪辑连接。" },
  };
  grid.append(field("片段关系", selectControl(settings.sequence_mode || "continuous", Object.entries(sequenceModes).map(([value, item]) => ({
    value,
    label: item.label,
  })), (value) => {
    const continuous = value === "continuous";
    Object.assign(settings, {
      sequence_mode: value,
      continuity: continuous,
      continuity_strategy: "masked_latent",
      context_frames: 39,
      video_feather_tokens: 0,
      audio_feather_ticks: 8,
      continuity_prompt_lock: continuous,
    });
    scheduleSave(); scheduleRender(); refreshDependencyBanner();
  }), sequenceModes[settings.sequence_mode || "continuous"]?.description || ""));
  panel.append(grid);

  const sequenceMode = settings.sequence_mode || "continuous";
  const sequenceSummary = html(
    "p",
    `h3dc-sequence-summary is-${sequenceMode}`,
    sequenceMode === "continuous"
      ? "连续长镜头：自动继承 39 帧运动与音视频 latent，并从上一段 latent 提取真实末帧锁定构图；关闭画面溶解羽化，覆盖后续提示词中的推拉摇移、景别和切镜指令。"
      : "独立分镜：每段从零生成，不继承上一段 latent；可以自由改变景别、机位和镜头运动。",
  );
  sequenceSummary.setAttribute("role", "note");

  const accelerationHost = html("div", "h3dc-dynamic-settings");
  const secondPassHost = html("div", "h3dc-dynamic-settings");

  function renderAccelerationHost() {
    accelerationHost.replaceChildren();
    if (!settings.acceleration_enabled) {
      accelerationHost.append(html("p", "h3dc-inference-summary", "未启用加速：使用 H3 默认 20 步 · res_multistep · beta，不加载任何加速 LoRA。"));
      return;
    }
    const acceleration = html("div", "h3dc-settings-grid h3dc-subsettings");
    const selectedLora = config.loras[settings.acceleration_lora];
    const loraHint = selectedLora?.experimental_modes?.includes(settings.mode)
      ? "跨模式实验用法：官方 Ref2VA 专用版仍是 Ref2V v0.1；建议对比画质、声音和稳定性"
      : "优先显示当前主模型的官方专用 LoRA；实验兼容项会单独标注";
    acceleration.append(field("加速 LoRA", selectControl(settings.acceleration_lora, compatibleLoraOptions(settings.mode), (value) => {
      applyAccelerationPreset(value); scheduleSave(); renderAccelerationHost(); refreshDependencyBanner();
    }), loraHint));
    acceleration.append(field("加速步数", selectControl(settings.steps, STEP_OPTIONS.map((value) => ({ value })), (value) => {
      settings.steps = Number(value); scheduleSave(); renderAccelerationHost();
    }), `当前 LoRA 推荐 ${config.loras[settings.acceleration_lora]?.recommended_steps || 4} 步，可自行对比`));
    acceleration.append(field("LoRA 强度", numberControl(settings.lora_strength, 0, 2, 0.05, (value) => {
      settings.lora_strength = value; scheduleSave();
    })));
    acceleration.append(field("采样器", selectControl(settings.sampler, ["euler", "res_multistep"].map((value) => ({ value })), (value) => {
      settings.sampler = value; scheduleSave();
    })));
    acceleration.append(field("调度器", selectControl(settings.scheduler, ["simple", "beta"].map((value) => ({ value })), (value) => {
      settings.scheduler = value; scheduleSave();
    })));
    accelerationHost.append(acceleration);
  }

  function renderSecondPassHost() {
    secondPassHost.replaceChildren();
    if (!settings.second_pass) return;
    const second = html("div", "h3dc-settings-grid h3dc-second-pass");
    const targetOptions = outputQualityOptions(settings);
    const availableTargets = targetOptions.filter((item) => !item.disabled);
    if (!availableTargets.length) {
      secondPassHost.append(html("p", "h3dc-warning", "当前基础分辨率已经是最高预设，无法再开启二次采样放大。"));
      return;
    }
    if (!availableTargets.some((item) => item.value === settings.output_quality)) {
      settings.output_quality = availableTargets[0].value;
      scheduleSave();
    }
    second.append(field("成片清晰度", selectControl(settings.output_quality, targetOptions, (value) => {
      settings.output_quality = value; scheduleSave(); renderSecondPassHost();
    }), `自动继承 ${config.aspect_ratios[settings.aspect_ratio]?.label || "当前宽高比"}；H3 对齐后为 ${sizeLabel(outputDimensions(settings))}`));
    second.append(field("细化步数", numberControl(settings.refine_steps, 1, Math.max(1, settings.steps - 1), 1, (value) => {
      settings.refine_steps = value; scheduleSave();
    }), "总步数不变，只把末段留给高分辨率细化"));
    const target = outputDimensions(settings);
    const targetMp = target.width * target.height / 1_000_000;
    const warning = targetMp > 1.2
      ? `当前目标约 ${targetMp.toFixed(2)}MP；4070 Ti 12GB 很可能显存不足，建议先用 720P。`
      : "4070 Ti 12GB 建议基础使用 0.6MP，并先用 720P 验证二采。";
    secondPassHost.append(second, html("p", "h3dc-warning", warning));
  }

  const switches = html("div", "h3dc-switches");
  switches.append(checkboxControl(settings.acceleration_enabled, (value) => {
    settings.acceleration_enabled = value;
    if (value) {
      const current = config.loras[settings.acceleration_lora];
      if (!current?.compatible_modes.includes(settings.mode)) {
        const compatible = recommendedLora(settings.mode);
        if (compatible) applyAccelerationPreset(compatible[0]);
      } else {
        applyAccelerationPreset(settings.acceleration_lora);
      }
    } else {
      Object.assign(settings, DEFAULT_INFERENCE);
    }
    scheduleSave();
    renderAccelerationHost();
    renderSecondPassHost();
    refreshDependencyBanner();
  }, "启用 LoRA 加速"));
  switches.append(checkboxControl(settings.second_pass, (value) => {
    settings.second_pass = value; scheduleSave(); renderSecondPassHost(); refreshDependencyBanner();
  }, "二次采样放大"));
  panel.append(sequenceSummary, switches, accelerationHost, secondPassHost);
  renderAccelerationHost();
  renderSecondPassHost();
  container.append(panel);
}

async function uploadAssets(shot, kind, files) {
  const mode = project.settings.mode;
  const limits = mode === "i2v" ? { images: 2, videos: 0, audios: 0 } : { images: 9, videos: 3, audios: 3 };
  const totalRoom = mode === "ref2v" ? 12 - Object.values(shot.assets).reduce((sum, items) => sum + items.length, 0) : limits[kind];
  const room = Math.min(limits[kind] - shot.assets[kind].length, totalRoom);
  if (room <= 0) throw new Error(`${kind} 已达到数量上限`);
  for (const file of [...files].slice(0, room)) {
    const body = new FormData();
    body.append("file", file);
    body.append("kind", kind);
    body.append("project_id", project.id);
    body.append("shot_id", shot.id);
    const result = await request("/director_console/upload", { method: "POST", body });
    shot.assets[kind].push({ path: result.asset.path, name: result.asset.name });
    setNotice(`已上传 ${file.name}`, "success");
  }
  await saveProject({ quiet: true });
  render();
}

function assetSection(shot, kind, label, accept, limit) {
  const section = html("section", "h3dc-assets");
  const heading = html("div", "h3dc-assets-head");
  heading.append(html("span", "", `${label} ${shot.assets[kind].length}/${limit}`));
  const input = html("input", "h3dc-file-input");
  input.type = "file";
  input.multiple = limit > 1;
  input.accept = accept;
  input.setAttribute("aria-label", `上传${label}`);
  input.addEventListener("change", () => uploadAssets(shot, kind, input.files).catch((error) => setNotice(error.message, "error")));
  const add = button("添加", () => input.click(), "quiet", `上传${label}`);
  heading.append(add, input);
  section.append(heading);
  const list = html("div", "h3dc-asset-list");
  shot.assets[kind].forEach((asset, index) => {
    const item = html("div", "h3dc-asset-chip");
    let reference;
    if (project.settings.mode === "i2v" && kind === "images") reference = index === 0 ? "<首帧>" : "<尾帧>";
    else reference = kind === "images" ? `<Picture ${index + 1}>` : kind === "videos" ? `<Video ${index + 1}>` : `<Audio ${index + 1}>`;
    const name = html("span", "", `${reference} ${asset.name}`);
    name.title = asset.path;
    const controls = html("span", "h3dc-chip-actions");
    controls.append(
      button("↑", () => moveAsset(shot, kind, index, -1), "icon", "向前移动"),
      button("↓", () => moveAsset(shot, kind, index, 1), "icon", "向后移动"),
      button("×", () => { shot.assets[kind].splice(index, 1); scheduleSave(); scheduleRender(); }, "icon", "从镜头移除"),
    );
    item.append(name, controls);
    list.append(item);
  });
  section.append(list);
  return section;
}

function moveAsset(shot, kind, index, delta) {
  const target = index + delta;
  if (target < 0 || target >= shot.assets[kind].length) return;
  [shot.assets[kind][index], shot.assets[kind][target]] = [shot.assets[kind][target], shot.assets[kind][index]];
  scheduleSave();
  scheduleRender();
}

function outputUrl(item) {
  const query = new URLSearchParams({ filename: item.filename, subfolder: item.subfolder, type: item.type || "output" });
  return api.apiURL(`/view?${query}`);
}

function renderShot(shot, index) {
  const card = html("article", "h3dc-shot");
  card.draggable = true;
  card.dataset.shotId = shot.id;
  card.addEventListener("dragstart", (event) => event.dataTransfer.setData("text/plain", shot.id));
  card.addEventListener("dragover", (event) => { event.preventDefault(); card.classList.add("is-dragover"); });
  card.addEventListener("dragleave", () => card.classList.remove("is-dragover"));
  card.addEventListener("drop", (event) => {
    event.preventDefault();
    card.classList.remove("is-dragover");
    reorderShot(event.dataTransfer.getData("text/plain"), shot.id);
  });

  const header = html("header", "h3dc-shot-head");
  const drag = html("span", "h3dc-drag", "⋮⋮");
  drag.title = "拖动排序";
  const order = html("span", "h3dc-order", String(index + 1).padStart(3, "0"));
  const title = html("input", "h3dc-title");
  title.value = shot.title;
  title.setAttribute("aria-label", `镜头 ${index + 1} 名称`);
  title.addEventListener("input", () => { shot.title = title.value; scheduleSave(); });
  const statusValue = runtime.get(shot.id)?.status || (shot.outputs?.length ? "已生成" : "待生成");
  const status = html("span", "h3dc-status", statusValue);
  status.dataset.status = statusValue;
  header.append(drag, order, title, status);
  card.append(header);

  const prompt = html("textarea", "h3dc-prompt");
  prompt.value = shot.prompt;
  const promptHints = {
    t2v: "输入 MiniMax H3 文生视频提示词；此模式只读取提示词。",
    i2v: "输入图生视频提示词；第一张图是首帧，第二张图是可选尾帧。",
    ref2v: "输入多参考提示词；素材按顺序映射到 <Picture N> / <Video N> / <Audio N>。",
  };
  const sequenceHint = (project.settings.sequence_mode || "continuous") === "continuous"
    ? (index === 0
      ? "这是长镜头起始段，可以定义初始机位；后续片段会继承该构图。"
      : "这是长镜头续段：只描述接下来的动作，不要写近景、远景、推拉摇移、切镜或转场。")
    : "这是独立分镜，可以自由描述新的景别、机位和镜头运动。";
  prompt.placeholder = `${promptHints[project.settings.mode]}\n${sequenceHint}`;
  prompt.setAttribute("aria-label", `${shot.title} 提示词`);
  prompt.addEventListener("input", () => { shot.prompt = prompt.value; scheduleSave(); });
  card.append(html("p", "h3dc-sequence-note", sequenceHint), prompt);

  if (project.settings.mode === "t2v") {
    card.append(html("p", "h3dc-mode-note", "文生视频不读取图片、视频或音频；切换回其他模式后，已保存的素材仍会保留。"));
  } else {
    const assets = html("div", "h3dc-asset-grid");
    if (project.settings.mode === "i2v") {
      assets.classList.add("is-single");
      assets.append(assetSection(shot, "images", "首帧 / 尾帧", "image/*", 2));
    } else {
      assets.append(
        assetSection(shot, "images", "图片", "image/*", 9),
        assetSection(shot, "videos", "视频", "video/*", 3),
        assetSection(shot, "audios", "音频", "audio/*", 3),
      );
      const total = Object.values(shot.assets).reduce((sum, items) => sum + items.length, 0);
      assets.prepend(html("p", "h3dc-asset-total", `参考素材 ${total}/12`));
    }
    card.append(assets);
  }

  const latest = shot.outputs?.[0];
  if (latest) {
    const preview = html("div", "h3dc-preview");
    const video = html("video", "");
    video.controls = true;
    video.preload = "metadata";
    video.src = outputUrl(latest);
    video.setAttribute("aria-label", `${shot.title} 最新生成片段`);
    preview.append(video, html("small", "", `最新结果 · ${(latest.size / 1024 / 1024).toFixed(1)} MB`));
    card.append(preview);
  }

  const actions = html("footer", "h3dc-shot-actions");
  actions.append(
    button(latest ? "重新生成" : "生成此镜头", () => runShots([shot.id]), "primary"),
    button("复制镜头", () => duplicateShot(index), "ghost"),
    button("删除", () => deleteShot(index), "danger"),
  );
  card.append(actions);
  return card;
}

function reorderShot(sourceId, targetId) {
  if (!sourceId || sourceId === targetId) return;
  const from = project.shots.findIndex((shot) => shot.id === sourceId);
  const to = project.shots.findIndex((shot) => shot.id === targetId);
  if (from < 0 || to < 0) return;
  const [moved] = project.shots.splice(from, 1);
  project.shots.splice(to, 0, moved);
  scheduleSave();
  setNotice("镜头顺序已变更；连续模式下请从受影响的最早镜头重新生成", "warning");
  render();
}

function duplicateShot(index) {
  const source = project.shots[index];
  const copy = structuredClone(source);
  copy.id = uid("shot");
  copy.title = `${source.title} · 副本`;
  copy.outputs = [];
  project.shots.splice(index + 1, 0, copy);
  scheduleSave();
  render();
}

function deleteShot(index) {
  if (project.shots.length === 1) {
    setNotice("至少保留一个镜头", "error");
    return;
  }
  if (!confirm(`删除“${project.shots[index].title}”？已生成的视频不会从磁盘删除。`)) return;
  project.shots.splice(index, 1);
  scheduleSave();
  render();
}

async function refreshOutputs(shot) {
  const query = new URLSearchParams({ project_id: project.id, shot_id: shot.id });
  const result = await request(`/director_console/outputs?${query}`);
  shot.outputs = result.outputs;
}

async function refreshAllOutputs() {
  if (!project) return;
  await Promise.all(project.shots.map((shot) => refreshOutputs(shot).catch(() => { shot.outputs ||= []; })));
}

async function waitForPrompt(promptId, shot) {
  const started = Date.now();
  while (!stopRequested) {
    const response = await api.fetchApi(`/history/${encodeURIComponent(promptId)}`);
    const history = await response.json();
    const entry = history[promptId];
    if (entry) {
      const status = entry.status || {};
      if (status.completed || status.status_str === "success") return entry;
      if (status.status_str === "error") throw new Error(status.messages?.at(-1)?.[1]?.exception_message || "ComfyUI 执行失败");
    }
    const elapsed = Math.floor((Date.now() - started) / 1000);
    runtime.set(shot.id, { status: `生成中 ${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, "0")}` });
    updateShotStatus(shot.id);
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  throw new Error("已停止后续镜头")
}

function updateShotStatus(shotId) {
  const card = rootElement?.querySelector(`[data-shot-id="${CSS.escape(shotId)}"]`);
  const status = card?.querySelector(".h3dc-status");
  if (status) status.textContent = runtime.get(shotId)?.status || "待生成";
}

async function runShots(ids) {
  const report = modeDependencies(project.settings.mode);
  if (!report.ready) {
    setNotice(`无法运行，缺少：${report.missing.join("、")}`, "error");
    return;
  }
  const selected = project.shots.filter((shot) => ids.includes(shot.id) && shot.enabled !== false);
  if (!selected.length) return;
  stopRequested = false;
  await saveProject({ quiet: true });
  for (const shot of selected) {
    if (stopRequested) break;
    try {
      runtime.set(shot.id, { status: "正在装配" });
      updateShotStatus(shot.id);
      const built = await request("/director_console/build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project, shot_id: shot.id }),
      });
      if (built.warnings?.length) setNotice(built.warnings.join("；"), "warning");
      const clientId = crypto.randomUUID();
      const queued = await request("/prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: built.prompt, client_id: clientId }),
      });
      runtime.set(shot.id, { status: "生成中 0:00", promptId: queued.prompt_id });
      updateShotStatus(shot.id);
      await waitForPrompt(queued.prompt_id, shot);
      await refreshOutputs(shot);
      runtime.set(shot.id, { status: "已生成" });
      await saveProject({ quiet: true });
      render();
    } catch (error) {
      runtime.set(shot.id, { status: "失败" });
      updateShotStatus(shot.id);
      setNotice(`${shot.title}：${error.message}`, "error");
      break;
    }
  }
  if (!stopRequested && selected.every((shot) => runtime.get(shot.id)?.status === "已生成")) {
    setNotice(`已按顺序完成 ${selected.length} 个镜头`, "success");
  }
}

async function stopRun() {
  stopRequested = true;
  try { await api.fetchApi("/interrupt", { method: "POST" }); } catch { /* Comfy may already be idle. */ }
  setNotice("已请求停止；当前 CUDA 操作结束后会中断", "warning");
}

function exportProject() {
  const blob = new Blob([JSON.stringify(project, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${project.id}.director.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function importProject() {
  const input = html("input");
  input.type = "file";
  input.accept = ".json,.director.json";
  input.addEventListener("change", async () => {
    try {
      const imported = JSON.parse(await input.files[0].text());
      project = imported;
      project = await saveProject();
      await refreshAllOutputs();
      render();
    } catch (error) {
      setNotice(`导入失败：${error.message}`, "error");
    }
  });
  input.click();
}

function renderToolbar(container) {
  const toolbar = html("div", "h3dc-toolbar");
  const name = html("input", "h3dc-project-name");
  name.value = project.name;
  name.setAttribute("aria-label", "项目名称");
  name.addEventListener("input", () => { project.name = name.value; scheduleSave(); });
  toolbar.append(name);
  const actions = html("div", "h3dc-toolbar-actions");
  actions.append(
    button("+ 镜头", () => { project.shots.push(defaultShot(project.shots.length + 1)); scheduleSave(); render(); }, "ghost"),
    button("保存", () => saveProject().catch((error) => setNotice(error.message, "error")), "ghost"),
    button("导入", importProject, "quiet"),
    button("导出", exportProject, "quiet"),
    button("停止", stopRun, "danger"),
    button("按顺序生成", () => runShots(project.shots.map((shot) => shot.id)), "primary"),
  );
  toolbar.append(actions);
  container.append(toolbar);
}

function render() {
  if (!rootElement || !project || !config) return;
  rootElement.replaceChildren();
  const appRoot = html("main", "h3dc-app");
  const header = html("header", "h3dc-header");
  header.append(html("div", "h3dc-mark", "H3"));
  const heading = html("div", "");
  heading.append(html("h1", "", "MiniMax H3 导演台"), html("p", "", "镜头编排 · 多参考 · 连续生成 · 一键重抽"));
  header.append(heading);
  appRoot.append(header);
  const live = html("div", "h3dc-live", "就绪");
  live.dataset.live = "";
  live.setAttribute("role", "status");
  live.setAttribute("aria-live", "polite");
  appRoot.append(live);
  renderDependencyBanner(appRoot);
  renderToolbar(appRoot);
  renderSettings(appRoot);
  const list = html("section", "h3dc-shot-list");
  list.setAttribute("aria-label", "镜头列表");
  project.shots.forEach((shot, index) => list.append(renderShot(shot, index)));
  appRoot.append(list);
  rootElement.append(appRoot);
}

function openSidebar() {
  const selectors = [
    `[data-tab-id="${TAB_ID}"]`,
    `[data-id="${TAB_ID}"]`,
    `[title="MiniMax H3 导演台"]`,
  ];
  const target = selectors.map((selector) => document.querySelector(selector)).find(Boolean);
  target?.click();
}

app.registerExtension({
  name: "H3DirectorConsole",
  async setup() {
    const style = document.createElement("link");
    style.rel = "stylesheet";
    style.href = new URL("./director_console.css", import.meta.url).href;
    document.head.append(style);
    app.extensionManager.registerSidebarTab({
      id: TAB_ID,
      title: "MiniMax H3 导演台",
      icon: "pi pi-video",
      type: "custom",
      render: (element) => {
        rootElement = element;
        element.classList.add("h3dc-root");
        initialize().catch((error) => {
          element.textContent = `导演台加载失败：${error.message}`;
        });
      },
    });
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      this.title = "🎬 MiniMax H3 导演台入口";
      this.addWidget("button", "打开导演台", null, openSidebar);
      this.size = [330, 120];
    };
  },
});
