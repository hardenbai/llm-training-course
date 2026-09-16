(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const scenes = $$(".scene");
  const railItems = $$(".rail-item");
  const sceneStatus = $("#sceneStatus");
  const progressBar = $("#progressBar");
  const liveRegion = $("#liveRegion");
  const stage = $("#stage");
  const scrollIndicator = $("#scrollIndicator");
  const scrollMeter = $("#scrollMeter");
  const scrollPercent = $("#scrollPercent");
  let currentScene = 0;

  function updateScrollIndicator() {
    const maxScroll = Math.max(0, stage.scrollHeight - stage.clientHeight);
    const progress = maxScroll > 1 ? Math.min(1, stage.scrollTop / maxScroll) : 0;
    scrollIndicator.classList.toggle("is-visible", maxScroll > 12);
    scrollMeter.style.transform = `scaleY(${progress})`;
    scrollPercent.value = String(Math.round(progress * 100)).padStart(2, "0");
  }

  function resetSceneScroll() {
    stage.scrollTop = 0;
    window.scrollTo(0, 0);
    window.requestAnimationFrame(updateScrollIndicator);
  }

  function showScene(index, updateHash = true) {
    currentScene = Math.max(0, Math.min(scenes.length - 1, index));
    scenes.forEach((scene, i) => {
      const active = i === currentScene;
      scene.hidden = !active;
      scene.classList.toggle("is-active", active);
    });
    railItems.forEach((item, i) => item.classList.toggle("is-active", i === currentScene));
    sceneStatus.textContent = `${String(currentScene).padStart(2, "0")} / ${String(scenes.length - 1).padStart(2, "0")}`;
    progressBar.style.width = `${(currentScene / (scenes.length - 1)) * 100}%`;
    $("#prevScene").disabled = currentScene === 0;
    $("#nextScene").disabled = currentScene === scenes.length - 1;
    const heading = $("h1, h2", scenes[currentScene]);
    liveRegion.textContent = `第 ${currentScene} 章：${heading ? heading.textContent.trim() : ""}`;
    resetSceneScroll();
    if (updateHash) history.replaceState(null, "", `#scene-${currentScene}`);
  }

  railItems.forEach(item => item.addEventListener("click", () => showScene(Number(item.dataset.go))));
  $("#prevScene").addEventListener("click", () => showScene(currentScene - 1));
  $("#nextScene").addEventListener("click", () => showScene(currentScene + 1));
  $(".brand").addEventListener("click", event => { event.preventDefault(); showScene(0); });

  document.addEventListener("keydown", event => {
    const tag = document.activeElement?.tagName;
    const editing = ["INPUT", "SELECT", "TEXTAREA", "BUTTON", "A", "PRE"].includes(tag);
    if (editing) return;
    if (event.key === "ArrowRight") { event.preventDefault(); showScene(currentScene + 1); }
    if (event.key === "ArrowLeft") { event.preventDefault(); showScene(currentScene - 1); }
    if (["PageDown", "ArrowDown"].includes(event.key)) { event.preventDefault(); stage.scrollBy({ top: stage.clientHeight * .82, behavior: "smooth" }); }
    if (["PageUp", "ArrowUp"].includes(event.key)) { event.preventDefault(); stage.scrollBy({ top: -stage.clientHeight * .82, behavior: "smooth" }); }
    if (event.key === "Home") { event.preventDefault(); stage.scrollTo({ top: 0, behavior: "smooth" }); }
    if (event.key === "End") { event.preventDefault(); stage.scrollTo({ top: stage.scrollHeight, behavior: "smooth" }); }
    if (event.code === "Space") { event.preventDefault(); toggleMotion(); }
  });

  stage.addEventListener("scroll", updateScrollIndicator, { passive: true });
  window.addEventListener("resize", () => window.requestAnimationFrame(updateScrollIndicator));
  if ("ResizeObserver" in window) new ResizeObserver(updateScrollIndicator).observe(stage);

  const hashScene = Number(location.hash.replace("#scene-", ""));
  if (Number.isInteger(hashScene) && hashScene >= 0 && hashScene < scenes.length) showScene(hashScene, false);
  else showScene(0, false);
  window.addEventListener("load", () => window.requestAnimationFrame(resetSceneScroll), { once: true });

  // Presentation timer
  let timerRunning = false;
  let elapsedSeconds = 0;
  let timerId = null;
  const timerToggle = $("#timerToggle");
  const timerText = $("#timerText");

  function renderTimer() {
    const minutes = Math.floor(elapsedSeconds / 60);
    const seconds = elapsedSeconds % 60;
    timerText.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }

  timerToggle.addEventListener("click", () => {
    timerRunning = !timerRunning;
    timerToggle.setAttribute("aria-pressed", String(timerRunning));
    if (timerRunning) timerId = window.setInterval(() => { elapsedSeconds += 1; renderTimer(); }, 1000);
    else window.clearInterval(timerId);
  });
  timerToggle.addEventListener("dblclick", () => { elapsedSeconds = 0; renderTimer(); });

  // Global animation control
  let motionPaused = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const motionToggle = $("#motionToggle");

  function setSvgMotion(paused) {
    $$('svg').forEach(svg => {
      if (paused && typeof svg.pauseAnimations === "function") svg.pauseAnimations();
      if (!paused && typeof svg.unpauseAnimations === "function") svg.unpauseAnimations();
    });
  }

  function renderMotion() {
    $("#appShell").classList.toggle("is-paused", motionPaused);
    motionToggle.setAttribute("aria-pressed", String(motionPaused));
    $("#motionIcon").textContent = motionPaused ? "▶" : "Ⅱ";
    $("#motionLabel").textContent = motionPaused ? "继续动画" : "暂停动画";
    setSvgMotion(motionPaused);
  }

  function toggleMotion() { motionPaused = !motionPaused; renderMotion(); }
  motionToggle.addEventListener("click", toggleMotion);
  renderMotion();

  // Lifecycle
  const lifeContent = {
    ingest: {
      data: "压缩文件、网页、代码、图像",
      hardware: "SSD / 对象存储 / NIC / CPU",
      bottleneck: "容量、顺序读带宽、解压与小文件",
      why: "训练前，GPU 还没有开始工作。数据获取和预处理若跟不上，昂贵加速器只能等待。"
    },
    tokenize: {
      data: "清洗后的样本 → token IDs",
      hardware: "CPU / DDR / 本地缓存",
      bottleneck: "CPU 核数、内存带宽、随机读取",
      why: "Tokenizer 和数据增强常在主机侧完成。预取、并行 worker 与 pinned memory 决定加速器能否持续拿到 batch。"
    },
    train: {
      data: "token、参数、激活、梯度",
      hardware: "GPU/TPU/NPU + HBM + 高速互联",
      bottleneck: "计算、显存容量、集合通信",
      why: "前向生成激活，反向生成梯度，优化器更新状态。模型越大，单卡容量与跨卡同步越先成为约束。"
    },
    align: {
      data: "指令、偏好、rollout、奖励",
      hardware: "训练集群 + 推理集群 + 存储",
      bottleneck: "生成速度、采样并发、训练推理切换",
      why: "RL 类后训练把生成放进训练闭环，系统同时包含推理和反向传播，调度与数据新鲜度变得重要。"
    },
    serve: {
      data: "请求、权重、KV Cache、输出 token",
      hardware: "加速器 + HBM + 调度器 + 网络",
      bottleneck: "首 token 延迟、显存带宽、KV 容量",
      why: "服务面对不同长度和到达时间的请求。连续批处理与缓存管理往往比单次 forward 的微小优化更影响总体效率。"
    }
  };

  $$("[data-life]").forEach(button => button.addEventListener("click", () => {
    $$("[data-life]").forEach(item => item.classList.toggle("is-selected", item === button));
    const item = lifeContent[button.dataset.life];
    $("#lifeData").textContent = item.data;
    $("#lifeHardware").textContent = item.hardware;
    $("#lifeBottleneck").textContent = item.bottleneck;
    $("#lifeWhy").textContent = item.why;
  }));

  // Model workload switcher
  const modelContent = {
    dense: { title: "每个 token 都经过同一组参数", caption: "大矩阵乘法占主要 FLOPs", compute: "大规模 GEMM", state: "参数、激活、KV", pressure: "算力 + HBM 带宽", regularity: 88, regularityText: "高", explanation: "Dense Transformer 的每个 token 激活全部层参数，计算规则、容易形成大矩阵，适合高吞吐加速器。" },
    moe: { title: "每个 token 只进入少量专家", caption: "总参数大，激活参数较少", compute: "路由 + Expert GEMM", state: "专家权重、路由 token", pressure: "HBM 容量 + All-to-All", regularity: 55, regularityText: "中", explanation: "MoE 用稀疏激活降低每 token 计算，但专家分布在不同卡上时，token 必须重排并跨卡发送，通信和负载不均成为新问题。" },
    vlm: { title: "视觉 token 与文本 token 汇合", caption: "高分辨率带来更长序列", compute: "视觉编码 + LLM GEMM", state: "图像特征、文本 KV", pressure: "预处理 + 序列长度", regularity: 68, regularityText: "中高", explanation: "VLM 多了一条图像解码与视觉编码路径。图片分辨率、patch 数量与多图输入会显著改变 Prefill 成本。" },
    diffusion: { title: "同一潜变量被反复去噪", caption: "迭代次数决定总工作量", compute: "重复的网络前向", state: "latent、条件特征", pressure: "计算吞吐 + 激活", regularity: 80, regularityText: "高", explanation: "Diffusion 不是逐 token 自回归，而是在多个时间步重复执行网络。并行性更好，但总延迟受采样步数影响。" }
  };

  $$("[data-model]").forEach(button => button.addEventListener("click", () => {
    $$("[data-model]").forEach(item => item.classList.toggle("is-selected", item === button));
    const key = button.dataset.model;
    const item = modelContent[key];
    $("#modelCanvas").dataset.modelView = key;
    $("#ffnTitle").textContent = item.title;
    $("#ffnCaption").textContent = item.caption;
    $("#modelCompute").textContent = item.compute;
    $("#modelState").textContent = item.state;
    $("#modelPressure").textContent = item.pressure;
    $("#regularityMeter").style.width = `${item.regularity}%`;
    $("#regularityText").textContent = item.regularityText;
    $("#modelExplanation").textContent = item.explanation;
  }));

  // Training steps and memory estimator
  const trainingContent = {
    forward: { stamp: "FORWARD / 01", title: "把输入变成预测", text: "读取参数，执行矩阵乘法，并保存反向传播所需的中间激活。计算繁重，激活随 batch 和序列长度快速增长。", read: "参数、输入", write: "激活、logits", hardware: "Tensor Core / HBM" },
    backward: { stamp: "BACKWARD / 02", title: "沿计算图追回梯度", text: "反向执行大量矩阵乘法，读取前向保存的激活，并为每个参数累积梯度。计算量通常约为前向的两倍。", read: "激活、参数、输出梯度", write: "参数梯度", hardware: "Tensor Core / HBM / 互联" },
    optimizer: { stamp: "OPTIMIZER / 03", title: "用梯度改写模型状态", text: "AdamW 读取参数、梯度、一阶矩和二阶矩，再写回更新后的状态。计算相对简单，却会产生大量显存读写。", read: "参数、梯度、m、v", write: "新参数、新 m、新 v", hardware: "HBM 带宽 / 分片通信" }
  };

  $$("[data-train-step]").forEach(button => button.addEventListener("click", () => {
    $$("[data-train-step]").forEach(item => item.classList.toggle("is-selected", item === button));
    const key = button.dataset.trainStep;
    const item = trainingContent[key];
    $("#trainingMachine").dataset.step = key;
    $("#trainStamp").textContent = item.stamp;
    $("#trainTitle").textContent = item.title;
    $("#trainText").textContent = item.text;
    $("#trainRead").textContent = item.read;
    $("#trainWrite").textContent = item.write;
    $("#trainHardware").textContent = item.hardware;
  }));

  function updateTrainMemory() {
    const params = Number($("#trainParams").value);
    const bytes = Number($("#trainPrecision").value);
    $("#trainParamsOut").textContent = params;
    $("#trainMemory").textContent = Math.round(params * bytes).toLocaleString("zh-CN");
  }
  $("#trainParams").addEventListener("input", updateTrainMemory);
  $("#trainPrecision").addEventListener("change", updateTrainMemory);

  // Parallel topology
  const parallelContent = {
    dp: { stamp: "DATA PARALLEL", title: "每卡一份模型，各算不同数据", text: "前向和反向独立进行；得到梯度后做 All-Reduce。扩 batch 容易，但每张卡都必须放下完整模型状态。", split: "batch", comm: "gradient", link: "跨节点网络可用" },
    tp: { stamp: "TENSOR PARALLEL", title: "一层矩阵切到多张卡", text: "每层都需要组合局部结果，通信频率高、延迟敏感。能让单卡放不下的一层继续计算，但强依赖节点内高速互联。", split: "matrix / hidden", comm: "activation", link: "NVLink / ICI" },
    pp: { stamp: "PIPELINE PARALLEL", title: "不同卡负责不同层", text: "微批次像工件一样流过多个 stage。通信内容是 stage 边界激活，代价是流水线气泡与复杂调度。", split: "layers", comm: "boundary activation", link: "低延迟点对点" },
    ep: { stamp: "EXPERT PARALLEL", title: "不同卡保存不同专家", text: "路由器把 token 发往对应专家，再把结果发回。MoE 总参数可扩展，但 All-to-All 和热点专家会限制效率。", split: "experts", comm: "routed tokens", link: "高双向带宽" }
  };

  const topologyPositions = [
    [115, 105], [290, 105], [465, 105], [640, 105],
    [115, 255], [290, 255], [465, 255], [640, 255]
  ];

  function linePath(a, b) { return `M ${a[0]} ${a[1]} L ${b[0]} ${b[1]}`; }

  function renderTopology(mode) {
    const links = $(".topology-links");
    const nodes = $(".gpu-nodes");
    const signals = $(".moving-signals");
    let pairs = [];
    if (mode === "dp") pairs = [[0,1],[1,2],[2,3],[3,7],[7,6],[6,5],[5,4],[4,0]];
    if (mode === "tp") pairs = [[0,1],[1,2],[2,3],[0,2],[1,3],[4,5],[5,6],[6,7],[4,6],[5,7]];
    if (mode === "pp") pairs = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]];
    if (mode === "ep") pairs = [[0,4],[0,5],[1,5],[1,6],[2,6],[2,7],[3,7],[3,4]];
    links.innerHTML = pairs.map(([a,b], index) => `<path class="link" id="link-${index}" d="${linePath(topologyPositions[a], topologyPositions[b])}"></path>`).join("");
    nodes.innerHTML = topologyPositions.map(([x,y], index) => {
      const modelWidth = mode === "dp" ? 54 : mode === "tp" ? 12 : mode === "pp" ? 54 : 23;
      const modelX = mode === "tp" ? -27 + (index % 4) * 14 : -27;
      const dataX = mode === "dp" ? -26 + (index % 4) * 13 : -26;
      const label = mode === "pp" ? `L${index * 4 + 1}–${index * 4 + 4}` : mode === "ep" ? `E${index * 2}–${index * 2 + 1}` : `GPU ${index}`;
      return `<g transform="translate(${x} ${y})"><rect class="node-shell" x="-48" y="-33" width="96" height="66"></rect><rect class="node-model" x="${modelX}" y="-17" width="${modelWidth}" height="13"></rect><rect class="node-data" x="${dataX}" y="4" width="${mode === "dp" ? 12 : 52}" height="8"></rect><text class="node-label" text-anchor="middle" y="26">${label}</text></g>`;
    }).join("");
    signals.innerHTML = pairs.slice(0, 8).map(([a,b], index) => {
      const path = linePath(topologyPositions[a], topologyPositions[b]);
      return `<circle class="signal" r="3"><animateMotion dur="${1.3 + (index % 3) * .35}s" begin="${index * .12}s" repeatCount="indefinite" path="${path}"></animateMotion></circle>`;
    }).join("");
    setSvgMotion(motionPaused);
  }

  $$("[data-parallel]").forEach(button => button.addEventListener("click", () => {
    $$("[data-parallel]").forEach(item => item.classList.toggle("is-selected", item === button));
    const key = button.dataset.parallel;
    const item = parallelContent[key];
    $("#topology").dataset.parallelView = key;
    $("#parallelStamp").textContent = item.stamp;
    $("#parallelTitle").textContent = item.title;
    $("#parallelText").textContent = item.text;
    $("#parallelSplit").textContent = item.split;
    $("#parallelComm").textContent = item.comm;
    $("#parallelLink").textContent = item.link;
    renderTopology(key);
  }));
  renderTopology("dp");

  // Inference phase and KV estimator
  const modelPresets = {
    "7b": { layers: 32, kvHeads: 8, headDim: 128 },
    "70b": { layers: 80, kvHeads: 8, headDim: 128 },
    "moe": { layers: 48, kvHeads: 8, headDim: 128 }
  };
  let inferMode = "prefill";

  function renderTokens() {
    const row = $("#tokenRow");
    const count = inferMode === "prefill" ? 28 : 18;
    row.innerHTML = Array.from({ length: count }, (_, index) => {
      const isNew = inferMode === "decode" && index === count - 1;
      return `<i class="${isNew ? "is-new" : ""}" style="animation-delay:${index * .025}s">${isNew ? "+1" : String(index + 1).padStart(2, "0")}</i>`;
    }).join("");
  }

  function updateKv() {
    const preset = modelPresets[$("#modelPreset").value];
    const context = Number($("#contextRange").value);
    const batch = Number($("#batchRange").value);
    const bytes = Number($("#kvPrecision").value);
    const cache = 2 * preset.layers * preset.kvHeads * preset.headDim * context * batch * bytes / 1e9;
    $("#contextOut").textContent = context.toLocaleString("zh-CN");
    $("#batchOut").textContent = batch;
    $("#kvValue").textContent = cache < 10 ? cache.toFixed(2) : cache.toFixed(1);
    const filled = Math.min(64, Math.max(1, Math.round(8 + 56 * cache / (cache + 18))));
    $("#kvGrid").innerHTML = Array.from({ length: 64 }, (_, index) => `<i class="${index < filled ? "is-filled" : ""}"></i>`).join("");
  }

  function setInferMode(mode) {
    inferMode = mode;
    $$("[data-infer]").forEach(item => item.classList.toggle("is-selected", item.dataset.infer === mode));
    $("#inferenceRig").dataset.inferView = mode;
    if (mode === "prefill") {
      $("#phaseAction").textContent = "一次处理大量输入 token";
      $("#phaseReuse").textContent = "权重被同一批 token 反复复用";
      $("#compassNeedle").style.left = "76%";
      $("#inferBottleneck").textContent = "大 GEMM 更容易喂饱计算单元";
      $("#inferReason").textContent = "矩阵的 batch/sequence 维度较大，同一份权重参与许多 token 的运算，算术强度较高。";
    } else {
      $("#phaseAction").textContent = "每一步只生成一个新 token";
      $("#phaseReuse").textContent = "为极少计算反复读取大量权重与 KV";
      $("#compassNeedle").style.left = "19%";
      $("#inferBottleneck").textContent = "小 GEMM 常在等待 HBM 数据";
      $("#inferReason").textContent = "低 batch Decode 难以复用权重，算术强度下降；显存带宽与跨卡边界开始主导逐 token 延迟。";
    }
    renderTokens();
  }

  $$("[data-infer]").forEach(button => button.addEventListener("click", () => setInferMode(button.dataset.infer)));
  ["modelPreset", "contextRange", "batchRange", "kvPrecision"].forEach(id => {
    $(`#${id}`).addEventListener(id.includes("Range") ? "input" : "change", updateKv);
  });
  renderTokens();
  updateKv();

  // Serving animation
  function renderServing(burst = 5) {
    $("#requestQueue").innerHTML = Array.from({ length: burst }, (_, index) => `<i style="width:${42 + ((index * 17) % 58)}%;animation-delay:${index * .08}s"></i>`).join("");
    $("#prefillJobs").innerHTML = Array.from({ length: Math.max(2, Math.ceil(burst / 2)) }, () => "<i></i>").join("");
    $("#decodeJobs").innerHTML = Array.from({ length: Math.max(6, burst * 2) }, () => "<i></i>").join("");
    $("#cacheBlocks").innerHTML = Array.from({ length: 24 }, (_, index) => `<i class="${index < burst * 3 ? "used" : ""}" style="animation-delay:${index * .05}s"></i>`).join("");
  }
  let burstSize = 5;
  $("#serveBurst").addEventListener("click", () => { burstSize = burstSize >= 8 ? 3 : burstSize + 1; renderServing(burstSize); });
  renderServing();

  // Hardware map
  const hardwareContent = {
    storage: { stamp: "PERSISTENT STORAGE", title: "对象存储 / NVMe", text: "保存原始数据、tokenized shards、checkpoint 与日志。容量和持续吞吐重要；大量小文件、解压和共享访问会把理论带宽吃掉。", wait: "GPU 等待下一个 batch 或 checkpoint", metric: "GB/s、IOPS、容量、恢复时间", mistake: "只看容量，不看并发读取和数据格式" },
    host: { stamp: "HOST SYSTEM", title: "CPU + DDR", text: "负责 Python 控制流、Tokenizer、解压、DataLoader、网络协议与算子发射。它通常不做主要 GEMM，却能让加速器因供给不足而空转。", wait: "数据准备与控制路径等待", metric: "核心数、内存带宽、NUMA、预取", mistake: "认为 CPU 与 AI 性能无关" },
    pcie: { stamp: "HOST–DEVICE LINK", title: "PCIe / DMA", text: "连接主机内存与设备显存。Pinned memory、异步拷贝和数据预取用于把传输隐藏在计算之后。", wait: "Host 到 Device 的搬移", metric: "单向带宽、拓扑、重叠比例", mistake: "把 PCIe 与 GPU 间高速互联混为一谈" },
    compute: { stamp: "ACCELERATOR", title: "GPU / TPU / NPU", text: "用大量并行乘加单元执行 GEMM、Attention 与其他张量算子。GPU 更通用；TPU/NPU 通过专用数据流和编译栈提高特定工作负载效率。", wait: "串行计算等待", metric: "有效 FLOPS、利用率、精度支持", mistake: "只看峰值 FLOPS，不看带宽和算子覆盖" },
    hbm: { stamp: "DEVICE MEMORY", title: "HBM 高带宽显存", text: "在计算附近保存权重、激活、梯度、优化器状态和 KV Cache。容量决定能否运行，带宽决定许多 Decode 与 optimizer 工作负载有多快。", wait: "计算单元等待权重与状态", metric: "容量、GB/s、访问局部性", mistake: "把显存容量和显存带宽当成同一指标" },
    scaleup: { stamp: "SCALE-UP FABRIC", title: "NVLink / NVSwitch / ICI", text: "连接同一节点或紧耦合域内的加速器，为高频、低延迟集合通信服务，尤其适合 Tensor Parallel 与 Expert Parallel。", wait: "层内激活和 token 路由同步", metric: "双向带宽、延迟、拓扑、bisection", mistake: "用 PCIe 带宽估算节点内 TP" },
    scaleout: { stamp: "SCALE-OUT NETWORK", title: "RDMA NIC / 数据中心网络", text: "把训练扩展到多个节点，承载梯度、激活、专家 token 与 checkpoint 流量。网络拓扑和拥塞控制会影响整个作业的 goodput。", wait: "跨节点集合通信与长尾", metric: "NIC 带宽、时延、拥塞、无阻塞比例", mistake: "只看单链路速率，不看全局通信模式" }
  };

  function selectHardware(button) {
    $$('[data-hardware]').forEach(item => item.classList.toggle("is-selected", item === button));
    const item = hardwareContent[button.dataset.hardware];
    $("#hardwareStamp").textContent = item.stamp;
    $("#hardwareTitle").textContent = item.title;
    $("#hardwareText").textContent = item.text;
    $("#hardwareWait").textContent = item.wait;
    $("#hardwareMetric").textContent = item.metric;
    $("#hardwareMistake").textContent = item.mistake;
  }
  $$('[data-hardware]').forEach(button => button.addEventListener("click", () => selectHardware(button)));
  selectHardware($('[data-hardware="compute"]'));

  // Workload decision lab
  const workloads = {
    denseTrain: { scores: { storage: 48, host: 42, compute: 92, memory: 82, network: 76 }, titles: { compute: ["让大矩阵持续占满计算单元", "Dense 预训练通常有高算术强度；集群效率取决于是否能同时隐藏通信与数据供给。", "优先：有效低精度算力、模型 FLOPS 利用率、稳定集群"] } },
    moeTrain: { scores: { storage: 45, host: 38, compute: 78, memory: 88, network: 96 }, titles: { network: ["先解决专家之间的 token 交换", "MoE 的稀疏激活减少计算，却引入高频 All-to-All；热点专家还会制造通信和计算长尾。", "优先：高 bisection 带宽、EP 拓扑、负载均衡与容量"] } },
    chat: { scores: { storage: 24, host: 42, compute: 66, memory: 92, network: 58 }, titles: { memory: ["先解决权重与 KV 的读取速度", "低 batch Decode 的算术强度低，HBM 容量和带宽通常比峰值 FLOPS 更能影响逐 token 延迟。", "优先：足够 HBM、量化、KV 管理、减少跨卡边界"] } },
    batch: { scores: { storage: 38, host: 35, compute: 91, memory: 72, network: 48 }, titles: { compute: ["用大 batch 把计算单元喂饱", "离线任务可以容忍排队，通过更大的 batch 提高权重复用，让推理从带宽受限向计算受限移动。", "优先：高有效算力、动态组批、吞吐/成本而非单请求延迟"] } },
    vlm: { scores: { storage: 54, host: 76, compute: 86, memory: 78, network: 50 }, titles: { compute: ["视觉编码与长 Prefill 共同吃算力", "图像解码和预处理会增加主机压力；高分辨率产生更多视觉 token，主要推高 Prefill 计算与 KV 占用。", "优先：预处理流水线、视觉 batch、足够 HBM 与计算吞吐"] } },
    diffusion: { scores: { storage: 28, host: 30, compute: 96, memory: 68, network: 35 }, titles: { compute: ["减少每一步成本或减少采样步数", "Diffusion 在多个时间步重复前向，矩阵并行度高；总延迟更多受算力、分辨率和采样步数影响。", "优先：低精度算力、算子融合、蒸馏或更少采样步"] } }
  };

  const resourceNames = { storage: "STORAGE", host: "HOST", compute: "COMPUTE", memory: "MEMORY", network: "NETWORK" };
  function updateDecision() {
    const item = workloads[$("#workloadSelect").value];
    const scores = { ...item.scores };
    const goal = $("#goalSelect").value;
    const scale = $("#scaleSelect").value;
    if (goal === "latency") { scores.memory += 8; scores.network += 4; }
    if (goal === "throughput") { scores.compute += 7; scores.host += 3; }
    if (goal === "cost") { scores.memory += 3; scores.compute += 3; scores.storage += 2; }
    if (scale === "cluster") scores.network += 7;
    Object.keys(scores).forEach(key => { scores[key] = Math.min(100, scores[key]); });
    const primary = Object.entries(scores).sort((a,b) => b[1] - a[1])[0][0];
    $$(".bottleneck-axis > div").forEach(row => {
      const key = row.dataset.resource;
      row.classList.toggle("is-primary", key === primary);
      $("i", row).style.width = `${scores[key]}%`;
      $("b", row).textContent = scores[key];
    });
    const fallback = item.titles[Object.keys(item.titles)[0]];
    const copy = item.titles[primary] || fallback;
    $("#decisionTag").textContent = `PRIMARY BOTTLENECK · ${resourceNames[primary]}`;
    $("#decisionTitle").textContent = copy[0];
    $("#decisionText").textContent = copy[1];
    $("#decisionAction").textContent = copy[2];
  }
  ["workloadSelect", "goalSelect", "scaleSelect"].forEach(id => $(`#${id}`).addEventListener("change", updateDecision));
  updateDecision();

  // Code evidence
  const codeContent = {
    train: {
      filename: "01_training_step.py", stamp: "TRAINING LOOP", title: "四行代码对应三类硬件行为",
      text: "forward 读取权重并保存激活；backward 生成梯度；optimizer 读取并改写参数状态。框架隐藏了调度，但没有消除数据搬移。",
      lab: "labs/01_training_step.py", source: "https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py", sourceLabel: "MiniMind · train_pretrain.py",
      code: `logits = model(input_ids)                 # 前向：参数 → 激活\nloss = cross_entropy(logits, labels)     # 比较预测与答案\noptimizer.zero_grad(set_to_none=True)\nloss.backward()                           # 反向：激活 → 梯度\nclip_grad_norm_(model.parameters(), 1.0)\noptimizer.step()                          # 更新：读写参数与状态`
    },
    kv: {
      filename: "03_kv_cache_demo.py", stamp: "INCREMENTAL DECODE", title: "缓存 K/V，避免重复投影历史 token",
      text: "每层保存历史 token 的 Key 和 Value；新一步只生成新的 Q/K/V。计算减少了，但 Cache 随上下文与 batch 线性增长。",
      lab: "labs/03_kv_cache_demo.py", source: "https://github.com/karpathy/nanochat/blob/master/nanochat/engine.py", sourceLabel: "nanochat · engine.py",
      code: `# 首次 Prefill：一次写入全部历史 K/V\nk, v = project_kv(prompt)\ncache.write(layer=layer_id, positions=range(T), k=k, v=v)\n\n# Decode：只投影一个新 token\nq_new, k_new, v_new = project(last_token)\ncache.append(k_new, v_new)\noutput = attention(q_new, cache.keys, cache.values)`
    },
    parallel: {
      filename: "04_parallelism_cost.py", stamp: "COLLECTIVE COMMUNICATION", title: "切开矩阵后，局部结果必须重新合并",
      text: "Tensor Parallel 往往在许多层中通信激活，因此偏好节点内高速互联；Data Parallel 主要同步梯度，更容易跨节点。",
      lab: "labs/04_parallelism_cost.py", source: "https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html", sourceLabel: "PyTorch · Tensor Parallel",
      code: `# 列并行：每卡计算一部分输出通道\nlocal_y = local_x @ local_weight\n\n# 行并行：每卡得到部分和，必须合并\ndist.all_reduce(local_y, op=dist.ReduceOp.SUM)\n\n# 通信频率 × 激活大小 × 拓扑\n# 决定 TP 能否扩展`
    },
    budget: {
      filename: "02_memory_budget.py", stamp: "CAPACITY FIRST", title: "先算是否装得下，再讨论能跑多快",
      text: "权重精度决定静态容量；训练还要保存梯度与优化器状态；在线推理则可能由并发请求的 KV Cache 吃掉剩余空间。",
      lab: "labs/02_memory_budget.py", source: "https://github.com/bojieli/ai-infra-book/tree/main/calculations", sourceLabel: "AI Infra Book · calculations",
      code: `# ZeRO-1/2/3 依次分片 optimizer、gradient、parameter\nfor name, bytes_per_param, sharded_from in states:\n    divisor = shards if zero_stage >= sharded_from else 1\n    per_gpu_gb += params_b * bytes_per_param / divisor\n\nkv_bytes = 2 * layers * kv_heads * head_dim\nkv_bytes *= context * batch * bytes_per_value`
    },
    moe: {
      filename: "06_moe_routing.py", stamp: "EXPERT PARALLEL", title: "稀疏计算把压力转移到路由与网络",
      text: "平均 expert load 不能代表尾部：热点专家会超过 capacity，其他设备则等待。Top-k 越大，激活专家计算和 dispatch/combine 数据量也越大。",
      lab: "labs/06_moe_routing.py", source: "https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py", sourceLabel: "MiniMind · MOEFeedForward",
      code: `scores = softmax(router(tokens), dim=-1)\nweights, experts = topk(scores, k=top_k)\n\n# token 按 expert 重排；跨设备时成为 All-to-All\nfor expert_id in range(num_experts):\n    routed = tokens[experts == expert_id]\n    outputs[expert_id] = expert(routed)\n\n# 热点 expert 决定整层尾延迟`
    },
    serving: {
      filename: "07_continuous_batching.py", stamp: "ITERATION SCHEDULING", title: "每个 token 边界都重新利用空槽",
      text: "静态批必须等待最长请求；Continuous Batching 让已完成请求立即释放槽位，并把等待队列中的请求补入活跃 batch。",
      lab: "labs/07_continuous_batching.py", source: "https://github.com/bojieli/ai-infra-book/tree/main/calculations/results", sourceLabel: "AI Infra Book · service calculations",
      code: `while waiting or active:\n    fill_free_slots(active, waiting)\n    logits = model.decode_one_token(active)\n    active = sample_and_advance(logits)\n\n    # 完成即回收 KV blocks；下一轮补入新请求\n    release_finished(active, kv_block_manager)`
    }
  };

  function renderCode(key) {
    const item = codeContent[key];
    $("#codeFilename").textContent = item.filename;
    $("#codeBlock code").textContent = item.code;
    $("#codeStamp").textContent = item.stamp;
    $("#codeTitle").textContent = item.title;
    $("#codeText").textContent = item.text;
    $("#labLink").href = item.lab;
    $("#labLink").textContent = item.lab;
    $("#sourceLink").href = item.source;
    $("#sourceLink").textContent = item.sourceLabel;
    $$("[data-code]").forEach(button => {
      const selected = button.dataset.code === key;
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-selected", String(selected));
    });
  }
  $$("[data-code]").forEach(button => button.addEventListener("click", () => renderCode(button.dataset.code)));
  renderCode("train");
})();
