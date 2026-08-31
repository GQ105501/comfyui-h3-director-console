# MiniMax H3 Director Console

一个面向多镜头制作的 ComfyUI 导演台。它用镜头卡片管理提示词和参考素材，按排序逐段提交 MiniMax H3，并在卡片内显示最新视频和提供单镜头重抽。

完整的模型清单、节点依赖、目录位置、迁移方法和故障排查见 [使用与部署说明](docs/使用与部署说明.md)。

## 功能

- 文生、图生、多参考三种输入模式；界面只显示当前模式允许的素材槽位。
- 基础尺寸分别选择宽高比和百万像素；界面显示 H3 对齐后的实际宽高。时长按秒输入并自动对齐到 H3 的 `17k+5` 帧网格。
- 镜头拖动排序、顺序执行、停止队列、单镜头重新生成、项目导入/导出。
- LoRA 加速独立开关；关闭时固定使用默认 20 步，开启后才选择兼容 LoRA、步数和强度。
- 片段关系分为“连续长镜头”和“独立分镜”：前者继承原生音视频 latent 并锁定运动/机位，后者允许正常切镜。
- 内置双端 Masked AV Bridge 节点，可在两个已知片段之间生成缺失的过渡区域。
- 可选的 3D latent 放大和二次采样；最终视频直接显示在镜头卡片中。

## 架构

导演台是一个本地单体扩展：前端边栏保存镜头 JSON；后端验证项目、接收素材并为单个镜头动态构建普通 Comfy API 图。模型推理仍由标准节点完成，不装配 TE-Speed，也不要求单独安装 Motion Context。ComfyUI 0.34.0 及以上直接使用原生 H3 逐 token 音视频掩码；0.33.4 会按能力检测仅在内存中补齐兼容行为，不修改 ComfyUI 磁盘文件。

```text
镜头卡片 → 项目 JSON → 连续性合同 → 工作流构建器 → Comfy /prompt
    ↑                                                ↓
视频预览 ← output 扫描 ← 裁掉保护前缀 ← Masked H3 采样链
```

## 安装

从仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\install_h3_director_console.ps1
```

脚本会在当前 ComfyUI 的 `custom_nodes` 下创建指向本目录的 Junction。重启 ComfyUI 后，在右侧边栏打开“MiniMax H3 导演台”，也可加载 `workflow_templates/H3-Director-Console-review.json`。

支持 ComfyUI `0.33.4+`；推荐 `0.34.0+`。从旧版本升级导演台后必须重启 ComfyUI，使内置节点完成注册。

## 片段关系与连续性

“连续长镜头”把所有片段视为同一次拍摄。S002 以后读取上一段保存的联合音视频 latent，将尾部直接复制到当前目标 latent 的开头，并设置 `noise_mask=0` 保护已知区域。上下文固定为 39 帧（1.625 秒），同时位于 H3 的 24fps 画面网格和 40Hz 音频网格上；音频使用 8 ticks（约 0.2 秒）羽化。画面羽化固定为 0，避免 latent 混合形成溶解、重影和双重构图。

后续片段还会从上一段已接受的 latent 自动解码真实末帧，并将其作为最高优先级构图锚点。多参考模式会把用户素材顺延为 `<Picture 2>` 起，提示词标签由系统自动重映射；每个后续镜头最多使用 11 个用户素材，为构图锚点预留 1 个位置。

导演台会把连续性合同注入 H3 的 `detailed_description` 或 `integrated_multimodal_description`：继承上一段机位高度、焦段、人物比例、屏幕位置、背景锚点、相机速度、动作惯性和声音相位。后续提示词中的近景、远景、推拉摇移、变焦、切镜和转场指令会被连续性合同覆盖；续段只应描述接下来的动作。

“独立分镜”不读取或保存连续 latent，每段从零生成，可以自由改变景别、机位和镜头运动。需要明显换镜头时应选择该模式，并在后期使用硬切、声音桥或明确转场，而不是伪装成长镜头。

## 生成模式与加速

| 模式 | 主模型 | 允许输入 |
|---|---|---|
| 文生视频 | FL2VA | 仅提示词 |
| 图生视频 | FL2VA | 提示词、必填首帧、可选尾帧 |
| 多参考生视频 | Ref2VA | 提示词；最多 9 图、3 视频、3 音频，总计不超过 12 个 |

“启用 LoRA 加速”关闭时使用 20 步、`res_multistep`、`beta`。开启后才显示 LoRA、步数、强度、采样器和调度器。FL2V v1.1 用于文生/图生，也允许在多参考模式中作为跨模式实验选项；Ref2V v0.1 仍是官方映射的多参考专用版本和导演台默认推荐。实验项会在界面提示，建议实际对比画质、声音和稳定性。多参考官方专用加速需要：

`minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`

放到 ComfyUI 的 `models/loras/`。导演台会检测文件，缺少时阻止错误模式进入队列。

## 尺寸与二次采样

基础生成分别选择 `16:9 / 9:16 / 1:1 / 4:3 / 3:4 / 21:9` 和 `0.2–0.98 MP`。二采不再重复选择宽高比，只选择 `720P / 1080P / 1440P / 2160P` 成片档位并继承基础比例。由于 H3 要求尺寸按 32 对齐，16:9 的 720P 实际为 1280×736，1080P 实际为 1920×1088，界面会直接显示实际值。

二采链路为“一采低分辨率 → 拆分联合 latent → 3D latent 放大 → 条件同步 → Masked Latent 连续性 → 二采细化”。总步数不增加，`细化步数` 表示留给目标分辨率的末段步数。4070 Ti 12GB 默认关闭；建议基础使用 0.6MP，先以 720P 验证；1080P 及以上很可能显存不足。

## 开发与测试

```powershell
D:\Comfy-Desktop\ComfyUI-Installs\comfy-Go\ComfyUI\.venv\Scripts\python.exe -m unittest discover -s .\tests -p test_*.py -v
node --check .\web\director_console.js
```

核心目录：`workflow_builder.py` 负责动态图和连续性合同；`continuity_nodes.py` 负责 latent 保存、读取和同步裁切；`vendor/h3_masked/` 包含 GPL-3.0 的 masked-latent 引擎；`web/` 是导演台前端；`tests/` 是构建回归测试。

## 数据位置

- 项目：`ComfyUI/user/director_console/projects/*.json`
- 上传：`ComfyUI/input/director_console/<project>/<shot>/`
- 视频：`ComfyUI/output/director_console/<project>/`
- 连续 latent：`ComfyUI/output/director_console/context/<project>/`

## HTTP API

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/director_console/config` | 读取模式、依赖和项目列表 |
| `GET` | `/director_console/project/{id}` | 读取项目 |
| `POST` | `/director_console/project` | 验证并保存项目 JSON |
| `POST` | `/director_console/upload` | 上传镜头素材；表单字段为 `file/kind/project_id/shot_id` |
| `POST` | `/director_console/build` | 返回指定镜头的 Comfy API prompt |
| `GET` | `/director_console/outputs` | 返回镜头已生成的视频 |

错误统一返回 `{ "ok": false, "error": "说明" }`。素材路径必须位于 ComfyUI input 目录，后端拒绝绝对路径和 `..`。

## 许可证

本插件按 GPL-3.0 发布。内置连续性引擎来源与固定上游提交见 `THIRD_PARTY_NOTICES.md`，完整许可证见 `LICENSE`。
