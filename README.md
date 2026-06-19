<!-- ============================================================ -->
<!--  HEADER: theme-aware venom banner (light/dark text variants) -->
<!-- ============================================================ -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://capsule-render.vercel.app/api?type=venom&color=gradient&customColorList=6,11,20&height=220&text=COMFYUI%20MLX&fontSize=70&fontColor=ffffff&animation=twinkling&fontAlignY=41&stroke=7c3aed&strokeWidth=2&desc=The%20Universal%20Apple%20Silicon%20Backend&descAlignY=63&descSize=17&descColor=c4b5fd" />
  <source media="(prefers-color-scheme: light)" srcset="https://capsule-render.vercel.app/api?type=venom&color=gradient&customColorList=6,11,20&height=220&text=COMFYUI%20MLX&fontSize=70&fontColor=ffffff&animation=twinkling&fontAlignY=41&stroke=e9d5ff&strokeWidth=2&desc=The%20Universal%20Apple%20Silicon%20Backend&descAlignY=63&descSize=17&descColor=c4b5fd" />
  <img width="100%" alt="COMFYUI MLX header banner" src="https://capsule-render.vercel.app/api?type=venom&color=gradient&customColorList=6,11,20&height=220&text=COMFYUI%20MLX&fontSize=70&fontColor=ffffff&animation=twinkling&fontAlignY=41&stroke=e9d5ff&strokeWidth=2&desc=The%20Universal%20Apple%20Silicon%20Backend&descAlignY=63&descSize=17&descColor=c4b5fd" />
</picture>

<!-- ============================================================ -->
<!--  BADGES                                                      -->
<!-- ============================================================ -->
<div align="center">

[![Python](https://img.shields.io/pypi/pyversions/mlx-lm.svg?style=for-the-badge&logo=python&logoColor=a78bfa&color=0a0a0a)](https://pypi.org/project/mlx-lm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-0a0a0a.svg?style=for-the-badge&color=0a0a0a)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Ready-0a0a0a?style=for-the-badge&logo=github&logoColor=a78bfa)](https://github.com/comfyanonymous/ComfyUI)
[![Apple Silicon](https://img.shields.io/badge/M--Series-Optimized-0a0a0a?style=for-the-badge&logo=apple&logoColor=a78bfa)](https://developer.apple.com/metal/)

</div>

<br/>

<!-- ============================================================ -->
<!--  TYPING SVG                                                  -->
<!-- ============================================================ -->
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=20&duration=3000&pause=1000&color=A78BFA&center=true&vCenter=true&letterSpacing=2px&width=680&height=50&lines=//+the+one+stop+shop+for+MLX;//+zero+memory+leaks;//+bridging+Apple+Silicon+to+ComfyUI;//+unified+tensors.+unified+cache." />
    <source media="(prefers-color-scheme: light)" srcset="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=20&duration=3000&pause=1000&color=5B21B6&center=true&vCenter=true&letterSpacing=2px&width=680&height=50&lines=//+the+one+stop+shop+for+MLX;//+zero+memory+leaks;//+bridging+Apple+Silicon+to+ComfyUI;//+unified+tensors.+unified+cache." />
    <img alt="Typing status lines" src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=20&duration=3000&pause=1000&color=5B21B6&center=true&vCenter=true&letterSpacing=2px&width=680&height=50&lines=//+the+one+stop+shop+for+MLX;//+zero+memory+leaks;//+bridging+Apple+Silicon+to+ComfyUI;//+unified+tensors.+unified+cache." />
  </picture>
</div>

<!-- ─────────────────────── DIVIDER ─────────────────────────── -->
<img width="100%" alt="" src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=3" />

The **ComfyUI MLX Universal Node Architecture** serves as a foundational runtime substrate designed to natively bridge the Apple Silicon MLX ecosystem with ComfyUI. 

**Our Ultimate Goal:** To be the definitive "One-Stop Shop" for everything MLX on ComfyUI. This architecture is explicitly built as a universal platform to support **all model modalities** (Text, Vision, Audio, Video, VAEs, Diffusion, and Segmentation) through a single, unified backend.

## 📋 Table of Contents
- [⚡ Core Capabilities](#-core-capabilities)
- [🏗️ Architecture Map](#️-architecture-map)
- [🗺️ Development Roadmap](#️-development-roadmap)
- [⚠️ Technical Edge Cases](#️-technical-considerations--known-edge-cases)
- [🤝 Contribution Guidelines](#-contribution-guidelines)

<!-- ─────────────────────── DIVIDER ─────────────────────────── -->
<img width="100%" alt="" src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=3" />

## ⚡ Core Capabilities

| Status | Modality | Tech Stack | Description |
|:---:|---|---|---|
| ![Active](https://img.shields.io/badge/●-ACTIVE-7c3aed?style=flat-square&labelColor=0d0d0d) | **LLMs** | `mlx-lm` | Universal text generation, speculative decoding, thinking tokens. |
| ![Active](https://img.shields.io/badge/●-ACTIVE-7c3aed?style=flat-square&labelColor=0d0d0d) | **VLMs** | `mlx-vlm` | Multimodal visual reasoning and prompt alignment. |
| ![Active](https://img.shields.io/badge/●-ACTIVE-7c3aed?style=flat-square&labelColor=0d0d0d) | **Diffusion** | `DiffusionKit` | Native MLX sampling/decoding for Flux architectures. |
| ![Active](https://img.shields.io/badge/●-ACTIVE-7c3aed?style=flat-square&labelColor=0d0d0d) | **Video** | `mlx_video` | CLI subprocess wrappers supporting Wan2.x, LTX-2, and CogVideoX. |
| ![Active](https://img.shields.io/badge/●-ACTIVE-7c3aed?style=flat-square&labelColor=0d0d0d) | **Segmentation** | `SAM3` | Open-vocabulary semantic segmentation and object detection. |
| ![Active](https://img.shields.io/badge/●-ACTIVE-7c3aed?style=flat-square&labelColor=0d0d0d) | **Adapters** | `registry.py` | Safe, dynamic LoRA fusions injected directly into the unified memory pool. |
| ![Active](https://img.shields.io/badge/●-ACTIVE-7c3aed?style=flat-square&labelColor=0d0d0d) | **System** | `mx.metal` | Explicit cache eviction to protect against Mac swap-memory death. |

<!-- ─────────────────────── DIVIDER ─────────────────────────── -->
<img width="100%" alt="" src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=3" />

## 🏗️ Architecture Map
This repository abandons "mega-nodes" in favor of strict separation of concerns.

```mermaid
graph TD
    subgraph ComfyUI Canvas
        UI_Load[MLX Load Model]
        UI_LoRA[MLX Apply LoRA]
        UI_GenText[MLX Generate Text]
        UI_VLM[MLX Understand Image]
        UI_SAM[MLX Segment Image]
        UI_Vid[MLX Generate Video]
        UI_Diff[MLX Generate Image]
        UI_Sys[MLX Clear Cache]
    end

    subgraph nodes/ [Frontend Nodes]
        UI_Load --> LN[loader_nodes.py]
        UI_LoRA --> LN
        UI_GenText --> GN[generate_nodes.py]
        UI_VLM --> GN
        UI_SAM --> SN[sam_nodes.py]
        UI_Vid --> VN[video_nodes.py]
        UI_Diff --> DN[diffusion_nodes.py]
        UI_Sys --> SysN[system_nodes.py]
    end

    subgraph runtime/ [Execution Substrate]
        LN --> Reg{registry.py}
        GN --> Reg
        SN --> Reg
        
        GN --> Bridge[bridge.py]
        SN --> Bridge
        VN --> Bridge
        DN --> Bridge
        
        Reg --> Data[data_types.py]
    end
    
    subgraph Apple Silicon
        Reg -.-> Unified_Memory[(Unified Memory Cache)]
        Bridge -.-> Tensors(Torch/PIL <-> MLX Arrays)
    end
```

<!-- ─────────────────────── DIVIDER ─────────────────────────── -->
<img width="100%" alt="" src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=3" />

## 🗺️ Development Roadmap

> We need community help to build the remaining modules to achieve the final vision.

**Phase 1: Stabilization (Completed)**
- [x] Unified pipelines for Text, Vision, Samplers, and Video.
- [x] Safe Native LoRA integration.
- [x] `registry.py` tracking and `bridge.py` conversions.

**Phase 2: Expansion (Help Wanted!)**

| Status | Target | Goal |
|:---:|---|---|
| ![WIP](https://img.shields.io/badge/○-WIP-555555?style=flat-square&labelColor=0d0d0d) | **Audio** | Integrate Whisper/Kokoro via `mlx-audio`. |
| ![WIP](https://img.shields.io/badge/○-WIP-555555?style=flat-square&labelColor=0d0d0d) | **SDXL / ControlNet** | Native image pipelines beyond base Flux. |
| ![WIP](https://img.shields.io/badge/○-WIP-555555?style=flat-square&labelColor=0d0d0d) | **VAEs** | Standalone causal video and image VAE encode/decode nodes. |

<!-- ─────────────────────── DIVIDER ─────────────────────────── -->
<img width="100%" alt="" src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=3" />

## ⚠️ Technical Considerations & Known Edge Cases
1. **Python Environment Execution**: Subprocess executions (such as video generation) must utilize `sys.executable` to inherit the parent ComfyUI environment. Direct calls to `python` will break in virtual environments.
2. **Unified Memory Management**: Relying on standard Python garbage collection is insufficient for MLX on Apple Silicon. When dropping heavy model references (e.g., swapping from a 20GB video generator to a 14B VLM), developers must explicitly invoke `mx.metal.clear_cache()` to prevent the OS from hard-swapping to disk.
3. **Lazy Evaluation Boundaries**: MLX employs lazy evaluation. Cross-framework boundaries must be strictly respected. Passing raw `mlx.core.array` objects to native ComfyUI nodes without explicit evaluation or routing through `bridge.py` will result in systemic failures.

<!-- ─────────────────────── DIVIDER ─────────────────────────── -->
<img width="100%" alt="" src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=3" />

## 🤝 Contribution Guidelines
This architecture is designed for the community. When submitting Pull Requests, please adhere to the architectural standards defined in `CONTRIBUTING.md`:
- **No business logic in UI nodes**: The `nodes/` directory should remain thin wrappers.
- **Data contracts**: Always utilize the `@dataclass` structures defined in `runtime/data_types.py`.
- **Tensor bridging**: All matrix and array conversions must be routed through `runtime/bridge.py`.
- **Cache management**: All model loading must be routed through `runtime/registry.py` to prevent OOM errors.

<!-- ============================================================ -->
<!--  FOOTER WAVE                                                  -->
<!-- ============================================================ -->
<img width="100%" alt="" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" />