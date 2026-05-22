# AI Coach

AI Coach 是一个本地运行的自适应训练原型系统。它把用户的自然语言训练需求解析为结构化意图，基于动作库生成可执行训练计划，并在训练过程中根据文字或语音反馈动态调整节奏、休息、组数和动作安排。

本项目目前定位为研究与演示用原型，重点验证生成式模型在个性化训练计划生成、训练过程反馈理解和实时自适应调整中的可行性。

[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat-square&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/VVViolet0/ai-coach)

## 当前能力

### 1. 训练意图建模

`intent_modeling.py` 使用本地 Ollama 模型解析用户输入，提取训练目标、目标肌群、训练时长、强度偏好、经验水平、可用器械和需要避开的身体部位等信息。

示例输入：

```text
今天想练腿，大概 30 分钟，强度不要太高。
```

示例结构化结果：

```json
{
  "session_goal": "general_fitness",
  "target_muscles": ["legs"],
  "duration_minutes": 30,
  "intensity_preference": "low",
  "experience_level": "unknown",
  "equipment_available": ["none"],
  "avoid_body_parts": []
}
```

### 2. 受动作库约束的训练计划生成

`workout_planner.py` 使用 `libraries/exercise_library.json` 作为动作库，只允许模型选择库内动作，并对训练轮次、组数、休息时间和动作时长进行边界校验。

训练计划示例：

```json
{
  "workout_plan": {
    "rounds": 2,
    "rest_between_rounds": 30,
    "exercises": [
      {
        "exercise": "push_up",
        "avg_set_time": 40,
        "total_sets": 3,
        "rest_seconds": 30
      }
    ]
  }
}
```

### 3. 自适应训练执行

`workout_executor.py` 负责按轮次、动作和组数执行训练，并维护运行时状态，包括：

- 当前轮次、动作、组次和阶段
- 倒计时与休息节奏
- 后续组数调整
- 节奏提示 `tempo_cue`
- 用户疲劳、疼痛、偏好等状态
- 对话记录、调整记录、失败解析记录和事件流

训练反馈理解由 `feedback_understanding.py` 负责。系统会把自然语言反馈解析为 `stop`、`pain`、`fatigue`、`pace_up`、`pace_down`、`preference_dislike` 等意图，再由执行器根据规则调整后续训练。

例如：

- 用户说“太累了”时，系统可以增加休息或减少后续组数。
- 用户说“膝盖疼”时，系统会触发安全确认并避免继续高风险动作。
- 用户说“这个动作不喜欢”时，系统可以跳过或替换后续动作。

### 4. 浏览器语音 AI Coach

`voice_demo/` 提供本地浏览器语音交互原型。

语音链路：

```text
浏览器麦克风
-> WebSocket PCM 音频流
-> Python VAD 分段
-> 本地 faster-whisper ASR
-> AI Coach 意图建模 / 训练反馈循环
-> 浏览器 speechSynthesis 语音播报
```

包含两个运行入口：

- `voice_demo/server.py`：独立语音链路 Demo，用于测试麦克风、VAD、ASR 和浏览器 TTS。
- `voice_demo/coach_server.py`：浏览器语音版 AI Coach，把首句语音路由到训练需求解析和计划生成，后续语音路由到训练反馈循环。

浏览器页面会展示当前动作、轮次、组次、训练阶段、倒计时、节奏提示、完整计划、动作说明、最近语音反馈和实时调整记录。

## 项目结构

```text
ai coach/
├── ai_coach_system.py              # 命令行端到端入口
├── intent_modeling.py              # 用户训练意图建模
├── workout_planner.py              # 受动作库约束的训练计划生成
├── workout_executor.py             # 自适应训练执行器
├── feedback_understanding.py       # 训练反馈语义理解
├── requirements.txt                # 项目整体依赖
├── libraries/
│   ├── exercise_library.json       # 训练计划生成使用的动作库
│   └── exercise_demo_library.json  # 动作文字说明库
├── data/                           # 运行日志、实验输出和临时结果
├── voice_demo/
│   ├── coach_server.py             # 浏览器语音 AI Coach
│   ├── server.py                   # 独立语音链路 Demo
│   ├── segmenter.py                # VAD 音频分段
│   ├── static/                     # 浏览器页面与音频脚本
│   ├── models/                     # 本地 faster-whisper 模型目录
│   └── requirements.txt            # 语音模块依赖
└── test_*.py                       # 单元测试与集成测试
```

## 环境要求

推荐环境：

- Python 3.10 到 3.12
- Ollama 本地服务
- 默认 LLM：`frob/qwen3.5-instruct:4b`
- 浏览器语音 Demo 需要可用麦克风和现代浏览器
- 如使用本地语音识别，需要准备 faster-whisper 模型

安装项目依赖：

```powershell
python -m pip install -r requirements.txt
```

准备本地 Ollama 模型：

```powershell
ollama pull frob/qwen3.5-instruct:4b
```

如果只运行语音模块，也可以单独安装：

```powershell
python -m pip install -r voice_demo/requirements.txt
```

## 如何运行

### 命令行完整流程

```powershell
python ai_coach_system.py --request "今天想练全身，30分钟，强度中等"
```

默认会把意图、训练计划和训练日志写入 `data/`。这些运行产物不会进入 Git 仓库。

### 独立语音链路 Demo

```powershell
python voice_demo/server.py --host 127.0.0.1 --port 8008 --model base --device cpu --compute-type int8
```

打开：

```text
http://127.0.0.1:8008
```

如果只想测试浏览器麦克风和 VAD，不加载 Whisper：

```powershell
python voice_demo/server.py --host 127.0.0.1 --port 8008 --asr-backend mock
```

### 浏览器语音 AI Coach

```powershell
python voice_demo/coach_server.py --host 127.0.0.1 --port 8008 --model "voice_demo/models/faster-whisper-base" --device cpu --compute-type int8 --cpu-threads 1 --num-workers 1 --language zh
```

打开：

```text
http://127.0.0.1:8008
```

首句有效语音会用于生成训练计划，之后的语音会作为训练过程反馈处理。

## 本地 Whisper 模型

如果无法在线下载 faster-whisper 模型，可以先在网络可用环境下下载 CTranslate2 / faster-whisper 格式模型，再复制到：

```text
voice_demo/models/faster-whisper-base/
```

然后通过 `--model` 指向该本地目录。

注意：这里需要的是 CTranslate2 / faster-whisper 格式模型，例如 `Systran/faster-whisper-base`，不是原始 OpenAI Whisper PyTorch checkpoint。

## 运行测试

推荐运行：

```powershell
python -m pytest test_voice_coach_io.py test_workout_executor.py test_workout_runtime_events.py
```

也可以运行全部测试：

```powershell
python -m pytest
```

部分端到端能力依赖本地 Ollama 或语音依赖；单元测试会尽量使用 mock 隔离外部服务。

## Git 同步说明

以下内容不会随 Git 同步，需要在新机器本地重新准备：

- `.env`
- `data/` 下的运行日志、实验输出和临时结果
- `voice_demo/models/` 下的本地 Whisper 模型文件
- Python 虚拟环境、缓存和 IDE 配置
