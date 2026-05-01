# ai-coach

AI Coach 是一个本地运行的自适应训练原型系统。它把用户的自然语言训练需求解析成结构化意图，生成受动作库约束的训练计划，并在训练过程中根据文本或语音反馈调整节奏、休息、组数和动作安排。

[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat-square&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/VVViolet0/ai-coach)

## 项目定位

这个仓库目前适合作为研究与演示用的 AI Coach 原型，而不是完整商业健身应用。它重点验证三件事：

1. 本地大语言模型能否把自然语言训练需求转换为结构化意图。
2. 系统能否基于动作库生成可执行、可校验的训练计划。
3. 训练过程中能否理解用户反馈，并以规则优先的方式做出自适应调整。

## 当前能力

### 意图建模

`intent_modeling.py` 通过 Ollama 本地模型把用户输入解析为结构化字段，包括训练目标、目标肌群、训练时长、强度偏好、经验水平、可用器械和需要避开的身体部位。

示例：

```text
今天想练腿，大概30分钟，强度不要太高。
```

输出会被标准化为类似结构：

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

### 受约束训练计划生成

`workout_planner.py` 使用 `data/exercise_library.json` 作为静态动作库，只允许模型选择库内动作，并对轮次、休息时间、组数和动作时长进行边界校验。

训练计划格式：

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

### 自适应训练执行

`workout_executor.py` 负责按轮次、动作和组数执行训练，并维护运行时状态：

- 当前轮次、动作、组次和阶段
- 倒计时与休息倍率
- 后续组数调整
- `tempo_cue`
- 用户疲劳、疼痛、偏好等条件状态
- 对话、调整记录、失败解析记录和事件流

反馈理解使用 `feedback_understanding.py`，会把自然语言反馈解析为 `stop`、`pain`、`fatigue`、`pace_up`、`pace_down`、`preference_dislike` 等意图。执行器再根据解析结果调整后续训练，例如增加休息、减少组数、放慢节奏、替换或跳过动作，以及在疼痛和停止意图下触发确认。

### 浏览器语音 Demo

`voice_demo/` 是新增的本地语音交互原型，包含两种运行方式：

- `voice_demo/server.py`：独立语音链路测试，验证浏览器麦克风、WebSocket PCM 流、VAD 分段、本地 faster-whisper ASR 和浏览器 TTS。
- `voice_demo/coach_server.py`：浏览器版 AI Coach，把首句语音路由到意图建模和计划生成，后续语音路由到训练反馈循环。

语音 Demo 使用 FastAPI 和 WebSocket，TTS 使用浏览器 `speechSynthesis`。本地 Whisper 模型文件不进入 Git 仓库。

## 项目结构

```text
ai coach/
├─ ai_coach_system.py              # 端到端流程入口
├─ intent_modeling.py              # 用户意图建模
├─ workout_planner.py              # 受动作库约束的计划生成
├─ workout_executor.py             # 自适应训练执行器
├─ feedback_understanding.py       # 训练反馈语义理解
├─ requirements.txt                # 项目整体 Python 依赖入口
├─ libraries/
│  ├─ exercise_library.json        # 训练计划生成使用的动作库
│  └─ exercise_demo_library.json   # 动作文字说明库
├─ data/                           # 运行日志、实验输出和临时结果
├─ voice_demo/
│  ├─ coach_server.py              # 浏览器语音 AI Coach
│  ├─ server.py                    # 独立语音链路 Demo
│  ├─ segmenter.py                 # VAD 分段
│  ├─ static/                      # 浏览器端页面与音频脚本
│  ├─ models/                      # 本地 faster-whisper 模型目录
│  └─ requirements.txt             # 语音模块依赖
└─ test_*.py
```

## 环境要求

推荐环境：

- Python 3.10 到 3.12，当前本地验证环境为 Python 3.12.4
- Ollama 本地服务
- 默认 LLM：`frob/qwen3.5-instruct:4b`
- 浏览器语音 Demo 需要可用麦克风和现代浏览器

安装项目整体依赖：

```powershell
python -m pip install -r requirements.txt
```

如果只安装语音 Demo 依赖，也可以单独运行：

```powershell
python -m pip install -r voice_demo/requirements.txt
```

准备本地 Ollama 模型：

```powershell
ollama pull frob/qwen3.5-instruct:4b
```

如果无法在线下载 faster-whisper 模型，可以先在网络可用环境下载 CTranslate2/faster-whisper 格式模型，再复制到：

```text
voice_demo/models/faster-whisper-base/
```

## 如何运行

### 命令行完整流程

```powershell
python ai_coach_system.py --request "今天想练全身，30分钟，强度中等"
```

默认会把意图、训练计划和训练日志写入 `data/`。这些运行产物已被 `.gitignore` 忽略。

### 独立语音链路测试

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

页面会展示当前动作、轮次/组次、训练阶段、倒计时、节奏提示、训练计划、动作说明、最近语音反馈和实时调整。

## 运行测试

```powershell
python -m pytest test_workout_runtime_events.py test_voice_demo_segmenter.py test_voice_coach_io.py
```

部分端到端能力依赖本机 Ollama 或语音依赖；单元测试会尽量使用 mock 隔离外部服务。

## Git 与迁移说明

仓库已配置远端：

```text
https://github.com/VVViolet0/ai-coach.git
```

迁移到另一台电脑时，推荐直接 clone：

```powershell
git clone https://github.com/VVViolet0/ai-coach.git
cd ai-coach
python -m pip install -r requirements.txt
ollama pull frob/qwen3.5-instruct:4b
```

以下内容不会随 Git 同步，需要在新机器本地重新准备：

- `.env`
- `data/` 下的运行日志、实验输出和临时测试输出
- `voice_demo/models/` 下的本地 Whisper 模型文件
- Python 虚拟环境、缓存和 IDE 配置

## 当前局限

- 动作库规模仍较小，适合原型验证，还不足以覆盖完整健身场景。
- 反馈理解依赖 LLM，但训练调整目前仍以显式规则为主。
- 浏览器语音链路已经接入原型，但还不是正式移动端或可穿戴设备体验。
- 动作演示仍以文字说明为主，尚未接入真实视频、动画或 Avatar。
- 尚未形成完整用户实验流程、问卷、分组和统计分析。

## 后续计划

- 扩展动作库、设备类型、训练目标和人群差异化规则。
- 将语音反馈链路进一步打磨成更稳定的训练交互。
- 接入多模态动作演示，例如视频、动画、Avatar 或节奏化语音提示。
- 建立更清晰的理论解释层，把反馈、动机状态和训练策略连接起来。
- 设计正式用户实验，评估计划合理性、自适应体验、完成率和用户信任。
