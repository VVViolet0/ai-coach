# ai-coach

利用大语言模型的理解与生成能力，为非专业训练者生成结构化训练计划，并在训练过程中根据自然语言反馈进行自适应调整。

[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat-square&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/VVViolet0/ai-coach)

## 项目定位

这个项目当前更适合作为一个 AI Coach 原型系统，而不是一个已经完成所有产品能力的健身应用。

它的核心目标有两个：

1. 验证 LLM 是否能够把自然语言训练需求转换成可执行的结构化训练计划。
2. 验证系统是否能够在训练过程中理解用户反馈，并据此调整后续训练节奏、休息时间、组数和动作安排。

## 已实现内容

### 1. 用户意图建模

系统支持将自然语言通过 Ollama 调用本地模型解析为结构化意图，当前主要字段包括：

- `session_goal`: `fat_loss | strength | general_fitness`
- `target_muscles`: `legs | core | chest | back | arms | full_body`
- `duration_minutes`
- `intensity_preference`: `low | moderate | high`
- `experience_level`: `beginner | intermediate | advanced | unknown`
- `equipment_available`: `dumbbell | resistance_band | none`
- `avoid_body_parts`

示例输入：

```text
今天想练腿，大概30分钟，强度不要太高。
```

示例输出：

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

### 2. 受约束的训练计划生成

依据用户意图，系统通过 Ollama 调用本地模型，来生成符合要求的训练计划；

系统使用静态动作库约束 LLM 的输出，避免模型凭空生成不在库中的动作。当前动作规划库位于：

- `libraries/exercise_library.json`

当前动作库包含 15 个常见动作，并统一使用以下肌群标签：

- `legs`
- `core`
- `chest`
- `back`
- `arms`
- `full_body`

训练计划输出为结构化 JSON，格式如下：

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

### 3. 训练执行原型

系统当前实现的是一个基于命令行的训练执行器。它会：

- 按轮次和动作顺序引导训练
- 展示动作名称和文字说明
- 执行组间休息和轮间休息
- 在运行过程中维护会话状态
- 在用户反馈后直接说明后续训练将如何调整
- 记录训练过程日志

当前动作演示库位于：

- `libraries/exercise_demo_library.json`

这份演示库只保留静态演示信息：

- `exercise_name`
- `instructions`

也就是说，当前系统已经完成了“动作知识”和“执行控制”的分层：

- 动作规划库负责训练计划生成
- 动作演示库负责动作展示说明
- 运行时状态负责节奏、速度和调整逻辑

### 4. 自适应反馈机制

训练过程中，用户可以通过自然语言文本反馈表达当前感受，例如：

```text
太累了
```

```text
快一点
```

```text
这个动作不喜欢
```

系统当前已经支持将反馈解析为结构化结果。具体来说，系统会通过 Ollama 调用本地大语言模型，对训练过程中的自然语言反馈进行语义理解，并将其转换为结构化字段。

这一过程对应当前代码中的反馈理解模块，目标不是直接生成训练建议，而是先把用户反馈标准化为可计算的状态表示，供后续规则调整与训练控制使用。

主要字段包括：

- `intent`: `stop | pain | fatigue | pace_up | pace_down | preference_dislike | preference_like | neutral | unknown`
- `fatigue_level`: `low | medium | high`
- `difficulty_level`: `easy | appropriate | hard`
- `preference`: `like | neutral | dislike`
- `confidence`
- `reason`

在此基础上，系统会根据规则对训练做出调整，包括：

- 缩短或延长后续休息时间
- 增减后续动作组数
- 调整节奏提示 `tempo_cue`
- 动态更新运行时 `demo_speed`
- 在可替换的情况下切换当前动作
- 在疼痛或停止意图下触发确认逻辑

### 5. 运行时状态建模

系统当前在 `SessionState` 中维护训练过程的关键运行信息，包括：

- 当前轮次、动作、组次
- 当前阶段 `phase`
- 剩余时间 `seconds_remaining`
- 休息倍率 `rest_multiplier`
- 组数调整量 `set_delta`
- 节奏提示 `tempo_cue`
- 运行时演示速度 `demo_speed`
- 用户条件状态 `UserConditionState`

这里需要特别说明：

- `tempo_cue` 表示当前训练节奏策略，例如 `normal / slower / faster`
- `demo_speed` 是运行时参数，不再属于静态动作库字段
- 当前实现中，`tempo_cue` 会驱动 `demo_speed` 的变化，用于后续接入动画、视频或语音节奏控制

### 6. 日志与测试

系统支持导出训练日志，日志中会保留：

- 初始训练计划
- 最终训练计划
- 用户状态时间线
- 调整记录
- 对话记录
- 反馈理解失败记录
- 事件流

当前仓库中已经包含针对系统主流程和自适应执行器的单元测试。

## 当前项目结构

```text
ai coach/
├─ ai_coach_system.py
├─ intent_modeling.py
├─ workout_planner.py
├─ workout_executor.py
├─ feedback_understanding.py
├─ libraries/
│  ├─ exercise_library.json
│  └─ exercise_demo_library.json
├─ data/
│  └─ ... 运行日志、测试输出与实验数据
└─ test_*.py
```

## 当前局限

虽然核心原型已经跑通，但当前系统仍有明显边界，这些边界也正是后续研究的空间。

### 1. 反馈输入仍然是文本，不是语音

当前系统使用文本输入模拟训练中的实时反馈，还没有接入语音识别或语音交互链路。

### 2. 动作演示仍然是文字说明，不是真实媒体播放

当前执行器展示的是动作名称、说明和运行时 `demo_speed`，但还没有接入：

- 动画播放
- 视频播放
- Avatar 动作控制
- 语音播报

### 3. 适应机制目前以规则驱动为主

反馈理解本身依赖 LLM，但理解结果到训练调整之间，当前仍然主要依赖显式规则，而不是一个更完整的学习型控制器。

### 4. 训练动作库规模仍然较小

当前动作库适合做原型验证，但还不足以支持更丰富的训练目标、设备条件和用户人群差异。

### 5. 尚未完成正式用户实验

当前仓库已经适合做原型演示和实验准备，但还没有形成完整的用户研究流程、问卷设计、实验分组和统计分析结果。

## 未来研究计划

### 1. 从文本反馈扩展到语音反馈

计划在现有文本反馈链路基础上扩展语音输入，使用户能够在训练过程中更自然地表达疲劳、疼痛、偏好和停止意图。

### 2. 从文字演示扩展到多模态演示

计划把当前的动作演示说明扩展为：

- 语音播报
- 动作动画
- 视频示范
- Avatar 演示

当前 `demo_speed` 的运行时设计，就是为这一阶段预留的控制接口。

### 3. 引入更系统的理论解释层

未来可以把当前的反馈理解与调整规则，进一步组织为一个更明确的理论引导模块，例如：

- 基于 SDT 的解释层
- 反馈到动机状态的映射层
- 从动机状态到训练策略的决策层

这部分更适合作为研究论文中的方法深化，而不是仓库当前已经完全落地的事实描述。

### 4. 扩展动作知识库与训练覆盖范围

后续可以增加：

- 更多动作与变式
- 更多设备类型
- 更细粒度的动作属性
- 更丰富的训练目标
- 人群差异化适配

### 5. 设计正式用户实验

后续研究可以围绕以下问题设计实验：

- AI 生成训练计划的合理性
- 自适应反馈机制对训练体验的影响
- 自适应调整对依从性和完成率的影响
- 用户对系统可解释性和信任感的评价

## 如何运行

### 运行完整流程

```bash
python ai_coach_system.py --request "今天想练全身，30分钟，强度中等"
```

### 运行自适应训练演示

```bash
python run_adaptive_demo.py
```

### 示例输出

```text
---------------------------
Exercise: Bodyweight Squat
Instructions:
 - Stand with feet shoulder-width apart
 - Push hips back
 - Lower until thighs parallel
 - Drive through heels to stand
---------------------------

Set 1/3 start (demo_speed: 1.00)
Set finished.
```

当用户输入类似 `i wanna rest`、`too hard` 或 `太累了` 的反馈时，系统除了记录内部调整日志，还会直接给出面向用户的说明，例如：

```text
[Coach] Thanks, I captured your feedback and updated upcoming blocks.
[Coach] I reduced upcoming sets by 1, increased upcoming rest by 20%, slowed the demo pace.
```

### 运行测试

```bash
python -m unittest test_ai_coach_system.py test_workout_executor.py
```

## 面向论文的表述建议

如果将本项目用于论文或开题报告，建议采用“已实现内容 + 未来研究计划”的写法：

- 已实现内容：意图建模、受约束计划生成、命令行训练执行、反馈理解、自适应规则调整、日志记录与测试
- 未来研究计划：语音交互、多模态动作演示、理论解释层、扩展动作库、正式用户实验

这样既能准确反映当前项目状态，也能保留研究工作的延展空间。
