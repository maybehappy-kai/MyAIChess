# POPUCOM Board Game: AlphaZero-based AI Research

本项目实现了一个基于 **AlphaZero** 范式的强化学习 AI，专用于策略游戏 **POPUCOM Board Game**。项目采用 **Python (训练/控制)** + **C++ (高性能 MCTS/推理)** 的混合架构，支持单机单卡训练与高效的双卡分布式训练。

## 🎮 游戏规则 (Game Rules)

POPUCOM 是一款基于 **9x9** 棋盘的策略游戏，核心机制为 **"三连消除 + 领地扩张"**。

1.  **基本设定**：黑方先手，白方后手。
2.  **胜利条件**：游戏结束时（默认 50 回合或无子可落），**领地 (Territory)** 多者获胜。注意：棋子本身不计分，只有被染色的领地计分。
3.  **核心机制 - Combo & Beam**：
    * **三连消除**：当一方在横、竖、斜任意方向连成 **3颗** 棋子时，这 3 颗棋子会立即**爆炸消失**。
    * **领地光束**：爆炸会沿着连线方向发射“光束”。光束经过的所有空地都会被标记为当前玩家的**领地**。
    * **阻挡与覆盖**：光束会一直延伸，直到遇到**棋盘边缘**或**对手的棋子**。如果光束经过对手的领地，会将其**覆盖**为己方领地。
4.  **落子限制**：只能在空地落子，且**不能直接下在对手的领地内**。

---

## 🚀 环境构建 (Build)

本项目针对 **Windows (MSVC)** 环境进行了编译优化。项目依赖 PyTorch 和 C++ 编译环境（支持 C++17）。

1.  **安装 Python 依赖**：
    ```bash 
    pip install -r requirements.txt

2.  **编译 C++ 扩展**：
    ```bash
    python setup.py build_ext --inplace

---

## ⚡ 运行操作流程 (Workflows)

本项目设计了两种工作流。**强烈推荐使用双卡模式**以获得最佳效率和体验。

### A. 双卡/分布式模式 (推荐)
*适用场景：拥有双 GPU 或希望数据生成与训练解耦。支持专家数据热重载。*

该模式采用 **生产者-消费者** 模型，通过文件系统异步通信。

**操作步骤**：

1.  **启动模型训练 Worker (GPU 1)**：
    **务必首先启动此脚本**。如果是首次运行，它会初始化并保存第一个随机权重的模型 (`model_0.pt`)。如果没有这一步，自对弈进程将因找不到模型文件而无法启动。
    ```bash
    python worker_train.py

2.  **启动数据生成 Worker (GPU 0)**：
    等待 `worker_train.py` 初始化模型完毕后启动。此进程会自动加载最新模型，持续进行高强度 MCTS 自对弈，并将数据写入 `data_buffer/`。
    ```bash
    python worker_selfplay.py

3.  **启动人机对战 GUI (可选)**：
    随时启动 GUI 与 AI 对战，并录入专家数据。
    ```bash
    python play_pixel_art.py

**✨ 关于专家数据注入 (Expert Data Injection)**：
在双卡模式下，`worker_train.py` 会实时监控 `human_games.pkl`。
* **操作**：你只需运行 `play_pixel_art.py` 并在游戏结束时保存数据。
* **结果**：训练进程会自动检测到文件变更并**热重载**新数据，**无需停止或重启**任何训练脚本。实现了真正的 "Human-in-the-loop" 训练。

---

### B. 单卡/单进程模式
*适用场景：单 GPU 环境或调试。*

该模式为串行流程：`自对弈 -> 训练 -> 评估` 循环执行。

**操作步骤**：

1.  **启动 Coach**：
    ```bash
    python coach.py

**⚠️ 关于专家数据注入的限制**：
单卡模式仅在**启动时**加载一次 `human_games.pkl`。
* 如果你在 `coach.py` 运行期间通过 GUI 生成了新数据，这些数据**不会**被当前进程读取。
* **必须停止并重启** `coach.py`，新数据才会被注入训练池。

---

## 📊 评估与基线 (Evaluation)

项目包含一个基于 **Alpha-Beta 剪枝** 的传统 AI 作为评估基线，用于验证 MCTS-AlphaZero 模型的有效性。

**运行评估**：
```text
python arena_ab_vs_mcts.py --games 100 --workers 8
```

* **Alpha-Beta AI**：采用综合评估函数（领地差 + 活二棋型），搜索深度可配置（默认 depth=3）。
* **MCTS AI**：加载最新的 `model_xxx.pt`。
* **输出**：脚本将利用多进程进行对战，并输出 MCTS AI 对阵传统算法的胜率。这是衡量模型是否掌握游戏策略（如利用三连机制）的关键指标。

---

## 📂 目录结构说明

* `worker_train.py`: **[双卡]** 训练与模型服务器（支持热重载）。**需最先启动**。
* `worker_selfplay.py`: **[双卡]** 自对弈数据生成器。
* `coach.py`: **[单卡]** 统筹训练逻辑。
* `play_pixel_art.py`: 游戏 GUI 及人机对战接口。
* `arena_ab_vs_mcts.py`: 强度评估工具。
* `cpp_src/`: C++ MCTS 引擎源码（核心性能所在）。
    * `Gomoku.cpp`: 游戏核心逻辑实现（Combo/Beam）。
    * `SelfPlayManager.cpp`: 多线程自对弈管理。
* `neural_net.py`: PyTorch 神经网络定义（SE-ResNet 架构）。