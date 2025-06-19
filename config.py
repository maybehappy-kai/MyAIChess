# file: config.py

args = {
    'C': 2,
    'num_searches': 800,
    'num_iterations': 1,
    'num_selfPlay_episodes': 100,
    'num_cpu_threads': 18,
    'num_epochs': 4,
    'batch_size': 128,
    'learning_rate': 0.001,
    'data_max_size': 200000,
    'num_res_blocks': 9,
    'num_hidden': 128,
    'dirichlet_alpha': 0.3,
    'dirichlet_epsilon': 0.25,
    'value_loss_weight': 0.25,

    # 温度采样参数
    'temperature_start': 1.0,
    'temperature_end': 0.1,
    'temperature_decay_moves': 10,

    # 评估体系参数
    'num_eval_games': 20,
    'board_size': 9,
    'num_rounds': 25, # 每局游戏的回合数，总步数为 num_rounds * 2
    'mcts_batch_size': 64,

    # =======================================================
    # --- 请确保您的配置包含以下所有参数 ---

    # 数据筛选开关
    'filter_zero_policy_data': True,  # True 表示开启筛选，False 表示关闭

    # 开局偏置参数
    'enable_opening_bias': True,  # True 表示启用开局偏置
    'opening_bias_strength': 0.1,  # 偏置的总体强度

    # 活二威胁检测参数
    'enable_threat_detection': False,  # [之前缺失] True 表示启用威胁检测
    'threat_detection_bonus': 10.0,  # 威胁点的额外奖励值

    # 领地价值启发参数
    'enable_territory_heuristic': True,  # [之前缺失] True 表示启用领地启发
    'territory_heuristic_weight': 0.3,  # 领地启发所占的权重
    # =======================================================
    'history_steps': 3, # 记录T-1, T-2, T-3三步历史
}