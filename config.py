# file: config.py (最终完整版 - 包含所有必需的参数)

args = {
    'C': 2,
    'num_searches': 800,
    'num_iterations': 1,
    'num_selfPlay_episodes': 100,
    'num_cpu_threads': 18,
    'training_steps_per_iteration': 1000,
    'batch_size': 128,
    'data_max_size': 200000,
    'num_res_blocks': 9,
    'num_hidden': 128,
    'dirichlet_alpha': 0.3,
    'dirichlet_epsilon': 0.25,
    'value_loss_weight': 1,

    # --- 训练与评估的核心超参数 ---
    'learning_rate': 0.001,
    'weight_decay': 0.0001,  # <--- 修正: 加上这一行
    'promotion_win_rate': 0.55,  # <--- 修正: 加上这一行
    'failed_selfplay_ratio': 0.1,  # <--- 修正: 加上这一行

    # --- 模仿学习相关 ---
    'expert_data_ratio': 0.25,
    'expert_data_max_size': 2000,
    'label_smoothing_alpha': 0.03,

    # --- 其他游戏和MCTS参数 ---
    'elo_k_factor': 32,
    'num_candidates_to_train': 3,
    'temperature_start': 1.0,
    'temperature_end': 0.1,
    'temperature_decay_moves': 10,
    'num_eval_games': 20,
    'board_size': 9,
    'num_rounds': 25,
    'mcts_batch_size': 64,
    'filter_zero_policy_data': True,
    'enable_opening_bias': True,
    'opening_bias_strength': 0.1,
    'enable_ineffective_connection_penalty': True,
    'ineffective_connection_penalty_factor': 0.1,
    'enable_territory_heuristic': True,
    'territory_heuristic_weight': 0.6,
    'enable_territory_penalty': True,
    'territory_penalty_strength': 0.5,
    'history_steps': 3,
}