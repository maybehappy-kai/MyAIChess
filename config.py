# config.py
args = {
    'C': 2,
    'num_searches': 600,  # 保证AI思考深度，确保CPU有足够的工作量
    'num_iterations': 10,  # 总迭代轮数
    'num_selfPlay_episodes': 320,  # 每轮自我对弈的局数
    'num_cpu_threads': 16,       # <-- 新增：C++工作线程池的大小
    'num_epochs': 2,  # 每轮训练的代数
    'batch_size': 256,  # 增大批处理大小，让GPU一次处理更多数据
    'learning_rate': 0.001,
    'data_max_size': 800000,
    'num_res_blocks': 20,
    'num_hidden': 256,
    'dirichlet_alpha': 0.3,
    'dirichlet_epsilon': 0.25, # <---【新增此行】噪声权重
    # 在config.py的args字典里任意位置加入这行
    'value_loss_weight': 0.25,
    # --- 温度采样参数 ---
        'temperature_start': 1.0,         # <--- 【新增】初始温度
        'temperature_end': 0.1,           # <--- 【新增】最终温度 (接近0表示贪心)
        'temperature_decay_moves': 10,    # <--- 【新增】温度衰减的步数 (前10步使用高温探索)

    # --- 评估体系参数 ---
    'num_eval_games': 320,  # 最终评估时，新旧模型对战的局数
    'board_size': 9,
}