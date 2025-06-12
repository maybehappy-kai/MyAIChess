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
    # 在config.py的args字典里任意位置加入这行
    'value_loss_weight': 0.25,
    # --- 评估体系参数 ---
    'num_eval_games': 50,  # 最终评估时，新旧模型对战的局数
    'board_size': 9,
}
