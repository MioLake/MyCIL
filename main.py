from utils.data_manager import DataManager

# 初始化管理器：第一个任务10类，之后每次增10类
data_manager = DataManager(dataset_name="cifar100", seed=1997, init_cls=10, increment=10)

# 模拟任务循环
cur_line = 0
for task_id in range(len(data_manager._increments)):
    # 获取当前任务的类别索引
    new_steps = data_manager._increments[task_id]
    task_classes = data_manager.class_order[cur_line : cur_line + new_steps]
    cur_line += new_steps
    
    # 构造数据集
    train_dset = data_manager.get_dataset(task_classes, source='train', mode='train')
    print(f"任务 {task_id}: 类别 {task_classes}, 训练样本数: {len(train_dset)}")