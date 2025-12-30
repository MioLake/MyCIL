import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets

class DummyDataset(Dataset):
    """
    一个通用的 PyTorch Dataset 包装器，用于将原始图像数据（Numpy 数组或文件路径）适配为标准 Dataset 接口。

    支持两种数据格式：
        - 内存中的 NumPy 图像数组（适用于 CIFAR 等小型数据集）
        - 图像文件路径列表（适用于 ImageNet 等大型数据集，避免内存溢出）
    
    返回格式：(样本索引, 转换后的图像张量, 标签)
    """
    def __init__(self, images, labels, trsf, use_path=False):
        """
        初始化包装器。

        Args:
            images (np.ndarray or List[str]): 
                - 若 use_path=False: 形状为 (N, H, W, C) 的 NumPy 图像数组。
                - 若 use_path=True: 长度为 N 的图像文件路径字符串列表。
            labels (np.ndarray): 长度为 N 的标签数组，类型通常为 int。
            trsf (torchvision.transforms.Compose): 图像预处理/增强变换流水线。
            use_path (bool): 是否以路径方式加载图像（默认 False）。
        """
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        """返回数据集中样本的总数"""
        return len(self.images)
    
    def __getitem__(self, index):
        """
        根据索引获取单个样本。

        Args:
            index (int): 样本索引。

        Returns:
            tuple: (index, transformed_image_tensor, label)
                - index: 原始数据中的索引（常用于持续学习中追踪样本来源）
                - transformed_image_tensor: 经过 trsf 处理后的 torch.Tensor，形状 [C, H, W]
                - label: 对应的类别标签（int）
        """
        if self.use_path:
            # 从磁盘动态加载图像（适用于大数据集）
            image = image.open(self.images[index]).convert('RGB')
        else:
            # 从内存中的 NumPy 数组构造 PIL 图像
            image = Image.fromarray(self.images[index])
        
        # 应用图像变换（如归一化、裁剪、翻转等）
        image = self.trsf(image)
        label = self.labels[index]
        return index, image, label
    
class DataManager(object):
    """
    持续学习（Continual Learning）中的数据管理器。
    
    功能包括：
        - 加载基础数据集（目前支持 CIFAR-100）
        - 打乱类别顺序（模拟任务增量学习）
        - 按任务划分数据（每次只暴露部分类别）
        - 支持从经验回放缓冲区（Buffer）中合并旧类样本
    """
    def __init__(self, dataset_name, seed, init_cls, increment):
        """
        初始化数据管理器。

        Args:
            dataset_name (str): 数据集名称，当前仅支持 "cifar100"。
            seed (int): 随机种子，用于打乱类别顺序，保证实验可复现。
            init_cls (int): 初始任务包含的类别数量。
            increment (int): 后续每个任务新增的类别数量。
        """
        self.dataset_name = dataset_name
        self._setup_data(dataset_name)  # 加载原始数据

        # 固定随机种子，生成一致的类别顺序
        np.random.seed(seed)
        self.class_order = np.arange(self.common_num_classes)
        np.random.shuffle(self.class_order) # 打乱类别 ID 顺序

        # 构建任务划分：例如 init_cls=10, increment=10 → [10, 10, 10, ..., 10]（共10个任务）
        self._increments = [init_cls]
        while sum(self._increments) < self.common_num_classes:
            self._increments.append(increment)
    
    def _setup_data(self, dataset_name):
        """
        加载指定数据集的训练集和测试集，并提取图像与标签。

        目前仅实现 CIFAR-100。CIFAR-100 的统计量（均值、标准差）已硬编码用于归一化。
        """
        if dataset_name == 'cifar100':
            # 自动下载并加载 CIFAR-100
            train_dset = datasets.CIFAR100('./data', train=True, download=True)
            test_dset = datasets.CIFAR100('./data', train=False, download=True)

            # 提取 NumPy 格式的图像数据和标签
            self.train_data = train_dset.data
            self.train_targets = np.array(train_dset.targets)
            self.test_data = test_dset.data
            self.test_targets = np.array(test_dset.targets)

            self.common_num_classes = 100
        else:
            raise NotImplementedError('暂时只支持 CIFAR-100')
    
    def get_dataset(self, indices, source='train', mode='train', appendent=None):
        """
        根据给定的类别索引，构建对应子集的 Dataset 对象。

        Args:
            indices (List[int]): 当前任务涉及的类别标签列表（按原始标签空间）。
            source (str): 'train' 或 'test'，选择使用训练集还是测试集。
            mode (str): 'train' 或 'test'，决定使用哪种数据增强策略。
            appendent (Optional[Tuple[np.ndarray, np.ndarray]]): 
                来自经验回放缓冲区（Buffer）的旧类样本，格式为 (images, labels)，
                将与当前任务新类样本拼接，用于缓解灾难性遗忘。

        Returns:
            DummyDataset: 可被 DataLoader 加载的 PyTorch Dataset 对象。
        """
        # 选择数据源
        if source == 'train':
            x, y = self.train_data, self.train_targets
        else:
            x, y = self.test_data, self.test_targets
        
        # 使用 np.isin 筛选出属于当前任务类别的样本
        data_mask = np.isin(y, indices)
        selected_x, selected_y = x[data_mask], y[data_mask]

        # 如果有来自 Buffer 的旧样本，则拼接到当前任务数据后
        if appendent is not None:
            appendent_x, appendent_y = appendent
            selected_x = np.concatenate((selected_x, appendent_x), axis=0)
            selected_y = np.concatenate((selected_y, appendent_y), axis=0)

        # 获取对应模式（训练/测试）的图像变换
        trsf = self._get_transform(mode)

        # 返回包装好的 Dataset
        return DummyDataset(selected_x, selected_y, trsf)

    def _get_transform(self, mode):
        """
        返回对应模式下的图像预处理流水线。

        CIFAR-100 的标准化参数（均值和标准差）来自官方统计：
            mean = [0.5071, 0.4867, 0.4408]
            std  = [0.2675, 0.2565, 0.2761]

        Args:
            mode (str): 'train' 启用数据增强；'test' 仅做标准化。

        Returns:
            torchvision.transforms.Compose: 变换流水线。
        """
        if mode == 'train':
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),      # 随机裁剪 + 填充
                transforms.RandomHorizontalFlip(),         # 随机水平翻转
                transforms.ToTensor(),                     # 转为 Tensor 并归一化到 [0,1]
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761)
                )
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761)
                )
            ])