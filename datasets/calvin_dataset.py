"""
CALVIN 数据集加载与多环境数据混合模块

支持:
  - LeRobot 格式的 CALVIN 数据集 (parquet 文件)
  - 单环境加载 (用于 Baseline 训练)
  - 多环境混合加载 (用于 Joint Training 和 Zero-shot 泛化实验)
  - 数据增强 (随机裁剪、颜色抖动等)

数据集结构 (LeRobot 格式):
  datasets/calvin/
  ├── splitA/
  │   ├── meta/
  │   │   ├── info.json          # 数据集元信息 (features, splits 等)
  │   │   ├── episodes.jsonl     # episode 列表 (episode_index, length, tasks 等)
  │   │   ├── modality.json
  │   │   └── tasks.jsonl
  │   └── data/
  │       └── chunk-XXX/
  │           └── episode_XXXXXX.parquet  # 每个 episode 一个 parquet 文件
  ├── splitB/
  ├── splitC/
  └── splitD/
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from pathlib import Path
from typing import List, Optional, Dict, Tuple


# ============================================================
# 图像预处理变换
# ============================================================

def build_transforms(cfg: dict, is_train: bool = True) -> transforms.Compose:
    """
    根据配置构建图像预处理流水线。

    Args:
        cfg:      data 配置字典 (来自 base_act.yaml)
        is_train: 是否为训练模式 (训练时启用数据增强)

    Returns:
        torchvision.transforms.Compose 对象
    """
    image_size = tuple(cfg.get("image_size", [224, 224]))
    mean = cfg.get("image_mean", [0.485, 0.456, 0.406])
    std  = cfg.get("image_std",  [0.229, 0.224, 0.225])

    if is_train and cfg.get("augmentation", True):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transform


# ============================================================
# LeRobot 格式的 CALVIN 数据集加载
# ============================================================

def _decode_image(image_data) -> np.ndarray:
    """
    从 LeRobot 格式的图像数据解码为 numpy 数组。
    
    LeRobot parquet 存储格式: dict with 'bytes' and 'path' keys
    - bytes: 图像字节数据
    - path: 图像文件路径 (通常为 None)
    
    Args:
        image_data: 图像数据 (dict 格式包含 'bytes' 键)
        
    Returns:
        numpy 数组 (H, W, C) uint8
    """
    import io
    from PIL import Image
    
    # LeRobot 格式: dict with 'bytes' key
    if isinstance(image_data, dict):
        raw_bytes = image_data.get('bytes')
        if raw_bytes is None:
            raise ValueError(f"图像数据中 'bytes' 字段为空: {image_data}")
        img = Image.open(io.BytesIO(raw_bytes))
    elif isinstance(image_data, bytes):
        # 原始 bytes 格式
        img = Image.open(io.BytesIO(image_data))
    else:
        raise TypeError(f"不支持的图像数据类型: {type(image_data)}")
    
    return np.array(img)


class CalvinLeRobotEnvDataset(Dataset):
    """
    加载单个 CALVIN 环境 (splitA/B/C/D) 的 LeRobot 格式数据。
    
    每个样本包含:
      - image:        (C, H, W) 机器人视角 RGB 图像
      - state:        (state_dim,) 机器人关节状态
      - actions:      (chunk_size, action_dim) 动作序列 (Action Chunking)
      - env_id:       环境标识符 (用于多环境实验分析)
    """

    def __init__(
        self,
        data_dir: str,
        env_name: str,
        chunk_size: int = 100,
        transform: Optional[transforms.Compose] = None,
        max_episodes: Optional[int] = None,
    ):
        """
        Args:
            data_dir:     该环境数据所在目录 (如 datasets/calvin/splitA)
            env_name:     环境名称 (e.g., "A", "B", "C", "D")
            chunk_size:   Action Chunking 的窗口大小
            transform:    图像预处理变换
            max_episodes: 最多使用的 episode 数量 (None 表示使用全部)
        """
        self.data_dir     = Path(data_dir)
        self.env_name     = env_name
        self.chunk_size   = chunk_size
        self.transform    = transform
        self.max_episodes = max_episodes

        # 加载元数据
        self.meta_info = self._load_meta_info()
        self.episodes = self._load_episode_index()
        
        # 缓存 parquet 数据 (避免重复读取)
        self._parquet_cache = {}

        print(f"[Dataset] 环境 {self.env_name}: 加载了 {len(self.episodes)} 条 episode")

    def _load_meta_info(self) -> dict:
        """加载 info.json 元数据"""
        info_path = self.data_dir / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"未找到元数据文件: {info_path}")
        
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        return info

    def _load_episode_index(self) -> List[Dict]:
        """
        加载 episodes.jsonl 构建 episode 索引列表。
        
        Returns:
            List of dicts: [{"episode_index": int, "length": int, ...}, ...]
        """
        episodes_path = self.data_dir / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(f"未找到 episode 索引文件: {episodes_path}")
        
        episodes = []
        with open(episodes_path, 'r') as f:
            for line in f:
                episode = json.loads(line.strip())
                episodes.append(episode)
        
        total = len(episodes)
        
        # 数据量限制
        if self.max_episodes is not None and self.max_episodes < total:
            rng = np.random.RandomState(seed=42 + ord(self.env_name[0]))
            selected_indices = rng.choice(total, size=self.max_episodes, replace=False)
            selected_indices.sort()
            episodes = [episodes[i] for i in selected_indices]
            print(f"[Dataset] 环境 {self.env_name}: 使用 {self.max_episodes}/{total} 条 episode")
        else:
            print(f"[Dataset] 环境 {self.env_name}: 使用全部 {total} 条 episode")
        
        return episodes

    def _load_episode_parquet(self, episode_index: int) -> pd.DataFrame:
        """
        加载指定 episode 的 parquet 文件。
        
        Args:
            episode_index: episode 索引
            
        Returns:
            pandas DataFrame
        """
        # 构建 parquet 文件路径
        # 格式: data/chunk-{chunk:03d}/episode_{idx:06d}.parquet
        # 需要遍历所有 chunk 目录找到对应的文件
        
        cache_key = episode_index
        if cache_key in self._parquet_cache:
            return self._parquet_cache[cache_key]
        
        # 计算 chunk 索引 (根据 chunks_size，默认为 1000)
        chunk_size_param = self.meta_info.get("chunks_size", 1000)
        chunk_idx = episode_index // chunk_size_param
        
        parquet_filename = f"episode_{episode_index:06d}.parquet"
        parquet_path = self.data_dir / "data" / f"chunk-{chunk_idx:03d}" / parquet_filename
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"未找到 parquet 文件: {parquet_path}")
        
        df = pd.read_parquet(parquet_path)
        self._parquet_cache[cache_key] = df
        return df

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep = self.episodes[idx]
        episode_index = ep["episode_index"]
        episode_length = ep["length"]
        
        # 加载 episode 数据
        df = self._load_episode_parquet(episode_index)
        
        # 随机选取起始帧 (确保有足够的 chunk_size 帧)
        max_start = max(0, episode_length - self.chunk_size)
        if max_start > 0:
            sample_start = np.random.randint(0, max_start + 1)
        else:
            sample_start = 0
        sample_end = min(sample_start + self.chunk_size, episode_length)
        
        # 从 DataFrame 获取数据
        # 注意: DataFrame 的行对应帧，但我们需要确认帧的索引方式
        # 使用 frame_index 或 index 列来定位
        if "frame_index" in df.columns:
            frame_indices = df["frame_index"].values
            start_row = np.searchsorted(frame_indices, frame_indices[sample_start])
            end_row = min(start_row + (sample_end - sample_start), len(df))
        else:
            start_row = sample_start
            end_row = min(sample_end, len(df))
        
        # 获取帧数据
        row_data = df.iloc[start_row:end_row]
        
        # 加载图像 (第一帧作为观测图像)
        image_bytes = row_data["image"].iloc[0]
        image = _decode_image(image_bytes)  # (H, W, C) uint8
        
        # 图像预处理
        if self.transform is not None:
            image = self.transform(image)  # -> (C, H, W) float32
        else:
            # 如果没有 transform，转换为 tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # 获取状态 (使用第一帧的状态)
        state = np.array(row_data["state"].iloc[0], dtype=np.float32)
        
        # 加载动作序列
        actions = []
        for i in range(len(row_data)):
            action = np.array(row_data["actions"].iloc[i], dtype=np.float32)
            actions.append(action)
        
        # 若动作序列不足 chunk_size，用最后一帧动作填充
        while len(actions) < self.chunk_size:
            actions.append(actions[-1])
        
        actions = np.stack(actions, axis=0)  # (chunk_size, action_dim)
        
        return {
            "image":   image,
            "state":   torch.tensor(state, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "env_id":  self.env_name,
        }


# ============================================================
# 多环境混合数据集
# ============================================================

class CalvinDataset(Dataset):
    """
    多环境 CALVIN 数据集，支持均匀采样或按比例混合。
    
    支持 LeRobot 格式的数据集结构:
      data_root/
      ├── splitA/
      │   ├── meta/
      │   │   ├── info.json
      │   │   └── episodes.jsonl
      │   └── data/
      │       └── chunk-XXX/
      │           └── episode_XXXXXX.parquet
      ├── splitB/
      ├── splitC/
      └── splitD/
    
    用法:
        # 单环境 (Baseline)
        dataset = CalvinDataset(data_root, env_names=["A"], ...)

        # 多环境联合训练
        dataset = CalvinDataset(data_root, env_names=["A", "B", "C"], ...)
    """

    def __init__(
        self,
        data_root: str,
        env_names: List[str],
        split: str = "training",
        chunk_size: int = 100,
        transform: Optional[transforms.Compose] = None,
        mixing_strategy: str = "uniform",
        max_episodes_per_env: Optional[int] = None,
    ):
        """
        Args:
            data_root:            CALVIN 数据集根目录 (包含 splitA/B/C/D 的目录)
            env_names:            要加载的环境列表 (e.g., ["A", "B", "C"])
            split:                数据集分割 ("training" 或 "validation")
            chunk_size:           Action Chunking 窗口大小
            transform:            图像预处理变换
            mixing_strategy:      多环境混合策略 ("uniform" 或 "proportional")
            max_episodes_per_env: 每个环境最多使用的 episode 数量 (None = 使用全部)
        """
        self.data_root            = Path(data_root)
        self.env_names            = env_names
        self.split                = split
        self.chunk_size           = chunk_size
        self.transform            = transform
        self.mixing_strategy      = mixing_strategy
        self.max_episodes_per_env = max_episodes_per_env

        # 构建各环境子数据集
        self.env_datasets: Dict[str, CalvinLeRobotEnvDataset] = {}
        for env in env_names:
            env_data_dir = self._get_env_data_dir(env)
            self.env_datasets[env] = CalvinLeRobotEnvDataset(
                data_dir=str(env_data_dir),
                env_name=env,
                chunk_size=chunk_size,
                transform=transform,
                max_episodes=max_episodes_per_env,
            )

        # 构建全局索引映射 (环境名 + 局部索引)
        self.index_map: List[Tuple[str, int]] = self._build_index_map()

    def _get_env_data_dir(self, env_name: str) -> Path:
        """
        根据环境名返回数据目录路径。
        LeRobot 格式: splitA/, splitB/, splitC/, splitD/
        
        搜索路径优先级:
        1. data_root/split{env}/ (LeRobot 标准格式)
        2. 相对于项目根的 datasets/calvin/split{env}/
        3. data_root 目录本身
        """
        # LeRobot 格式: split{env}/
        candidate = self.data_root / f"split{env_name}"
        if candidate.exists():
            return candidate
        
        # 备用: 旧格式 env_{env}/training/ 或 env_{env}/validation/
        candidate = self.data_root / f"env_{env_name}" / self.split
        if candidate.exists():
            return candidate
        
        # 尝试相对于项目根的 datasets/calvin/split{env}/ 路径
        # 当 data_root 是相对路径时，尝试从当前工作目录解析
        project_candidate = Path("datasets/calvin") / f"split{env_name}"
        if project_candidate.exists():
            return project_candidate
        
        # 尝试相对于当前文件的父目录
        import os
        base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        project_candidate = base_dir / "calvin" / f"split{env_name}"
        if project_candidate.exists():
            return project_candidate
        
        project_candidate = base_dir / "datasets" / "calvin" / f"split{env_name}"
        if project_candidate.exists():
            return project_candidate
        
        # 最后返回原始 data_root
        return self.data_root

    def _build_index_map(self) -> List[Tuple[str, int]]:
        """
        构建全局索引到 (环境名, 局部索引) 的映射。

        均匀采样策略: 对每个环境循环采样，使各环境样本数相等。
        比例采样策略: 按各环境实际数据量拼接。
        """
        if self.mixing_strategy == "uniform":
            # 找到最小数据集大小，截断所有环境到相同长度
            min_len = min(len(ds) for ds in self.env_datasets.values())
            index_map = []
            for env in self.env_names:
                for i in range(min_len):
                    index_map.append((env, i))
        else:  # proportional
            index_map = []
            for env, ds in self.env_datasets.items():
                for i in range(len(ds)):
                    index_map.append((env, i))

        return index_map

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        env_name, local_idx = self.index_map[idx]
        return self.env_datasets[env_name][local_idx]


# ============================================================
# DataLoader 构建工厂函数
# ============================================================

def build_dataloader(
    cfg: dict,
    env_names: List[str],
    split: str = "training",
    is_train: bool = True,
) -> DataLoader:
    """
    根据配置字典构建 DataLoader。

    Args:
        cfg:       完整配置字典 (包含 data, model, paths 等子配置)
        env_names: 要加载的环境列表
        split:     数据集分割 ("training" 或 "validation")
        is_train:  是否为训练模式 (影响数据增强和 shuffle)

    Returns:
        torch.utils.data.DataLoader
    """
    data_cfg  = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    paths_cfg = cfg.get("paths", {})

    transform = build_transforms(data_cfg, is_train=is_train)

    # 读取数据量限制参数 (None 表示使用全部数据)
    max_episodes_per_env = data_cfg.get("max_episodes_per_env", None)

    dataset = CalvinDataset(
        data_root            = paths_cfg.get("data_root", "./data/calvin"),
        env_names            = env_names,
        split                = split,
        chunk_size           = model_cfg.get("chunk_size", 100),
        transform            = transform,
        mixing_strategy      = data_cfg.get("mixing_strategy", "uniform"),
        max_episodes_per_env = max_episodes_per_env,
    )

    dataloader = DataLoader(
        dataset,
        batch_size  = cfg.get("training", {}).get("batch_size", 32),
        shuffle     = is_train,
        num_workers = data_cfg.get("num_workers", 4),
        pin_memory  = True,
        drop_last   = is_train,
    )

    return dataloader