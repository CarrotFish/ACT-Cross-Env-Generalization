"""
日志记录模块 - 封装 WandB / SwanLab 日志后端

提供统一的日志接口，支持：
  - 标量指标记录 (Loss, Success Rate 等)
  - 超参数配置记录
  - 模型权重上传
  - 训练曲线可视化

用法:
    logger = build_logger(cfg)
    logger.log({"train/loss": 0.5, "train/action_loss": 0.3}, step=100)
    logger.finish()
"""

import os
from typing import Dict, Any, Optional


# ============================================================
# 基础 Logger 接口
# ============================================================

class Logger:
    """
    日志记录器基类，定义统一接口。
    具体实现由 WandBLogger 或 SwanLabLogger 提供。
    """

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """记录标量指标"""
        raise NotImplementedError

    def log_config(self, config: Dict[str, Any]):
        """记录超参数配置"""
        raise NotImplementedError

    def log_summary(self, summary: Dict[str, Any]):
        """记录实验总结 (最终结果)"""
        raise NotImplementedError

    def finish(self):
        """结束日志记录会话"""
        raise NotImplementedError


# ============================================================
# WandB Logger
# ============================================================

class WandBLogger(Logger):
    """
    基于 Weights & Biases (WandB) 的日志记录器。

    需要提前安装: pip install wandb
    使用前需登录: wandb login
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        config: Dict[str, Any],
        tags: Optional[list] = None,
        log_dir: str = "./logs",
    ):
        """
        Args:
            project:  WandB 项目名称
            run_name: 本次运行的名称
            config:   超参数配置字典
            tags:     运行标签列表
            log_dir:  本地日志保存目录
        """
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "WandB 未安装，请运行: pip install wandb\n"
                "或切换到 SwanLab: 在配置中设置 logging.backend: swanlab"
            )

        os.makedirs(log_dir, exist_ok=True)

        # 获取 entity (用户名或组织名)
        # 1. 优先使用配置中指定的 entity
        # 2. 如果未指定，尝试从 WandB API 获取当前登录用户的默认实体
        # 3. 如果都失败，不传递 entity 让 WandB 自行处理
        entity_name = config.get("logging", {}).get("entity", None)

        if not entity_name:
            try:
                api = wandb.Api()
                entity_name = api.default_entity
                print(f"[WandB] Using default entity from API: {entity_name}")
            except Exception as e:
                print(f"[WandB] Failed to get default entity from API: {e}")
                entity_name = None

        if entity_name:
            print(f"[WandB] Using entity: {entity_name}")
        else:
            print("[WandB] Entity not set, using viewer's default")

        init_kwargs = dict(
            project = project,
            name    = run_name,
            config  = config,
            tags    = tags or [],
            dir     = log_dir,
        )
        if entity_name:
            init_kwargs["entity"] = entity_name

        self.run = wandb.init(**init_kwargs)
        print(f"[WandB] 实验已启动: {self.run.url}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        记录标量指标。

        Args:
            metrics: 指标字典，键为指标名，值为数值
            step:    当前训练步数
        """
        self.wandb.log(metrics, step=step)

    def log_config(self, config: Dict[str, Any]):
        """更新超参数配置"""
        self.wandb.config.update(config)

    def log_summary(self, summary: Dict[str, Any]):
        """记录实验总结"""
        for key, value in summary.items():
            self.wandb.run.summary[key] = value

    def finish(self):
        """结束 WandB 会话"""
        self.wandb.finish()
        print("[WandB] 实验记录已完成")


# ============================================================
# SwanLab Logger
# ============================================================

class SwanLabLogger(Logger):
    """
    基于 SwanLab 的日志记录器 (国内友好的 WandB 替代方案)。

    需要提前安装: pip install swanlab
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        config: Dict[str, Any],
        tags: Optional[list] = None,
        log_dir: str = "./logs",
    ):
        """
        Args:
            project:  SwanLab 项目名称
            run_name: 本次运行的名称
            config:   超参数配置字典
            tags:     运行标签列表
            log_dir:  本地日志保存目录
        """
        try:
            import swanlab
            self.swanlab = swanlab
        except ImportError:
            raise ImportError(
                "SwanLab 未安装，请运行: pip install swanlab\n"
                "或切换到 WandB: 在配置中设置 logging.backend: wandb"
            )

        os.makedirs(log_dir, exist_ok=True)

        self.run = swanlab.init(
            project    = project,
            experiment_name = run_name,
            config     = config,
            logdir     = log_dir,
        )
        print(f"[SwanLab] 实验已启动: {project}/{run_name}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        记录标量指标。

        Args:
            metrics: 指标字典
            step:    当前训练步数
        """
        self.swanlab.log(metrics, step=step)

    def log_config(self, config: Dict[str, Any]):
        """更新超参数配置"""
        # SwanLab 在 init 时已记录 config，此处为兼容接口
        pass

    def log_summary(self, summary: Dict[str, Any]):
        """记录实验总结"""
        self.swanlab.log(summary)

    def finish(self):
        """结束 SwanLab 会话"""
        self.swanlab.finish()
        print("[SwanLab] 实验记录已完成")


# ============================================================
# 控制台 Logger (无需外部依赖的备用方案)
# ============================================================

class ConsoleLogger(Logger):
    """
    仅输出到控制台的简单日志记录器。
    当 WandB 和 SwanLab 均不可用时作为备用。
    """

    def __init__(self, log_dir: str = "./logs"):
        import json
        from datetime import datetime

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"train_{timestamp}.jsonl")
        self._step = 0
        print(f"[ConsoleLogger] 日志将保存至: {self.log_file}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        import json
        self._step = step if step is not None else self._step + 1
        log_entry = {"step": self._step, **metrics}
        # 打印到控制台
        metrics_str = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        print(f"[Step {self._step:6d}] {metrics_str}")
        # 写入文件
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_config(self, config: Dict[str, Any]):
        import json
        print(f"[Config] {json.dumps(config, indent=2, ensure_ascii=False)}")

    def log_summary(self, summary: Dict[str, Any]):
        print("\n" + "="*60)
        print("[实验总结]")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("="*60)

    def finish(self):
        print(f"[ConsoleLogger] 训练完成，日志已保存至: {self.log_file}")


# ============================================================
# Logger 工厂函数
# ============================================================

def build_logger(
    cfg: dict,
    run_name: str,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Logger:
    """
    根据配置字典构建日志记录器。

    Args:
        cfg:          完整配置字典
        run_name:     本次运行的名称 (e.g., "single_env_A_epoch100")
        extra_config: 额外的配置信息 (会合并到日志配置中)

    Returns:
        Logger 实例
    """
    log_cfg  = cfg.get("logging", {})
    paths_cfg = cfg.get("paths", {})

    backend = log_cfg.get("backend", "wandb").lower()
    project = log_cfg.get("project", "ACT-Cross-Env-Generalization")
    log_dir = paths_cfg.get("log_dir", "./logs")

    # 合并配置信息用于记录
    log_config = dict(cfg)
    if extra_config:
        log_config.update(extra_config)

    # 获取实验标签
    tags = cfg.get("experiment", {}).get("tags", [])

    if backend == "wandb":
        try:
            return WandBLogger(
                project  = project,
                run_name = run_name,
                config   = log_config,
                tags     = tags,
                log_dir  = log_dir,
            )
        except ImportError as e:
            print(f"[Warning] {e}")
            print("[Warning] 回退到 ConsoleLogger")
            return ConsoleLogger(log_dir=log_dir)

    elif backend == "swanlab":
        try:
            return SwanLabLogger(
                project  = project,
                run_name = run_name,
                config   = log_config,
                tags     = tags,
                log_dir  = log_dir,
            )
        except ImportError as e:
            print(f"[Warning] {e}")
            print("[Warning] 回退到 ConsoleLogger")
            return ConsoleLogger(log_dir=log_dir)

    else:
        print(f"[Warning] 未知日志后端: {backend}，使用 ConsoleLogger")
        return ConsoleLogger(log_dir=log_dir)
