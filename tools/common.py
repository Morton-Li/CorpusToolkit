import functools
import os
import sys
import time
from pathlib import Path
from typing import Optional


def prepare_environment():
    """
    添加项目根路径到 sys.path，确保工具脚本可正确导入主包；
    如果用户在 tools/ 目录下运行，自动切换回项目根路径。
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    sys.path.insert(0, str(project_root))

    from tools.logger import logger

    if Path.cwd().resolve().name == "tools":
        os.chdir(project_root)
        logger.warning(f'当前工作目录为 tools/，已自动切换到项目根目录：{project_root}')


def timeit(name: Optional[str] = None):
    """
    记录函数运行时间的装饰器
    :param name: 可选的名称标识，打印时使用
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            label = f"[{name}]" if name else f"[{func.__name__}]"
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"{label} executed in {elapsed:.2f} seconds")
            return result
        return wrapper
    return decorator


def timed_block(label):
    """
    计算运行时间辅助函数
    :param label:
    :return:
    """
    class Timer:
        def __enter__(self): self.start = time.time()
        def __exit__(self, *args): print(f"[{label}] took {time.time() - self.start:.2f} seconds")
    return Timer()
