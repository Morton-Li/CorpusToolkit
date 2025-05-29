#!/usr/bin/env python3
import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from common import prepare_environment

prepare_environment()

from tools.logger import logger

def detect_venv() -> Optional[str]:
    """检测是否存在虚拟环境目录并返回其激活脚本路径"""
    for name in ["venv", ".venv"]:
        venv_dir = Path(name)
        if venv_dir.exists() and venv_dir.is_dir():
            if os.name == "nt":
                activate = venv_dir / "Scripts" / "activate"
            else:
                activate = venv_dir / "bin" / "activate"
            if activate.exists() and activate.is_file():
                return str(activate.resolve())
    return None

def clean_build_artifacts():
    for folder in ["dist", "build"]:
        shutil.rmtree(folder, ignore_errors=True)
    for path in Path(".").glob("*.egg-info"):
        if path.is_dir():
            shutil.rmtree(path)

def run_command(command, use_shell=True):
    subprocess.run(command, shell=use_shell, check=True)

def main():
    venv_activate = detect_venv()
    if not venv_activate:
        raise RuntimeError("未检测到虚拟环境，请先创建虚拟环境。")
    logger.info(f'检测到虚拟环境：{Path(venv_activate).parent.parent}')

    logger.info(f'清理旧构建产物...')
    clean_build_artifacts()

    # 在 shell 中激活虚拟环境并执行后续命令
    shell_command = f'''source "{venv_activate}" && python -m build'''
    logger.info(f'开始构建：')
    run_command(shell_command)

    logger.info(f'构建完成！请检查 dist/ 目录中的产物。')

if __name__ == "__main__":
    main()
