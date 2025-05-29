import datetime
import threading

# 日志级别
# DEBUG: 调试信息
# INFO: 一般信息
# WARNING: 警告
# ERROR: 错误
# CRITICAL: 严重错误
# 日志级别及其对应的名称和颜色
LOG_LEVELS = {
    1: {"name": "DEBUG", "color": "\033[0;36m"},    # Cyan
    2: {"name": "INFO", "color": "\033[0;32m"},     # Green
    3: {"name": "WARNING", "color": "\033[0;33m"},  # Yellow
    4: {"name": "ERROR", "color": "\033[0;31m"},    # Red
    5: {"name": "CRITICAL", "color": "\033[0;41m"}  # Background Red
}
RESET_COLOR = "\033[0m"


class Logger:
    def __init__(self):
        self.log_level = 2  # 默认日志级别
        self.lock = threading.Lock()  # 线程锁

    def log(self, message: str, level: int, newline: bool = True):
        # 如果日志级别低于指定的日志级别，则不输出
        if level < self.log_level:
            return

        # 选择的日志级别
        log_level_selected = LOG_LEVELS[level]

        # 构造日志消息
        log_message = f"{datetime.datetime.now()} [{log_level_selected["name"]}] {message}"

        with self.lock:
            print(f"{log_level_selected["color"]}{log_message}{RESET_COLOR}") if newline else print(f"{log_level_selected["color"]}{log_message}{RESET_COLOR}", end='', flush=(True if newline else False))


    def debug(self, message: str, newline: bool = True):    self.log(message, 1, newline)


    def info(self, message: str, newline: bool = True):     self.log(message, 2, newline)


    def warning(self, message: str, newline: bool = True):  self.log(message, 3, newline)


    def error(self, message: str, newline: bool = True):    self.log(message, 4, newline)


    def critical(self, message: str, newline: bool = True): self.log(message, 5, newline)

    def set_level(self, level: int):
        if level not in LOG_LEVELS:
            raise ValueError("Invalid log level")
        self.log_level = level

logger = Logger()
