import html
import re
import sys

from .utils import get_punct_mappings


def filter_url(text: str) -> str:
    """
    过滤URL
    :param text: 文本
    :return: 过滤后的文本
    """
    return re.sub(r'[:：]?\s*https?://[^\s～·！（）【】「」、｜；：‘“’”《，》。？]*', '', text)


def filter_repeated_punctuation(text: str) -> str:
    """
    过滤重复标点
    :param text: 文本
    :return: 过滤后的文本
    """
    return re.sub(r'([～·！@#¥%…&*（）—+【】「」、｜；：‘“’”《，》。？/~`!$^()_\-={}|\[\]\\:";\'<>?,. ])\1+', r'\1', text)


def filter_uncommon_characters(text: str) -> str:
    """
    过滤不常用字
    :param text: 文本
    :return: 过滤后的文本
    """
    pattern = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9\s～·！@#¥%…&*（）—+【】「」、｜；：‘“’”《，》。？/~`!$^()_\-={}|\[\]\\:";\'<>?,.]')
    return pattern.sub('', text)


def remove_excessive_punctuations(
    text: str,
    min_repeat: int = 3,
    protect_urls: bool = False
) -> str:
    """
    去除连续{num}个及以上的标点符号
    :param text: 文本
    :param min_repeat: 最小重复次数，默认为3
    :param protect_urls: 是否保护 URL，不清除 URL 中的连续标点，默认为 False
    Returns:

    """
    def _clean(t: str) -> str:
        pattern = re.compile(rf'([·@#¥%…&*（）—+「」、｜；：‘“’”《，》。/`$^()_\-={{}}|\[\]\\:";\'<>,.]){{{min_repeat},}}')
        return pattern.sub('', t)

    if protect_urls:
        # 1. 用唯一占位符替换协议头
        protocol_pattern = re.compile(r'https?://')
        protocols = protocol_pattern.findall(text)
        protocol_placeholders = {f"protect_protocol_{i}>>": url for i, url in enumerate(protocols)}
        for i, url in enumerate(protocols):
            text = text.replace(url, f"protect_protocol_{i}>>")

        # 2. 清除连续标点
        text = _clean(t=text)

        # 3. 还原 URL
        for placeholder, url in protocol_placeholders.items():
            text = text.replace(placeholder, url)

        return text
    else:
        return _clean(t=text)


def remove_all_newlines(text: str) -> str:
    """
    移除所有换行符
    :param text: 文本
    :return: 处理后的文本
    """
    return re.sub(r'(?:\r\n|\r|\n|<br\s*/?>)+', ' ', text, flags=re.IGNORECASE).strip()


def remove_html_entities(text: str) -> str:
    """
    移除 HTML 实体字符
    :param text: 文本
    :return: 处理后的文本
    """
    return html.unescape(text)


def normalize_punctuation(
    text: str,
    # all_to: Optional[str] = None,
    all_to: str,
) -> str:
    """
    标准化标点符号
    :param text:
    :param all_to: 'full' 或 'half'，表示将所有标点符号转换为全角或半角
    :return:
    """
    if len(text) == 0:
        return text

    full_to_half, half_to_full = get_punct_mappings()
    if all_to is not None:
        if all_to == 'half':
            mapping = full_to_half
        elif all_to == 'full':
            mapping = half_to_full
        else:
            raise ValueError(f"Invalid value for all_to: {all_to}. Must be 'full' or 'half'.")
        return ''.join(str(mapping.get(ch, ch)) for ch in text)

    # 统计英文占比
    english_count = sum(1 for ch in text if 'a' <= ch.lower() <= 'z')
    total_count = len(text)

    english_ratio = english_count / total_count

    # TODO: 通过合理的判断调整标点符号大小写，完成后 all_to 变量可以为 None


class Cleaner:
    def __init__(
        self,
        rules: list[dict],
    ):
        """
        初始化数据清洗器
        :param rules: 规则列表
        """
        self.actions = {}
        for rule in rules:
            if isinstance(rule['func'], str):
                rule['func'] = getattr(sys.modules[__name__], rule['func'])
                if not callable(rule['func']):
                    raise ValueError(f"Function {rule['func']} is not callable.")
                self.actions[rule['func']] = rule.get('kwargs', {})
            elif callable(rule['func']):
                self.actions[rule['func']] = rule.get('kwargs', {})
            else:
                raise ValueError(f"Invalid function type: {rule['func']} must be a string or callable.")

    def __call__(self, text: str) -> str:
        """
        处理数据
        Returns:

        """
        for func, kwargs in self.actions.items(): text = func(text=text, **kwargs)
        return text
