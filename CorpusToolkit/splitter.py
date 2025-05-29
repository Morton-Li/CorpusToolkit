
SPLIT_PUNCTUATIONS = {
    1: ['。', '！', '？', '!', '?', ',', '，', ';', '；', '～', '~'],     # 一级标点符号
    2: ['…', '》', '】', '）', ')', '」', '”', '’'],        # 二级标点符号
}

def split_sentence(text: str, max_len: int = 128, min_len: int = 16) -> list[str]:
    """
    将文本按句子分割，确保每个句子不超过指定长度
    Args:
        text: 输入文本
        max_len: 每个句子的最大长度
        min_len: 每个句子的最小长度，若剩余文本长度小于此值则会舍弃
        strict: TODO: 是否强制丢弃无切点文本，如果为 True 则不会强制切割没有分割符的文本

    Returns:
        分割后的句子列表
    """
    if len(text) < min_len:
        return []
    if len(text) <= max_len:
        return [text.strip()]

    for group_num, punctuation_list in SPLIT_PUNCTUATIONS.items():
        for i in range(max_len, min_len - 1, -1):  # 从 max_len 向前找合适切点
            # 检查当前字符是否为一级或二级标点符号
            if text[i - 1] in punctuation_list:
                # 当前分割点分割的文本 + split_sentence 递归调用后的文本
                return [text[:i].strip()] + split_sentence(text[i:].strip(), max_len=max_len, min_len=min_len)

    # 如果没有找到分割符号则向后滑动直到有分隔符号
    for i in range(max_len, len(text)):  # (max_len, len(text))
        char = text[i]  # 当前字符
        if char in SPLIT_PUNCTUATIONS[1] or char in SPLIT_PUNCTUATIONS[2]:
            # 舍弃前段，跳过分割点后继续处理
            split_text = text[i + 1:]  # 跳过分割点
            return split_sentence(split_text, max_len=max_len, min_len=min_len)

    # 如果向后滑动没有找到分割符号，则直接返回空列表
    return []
