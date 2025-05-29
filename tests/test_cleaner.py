from CorpusToolkit import Cleaner

def test_cleaner():
    rules = [
        {'func': 'filter_url'},  # 过滤URL
        {'func': 'filter_repeated_punctuation'},  # 过滤重复标点
        {'func': 'filter_uncommon_characters'},  # 过滤不常用字
        {'func': 'remove_excessive_punctuations', 'kwargs': {'min_repeat': 3, 'protect_urls': True}},  # 去除连续{num}个及以上的标点符号
        {'func': 'remove_all_newlines'},  # 移除所有换行符
        {'func': 'remove_html_entities'},  # 移除 HTML 实体字符
        {'func': 'normalize_punctuation', 'kwargs': {'all_to': 'half'}},  # 转换标点符号
    ]
    cleaner = Cleaner(rules=rules)
    text = '今天心情超棒~(*^▽^*)！来看我最新VLOG啦：https://t.cn/A6xxx 【#生活日常#】#vlog# 赞赞赞~~~~'
    cleaned_text = cleaner(text)
    print(cleaned_text)

if __name__ == '__main__':
    test_cleaner()
