from CorpusToolkit.scorer import compute_perplexity

def test_compute_perplexity():
    # 测试困惑度评分
    sample_texts = [
        "他走进了咖啡店，点了一杯拿铁。",
        "中国是一个拥有悠久历史的国家。",
        "树立科学思想，掌握科学方法，了解科技知识。",
        "人工智能正在改变我们的生活方式。",
        "啊发疯开i句i阶段哦小脾气。",  # 无意义文本示例
    ]

    print(f"Perplexity of single text: {compute_perplexity(sample_texts)}")  # [9.5992, 14.1634, 26.9556, 10.4854, 3445.8342]


if __name__ == "__main__":
    # 测试困惑度评分
    test_compute_perplexity()
