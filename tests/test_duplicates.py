from CorpusToolkit import DuplicateDetector

def test_duplicate_detector():
    # 创建 DuplicateDetector 实例
    detector = DuplicateDetector(threshold=0.32)

    # 添加文本
    texts = [
        "今天天气不错",
        "我喜欢人工智能。",
        "我非常喜欢人工智能。",
        "我喜欢人工智能。",
    ]

    ids = detector.add_batch(texts)
    assert len(ids) == len(texts), "添加文本数量不匹配"
    print(f"Added {len(ids)} texts.")

    # 查询相似文本
    for text in texts:
        similar_ids = detector.query(text)
        print(f"Text: '{text}' has similar IDs: {similar_ids}")

    # 查找所有重复文本
    duplicates = detector.find_all_duplicates()
    print("All duplicate groups:", duplicates)

if __name__ == "__main__":
    # 测试 DuplicateDetector
    test_duplicate_detector()

