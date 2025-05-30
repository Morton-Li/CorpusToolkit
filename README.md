# CorpusToolkit

**CorpusToolkit** 是一个面向中文语料预处理、去重与质量评估的工具包，适用于 NLP 数据清洗与训练语料准备场景。

---

## 📦 Installation / 安装

```bash
pip install CorpusKit
```

或从源码安装：

```bash
git clone https://github.com/Morton-Li/CorpusToolkit.git
cd CorpusToolkit
pip install .
```

<details>
<summary><strong>⚠️ 注意 / Note</strong></summary>

如需使用机器学习相关功能，请确保安装了 `ml` 可选依赖项：  
To use neural network-related features, make sure to install the optional dependencies group `ml`:  

* For PyPI install / 使用 PyPI 安装：
  ```bash
  pip install CorpusKit[ml]
  ```

* For source install / 从源码安装：
  ```bash
  pip install .[ml]
  ```

</details>

---

## 🧰 模块功能简介 / Module Overview

| 模块                                | 功能                                   |
|-----------------------------------|--------------------------------------|
| `CorpusToolkit.scorer`            | 计算中文语料的质量评分，如困惑度（Perplexity）         |
| `CorpusToolkit.Cleaner`           | 标点规范、空白符清洗、HTML 实体解码、emoji 过滤等语料清洗功能 |
| `CorpusToolkit.DuplicateDetector` | 基于 MinHash + LSH 实现语句级重复检测           |
| `CorpusToolkit.split_sentence`    | 中文文本长句分割工具                           |

---

## 🪄 快速使用示例 / Quick Usage Examples

### 1. **计算困惑度 / Compute Perplexity**

```python
from CorpusToolkit.scorer import compute_perplexity

sample_texts = [
    "他走进了咖啡店，点了一杯拿铁。",
    "中国是一个拥有悠久历史的国家。",
    "树立科学思想，掌握科学方法，了解科技知识。",
    "人工智能正在改变我们的生活方式。",
    "啊发疯开i句i阶段哦小脾气。",  # 无意义文本示例
]
ppl_scores = compute_perplexity(sample_texts)
print(ppl_scores)  # [9.5992, 14.1634, 26.9556, 10.4854, 3445.8342]
```

### 2. **检测与去除重复语句 / Detect and Remove Duplicates**

```python
from CorpusToolkit import DuplicateDetector

sample_texts = [
    "今天天气不错",
    "我喜欢人工智能。",
    "我非常喜欢人工智能。",
    "我喜欢人工智能。",
]

detector = DuplicateDetector()
detector.add_batch(sample_texts)

for text in sample_texts:
    similar_ids = detector.query(text)
    print(f"Text: '{text}' has similar IDs: {similar_ids}")

# Text: '今天天气不错' has similar IDs: [0]
# Text: '我喜欢人工智能。' has similar IDs: [3, 1, 2]
# Text: '我非常喜欢人工智能。' has similar IDs: [3, 1, 2]
# Text: '我喜欢人工智能。' has similar IDs: [3, 1, 2]

duplicates = detector.find_all_duplicates()
print("All duplicate groups:", duplicates)  # All duplicate groups: {1: [3, 2]}
```

> 更多示例请参考 [examples](./examples) 目录。

---

## 📄 License / 许可证

本项目采用 Apache License 2.0 协议。

