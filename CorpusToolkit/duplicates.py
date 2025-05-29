from typing import Optional

from datasketch import MinHashLSH, MinHash


class DuplicateDetector:
    def __init__(
        self,
        threshold: float = 0.32,
        num_perm: int = 256,
        n: int = 3,  # n-gram 大小
    ):
        """
        重复检测器
        :param threshold: 判定为重复的 Jaccard 相似度阈值
        :param num_perm: MinHash 使用的排列数（越大越精确，消耗越高）
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.n = n  # n-gram 大小
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes: dict[int, MinHash] = {}
        self.texts: dict[int, str] = {}
        self.counter = 0  # 自增 ID 分配器

    def _char_ngrams(self, text: str):
        return {text[i:i + self.n] for i in range(len(text) - self.n + 1)}

    def _text_to_minhash(self, text: str) -> MinHash:
        """
        将文本转换为 MinHash 对象
        :param text:
        :return:
        """
        mh = MinHash(num_perm=self.num_perm)
        for token in self._char_ngrams(text):  # 可替换为 n-gram/词粒度
            mh.update(token.encode("utf8"))
        return mh

    def add(self, text: str) -> int:
        """
        添加一条文本，返回其内部 ID
        """
        idx = self.counter
        self.counter += 1
        mh = self._text_to_minhash(text)
        self.minhashes[idx] = mh
        self.texts[idx] = text
        self.lsh.insert(str(idx), mh)
        return idx

    def add_batch(self, texts: list[str]) -> list[int]:
        """
        批量添加文本，返回其 ID 列表
        """
        return [self.add(text) for text in texts]

    def query(self, text: str) -> list[int]:
        """
        查询与某文本相似的已有文本 ID（不包含自己）
        """
        mh = self._text_to_minhash(text)
        matches = self.lsh.query(mh)
        return [int(m) for m in matches]

    def find_all_duplicates(self) -> dict[int, list[int]]:
        """
        返回所有重复组（主 ID → 相似 ID 列表），已去重
        """
        seen = set()
        groups = {}
        for idx, mh in self.minhashes.items():
            if idx in seen:
                continue
            matches = map(int, self.lsh.query(mh))
            group = [m for m in matches if m != idx and m not in seen]
            if group:
                groups[idx] = group
                seen.update(group)
        return groups

    def get_duplicate_ids(self) -> set[int]:
        """返回所有重复文本的 ID（排除主 ID）"""
        return set(sid for ids in self.find_all_duplicates().values() for sid in ids)

    def get_text(self, idx: int) -> str:
        """
        获取指定 ID 的文本内容
        :param idx: 文本 ID
        :return: 文本内容
        """
        return self.texts.get(idx, "")

    def get_all_texts(self) -> dict[int, str]:
        """
        获取所有文本内容
        :return: 文本 ID → 文本内容 的字典
        """
        return self.texts.copy()

    def get_idx_by_text(self, text: str) -> Optional[int]:
        """
        根据文本内容获取其对应的 ID
        :param text: 文本内容
        :return: 文本 ID，如果不存在则返回 -1
        """
        for idx, t in self.texts.items():
            if t == text:
                return idx
        return None

    def get_all_minhashes(self) -> dict[int, MinHash]:
        """
        获取所有 MinHash 对象
        :return: 文本 ID → MinHash 对象 的字典
        """
        return self.minhashes.copy()
