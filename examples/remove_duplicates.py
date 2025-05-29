import os

from CorpusToolkit import DuplicateDetector
from CorpusToolkit.utils import load_data
from tools.common import timeit, timed_block


@timeit("Remove Duplicates Script")
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    with timed_block("Load Data"):
        # 加载数据
        data_df = load_data([
            os.path.join(current_dir, file)
            for file in os.listdir(current_dir) if file.endswith('.parquet')
        ])
    print(f'Loaded data with {len(data_df)} rows and {len(data_df.columns)} columns.')

    with timed_block('Remove Duplicates'):
        detector = DuplicateDetector()
        id2df_index = {}
        for idx, row in data_df.iterrows():
            text = row['content']
            minhash_id = detector.add(text)
            id2df_index[minhash_id] = idx

        # 获取所有重复文本的 minhash ID（排除主 ID，仅保留一份）
        duplicate_ids = detector.get_duplicate_ids()

        # 将 minhash ID 转为 DataFrame 索引
        duplicate_df_indices = {id2df_index[mid] for mid in duplicate_ids}

        # 删除重复项
        data_df.drop(index=duplicate_df_indices, inplace=True)

    print(f'Removed {len(duplicate_df_indices)} duplicate groups, remaining data size: {len(data_df)}.')

    # 保存去重后的数据
    output_file = os.path.join(current_dir, f'removed_duplicates_{len(data_df)}.parquet')
    data_df.to_parquet(output_file, index=False)


if __name__ == "__main__":
    main()
