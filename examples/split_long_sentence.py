import os

import pandas

from CorpusToolkit import split_sentence
from CorpusToolkit.utils import load_data
from tools.common import timeit, timed_block


@timeit("Split Long Sentence Script")
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    with timed_block("Load Data"):
        # 加载数据
        data_df = load_data([
            os.path.join(current_dir, file)
            for file in os.listdir(current_dir) if file.endswith('.parquet')
        ])
    print(f'Loaded data with {len(data_df)} rows and {len(data_df.columns)} columns.')

    with timed_block("Split Sentences"):
        sentences = []
        # 分割长句子
        for text in data_df['content']:
            split_sentences = split_sentence(text, max_len=128, min_len=6)
            sentences.extend(split_sentences)

    print(f'Split into {len(sentences)} sentences.')
    # 保存结果
    output_file = os.path.join(current_dir, f'split_sentences_{len(sentences)}.parquet')
    pandas.DataFrame({'sentence': sentences}).to_parquet(output_file, index=False)

if __name__ == "__main__":
    main()
