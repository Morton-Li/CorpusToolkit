import os

import pandas

from CorpusToolkit.scorer import compute_perplexity
from CorpusToolkit.utils import load_data
from tools.common import timeit, timed_block


@timeit("Fluency Scoring Script")
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    with timed_block("Load Data"):
        # 加载数据
        data_df = load_data([
            os.path.join(current_dir, file)
            for file in os.listdir(current_dir) if file.endswith('.parquet')
        ])
    print(f'Loaded data with {len(data_df)} rows and {len(data_df.columns)} columns.')

    with timed_block("Compute Perplexity"):
        # 计算困惑度
        ppl_series = compute_perplexity(data_df['content'], batch_size=32, return_format=pandas.Series)
    print(f'Perplexity computed for {len(ppl_series)} texts.')

    # 将困惑度添加到 DataFrame
    data_df['perplexity'] = ppl_series
    # 保存结果
    output_file = os.path.join(current_dir, f'fluency_scores_{len(data_df)}.parquet')
    data_df.to_parquet(output_file, index=False)


if __name__ == "__main__":
    main()
