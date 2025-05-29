from functools import lru_cache

import pandas


def load_data(
    files: list[str],
) -> pandas.DataFrame:
    """
    Load data from files in the specified list.
    Args:
        files (list[str]): list of files to load
    Returns:
        pandas.DataFrame: Concatenated DataFrame from all files
    """
    file_list = []
    for file in files:
        if file.endswith('.tsv'):
            file_list.append(pandas.read_csv(file, sep='\t'))
        elif file.endswith('.parquet'):
            file_list.append(pandas.read_parquet(file))
        elif file.endswith('.json') or file.endswith('.jsonl'):
            file_list.append(pandas.read_json(file, lines=True))
        else:
            raise ValueError(f"Unsupported file format: {file}")

    return pandas.concat(file_list)


@lru_cache()
def get_punct_mappings() -> tuple[dict[str, str], dict[str, str]]:
    """
    Get mappings for full-width and half-width punctuation characters.
    :return:
        tuple: (full_to_half, half_to_full) dictionaries
    """
    full_to_half = {
        '，': ',', '。': '.', '！': '!', '？': '?',
        '；': ';', '：': ':',
        '（': '(', '）': ')', '【': '[', '】': ']',
        '《': '<', '》': '>', '〈': '<', '〉': '>',
        '「': '"', '」': '"', '‘': "'", '’': "'",
        '“': '"', '”': '"', '、': ','
    }
    half_to_full = {v: k for k, v in full_to_half.items()}
    return full_to_half, half_to_full

