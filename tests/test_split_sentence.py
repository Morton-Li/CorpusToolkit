from CorpusToolkit import split_sentence


def test_split_sentence():
    sample_text = '扩大单边免签政策实施范围。拓展单边免签国家范围，拓宽免签事由。建议对欧盟、东盟、海合会等符合条件的国家实施单边免签负面清单政策。在免签事由方面，尽快增加短期就医、短期教育培训、短期学术交流、短期演出等；延长单边免签入境停留时间。建议将单边免签入境最长停留时间由30天延长至90天；在海南自由贸易港，可酌情延长至180天；放宽第一入境点限制，实施“多点入境”和联动过境免签等政策；以单边免签吸引更多对我国友好的“顶流”，鼓励其在国内开展演出和交流等，更好地向全球宣介中国。'

    sentences = split_sentence(sample_text, max_len=20, min_len=5)
    print(f'Source text: {sample_text}')
    print(f'Split sentences: {sentences}')


if __name__ == "__main__":
    test_split_sentence()
