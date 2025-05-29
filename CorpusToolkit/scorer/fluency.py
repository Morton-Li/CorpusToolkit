import pandas
import torch
from torch.nn import functional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

_tokenizer = None
_model = None

def get_tokenizer(model_name: str = 'Qwen/Qwen3-0.6B-Base'):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer

def get_model(model_name: str = 'Qwen/Qwen3-0.6B-Base'):
    global _model
    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype='auto',
            device_map='auto'
        ).eval()
    return _model

def compute_perplexity(
    texts: str | list[str] | pandas.DataFrame | pandas.Series,
    batch_size: int = 32,
    model_name: str = 'Qwen/Qwen3-0.6B-Base',
    progress_bar: bool = True,
    return_format: type = list
) -> list[float] | pandas.Series:
    """
    使用 Qwen3-0.6B 对单句中文文本进行困惑度打分
    :param texts: 输入文本，可以是单个字符串、字符串列表、pandas DataFrame 或 pandas Series
    :param batch_size: 批处理大小
    :param model_name: 模型名称，默认为 'Qwen/Qwen3-0.6B-Base'
    :param progress_bar: 是否显示进度条
    :param return_format: 返回格式，可以是 list 或 pandas.Series
    """
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, pandas.DataFrame):
        # 如果是 DataFrame，不能为多列
        if len(texts.columns) != 1:
            raise ValueError("DataFrame must have exactly one column.")
        texts = texts.iloc[:, 0].tolist()
    elif isinstance(texts, pandas.Series):
        texts = texts.tolist()
    elif not isinstance(texts, list):
        raise TypeError("Input must be a string, list of strings, pandas DataFrame, or pandas Series.")

    if len(texts) == 0:
        raise ValueError("Input list cannot be empty.")

    tokenizer = get_tokenizer(model_name=model_name)
    model = get_model(model_name=model_name)

    ppls = []
    with torch.inference_mode(), torch.autocast(model.device.type):
        for i in tqdm(
            range(0, len(texts), batch_size),
            disable=not progress_bar,
            desc="Calculating perplexity",
            dynamic_ncols=True,
            leave=False,
            unit="batch",
        ):
            # 截断列表
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(model.device)
            outputs = model(**inputs, use_cache=False)
            shift_labels = functional.pad(inputs.input_ids, (0, 1), value=tokenizer.pad_token_id)[..., 1:].contiguous()  # [batch, 1+seq]
            loss = functional.cross_entropy(
                input=outputs.logits.view(-1, model.config.vocab_size),  # [batch*seq, vocab]
                target=shift_labels.view(-1).to(model.device),  # [batch*seq]
                ignore_index=tokenizer.pad_token_id,
                reduction='none'
            ).view(shift_labels.size())

            # valid_mask = inputs.attention_mask.bool() & (shift_labels != -100)
            valid_mask = shift_labels != tokenizer.pad_token_id
            loss_per_sample = (loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
            ppls.extend([round(ppl.item(), 4) for ppl in torch.exp(loss_per_sample)])

            del inputs, outputs, shift_labels, loss, loss_per_sample, valid_mask
            if model.device.type == 'cuda':
                torch.cuda.empty_cache()

    if return_format == list:
        return ppls
    elif return_format == pandas.Series:
        return pandas.Series(ppls)
    else:
        raise ValueError("return_format must be either list or pandas.Series.")
