import pandas
import torch
from torch.nn import functional
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
_tokenizer = None
_model = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer

def get_model():
    global _model
    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype='auto',
            device_map='auto'
        ).eval()
    return _model

def compute_perplexity(
    texts: str | list[str] | pandas.DataFrame | pandas.Series,
    batch_size: int = 32,
    return_format: type = list
) -> list[float] | pandas.Series:
    """
    使用 Qwen3-0.6B 对单句中文文本进行困惑度打分
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

    tokenizer = get_tokenizer()
    model = get_model()

    ppls = []
    with torch.inference_mode(), torch.autocast(model.device.type):
        for i in range(0, len(texts), batch_size):
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
