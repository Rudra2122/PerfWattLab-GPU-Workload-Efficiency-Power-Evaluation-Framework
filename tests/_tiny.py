"""A tiny random Llama (GQA) so engine tests run on CPU in seconds."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM


def tiny_llama(seed=0, attn="eager"):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, attn_implementation=attn)
    m = LlamaForCausalLM(cfg).eval()
    return m


def hf_greedy(model, ids, n):
    out = model.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                         max_new_tokens=n, min_new_tokens=n, do_sample=False,
                         eos_token_id=None, pad_token_id=0)
    return out[0, len(ids):].tolist()


def tiny_tokenizer():
    """Offline word-level tokenizer with a 512-token vocab (for pipeline-API tests)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for i in range(3, 512):
        vocab[f"w{i}"] = i
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    t = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>",
                                eos_token="</s>", pad_token="</s>",
                                model_input_names=["input_ids", "attention_mask"])
    return t
