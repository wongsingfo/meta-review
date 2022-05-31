import torch
import os
import pdb
import html
import json

class KnownBugsException(Exception):
    def __init__(self, exce):
        self.exce = exce

def write_file(file, s):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(s)


def read_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        s = f.read()
    return s


def _assert_eq_list(a, b):
    for k, (i, j) in enumerate(zip(a, b)):
        assert i == j, f"the {k}-th elements are different: {i} {j}"

    # ignore the padding
    # assert len(a) == len(b)


def _index_text(tokenizer, token_id):

    def _should_apply_monkey_patch(redo_token_id, offset_mapping):
        sos_token_id = 0
        eos_token_id = 2

        i = 0
        while token_id[i] == sos_token_id or token_id[i] == eos_token_id:
            i += 1
        num_pre_pad = i - 1

        if num_pre_pad > 0:
            redo_token_id = [eos_token_id] + \
                    [sos_token_id] * (num_pre_pad-1) + redo_token_id
            offset_mapping = [(0, 0)] * num_pre_pad + offset_mapping
        return redo_token_id, offset_mapping

    text = tokenizer.decode(token_id, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)
    helper = tokenizer(text, return_offsets_mapping=True)

    redo_token_id = helper["input_ids"]
    offset_mapping = helper["offset_mapping"]

    redo_token_id, offset_mapping = \
        _should_apply_monkey_patch(redo_token_id, offset_mapping)

    # BUG: sometimes the restored token_ids differs from the orignal :(
    try:
        _assert_eq_list(token_id, redo_token_id)
    except Exception as exce:
        raise KnownBugsException(exce)

    return text, offset_mapping


def _visualize_text_html(text, offset):
    output = []
    last_end = 0
    for i, (start, end) in enumerate(offset):
        if start < end:
            if not last_end == start:
                output.append('\n')
            elem = html.escape(text[start:end])
            s = f'<span :style="styles[{i}]">{elem}</span>'
            output.append(s)
            last_end = end
    return ''.join(output)


def _visualize_summary_html(text, offset, seg):
    output = []
    for i, (start, end) in enumerate(seg):
        start_t, _ = offset[start]
        _, end_t = offset[end - 1]
        elem = html.escape(text[start_t:end_t])
        s = f'<span @click="onSelect({i})" :class="selected=={i} && \'selected\'">{elem}</span>'
        output.append(s)
    return '\n'.join(output)


def _visualize_split(token_id):
    dot_token_id = 4
    split_index = [-1]
    for i, v in enumerate(token_id):
        if v == dot_token_id:
            split_index.append(i)
    # if split_index[-1] != len(token_id) - 1:
        # split_index.append(len(token_id) - 1)

    return [(i+1, j+1) for i, j in zip(split_index, split_index[1:])]


def _tensor_sentence_max(tensor, sentences):
    results = []
    for start, end in sentences:
        # BUG!!: tensor.shape[0] < end
        if tensor.shape[0] < end:
            raise KnownBugsException("the attention output dim is too small")
        arr = tensor[start:end].max(0).tolist()
        results.append(arr)
    return results


def _visualize(tensor, tokenizer, input_text, output_text, file):
    text, text_o = _index_text(tokenizer, input_text)
    text_html = _visualize_text_html(text, text_o)
    sentences = _visualize_split(output_text)

    summary, summary_o = _index_text(tokenizer, output_text)
    summary_html = _visualize_summary_html(summary, summary_o, sentences)

    weight = _tensor_sentence_max(tensor, sentences)
    weight_json = f"const data = {json.dumps(weight)};"

    template = read_file('src/template.html')
    html = template
    html = html.replace('{{weight}}', weight_json)
    html = html.replace('{{summary}}', summary_html)
    html = html.replace('{{text}}', text_html)
    write_file(file, html)


def visualize_cross_attention(tensor, tokenizer, input_text, output_text, filedir):
    input_text = input_text["input_ids"]

    for i in range(len(input_text)):
        file = f"{filedir}/{i}.html"
        try:
            _visualize(tensor[i], tokenizer, input_text[i], output_text[i], file)
        except KnownBugsException as e:
            print(f"bug for #{i}: {str(e)}")
            pass
