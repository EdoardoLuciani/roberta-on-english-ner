import json, os

def load_label_mapping(file_name: str = 'label.json') -> tuple[dict[str, int], dict[int, str]]:
    with open('label.json') as f:
        ds_label_tag_mapping = json.load(f)
        ds_tag_label_mapping = dict((v,k) for k,v in ds_label_tag_mapping.items())
        assert len(ds_label_tag_mapping) == len(ds_tag_label_mapping)
    return ds_label_tag_mapping, ds_tag_label_mapping


def tokenize_and_align_labels(input, idx: int, tokenizer, ds_tag_label_mapping: dict, ds_label_tag_mapping: dict):
    tokenized_inputs = tokenizer(input["tokens"], truncation=True, is_split_into_words=True)

    input_ids = tokenized_inputs['input_ids']
    word_ids: list[int] = tokenized_inputs.word_ids()
    labels: list[int] = [0] * len(input_ids)

    assert len(input_ids) == len(word_ids) == len(labels)

    for i, word_id in enumerate(word_ids):
        # word_id is none on first and last padding tokens which are automatically introduced by the tokenizer, setting those to -100
        if word_id == None:
            assert i == 0 or i == (len(input_ids) - 1)
            labels[i] = -100
        # if this is a continuation of a previous word and the word has an actual meaning
        elif word_id == word_ids[i-1] and input['tags'][word_id] != 0:
            prev_word_tag: int = input['tags'][word_ids[i-1]]
            prev_word_label: str = ds_tag_label_mapping[prev_word_tag]

            if prev_word_label.startswith('I-'):
                labels[i] = prev_word_tag
            elif prev_word_label.startswith('B-'):
                labels[i] = ds_label_tag_mapping[prev_word_label.replace('B-', 'I-')]
            else:
                raise Exception(f"Cannot determine label for word_id {word_id} and dataset row {idx}")
        else:
            labels[i] = input['tags'][word_id]

    tokenized_inputs['labels'] = labels

    return tokenized_inputs


def group_entries(input, max_list_size=512):
    input: dict = dict(input)
    output: dict[str, list[int]] = {k:[[]] for k in input.keys()}

    for x,y,z in zip(input['input_ids'], input['attention_mask'], input['labels']):
        assert len(x) == len(y) == len(z)

        if len(output['input_ids'][-1]) + len(x) < max_list_size:
            output['input_ids'][-1].extend(x)
            output['attention_mask'][-1].extend(y)
            output['labels'][-1].extend(z)
        else:
            output['input_ids'].append(x)
            output['attention_mask'].append(y)
            output['labels'].append(z)


    for x,y,z in zip(output['input_ids'], output['attention_mask'], output['labels']):
        elem_diff = max_list_size - len(x)

        x.extend(elem_diff * [1])
        y.extend(elem_diff * [0])
        z.extend(elem_diff * [-100])
    
    return output


def process_dataset(ds, tokenizer):
    ds_label_tag_mapping, ds_tag_label_mapping = load_label_mapping()
    args = {"tokenizer" : tokenizer, "ds_label_tag_mapping" : ds_label_tag_mapping, "ds_tag_label_mapping" : ds_tag_label_mapping}

    return (ds.map(tokenize_and_align_labels, with_indices=True, remove_columns=['tokens', 'tags'], num_proc=os.cpu_count(), fn_kwargs=args)
            .shuffle(seed=5473)
            .map(group_entries, batched=True, batch_size=None))