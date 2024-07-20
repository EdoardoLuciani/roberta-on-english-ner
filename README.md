# NER detection powered by RoBERTa
Presented here is an english NER model, finetuned from [roberta-base](https://huggingface.co/FacebookAI/roberta-base). Weights available on [huggingface](https://huggingface.co/EdoardoLuciani/roberta-on-english-ner). Code available on [github](https://github.com/EdoardoLuciani/roberta-on-english-ner)

### Sample
The setup closely follows the [RobertaForTokenClassification](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaForTokenClassification) sample code:

```python
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaForTokenClassification.from_pretrained("EdoardoLuciani/roberta-on-english-ner")

inputs = tokenizer(
    "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_token_class_ids = logits.argmax(-1)
```

More specific integration, along with pretty printing and output parsing, is available on [example.ipynb](example.ipynb). Here is an extract of the output:

```
Barack Obama was born in Hawaii and served as the 44th President of the United States.
------------             ------                   --                -----------------
PERSON                   GPE                      ORDINAL           GPE


Apple Inc. is headquartered in Cupertino, California, and was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.
----------                     ---------  ----------                     ----------  -------------      ------------
ORG                            GPE        GPE                            PERSON      PERSON             PERSON


On July 20, 1969, Neil Armstrong and Buzz Aldrin became the first humans to walk on the moon as part of the Apollo 11 mission.
   -------------  --------------     -----------            -----                                           ---------
   DATE           PERSON             PERSON                 ORDINAL                                         EVENT
```

### Labels
The labels used follow the ones provided by [ontonotes5](https://paperswithcode.com/dataset/ontonotes-5-0) which are available [here](label.json). They are formatted in a part of the [TNER](https://github.com/asahi417/tner) project. They include:
```
CARDINAL, DATE, PERSON, NORP, GPE, LAW, PERCENT, ORDINAL, MONEY, WORK_OF_ART, FAC, TIME, QUANTITY, PRODUCT, LANGUAGE, ORG, LOC, EVENT
```


### Evaluation
Model has been evaluated both manually and with a portion of the ontonotes5 dataset never seen in training. Accuracy scores for the latter amount to 99.5%.
Full code used for the evaluation is available on [test.ipynb](test.ipynb)


### Dataset
Training data has been provided by the [ontonotes5](https://paperswithcode.com/dataset/ontonotes-5-0) dataset, specifically using the postprocessed dataset by [tner available on huggingface](https://huggingface.co/datasets/tner/ontonotes5)

Dataset has been further processed to split the labels between the model's tokens, assuring consistency in the model output. Full code for training and dataset processing is available on [train.ipynb](train.ipynb).