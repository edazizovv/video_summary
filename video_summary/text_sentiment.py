
from transformers import pipeline

# I would prefer the first one!
# model_path = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

# model_path = 'finiteautomata/bertweet-base-sentiment-analysis'
# I don't like it # model_path = 'nlptown/bert-base-multilingual-uncased-sentiment'
# does not work # model_path = 'mrm8488/t5-base-finetuned-span-sentiment-extraction'

sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
result = sentiment_task("Covid cases are increasing fast!")
