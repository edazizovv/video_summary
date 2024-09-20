from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def Summarizer(link, model, min_length=128, max_length=1024):
    video_id = link.split("=")[1]

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        FinalTranscript = ' '.join([i['text'] for i in transcript])

        if model is None:
            return FinalTranscript
        else:

            if model == "Pegasus":
                checkpoint = "google/pegasus-large"
            if model == "Pegasus_Dailymail":
                checkpoint = "google/pegasus-cnn_dailymail"
            if model == "Pegasus_Multi":
                checkpoint = "google/pegasus-multi_news"
            elif model == "mT5":
                checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
            elif model == "BART":
                checkpoint = "sshleifer/distilbart-cnn-12-6"
            elif model == "FACE_BART":
                checkpoint = "facebook/bart-large-cnn"

            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

            inputs = tokenizer(FinalTranscript,
                               max_length=max_length,
                               return_tensors="pt",
                               truncation=True,
                               verbose=True)

            summary_ids = model.generate(inputs["input_ids"],
                                         max_length=max_length,
                                         min_length=min_length)
            summary = tokenizer.batch_decode(summary_ids,
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)

            return summary[0]
    except Exception as e:
        raise e
