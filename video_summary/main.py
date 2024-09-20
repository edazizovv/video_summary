# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from summarize import Summarizer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # y = 'https://www.youtube.com/watch?v=Y3LufB6DK4k'
    # y = 'https://www.youtube.com/watch?v=7TEkzCObKwg'
    # y = 'https://www.youtube.com/watch?v=uz58FQvzGgc'
    # y = 'https://www.youtube.com/watch?v=Pz2ccD3It0M'
    ## y = 'https://www.youtube.com/watch?v=2jA65esP-7s'
    ## y = 'https://www.youtube.com/watch?v=7MO798RgAhE'
    # y = 'https://www.youtube.com/watch?v=Vy4JXjrPFmc'
    y = 'https://www.youtube.com/watch?v=6IRweY8KJzM'

    model = 'FACE_BART'
    result = Summarizer(link=y, model=model)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

"""
Detecting dialogue actors: exclude what says interviewer, analyse only what say(s) expert(s)

Expert(s) speech should be detected in terms of: text, video, voice (from source video), only relevant to the expert, 
so that:
    1. Text is processed into:
        a -- summary text
        b -- sentiment sub-index
    2. Video is processed into:
        a -- description of detected person(s) 
             (relevant to the topic / summary text data is retrieved from the knowledge graph) 
        b -- detected persons(s)' images (extracted)
        c -- sentiment sub-index (based on emotion detection)
    3. Voice is processed into:
        a -- sentiment sub-index

As a result, the following features are presented:

    0. Video name, source link, video id (ours)
    1. Summary text
    2. Integrated sentiment (based on the sub-indices)
    3. Persons description (images extracted + text)
"""
