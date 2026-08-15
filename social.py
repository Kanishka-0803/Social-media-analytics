import social_tests as test

### PHASE 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def parse_label(label):
    info = dict()
    start = label.index(":")
    end = label.index("(")
    indx_from = label.index("from")

    name = label[start + 2:end - 1]
    info["name"] = name

    position = label[end + 1:indx_from - 1]
    info["position"] = position

    state = label[indx_from + 5:len(label) - 1]
    info["state"] = state
    return info

def get_region_from_state(state_df, state):
    row = state_df[state_df["state"] == state]
    column = row.iloc[0]["region"]
    return column

end_chars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]
def find_hashtags(message):
    hashtags = []
    for i in range(len(message)):
        if message[i] == "#":
            j = i + 1
            while j < len(message) and message[j] not in end_chars:
                j = j + 1
            hashtags.append(message[i:j])
    return hashtags

def find_sentiment(classifier, message):
    sentiment = classifier.polarity_scores(message)
    score = sentiment["compound"]
    if score > 0.1:
        return (score, "positive")
    elif score < -0.1:
        return (score, "negative")
    else:
        return (score, "neutral")  
    
def add_columns(data, state_df):
    classifier = SentimentIntensityAnalyzer()
    names, positions, states, regions = [], [], [], []
    for label in data["label"]:
        label_result = parse_label(label)
        names.append(label_result["name"])
        positions.append(label_result["position"])
        state = label_result["state"]
        states.append(state)
        regions.append(get_region_from_state(state_df, state))

    data["name"] = names
    data["position"] = positions
    data["state"] = states
    data["region"] = regions

    hashtags, scores, sentiment = [], [], []
    for text in data["text"]:
        hashtags.append(find_hashtags(text))
        (score, category) = find_sentiment(classifier, text)
        scores.append(score)
        sentiment.append(category)

    data["hashtags"] = hashtags
    data["score"] = scores
    data["sentiment"] = sentiment
    return

### PHASE 2 ###

def get_sentiment_quantiles(data, col_name, col_value):
    if col_name != "":
        data = data[data[col_name] == col_value]
    result = [data["score"].min()]
    result.extend(list(round(data["score"].quantile([0.25, 0.5, 0.75]), 5)))
    result.append(data["score"].max())
    return result

def get_hashtag_subset(data, col_name, col_value):
    data = data[data[col_name] == col_value]
    all_hashtags = set()
    for hashtags in data["hashtags"]:
        for tag in hashtags:
            all_hashtags.add(tag)
    return all_hashtags

def get_hashtag_rates(data):
    d = {}
    for hashtags in data["hashtags"]:
        for tag in hashtags:
            if tag not in d:
                d[tag] = 0
            d[tag] += 1
    return d

def most_common_hashtags(hashtags, count):
    best_only = {}
    while len(best_only) < count:
        curr_best = None
        curr_count = 0
        for k in hashtags:
            if hashtags[k] > curr_count and k not in best_only:
                curr_best = k
                curr_count = hashtags[k]
        best_only[curr_best] = curr_count
    return best_only

def get_hashtag_sentiment(data, hashtag):
    total = 0
    count = 0
    for index, row in data.iterrows():
        hashtags = row["hashtags"]
        sent = row["sentiment"]
        if hashtag in hashtags:
            count += 1
            if sent == "positive":
                total += 1
            elif sent == "negative":
                total -= 1
    return total / count 

### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    test.test_all()
    test.run()