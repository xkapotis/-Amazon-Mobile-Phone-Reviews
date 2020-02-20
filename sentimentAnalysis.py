import pandas as pd

data = pd.read_csv("./reviews.csv")

reviews = data["Reviews"]
# print(type(reviews))

# print(data)

# import SentimentIntensityAnalyzer class 
# from vaderSentiment.vaderSentiment module. 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
  
x = []  
# function to print sentiments 
# of the sentence. 
def sentiment_scores(sentence): 
  
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
  
    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
      
    # print("Overall sentiment dictionary is : ", sentiment_dict) 
    # print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
    # print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    # print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 
  
    # print("Sentence Overall Rated As", end = " ") 
  
    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] >= 0.05 : 
        # print("Positive") 
        x.append("Positive")
  
    elif sentiment_dict['compound'] <= - 0.05 : 
        # print("Negative") 
        x.append("Negative")
  
    else : 
        # print("Neutral") 
        x.append("Neutral")
    print(len(x))
    # print(x)

  




for row in reviews:
    # print(row)
    sentiment_scores(str(row))

# revs = data["Reviews"].head(100)
reviews = x
reviews = pd.DataFrame(reviews)
# print(type(reviews))
# print(type(reviews))


reviews.to_csv(r"./reviews_anlzd.csv", index=None, header="Reviews")






# etrexe apo 19:35 mexri 21:44
