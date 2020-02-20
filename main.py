import pandas as pd

data = pd.read_csv("./Amazon_Unlocked_Mobile.csv")

reviews = data["Reviews"].str.upper()

print(reviews)

reviews.to_csv(r"./reviews.csv", index=None, header="Reviews")