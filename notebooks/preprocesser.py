import pandas as pd


def load_balanced_reviews(csv_path):
    df = pd.read_csv(csv_path)
    # Filter for 1, 2, and 5 star reviews
    low = df[df["Score"].isin([1, 2])]
    high = df[df["Score"] == 5]
    # Balance the classes
    n_samples = min(len(low), len(high))
    low_balanced = low.sample(n=n_samples, random_state=42)
    high_balanced = high.sample(n=n_samples, random_state=42)
    balanced_df = (
        pd.concat([low_balanced, high_balanced])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    balanced_df.to_csv(data_path + "FilteredReviews.csv", index=False)
    return balanced_df


# Calls the function
data_path = "../data/"
balanced_reviews = load_balanced_reviews(data_path + "Reviews.csv")
