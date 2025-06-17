"""
Script for pulling historic data for Reddit API.

Documentation:
Reddit API:
https://praw.readthedocs.io/en/stable/index.html

Reddit Instance:
https://praw.readthedocs.io/en/stable/code_overview/reddit_instance.html

Subreddit class
https://praw.readthedocs.io/en/stable/code_overview/models/subreddit.html
"""
from typing import Dict, List
import praw
import pandas as pd
from datetime import datetime
import time

subreddits_list = [
    "worldnews",
    "askreddit",
    "google",
    "globaltalk",
    "conspiracy",
    "inthenews",
    "breakingnews",
    "news",
    "geopolitics",
    "politics",
    "politicaldiscussion",
    "worldpolitics",
    "politicsandwar",
    "military",
    "foodforthought",
    "collapse",
    "moderatepolitics",
    "history"
]

countries_dict = {
    "US": "(USA OR America OR 'United States')",
    "GB": "(UK OR British OR 'United Kingdom' OR Britain)",
    "CA": "(Canada OR Canadian)",
    "AU": "(Australia)",
    "UA": "(Ukraine)",
    "RU": "(Russia)",
    "FR": "(France OR French)",
    "DE": "(German)",
    "BR": "(Brazil)",
    "CN": "(China OR Chinese)",
    "JP": "(Japan)",
    "PK": "(Pakistan)",
    "KP": "('North Korea')",
    "KR": "('South Korea')",
    "IN": "(India)",
    "TW": "(Taiwan)",
    "NL": "(NetherLands OR Holland OR Dutch)",
    "ES": "(Spain OR Spanish)",
    "SE": "(Sweden OR Swedish)",
    "MX": "(Mexic)",
    "IR": "(Iran)",
    "IL": "(Israel)",
    "SA": "(Saudi)",
    "SY": "(Syria)",
    "FI": "(Finland OR Finnish)",
    "IE": "(Ireland OR Irish)",
    "AT": "(Austria)",
    "NO": "(Norway OR Norwegian)",
    "CH": "(Switzerland OR Swiss)",
    "IT": "(Italy OR Italian)",
    "MY": "(Malaysia)",
    "EG": "(Egypt)",
    "TR": "(Turkey OR Turkish)",
    "PT": "(Portugal OR Portuguese)",
    "PS": "(Palestin OR 'West Bank' OR Gaza)",
    "AE": "(UAE OR 'United Arab Emirates' OR Emarat)"
}

# Date range
start_date = "2011-07"
end_date = "2025-05"
dt_range = pd.period_range(start_date, end_date, freq="M")

# Init output empty dataframe
output_df = pd.DataFrame()
output_df.index = dt_range


def aggr_dates(
    dates: Dict[str, List[datetime]],
    date_range: pd.PeriodIndex
) -> pd.DataFrame:
    """
    :param dates: Dictionary of the form {"Dates": [datetime, datetime, ...]}
    :param date_range: Range of dates in format YYYY-MM

    :return: Dataframe with counts of dates found per month, ordered by the
    date range PeriodIndex.
    """
    # Create mini dataframe and drop duplicate dates (probably same posts)
    count_df = pd.DataFrame(dates).drop_duplicates(subset="Dates")

    # Count of posts per month and re-indexing for all required months
    count_df = count_df["Dates"].dt.to_period("M").value_counts().sort_index()
    return count_df.reindex(date_range, fill_value=0)


if __name__ == "__main__":

    # Init Reddit client
    reddit = praw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="PersonalScript by u/AcademicApp"
    )

    for code, country in countries_dict.items():
        col_name = f"War_Conflict_{code}"  # Column name with country data
        dates_dict = {"Dates": []}  # Dictionary to store posts dates

        # Search query
        query = f"({country} AND War) OR " + \
                f"({country} AND 'Military Conflict') OR " + \
                f"({country} AND 'Military Attack') OR " + \
                f"({country} AND 'Political Tension') OR " + \
                f"({country} AND 'Political Conflict') OR " + \
                f"({country} AND 'Armed Force Attack')"

        print(col_name)
        print(query)

        for sub_reddit in subreddits_list:
            try:
                posts = reddit.subreddit(sub_reddit).search(
                    query,
                    sort="relevance",
                    time_filter="all",
                    limit=800
                )

                # Get date per post
                for post in posts:
                    post_date = datetime.utcfromtimestamp(post.created_utc)
                    dates_dict["Dates"].append(post_date)

                time.sleep(0.1)
            except Exception as e:
                print(f"Error fetching {code}: {e}")

        # Store country data to the output dataframe
        df = aggr_dates(dates_dict, dt_range)
        output_df[col_name] = df.values

    # Store output to CSV
    output_df.to_csv("RedditWarConflicts.csv", index=True)
    print("Saved to RedditWarConflicts.csv")
