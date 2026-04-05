"""
Script for pulling historic Reddit data using the free JSON endpoint.

Documentation for subreddit search:
https://www.reddit.com/dev/api#GET_subreddits_search

Useful URLs:
https://dev.to/agenthustler/how-to-scrape-reddit-in-2026-subreddits-posts-comments-via-python-4el5
https://apidirect.io/blog/reddit-search-api
https://data365.co/blog/reddit-search-api
https://praw.readthedocs.io/en/stable/code_overview/models/subreddit.html#praw.models.Subreddit.search
"""
import requests
from typing import Dict, List
import pandas as pd
from datetime import datetime
import time

from subreddits import subreddits_list
from countries import countries_dict

# Date range
START_DATE = "2011-07"
END_DATE = "2026-02"

# For requests
HEADERS = {"User-Agent": "PythonScraper/1.0 (research)"}

# Keywords defined with double quotes for multi-word phrases in lucene
conflict_keywords = (
    'War OR "Military Conflict" OR "Military Attack" OR '
    '"Political Tension" OR "Political Conflict" OR "Armed Force Attack"'
)

# For queries
TIME_FILTER = "all"
SORT = "relevance"
SYNTAX = "lucene"  # Can be 'cloudsearch' or 'plain'
MAX_POSTS = 1_000


def aggr_dates(
    dates: Dict[str, List[datetime]],
    period_range: pd.PeriodIndex
) -> pd.DataFrame:
    """
    :param dates: Dictionary of the form {"Dates": [datetime, datetime, ...]}
    :param period_range: Range of dates in format YYYY-MM

    :return: Dataframe with counts of dates found per month, ordered by the
    preriod range PeriodIndex.
    """
    # Create mini dataframe and drop duplicate dates (probably same posts)
    count_df = pd.DataFrame(dates).drop_duplicates(subset="Dates")

    # Count of posts per month and re-indexing for all required months
    count_df = count_df["Dates"].dt.to_period("M").value_counts().sort_index()
    return count_df.reindex(period_range, fill_value=0)


if __name__ == "__main__":
    # Init output empty dataframe
    date_range = pd.period_range(START_DATE, END_DATE, freq="M")
    output_df = pd.DataFrame(index=date_range)
    output_df.index.name = "Date"

    for code, country in countries_dict.items():
        col_name = f"War_Conflict_{code}"  # Column name with country data
        dates_dict = {"Dates": []}  # Dictionary to store posts dates

        # Search query
        QUERY = f"({country} AND ({conflict_keywords}))"

        print(f"Searching posts for {country}...")

        for sub_reddit in subreddits_list:
            print(f"Searching subreddit: {sub_reddit}")
            try:
                # JSON API URL
                URL = f"https://www.reddit.com/r/{sub_reddit}/search.json"

                # List to store subreddit posts
                sub_results = []

                # Parameter for output pagination
                after = None

                # Request parameters
                params = {
                    "q": QUERY,
                    "sort": SORT,
                    "time_filter": TIME_FILTER,
                    "limit": 100,
                    "raw_json": 1,
                    "syntax": SYNTAX,
                    "restrict_sr": 1 if sub_reddit else 0
                }

                while len(sub_results) < MAX_POSTS:
                    if after:
                        params["after"] = after

                    print("Making request...")
                    response = requests.get(URL, params=params, headers=HEADERS)

                    # Rate limit reached - sleep
                    if response.status_code == 429:
                        print("Rate limited — waiting 500s...")
                        time.sleep(500)
                        continue

                    # Status code for banned subreddits (e.g., arabasian)
                    elif response.status_code == 404:
                        print(f"{sub_reddit} banned!")
                        break

                    # Fetch current page posts
                    data = response.json()
                    children = data["data"]["children"]
                    print(f"Fetched {len(children)} posts...!")

                    # Get current page post dates
                    for child in children:
                        post = child["data"]
                        post_utc = post["created_utc"]
                        post_date = datetime.utcfromtimestamp(post_utc)
                        sub_results.append(post_date)

                    # Update 'after' parameter to get posts of next page
                    after = data["data"]["after"]
                    if not after:
                        break

                    time.sleep(2)  # Reddit rate limit: ~30 req/min without auth

                # Append all posts to the list of current country posts
                print(f"Fetched {len(sub_results)} posts in total!")
                dates_dict["Dates"] += sub_results
                time.sleep(1)

            except Exception as e:
                print(f"Error fetching {code}: {e}")

        # Store country data to the output dataframe
        df = aggr_dates(dates_dict, date_range)
        df.to_csv(f"reddit_interm_{code}.csv", index=True)
        output_df[col_name] = df.values

    # Create column with monthly sum across all countries.
    output_df["War_Conflict_All"] = output_df.sum(axis=1).astype(int)

    # Export as CSV
    output_csv = "RedditData_v3.csv"
    output_df.to_csv(output_csv, index=True)
    print(f"Saved to {output_csv}")
