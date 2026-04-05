"""
Script for pulling historic trends from YouTube API.

API documentation:
https://developers.google.com/youtube/v3

Python client repository and documentation:
https://github.com/googleapis/google-api-python-client

Documentation on Google client libraries and cloud APIs:
https://docs.cloud.google.com/apis/docs/client-libraries-explained#google_api_client_libraries
"""
import datetime
import time
import json

import pandas as pd
from googleapiclient.discovery import build

# Load YouTube API key
with open("config.json") as config_file:
    config = json.load(config_file)

API_KEY = config["apikey"]

# Search topics (| is the OR operator for YouTube API).
TOPICS = [
    '"war conflict"',
    '"armed force attack"',
    '"political tension"',
    '"military attack"',
    '"armed force war"'
]
TOPIC_QUERY = "|".join(TOPICS)

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

# Date range, year range and months
START_DATE = "2011-07"
END_DATE = "2026-02"
date_range = pd.period_range(START_DATE, END_DATE, freq="M")
year_range = range(2011, 2026 + 1)
month_list = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]

# List for all count data
counts_list = []

# Dataframe columns
df_columns = [f"War_Conflict_{code}" for code in countries_dict.keys()]

# tracking counters
counter = 0
start = 0

# YouTube Build object
youtube = build("youtube", "v3", developerKey=API_KEY)

for year in year_range:

    # Start enumerate from 1 to match month indexing
    for idx, month in enumerate(month_list, start=1):
        print(f"Searching in {str(year)}-{month}:")

        # Skip unnecessary months:
        if any([
            year == 2011 and month in month_list[:6],
            year == 2026 and month in month_list[2:]
        ]):
            print("Date outside the period of interest.")
            continue

        # Start date
        start_date = datetime.datetime(year, idx, 1).isoformat() + "Z"

        # End date
        if month == 12:
            end_date = datetime.datetime(year + 1, 1, 1).isoformat() + "Z"
        else:
            end_date = datetime.datetime(year, idx + 1, 1).isoformat() + "Z"

        # Init list with month video counts
        month_count_list = []
        counter += 1

        for code, country in countries_dict.items():
            print(f"--Searching {country}:")
            try:
                # 'type="video"' ensures we ignore playlists and channels
                request = youtube.search().list(
                    q=TOPIC_QUERY,
                    part="snippet",
                    type="video",
                    regionCode=code,
                    publishedAfter=start_date,
                    publishedBefore=end_date,
                    maxResults=1
                )
                response = request.execute()

                # totalResults provides the estimated count for the query
                count = response.get("pageInfo", {}).get("totalResults", 0)
                month_count_list.append(count)
                print(f"    Estimate of total videos: {count}")

                # Small sleep to prevent rate limit flags
                time.sleep(0.2)

            except Exception as e:
                raise InterruptedError(f"Check Quota or Error: {e}")

        # Store month results
        counts_list.append(month_count_list)

        # Export intermediate data
        index = date_range[start:counter]
        int_df = pd.DataFrame(counts_list, columns=df_columns, index=index)
        int_df.to_csv(f"youtube_trends_interm_{counter}.csv", index=True)

# Create final dataframe
df = pd.DataFrame(
    counts_list,
    columns=df_columns,
    index=date_range[start:counter]
)

df.index.name = "Date"

# Export final dataset as CSV
output_csv = "youtube_trends.csv"
df.to_csv(output_csv, index=True)
print(f"Saved to {output_csv}")
