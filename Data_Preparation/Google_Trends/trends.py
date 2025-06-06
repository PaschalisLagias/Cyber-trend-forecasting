from itertools import product
from datetime import datetime

import pandas as pd
import pytrends
from pytrends.request import TrendReq
# TODO: Downgrade to Python 3.8


countries = [
    "US", "GB", "CA", "AU", "UA", "RU", "FR", "DE", "BR", "CN2", "JP", "PK",
    "KP", "KR", "IN", "TW", "NL", "ES", "SE", "MX", "IR", "IL", "SA", "SY",
    "FI", "IE", "AT", "NO", "CH", "IT", "MY", "EG", "TR", "PT", "PS", "AE"
]

keywords = [
    "war",
    "war conflict",
    "armed force attack"
]

time_window = "2017-01-01 2022-12-31"
trends_requester = TrendReq(hl="en-US", tz=0)
output = pd.DataFrame()  # Output dataframe

for country, keyword in product(countries, keywords):
    print(f"Fetching {keyword} for {country}...")
    try:
        trends_requester.build_payload(
            kw_list=[keyword],
            timeframe=time_window,
            geo=country
        )
        df = trends_requester.interest_over_time()
        if len(df) > 0:
            df = df.resample("M").mean()  # Monthly average score
            col_name = f"{keyword.upper()}_{country}"
            df = df.rename(columns={keyword: col_name})
    except Exception as e:
        print(f"Failure for {keyword} in {country} - {e}")
