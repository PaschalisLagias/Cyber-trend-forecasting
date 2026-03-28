# Replacing Hackmageddon dataset for cyber event forecasting
https://github.com/ehshan/Cyber-trend-forecasting

The Hackmageddon dataset has a gap from 2025-03-01 up to 2025-12-31, the author (Paolo Passeri) did not publish any timelines during this period though he has since resumed.

CSIS and VCDB datasets were proposed as alternatives. These folders contain instructions and scripts for processing those datasets into a compatible format as the Hackmageddon data.

## CSIS processing
1. Download Significant Cyber Events List (e.g. `260306_Cyber_Events.pdf`) from this page: https://www.csis.org/programs/strategic-technologies-program/significant-cyber-incidents
2. Convert PDF to text using Adobe Acrobat (`260306_Cyber_Events.txt`)
3. Manually convert text file to CSV (`csis_260306.csv`):
   - Remove introductory text, blank lines, header/footers.
   - Find and replace date strings at the start of lines (e.g. 'July 2020. ' to '2020-07,"').  
     Note that October 2015 and November 2015 are out of order in the PDF, and that some year/month strings are also included in the incident descriptions.
   - Add " to end of incident descriptions and remove trailing spaces.
4. Manually identify the countries involved in each incident (3rd column of CSV)
   - List countries by 2 letter country code, separated by a dash e.g. "DK-SE-IR-CN"
   - Where the country is unknown or unclear, use "?"
   - Where the row should be ignored (e.g. it does not describe an incident) use "ignore"
   - Where the row describes an international incident, use >1. The script will replace this with a list of all countries.
5. Check the options in the `csis.py` then run it to produce the output file (e.g. `csis_output_20260326-01.csv`). This file contains data in the same format as the Hackmageddon scripts.

## VCDB
Source data is available here: 
https://verisframework.org/vcdb.html
https://github.com/vz-risk/VCDB/tree/master/data/csv/vcdb.csv.zip

The VCDB data is too large for GitHub storage, so I have not uploaded (could split into separate files and combine in script).

No pre/manual processing of the CSV is necessary before running the script. The script has similar options to the CSIS script (Hackmageddon style keyword matching), but also includes extra options for reducing the number of "Others" categorisations by attempting to map them based on the VCDB category. 

There is also an option to output the VCDB categories instead, but these are more granular than the original Hackmageddon categories.

Script filters on:
  - Security incident [Confirmed, Suspected] and ignore [False Positive, Near Miss]
  - Confidence [High, Medium, Low] and ignore [None]
  - The VCDB Github README mentions that some entries are not randomly chosen (healthcare and priority issues). A flag is set in the script to exclude these non-random entries.

Output file is `vcdb_{mode}_yyyymmdd-nn.csv`.

Two caveats noted on the VCDB site re: single vs multiple incidents

> It is sometimes difficult to determine the difference between multiple separate incidents and a single incident with multiple sequential events. For instance, the August 2013 attack against the New York Times could be seen as one multi-staged incident involving three victims (the Indian ISP, Melbourne IT, and NYT), but could also be considered three distinct, but related, incidents. In such cases, there is no “right or wrong” method, but data quality requires a consistent approach. Standard procedure with VERIS is to treat these as separate incidents that can be associated using the related_incidents field. As a rule, incidents in VERIS have only one primary victim.

> Mass defacements and DDoS/DoS attacks can make this single vs multiple incident determination particularly difficult. Such reports often state that “547 sites defaced/DoS'd by #Actor” with few other details. It's not possible (at least not without a lot of effort) to determine if that involves a single compromised hosting provider that affected 547 hosted sites, whether those sites represent 547 unique victims, etc. In such cases, record “Mass Event” for the victim_id. Leave country, employee_count, and industry blank unless they are shared by the entire set of victims/sites/etc (e.g., they are all Indonesian). Record the total number of sites (or whatever) defaced/DoS'd in the secondary_amount field. If specific sites or names are given, copy/paste those in the secondary_id feld. Then record whatever other details are given as normal.

## Comparison notebook
This was mostly written with Github Copilot because I am not very good with matplotlib. Basically normalising the datasets and comparing the country and incident distributions for Hackmageddon, CSIS and VCDB (for period 07/2011 to 12/2022).

The VCDB database is heavily distributed towards US incidents, and it looks like the CSIS dataset better aligns with the Hackmageddon data. Both CSIS + VCDB have far fewer entries than the Hackmageddon dataset.

## Questions
1. How to handle countries that are not in the original Hackmageddon script? The scripts have options for:
   - include (e.g. DDoS-AD, DDoS-EE, etc. will be categories in the output dataset)
   - include as 'other' (any incidents will be tallied under DDoS-?, Phishing-?, etc.)
   - exclude (do not track incidents against these countries)
2. How to handle incidents that explicitly names some countries, but not all?
   - e.g. "Europol disrupted a pro-Russian hacktivist group behind DDoS attacks on Ukraine and NATO countries. The operation led to arrests in France and Spain, as well as warrants for six Russian nationals." should this be "?" or "UA-?" or list UA and all NATO countries? 
   - e.g. "A Trojan nicknamed 'Mahdi' found gathering data from approximately 800 critical infrastructure engineering firms, government agencies, financial houses, and academia throughout the Middle East and beyond, predominantly in Israel and Iran. The virus contains Persian language strings." should this be "?", "IL-IR", or "IL-IR-?"
   - e.g. "Four Chinese nationals targeted companies, universities, and government entities in the United States and abroad between 2011 and 2018. The campaign focused on information of economic benefit to China's commercial sectors." should this be "?", "US", or "US-?"
3. How to handle incidents that describe regions?
     - "The United States, the European Union, NATO and other world powers released joint statements condemning the Chinese government for a series of malicious cyber activities. They attributed responsibility to China for the Microsoft Exchange hack from early 2021 and the compromise of more than 100,000 servers worldwide." Should this list the US and all EU+NATO members, just the "US", "US-?" or "?"
4. Do we want to use the exact same keyword matching as the original Hackmageddon script, or add additional keywords (to reduce the number of incidents classified as "others"). This is currently in the scripts as an "enhanced" flag.
