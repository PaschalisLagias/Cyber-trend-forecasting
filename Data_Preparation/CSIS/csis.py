"""
CSIS data preparation script
--------------------------------------------------------------------------------
This script processes CSIS cyber incident data and generates a monthly number of
incidents (NoI) aggregation similar to the Hackmageddon data preparation.

The CSIS significant cyber events list (e.g. `260306_Cyber_Events.pdf`) 
is available for download here:
https://www.csis.org/programs/strategic-technologies-program/significant-cyber-incidents

The PDF must be converted to text (e.g. `pdftotext` or Acrobat) and then
manually converted to CSV:
- Remove introductory text, blank lines, header/footers.
- Find and replace date strings at the start of lines
  (e.g. 'July 2020. ' to '2020-07,"'). Note that October 2015 and November 2015
  are out of order in the PDF, and that some year/month strings also appear in
  the incident descriptions.
- Add " to end of incident descriptions and remove trailing spaces.
- Manually identify the countries involved in each incident (3rd column of CSV)
   - List countries by 2 letter country code, separated
     by a dash e.g. "DK-SE-IR-CN"
   - Where countries are unknown or unspecific, use "?". If some countries are
     specified and others are unknown, list the known countries
     and add "?" e.g. "US-GB-?":
   - Where the row should be ignored (e.g. it does not 
     describe an incident) use "ignore"
   - Where the row describes an international incident,
     use >1. The script will replace this with a list of all countries.

Attack identification and country categorization follows the same methodology
as the Hackmageddon processing scripts, however there are additional options:

- Unrecognised country codes (those not appearing in the `countries` list) can
  be `include`d, `excluded`, or considered `unknown` and added
  to the ? category.
- The keyword matching (to identify attack types) can use the original 
  `hackmageddon` matches, or an `enhanced` keyword list.

--------------------------------------------------------------------------------
2026-04-04 BG
After discussion with Paschalis, the following settings were used:
- In the input CSV file, incidents that describe regions or groups of countries
  (e.g. Middle East, NATO) these are been treated as "?" rather than listing
  all countries in the region/org.
- UNRECOGNISED_COUNTRY_HANDLING = 'unknown'
- SCANNER_MODE = 'enhanced'
- 11 input rows do not have specific dates, these are ignored by the script.
- 6 rows don't describe incidents and have been manually flagged to be ignored.

"""

import csv
import os
from datetime import datetime
import sys

# date range (inclusive)
START_YEAR = 2006
START_MONTH = 4
END_YEAR = 2025
END_MONTH = 12

# set to 'include' to keep unrecognised countries as their own output codes
# set to 'exclude' to exclude unrecognised countries
# set to 'unknown' to put unrecognised countries into '?' category
UNRECOGNISED_COUNTRY_HANDLING = 'unknown'

INPUT_FILE = 'csis_260306.csv'

OUTPUT_FILE = ''  # if not specified: csis_output_yyyymmdd-nn.csv

OUTPUT_TRANSPOSED = True  # True = periods as row

# scanner mode:
# - 'hackmageddon' keeps the original hackmageddon-style keyword scanner
# - 'enhanced' uses broader matching to reduce incidents classified as Others
SCANNER_MODE = 'enhanced'


# valid countries list (same as Hackmageddon script)
countries = ['US','GB','CA','AU','UA','RU','FR','DE','BR','CN','JP','PK',
             'KP','KR','IN','TW','NL','ES','SE','MX','IR','IL','SA','SY',
             'FI','IE','AT','NO','CH','IT','MY','EG','TR','PT','PS','AE','?','ALL']

# Attack types
attacks = [
    'DDoS',
    'Phishing',
    'Ransomware',
    'Password Attack',
    'SQL Injection',
    'Account Hijacking',
    'Defacement',
    'Trojan',
    'Vulnerability',
    'Zero-day',
    'Advanced persistent threat',
    'XSS',
    'Malware',
    'Data Breach',
    'Disinformation/Misinformation',
    'Targeted Attack',
    'Adware',
    'Brute Force Attack',
    'Malvertising',
    'Backdoor',
    'Botnet',
    'Cryptojacking',
    'Worms',
    'Spyware',
    'Unknown',
    'Others'
]

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
malwares = ['MALWARE', 'WIPER', 'TROJAN', 'DROPPER', 'SPYWARE', 'PEGASUS', 'MALVERTIS',
            'RANSOMWARE', 'VIRUS', 'WORM', 'KEYLOG', 'KEYSTROKE LOG', 'MALICIOUS CODE',
            'MALICIOUS SOFTWARE', 'ADWARE', 'ROOTKIT', 'BOT', 'BACKDOOR']


def get_output_filename(configured_name):
    if configured_name and configured_name.strip():
        return configured_name.strip()

    date_part = datetime.now().strftime('%Y%m%d')
    seq = 1
    while True:
        candidate = f"csis_output_{date_part}-{seq:02d}.csv"
        if not os.path.exists(candidate):
            return candidate
        seq += 1


def detect_attacks_hackmageddon(text):
    detected = set()

    if 'DDOS' in text or 'DENIAL OF SERVICE' in text:
        detected.add('DDoS')
    if 'PHISHING' in text:
        detected.add('Phishing')
    if 'RANSOMWARE' in text:
        detected.add('Ransomware')
    if 'PASSWORD' in text:
        detected.add('Password Attack')
    if 'SQLI' in text or 'SQL I' in text:
        detected.add('SQL Injection')
    if 'ACCOUNT HIJACK' in text or 'ACCOUNT TAKE' in text:
        detected.add('Account Hijacking')
    if 'DEFACE' in text:
        detected.add('Defacement')
    if 'TROJAN' in text or 'DROPPER' in text:
        detected.add('Trojan')
    if 'VULNERABILITY' in text or '0 DAY' in text or 'ZERO DAY' in text:
        detected.add('Vulnerability')
    if '0 DAY' in text or 'ZERO DAY' in text:
        detected.add('Zero-day')
    if 'APT' in text or 'ADVANCED PERSISTENT THREAT' in text:
        detected.add('Advanced persistent threat')
    if 'XSS' in text or 'CROSS SITE SCRIPT' in text:
        detected.add('XSS')
    if any(malware_keyword in text for malware_keyword in malwares):
        detected.add('Malware')
    if 'BREACH' in text or 'LEAK' in text or 'SPILL' in text or 'EXPOSE' in text:
        detected.add('Data Breach')
    if 'DISINFORMATION' in text or 'MISINFORMATION' in text or 'FALSE INFORMATION' in text or 'MISLEADING' in text:
        detected.add('Disinformation/Misinformation')
    if 'TARGETED ATTACK' in text:
        detected.add('Targeted Attack')
    if 'ADWARE' in text:
        detected.add('Adware')
    if 'BRUTE FORCE' in text:
        detected.add('Brute Force Attack')
    if 'MALVERTIS' in text:
        detected.add('Malvertising')
    if 'BACKDOOR' in text:
        detected.add('Backdoor')
    if 'BOTNET' in text:
        detected.add('Botnet')
    if 'CRYPTOJACK' in text or 'CRYPTO JACK' in text:
        detected.add('Cryptojacking')
    if 'WORM' in text:
        detected.add('Worms')
    if 'SPYWARE' in text:
        detected.add('Spyware')
    if 'UNKNOWN' in text:
        detected.add('Unknown')

    return detected


def detect_attacks_enhanced(text):
    """Enhanced scanner with broader CSIS-oriented matching."""
    detected = set(detect_attacks_hackmageddon(text))

    # ddos
    if 'DOS ATTACK' in text or 'DENIAL-OF-SERVICE' in text or 'TRAFFIC FLOOD' in text:
        detected.add('DDoS')

    # phishing
    if 'SPEAR PHISH' in text or 'SMISHING' in text or 'VISHING' in text or 'SOCIAL ENGINEER' in text:
        detected.add('Phishing')

    # credential attacks
    if 'PASSWORD SPRAY' in text or 'CREDENTIAL STUFF' in text:
        detected.add('Password Attack')

    # SQL injection variants
    # none

    # account takeover
    if 'ACCOUNT TAKEOVER' in text or 'UNAUTHORIZED ACCOUNT ACCESS' in text or 'CREDENTIAL THEFT' in text:
        detected.add('Account Hijacking')

    # trojan/remote access trojan
    if ' RAT,' in text or ' RAT ' in text or ' RAT.' in text:
        detected.add('Trojan')

    # vulnerability exploitation
    if 'EXPLOIT' in text or 'CVE-' in text or 'UNPATCHED' in text:
        detected.add('Vulnerability')

    # apt/espionage
    if 'ESPIONAGE' in text or 'CYBER ESPIONAGE' in text or 'NATION STATE' in text or 'STATE-SPONSORED' in text:
        detected.add('Advanced persistent threat')

    # malware
    if 'MALICIOUS' in text or 'WIPER' in text or 'KEYLOGGER' in text or 'ROOTKIT' in text:
        detected.add('Malware')

    # breach / intrusion / exfiltration language
    if ('INTRUSION' in text or 'COMPROMISE' in text or 'EXFILTRAT' in text or
            'STOLEN DATA' in text or 'DATA THEFT' in text or 'UNAUTHORIZED ACCESS' in text):
        detected.add('Data Breach')

    # disinformation variants
    if 'INFLUENCE OPERATION' in text or 'PROPAGANDA' in text or 'DEEPFAKE' in text:
        detected.add('Disinformation/Misinformation')

    # targeted campaigns
    if 'TARGETED' in text and ('CAMPAIGN' in text or 'SECTOR' in text or 'ORGANIZATION' in text):
        detected.add('Targeted Attack')

    # backdoor/botnet variants
    if 'WEB SHELL' in text:
        detected.add('Backdoor')

    if 'BOT HERDER' in text:
        detected.add('Botnet')

    # cryptojacking
    if 'CRYPTO MINING' in text:
        detected.add('Cryptojacking')

    return detected


def detect_attacks(text, scanner_mode):
    if scanner_mode == 'hackmageddon':
        return detect_attacks_hackmageddon(text)
    if scanner_mode == 'enhanced':
        return detect_attacks_enhanced(text)
    raise ValueError(f"Unsupported SCANNER_MODE '{scanner_mode}'. Use 'hackmageddon' or 'enhanced'.")


def increment_attack_counts(counter, attack_name, country, date_key):
    counter[attack_name + '-' + country][date_key] += 1
    counter[attack_name + '-ALL'][date_key] += 1


def initialise_country_counter(counter, country, periods):
    for attack in attacks:
        key = attack + '-' + country
        if key not in counter:
            counter[key] = {period: 0 for period in periods}


# ------------------------------------------------------------------------------

c = dict()
periods = []
unrecognised_countries = set()
known_countries = set(countries)
included_unrecognised_countries = set()

# initialise attack-country counters
for attack in attacks:
    for country in countries:
        key = attack + '-' + country
        c[key] = dict()
        for year in range(START_YEAR, END_YEAR + 1):
            for month in months:
                # Skip months outside the configured range
                if year == START_YEAR and int(month) < START_MONTH:
                    continue
                if year == END_YEAR and int(month) > END_MONTH:
                    continue

                date = month + '/' + str(year)
                c[key][date] = 0
                if date not in periods:
                    periods.append(date)

# sort periods
periods.sort(key=lambda x: (int(x.split('/')[1]), int(x.split('/')[0])))

OUTPUT_FILE = get_output_filename(OUTPUT_FILE)

print(f"Processing {INPUT_FILE}...")

# process csv
try:
    with open(INPUT_FILE, 'r', encoding='latin-1') as datafile:
        reader = csv.DictReader(datafile)
        data = list(reader)
except Exception as e:
    print(f"Error reading input file: {e}", file=sys.stderr)
    sys.exit(1)

print(f"Data has {len(data)} rows")
print(f"Scanner mode: {SCANNER_MODE}")

# process rows (incidents)
for idx, row in enumerate(data):
    try:
        date_str = row.get('date', '').strip()
        summary = row.get('summary', '').strip()
        countries_str = row.get('countries', '').strip()

        # skip empty rows
        if not date_str or not summary or not countries_str:
            continue

        # skip lines with 'ignore' country code
        if countries_str.lower() == 'ignore':
            continue

        # parse YYYY-MM date
        try:
            date_parts = date_str.split('-')
            year = int(date_parts[0])
            month = int(date_parts[1])

            # skip if outside date range
            if year < START_YEAR or year > END_YEAR:
                continue
            if year == START_YEAR and month < START_MONTH:
                continue
            if year == END_YEAR and month > END_MONTH:
                continue

            # format date for matching
            month_str = f"{month:02d}"
            date_key = month_str + '/' + str(year)
        except (IndexError, ValueError):
            print(f"Warning: Could not parse date '{date_str}' on row {idx + 2}")
            continue

        # normalise text
        text = (summary).upper().replace("-", " ")

        # parse countries for incident
        if '>1' in countries_str:
            country_list=countries[:-1]                                         # excluding 'ALL'
        else:
            # split hypenated list of countries
            country_list = [country.strip().upper() for country in countries_str.split('-')]

        # process countries found
        for country in country_list:
            # check if country is valid
            if country not in known_countries:
                unrecognised_countries.add(country)
                # handle unrecognised country based on configuration
                if UNRECOGNISED_COUNTRY_HANDLING == 'exclude':
                    continue
                elif UNRECOGNISED_COUNTRY_HANDLING == 'include':
                    included_unrecognised_countries.add(country)
                    known_countries.add(country)
                    initialise_country_counter(c, country, periods)
                elif UNRECOGNISED_COUNTRY_HANDLING == 'unknown':
                    country = '?'
                else:
                    continue

            detected_attacks = detect_attacks(text, SCANNER_MODE)

            if detected_attacks:
                for attack_name in detected_attacks:
                    increment_attack_counts(c, attack_name, country, date_key)
            else:
                increment_attack_counts(c, 'Others', country, date_key)

    except Exception as e:
        print(f"Error processing row {idx + 2}: {e}", file=sys.stderr)
        continue

# write output CSV
print(f"Writing output to {OUTPUT_FILE}...")

if included_unrecognised_countries:
    output_countries = countries[:-1] + sorted(included_unrecognised_countries) + ['ALL']
else:
    output_countries = countries

ordered_keys = [attack + '-' + country for attack in attacks for country in output_countries]

try:
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        if OUTPUT_TRANSPOSED:
            fields = ['Period'] + ordered_keys
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for period in periods:
                row = {'Period': period}
                for key in ordered_keys:
                    row[key] = c[key][period]
                writer.writerow(row)
        else:
            fields = ['Attack-Country'] + periods
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for key in ordered_keys:
                row = {'Attack-Country': key}
                row.update(c[key])
                writer.writerow(row)
except Exception as e:
    print(f"Error writing output file: {e}", file=sys.stderr)
    sys.exit(1)

# report unrecognised countries
if unrecognised_countries:
    print(f"\nUnrecognised country codes found: {sorted(unrecognised_countries)}")
    if UNRECOGNISED_COUNTRY_HANDLING == 'include':
        print(f"These were included as separate country code columns")
    elif UNRECOGNISED_COUNTRY_HANDLING == 'unknown':
        print(f"These were grouped into '?'")
    elif UNRECOGNISED_COUNTRY_HANDLING == 'exclude':
        print(f"These were excluded from the analysis")
else:
    print(f"\nNo unrecognised country codes found.")

print(f"\nProcessing complete. Output saved to {OUTPUT_FILE}")