import csv
import os
import sys
from datetime import datetime

# date range (inclusive)
START_YEAR = 2011
START_MONTH = 7
END_YEAR = 2022
END_MONTH = 12

# set to 'exclude' to exclude unrecognised countries
# set to 'others' to put unrecognised countries into 'Others' category
# set to 'unknown' to put unrecognised countries into '?' category
UNRECOGNIZED_COUNTRY_HANDLING = 'exclude'

# scanner mode:
# - 'hackmageddon': use the hackmageddon keyword scanner
# - 'enhanced': use broader keyword matching
# - 'vcdb_categories': tally vcdb attack varieties directly
# - 'vcdb_mapped': map vcdb varieties to hackmageddon categories
SCANNER_MODE = 'enhanced'

# for keyword modes, try mapping 'Others' rows via VCDB_VARIETY_TO_ORIGINAL_MAP
MAP_OTHERS_WITH_VCDB = True

# set to True to output in transposed format (periods as rows)
OUTPUT_TRANSPOSED = True

INPUT_FILE = 'vcdb.csv'

OUTPUT_FILE = ''  # if not specified: vcdb_<SCANNER_MODE>_yyyymmdd-nn.csv

# set to True to filter out NON_RANDOM_SUBSOURCES
EXCLUDE_NON_RANDOM_SUBSOURCES = True

# set values to include only specific sub sources, or keep empty for all
ALLOWED_SUB_SOURCES = []

# confidence filtering
ALLOWED_CONFIDENCE = ['High', 'Medium', 'Low']
EXCLUDE_UNKNOWN_CONFIDENCE = False

# incident confirmation filtering
ALLOWED_CONFIRMATION = ['Confirmed', 'Suspected', 'Near miss']
EXCLUDE_UNKNOWN_CONFIRMATION = False

# ====== END CONFIGURATION ======

# valid countries list (from hackmageddon script)
countries = ['US','GB','CA','AU','UA','RU','FR','DE','BR','CN','JP','PK',
             'KP','KR','IN','TW','NL','ES','SE','MX','IR','IL','SA','SY',
             'FI','IE','AT','NO','CH','IT','MY','EG','TR','PT','PS','AE','?','ALL']

# attack types (hackmageddon categories)
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

NON_RANDOM_SUBSOURCES = {'phidbr', 'priority'}

INCIDENT_CONFIRMATION_COLUMNS = [
    ('security_incident.Confirmed', 'Confirmed'),
    ('security_incident.Suspected', 'Suspected'),
    ('security_incident.Near miss', 'Near miss'),
    ('security_incident.False positive', 'False positive'),
]

CONFIDENCE_COLUMNS = [
    ('confidence.High', 'High'),
    ('confidence.Medium', 'Medium'),
    ('confidence.Low', 'Low'),
    ('confidence.None', 'None'),
]

# vcdb variety to hackmageddon attack mapping
# THIS IS JUST A ROUGH GUESS - NEEDS REVIEW
VCDB_VARIETY_TO_ORIGINAL_MAP = {
    "hacking_dos": "DDoS",
    "hacking_sqli": "SQL Injection",
    "hacking_brute_force": "Brute Force Attack",
    "hacking_use_of_backdoor_or_c2": "Backdoor",
    "hacking_session_riding_and_hijacking": "Account Hijacking",
    "hacking_xss": "XSS",
    "hacking_defacement": "Defacement",
    "hacking_vulnerability": "Vulnerability",
    "hacking_zero_day": "Zero-day",
    "malware_ransomware": "Ransomware",
    "malware_trojan_or_backdoor": "Trojan",
    "malware_backdoor": "Backdoor",
    "malware_adware": "Adware",
    "malware_bot": "Botnet",
    "malware_worm": "Worms",
    "malware_keylogger": "Spyware",
    "malware_spyware_or_monitoring_software": "Spyware",
    "social_phishing": "Phishing",
    "social_pretexting": "Phishing",
}


def get_output_filename(configured_name, suffix):
    """generate output filename if not configured."""
    if configured_name and configured_name.strip():
        return configured_name.strip()

    date_part = datetime.now().strftime('%Y%m%d')
    seq = 1
    while True:
        candidate = f"vcdb_{suffix}_{date_part}-{seq:02d}.csv"
        if not os.path.exists(candidate):
            return candidate
        seq += 1


def normalize_variety_token(value):
    token = value.strip().lower().replace(' ', '_').replace('-', '_')
    while '__' in token:
        token = token.replace('__', '_')
    return token.strip('_')


def is_true(value):
    return str(value).strip().lower() in {'true', '1', 'yes', 'y', 't'}


def extract_confidences(row):
    return {label for col, label in CONFIDENCE_COLUMNS if is_true(row.get(col, ''))}


def extract_confirmations(row):
    return {label for col, label in INCIDENT_CONFIRMATION_COLUMNS if is_true(row.get(col, ''))}


def parse_incident_year_month(row):
    """parse incident year/month from supported vcdb date columns."""
    year_str = str(
        row.get('incident.year', '') or row.get('timeline.incident.year', '')
    ).strip()
    month_str = str(
        row.get('incident.month', '') or row.get('timeline.incident.month', '')
    ).strip()

    if not year_str or not month_str:
        return None, None

    try:
        year = int(year_str)
        month = int(month_str)
    except (ValueError, TypeError):
        return None, None

    if month < 1 or month > 12:
        return None, None

    return year, month


def row_passes_filters(row, stats):
    """apply sub source, confidence, and confirmation filters."""
    sub_source = str(row.get('plus.sub_source', '')).strip()

    if EXCLUDE_NON_RANDOM_SUBSOURCES and sub_source.lower() in NON_RANDOM_SUBSOURCES:
        stats['dropped_non_random'] += 1
        return False

    if ALLOWED_SUB_SOURCES and sub_source not in set(ALLOWED_SUB_SOURCES):
        stats['dropped_sub_source_allowlist'] += 1
        return False

    row_confidences = extract_confidences(row)
    allowed_confidence = set(ALLOWED_CONFIDENCE)
    if allowed_confidence:
        if not row_confidences and EXCLUDE_UNKNOWN_CONFIDENCE:
            stats['dropped_confidence'] += 1
            return False
        if row_confidences and not (row_confidences & allowed_confidence):
            stats['dropped_confidence'] += 1
            return False

    row_confirmations = extract_confirmations(row)
    allowed_confirmation = set(ALLOWED_CONFIRMATION)
    if allowed_confirmation:
        if not row_confirmations and EXCLUDE_UNKNOWN_CONFIRMATION:
            stats['dropped_confirmation'] += 1
            return False
        if row_confirmations and not (row_confirmations & allowed_confirmation):
            stats['dropped_confirmation'] += 1
            return False

    return True


def detect_attacks_hackmageddon(text):
    """original hackmageddon-style keyword scanner."""
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
    """enhanced scanner with broader matching for VCDB data."""
    detected = set(detect_attacks_hackmageddon(text))

    # ddos variants
    if 'DOS ATTACK' in text or 'DENIAL-OF-SERVICE' in text or 'TRAFFIC FLOOD' in text:
        detected.add('DDoS')

    # phishing variants
    if 'SPEAR PHISH' in text or 'SMISHING' in text or 'VISHING' in text or 'SOCIAL ENGINEER' in text:
        detected.add('Phishing')

    # credential attacks
    if 'PASSWORD SPRAY' in text or 'CREDENTIAL STUFF' in text:
        detected.add('Password Attack')

    # account takeover
    if 'ACCOUNT TAKEOVER' in text or 'UNAUTHORIZED ACCOUNT ACCESS' in text or 'CREDENTIAL THEFT' in text:
        detected.add('Account Hijacking')

    # trojan/RAT
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
    """dispatch to appropriate scanner based on mode."""
    if scanner_mode == 'hackmageddon':
        return detect_attacks_hackmageddon(text)
    if scanner_mode == 'enhanced':
        return detect_attacks_enhanced(text)
    raise ValueError(
        f"unsupported SCANNER_MODE '{scanner_mode}'. "
        "use 'hackmageddon', 'enhanced', 'vcdb_categories', or 'vcdb_mapped'."
    )


def parse_action_varieties(fieldnames):
    """extract action variety field names and normalize them."""
    variety_columns = {}
    variety_map_keys = {}
    varieties_by_action = {}

    for col in fieldnames:
        if not col.startswith('action.') or '.variety.' not in col:
            continue

        parts = col.split('.')
        if len(parts) < 4:
            continue

        action_name = parts[1]
        variety_name = parts[3]

        # normalize to action_variety format for output labels
        action_normalized = action_name.replace('_', ' ').title().replace(' ', '')
        variety_normalized = variety_name.replace('_', ' ').title().replace(' ', '')
        label = f"{action_normalized}_{variety_normalized}"
        map_key = f"{normalize_variety_token(action_name)}_{normalize_variety_token(variety_name)}"

        variety_columns[col] = label
        variety_map_keys[col] = map_key
        if label not in varieties_by_action:
            varieties_by_action[label] = col

    return variety_columns, variety_map_keys, sorted(set(varieties_by_action.keys()))


def extract_true_variety_labels(row, granular_variety_cols):
    out = set()
    for col, label in granular_variety_cols.items():
        val = row.get(col, '').strip().lower()
        if val in {'true', '1', 'yes', 'y', 't'}:
            out.add(label)
    return out


def extract_mapped_attacks(row, variety_map_keys):
    mapped = set()
    for col, map_key in variety_map_keys.items():
        val = row.get(col, '').strip().lower()
        if val in {'true', '1', 'yes', 'y', 't'} and map_key in VCDB_VARIETY_TO_ORIGINAL_MAP:
            mapped.add(VCDB_VARIETY_TO_ORIGINAL_MAP[map_key])
    return mapped


def increment_attack_counts(counter, attack_name, country, date_key):
    counter[attack_name + '-' + country][date_key] += 1
    counter[attack_name + '-ALL'][date_key] += 1


# main processing

c = dict()
periods = []
unrecognised_countries = set()

if SCANNER_MODE in {'hackmageddon', 'enhanced', 'vcdb_mapped'}:
    output_attacks = list(attacks)
elif SCANNER_MODE == 'vcdb_categories':
    output_attacks = []
else:
    raise ValueError(
        f"unsupported SCANNER_MODE '{SCANNER_MODE}'. "
        "use 'hackmageddon', 'enhanced', 'vcdb_categories', or 'vcdb_mapped'."
    )

# initialise attack-country counters
for attack in output_attacks:
    for country in countries:
        key = attack + '-' + country
        c[key] = dict()
        for year in range(START_YEAR, END_YEAR + 1):
            for month in months:
                # skip months outside the configured range
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

OUTPUT_FILE = get_output_filename(OUTPUT_FILE, SCANNER_MODE)

print(f"processing {INPUT_FILE}...")
print(f"scanner mode: {SCANNER_MODE}")
print(f"country handling: {UNRECOGNIZED_COUNTRY_HANDLING}")
if SCANNER_MODE in {'hackmageddon', 'enhanced'}:
    print(f"map others with vcdb: {MAP_OTHERS_WITH_VCDB}")
print(f"exclude non-random sub sources: {EXCLUDE_NON_RANDOM_SUBSOURCES}")
print(f"allowed sub sources: {ALLOWED_SUB_SOURCES if ALLOWED_SUB_SOURCES else 'ALL'}")
print(f"allowed confidence: {ALLOWED_CONFIDENCE if ALLOWED_CONFIDENCE else 'ALL'}")
print(f"allowed confirmation: {ALLOWED_CONFIRMATION if ALLOWED_CONFIRMATION else 'ALL'}")

# process csv
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as datafile:
        reader = csv.DictReader(datafile)
        fieldnames = reader.fieldnames or []
        data = list(reader)
except Exception as e:
    print(f"error reading input file: {e}", file=sys.stderr)
    sys.exit(1)

print(f"data has {len(data)} rows")

# extract granular variety columns if needed
granular_variety_cols = {}
variety_map_keys = {}
granular_varieties = []
if SCANNER_MODE in {'vcdb_categories', 'vcdb_mapped'} or MAP_OTHERS_WITH_VCDB:
    granular_variety_cols, variety_map_keys, granular_varieties = parse_action_varieties(fieldnames)

if SCANNER_MODE == 'vcdb_categories':
    output_attacks = list(granular_varieties)

    # initialise granular counters for direct vcdb categories output
    for variety in granular_varieties:
        for country in countries:
            key = variety + '-' + country
            c[key] = dict()
            for year in range(START_YEAR, END_YEAR + 1):
                for month in months:
                    if year == START_YEAR and int(month) < START_MONTH:
                        continue
                    if year == END_YEAR and int(month) > END_MONTH:
                        continue
                    date = month + '/' + str(year)
                    c[key][date] = 0

# extract victim country columns
victim_country_columns = [col for col in fieldnames if col.startswith('victim.country.')]

stats = {
    'rows_processed': 0,
    'rows_no_date': 0,
    'rows_no_country': 0,
    'rows_no_attack': 0,
    'rows_no_mapped': 0,
    'rows_no_variety': 0,
    'dropped_non_random': 0,
    'dropped_sub_source_allowlist': 0,
    'dropped_confidence': 0,
    'dropped_confirmation': 0,
    'unknown_countries': set()
}

# process rows (incidents)
for idx, row in enumerate(data):
    try:
        # parse year/month from supported date fields
        year, month = parse_incident_year_month(row)
        if year is None or month is None:
            stats['rows_no_date'] += 1
            continue

        # skip if outside date range
        if year < START_YEAR or year > END_YEAR:
            stats['rows_no_date'] += 1
            continue
        if year == START_YEAR and month < START_MONTH:
            stats['rows_no_date'] += 1
            continue
        if year == END_YEAR and month > END_MONTH:
            stats['rows_no_date'] += 1
            continue

        # apply vcdb row filters from the original script
        if not row_passes_filters(row, stats):
            continue

        # format date for matching
        month_formatted = f"{month:02d}"
        date_key = month_formatted + '/' + str(year)

        # extract victim countries
        victim_countries = []
        for col in victim_country_columns:
            country_code = col.split('.')[-1]

            # check if column is true (marked)
            val = row.get(col, '').strip().lower()
            if val in {'true', '1', 'yes', 'y', 't'}:
                # check if country is valid
                if country_code not in countries and country_code != 'Unknown':
                    stats['unknown_countries'].add(country_code)
                    # handle unrecognized country based on configuration
                    if UNRECOGNIZED_COUNTRY_HANDLING == 'exclude':
                        continue
                    elif UNRECOGNIZED_COUNTRY_HANDLING == 'others':
                        # skip if no 'Others' equivalent in hackmageddon format
                        continue
                    elif UNRECOGNIZED_COUNTRY_HANDLING == 'unknown':
                        victim_countries.append('?')
                        continue

                if country_code == 'Unknown':
                    victim_countries.append('?')
                else:
                    victim_countries.append(country_code)

        # if no countries extracted, mark as unknown
        if not victim_countries:
            victim_countries = ['?']
            stats['rows_no_country'] += 1

        stats['rows_processed'] += 1

        # normalise text for attack detection
        summary = row.get('summary', '').strip()
        notes = row.get('notes', '').strip()
        text = (summary + ' ' + notes).upper().replace("-", " ")

        if SCANNER_MODE in {'hackmageddon', 'enhanced'}:
            # detect attacks from text scanner
            detected_attacks = detect_attacks(text, SCANNER_MODE)
            if not detected_attacks:
                detected_attacks = {'Others'}
                stats['rows_no_attack'] += 1

            # optionally remap Others using vcdb variety mapping
            if MAP_OTHERS_WITH_VCDB and 'Others' in detected_attacks:
                mapped_attacks = extract_mapped_attacks(row, variety_map_keys)
                if mapped_attacks:
                    detected_attacks.discard('Others')
                    detected_attacks.update(mapped_attacks)

        elif SCANNER_MODE == 'vcdb_categories':
            # use vcdb categories directly from action.variety columns
            detected_attacks = extract_true_variety_labels(row, granular_variety_cols)
            if not detected_attacks:
                stats['rows_no_variety'] += 1
                continue

        elif SCANNER_MODE == 'vcdb_mapped':
            # use vcdb variety map to produce hackmageddon categories
            detected_attacks = extract_mapped_attacks(row, variety_map_keys)
            if not detected_attacks:
                detected_attacks = {'Others'}
                stats['rows_no_mapped'] += 1
        else:
            raise ValueError(
                f"unsupported SCANNER_MODE '{SCANNER_MODE}'. "
                "use 'hackmageddon', 'enhanced', 'vcdb_categories', or 'vcdb_mapped'."
            )

        for attack in detected_attacks:
            for country in victim_countries:
                increment_attack_counts(c, attack, country, date_key)

    except Exception as e:
        print(f"warning: error processing row {idx + 1}: {e}", file=sys.stderr)
        continue

# write output csv for selected mode
print(f"writing output to {OUTPUT_FILE}...")

ordered_keys = [attack + '-' + country for attack in output_attacks for country in countries]

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
    print(f"error writing output file: {e}", file=sys.stderr)
    sys.exit(1)

# report summary
if stats['unknown_countries']:
    print(f"\nunrecognised country codes found: {sorted(stats['unknown_countries'])}")
    if UNRECOGNIZED_COUNTRY_HANDLING == 'unknown':
        print(f"these were grouped into '?'")
    elif UNRECOGNIZED_COUNTRY_HANDLING == 'exclude':
        print(f"these were excluded from the analysis")
else:
    print(f"\nno unrecognised country codes found.")

print(f"\nprocessing complete.")
print(f"  rows processed: {stats['rows_processed']}")
print(f"  rows with no date: {stats['rows_no_date']}")
print(f"  rows with no country: {stats['rows_no_country']}")
print(f"  dropped non-random sub source: {stats['dropped_non_random']}")
print(f"  dropped sub source allow-list: {stats['dropped_sub_source_allowlist']}")
print(f"  dropped by confidence: {stats['dropped_confidence']}")
print(f"  dropped by confirmation: {stats['dropped_confirmation']}")
if SCANNER_MODE in {'hackmageddon', 'enhanced'}:
    print(f"  rows with no keyword attack detected: {stats['rows_no_attack']}")
if SCANNER_MODE == 'vcdb_categories':
    print(f"  rows with no vcdb variety detected: {stats['rows_no_variety']}")
if SCANNER_MODE == 'vcdb_mapped':
    print(f"  rows with no vcdb mapping (fell back to Others): {stats['rows_no_mapped']}")
print(f"  output: {OUTPUT_FILE}")
