"""
Constants to be used in various parts of the project
"""
from pathlib import Path

BASE_FILEPATH = Path(__file__).resolve().parent.parent
# returns the base_path to the directory

MI_EXP_FILEPATH = BASE_FILEPATH / "data" / "raw" / "MI" / "Expenditure"

MI_CON_FILEPATH = BASE_FILEPATH / "data" / "raw" / "MI" / "Contribution"

AZ_TRANSACTIONS_FILEPATH = (
    BASE_FILEPATH / "data" / "raw" / "AZ" / "az_transactions_demo.csv"
)

AZ_INDIVIDUALS_FILEPATH = (
    BASE_FILEPATH / "data" / "raw" / "AZ" / "az_individuals_demo.csv"
)

AZ_ORGANIZATIONS_FILEPATH = BASE_FILEPATH / "data" / "raw" / "AZ" / "az_orgs_demo.csv"

USER_AGENT = """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
                (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"""

HEADERS = {"User-Agent": USER_AGENT}

MI_SOS_URL = "https://miboecfr.nictusa.com/cfr/dumpall/cfrdetail/"

MI_CONTRIBUTION_COLUMNS = [
    "doc_seq_no",
    "page_no",
    "contribution_id",
    "cont_detail_id",
    "doc_stmnt_year",
    "doc_type_desc",
    "com_legal_name",
    "common_name",
    "cfr_com_id",
    "com_type",
    "can_first_name",
    "can_last_name",
    "contribtype",
    "f_name",
    "l_name_or_org",
    "address",
    "city",
    "state",
    "zip",
    "occupation",
    "employer",
    "received_date",
    "amount",
    "aggregate",
    "extra_desc",
]

# MN State Cleaner Constants:
here = Path(__file__).resolve()
repo_root = here.parent.parent

MN_FILEPATHS_LST = [
    repo_root / "data" / "raw" / "MN" / "AG.csv",
    repo_root / "data" / "raw" / "MN" / "AP.csv",
    repo_root / "data" / "raw" / "MN" / "DC.csv",
    repo_root / "data" / "raw" / "MN" / "GC.csv",
    repo_root / "data" / "raw" / "MN" / "House.csv",
    repo_root / "data" / "raw" / "MN" / "SA.csv",
    repo_root / "data" / "raw" / "MN" / "SC.csv",
    repo_root / "data" / "raw" / "MN" / "Senate.csv",
    repo_root / "data" / "raw" / "MN" / "SS.csv",
    repo_root / "data" / "raw" / "MN" / "ST.csv",
    repo_root / "data" / "raw" / "MN" / "non_candidate_con.csv",
    repo_root / "data" / "raw" / "MN" / "independent_exp.csv",
]

MN_CANDIDATE_CONTRIBUTION_COL = [
    "OfficeSought",
    "CandRegNumb",
    "CandFirstName",
    "CandLastName",
    "DonationDate",
    "DonorType",
    "DonorName",
    "DonationAmount",
    "InKindDonAmount",
    "InKindDescriptionText",
]

MN_CANDIDATE_CONTRIBUTION_MAP = {
    "OfficeSought": "office_sought",
    "CandRegNumb": "recipient_id",
    "CandFirstName": "recipient_first_name",
    "CandLastName": "recipient_last_name",
    "DonationDate": "date",
    "DonorType": "donor_type",
    "DonorName": "donor_full_name",
    "DonationAmount": "amount",
    "InKindDonAmount": "inkind_amount",
    "InKindDescriptionText": "purpose",
}

MN_NONCANDIDATE_CONTRIBUTION_COL = [
    "PCFRegNumb",
    "Committee",
    "ETType",
    "DonationDate",
    "DonorType",
    "DonorRegNumb",
    "DonorName",
    "DonationAmount",
    "InKindDonAmount",
    "InKindDescriptionText",
]

MN_NONCANDIDATE_CONTRIBUTION_MAP = {
    "PCFRegNumb": "recipient_id",
    "Committee": "recipient_full_name",
    "ETType": "recipient_type",
    "DonationDate": "date",
    "DonorType": "donor_type",
    "DonorRegNumb": "donor_id",
    "DonorName": "donor_full_name",
    "DonationAmount": "amount",
    "InKindDonAmount": "inkind_amount",
    "InKindDescriptionText": "purpose",
}

MN_INDEPENDENT_EXPENDITURE_COL = [
    "Spender",
    "Spender Reg Num",
    "Spender type",
    "Affected Comte Name",
    "Affected Cmte Reg Num",
    "For /Against",
    "Date",
    "Type",
    "Amount",
    "Purpose",
    "Vendor State",
]

MN_INDEPENDENT_EXPENDITURE_MAP = {
    "Spender": "donor_full_name",
    "Spender Reg Num": "donor_id",
    "Spender type": "donor_type",
    "Affected Comte Name": "recipient_full_name",
    "Affected Cmte Reg Num": "recipient_id",
    "Date": "date",
    "Amount": "amount",
    "Purpose": "purpose",
    "Type": "transaction_type",
    "Vendor State": "state",
}

MN_RACE_MAP = {
    "GC": "Governor",
    "AG": "Attorney General",
    "SS": "Secretary of State",
    "SA": "State Auditor",
    "ST": "State Treasurer",
    "Senate": "State Senator",
    "House": "State Representative",
    "SC": "State Supreme Court Justice",
    "AP": "State Appeals Court Judge",
    "DC": "State District Court Judge",
}


MI_CONT_DROP_COLS = [
    "doc_seq_no",
    "page_no",
    "cont_detail_id",
    "doc_type_desc",
    "address",
    "city",
    "zip",
    "occupation",
    "received_date",
    "aggregate",
    "extra_desc",
]

MI_EXP_DROP_COLS = [
    "doc_seq_no",
    "expenditure_type",
    "gub_account_type",
    "gub_elec_type",
    "page_no",
    "detail_id",
    "doc_type_desc",
    "extra_desc",
    "address",
    "city",
    "zip",
    "exp_date",
    "state_loc",
    "supp_opp",
    "can_or_ballot",
    "county",
    "debt_payment",
    "vend_addr",
    "vend_city",
    "vend_state",
    "vend_zip",
    "gotv_ink_ind",
    "fundraiser",
]


# PA EDA constants:

PA_CONT_COLS_NAMES_PRE2022: list = [
    "RECIPIENT_ID",
    "YEAR",
    "CYCLE",
    "SECTION",
    "DONOR",
    "ADDRESS_1",
    "ADDRESS_2",
    "CITY",
    "STATE",
    "ZIPCODE",
    "OCCUPATION",
    "E_NAME",
    "E_ADDRESS_1",
    "E_ADDRESS_2",
    "E_CITY",
    "E_STATE",
    "E_ZIPCODE",
    "CONT_DATE_1",
    "CONT_AMT_1",
    "CONT_DATE_2",
    "CONT_AMT_2",
    "CONT_DATE_3",
    "CONT_AMT_3",
    "PURPOSE",
]

PA_CONT_COLS_NAMES_POST2022: list = [
    "RECIPIENT_ID",
    "REPORTER_ID",
    "TIMESTAMP",
    "YEAR",
    "CYCLE",
    "SECTION",
    "DONOR",
    "ADDRESS_1",
    "ADDRESS_2",
    "CITY",
    "STATE",
    "ZIPCODE",
    "OCCUPATION",
    "E_NAME",
    "E_ADDRESS_1",
    "E_ADDRESS_2",
    "E_CITY",
    "E_STATE",
    "E_ZIPCODE",
    "CONT_DATE_1",
    "CONT_AMT_1",
    "CONT_DATE_2",
    "CONT_AMT_2",
    "CONT_DATE_3",
    "CONT_AMT_3",
    "PURPOSE",
]

PA_FILER_COLS_NAMES_PRE2022: list = [
    "RECIPIENT_ID",
    "YEAR",
    "CYCLE",
    "AMEND",
    "TERMINATE",
    "RECIPIENT_TYPE",
    "RECIPIENT",
    "RECIPIENT_OFFICE",
    "DISTRICT",
    "RECIPIENT_PARTY",
    "ADDRESS_1",
    "ADDRESS_2",
    "CITY",
    "STATE",
    "ZIPCODE",
    "COUNTY",
    "PHONE",
    "BEGINNING",
    "MONETARY",
    "INKIND",
]

PA_FILER_COLS_NAMES_POST2022: list = [
    "RECIPIENT_ID",
    "REPORTER_ID",
    "TIMESTAMP",
    "YEAR",
    "CYCLE",
    "AMEND",
    "TERMINATE",
    "RECIPIENT_TYPE",
    "RECIPIENT",
    "RECIPIENT_OFFICE",
    "DISTRICT",
    "RECIPIENT_PARTY",
    "ADDRESS_1",
    "ADDRESS_2",
    "CITY",
    "STATE",
    "ZIPCODE",
    "COUNTY",
    "PHONE",
    "BEGINNING",
    "MONETARY",
    "INKIND",
]

PA_EXPENSE_COLS_NAMES_PRE2022: list = [
    "DONOR_ID",
    "YEAR",
    "EXPENSE_CYCLE",
    "RECIPIENT",
    "EXPENSE_ADDRESS_1",
    "EXPENSE_ADDRESS_2",
    "EXPENSE_CITY",
    "EXPENSE_STATE",
    "EXPENSE_ZIPCODE",
    "EXPENSE_DATE",
    "AMOUNT",
    "PURPOSE",
]

PA_EXPENSE_COLS_NAMES_POST2022: list = [
    "DONOR_ID",
    "EXPENSE_REPORTER_ID",
    "EXPENSE_TIMESTAMP",
    "YEAR",
    "EXPENSE_CYCLE",
    "RECIPIENT",
    "EXPENSE_ADDRESS_1",
    "EXPENSE_ADDRESS_2",
    "EXPENSE_CITY",
    "EXPENSE_STATE",
    "EXPENSE_ZIPCODE",
    "EXPENSE_DATE",
    "AMOUNT",
    "PURPOSE",
]

PA_OFFICE_ABBREV_DICT: dict = {
    "GOV": "Governor",
    "LTG": "Lieutenant Gov",
    "ATT": "Attorney General",
    "AUD": "Auditor General",
    "TRE": "State Treasurer",
    "SPM": "Justice of the Supreme Crt",
    "SPR": "Judge of the Superior Crt",
    "CCJ": "Judge of the CommonWealth Crt",
    "CPJ": "Judge of the Crt of Common Pleas",
    "CPJA": "Judge of the Crt of Common Pleas",
    "CPJP": "Judge of the Crt of Common Pleas",
    "MCJ": "Judge of the Municipal Crt",
    "TCJ": "Judge of the Traffic Crt",
    "STS": "Senator (General Assembly)",
    "STH": "Rep (General Assembly)",
    "USC": "United States Congress",
    "USS": "United States Senate",
    "DSC": "Member of Dem State Committee",
    "RSC": "Member of Rep State Committee",
    "OTH": "Other(local offices)",
}
PA_FILER_ABBREV_DICT: dict = {1.0: "Candidate", 2.0: "Committee", 3.0: "Lobbyist"}

PA_ORGANIZATION_IDENTIFIERS: list = [
    "FRIENDS",
    "CITIZENS",
    "UNION",
    "STATE",
    "TEAM",
    "PAC",
    "PA",
    "GOVT",
    "WARD",
    "DEM",
    "COM",
    "COMMITTEE",
    "CORP",
    "ASSOCIATIONS",
    "FOR",
    "FOR THE",
    "SENATE",
    "COMMONWEALTH",
    "ELECT",
    "POLITICAL ACTION COMMITTEE",
    "REPUBLICANS",
    "REPUBLICAN",
    "DEMOCRAT",
    "DEMOCRATS",
    "CORPORATION",
    "CORP",
    "COMPANY",
    "CO",
    "LIMITED",
    "LTD",
    "INC",
    "INCORPORATED",
    "LLC",
    "FUND",
]

MI_EXPENDITURE_COLUMNS = [
    "doc_seq_no",
    "expenditure_type",
    "gub_account_type",
    "gub_elec_type",
    "page_no",
    "expense_id",
    "detail_id",
    "doc_stmnt_year",
    "doc_type_desc",
    "com_legal_name",
    "common_name",
    "cfr_com_id",
    "com_type",
    "schedule_desc",
    "exp_desc",
    "purpose",
    "extra_desc",
    "f_name",
    "lname_or_org",
    "address",
    "city",
    "state",
    "zip",
    "exp_date",
    "amount",
    "state_loc",
    "supp_opp",
    "can_or_ballot",
    "county",
    "debt_payment",
    "vend_name",
    "vend_addr",
    "vend_city",
    "vend_state",
    "vend_zip",
    "gotv_ink_ind",
    "fundraiser",
]

MICHIGAN_CONTRIBUTION_COLS_REORDER = [
    "doc_seq_no",
    "page_no",
    "contribution_id",
    "cont_detail_id",
    "doc_stmnt_year",
    "doc_type_desc",
    "common_name",
    "com_type",
    "can_first_name",
    "can_last_name",
    "contribtype",
    "f_name",
    "l_name_or_org",
    "address",
    "city",
    "state",
    "zip",
    "occupation",
    "employer",
    "amount",
    "received_date",
    "aggregate",
    "extra_desc",
    "amount",
]

MICHIGAN_CONTRIBUTION_COLS_RENAME = [
    "doc_seq_no",
    "page_no",
    "contribution_id",
    "cont_detail_id",
    "doc_stmnt_year",
    "doc_type_desc",
    "com_legal_name",
    "common_name",
    "cfr_com_id",
    "com_type",
    "can_first_name",
    "can_last_name",
    "contribtype",
    "f_name",
    "l_name_or_org",
    "address",
    "city",
    "state",
    "zip",
    "occupation",
    "employer",
    "received_date",
    "amount",
    "aggregate",
]


AZ_pages_dict = {
    "Candidate": 1,
    "PAC": 2,
    "Political Party": 3,
    "Organizations": 4,
    "Independent Expenditures": 5,
    "Ballot Measures": 6,
    "Individual Contributors": 7,
    "Vendors": 8,
    "Name": 11,
    "Candidate/Income": 20,
    "Candidate/Expense": 21,
    "Candidate/IEFor": 22,
    "Candidate/IEAgainst": 23,
    "Candidate/All Transactions": 24,
    "PAC/Income": 30,
    "PAC/Expense": 31,
    "PAC/IEFor": 32,
    "PAC/IEAgainst": 33,
    "PAC/BMEFor": 34,
    "PAC/BMEAgainst": 35,
    "PAC/All Transactions": 36,
    "Political Party/Income": 40,
    "Political Party/Expense": 41,
    "Political Party/All Transactions": 42,
    "Organizations/IEFor": 50,
    "Organizations/IEAgainst": 51,
    "Organizations/BMEFor": 52,
    "Organizations/BME Against": 53,
    "Organizations/All Transactions": 54,
    "Independent Expenditures/IEFor": 60,
    "Independent Expenditures/IEAgainst": 61,
    "Independent Expenditures/All Transactions": 62,
    "Ballot Measures/Amount For": 70,
    "Ballot Measures/Amount Against": 71,
    "Ballot Measures/All Transactions": 72,
    "Individuals/All Transactions": 80,
    "Vendors/All Transactions": 90,
}

AZ_head = {
    "authority": "seethemoney.az.gov",
    "accept": "application/json, text/javascript, */*; q=0.01",
    "accept-language": "en-US,en;q=0.7",
    "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    "origin": "https://seethemoney.az.gov",
    "referer": "https://seethemoney.az.gov/Reporting/Explore",
    "sec-ch-ua": '"Chromium";v="116", "Not)A;Brand";v="24", "Brave";v="116"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "sec-gpc": "1",
    "user-agent": """Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)
    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36""",
    "x-requested-with": "XMLHttpRequest",
}


AZ_valid_detailed_pages = [
    20,
    21,
    22,
    23,
    24,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    40,
    41,
    42,
    50,
    51,
    52,
    53,
    54,
    60,
    61,
    62,
    70,
    71,
    72,
    80,
    90,
]


AZ_base_data = {
    "draw": "2",
    "order[0][column]": "0",
    "order[0][dir]": "asc",
    "start": "0",
    "length": "500000",
    "search[value]": "",
    "search[regex]": "false",
}

state_abbreviations = [
    " AK ",
    " AL ",
    " AR ",
    " AZ ",
    " CA ",
    " CO ",
    " CT ",
    " DC ",
    " DE ",
    " FL ",
    " GA ",
    " GU ",
    " HI ",
    " IA ",
    " ID ",
    " IL ",
    " IN ",
    " KS ",
    " KY ",
    " LA ",
    " MA ",
    " MD ",
    " ME ",
    " MI ",
    " MN ",
    " MO ",
    " MS ",
    " MT ",
    " NC ",
    " ND ",
    " NE ",
    " NH ",
    " NJ ",
    " NM ",
    " NV ",
    " NY ",
    " OH ",
    " OK ",
    " OR ",
    " PA ",
    " PR ",
    " RI ",
    " SC ",
    " SD ",
    " TN ",
    " TX ",
    " UT ",
    " VA ",
    " VI ",
    " VT ",
    " WA ",
    " WI ",
    " WV ",
    " WY ",
]
