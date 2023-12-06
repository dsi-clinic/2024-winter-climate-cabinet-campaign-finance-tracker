"""
Constants to be used in various parts of the project.
"""
from pathlib import Path

MI_FILEPATH = "../data/Contributions/"

MI_VALUES_TO_CHECK = ["1998", "1999", "2000", "2001", "2002", "2003"]

BASE_FILEPATH = Path(__file__).resolve().parent.parent

USER_AGENT = """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
                (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"""

HEADERS = {"User-Agent": USER_AGENT}

MI_EXP_FILEPATH = str(BASE_FILEPATH / "data" / "Expenditure")

MI_CON_FILEPATH = str(BASE_FILEPATH / "data" / "Contribution")

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

PA_MAIN_URL = "https://www.dos.pa.gov"
PA_ZIPPED_URL = (
    "/VotingElections/CandidatesCommittees/CampaignFinance/Resources/Documents/"
)

# PA EDA constants:

PA_CONT_COLS_NAMES_PRE2022: list = [
    "FILER_ID",
    "YEAR",
    "CYCLE",
    "SECTION",
    "CONTRIBUTOR",
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
    "CONT_DESCRIP",
]

PA_CONT_COLS_NAMES_POST2022: list = [
    "FILER_ID",
    "REPORTER_ID",
    "TIMESTAMP",
    "YEAR",
    "CYCLE",
    "SECTION",
    "CONTRIBUTOR",
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
    "CONT_DESCRIP",
]

PA_FILER_COLS_NAMES_PRE2022: list = [
    "FILER_ID",
    "YEAR",
    "CYCLE",
    "AMEND",
    "TERMINATE",
    "FILER_TYPE",
    "FILER_NAME",
    "OFFICE",
    "DISTRICT",
    "PARTY",
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
    "FILER_ID",
    "REPORTER_ID",
    "TIMESTAMP",
    "YEAR",
    "CYCLE",
    "AMEND",
    "TERMINATE",
    "FILER_TYPE",
    "FILER_NAME",
    "OFFICE",
    "DISTRICT",
    "PARTY",
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
    "FILER_ID",
    "YEAR",
    "EXPENSE_CYCLE",
    "EXPENSE_NAME",
    "EXPENSE_ADDRESS_1",
    "EXPENSE_ADDRESS_2",
    "EXPENSE_CITY",
    "EXPENSE_STATE",
    "EXPENSE_ZIPCODE",
    "EXPENSE_DATE",
    "EXPENSE_AMT",
    "EXPENSE_DESC",
]

PA_EXPENSE_COLS_NAMES_POST2022: list = [
    "FILER_ID",
    "EXPENSE_REPORTER_ID",
    "EXPENSE_TIMESTAMP",
    "YEAR",
    "EXPENSE_CYCLE",
    "EXPENSE_NAME",
    "EXPENSE_ADDRESS_1",
    "EXPENSE_ADDRESS_2",
    "EXPENSE_CITY",
    "EXPENSE_STATE",
    "EXPENSE_ZIPCODE",
    "EXPENSE_DATE",
    "EXPENSE_AMT",
    "EXPENSE_DESC",
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


AZ_pages_dict = {
    "Candidate": 1,
    "PAC": 2,
    "Political Party": 3,
    "Organzations": 4,
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
