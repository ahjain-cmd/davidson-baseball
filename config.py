"""
Davidson Baseball Analytics — Configuration & Constants.

All roster data, pitch mappings, color definitions, zone constants,
and name normalization utilities live here.
"""
import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("TRACKMAN_CSV_ROOT", os.path.join(_APP_DIR, "v3"))
PARQUET_FIXED_PATH = os.path.join(_APP_DIR, "all_trackman_fixed.parquet")
PARQUET_PATH = PARQUET_FIXED_PATH if os.path.exists(PARQUET_FIXED_PATH) else os.path.join(_APP_DIR, "all_trackman.parquet")
DUCKDB_PATH = os.path.join(_APP_DIR, "davidson.duckdb")
DAVIDSON_TEAM_ID = "DAV_WIL"

# ── TrueMedia API credentials ────────────────
TM_USERNAME = "frhowden@davidson.edu"
TM_SITENAME = "davidson-ncaabaseball"
TM_MASTER_TOKEN = os.environ.get(
    "TM_MASTER_TOKEN",
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiZjZlZWEwYzViZmUwZTY4ZmEwZDUyMGQyMDU2NTNmYzciLCJpYXQiOjE3NzAwMDM4NTd9.c2QwNDh0Sy54ystStrYvORy4PrEQEJbUFDAacCH55EA",
)
CACHE_DIR = os.path.join(_APP_DIR, ".cache")
TUNNEL_BENCH_PATH = os.path.join(CACHE_DIR, "tunnel_benchmarks.json")
TUNNEL_WEIGHTS_PATH = os.path.join(CACHE_DIR, "tunnel_weights.json")

# ── Roster ─────────────────────────────────────
ROSTER_2026 = {
    "Higgins, Justin", "Diaz, Fredy", "Vokal, Jacob", "Vannoy, Matthew",
    "Rice, Aidan", "Thomas, Gavin", "Collins, Cooper", "McCullough, Will",
    "Manjooran, Matthew", "Berman, Connor", "Daly, Jameson", "Daly, Jamie",
    "Lietz, Forrest", "Champey, Brycen", "Furr, Keely", "Loughlin, Theo",
    "Laughlin, Theo", "Cavanaugh, Cooper", "Edwards, Scotty", "Ludwig, Landon",
    "Banks, Will", "Hall, Edward", "Hall, Ed", "Torreso, Anthony", "Brooks, Will",
    "Hoyt, Ivan", "Fritch, Brendan", "Papciak, Will", "Wille, Tyler",
    "Smith, Daniel", "Jimenez, Ethan", "Jones, Parker", "Marenghi, Will",
    "Hultquist, Henry", "Whelan, Thomas", "Pyne, Garrett", "Taggart, Carson",
    "Howard, Jed", "Perkins, Wilson", "Hamilton, Matt", "Hamilton, Matthew",
    "Yochum, Simon", "Suarez, Jake", "Ban, Jason", "Katz, Adam",
}

JERSEY = {
    "Higgins, Justin": 1, "Diaz, Fredy": 2, "Vokal, Jacob": 4, "Vannoy, Matthew": 5,
    "Rice, Aidan": 6, "Thomas, Gavin": 7, "Collins, Cooper": 8, "McCullough, Will": 9,
    "Manjooran, Matthew": 10, "Berman, Connor": 11, "Daly, Jamie": 12, "Daly, Jameson": 12,
    "Lietz, Forrest": 13, "Champey, Brycen": 14, "Furr, Keely": 15,
    "Loughlin, Theo": 16, "Laughlin, Theo": 16, "Cavanaugh, Cooper": 17,
    "Edwards, Scotty": 18, "Ludwig, Landon": 19, "Banks, Will": 20,
    "Hall, Ed": 21, "Hall, Edward": 21, "Torreso, Anthony": 22, "Brooks, Will": 23,
    "Hoyt, Ivan": 24, "Fritch, Brendan": 25, "Papciak, Will": 26, "Wille, Tyler": 27,
    "Smith, Daniel": 28, "Jimenez, Ethan": 29, "Jones, Parker": 31, "Marenghi, Will": 32,
    "Hultquist, Henry": 33, "Whelan, Thomas": 34, "Pyne, Garrett": 35, "Taggart, Carson": 36,
    "Howard, Jed": 38, "Perkins, Wilson": 39, "Hamilton, Matt": 40, "Hamilton, Matthew": 40,
    "Yochum, Simon": 41, "Suarez, Jake": 42, "Ban, Jason": 43, "Katz, Adam": 44,
}

POSITION = {
    "Higgins, Justin": "INF", "Diaz, Fredy": "INF", "Vokal, Jacob": "RHP/INF",
    "Vannoy, Matthew": "C/INF", "Rice, Aidan": "INF/OF", "Thomas, Gavin": "OF",
    "Collins, Cooper": "INF", "McCullough, Will": "OF", "Manjooran, Matthew": "3B/OF",
    "Berman, Connor": "INF/OF", "Daly, Jamie": "OF", "Daly, Jameson": "OF",
    "Lietz, Forrest": "LHP/OF", "Champey, Brycen": "RHP", "Furr, Keely": "RHP",
    "Loughlin, Theo": "1B", "Laughlin, Theo": "1B", "Cavanaugh, Cooper": "LHP",
    "Edwards, Scotty": "OF", "Ludwig, Landon": "RHP", "Banks, Will": "RHP",
    "Hall, Ed": "RHP", "Hall, Edward": "RHP", "Torreso, Anthony": "C",
    "Brooks, Will": "C/INF", "Hoyt, Ivan": "RHP/INF", "Fritch, Brendan": "C/OF",
    "Papciak, Will": "RHP", "Wille, Tyler": "LHP", "Smith, Daniel": "RHP",
    "Jimenez, Ethan": "1B/RHP", "Jones, Parker": "RHP", "Marenghi, Will": "RHP",
    "Hultquist, Henry": "RHP", "Whelan, Thomas": "RHP", "Pyne, Garrett": "RHP",
    "Taggart, Carson": "LHP", "Howard, Jed": "OF/LHP", "Perkins, Wilson": "RHP",
    "Hamilton, Matt": "RHP", "Hamilton, Matthew": "RHP", "Yochum, Simon": "RHP",
    "Suarez, Jake": "C", "Ban, Jason": "LHP", "Katz, Adam": "Util",
}

# ── Bryant 2026 Roster ────────────────────────
BRYANT_TEAM_NAME = "Bryant University"
BRYANT_COMBINED_TEAM_ID = "BRYANT_COMBINED_2026"

# Transfers: "Last, First" -> previous school search name.
# Players NOT in this dict are assumed to have been at Bryant for 2024-2025.
BRYANT_TRANSFERS = {
    "Wensley, Casey": "Wheaton College (Massachusetts)",
    "Greger, Gavin": "University of Connecticut",
    "Story, Hudson": "LA Mission",  # community college — not in TrueMedia
    "Galusha, Thomas": "University of Connecticut",
    "Irizarry, Carlos": "Penn State Harrisburg",
    "Scudder, Dylan": "Eastern Connecticut",
    "Garcia, Ellis": "West Virginia",
    "Dressler, Justin": "Pace",
    "Vining, Aidan": "Johnson and Wales",
    "Durand, Brandyn": "Chipola",
    "Salsberg, Zev": "Ohio State",
    "Flaherty, Tommy": "Clark University",
    "Schiff, Cole": "UNC Asheville",
    "White, Landon": "Ithaca",
}
BRYANT_ROSTER_2026 = {
    "Vazquez, Alejandro", "Kingsbury, Hunter", "Prince, Dylan", "Zyons, Zac",
    "Belcher, Michael", "Wensley, Casey", "Greger, Gavin", "Ferrell, Vince",
    "Papetti, Cam", "Fiatarone, Mike", "Carter, Ian", "Hilburger, Kaden",
    "Story, Hudson", "Hackett, Justin", "Gaudreau, Jacob", "Davis, Ty",
    "Gorman, Greg", "Galusha, Thomas", "Saul, Charlie", "Irizarry, Carlos",
    "Scudder, Dylan", "Garcia, Ellis", "Dressler, Justin", "Hurley, Will",
    "Burkholz, Max", "Vining, Aidan", "Durand, Brandyn", "Soroko, Cameron",
    "Birchard, Owen", "Malloy, Tommy", "Davis, Zach", "Vanesko, Jackson",
    "Perez, Yamil", "Zaslaw, Sean", "Lewis, Bradley", "Salsberg, Zev",
    "Flaherty, Tommy", "Schiff, Cole", "White, Landon", "Mulholland, Billy",
    "Maher, Thomas", "Dobis, Jameson", "Clifford, Sean",
}

BRYANT_JERSEY = {
    "Vazquez, Alejandro": 1, "Kingsbury, Hunter": 2, "Prince, Dylan": 3,
    "Zyons, Zac": 4, "Belcher, Michael": 5, "Wensley, Casey": 6,
    "Greger, Gavin": 7, "Ferrell, Vince": 9, "Papetti, Cam": 10,
    "Fiatarone, Mike": 11, "Carter, Ian": 12, "Hilburger, Kaden": 13,
    "Story, Hudson": 14, "Hackett, Justin": 15, "Gaudreau, Jacob": 16,
    "Davis, Ty": 17, "Gorman, Greg": 18, "Galusha, Thomas": 19,
    "Saul, Charlie": 20, "Irizarry, Carlos": 21, "Scudder, Dylan": 22,
    "Garcia, Ellis": 23, "Dressler, Justin": 24, "Hurley, Will": 25,
    "Burkholz, Max": 26, "Vining, Aidan": 27, "Durand, Brandyn": 28,
    "Soroko, Cameron": 29, "Birchard, Owen": 30, "Malloy, Tommy": 31,
    "Davis, Zach": 32, "Vanesko, Jackson": 33, "Perez, Yamil": 34,
    "Zaslaw, Sean": 35, "Lewis, Bradley": 36, "Salsberg, Zev": 37,
    "Flaherty, Tommy": 39, "Schiff, Cole": 40, "White, Landon": 41,
    "Mulholland, Billy": 42, "Maher, Thomas": 43, "Dobis, Jameson": 50,
}

BRYANT_POSITION = {
    "Vazquez, Alejandro": "NF", "Kingsbury, Hunter": "OF", "Prince, Dylan": "INF",
    "Zyons, Zac": "INF", "Belcher, Michael": "RHP", "Wensley, Casey": "INF",
    "Greger, Gavin": "OF", "Ferrell, Vince": "OF", "Papetti, Cam": "C/INF",
    "Fiatarone, Mike": "INF", "Carter, Ian": "RHP", "Hilburger, Kaden": "OF/RHP",
    "Story, Hudson": "INF", "Hackett, Justin": "INF", "Gaudreau, Jacob": "C",
    "Davis, Ty": "RHP", "Gorman, Greg": "OF", "Galusha, Thomas": "RHP",
    "Saul, Charlie": "OF", "Irizarry, Carlos": "INF", "Scudder, Dylan": "RHP",
    "Garcia, Ellis": "INF", "Dressler, Justin": "RHP", "Hurley, Will": "RHP",
    "Burkholz, Max": "OF", "Vining, Aidan": "LHP", "Durand, Brandyn": "C",
    "Soroko, Cameron": "INF/OF", "Birchard, Owen": "RHP", "Malloy, Tommy": "C",
    "Davis, Zach": "RHP", "Vanesko, Jackson": "LHP", "Perez, Yamil": "C",
    "Zaslaw, Sean": "RHP", "Lewis, Bradley": "LHP", "Salsberg, Zev": "RHP",
    "Flaherty, Tommy": "RHP", "Schiff, Cole": "LHP", "White, Landon": "RHP",
    "Mulholland, Billy": "RHP/C", "Maher, Thomas": "LHP", "Dobis, Jameson": "RHP",
    "Clifford, Sean": "RHP",
}

# ── UNCG (UNC Greensboro) 2026 Roster ────────
UNCG_TEAM_NAME = "UNC Greensboro"
UNCG_COMBINED_TEAM_ID = "UNCG_COMBINED_2026"

# Transfers: "Last, First" -> previous school search name.
# Players NOT in this dict are assumed to have been at UNCG for 2024-2025.
UNCG_TRANSFERS = {
    "Ruocco, Anthony": "Florida Southwestern",
    "Mueller, Jake": "Florida Southwestern",
    "Barbour, Jake": "Queens University",
    "Gardner, Brody": "Greensboro College",
    "Weaver, JT": "Shippensburg University",
    "Holland, Tucker": "Gaston College",
    "Horton, John": "Virginia Beach",  # Grad student
}

UNCG_ROSTER_2026 = {
    "Ruocco, Anthony", "Truitt, Brantley", "Budzik, Jacob", "Parsons, JJ",
    "Aycock, Grant", "Brittain, Ethan", "Dilley, Jacob", "Williams, Ian",
    "Jenkins, Luke", "Berry, Tanner", "Wight, Parker", "Mueller, Jake",
    "Dear, Mayson", "Rogers, Eddie", "Gardner, Brody", "Polk, Landon",
    "Bush, Wyatt", "Lee, Cannon", "Barbour, Jake", "Chapman, Noah",
    "Hester, Thomas", "Shuey, Hunter", "Miles, Isaac", "Holland, Luke",
    "Hudson, Brandon", "Thomas, Luke", "West, Ayden", "Nobles, Ethan",
    "Horton, John", "Winfield, Nolan", "Watson, Parker", "Weaver, JT",
    "Smith, Dylan", "Lancaster, Hazen", "Colucci, Jake", "Barnes, Cole",
    "Holland, Tucker",
}

UNCG_JERSEY = {
    "Ruocco, Anthony": 2, "Truitt, Brantley": 3, "Budzik, Jacob": 4,
    "Parsons, JJ": 5, "Aycock, Grant": 6, "Brittain, Ethan": 7,
    "Dilley, Jacob": 8, "Williams, Ian": 9, "Jenkins, Luke": 10,
    "Berry, Tanner": 11, "Wight, Parker": 12, "Mueller, Jake": 13,
    "Dear, Mayson": 14, "Rogers, Eddie": 15, "Gardner, Brody": 16,
    "Polk, Landon": 17, "Bush, Wyatt": 18, "Lee, Cannon": 19,
    "Barbour, Jake": 20, "Chapman, Noah": 21, "Hester, Thomas": 22,
    "Shuey, Hunter": 23, "Miles, Isaac": 24, "Holland, Luke": 25,
    "Hudson, Brandon": 26, "Thomas, Luke": 27, "West, Ayden": 29,
    "Nobles, Ethan": 30, "Horton, John": 31, "Winfield, Nolan": 32,
    "Watson, Parker": 34, "Weaver, JT": 35, "Smith, Dylan": 36,
    "Lancaster, Hazen": 37, "Colucci, Jake": 39, "Barnes, Cole": 40,
    "Holland, Tucker": 44,
}

UNCG_POSITION = {
    "Ruocco, Anthony": "IF", "Truitt, Brantley": "IF/OF", "Budzik, Jacob": "IF",
    "Parsons, JJ": "C", "Aycock, Grant": "IF/RHP", "Brittain, Ethan": "IF",
    "Dilley, Jacob": "C/OF", "Williams, Ian": "UTIL", "Jenkins, Luke": "UTIL",
    "Berry, Tanner": "C", "Wight, Parker": "IF", "Mueller, Jake": "IF",
    "Dear, Mayson": "RHP/C", "Rogers, Eddie": "RHP", "Gardner, Brody": "OF",
    "Polk, Landon": "OF", "Bush, Wyatt": "UTIL", "Lee, Cannon": "RHP",
    "Barbour, Jake": "IF", "Chapman, Noah": "LHP", "Hester, Thomas": "LHP",
    "Shuey, Hunter": "RHP", "Miles, Isaac": "RHP", "Holland, Luke": "OF",
    "Hudson, Brandon": "RHP", "Thomas, Luke": "RHP", "West, Ayden": "RHP",
    "Nobles, Ethan": "RHP", "Horton, John": "RHP", "Winfield, Nolan": "RHP",
    "Watson, Parker": "IF", "Weaver, JT": "RHP", "Smith, Dylan": "RHP",
    "Lancaster, Hazen": "LHP", "Colucci, Jake": "RHP", "Barnes, Cole": "RHP",
    "Holland, Tucker": "LHP",
}

# ── Fairfield 2026 Roster ────────
FAIRFIELD_TEAM_NAME = "Fairfield"
FAIRFIELD_COMBINED_TEAM_ID = "FAIRFIELD_COMBINED_2026"
FAIRFIELD_TM_TEAM_ID = 730380544

FAIRFIELD_TRANSFERS = {
    "Spencer, Luke": "Wilkes",
    "Stephenson, Zach": "Endicott",
    "Haarde, Jake": "Penn State",
    "Camera, Dom": "Hofstra",
}

FAIRFIELD_ROSTER_2026 = {
    "Nomura, Luke", "Spencer, Luke", "Wawruck, Payten", "Colby, Nolan",
    "Stephenson, Zach", "McIlroy, Liam", "Baglino, Aidan", "Paine, Ryan",
    "Scanlan, Connor", "Kuczik, JP", "Byrne, Jack", "Bucciero, Matthew",
    "Swanson, Boden", "Schmalzle, TJ", "Haarde, Jake", "Grande, Carter",
    "Camera, Dom", "Gabardi, Dominic",
    "Mulvaney, Jimmy", "Kalfas, Matt", "Engle, Harrison", "Hoxie, Hunter",
    "Miller, Brendon", "Chambers, Joseph", "Wolff, Devin", "Grabmann, Matthew",
    "Youngman, Will", "Maiorano, Ryan", "Sheldon, Jack", "Frank, Alex",
    "Kurek, Matthew", "Memoli, Jake", "Ramchandran, NJ", "Kell, Kevin",
    "Alekson, Ben", "Kelly, Nick",
}

FAIRFIELD_JERSEY = {
    "Mulvaney, Jimmy": 1, "Nomura, Luke": 2, "Spencer, Luke": 3,
    "Wawruck, Payten": 4, "Colby, Nolan": 5, "Stephenson, Zach": 6,
    "McIlroy, Liam": 7, "Baglino, Aidan": 8, "Kalfas, Matt": 9,
    "Paine, Ryan": 10, "Engle, Harrison": 11, "Scanlan, Connor": 13,
    "Kuczik, JP": 14, "Byrne, Jack": 15, "Hoxie, Hunter": 16,
    "Bucciero, Matthew": 17, "Miller, Brendon": 18, "Swanson, Boden": 19,
    "Chambers, Joseph": 20, "Wolff, Devin": 21, "Schmalzle, TJ": 22,
    "Haarde, Jake": 23, "Grabmann, Matthew": 24, "Youngman, Will": 25,
    "Maiorano, Ryan": 26, "Grande, Carter": 27, "Camera, Dom": 29,
    "Sheldon, Jack": 30, "Frank, Alex": 31, "Gabardi, Dominic": 33,
    "Kurek, Matthew": 34, "Memoli, Jake": 35, "Ramchandran, NJ": 36,
    "Kell, Kevin": 37, "Alekson, Ben": 38, "Kelly, Nick": 39,
}

FAIRFIELD_POSITION = {
    "Nomura, Luke": "INF", "Spencer, Luke": "INF", "Wawruck, Payten": "OF",
    "Colby, Nolan": "INF", "Stephenson, Zach": "OF", "McIlroy, Liam": "UTL",
    "Baglino, Aidan": "C", "Paine, Ryan": "INF", "Scanlan, Connor": "INF",
    "Kuczik, JP": "C", "Byrne, Jack": "OF", "Bucciero, Matthew": "OF",
    "Swanson, Boden": "C", "Schmalzle, TJ": "OF", "Haarde, Jake": "OF",
    "Grande, Carter": "INF", "Camera, Dom": "C", "Gabardi, Dominic": "OF/RHP",
    "Mulvaney, Jimmy": "RHP", "Kalfas, Matt": "RHP", "Engle, Harrison": "RHP",
    "Hoxie, Hunter": "RHP", "Miller, Brendon": "RHP", "Chambers, Joseph": "RHP",
    "Wolff, Devin": "RHP", "Grabmann, Matthew": "RHP", "Youngman, Will": "RHP",
    "Maiorano, Ryan": "RHP", "Sheldon, Jack": "RHP", "Frank, Alex": "RHP",
    "Kurek, Matthew": "RHP", "Memoli, Jake": "RHP", "Ramchandran, NJ": "RHP",
    "Kell, Kevin": "RHP", "Alekson, Ben": "RHP", "Kelly, Nick": "RHP",
}

# ── Wofford 2026 Roster ────────
WOFFORD_TEAM_NAME = "Wofford"
WOFFORD_COMBINED_TEAM_ID = "WOFFORD_COMBINED_2026"
WOFFORD_TM_TEAM_ID = 730176512

WOFFORD_TRANSFERS = {
    "Quarrie, Marc": "Muhlenberg",
    "Manning, Lucas": "Holy Cross",
    "Burroughs, Keeton": "Pittsburgh",
    "Newman, Blayne": "USC Aiken",
    "Bouknight, Cade": "USC Aiken",
    "Childers, Corben": "Anderson University",
    "Egger, Sheldon": "Edmonds",
    "Feliz, Raul": "Indian River State",
    "Herndon, Hunter": "Sewanee",
    "Brini, Niko": "UConn",
}

WOFFORD_ROSTER_2026 = {
    "Tribble, Logan", "Belk, Ethan", "Porter, Zach", "Layman, James",
    "Campi, Harrison", "Mannelly, Andrew", "Quarrie, Marc", "Timblin, Ben",
    "Hardin, Tanner", "Manning, Lucas", "Little, Branton", "Rivers, Davis",
    "Weaver, Mason", "Burroughs, Keeton", "Collins, Cade", "Davis, Champ",
    "Norris, Will", "Alston, Miller", "Laughlin, Austin", "Draper, Cole",
    "Lewis, Hiram", "Smith, JT", "Egger, Sheldon", "Newman, Blayne",
    "Bouknight, Cade", "Childers, Corben", "Estes, Wes", "Bouchard, Alec",
    "Gold, Brady", "Fitzpatrick, PJ", "Snow, Brady", "Gerrick, Noah",
    "Michaels, Kenny", "Myers, Alex", "Feliz, Raul", "Kirwan, Tommy",
    "Herndon, Hunter", "Condon, Cullen", "James, Trey", "Brini, Niko",
    "Euart, Jack", "Gray, John", "Clemente, Michael", "Compton, Mason",
    "Howard, Will", "Rembish, Logan", "Vargo, Lucas",
}

WOFFORD_JERSEY = {
    "Tribble, Logan": 1, "Belk, Ethan": 2, "Porter, Zach": 3,
    "Layman, James": 4, "Campi, Harrison": 5, "Mannelly, Andrew": 6,
    "Quarrie, Marc": 7, "Timblin, Ben": 8, "Hardin, Tanner": 9,
    "Manning, Lucas": 10, "Little, Branton": 11, "Rivers, Davis": 12,
    "Weaver, Mason": 13, "Burroughs, Keeton": 14, "Collins, Cade": 15,
    "Davis, Champ": 16, "Norris, Will": 17, "Alston, Miller": 18,
    "Laughlin, Austin": 19, "Draper, Cole": 20, "Lewis, Hiram": 21,
    "Smith, JT": 22, "Egger, Sheldon": 23, "Newman, Blayne": 24,
    "Bouknight, Cade": 26, "Childers, Corben": 27, "Estes, Wes": 28,
    "Bouchard, Alec": 30, "Gold, Brady": 31, "Fitzpatrick, PJ": 34,
    "Snow, Brady": 35, "Gerrick, Noah": 36, "Michaels, Kenny": 37,
    "Myers, Alex": 38, "Feliz, Raul": 39, "Kirwan, Tommy": 41,
    "Herndon, Hunter": 43, "Condon, Cullen": 44, "James, Trey": 45,
    "Brini, Niko": 46, "Euart, Jack": 48, "Gray, John": 50,
}

WOFFORD_POSITION = {
    "Tribble, Logan": "OF", "Belk, Ethan": "OF", "Porter, Zach": "C",
    "Layman, James": "INF", "Campi, Harrison": "INF", "Mannelly, Andrew": "OF",
    "Quarrie, Marc": "OF", "Timblin, Ben": "OF", "Hardin, Tanner": "INF",
    "Manning, Lucas": "C", "Little, Branton": "RHP", "Rivers, Davis": "RHP",
    "Weaver, Mason": "RHP", "Burroughs, Keeton": "INF", "Collins, Cade": "INF",
    "Davis, Champ": "RHP", "Norris, Will": "OF", "Alston, Miller": "LHP",
    "Laughlin, Austin": "RHP", "Draper, Cole": "RHP", "Lewis, Hiram": "RHP",
    "Smith, JT": "C", "Egger, Sheldon": "LHP", "Newman, Blayne": "RHP",
    "Bouknight, Cade": "RHP", "Childers, Corben": "RHP", "Estes, Wes": "RHP",
    "Bouchard, Alec": "RHP", "Gold, Brady": "INF", "Fitzpatrick, PJ": "RHP",
    "Snow, Brady": "RHP", "Gerrick, Noah": "OF", "Michaels, Kenny": "LHP",
    "Myers, Alex": "INF", "Feliz, Raul": "C", "Kirwan, Tommy": "INF",
    "Herndon, Hunter": "INF/OF", "Condon, Cullen": "RHP", "James, Trey": "LHP",
    "Brini, Niko": "OF", "Euart, Jack": "INF", "Gray, John": "RHP",
    "Clemente, Michael": "RHP", "Compton, Mason": "RHP", "Howard, Will": "RHP",
    "Rembish, Logan": "RHP", "Vargo, Lucas": "RHP",
}

# ── LMU (Loyola Marymount) 2026 Roster ────────
LMU_TEAM_NAME = "Loyola Marymount"
LMU_COMBINED_TEAM_ID = "LMU_COMBINED_2026"
LMU_TM_TEAM_ID = 730149120

LMU_TRANSFERS = {
    # D1 transfers (new in 2026, previous D1 school for 2025 data)
    "Whitton, Cooper": "Washington",
    "Wadas, Zach": "TCU",
    "Stiveson, Nate": "Stanford",
    "Riera, Niko": "UC Irvine",
    "Champion, Matt": "Oregon",
    "Estrella, Andrew": "UCF",
    "Gallegos, Elliot": "UC Santa Barbara",
}

LMU_ROSTER_2026 = {
    # Returning from 2025
    "Behrens, Adam", "Laine, Avery", "Ghiorso, DJ", "Dunn, JD",
    "Fried, Jacob", "Geis, Jake", "Lyall, Jake", "Johnson, Jonah",
    "Moreno, Matthew", "Malone, Noah", "Warady, Tanner", "Bender, Zach",
    "Williams, Zion", "Jacobsen, Gavin", "Casale, Johnny", "Danos, Luca",
    "Chavez, Alex", "Stucky, Cole",
    # New in 2026 — D1 transfers
    "Whitton, Cooper", "Wadas, Zach", "Stiveson, Nate", "Riera, Niko",
    "Champion, Matt", "Estrella, Andrew", "Gallegos, Elliot",
    # New in 2026 — JC / other transfers
    "Singh, Dylan", "Wall, Jaxson", "Elward, Luke", "Johnson, Alec",
    "Yamanaka, Eli", "DenDekker, Zach", "Carmona Jr., Jose", "Gabay, Lucas",
    # New in 2026 — freshmen
    "Schneider, Max", "Sweeney, Caleb", "Klosek, Richie", "Friend, Travis",
    "Gamboa, Alex", "Mhoon, Andrew", "Ortiz, Jordan", "Gurney, Win",
    "Erdmann, Eric", "Aguirre, David",
}

LMU_JERSEY = {
    "Moreno, Matthew": 0, "Wall, Jaxson": 2, "Singh, Dylan": 3,
    "Malone, Noah": 4, "Dunn, JD": 5, "Estrella, Andrew": 6,
    "Williams, Zion": 7, "Jacobsen, Gavin": 8, "Casale, Johnny": 9,
    "Stiveson, Nate": 11, "Schneider, Max": 12, "Sweeney, Caleb": 13,
    "DenDekker, Zach": 14, "Klosek, Richie": 15, "Lyall, Jake": 16,
    "Gallegos, Elliot": 17, "Ghiorso, DJ": 18, "Danos, Luca": 19,
    "Geis, Jake": 20, "Riera, Niko": 21, "Erdmann, Eric": 22,
    "Johnson, Jonah": 23, "Behrens, Adam": 24, "Elward, Luke": 25,
    "Champion, Matt": 26, "Warady, Tanner": 27, "Aguirre, David": 28,
    "Yamanaka, Eli": 29, "Friend, Travis": 30, "Johnson, Alec": 31,
    "Gamboa, Alex": 32, "Mhoon, Andrew": 33, "Wadas, Zach": 34,
    "Ortiz, Jordan": 35, "Chavez, Alex": 36, "Fried, Jacob": 37,
    "Whitton, Cooper": 41, "Bender, Zach": 42, "Stucky, Cole": 45,
    "Carmona Jr., Jose": 46, "Gabay, Lucas": 47, "Laine, Avery": 55,
    "Gurney, Win": 99,
}

LMU_POSITION = {
    # Hitters / position players
    "Moreno, Matthew": "INF/RHP", "Wall, Jaxson": "UTL", "Malone, Noah": "OF",
    "Dunn, JD": "OF", "Estrella, Andrew": "INF", "Williams, Zion": "OF",
    "Casale, Johnny": "OF/LHP", "Stiveson, Nate": "UTL", "Lyall, Jake": "C",
    "Ghiorso, DJ": "INF", "Danos, Luca": "INF", "Klosek, Richie": "OF/1B",
    "Whitton, Cooper": "OF", "Wadas, Zach": "OF/1B", "Gamboa, Alex": "INF",
    "Mhoon, Andrew": "INF", "Friend, Travis": "OF", "Ortiz, Jordan": "C",
    "Gurney, Win": "OF/1B", "Erdmann, Eric": "1B/RHP", "Aguirre, David": "INF",
    "Carmona Jr., Jose": "C", "Gabay, Lucas": "INF",
    # Pitchers
    "Behrens, Adam": "RHP", "Laine, Avery": "RHP", "Geis, Jake": "RHP",
    "Johnson, Jonah": "RHP", "Warady, Tanner": "RHP", "Fried, Jacob": "RHP",
    "Bender, Zach": "RHP", "Singh, Dylan": "LHP", "Elward, Luke": "RHP",
    "Riera, Niko": "RHP", "Johnson, Alec": "LHP", "Champion, Matt": "INF/RHP",
    "Schneider, Max": "RHP", "DenDekker, Zach": "RHP", "Sweeney, Caleb": "LHP",
    "Yamanaka, Eli": "RHP", "Jacobsen, Gavin": "LHP", "Chavez, Alex": "RHP",
    "Stucky, Cole": "RHP", "Gallegos, Elliot": "RHP",
}

# ── Lehigh 2026 Roster ────────
LEHIGH_TEAM_NAME = "Lehigh"
LEHIGH_COMBINED_TEAM_ID = "LEHIGH_COMBINED_2026"
LEHIGH_TM_TEAM_ID = 730333184

# Transfers: "Last, First" -> previous school search name.
# Players NOT in this dict are assumed to have been at Lehigh for 2025.
LEHIGH_TRANSFERS = {
    "Fairhurst, Sam": "Catholic University",
    "Esposito, Joey": "Rutgers",
    # Remaining new pitchers (freshmen / JC — no D1 2025 data):
    # Coughlin, Brandon; Nell, Christopher; Wilkes, Declan;
    # Shepelsky, Jackson; Holman, Ryan; ONeill, Shane
}

LEHIGH_ROSTER_2026 = {
    # Returning pitchers (on 2025 roster)
    "Kochanowicz, Cole", "Leaman, Cole", "Andolina, David",
    "Mulvehill, Jake", "O'Hearen, Liam", "Treonze, Max",
    "Gyauch-Quirk, Noah", "Gariano, Ralph",
    # New pitchers in 2026
    "Coughlin, Brandon", "Nell, Christopher", "Wilkes, Declan",
    "Shepelsky, Jackson", "Holman, Ryan", "Fairhurst, Sam", "ONeill, Shane",
    "Ermigiotti, Julio", "Correll, Logan", "Hayden, Nick",
    # Returning hitters
    "Quinn, Aidan", "Patrizi, Dom", "Betances, Edwin", "Golier, Grady",
    "Wirtz, Ian", "Frankovic, Jack", "Adelman, Jasper", "Esposito, Joey",
    "Tsiaras, Matt", "Walewander, Owen", "Rogers, Raffaele", "Carvelli, Robbie",
    "Cochran, Ryan", "Davis, Ryan", "Crawford, Trystan", "Lamar, Tommy",
    "Kleckner, Bobby",
    # New hitters in 2026
    "Dantoni, Max", "Ivy, Parker", "Grasso, Robby", "Giardina, Silvio",
    # Two-way players
    "Ahearn, Cadeyrn",
}

LEHIGH_JERSEY = {
    "Quinn, Aidan": 1, "Patrizi, Dom": 2, "Cochran, Ryan": 3,
    "Giardina, Silvio": 4, "Walewander, Owen": 5, "Crawford, Trystan": 6,
    "Coughlin, Brandon": 7, "ONeill, Shane": 8, "Frankovic, Jack": 9,
    "Dantoni, Max": 10, "Lamar, Tommy": 11, "Adelman, Jasper": 12,
    "Fairhurst, Sam": 13, "Golier, Grady": 14, "Rogers, Raffaele": 15,
    "Kleckner, Bobby": 16, "Grasso, Robby": 17, "Carvelli, Robbie": 18,
    "Holman, Ryan": 19, "O'Hearen, Liam": 20, "Leaman, Cole": 21,
    "Esposito, Joey": 22, "Davis, Ryan": 23, "Shepelsky, Jackson": 24,
    "Ermigiotti, Julio": 25, "Betances, Edwin": 27, "Treonze, Max": 28,
    "Wilkes, Declan": 29, "Ivy, Parker": 30, "Correll, Logan": 31,
    "Hayden, Nick": 32, "Wirtz, Ian": 34, "Gariano, Ralph": 37,
    "Tsiaras, Matt": 38, "Kochanowicz, Cole": 40, "Andolina, David": 41,
    "Gyauch-Quirk, Noah": 42, "Mulvehill, Jake": 43, "Ahearn, Cadeyrn": 44,
    "Nell, Christopher": 45,
}

LEHIGH_POSITION = {
    # Pure pitchers
    "Kochanowicz, Cole": "RHP", "Leaman, Cole": "RHP", "Andolina, David": "RHP",
    "Mulvehill, Jake": "LHP", "O'Hearen, Liam": "RHP", "Treonze, Max": "RHP",
    "Gariano, Ralph": "LHP", "Nell, Christopher": "RHP", "Wilkes, Declan": "RHP",
    "Shepelsky, Jackson": "RHP", "Holman, Ryan": "RHP", "Fairhurst, Sam": "RHP",
    "ONeill, Shane": "RHP", "Ermigiotti, Julio": "RHP", "Correll, Logan": "LHP",
    "Hayden, Nick": "RHP",
    # Two-way / utility (get hitter reports)
    "Gyauch-Quirk, Noah": "P/OF/DH", "Coughlin, Brandon": "RHP/OF",
    "Frankovic, Jack": "UT/RHP", "Ahearn, Cadeyrn": "P/IF",
    "Golier, Grady": "RHP/UTL", "Lamar, Tommy": "RHP/UTL",
    "Grasso, Robby": "OF/INF/P",
    # Hitters / position players
    "Quinn, Aidan": "IF", "Patrizi, Dom": "IF", "Betances, Edwin": "IF",
    "Wirtz, Ian": "C", "Adelman, Jasper": "OF/1B", "Esposito, Joey": "C",
    "Tsiaras, Matt": "C", "Walewander, Owen": "C", "Rogers, Raffaele": "IF",
    "Carvelli, Robbie": "OF", "Cochran, Ryan": "IF", "Davis, Ryan": "OF",
    "Crawford, Trystan": "IF/OF", "Dantoni, Max": "INF", "Ivy, Parker": "C",
    "Giardina, Silvio": "INF", "Kleckner, Bobby": "OF",
}

# ── Gardner-Webb 2026 Roster ────────
GW_TEAM_NAME = "Gardner Webb"
GW_COMBINED_TEAM_ID = "GW_COMBINED_2026"

# D1 transfers: "Last, First" -> previous school search name.
# Players NOT in this dict are assumed to have been at Gardner-Webb for 2025.
GW_TRANSFERS = {
    "Carter, Merik": "Alabama-Huntsville",
    "Thompson, Zack": "Mercer",
    "Ector, Thad": "UNC Charlotte",
    "Mako, Chance": "North Carolina State",
    "Ellison, Oliver": "Coastal Carolina",
    "Ripepi, Drew": "Pittsburgh",
    "Umbach, Anthony": "Southern Indiana",
    "Agosto, Kelvin": "Alabama State",
    "Hobb, Jerek": "Stony Brook",
    "Shealor, Bennett": "George Mason",
    "Stockton, Jaden": "Florida State",
    "Paz, Sebastian": "Delaware State",
    "Manley, Matt": "Samford",
    "Hausner, Anthony": "The Citadel",
    "Iannibelli, Jack": "Stonehill",
    "Bitter, Jaden": "Ball State",
    "Lysik, Brendan": "Texas Tech",
    "Eldridge, Brandon": "Western Carolina",
}

GW_ROSTER_2026 = {
    "Carter, Merik", "Rossow, Ethan", "Camarillo, Allan", "Thompson, Zack",
    "Putnam, Colby", "Niehus, Patrick", "Ector, Thad", "Mako, Chance",
    "Bertram, Reid", "Ellison, Oliver", "Liao, Ethan", "Ripepi, Drew",
    "Dixon, Parker", "Kennell, Ryan", "Stanzione, Joe", "Ilgenfritz, Matt",
    "Umbach, Anthony", "Stuart, Matt", "Agosto, Kelvin", "Hobb, Jerek",
    "Shealor, Bennett", "Humphries, Colby", "Smith, Daniel", "Stockton, Jaden",
    "Littrell, Jude", "Pressley, Devin", "Paz, Sebastian", "Busson, Connor",
    "Manley, Matt", "Gentile, Josh", "Hausner, Anthony", "Rawlings, Burton",
    "Graydon, Trey", "Winters, Dean", "Robinson, Zachary", "Guevara, Diego",
    "Iannibelli, Jack", "Blaszczak, Nick", "Lopez, Marco", "Bitter, Jaden",
    "Sanchez, Alejandro", "Murcer, Holden", "Murcer, Jackson", "Maurer, Abe",
    "Lysik, Brendan", "Acuna, Jesus", "Angelakos, Nicholas", "Cox, Miller",
    "Creech, Marion", "Eldridge, Brandon", "Emswiler, Caleb",
}

GW_JERSEY = {
    "Carter, Merik": 1, "Rossow, Ethan": 2, "Camarillo, Allan": 3,
    "Thompson, Zack": 4, "Putnam, Colby": 5, "Niehus, Patrick": 6,
    "Ector, Thad": 7, "Mako, Chance": 8, "Bertram, Reid": 10,
    "Ellison, Oliver": 11, "Liao, Ethan": 12, "Ripepi, Drew": 13,
    "Dixon, Parker": 14, "Kennell, Ryan": 15, "Stanzione, Joe": 16,
    "Ilgenfritz, Matt": 18, "Umbach, Anthony": 20, "Stuart, Matt": 21,
    "Agosto, Kelvin": 23, "Hobb, Jerek": 24, "Shealor, Bennett": 25,
    "Humphries, Colby": 26, "Smith, Daniel": 28, "Stockton, Jaden": 29,
    "Littrell, Jude": 30, "Pressley, Devin": 31, "Paz, Sebastian": 32,
    "Busson, Connor": 33, "Manley, Matt": 34, "Gentile, Josh": 35,
    "Hausner, Anthony": 36, "Rawlings, Burton": 37, "Graydon, Trey": 38,
    "Winters, Dean": 40, "Robinson, Zachary": 41, "Guevara, Diego": 44,
    "Iannibelli, Jack": 45, "Blaszczak, Nick": 46, "Lopez, Marco": 47,
    "Bitter, Jaden": 48, "Sanchez, Alejandro": 49, "Murcer, Holden": 51,
    "Murcer, Jackson": 52, "Maurer, Abe": 54, "Lysik, Brendan": 55,
}

GW_POSITION = {
    # Hitters / position players
    "Carter, Merik": "SS", "Rossow, Ethan": "UTL", "Camarillo, Allan": "INF",
    "Thompson, Zack": "C", "Niehus, Patrick": "INF", "Ector, Thad": "OF",
    "Liao, Ethan": "UTL", "Ripepi, Drew": "INF", "Kennell, Ryan": "INF",
    "Stanzione, Joe": "INF", "Ilgenfritz, Matt": "INF", "Umbach, Anthony": "INF",
    "Agosto, Kelvin": "OF", "Shealor, Bennett": "C", "Smith, Daniel": "OF",
    "Littrell, Jude": "OF", "Paz, Sebastian": "UTL", "Hausner, Anthony": "INF",
    "Robinson, Zachary": "UTL", "Guevara, Diego": "OF", "Lopez, Marco": "C",
    "Sanchez, Alejandro": "OF", "Murcer, Holden": "C", "Maurer, Abe": "C",
    "Acuna, Jesus": "OF", "Cox, Miller": "UTL",
    # Pitchers
    "Putnam, Colby": "RHP", "Mako, Chance": "RHP", "Bertram, Reid": "LHP",
    "Ellison, Oliver": "RHP", "Dixon, Parker": "RHP", "Stuart, Matt": "RHP",
    "Hobb, Jerek": "LHP", "Humphries, Colby": "RHP", "Stockton, Jaden": "RHP",
    "Pressley, Devin": "RHP", "Busson, Connor": "RHP", "Manley, Matt": "RHP",
    "Gentile, Josh": "RHP", "Rawlings, Burton": "RHP", "Graydon, Trey": "RHP",
    "Winters, Dean": "LHP", "Iannibelli, Jack": "LHP", "Blaszczak, Nick": "RHP",
    "Bitter, Jaden": "RHP", "Murcer, Jackson": "RHP", "Lysik, Brendan": "LHP",
    "Angelakos, Nicholas": "RHP", "Creech, Marion": "LHP",
    "Eldridge, Brandon": "RHP", "Emswiler, Caleb": "RHP",
}

# ── Wake Forest 2026 Roster ────────
WF_TEAM_NAME = "Wake Forest"
WF_COMBINED_TEAM_ID = "WF_COMBINED_2026"
WF_TM_TEAM_ID = 730373376

WF_TRANSFERS = {
    # D1 transfers (new in 2026, previous D1 school for 2025 data)
    "Schaaf, Blake": "Georgetown",
    "Figueroa, Tyler": "Appalachian State",
    "Bagwell, Cam": "UNC Wilmington",
    "Miller, Jackson": "Ole Miss",
    "Torres, Boston": "VMI",
}

WF_ROSTER_2026 = {
    # Returning from 2025
    "Baxter, Cuyler", "Wentz, Dalton", "Ray, Will", "Morningstar, Blake",
    "Lewis, Kade", "Conte, Matt", "Hawke, Austin", "Dallas, Matthew",
    "Costello, Luke", "Williams, Javar", "Whysong, Nate", "Johnston, Zach",
    "Levonas, Chris", "Dressler, Troy", "Schmolke, Luke", "Bowie, Rhys",
    "Marsten, Duncan", "Keenan, Jimmy", "Billings, Luke", "Preisano, Ryan",
    # New in 2026 — D1 transfers
    "Schaaf, Blake", "Figueroa, Tyler", "Bagwell, Cam", "Miller, Jackson",
    "Torres, Boston",
    # New in 2026 — freshmen / prep
    "Stein, JD", "Harsch, Marcelo", "Jones, Evan", "Roper, Jackson",
    "Brennecke, Ryan", "Nicholson, Grant", "Costello, Andrew",
    "Wood, Tyler", "Serrano, Jordan", "Bosch, Ryan",
    "Rubino, Nick",
}

WF_JERSEY = {
    "Baxter, Cuyler": 0, "Wentz, Dalton": 1, "Ray, Will": 2,
    "Stein, JD": 3, "Morningstar, Blake": 4, "Harsch, Marcelo": 5,
    "Lewis, Kade": 6, "Schaaf, Blake": 7, "Conte, Matt": 8,
    "Hawke, Austin": 9, "Dallas, Matthew": 10, "Costello, Luke": 11,
    "Figueroa, Tyler": 12, "Bagwell, Cam": 13, "Williams, Javar": 14,
    "Whysong, Nate": 15, "Jones, Evan": 16, "Johnston, Zach": 17,
    "Levonas, Chris": 18, "Dressler, Troy": 19, "Miller, Jackson": 21,
    "Roper, Jackson": 22, "Brennecke, Ryan": 23, "Nicholson, Grant": 24,
    "Torres, Boston": 25, "Costello, Andrew": 26, "Schmolke, Luke": 27,
    "Bowie, Rhys": 28, "Wood, Tyler": 29, "Marsten, Duncan": 30,
    "Rubino, Nick": 33, "Keenan, Jimmy": 34, "Billings, Luke": 35,
    "Serrano, Jordan": 40, "Preisano, Ryan": 51, "Bosch, Ryan": 99,
}

WF_POSITION = {
    # Hitters / position players
    "Baxter, Cuyler": "INF", "Wentz, Dalton": "INF", "Lewis, Kade": "INF",
    "Schaaf, Blake": "INF", "Conte, Matt": "C", "Hawke, Austin": "INF",
    "Costello, Luke": "INF", "Figueroa, Tyler": "INF/OF",
    "Williams, Javar": "OF", "Miller, Jackson": "OF", "Roper, Jackson": "INF",
    "Torres, Boston": "OF", "Costello, Andrew": "C", "Rubino, Nick": "INF",
    "Keenan, Jimmy": "C", "Serrano, Jordan": "OF", "Preisano, Ryan": "INF",
    "Stein, JD": "INF",
    # Pitchers
    "Ray, Will": "RHP", "Morningstar, Blake": "RHP", "Harsch, Marcelo": "RHP",
    "Dallas, Matthew": "LHP", "Whysong, Nate": "RHP", "Jones, Evan": "RHP",
    "Johnston, Zach": "LHP", "Levonas, Chris": "RHP", "Dressler, Troy": "RHP",
    "Brennecke, Ryan": "LHP", "Nicholson, Grant": "RHP", "Schmolke, Luke": "RHP",
    "Bowie, Rhys": "LHP", "Wood, Tyler": "RHP", "Marsten, Duncan": "RHP",
    "Billings, Luke": "RHP", "Bosch, Ryan": "LHP", "Bagwell, Cam": "RHP",
}

# ── Fordham 2026 Roster ────────
FORDHAM_TEAM_NAME = "Fordham"
FORDHAM_COMBINED_TEAM_ID = "FORDHAM_COMBINED_2026"

# D1 transfers: "Last, First" -> previous school search name.
FORDHAM_TRANSFERS = {
    "Donnelly, Joey": "California",
}

FORDHAM_ROSTER_2026 = {
    "Dieguez, Matt", "Scarlata, Anthony", "Beaudreau, Bradley", "Pino, A.J.",
    "Ocko, Madden", "Little, Ernie", "Berg, Aric", "Stewart, Robbie",
    "Forney, Colden", "Elson, Beau", "Kirk, Taylor", "Chavez, Carson",
    "Vieira, Will", "Markey, Tommy", "Dowd, Aidan", "Cawley, Declan",
    "Smith, Koen", "Chavez, Caden", "Rodarte, Jordan", "Kapica, Andrew",
    "Egan, Ryan", "Young, Caden", "Redick, James", "Grabau, Anthony",
    "Donnelly, Joey", "Murray, Alec", "Rubin, Jack", "Reilly-Bell, Diego",
    "Morello, Mason", "Dean, Mason", "Travaglia, Giacomo", "Swaim, Carson",
    "Hanawalt, Chase", "Osterhus, Eric", "McAndrews, Tommy",
    "O'Brien-Gonzalez, Aidan", "Ford, Tim",
}

FORDHAM_JERSEY = {
    "Dieguez, Matt": 1, "Scarlata, Anthony": 3, "Beaudreau, Bradley": 4,
    "Pino, A.J.": 5, "Ocko, Madden": 6, "Little, Ernie": 7,
    "Berg, Aric": 8, "Stewart, Robbie": 9, "Forney, Colden": 10,
    "Elson, Beau": 11, "Kirk, Taylor": 12, "Chavez, Carson": 13,
    "Vieira, Will": 14, "Markey, Tommy": 15, "Dowd, Aidan": 16,
    "Cawley, Declan": 17, "Smith, Koen": 18, "Chavez, Caden": 19,
    "Rodarte, Jordan": 20, "Kapica, Andrew": 22, "Egan, Ryan": 23,
    "Young, Caden": 24, "Redick, James": 25, "Grabau, Anthony": 26,
    "Donnelly, Joey": 27, "Murray, Alec": 28, "Rubin, Jack": 29,
    "Reilly-Bell, Diego": 30, "Morello, Mason": 31, "Dean, Mason": 32,
    "Travaglia, Giacomo": 33, "Swaim, Carson": 34, "Hanawalt, Chase": 35,
    "Osterhus, Eric": 36, "McAndrews, Tommy": 37,
    "O'Brien-Gonzalez, Aidan": 40, "Ford, Tim": 46,
}

FORDHAM_POSITION = {
    # Hitters / position players
    "Dieguez, Matt": "IF", "Beaudreau, Bradley": "IF", "Ocko, Madden": "IF",
    "Little, Ernie": "OF", "Forney, Colden": "IF", "Kirk, Taylor": "IF/OF",
    "Chavez, Carson": "C", "Markey, Tommy": "OF", "Young, Caden": "C",
    "Grabau, Anthony": "IF", "Donnelly, Joey": "OF", "Rubin, Jack": "OF/1B",
    "Dean, Mason": "OF", "McAndrews, Tommy": "C", "Ford, Tim": "IF",
    # Pitchers
    "Scarlata, Anthony": "RHP", "Pino, A.J.": "LHP", "Berg, Aric": "RHP",
    "Stewart, Robbie": "RHP", "Elson, Beau": "RHP", "Vieira, Will": "RHP",
    "Dowd, Aidan": "RHP", "Cawley, Declan": "RHP", "Smith, Koen": "RHP",
    "Chavez, Caden": "LHP", "Rodarte, Jordan": "RHP", "Kapica, Andrew": "RHP",
    "Egan, Ryan": "RHP", "Redick, James": "RHP", "Murray, Alec": "RHP",
    "Reilly-Bell, Diego": "RHP", "Morello, Mason": "RHP",
    "Travaglia, Giacomo": "RHP", "Swaim, Carson": "RHP",
    "Hanawalt, Chase": "LHP", "Osterhus, Eric": "RHP",
    "O'Brien-Gonzalez, Aidan": "RHP",
}

NAME_MAP = {
    "Laughlin, Theo": "Loughlin, Theo",
    "Laughlin , Theo": "Loughlin, Theo",
    "Daly, Jameson": "Daly, Jamie",
    "Hall, Edward": "Hall, Ed",
    "Hamilton, Matthew": "Hamilton, Matt",
    "Edwards, Scott": "Edwards, Scotty",
    "Edwards , Scott": "Edwards, Scotty",
    "Edwards , Scotty": "Edwards, Scotty",
    "Lietz, Foresst": "Lietz, Forrest",
    "McCoullough, Will": "McCullough, Will",
}

# ── Pitch colors & mappings ───────────────────
PITCH_COLORS = {
    "Fastball": "#d22d49", "Sinker": "#fe6100", "Cutter": "#933f8e",
    "Slider": "#f7c631", "Curveball": "#00d1ed", "Changeup": "#1dbe3a",
    "Splitter": "#c99b6e", "Knuckle Curve": "#2d7fc1", "Sweeper": "#dbab00",
    "Other": "#aaaaaa",
}

PITCH_TYPE_MAP = {
    "FourSeamFastBall": "Fastball",
    "OneSeamFastBall": "Sinker",
    "TwoSeamFastBall": "Sinker",
    "ChangeUp": "Changeup",
    "Sweeper": "Slider",
    "Knuckle Curve": "Curveball",
    "Knuckleball": "Other",
    "Undefined": "Other",
    "UN": "Other",
    "Unknown": "Other",
    "UNK": "Other",
}
PITCH_TYPES_TO_DROP = {"Other", "Undefined", "UN", "Unknown", "UNK"}

TM_PITCH_PCT_COLS = {
    "4Seam%": "Fastball", "Sink2Seam%": "Sinker", "Cutter%": "Cutter",
    "Slider%": "Slider", "Curve%": "Curveball", "Change%": "Changeup",
    "Split%": "Splitter", "Sweeper%": "Sweeper",
}

# ── Strike zone constants ──────────────────────
ZONE_SIDE = 0.83
ZONE_HEIGHT_BOT = 1.5
ZONE_HEIGHT_TOP = 3.5
MIN_CALLED_STRIKES_FOR_ADAPTIVE_ZONE = 20
PLATE_SIDE_MAX = 2.5
PLATE_HEIGHT_MIN = 0.0
PLATE_HEIGHT_MAX = 5.5
MIN_PITCH_USAGE_PCT = 5.0
MIN_TUNNEL_SEQ_PCT = 10.0

# ── Swing / contact call lists ─────────────────
SWING_CALLS = ["StrikeSwinging", "FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]
CONTACT_CALLS = ["FoulBall", "FoulBallNotFieldable", "FoulBallFieldable", "InPlay"]

# SQL-ready versions
_SWING_CALLS_SQL = "('StrikeSwinging','FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay')"
_CONTACT_CALLS_SQL = "('FoulBall','FoulBallNotFieldable','FoulBallFieldable','InPlay')"
_IZ_COND = "ABS(PlateLocSide) <= 0.83 AND PlateLocHeight BETWEEN 1.5 AND 3.5"
_HAS_LOC = "PlateLocSide IS NOT NULL AND PlateLocHeight IS NOT NULL"
_OZ_COND = f"NOT ({_IZ_COND}) AND {_HAS_LOC}"

# ── Stuff+ weights per pitch type ──────────────
STUFF_WEIGHTS = {
    "Fastball":       {"RelSpeed": 2.0, "InducedVertBreak": 2.5, "HorzBreak": 0.3, "Extension": 0.5, "VertApprAngle": 2.5, "SpinRate": 1.0},
    "Sinker":         {"RelSpeed": 2.5, "InducedVertBreak": -0.5, "HorzBreak": 1.5, "Extension": 0.5, "VertApprAngle": -1.5, "SpinRate": 0.8},
    "Cutter":         {"RelSpeed": 0.8, "InducedVertBreak": 0.3, "HorzBreak": -1.5, "Extension": -1.0, "VertApprAngle": -0.5, "SpinRate": 2.0},
    "Slider":         {"RelSpeed": 1.0, "InducedVertBreak": -0.5, "HorzBreak": 1.0, "Extension": 0.3, "VertApprAngle": -2.5, "SpinRate": 1.5},
    "Curveball":      {"RelSpeed": 1.5, "InducedVertBreak": -1.5, "HorzBreak": -1.5, "Extension": -1.5, "VertApprAngle": -2.0, "SpinRate": 0.5},
    "Changeup":       {"RelSpeed": 0.5, "InducedVertBreak": 1.5, "HorzBreak": 1.0, "Extension": 0.5, "VertApprAngle": -2.5, "SpinRate": 1.0, "VeloDiff": 2.0},
    "Sweeper":        {"RelSpeed": 1.5, "InducedVertBreak": -1.0, "HorzBreak": 2.0, "Extension": 0.8, "VertApprAngle": -1.5, "SpinRate": 0.5},
    "Splitter":       {"RelSpeed": 1.0, "InducedVertBreak": -2.0, "HorzBreak": 0.5, "Extension": 1.0, "VertApprAngle": -2.0, "SpinRate": -0.3, "VeloDiff": 1.5},
    "Knuckle Curve":  {"RelSpeed": 1.5, "InducedVertBreak": -1.5, "HorzBreak": -1.5, "Extension": -1.5, "VertApprAngle": -2.0, "SpinRate": 0.5},
}
STUFF_WEIGHTS_DEFAULT = {"RelSpeed": 1.0, "InducedVertBreak": 1.0, "HorzBreak": 1.0, "Extension": 1.0, "VertApprAngle": 1.0, "SpinRate": 1.0}


# ── Utility functions ──────────────────────────

def _norm_name_sql(col):
    """Normalize whitespace and comma spacing in SQL."""
    return (
        f"regexp_replace("
        f"regexp_replace("
        f"regexp_replace(trim({col}), '\\\\s+', ' '),"
        f"'\\\\s+,', ','),"
        f"',\\\\s*', ', ')"
    )


def _name_case_sql(col):
    """SQL CASE expression that normalizes player names to match NAME_MAP."""
    norm = _norm_name_sql(col)
    def _esc(s):
        return s.replace("'", "''")
    parts = " ".join(f"WHEN {norm} = '{_esc(old)}' THEN '{_esc(new)}'" for old, new in NAME_MAP.items())
    return f"CASE {parts} ELSE {norm} END"


def _name_sql(col):
    """Alias for _name_case_sql."""
    return _name_case_sql(col)


def _normalize_hand(series):
    """Normalize handedness to Left/Right; others become NA."""
    s = series.astype(str).str.strip()
    s = s.replace({"L": "Left", "R": "Right", "B": "Both"})
    s = s.where(s.isin(["Left", "Right"]))
    return s


def safe_mode(series, default=""):
    """Return the mode of a Series, or *default* if no mode exists."""
    m = series.mode()
    return m.iloc[0] if len(m) > 0 else default


def is_barrel(ev, la):
    """Statcast barrel definition."""
    if pd.isna(ev) or pd.isna(la):
        return False
    if ev < 98:
        return False
    la_min = max(26 - 2 * (ev - 98), 8)
    la_max = min(30 + 3 * (ev - 98), 50)
    return la_min <= la <= la_max


def is_barrel_mask(df):
    """Vectorised Statcast barrel mask for a DataFrame with ExitSpeed & Angle."""
    ev = pd.to_numeric(df["ExitSpeed"], errors="coerce")
    la = pd.to_numeric(df["Angle"], errors="coerce")
    la_min = (26 - 2 * (ev - 98)).clip(lower=8)
    la_max = (30 + 3 * (ev - 98)).clip(upper=50)
    return (ev >= 98) & (la >= la_min) & (la <= la_max)


def in_zone_mask(df, batter_zones=None, batter_col="Batter"):
    """Per-pitch boolean mask: True if pitch is inside the batter's strike zone."""
    valid_loc = (
        df["PlateLocSide"].between(-PLATE_SIDE_MAX, PLATE_SIDE_MAX) &
        df["PlateLocHeight"].between(PLATE_HEIGHT_MIN, PLATE_HEIGHT_MAX)
    )
    side_ok = df["PlateLocSide"].abs() <= ZONE_SIDE
    if batter_zones and batter_col in df.columns:
        bot = df[batter_col].map(lambda b: batter_zones.get(b, (ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP))[0])
        top = df[batter_col].map(lambda b: batter_zones.get(b, (ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP))[1])
        height_ok = (df["PlateLocHeight"] >= bot) & (df["PlateLocHeight"] <= top)
    else:
        height_ok = df["PlateLocHeight"].between(ZONE_HEIGHT_BOT, ZONE_HEIGHT_TOP)
    return (valid_loc & side_ok & height_ok).fillna(False)


def normalize_pitch_types(df):
    """Normalize pitch type names and null out junk/undefined."""
    if "TaggedPitchType" not in df.columns:
        return df
    df = df.copy()
    df["TaggedPitchType"] = df["TaggedPitchType"].replace(PITCH_TYPE_MAP)
    df.loc[df["TaggedPitchType"].isin(PITCH_TYPES_TO_DROP), "TaggedPitchType"] = np.nan
    return df


def filter_minor_pitches(df, min_pct=5.0):
    """Remove pitch types that make up less than min_pct% of a pitcher's arsenal."""
    if df.empty or "TaggedPitchType" not in df.columns:
        return df
    df = df[df["TaggedPitchType"].notna()]
    total = len(df)
    if total == 0:
        return df
    counts = df["TaggedPitchType"].value_counts()
    keep = counts[counts / total * 100 >= min_pct].index
    return df[df["TaggedPitchType"].isin(keep)]


def _is_position_player(name):
    """True if player is a position player (not a pure pitcher)."""
    pos = POSITION.get(name, "")
    return pos not in ("RHP", "LHP")


def filter_davidson(data, role="pitcher"):
    if role == "pitcher":
        return data[(data["PitcherTeam"] == DAVIDSON_TEAM_ID) & (data["Pitcher"].isin(ROSTER_2026))].copy()
    else:
        return data[(data["BatterTeam"] == DAVIDSON_TEAM_ID) & (data["Batter"].isin(ROSTER_2026))].copy()


def display_name(name, escape_html=True):
    import html as html_mod
    if not name:
        return "Unknown"
    parts = name.split(", ")
    result = f"{parts[1]} {parts[0]}" if len(parts) == 2 else name
    return html_mod.escape(result) if escape_html else result


def tm_name_to_trackman(full_name):
    """Convert TrueMedia 'First Last' or 'First Last Jr.' to Trackman 'Last, First' / 'Last, First Jr.'."""
    if not full_name or not isinstance(full_name, str):
        return full_name
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    suffixes = {"Jr.", "Jr", "Sr.", "Sr", "II", "III", "IV", "V"}
    suffix = ""
    if parts[-1] in suffixes:
        suffix = " " + parts[-1]
        parts = parts[:-1]
    if len(parts) < 2:
        return full_name
    first = parts[0]
    last = " ".join(parts[1:])
    return f"{last}, {first}{suffix}"


def get_percentile(value, series):
    from scipy.stats import percentileofscore
    if pd.isna(value) or series.dropna().empty:
        return np.nan
    return percentileofscore(series.dropna(), value, kind='rank')


# ── Team Name Mapping ────────────────────────────
TEAM_NAMES = {
    "DAV_WIL": "Davidson", "BRY_BUL": "Bryant", "DUK_BLU": "Duke",
    "CLE_TIG": "Clemson", "VCU_RAM": "VCU", "RIC_SPI": "Richmond",
    "FOR_RAM": "Fordham", "DAY_FLY": "Dayton", "SIE_SAI": "Siena",
    "STJ_HAW": "Saint Joseph's", "STB_BON": "St. Bonaventure",
    "LAF_LEP": "Lafayette", "GEO_PAT": "George Mason",
    "GEO_COL": "Georgetown", "RHO_RAM": "Rhode Island",
    "SLU_BILL": "Saint Louis", "VIL_WIL": "Villanova",
    "PEN_NIT": "Penn State", "UMA_AMH": "UMass",
    "BRO_BEA": "Brown", "HIG_PAN": "High Point",
    "GAR_RUN": "Gardner-Webb", "WOF_TER": "Wofford",
    "WIN_EAG": "Winthrop", "CHA_FOR": "Charleston",
    "MOR_EAG": "Morehead St.", "NOR_AGG": "NC A&T",
    "BUC_BIS": "Bucknell", "ERS_COL": "Erskine",
    "SOU_GAM": "South Carolina", "QUN_RYL": "Quinnipiac",
    "IND_SPI8": "Indiana St.", "ONT_BLU": "Ontario",
    "202_ONT1": "Ontario",
    "FAI_STA": "Fairfield",
}


def _friendly_team_name(team_id):
    """Map Trackman team ID to a readable name."""
    if not team_id or pd.isna(team_id):
        return "?"
    return TEAM_NAMES.get(str(team_id), str(team_id))
