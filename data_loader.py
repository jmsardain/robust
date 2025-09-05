from datasets import load_dataset
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
import sqlite3
import random
from commonPerturbations import _COMMON_ABBREVIATIONS, _COMMON_SYNONYMS, _COMMON_TRANSLATIONS

def load_bird_data(dataset_name):
    dataset = load_dataset("birdsql/bird_mini_dev")
    return dataset[dataset_name]

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
        return json.load(f)
#
#
###################################################################################################
def perturb_synonyms(schema: dict) -> dict:
    """
    Change some names with synonyms
    """
    new_tables = {}
    for t, cols in schema.items():
        new_t = random.choice(_COMMON_SYNONYMS.get(t.lower(), [t]))
        new_cols = [random.choice(_COMMON_SYNONYMS.get(c.lower(), [c])) for c in cols]
        new_tables[new_t] = new_cols
    return new_tables
#
###################################################################################################
## Milder perturbation
def abbrev(name: str) -> str:
    # return ''.join(c for c in name if c.lower() not in 'aeiou') or name ## too harsh
    ## a milder version
    lowercase = name.lower()
    for full, abbr in _COMMON_ABBREVIATIONS.items():
        if full in lowercase:
            return lowercase.replace(full, abbr)
    ## if abbreviation not in the dict, just truncate
    return lowercase[:max(3, len(lowercase)//2)]

def perturb_abbreviations(schema: dict) -> dict:
    """
    Implement abbreviations, change words based on common abbreviations
    """
    return {abbrev(t): [abbrev(c) for c in cols] for t, cols in schema.items()}

###################################################################################################
def perturb_typos(schema: dict) -> dict:
    """
    Implement a typo randomly anywhere in the string
    """
    def typo(name: str) -> str:
        if len(name) < 2: return name
        i = random.randint(0, len(name)-1)
        return name[:i] + random.choice('abcdefghijklmnopqrstuvwxyz') + name[i+1:]
    return {typo(t): [typo(c) for c in cols] for t, cols in schema.items()}
#
#
###################################################################################################
def perturb_translate(schema: dict) -> dict:
    """
    Change some names with synonyms
    """
    new_tables = {}
    for t, cols in schema.items():
        new_t = random.choice(_COMMON_TRANSLATIONS.get(t.lower(), [t]))
        new_cols = [random.choice(_COMMON_TRANSLATIONS.get(c.lower(), [c])) for c in cols]
        new_tables[new_t] = new_cols
    return new_tables
###################################################################################################

def appendSchema(path:str, dataset: List[Dict[str, Any]], systematic: str) -> List[Dict[str, Any]]:

    dev_database_dict = [
        'california_schools', ##
        'card_games', ##
        'codebase_community', ##
        'debit_card_specializing', ##
        'european_football_2', ##
        'financial',
        'formula_1', ##
        'student_club', ##
        'superhero', ##
        'thrombosis_prediction', ##
        'toxicology', ##
    ]

    dataset = [entry for entry in dataset if entry["db_id"] != "financial"]

    for key in dataset:
        if key['db_id'] in dev_database_dict:
            db_path = f"{path}/dev_databases/{key['db_id']}/{key['db_id']}.sqlite"
            key['schema']  = getSchema(db_path)
            if systematic == "typo":
                key['schema'] = perturb_typos(key['schema'])
            if systematic == "abbrev":
                key['schema'] = perturb_abbreviations(key['schema'])
                # key['SQL'] = abbrev(key['SQL'])
            if systematic == "synonyms":
                key['schema'] = perturb_synonyms(key['schema'])
            if systematic == "translate":
                key['schema'] = perturb_translate(key['schema'])
            key['db_path'] = db_path

    return dataset

def getSchema(db_path: str) -> dict:

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # List tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]

    # List columns per table
    schema = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [row[1] for row in cur.fetchall()]
        schema[t] = cols

    conn.close()
    return schema
