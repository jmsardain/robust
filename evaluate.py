from models import OpenAIClient, QwenClient
import re
import sqlparse
import sqlite3
from tqdm import tqdm
import os 
from huggingface_hub import login

def schema_to_str(schema: dict) -> str:
    lines = []
    for table, cols in schema.items():
        lines.append(f"- {table}({', '.join(cols)})")
    return "\n".join(lines)

def normalize_sql(sql: str) -> str:
    if not sql:
        return ""
    s = sql.strip()
    s = sqlparse.format(s, keyword_case="upper", strip_comments=True, reindent=True)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def strip_code_fences(s: str) -> str:
    """
    Remove Markdown code fences like ```sql ... ``` or ``` ... ```
    """
    match = re.search(r"```sql\n(.*?)\n```", s, re.DOTALL)
    if match:
        sql_code = match.group(1)
        return sql_code
    else:
        return ''

def execute_sqlite(db_path: str, sql: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("BEGIN")
    try:
        cur.execute(sql)
        rows = cur.fetchall()
    except Exception:
        rows = None
    conn.rollback()
    conn.close()
    return rows

def execution_match(db_path: str, gold_sql: str, pred_sql: str) -> bool:
    gold_rows = execute_sqlite(db_path, gold_sql)
    # print(f"HERE Gold!!!! {gold_rows}")
    pred_rows = execute_sqlite(db_path, pred_sql)
    # print(f"HERE PRED!!!! {pred_rows}")
    if gold_rows is None or pred_rows is None:
        return False

    # sort rows for deterministic comparison
    normalize = lambda rows: sorted([tuple(str(x) for x in row) for row in rows])
    return normalize(gold_rows) == normalize(pred_rows)
    # return sorted(gold_rows) == sorted(pred_rows)

def execution_match_scalar(db_path: str, gold_sql: str, pred_sql: str) -> bool:
    def scalar(sql):
        rows = execute_sqlite(db_path, sql)
        if not rows or not rows[0]:
            return None
        return rows[0][0] # first column of first row
    return scalar(gold_sql) == scalar(pred_sql)

DEFAULT_PROMPT = (
"You are given a database schema and a natural language question.\n"
"Write a correct SQL query. Use only the provided tables/columns.\n\n"
"Schema: {schema}\n"
"Question: {question}\n"
"SQL:"
)


def evaluate(dataset, model="gpt-4o-mini", n=10):

    if 'gpt' in model:
        client = OpenAIClient(model=model, temperature=0)
    elif 'qwen':
        ACCESS_TOKEN=os.environ.get("HUGGINGFACE_HUB_TOKEN")
        try:
            if ACCESS_TOKEN:
                login(token=ACCESS_TOKEN)
                print( "Connecting to Hugging Face Hub.")
            else:
                print( "ACCESS_TOKEN not found")
        except Exception as e:
            print( f"Not connected to Hugging Face Hub: {e}")
        client = QwenClient(model='Qwen/Qwen2.5-Coder-0.5B-Instruct')
    else:
        return -999
    correct = 0
    # schema_str = schema_to_str(schema)

    denominator = (n if n != -1 else len(dataset))

    for i, ex in tqdm(enumerate(dataset), total=(n if n != -1 else len(dataset))):
        if n != -1 and i >= n:
            break

        question = ex["question"]
        gold_sql = ex["SQL"]
        schema_str = schema_to_str(ex["schema"])
        path_database = ex["db_path"]

        prompt = DEFAULT_PROMPT.format(schema=schema_str, question=question)
        pred = client.generate(prompt)
        # print(pred)
        # pred = pred.replace("```sql", "")
        # pred_sql = pred.replace("```", "")
        pred_sql = strip_code_fences(pred) ## models love to add markdown when it's a code .. get code in between markdown style

        # if normalize_sql(pred) == normalize_sql(gold_sql): ## semantic similarity is not really the best approach here  ....
            # correct += 1
        exec_correct = execution_match(path_database, normalize_sql(gold_sql), normalize_sql(pred_sql))
        # exec_correct = execution_match_scalar(path_database, normalize_sql(gold_sql), normalize_sql(pred))

        if exec_correct:
            correct += 1

        # print(f"Q: {question}\nGold: {gold_sql}\nPred: {pred}\n---")
    return correct / denominator
