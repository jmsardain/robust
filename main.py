import argparse
from data_loader import load_bird_data, read_jsonl, getSchema, appendSchema
from evaluate import evaluate
from tqdm import tqdm
import os 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument("--syst", default="nominal", help="nominal, typo, abbrev, synonyms, translate")
    parser.add_argument("--n", type=int, default=10, help="Number of questions. -1 means all")
    args = parser.parse_args()

    path = "/data/jmsardain/Projects/robust/data/minidev/MINIDEV/"
    dataset = read_jsonl('/data/jmsardain/Projects/robust/data/minidev/MINIDEV/mini_dev_sqlite.json')

    data = appendSchema(path, dataset, args.syst)
    acc = evaluate(data, model=args.model, n=args.n)
    
    print(f"For {args.syst} Accuracy: {acc}")
    with open("performance_"+args.model+".txt", "a") as f:
        f.write(f"For {args.syst} Accuracy: {acc}\n")
    
    # dev_database_dict = [
    #     'california_schools', ##
    #     'card_games', ##
    #     'codebase_community', ##
    #     'debit_card_specializing', ##
    #     'european_football_2', ##
    #     'formula_1', ##
    #     'student_club', ##
    #     'superhero', ##
    #     'thrombosis_prediction', ##
    #     'toxicology', ##
    # ]
    # path = "/Users/jmsardain/Downloads/minidev/MINIDEV/"
    # for dev in dev_database_dict:
    #     db_path = f"{path}/dev_databases/{dev}/{dev}.sqlite"
    #     schema = getSchema(db_path)
    #     print(f"Database {dev}\n Schema {schema}\n")

if __name__ == "__main__":

    # Disable parallelism in tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Limit PyTorch to 1 thread
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    main()
