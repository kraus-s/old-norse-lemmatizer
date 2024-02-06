import src.importer as importer
from src.config import *

parsed_docs_list = importer.import_menota_data(path= OLD_NORSE_CORPUS_FILES, test=False)
best_list = [x for x in parsed_docs_list if x.diplomatic and x.normalized]
print(f"Found {len(best_list)} documents with diplomatic and normalized tokens")
num_toks = sum([len(x.tokens) for x in best_list])
print(f"Found {num_toks} tokens in the best list")