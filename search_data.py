from pymilvus import Collection, CollectionSchema, FieldSchema, utility
from dotenv import load_dotenv
from src import MilVus, DataMilVus, DataProcessor
import os 
import argparse
import json

def main(args):
    load_dotenv()
    db_name = 'finger'
    ip_addr = os.getenv('ip_addr')

    with open(os.path.join(args.config_path, args.config_name)) as f:
        db_args = json.load(f)
    db_args['ip_addr'] = ip_addr

    milvus_db = MilVus(db_args)
    milvus_db.set_env()
    print(f'client: {milvus_db.client}')
    
    collection = Collection(args.collection_name)
    collection.load()
    
    data_milvus = DataMilVus(db_args)
    # User Query
    text = "제2장 국내 출장 여비 "
    # cleansed_text = data_milvus.cleanse_text(text)
    # print(cleansed_text)
    query_emb = data_milvus.bge_embed_data(text)
    # print(query_emb)
    data_milvus.set_search_params()
    search_result = data_milvus.search_data(collection, query_emb, output_fields='text')
    print(search_result)
    # emb2 = search_result[0][0].entity.get('text_emb')
    # print(f'-' * 20)
    print(data_milvus.decode_search_result(search_result))
    # print(data_milvus.calc_emb_similarity(query_emb, emb2))

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='./config/')
    cli_parser.add_argument('--config_name', type=str, default='db_config.json')
    cli_parser.add_argument('--output_dir', type=str, default='./data/pdf/embed_output')
    cli_parser.add_argument('--file_name', type=str, default='취업규칙.csv')
    cli_parser.add_argument('--collection_name', type=str, default='rule_book')
    cli_parser.add_argument('--partition_name', type=str, default=None)
    cli_argse = cli_parser.parse_args()
    main(cli_argse)