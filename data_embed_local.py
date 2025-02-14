from dotenv import load_dotenv
from pymilvus import Collection
from src import milvus 
import os 
import argparse
import pandas as pd
import json
import numpy as np

def main(args):
    load_dotenv()
    ip_addr = os.getenv('ip_addr')
    with open(os.path.join(args.config_path, args.config_name)) as f:
        db_args = json.load(f)
    db_args['ip_addr'] = ip_addr

    milvus_db = milvus.MilvusEnvManager(db_args)
    milvus_db.set_env()
    print(f'ip: {milvus_db.ip_addr}')
    milvus_meta = milvus.MilvusMeta()
    milvus_meta.set_rulebook_map()
    id_code = milvus_meta.rulebook_id_code
    partition_to_kor = milvus_meta.rulebook_eng_to_kor
    print(id_code, partition_to_kor[args.partition_name])
    
    collection = Collection(args.collection_name)
    collection.load()

    data_df = pd.read_csv(os.path.join(args.output_dir, args.file_name), index_col=0)
    print(data_df.head(3), len(data_df))

    milvus_dp = milvus.DataMilVus(db_args)
    code = id_code[partition_to_kor[args.partition_name]]
    data_id = [code + str(id).zfill(4) for id in range(len(data_df))]   # 0001, 0002, ... 같은 4자리 아이디 생성
    print(data_id[:3])
    
    text = data_df['text'].values
    text_emb = milvus_dp.bge_embed_data(text)
    source = list(data_df['source'].values)
    page_no = list(data_df['page_no'].values)
    chunk_size = list(data_df['tok_chunk_size'].values)
    data = [data_id, text_emb, text, source, page_no, chunk_size]
    milvus_dp.insert_data(data, args.collection_name, args.partition_name)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='./config/')
    cli_parser.add_argument('--config_name', type=str, default='db_config.json')
    cli_parser.add_argument('--output_dir', type=str, default='./data/')
    cli_parser.add_argument('--file_name', type=str, default=None)
    cli_parser.add_argument('--collection_name', type=str, default='rule_book')
    cli_parser.add_argument('--partition_name', type=str, default=None)
    cli_argse = cli_parser.parse_args()
    main(cli_argse)