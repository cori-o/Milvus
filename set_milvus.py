from dotenv import load_dotenv
from src import milvus, MilvusEnvManager
import os 
import argparse
import json

def main(args):
    load_dotenv()
    ip_addr = os.getenv('ip_addr')
    collection_name = 'rule_book'

    with open(os.path.join(args.config_path, args.config_name)) as f:
        db_args = json.load(f)
    db_args['ip_addr'] = ip_addr

    milvus_db = MilvusEnvManager(db_args)
    print(f'ip: {milvus_db.ip_addr}')

    milvus_db.set_env()
    milvus_db.create_db(db_args['db'])
    print(f'client: {milvus_db.client}')

    # define schema field 
    data_id = milvus_db.create_field_schema('id', dtype='str', is_primary=True, max_length=20)
    data_text_emb = milvus_db.create_field_schema('text_emb', dtype='float', is_primary=False, dim=1024)
    data_text = milvus_db.create_field_schema('text', dtype='str', is_primary=False, max_length=2048)
    data_source = milvus_db.create_field_schema('source', dtype='str', is_primary=False, max_length=200)
    schema_field_list = [data_id, data_text_emb, data_text, data_source]

    # create schema 
    schema = milvus_db.create_schema(schema_field_list, 'schema for ~')

    # create collection
    collection = milvus_db.create_collection(collection_name, schema, shards_num=2)
    milvus_db.get_collection_info(collection)
    milvus_db.create_index(collection, field_name='text_emb')   # text 필드에 index 생성 

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='./config/')
    cli_parser.add_argument('--config_name', type=str, default='db_config.json')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)