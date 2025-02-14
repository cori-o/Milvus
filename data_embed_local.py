from dotenv import load_dotenv
from pymilvus import Collection
from src import milvus 
from src import LLMOpenAI, FileProcessor
import os 
import openai
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

    with open(os.path.join(args.config_path, args.llm_config)) as f:
        llm_config = json.load(f)
    
    milvus_db = milvus.MilvusEnvManager(db_args)
    milvus_db.set_env()
    print(f'ip: {milvus_db.ip_addr}')
    
    milvus_meta = milvus.MilvusMeta()
    milvus_meta.set_congress_map()
    id_code = milvus_meta.congress_id_code
    partition_to_kor = milvus_meta.congress_eng_to_kor
    print(id_code, partition_to_kor[args.partition_name])
    
    collection = Collection(args.collection_name)
    collection.load()

    img_file = '예산결산특별위원회_전체회의.jpg'
    img_description = "이 이미지는 예산결산특별위원회 전체회의에서 촬영된 것으로 보입니다. 사진 속 인물은 의회나 입법 회의의 의장석에 앉아 있으며, 앞에는 마이크가 놓여 있습니다. 그는 어두운 색의 정장을 입고 있으며, 성인 남성으로 보입니다. 그의 가슴에는 상이나 계급을 나타낼 수 있는 리본이 달려 있지만, 이미지 만으로는 명확하지 않습니다. 배경에는 큰 문양이 벽에 걸려 있으며, 이는 한국의 의회 홀에서 촬영된 것임을 암시하는 상징과 글자가 포함되어 있습니다. 디자인에는 국기, 왕관, 그리고 아마도 회의나 국가의 이름을 나타내는 텍스트가 포함되어 있습니다. 이 장면은 입법 절차와 일치하는 공식적이고 격식 있는 분위기를 띠고 있습니다."
    
    # llm_openai = LLMOpenAI(llm_config)
    # embeddings = llm_openai.create_embeddings(emb_model=llm_config['openai_embedding_model'], emb_text=img_description)
    # print(np.shape(embeddings))
    
    milvus_dp = milvus.DataMilVus(db_args)
    code = id_code[partition_to_kor[args.partition_name]]
    data_id = code + '01'.zfill(4)
    
    text_emb = milvus_dp.bge_embed_data(img_description)
    source = img_file
    data = [data_id, [text_emb], img_description, [source]]
    milvus_dp.insert_data(data, args.collection_name, args.partition_name)

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='./config/')
    cli_parser.add_argument('--config_name', type=str, default='db_config.json')
    cli_parser.add_argument('--llm_config', type=str, default='llm_config.json')
    cli_parser.add_argument('--output_dir', type=str, default='./data/')
    cli_parser.add_argument('--file_name', type=str, default=None)
    cli_parser.add_argument('--collection_name', type=str, default='congress')
    cli_parser.add_argument('--partition_name', type=str, default=None)
    cli_argse = cli_parser.parse_args()
    main(cli_argse)