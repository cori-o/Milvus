from pymilvus import Collection, CollectionSchema, FieldSchema, utility
from dotenv import load_dotenv
from src import MilVus, DataMilVus, DataProcessor
from src import LLMOpenAI
import os 
import argparse
import numpy as np
import json

def main(args):
    load_dotenv()
    db_name = 'finger'
    ip_addr = os.getenv('ip_addr')

    with open(os.path.join(args.config_path, args.config_name)) as f:
        db_args = json.load(f)

    with open(os.path.join(args.config_path, args.llm_config)) as f:
        llm_config = json.load(f)

    db_args['ip_addr'] = ip_addr

    milvus_db = MilVus(db_args)
    milvus_db.set_env()
    print(f'client: {milvus_db.client}')
    
    llm_openai = LLMOpenAI(llm_config)
    llm_openai.set_generation_config()
    llm_openai.set_response_guideline()

    collection = Collection(args.collection_name)
    collection.load()
    
    data_milvus = DataMilVus(db_args)
    text = "개회겸 개최식"
    # text2 = "2025년 신년을 맞아 현충원에서 열린 참배 행사에서 촬영된 컬러 사진입니다. 사진 속 오른쪽 남성은 안경을 쓰고 넥타이를 맨 어두운 정장을 입고 있으며, 기념비 위에 손을 얹고 약간 몸을 숙이고 있습니다. 왼쪽에는 군복을 입고 계급장이 달린 모자를 쓴 또 다른 남성이 그 앞에 서서 경례를 하고 있습니다. 배경에는 기념비나 동상들이 보이는 것으로 보아, 이곳은 역사적으로 중요한 장소에서 열린 기념 행사로 추측됩니다. 날씨는 흐린 것으로 보입니다."
    query_emb = data_milvus.bge_embed_data(text)
    # emb2 = data_milvus.bge_embed_data(text2)
    # emb3 = data_milvus.get_embedding_by_id(args.collection_name, '000001')
    # query_emb = llm_openai.create_embeddings(llm_config['openai_embedding_model'], text)
    print(np.shape(query_emb))
    
    data_milvus.set_search_params()
    search_result = data_milvus.search_data(collection, query_emb, args.partition_name, output_fields='text')
    print(search_result)
    # emb2 = search_result[0][0].entity.get('text_emb')
    print(f'-' * 20)
    print(data_milvus.decode_search_result(search_result))
    #print(data_milvus.calc_emb_similarity(emb3, query_emb))

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='./config/')
    cli_parser.add_argument('--config_name', type=str, default='db_config.json')
    cli_parser.add_argument('--llm_config', type=str, default='llm_config.json')
    cli_parser.add_argument('--output_dir', type=str, default='./data/pdf/embed_output')
    cli_parser.add_argument('--collection_name', type=str, default='congress')
    cli_parser.add_argument('--partition_name', type=str, default='national_assembly_library')
    cli_argse = cli_parser.parse_args()
    main(cli_argse)