from dotenv import load_dotenv
from src import milvus 
import os 
import argparse
import pandas as pd
import json
import numpy as np

def main(args):
    load_dotenv()
    ip_addr = os.getenv('ip_addr')
    id_code = {'취업규칙': '00', '윤리규정': '01', '신여비교통비': '02', '경조금지급규정': '03'}
    with open(os.path.join(args.config_path, args.config_name)) as f:
        db_args = json.load(f)
    db_args['ip_addr'] = ip_addr

    milvus_db = milvus.MilVus(db_args)
    milvus_db.set_env()
    print(f'client: {milvus_db.client}')

    # data_df = pd.read_csv(os.path.join(args.output_dir, args.file_name), index_col=0)
    # print(data_df.head(3), len(data_df))
    milvus_dp = milvus.DataMilVus(db_args)
    # print(data_df['text'][1])
    emb1 = milvus_dp.bge_embed_data('제2장 국내 출장 여비 ')
    # print(emb1)
    text = """제2장 국내 출장 여비 

        제9조 (국내출장여비  지급기준)   
        국내출장여비지급은  별표 제1호에 정하는 바에 의한다.  

        제10조 (국내출장 기간에 따른 여비 조정) 
        ① 출장에 따른 숙박비, 일비, 식비는 출장기간에  따라 아래 각호와 같이 조정하여 지급한다.  
        출장기간의  적용은 출장명령서(국내출장신청서 )의 기간에 의한다.  단, 해당 출장자가 이전에",../../pdf/신여비교통비.pdf,2,200
        11,"동일지역  출장 경력이 있는 경우에는  이전 출장기간을  합산할 수 있다."""
    emb2 = milvus_dp.bge_embed_data(text)
    # print(emb2) 
    sim = milvus_dp.calc_emb_similarity(emb1, emb2, metric='L2')
    print(sim)
    

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config_path', type=str, default='../db_config/')
    cli_parser.add_argument('--config_name', type=str, default='db_config.json')
    cli_parser.add_argument('--output_dir', type=str, default='../../data/pdf/embed_output')
    cli_parser.add_argument('--file_name', type=str, default=None)
    cli_parser.add_argument('--collection_name', type=str, default='rule_book')
    cli_parser.add_argument('--partition_name', type=str, default=None)
    cli_argse = cli_parser.parse_args()
    main(cli_argse)