from pymilvus import MilvusClient, DataType
from pymilvus import connections, db
from pymilvus import Collection, CollectionSchema, FieldSchema, utility
from pymilvus import connections
from . import DataProcessor
import numpy as np
import logging

class MilVus():
    def __init__(self, args):
        self.ip_addr = args['ip_addr'] 
        self.port = '19530'

    def set_env(self):
        self.client = MilvusClient(
            uri="http://" + self.ip_addr + ":19530", port=19530
        )
        self.conn = connections.connect(
            alias="default", 
            host=self.ip_addr, 
            port='19530'
        )

    def _get_data_type(self, dtype):
        if dtype == 'int':
            return DataType.INT64 
        elif dtype == 'str':
            return DataType.VARCHAR
        elif dtype == 'float':
            return DataType.FLOAT_VECTOR
        return None

    def get_partition_info(self, collection):
        self.partitions = collection.partitions 
        self.partition_names = []; self.partition_num_entities = [] 
        for partition in self.partitions: 
            self.partition_names.append(partition.name)
            self.partition_num_entities.append(partition.num_entities)

    def get_collection_info(self, collection):
        print(f'primary key of collection: {collection.primary_field}')
        print(f'schema info: {collection.schema}') 
        # print(f'is collection empty ?: {collection.is_empty}')
        print(f'num of data: {collection.num_entities}')
        

class MilvusEnvManager(MilVus):
    def __init__(self, args):
        super().__init__(args)
        self.logger = logging.getLogger(__name__)

    def create_db(self, db_name):
        if not db.has_database(db_name):
            db.create_database(db_name)
            self.logger.info(f'Created database: {db_name}')
        else:
            self.logger.warning(f'Database {db_name} already exists.')

    def create_collection(self, collection_name, schema, shards_num):
        collection = Collection(
            name=collection_name,
            schema=schema,
            using='default',
            shards_num=shards_num
        )
        return collection 

    def create_field_schema(self, schema_name, dtype=None, dim=1024, max_length=200, is_primary=False):
        data_type = self._get_data_type(dtype)
        field_schema = FieldSchema(
            name=schema_name,
            dtype=data_type,
            is_primary=is_primary,
            dim=dim, 
            max_length=max_length
        )
        return field_schema

    def create_schema(self, field_schema_list, desc, enable_dynamic_field=True):
        schema = CollectionSchema(
            fields=field_schema_list,
            description=desc,
            enable_dynamic_field=enable_dynamic_field
        )
        self.logger.info('Created schema')
        return schema

    def create_index(self, collection, field_name):
        index_params = {
            "metric_type": "L2",    # L2: Euclidean, IP: Cosine similarity
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }   
        collection.create_index(
            field_name=field_name,
            index_params=index_params
        )
        self.logger.info(f'Created index on field: {field_name}')
    
    def create_partition(self, collection, partition_name):
        if not collection.has_partition(partition_name):
            collection.create_partition(partition_name)
            self.logger.info(f'Created partition: {partition_name}')
        else:
            self.logger.warning(f'Partition {partition_name} already exists.')

    def delete_collection(self, collection_name):
        try:
            assert utility.has_collection(collection_name), f'{collection_name}이 존재하지 않습니다.'
            utility.drop_collection(collection_name)
        except:
            pass
    
    def delete_partition(self, collection_name, partition_name):
        try:
            assert utility.has_collection(collection_name)
            self.client.release_collection(collection_name=collection_name)
            self.client.drop_partition(
                collection_name=collection_name,
                partition_name=partition_name
            )
        except Exception as e:
            print(f"Error while deleting partition {partition_name} in {collection_name}: {e}")


class DataMilVus(DataProcessor):   #  args: (DataProcessor)
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def set_env(self):
        self.client = MilvusClient(
            uri="http://" + self.args.ip_addr + ":19530", port=19530
        )
        self.conn = connections.connect(
            alias="default", 
            host=self.args.ip_addr, 
            port='19530'
        )

    def bge_milvus_embed(self, text):   
        ''' 2.4 x ''' 
        from pymilvus.model.hybrid import BGEM3EmbedddingFunction
        bge_m3_ef = BGEM3EmbedddingFunction(
            model_name='BAAI/bge-m3',
            device='cpu',
            use_fp16=False
        )
        
        if isinstance(text, str):
            bge_emb = bge_m3_ef.encode_queries(text)
            print(f"embeddings (dense): {bge_emb['dense']}")
        else:
            bge_emb = bge_m3_ef.encode_documents(text)
        return bge_emb

    def bge_embed_data(self, text):
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
        if isinstance(text, str):
            embeddings = model.encode(text, batch_size=12, max_length=8192)['dense_vecs']
        else:       
            embeddings = model.encode(list(text), batch_size=12, max_length=1024)['dense_vecs']   # dense_vecs, lexical weights, colbert_vecs 
        # print(f"lexical_weights:{model.convert_id_to_token(embeddings['lexical_weights'])}")
        # print(f'key of emb: {embeddings.keys()}')
        embeddings = list(map(np.float32, embeddings))
        return embeddings

    def insert_data(self, m_data, collection_name, partition_name=None):
        collection = Collection(collection_name)
        collection.insert(m_data, partition_name)
        
    def get_len_data(self, collection):
        print(collection.num_entities)

    def set_search_params(self, metric_type="L2", offset=5, ignore_growing=False):
        ''' 옵션을 주지 않는게 훨씬 더 검색 잘 함 .. !! '''
        self.search_params = {
            "metric_type": metric_type, 
            # "offset": offset, 
            # "ignore_growing": ignore_growing, 
            # "params": {"nprobe": 80}
        }
    
    def search_data(self, collection, query_emb, limit=5, anns_field='text_emb', output_fields=None, consistency_level="Strong"):
        results = collection.search(
                data=[query_emb], 
                anns_field=anns_field, 
                # the sum of `offset` in `param` and `limit` should be less than 16384.
                param=self.search_params,
                limit=5,
                expr=None,
                # set the names of the fields you want to retrieve from the search result.
                output_fields=[output_fields],
                consistency_level=consistency_level,
                partition_names=['transport_expenses']
            )
        return results

    def decode_search_result(self, search_result):
        print(f'ids: {search_result[0][0].id}')
        print(f"entity: {search_result[0][0].entity.get('text')}") 
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
        # model.convert_id_to_token(search_result[0][0].entity,)
        return search_result[0][0].entity.get('text')

    def calc_emb_similarity(self, emb1, emb2, metric='L2'):
        import numpy as np 
        np_emb1 = np.array(emb1)
        np_emb2 = np.array(emb2)
        if metric == 'L2':   # Euclidean distance
            print('calc Euclidean distance')
            l2_distance = np.linalg.norm(np_emb1 - np_emb2)
            return l2_distance


class MilvusMeta():
    ''' 
    파일이름 - ID Code, 파일이름 - 영문이름 (파티션) 매핑 정보 관리 클래스 
    '''
    def set_rulebook_map(self):
        self.rulebook_id_code = {
            '취업규칙': '00', 
            '윤리규정': '01', 
            '신여비교통비': '02', 
            '경조금지급규정': '03',
            '직무발명보상': '04',
            '투자업무_운영관리': '05',
        }
        self.rulebook_kor_to_eng = {
            '취업규칙': 'employment_rules',
            '윤리규정': 'code_of_ethics',
            '신여비교통비': 'transport_expenses',
            '경조금지급규정': 'extra_expenditure',
            '직무발명보상': 'ei_compensation',
            '투자업무_운영관리': 'io_management'
        }
        self.rulebook_eng_to_kor = {value: key for key, value in self.rulebook_kor_to_eng.items()}