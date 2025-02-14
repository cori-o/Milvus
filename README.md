# Milvus
The files required to run Milvus & code related to Milvus tasks. 

### Milvus Docker Setting
##### 01. Get milvus.yaml file 
```bash 
$ wget https://raw.githubusercontent.com/milvus-io/milvus/v2.4.9/configs/milvus.yaml
```
##### 02. Get docker compose file 
```bash
$ wget https://github.com/milvus-io/milvus/releases/download/v2.4.9/milvus-standalone-docker-compose.yml -O docker-compose.yml
```
##### 03. Start container 
```bash
$ sudo docker compose up -d
```
##### 04. Stop container 
```bash
$ sudo docker compose down
```
##### 05. Remove volume
```bash
$ sudo rm -rf volumes
```

### Milvus Setting
##### 01. set env
we can define schema field, create schema, create collection by running next command 
```bash
$ python set_milvus.py
```
##### 02. check info 
we can check info, create or delete partition by running next command 
```bash
$ python manage_milvus.py    # check info
$ python manage_milvus.py --task_name create --collection_name ... --partition_name ...   # create partition
$ python manage_milvus.py --task_name delete --collection_name ... --partition_name ...   # delete partition
```
