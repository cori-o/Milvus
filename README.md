# Milvus
The files required to run Milvus & code related to Milvus tasks. 

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
