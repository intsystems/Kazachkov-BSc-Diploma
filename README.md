# Docker
Собрать docker можно следующими командами:
```bash
# Сборка и запуск контейнера (интерактивный bash)
docker compose -f docker/docker-compose.yml up --build

# Или в фоне, с последующим подключением
docker compose -f docker/docker-compose.yml up --build -d
docker exec -it python_gpu_science_stack bash

# Запустить эксперименты внутри контейнера
cd code && python experiments.py

# Остановить контейнер
docker compose -f docker/docker-compose.yml down
```

> **Примечание:** `docker-compose.yml` написан для GPU. Если на CPU, то секцию `deploy.resources` из `docker/docker-compose.yml` нужно убрать.

```bash
GPU=0 DETACH=1 LOG=exp_m scripts/remote_run.sh 'EXP=m ALGO=sgd python code/experiments.py'
GPU=1 DETACH=1 LOG=exp_n scripts/remote_run.sh 'EXP=n ALGO=sgd python code/experiments.py'
```