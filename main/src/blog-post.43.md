# Hadoop Ecosystem
하둡 에코시스템은 대규모 데이터를 분산 환경에서 저장, 처리, 분석 및 관리하기 위해 하둡 코어 프레임워크를 중심으로 구축된 다양한 도구와 기술의 집합체입니다.

## 코어 레이어 (Core Layers)
하둡 시스템의 근간을 이루는 핵심 컴포넌트입니다.

- HDFS (Hadoop Distributed File System): 대용량의 정형/비정형 데이터를 여러 노드에 분산 저장하는 시스템입니다.
- YARN (Yet Another Resource Negotiator): 클러스터의 자원(CPU, 메모리 등)을 관리하고 작업 스케줄링을 담당하는 자원 관리 레이어입니다.

## 데이터 처리 및 분석 (Data Processing & Analysis)
저장된 데이터를 가공하고 분석하는 도구들입니다.

- MapReduce: 대규모 데이터셋을 병렬로 처리하기 위한 전통적인 소프트웨어 프레임워크입니다. 데이터를 필터링하고 조직화하는 Map단계와 이를 필터링·집계하는 Reduce단계로 나뉩니다.
- Apache Spark: 인메모리(In-Memory) 컴퓨팅 기술을 사용하여 기존 MapReduce보다 훨씬 빠른 속도로 배치, 실시간 스트리밍, 그래프 계산 등을 수행하는 고성능 데이터 처리 플랫폼입니다.
- HIVE: 데이터웨어하우스 인프라로, HQL(Hive Query Language)이라는 SQL 유사 언어를 사용하여 대규모 데이터셋을 쿼리하고 분석할 수 있게 해줍니다. Standard SQL 데이터 타입을 지원하여 기존 SQL 사용자도 쉽게 쓸 수 있습니다.

## 데이터베이스 및 머신러닝 (NoSQL & Machine Learning)
실시간 조회 및 고도화된 데이터 활용을 위한 도구입니다.

- Apache HBase: 구글의 BigTable을 모델로 한 분산 NoSQL 데이터베이스입니다. 대규모 데이터 속에서 특정 부분을 아주 빠르게 읽고 쓰는(Real-time Read/Write) 작업에 최적화되어 있으며 결함 허용(Fault-tolerant) 특성을 가집니다.
- Mahout & Spark MLlib: 하둡 환경에서 기계학습(Machine Learning)을 구현할 수 있도록 클러스터링, 분류, 추천 협업 필터링 등의 알고리즘 라이브러리를 제공합니다.

## 관리 및 보조 도구 (Coordination & Workflow Management)
에코시스템이 원활하게 돌아가도록 돕는 유틸리티입니다.

- Zookeeper: 분산 환경에서 각 컴포넌트 간의 동기화, 구성 관리, 그룹화 등을 조율하여 데이터 일관성을 유지하는 코디네이터입니다.
- Oozie: 하둡 작업들의 흐름을 관리하는 워크플로우 스케줄러입니다. 순차적으로 실행되는 Workflow 작업과 시간/데이터 이벤트에 의해 트리거되는 Coordinator 작업을 관리합니다.

## Reference
https://www.geeksforgeeks.org/dbms/hadoop-ecosystem/