start-redpanda:
	docker compose -f redpanda.yml up -d

stop-redpanda:
	docker compose -f redpanda.yml down

# Elastic Search and Kibana
start-elasticsearch:
	docker compose -f elastic_search_and_kibana.yml up -d
stop-elasticsearch:
	docker compose -f elastic_search_and_kibana.yml down

start-feature-pipeline:
	docker compose -f feature_pipeline.yml up --build -d

stop-feature-pipeline:
	docker compose -f feature_pipeline.yml down

start-backfill-pipeline:
    # generate a unique backfill job id using the current timestamp and a random number
	export BACKFILL_JOB_ID=$$(date +%s%N)-$$RANDOM && \
	docker compose -f backfill_pipeline.yml up --build -d && \
	echo "Started backfill pipeline with BACKFILL_JOB_ID=$$BACKFILL_JOB_ID"

stop-backfill-pipeline:
	echo "Stopping backfill pipeline with BACKFILL_JOB_ID=$$BACKFILL_JOB_ID"
	docker compose -f backfill_pipeline.yml down

	echo "Deleting Redpanda resources for BACKFILL_JOB_ID=$$BACKFILL_JOB_ID"
	make delete-redpanda-resources BACKFILL_JOB_ID=$$BACKFILL_JOB_ID

delete-redpanda-resources:
	rpk group delete trade_to_ohlcv_historical_consumer_group_$(BACKFILL_JOB_ID) --brokers redpanda:9092
	rpk group delete ohlcv_historical_to_feature_store_consumer_group_${BACKFILL_JOB_ID} --brokers redpanda:9092
	rpk topic delete trade_historical_$(BACKFILL_JOB_ID) --brokers redpanda:9092
	rpk topic delete ohlcv_historical_$(BACKFILL_JOB_ID) --brokers redpanda:9092

reset-docker:
	docker container prune
	docker image prune
	docker volume prune
	docker network prune
	docker system prune -a