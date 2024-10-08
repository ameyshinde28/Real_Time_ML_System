start-redpanda:
	docker compose -f redpanda.yml up -d

stop-redpanda:
	docker compose -f redpanda.yml down

start-feature-pipeline:
	docker compose -f feature_pipeline.yml up --build -d

stop-feature-pipeline:
	docker compose -f feature_pipeline.yml down

start-backfill-pipeline:
	docker compose -f backfill_pipeline.yml up --build -d

stop-backfill-pipeline:
	docker compose -f backfill_pipeline.yml down