training-dev:
	poetry run python src/training.py

api-dev:
	poetry run uvicorn src.api:app --reload --port 8000

check-health-local:
	curl -X GET http://localhost:8000/health

predict-request-local:
		curl -X GET "http://localhost:8000/predict?product_id=ETH%2FUSD"

# Build the docker image
build:
	docker build -t price_predictor .

# Run the training script instide theDocker container
training: build
	docker run \
		--env-file .env \
		-e TRAINING_OR_PREDICTION=training \
		price_predictor \
  
# Run the API inside Docker container
api: build
	docker run \
		-p 8000:80 \
		--env-file .env \
		-e TRAINING_OR_PREDICTION=prediction \
		price_predictor 

check-health-prod:
	curl -X GET https://price-predictor-aammeeyy28-realtimemlsystem-production1.deployments.quix.io/health

predict-request-prod:
		curl -X GET "https://price-predictor-aammeeyy28-realtimemlsystem-production1.deployments.quix.io/predict?product_id=ETH%2FUSD"

predict-request-dev:
		curl -X GET "https://price-predictor-aammeeyy28-realtimemlsystem-dev.deployments.quix.io/predict?product_id=ETH%2FUSD"


test:
	poetry run pytest tests/test_utils.py