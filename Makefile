deploy:
	docker build -t mlops_deploy_service .
	docker run -it -d -p 9000:9000 mlops_deploy_service

simulate:
	python simulate_traffic.py

train:
	python train.py

test:
	pytest tests

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 6677 --host localhost

prefect:
	prefect orion start
