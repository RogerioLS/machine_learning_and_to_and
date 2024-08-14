
conda:
	conda info --envs

docker:
	docker build -t real_estate_app .
	docker run -p 8501:8501 real_estate_app

porta:
	sudo lsof -i :8501
	sudo kill -9 <PID>