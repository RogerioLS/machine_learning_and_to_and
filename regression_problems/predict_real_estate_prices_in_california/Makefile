
conda:
	conda info --envs

docker:
	docker build -t real_estate_app .
	docker run -p 8501:8501 real_estate_app

porta:
	sudo lsof -i :8501
	sudo kill -9 <PID>

predict_flask:
	@echo "curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{\
	"MedInc": 3.0,\
	"HouseAge": 20,\
	"AveRooms": 5.0,\
	"AveBedrms": 1.0,\
	"Population": 300,\
	"AveOccup": 3.0,\
	"Latitude": 34.0,\
	"Longitude": -118.0\
	}"\