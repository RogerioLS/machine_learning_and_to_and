from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Carregar o modelo treinado
model = pickle.load(open('../model_trained/model_regression_immobile.pkl', 'rb'))

# Carregar a base de dados
#data = pd.read_csv('../data/arquivo.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Receber dados do cliente
    req_data = request.get_json()
    
    # Verificar se todos os campos necessários estão presentes
    required_fields = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    if not all(field in req_data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Criar DataFrame a partir dos dados recebidos
    input_data = {
        'MedInc': req_data['MedInc'],
        'HouseAge': req_data['HouseAge'],
        'AveRooms': req_data['AveRooms'],
        'AveBedrms': req_data['AveBedrms'],
        'Population': req_data['Population'],
        'AveOccup': req_data['AveOccup'],
        'Latitude': req_data['Latitude'],
        'Longitude': req_data['Longitude']
    }
    
    input_df = pd.DataFrame([input_data])
    
    try:
        # Prever usando o modelo carregado
        prediction = model.predict(input_df)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)