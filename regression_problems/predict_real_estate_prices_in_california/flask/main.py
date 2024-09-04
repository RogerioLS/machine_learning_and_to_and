from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import pandas as pd
from marshmallow import Schema, fields, ValidationError
from flask_swagger_ui import get_swaggerui_blueprint
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Configuração da proteção CSRF
csrf = CSRFProtect(app)

# Configuração de rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Swagger setup http://localhost:5000/swagger
SWAGGER_URL = '/swagger'
API_URL = '/static_swagger/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "Real Estate Price Prediction API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Adicionar uma rota para servir o arquivo swagger.json
@app.route('/static_swagger/<path:filename>')
def serve_static(filename):
    return send_from_directory('static_swagger', filename)

# Carregar o modelo treinado
model = pickle.load(open('../model_trained/model_regression_immobile.pkl', 'rb'))

# Definição do schema de validação usando marshmallow
class InputSchema(Schema):
    MedInc = fields.Float(required=True)
    HouseAge = fields.Float(required=True)
    AveRooms = fields.Float(required=True)
    AveBedrms = fields.Float(required=True)
    Population = fields.Int(required=True)
    AveOccup = fields.Float(required=True)
    RoomDensity = fields.Float(required=True)
    Latitude = fields.Float(required=True)
    Longitude = fields.Float(required=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")  # Limite de 10 requisições por minuto para este endpoint
@csrf.exempt  # Exemplo: desabilitar CSRF para a rota de API (opcional, com cuidado)
def predict():
    try:
        # Receber dados do cliente
        req_data = request.get_json()

        # Validar entrada
        schema = InputSchema()
        validated_data = schema.load(req_data)

        # Criar DataFrame a partir dos dados validados
        input_df = pd.DataFrame([validated_data])

        # Prever usando o modelo carregado
        prediction = model.predict(input_df)
        return jsonify({'prediction': prediction[0]})
    except ValidationError as err:
        # Erros de validação do input
        return jsonify({'error': 'Validation Error', 'messages': err.messages}), 400
    except KeyError as err:
        # Se algum campo esperado não estiver presente
        return jsonify({'error': f'Missing required field: {err}'}), 400
    except ValueError as err:
        # Erros relacionados ao valor dos dados de entrada
        return jsonify({'error': f'Invalid input: {err}'}), 400
    except Exception as e:
        # Erro geral
        return jsonify({'error': 'An error occurred', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
