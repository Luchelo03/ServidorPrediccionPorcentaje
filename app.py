from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime

# Cargar modelo entrenado
model = joblib.load("modelo_lleno.pkl")

# Crear app Flask
app = Flask(__name__)

# Códigos de tacho (ajusta si cambia)
TACHO_CODES = {
    'TCH001': 0, 'TCH002': 1, 'TCH003': 2, 'TCH004': 3,
    'TCH005': 4, 'TCH006': 5, 'TCH007': 6, 'TCH008': 7
}

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.get_json()

    # Extraer datos
    distancia = datos.get('DistanciaCM')
    hora = datos.get('Hora')
    minuto = datos.get('Minuto')
    segundo = datos.get('Segundo')
    tacho = datos.get('TachoID')

    if tacho not in TACHO_CODES:
        return jsonify({'error': 'TachoID no válido'}), 400

    # Preparamos entrada como DataFrame
    entrada = pd.DataFrame([{
        'DistanciaCM': distancia,
        'Hora': hora,
        'Minuto': minuto,
        'Segundo': segundo,
        'TachoID_Cod': TACHO_CODES[tacho]
    }])

    pred = model.predict(entrada)[0]
    return jsonify({'PorcentajeLlenado': round(pred, 2)})

if __name__ == '__main__':
    app.run(debug=True)
