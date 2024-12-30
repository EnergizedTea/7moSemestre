from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Flask app and OpenAI client
app = Flask(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

@app.route('/analyze', methods=['POST'])
def analyze_post():
    try:
        # Get data from the request
        data = request.json
        user_post = data.get("post")
        user_emotion = data.get("emotion")

        if not user_post or not user_emotion:
            return jsonify({"error": "Post and emotion are required."}), 400

        # Interact with OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Se te dara la clasificación emocional de una publicacion y el rostro de quien la hizo. Usa esta información para recomendar una actividad, producto o curso específico. "
                        "Ejemplos: "
                        "- Si la clasificación es 'Felicidad' y 'Positiva', sugiere actividades como clases creativas o experiencias al aire libre. "
                        "- Si es 'Tristeza' y 'Negativa', sugiere buscar ayuda o una actividad relajante. Sé creativo y útil."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Post: {user_post}, Rostro: {user_emotion}"
                },
            ]
        )


        # Extract and return the AI response
        result = response.choices[0].message.content
        return jsonify({"suggestion": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
