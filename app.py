from flask import Flask, jsonify, request, abort
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic.fields import Field
import requests
from typing import Optional, List
import json
import threading
import re

app = Flask(__name__)

# In-memory data storage for students and concurrency lock
students = {}
current_id = 1
students_lock = threading.Lock() 

class OllamaLLM(LLM):
    """Custom LLM wrapper for Ollama integration with LangChain."""
    
    model: str = Field(default="llama3.2:1b")
    api_url: str = Field(default="http://127.0.0.1:11434/api/generate")

    def __init__(self, model: str = "llama3.2:1b", api_url: str = "http://127.0.0.1:11434/api/generate"):
        super().__init__(model=model, api_url=api_url)

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Send a prompt to the Ollama API and return the response."""
        payload = {
            "model": self.model,
            "prompt": prompt
        }
        try:
            response = requests.post(self.api_url, json=payload, stream=True)
            response.raise_for_status()
            
            full_response = ""
            
            # Process each line in the streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        result = json.loads(line)
                        full_response += result.get("response", "")
                    except json.JSONDecodeError as e:
                        print("JSON parsing error for line:", line)
                        print("Error message:", e)
                        return "Error in parsing Ollama response."

            # Return the complete response text after processing all lines
            return full_response if full_response else "No summary generated from Ollama."

        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return "Error generating response from Ollama."

# Prompt template with clear indication for fictional scenario
prompt_template = PromptTemplate(
    input_variables=["name", "age", "email"],
    template=(
        "Create a fictional summary for a student named {name}, who is {age} years old with email {email}. "
        "This is for a testing scenario only, and no real personal information is involved. "
        "Describe this fictional student's profile in a few sentences."
    )
)

# Initialize the LLMChain with the prompt template
ollama_llm = OllamaLLM(model="llama3.2:1b")
llm_chain = LLMChain(llm=ollama_llm, prompt=prompt_template)

# Helper function to validate student data
def validate_student_data(data):
    if "name" not in data or not isinstance(data["name"], str) or len(data["name"].strip()) == 0:
        abort(400, description="Invalid or missing 'name'.")
    if "age" not in data or not isinstance(data["age"], int) or data["age"] <= 0:
        abort(400, description="Invalid or missing 'age'.")
    if "email" not in data or not isinstance(data["email"], str) or not re.match(r"[^@]+@[^@]+\.[^@]+", data["email"]):
        abort(400, description="Invalid or missing 'email'.")

# REST API Endpoints

@app.route('/students', methods=['POST'])
def create_student():
    global current_id
    data = request.get_json()
    validate_student_data(data)  # Validate input data

    with students_lock:  # Ensure thread-safe access
        student = {"id": current_id, "name": data["name"], "age": data["age"], "email": data["email"]}
        students[current_id] = student
        current_id += 1

    return jsonify(student), 201

@app.route('/students', methods=['GET'])
def get_students():
    with students_lock:  # Ensure thread-safe read access
        return jsonify(list(students.values()))

@app.route('/students/<int:id>', methods=['GET'])
def get_student(id):
    with students_lock:
        student = students.get(id)
    if student is None:
        abort(404, description="Student not found.")
    return jsonify(student)

@app.route('/students/<int:id>', methods=['PUT'])
def update_student(id):
    data = request.get_json()
    validate_student_data(data)  # Validate input data

    with students_lock:  # Ensure thread-safe access
        student = students.get(id)
        if student is None:
            abort(404, description="Student not found.")
        
        student.update({k: data[k] for k in ("name", "age", "email") if k in data})

    return jsonify(student)

@app.route('/students/<int:id>', methods=['DELETE'])
def delete_student(id):
    with students_lock:  # Ensure thread-safe access
        if id in students:
            del students[id]
            return '', 204
        else:
            abort(404, description="Student not found.")

@app.route('/students/<int:id>/summary', methods=['GET'])
def student_summary(id):
    with students_lock:
        student = students.get(id)
    if student is None:
        abort(404, description="Student not found.")

    # Generate summary using the custom LLM chain
    summary = llm_chain.invoke({
        "name": student["name"],
        "age": student["age"],
        "email": student["email"]
    })
    return jsonify({"summary": summary})

# Error handling
@app.errorhandler(404)
def resource_not_found(error):
    return jsonify({"error": str(error)}), 404

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": str(error)}), 400

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "An internal error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True)
