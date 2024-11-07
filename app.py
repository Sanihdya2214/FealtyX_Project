from flask import Flask, jsonify, request, abort
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic.fields import Field
import requests
from typing import Optional, List
import json

app = Flask(__name__)

# In-memory data storage for students
students = {}
current_id = 1

# Custom Ollama LLM class for LangChain with streaming response handling
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
            
            # Initialize a variable to store the full response text
            full_response = ""
            
            # Iterate through each line in the streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse each line as a JSON object
                        result = json.loads(line)
                        # Concatenate the 'response' text to form the full response
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

# Updated prompt template with clear indication for testing
prompt_template = PromptTemplate(
    input_variables=["name", "age", "email"],
    template=(
        "Create a fictional summary for a student named {name}, who is {age} years old with email {email}. "
        "This is for a testing scenario only, and no real personal information is involved. "
        "Describe this fictional student's profile in a few sentences."
    )
)

# Initialize the LLMChain with the updated prompt template
ollama_llm = OllamaLLM(model="llama3.2:1b")
llm_chain = LLMChain(llm=ollama_llm, prompt=prompt_template)

# REST API Endpoints

@app.route('/students', methods=['POST'])
def create_student():
    global current_id
    data = request.get_json()
    if not all(k in data for k in ("name", "age", "email")):
        abort(400, "Missing fields in request data.")
    
    student = {"id": current_id, "name": data["name"], "age": data["age"], "email": data["email"]}
    students[current_id] = student
    current_id += 1
    return jsonify(student), 201

@app.route('/students', methods=['GET'])
def get_students():
    return jsonify(list(students.values()))

@app.route('/students/<int:id>', methods=['GET'])
def get_student(id):
    student = students.get(id)
    if student is None:
        abort(404, "Student not found.")
    return jsonify(student)

@app.route('/students/<int:id>', methods=['PUT'])
def update_student(id):
    if id not in students:
        abort(404, "Student not found.")
    
    data = request.get_json()
    student = students[id]
    student.update({k: data[k] for k in ("name", "age", "email") if k in data})
    return jsonify(student)

@app.route('/students/<int:id>', methods=['DELETE'])
def delete_student(id):
    if id in students:
        del students[id]
        return '', 204
    else:
        abort(404, "Student not found.")

@app.route('/students/<int:id>/summary', methods=['GET'])
def student_summary(id):
    student = students.get(id)
    if student is None:
        abort(404, "Student not found.")

    # Generate summary using the custom LLM chain
    summary = llm_chain.invoke({
        "name": student["name"],
        "age": student["age"],
        "email": student["email"]
    })
    return jsonify({"summary": summary})

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
