# Code Review Agent Trio: Coder, Reviewer, Manager
import os
import json
import requests
import time
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from openai import OpenAI
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_KEY_1 = os.getenv("OPENROUTER_KEY_1")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROKCLOUD_API_KEY = os.getenv("GROKCLOUD_API_KEY")

response_cache = {}

def api_call(config, prompt, retries=3, timeout=10):
    error_details = ""
    for attempt in range(retries):
        try:
            if config["type"] == "openrouter":
                client = OpenAI(api_key=OPENROUTER_KEY_1, base_url="https://openrouter.ai/api/v1")
                models = [config["model"], "mistral-7b-instruct:free"]  # Fallback model
                for model in models:
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            timeout=timeout
                        )
                        if not response.choices or len(response.choices) == 0:
                            raise ValueError("Empty choices in OpenRouter response")
                        time.sleep(2)  
                        return response.choices[0].message.content
                    except Exception as e:
                        error_details = f"Model {model} failed: {str(e)}"
                        continue
                raise ValueError(f"All OpenRouter models failed: {error_details}")
            elif config["type"] == "grokcloud":
                headers = {
                    "Authorization": f"Bearer {GROKCLOUD_API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "llama3-70b-8192",
                    "messages": [{"role": "user", "content": prompt}]
                }
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers, timeout=timeout)
                resp.raise_for_status()
                response = resp.json()
                if not response.get("choices") or len(response["choices"]) == 0:
                    raise ValueError("Empty choices in GrokCloud response")
                return response["choices"][0]["message"]["content"]
            elif config["type"] == "cohere":
                headers = {
                    "Authorization": f"Bearer {COHERE_API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "message": prompt,
                    "model": "command-r-plus",
                    "temperature": 0.5
                }
                resp = requests.post("https://api.cohere.ai/v1/chat", json=data, headers=headers, timeout=timeout)
                resp.raise_for_status()
                response = resp.json()
                if not response.get("text"):
                    raise ValueError("Empty text in Cohere response")
                return response["text"]
            elif config["type"] == "gemini":
                model_instance = genai.GenerativeModel(config["model"])
                response = model_instance.generate_content(prompt)
                if not response.text:
                    raise ValueError("Empty text in Gemini response")
                return response.text
        except Exception as e:
            error_details = f"{config['type'].capitalize()} API failed - {str(e)}. Response: {resp.text if 'resp' in locals() else 'No response'}"
            if attempt == retries - 1:
                return f"Error: {error_details}"
            time.sleep(2 ** attempt) 
    return f"Error: {config['type'].capitalize()} API failed after {retries} retries - {error_details}"

class Coder:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["task", "critique"],
            template=(
                "Write Python code for the following task: {task}\n"
                "{critique}\n"
                "Ensure the code is correct, efficient, and follows PEP 8. Include comments for clarity."
            )
        )

    def generate_code(self, task, critique=""):
        cache_key = f"code:{task}:{critique}"
        if cache_key in response_cache:
            return response_cache[cache_key]

        prompt = self.prompt_template.format(task=task, critique=critique)
        code = api_call({"type": "gemini", "model": "gemini-2.0-flash"}, prompt)
        if "Error:" in code:
            code = api_call({"type": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct:free"}, prompt)
        response_cache[cache_key] = code
        return code

class Reviewer:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["code", "task"],
            template=(
                "Review the following Python code for the task: {task}\n"
                "Code:\n{code}\n"
                "Identify specific issues (e.g., bugs, performance, PEP 8 violations, clarity) and suggest actionable improvements. "
                "Be concise and focus on critical flaws."
            )
        )

    def review_code(self, code, task):
        prompt = self.prompt_template.format(code=code, task=task)
        critique = api_call({"type": "cohere"}, prompt)
        if "Error:" in critique:
            critique = api_call({"type": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct:free"}, prompt)
        return critique

class Manager:
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["code", "critique", "task"],
            template=(
                "Evaluate the following Python code and critique for the task: {task}\n"
                "Code:\n{code}\n"
                "Critique:\n{critique}\n"
                "Decide if the code should be accepted or requires a redo. "
                "Accept if the code is correct, efficient, and has no significant issues (minor style issues are acceptable). "
                "Request a redo if there are bugs, performance problems, or major issues. "
                "Explain your reasoning and return 'Accept' or 'Redo' as the final word."
            )
        )

    def decide(self, code, critique, task):
        cache_key = f"decision:{task}:{code}"
        if cache_key in response_cache:
            return response_cache[cache_key]

        prompt = self.prompt_template.format(code=code, critique=critique, task=task)
        decision = api_call({"type": "grokcloud"}, prompt)
        if "Error:" in decision:
            decision = api_call({"type": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct:free"}, prompt)
        response_cache[cache_key] = decision
        return decision

class Orchestrator:
    def __init__(self):
        self.coder = Coder()
        self.reviewer = Reviewer()
        self.manager = Manager()
        self.max_iterations = 3

    def run_review(self, task):
        reasoning_log = []
        current_code = ""
        critique = ""
        decision = None 
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            current_code = self.coder.generate_code(task, critique)
            reasoning_log.append({
                "step": f"Code Generation Iteration {iteration}",
                "code": current_code
            })

            if "Error:" in current_code:
                reasoning_log.append({
                    "step": f"Error Iteration {iteration}",
                    "error": current_code
                })
                break

            critique = self.reviewer.review_code(current_code, task)
            reasoning_log.append({
                "step": f"Review Iteration {iteration}",
                "critique": critique
            })

            if "Error:" in critique:
                reasoning_log.append({
                    "step": f"Error Iteration {iteration}",
                    "error": critique
                })
                break

            decision = self.manager.decide(current_code, critique, task)
            reasoning_log.append({
                "step": f"Decision Iteration {iteration}",
                "decision": decision
            })

            if "Error:" in decision:
                reasoning_log.append({
                    "step": f"Error Iteration {iteration}",
                    "error": decision
                })
                break

            if "Accept" in decision.split()[-1]:
                final_response = (
                    f"Final Code:\n{current_code}\n"
                    f"Manager Decision: Accepted\n"
                    f"Reasoning:\n{decision}"
                )
                return {
                    "final_response": final_response,
                    "reasoning_log": reasoning_log
                }

            critique = f"Previous critique: {critique}\nManager decision: {decision}\nPlease address the issues and revise the code."

        final_response = (
            f"Final Code (Max Iterations Reached or Error):\n{current_code}\n"
            f"Manager Decision: Not Accepted\n"
            f"Last Critique:\n{critique}\n"
            f"Last Decision:\n{decision if decision is not None else 'None (Error or Early Termination)'}"  # Handle undefined decision
        )
        return {
            "final_response": final_response,
            "reasoning_log": reasoning_log
        }

if __name__ == "__main__":
    orchestrator = Orchestrator()
    
    while True:
        task = input("Enter your coding task (or 'quit' to exit): ").strip()
        if task.lower() == 'quit':
            print("Exiting...")
            break
        if not task:
            task = "Write a Python function to reverse a string"
            print(f"Using default task: {task}")

        print(f"\nRunning code review for task: '{task}'...")
        result = orchestrator.run_review(task)
        
        print("\nFinal Response:", result["final_response"])
        print("\nReasoning Log:")
        for log in result["reasoning_log"]:
            print(json.dumps(log, indent=2))
        print("\n" + "="*50 + "\n")