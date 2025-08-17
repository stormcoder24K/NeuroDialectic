import json
import os
import uuid
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GoalAGISystem:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize transformers pipeline
        try:
            logger.info("Initializing transformers pipeline with model: google/flan-t5-small")
            self.pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                device=-1,  # CPU
                max_length=256,
                model_kwargs={"torch_dtype": "auto"}
            )
            logger.info("Transformers pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize transformers pipeline: {e}")
            raise
        
        # Initialize Hugging Face Embeddings for memory
        try:
            logger.info("Initializing HuggingFaceEmbeddings with model: sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.vector_store = Chroma(
                collection_name="agent_memory",
                embedding_function=self.embeddings,
                persist_directory="./chroma_db"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def generate_response(self, prompt):
        """Generate a response using the transformers pipeline."""
        try:
            response = self.pipeline(prompt, max_length=256, temperature=0.7)[0]["generated_text"]
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def store_memory(self, goal, clarified_goal, plan, execution_results, feedback):
        """Store interaction in ChromaDB."""
        try:
            document = json.dumps({
                "goal": goal,
                "clarified_goal": clarified_goal,
                "plan": plan,
                "execution_results": execution_results,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            })
            doc_id = str(uuid.uuid4())
            self.vector_store.add_texts(
                texts=[document],
                metadatas=[{"goal": goal}],
                ids=[doc_id]
            )
            logger.info(f"Stored interaction for goal: {goal}")
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
    
    def retrieve_memory(self, goal, max_results=2):
        """Retrieve relevant past interactions from ChromaDB."""
        try:
            results = self.vector_store.similarity_search(
                query=goal,
                k=max_results
            )
            context = [json.loads(doc.page_content) for doc in results]
            logger.info(f"Retrieved {len(context)} past interactions for goal: {goal}")
            return context if context else "No relevant past context found."
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            return "Error retrieving past context."
    
    def process_goal(self, goal):
        """Process a goal through a simplified perception-planning-action-feedback loop."""
        logger.info(f"Processing goal: {goal}")
        
        # Step 1: Goal Clarification
        clarification_prompt = f"""Clarify and reformulate the goal: '{goal}'. Ensure it is specific, measurable, and actionable. 
Use past context if relevant: {self.retrieve_memory(goal)}. 
Output in JSON format:
{{
    "original_goal": "{goal}",
    "clarified_goal": "<clarified goal>",
    "reasoning": "<reasoning>"
}}"""
        try:
            goal_result = self.generate_response(clarification_prompt)
            goal_data = json.loads(goal_result)
            clarified_goal = goal_data["clarified_goal"]
        except json.JSONDecodeError:
            logger.error("Failed to parse goal clarification response")
            goal_data = {"error": "Failed to parse goal clarification response", "clarified_goal": goal}
            clarified_goal = goal
        except Exception as e:
            logger.error(f"Goal clarification error: {e}")
            goal_data = {"error": str(e), "clarified_goal": goal}
            clarified_goal = goal
        
        # Step 2: Task Planning
        planning_prompt = f"""Break down the clarified goal: '{clarified_goal}' into actionable subtasks. 
Use chain-of-thought reasoning and past context: {self.retrieve_memory(goal)}. 
Output in JSON format:
{{
    "goal": "{clarified_goal}",
    "reasoning": "<reasoning>",
    "plan": [
        {{"task_id": 1, "description": "<task>", "status": "pending"}},
        ...
    ]
}}"""
        try:
            plan_result = self.generate_response(planning_prompt)
            plan_data = json.loads(plan_result)
        except json.JSONDecodeError:
            logger.error("Failed to parse planning response")
            plan_data = {"error": "Failed to parse planning response", "raw_response": plan_result}
            return {
                "goal": goal,
                "clarified_goal": clarified_goal,
                "past_context": self.retrieve_memory(goal),
                "plan": plan_data,
                "execution_results": [],
                "feedback": {"error": "Planning failed"}
            }
        except Exception as e:
            logger.error(f"Planning error: {e}")
            plan_data = {"error": str(e), "raw_response": ""}
            return {
                "goal": goal,
                "clarified_goal": clarified_goal,
                "past_context": self.retrieve_memory(goal),
                "plan": plan_data,
                "execution_results": [],
                "feedback": {"error": "Planning failed"}
            }
        
        # Step 3: Execution (Simulated)
        execution_results = []
        for task in plan_data.get("plan", []):
            execution_prompt = f"""Simulate executing the task: '{task["description"]}'. 
Describe the outcome as if completed using tools (e.g., search, API calls). 
Output in JSON format:
{{
    "task": "{task["description"]}",
    "outcome": "<outcome>",
    "success": true
}}"""
            try:
                exec_result = self.generate_response(execution_prompt)
                result_data = json.loads(exec_result)
                execution_results.append(result_data)
                task["status"] = "completed" if result_data["success"] else "failed"
            except json.JSONDecodeError:
                execution_results.append({
                    "task": task["description"],
                    "outcome": "Failed to parse execution result",
                    "success": False
                })
                task["status"] = "failed"
            except Exception as e:
                logger.error(f"Execution error for task {task['description']}: {e}")
                execution_results.append({
                    "task": task["description"],
                    "outcome": f"Error: {str(e)}",
                    "success": False
                })
                task["status"] = "failed"
        
        # Step 4: Self-Evaluation
        evaluation_prompt = f"""Evaluate the plan: {json.dumps(plan_data)} and execution results: {json.dumps(execution_results)}. 
Assess feasibility, clarity, and completeness. Suggest improvements or confirm quality. 
Use chain-of-thought reasoning. Output in JSON format:
{{
    "evaluation": "<evaluation>",
    "suggestions": ["<suggestion1>", "<suggestion2>"],
    "is_feasible": true,
    "goal_achieved": true
}}"""
        try:
            eval_result = self.generate_response(evaluation_prompt)
            feedback_data = json.loads(eval_result)
        except json.JSONDecodeError:
            logger.error("Failed to parse evaluation response")
            feedback_data = {"error": "Failed to parse evaluation response", "raw_response": eval_result}
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            feedback_data = {"error": str(e), "raw_response": ""}
        
        # Step 5: Store interaction in memory
        self.store_memory(goal, clarified_goal, plan_data, execution_results, feedback_data)
        
        return {
            "goal": goal,
            "clarified_goal": clarified_goal,
            "past_context": self.retrieve_memory(goal),
            "plan": plan_data,
            "execution_results": execution_results,
            "feedback": feedback_data
        }

def main():
    # Initialize the system
    try:
        system = GoalAGISystem()
    except Exception as e:
        print(f"Error initializing system: {e}")
        return
    
    # Prompt for user input
    goal = input("Enter your goal: ")
    if not goal.strip():
        print("Error: Goal cannot be empty.")
        return
    
    # Process the goal
    try:
        result = system.process_goal(goal)
        print("\nResult:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error processing goal: {e}")

if __name__ == "__main__":
    main()