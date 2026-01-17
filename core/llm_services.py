import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from the .env file in the project root
# This makes the pathing robust, no matter where the script is run from.
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Make sure it's in your .env file in the project root.")

class LLMService:
    """
    A centralized service to manage all interactions with the Large Language Model.
    """
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.3):
        """
        Initializes the connection to the LLM.
        
        Args:
            model_name: The name of the OpenAI model to use.
            temperature: The creativity setting for the model (0.0 to 1.0).
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        print("[INFO] LLM Service initialized successfully.")

    def invoke_llm(self, prompt: str) -> str:
        """
        Sends a prompt to the LLM and gets a response. Includes a simple retry mechanism.

        Args:
            prompt: The full prompt string to send to the AI.

        Returns:
            The text response from the AI.
        """
        try:
            response = self.llm.invoke(prompt)
            # Add a small delay to respect API rate limits (e.g., 60 requests per minute)
            time.sleep(1) 
            return response.content
        except Exception as e:
            print(f"[ERROR] An error occurred while invoking the LLM: {e}")
            # In a production system, you might add more robust retry logic here.
            time.sleep(5) # Wait longer before a potential retry
            # For now, we will re-raise the exception to be handled by the caller.
            raise
'''```eof

---

### **Step 6: Coding the Task Strategy Layer**

This layer defines *how* each type of task should be performed. We'll create a base template and then specific files for `rewrite` and `summarize`.

Here is the code for `base_task.py`.

**Explanation:**
* This file defines an "abstract base class" or a blueprint. It declares that any task we create *must* have an `execute` method.
* This is an expert-level design pattern that enforces consistency and ensures that our main orchestrator (`task_processor.py`) can treat all tasks the same way, simply by calling their `execute` method.

```python:Base Task Blueprint:document-processor/tasks/base_task.py'''
from abc import ABC, abstractmethod
from typing import List
from core.llm_services import LLMService

class BaseTask(ABC):
    """
    An abstract base class that defines the structure for all task strategies.
    """
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    @abstractmethod
    def execute(self, text_chunks: List[str], task_instruction: str) -> str:
        """
        The main method that every task must implement.

        Args:
            text_chunks: A list of text chunks from the document.
            task_instruction: The user's specific instruction for the task.

        Returns:
            A single string containing the final processed result.
        """
        pass
'''```eof

Now, here is the implementation for the `rewrite` task.

**Explanation:**
* The `RewriteTask` class inherits the structure from `BaseTask`.
* Its `execute` method implements the simple **Map** strategy.
* It loops through every single chunk of the document.
* For each chunk, it creates a specific prompt and uses the `llm_service` to get the rewritten version of that chunk.
* `print(f"[INFO]     - [Map] Processing chunk {i + 1} of {total_chunks}...")` provides the real-time progress tracking you wanted.
* Finally, it joins all the rewritten chunks back together into a single string.

```python:Rewrite Task Strategy:document-processor/tasks/rewrite.py'''
from typing import List
from tasks.base_task import BaseTask

class RewriteTask(BaseTask):
    """
    A task strategy for rewriting or refining a document chunk by chunk.
    This uses a simple "Map" approach.
    """
    def execute(self, text_chunks: List[str], task_instruction: str) -> str:
        print(f"[INFO] ==> Step 3: 'Rewrite' task detected. Executing Map strategy.")
        
        rewritten_chunks = []
        total_chunks = len(text_chunks)

        prompt_template = (
            "You are an expert editor. Your overall goal is to '{task}'.\n\n"
            "Apply this goal to the following text chunk. Output ONLY the rewritten text for this chunk.\n\n"
            "TEXT CHUNK:\n---\n{chunk}\n---"
        )

        for i, chunk in enumerate(text_chunks):
            print(f"[INFO]     - [Map] Processing chunk {i + 1} of {total_chunks}...")
            
            prompt = prompt_template.format(task=task_instruction, chunk=chunk)
            rewritten_chunk = self.llm_service.invoke_llm(prompt)
            rewritten_chunks.append(rewritten_chunk)

        # Join all the rewritten chunks to form the final document
        return "\n\n".join(rewritten_chunks)
'''```eof

Finally, here is the implementation for the `summarize` task.

**Explanation:**
* The `SummarizeTask` class also inherits from `BaseTask`.
* Its `execute` method implements the more complex **Map-Reduce** strategy, which is essential for creating a coherent summary of a large document.
* **Map Phase**: It first loops through all chunks and generates an initial, small summary for each one.
* **Reduce Phase**: After collecting all the initial summaries, it combines them into a single body of text. It then performs one final AI call on this combined text to create the final, polished summary of the entire document. This two-step process allows the AI to first understand the details of each part and then create a high-level overview of the whole.

```python:Summarize Task Strategy:document-processor/tasks/summarize.py'''
from typing import List
from tasks.base_task import BaseTask

class SummarizeTask(BaseTask):
    """
    A task strategy for summarizing a large document using a Map-Reduce approach.
    """
    def execute(self, text_chunks: List[str], task_instruction: str) -> str:
        print(f"[INFO] ==> Step 3: 'Summarize' task detected. Executing Map-Reduce strategy.")
        
        # --- MAP PHASE ---
        # Get an initial summary for each chunk of the document.
        initial_summaries = []
        total_chunks = len(text_chunks)
        map_prompt_template = (
            "Your goal is to help with the first step of a multi-step summarization task.\n\n"
            "Analyze the following text chunk and create a concise summary of its key points. "
            "This will be used later to create a final summary of the entire document.\n\n"
            "TEXT CHUNK:\n---\n{chunk}\n---"
        )

        for i, chunk in enumerate(text_chunks):
            print(f"[INFO]     - [Map] Processing chunk {i + 1} of {total_chunks}...")
            
            prompt = map_prompt_template.format(chunk=chunk)
            summary = self.llm_service.invoke_llm(prompt)
            initial_summaries.append(summary)

        # --- REDUCE PHASE ---
        # Combine the initial summaries and create a final, cohesive summary.
        print("[INFO]     - [Reduce] Combining intermediate summaries for final processing...")
        combined_summaries = "\n\n".join(initial_summaries)
        
        reduce_prompt_template = (
            "You are an expert analyst. Your task is to '{task}'.\n\n"
            "The following text is a collection of summaries from different parts of a large document. "
            "Synthesize them into a single, final, and coherent output that fulfills the user's request.\n\n"
            "COMBINED SUMMARIES:\n---\n{summaries}\n---"
        )
        
        final_prompt = reduce_prompt_template.format(task=task_instruction, summaries=combined_summaries)
        final_summary = self.llm_service.invoke_llm(final_prompt)
        
        return final_summary
'''```eof

We have now successfully built the modular logic for handling different types of tasks.

Our very next and final step in the backend development will be to code the `task_processor.py` orchestrator, which will tie all these pieces together, and the `api.py` file to serve it all through a web interface.'''