from typing import Dict, Type
from .document_loader import load_text_from_docx
from .text_splitter import split_text_into_chunks
from .llm_services import LLMService
from tasks.base_task import BaseTask
from tasks.summarize import SummarizeTask
from tasks.rewrite import RewriteTask
from utils.file_utils import create_docx_from_text

class TaskProcessor:
    """
    The main orchestrator that manages the entire document processing workflow.
    """
    def __init__(self):
        """
        Initializes the TaskProcessor by setting up the LLM service and
        registering the available task strategies.
        """
        self.llm_service = LLMService()
        self.tasks: Dict[str, Type[BaseTask]] = {
            "summarize": SummarizeTask,
            "rewrite": RewriteTask,
        }
        print("[INFO] Task Processor initialized with available tasks: summarize, rewrite.")

    def _select_task(self, task_instruction: str) -> BaseTask:
        """
        Intelligently selects the appropriate task strategy based on keywords
        in the user's instruction.
        
        Args:
            task_instruction: The user's plain-text instruction.
            
        Returns:
            An initialized instance of the selected task class.
        """
        instruction_lower = task_instruction.lower()
        summary_keywords = ["summary", "summarize", "summarise", "extract", "bullet points", "overview"]

        # If any summary keyword is found, select the SummarizeTask.
        if any(keyword in instruction_lower for keyword in summary_keywords):
            print("[INFO] Selected 'SummarizeTask' strategy.")
            return self.tasks["summarize"](self.llm_service)
        
        # Otherwise, default to the RewriteTask for all other types of edits.
        print("[INFO] Defaulting to 'RewriteTask' strategy.")
        return self.tasks["rewrite"](self.llm_service)

    def process_task(self, file_path: str, task_instruction: str, original_filename: str) -> str:
        """
        Executes the full end-to-end document processing pipeline.

        Args:
            file_path: The path to the uploaded .docx file.
            task_instruction: The user's instruction.
            original_filename: The original name of the uploaded file.

        Returns:
            The file path of the final, processed .docx document.
        """
        # Step 2a: Load the document text
        document_text = load_text_from_docx(file_path)

        # Step 2b: Split the text into manageable chunks
        text_chunks = split_text_into_chunks(document_text)

        # Step 3: Select and execute the appropriate task strategy
        task_strategy = self._select_task(task_instruction)
        final_text = task_strategy.execute(text_chunks, task_instruction)

        # Step 4a: Create the final .docx file from the result text
        output_path = create_docx_from_text(final_text, original_filename)
        
        return output_path
'''```eof

---

### **Step 8: Coding the API Layer**

This is the final layer that exposes our powerful backend to the world (and to a future frontend).

First, here is the code for `api_models.py`.

**Explanation:**
* This file is a FastAPI best practice. It uses `Pydantic` to define the exact structure of our API's responses.
* This ensures our API always returns data in a consistent format and provides automatic validation and documentation.

```python:API Data Models:document-processor/app/api_models.py'''
from pydantic import BaseModel

class TaskResponse(BaseModel):
    """
    A Pydantic model to define the structure of a successful task response.
    This is not used for the file download but could be used for other endpoints.
    """
    message: str
    task: str
    input_file: str
    output_file: str

class ErrorResponse(BaseModel):
    """
    A Pydantic model for returning structured error messages.
    """
    detail: str
'''```eof

Finally, here is the code for `main.py`, the entry point for our FastAPI server.

**Explanation:**
* This file sets up the FastAPI application.
* `TaskProcessor()` is initialized once when the server starts. This is efficient as it prevents reloading the AI models on every request.
* It defines the single `POST /process-document/` endpoint.
* This endpoint handles file uploads, saves the uploaded file to a temporary location, and then calls our `TaskProcessor` to do the heavy lifting.
* It uses FastAPI's `BackgroundTasks` to reliably clean up the temporary input and output files after the response has been sent to the user. This is a robust, professional pattern for handling file-based workflows.

```python:FastAPI Server:document-processor/app/main.py'''
import os
import tempfile
import traceback
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from core.task_processor import TaskProcessor

# Initialize the FastAPI app
app = FastAPI(
    title="Scalable Document Processor",
    description="An expert-level API to process large documents with AI."
)

# Initialize our main processor once at startup for efficiency
processor = TaskProcessor()

def safe_delete(path: str):
    """A robust function to delete a file, ignoring errors if it's already gone."""
    try:
        if os.path.exists(path):
            os.unlink(path)
            print(f"[INFO] Cleaned up temporary file: {path}")
    except Exception as e:
        print(f"[ERROR] Could not clean up file {path}. Reason: {e}")

@app.post("/process-document/")
async def process_document_endpoint(
    background_tasks: BackgroundTasks,
    task_instruction: str = Form(..., description="The specific task to perform (e.g., 'Summarize this document')."),
    file: UploadFile = File(...)
):
    """
    The main endpoint to upload a .docx file and process it.
    """
    if not file.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .docx file.")

    print(f"[INFO] ==> Step 1: Request received for task '{task_instruction}'. File: '{file.filename}'.")
    
    uploaded_file_path = None
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_upload:
            uploaded_file_path = tmp_upload.name
            tmp_upload.write(await file.read())
            print(f"[INFO]     - Uploaded file saved to temporary path: {uploaded_file_path}")

        # Call the core processor to handle the entire workflow
        result_file_path = processor.process_task(
            file_path=uploaded_file_path,
            task_instruction=task_instruction,
            original_filename=file.filename
        )
        
        # Schedule the temporary files to be deleted after the response is sent
        background_tasks.add_task(safe_delete, uploaded_file_path)
        background_tasks.add_task(safe_delete, result_file_path)
        
        print(f"[INFO] ==> Step 4b: Task complete. Sending response to client.")
        
        return FileResponse(
            path=result_file_path,
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            filename=f"result_{file.filename}"
        )
        
    except Exception as e:
        # If anything goes wrong, log the error and clean up the uploaded file
        traceback.print_exc()
        if uploaded_file_path:
            background_tasks.add_task(safe_delete, uploaded_file_path)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

if __name__ == "__main__":
    # This allows running the app directly for development
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''```eof

Your scalable, modular, and robust `document-processor` is now complete. The final step is to install the dependencies and run the server.'''