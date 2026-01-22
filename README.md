
#  AI Document Processor


## Overview

This project provides a robust, scalable backend service to apply Large Language Model (LLM) tasks—such as summarization or rewriting—to documents of virtually unlimited size.

The core problem this system solves is the **fixed context window** of modern LLMs. A 500-page report cannot be processed in a single pass. This application acts as an intelligent orchestrator, breaking large documents into semantically-aware chunks, processing them individually, and then intelligently combining the results using a **Map-Reduce** strategy.

It is built as an API-first service, making it a perfect, scalable microservice for any application stack.

### Key Features

  * **Process Massive Documents**: Handles `.docx` files far exceeding standard LLM token limits (500+ pages).
  * **Intelligent Task Strategies**:
      * **Map-Reduce**: For holistic tasks like summarization, the system generates summaries of chunks and then combines them into a final, coherent summary.
      * **Simple Map**: For localized tasks like tone rewrites or translation, the system processes and stitches back each chunk in order.
  * **Modular & Extensible**: Built on a 4-layer architecture, allowing new tasks and capabilities to be added easily.
  * **Production-Ready API**: A high-performance [FastAPI](https://fastapi.tiangolo.com/) server with asynchronous-ready components.

### Target Audience

This tool is built for developers, data scientists, and organizations that need to automate workflows for large-scale document analysis, legal tech, academic research, or corporate reporting.

-----

## Architecture & Technical Summary

The system is engineered for scalability and maintainability, adhering to a 4-layer modular design.

1.  **API Layer (`/app`)**: A [FastAPI](https://fastapi.tiangolo.com/) server that handles HTTP requests, file uploads, and response delivery. It is the single entry point to the system.
2.  **Core Logic Layer (`/core`)**: The "brain" of the application. The `TaskProcessor` orchestrates the entire workflow, from loading the document (`DocumentLoader`) to splitting it (`TextSplitter`) and managing the AI interaction (`LLMService`).
3.  **Task Strategy Layer (`/tasks`)**: This layer defines *how* to perform a task. It contains distinct classes (`SummarizeTask`, `RewriteTask`) that inherit from a `BaseTask`, making the system "soft-coded" and extensible.
4.  **Utilities Layer (`/utils`)**: Contains helper functions for tasks like creating the final output `.docx` file.

The core of the system relies on [LangChain](https://www.langchain.com/) for intelligent text splitting and [OpenAI](https://openai.com/) for the LLM inference.

### Major Dependencies

  * **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
  * **Web Server**: [Uvicorn](https://www.uvicorn.org/)
  * **LLM Orchestration**: [LangChain](https://python.langchain.com/)
  * **LLM Provider**: [LangChain-OpenAI](https://python.langchain.com/docs/integrations/llms/openai)
  * **Document Handling**: [python-docx](https://python-docx.readthedocs.io/en/latest/)
  * **Data Validation**: [Pydantic](https://www.google.com/search?q=https://docs.pydantic.dev/)

-----

## Installation & Setup

Follow these steps to set up and run the project locally.

### Prerequisites

  * Python 3.10+
  * An [OpenAI API Key](https://platform.openai.com/api-keys)

### Step-by-Step Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/<your-github-user>/<PROJECT_NAME>.git
    cd <PROJECT_NAME>
    ```

2.  **Create and Activate a Virtual Environment**

    ```bash
    # Create the environment
    python -m venv .venv

    # Activate on Windows
    .\.venv\Scripts\activate

    # Activate on macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file in the project root. See the [Configuration](https://www.google.com/search?q=%23configuration) section for details.

    ```bash
    cp .env.example .env
    ```

    Now, edit the `.env` file and add your `OPENAI_API_KEY`.

5.  **Run the Server**

    ```bash
    uvicorn app.main:app --reload
    ```

    The server will start, and you will see a confirmation message:

    ```bash
    =====================================================================
    🚀 Your AI Document Processor is running!
    Access the API documentation (Swagger UI) here:
    👉 http://127.0.0.1:8000/docs 👈 (CTRL + Click to open)
    =====================================================================
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
    ```

-----

## Usage Examples

The easiest way to interact with the API is via the built-in documentation, accessible at `http://127.0.0.1:8000/docs`.

### API Request (cURL)

You can also send a `POST` request using cURL to process a document.

```bash
curl -X POST "http://127.0.0.1:8000/process-document/" \
     -H "Content-Type: multipart/form-data" \
     -F "task_instruction=Summarize this document into three key bullet points" \
     -F "file=@/path/to/your/large_report.docx" \
     --output result_report.docx
```

### Expected Server Output

When you send a request, the server terminal will provide real-time logs of the process, which is especially useful for long-running jobs.

```bash
INFO:     [STARTUP] Your AI Document Processor is running!
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     ==> Step 1: Request received for task 'Summarize...'. File: 'large_report.docx'.
INFO:     - Uploaded file saved to temporary path: C:\...\Temp\tmp123abc
INFO:     LLM Service initialized successfully.
INFO:     ==> Step 2a: Document loading successful.
INFO:     ==> Step 2b: Document successfully split into 528 processable chunks.
INFO:     ==> Step 3: 'Summarize' task detected. Executing Map-Reduce strategy.
INFO:     - [Map] Processing chunk 1 of 528...
INFO:     - [Map] Processing chunk 2 of 528...
...
INFO:     - [Map] Processing chunk 528 of 528...
INFO:     - [Reduce] Combining intermediate summaries for final processing...
INFO:     ==> Step 4a: Final result compiled. Creating output document...
INFO:     ==> Step 4b: Task complete. Sending response to client.
INFO:     "POST /process-document/" 200 OK
INFO:     Cleaned up temporary file: C:\...\Temp\tmp123abc
INFO:     Cleaned up temporary file: C:\...\Temp\result_large_report_xyz.docx
```

-----

## Configuration

Configuration is managed via environment variables loaded from a `.env` file in the project root.

| Variable | Description | Required | Default |
| :--- | :--- | :--- | :--- |
| `OPENAI_API_KEY` | Your secret key from OpenAI to authenticate API requests. | **Yes** | `None` |
| `HOST` | The host address to bind the server to. | No | `127.0.0.1` |
| `PORT` | The port to run the server on. | No | `8000` |

### `.env` Example

```text
# This file stores your secret API keys and configuration.
# DO NOT commit this file to version control.

# Replace sk-... with your actual OpenAI API key.
OPENAI_API_KEY="sk-..."

# Optional: Server configuration
HOST="127.0.0.1"
PORT=8000
```

-----

## Project Structure

The project follows a modular structure to separate concerns, making it easy to maintain and extend.

```plaintext
/document-processor/
│
├── .env                  # Secret keys and configuration (local)
├── .env.example          # Example configuration file
├── requirements.txt      # Python dependencies
├── README.md             # This file
│
├── app/
│   ├── __init__.py
│   ├── api_models.py     # Pydantic models for API request/response
│   └── main.py           # FastAPI server and API endpoints
│
├── core/
│   ├── __init__.py
│   ├── document_loader.py  # Reads and extracts text from .docx files
│   ├── llm_services.py   # Manages all communication with the LLM API
│   ├── task_processor.py # The main orchestrator for all tasks
│   └── text_splitter.py  # Intelligently chunks large documents
│
├── tasks/
│   ├── __init__.py
│   ├── base_task.py      # Abstract base class (blueprint) for all tasks
│   ├── rewrite.py        # Implements the "Simple Map" strategy
│   └── summarize.py      # Implements the "Map-Reduce" strategy
│
└── utils/
    ├── __init__.py
    └── file_utils.py     # Helper functions (e.g., create output .docx)
```

-----

## Contributing Guidelines

We welcome contributions\! Please follow these guidelines to help us maintain the project's quality.

1.  **Fork** the repository and clone it to your local machine.
2.  Create a new branch for your feature or bug fix: `git checkout -b feature/my-new-feature`
3.  Make your changes and ensure your code adheres to the project's style.

### Linting & Formatting

We use [Black](https://github.com/psf/black) for formatting and [Flake8](https://flake8.pycqa.org/en/latest/) for linting. Before committing, please run:

```bash
# Install dev dependencies
pip install black flake8

# Run formatter
black .

# Run linter
flake8 .
```

### Commit Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This helps in automating changelogs and versioning.

  * `feat:`: A new feature
  * `fix:`: A bug fix
  * `docs:`: Documentation only changes
  * `style:`: Code style changes (formatting, etc.)
  * `refactor:`: A code change that neither fixes a bug nor adds a feature
  * `test:`: Adding missing tests or correcting existing tests

**Example commit:** `git commit -m "feat: add support for PDF file ingestion"`

### Submitting a Pull Request

1.  Push your branch to your fork: `git push origin feature/my-new-feature`
2.  Open a **Pull Request (PR)** against the `main` branch of the original repository.
3.  In your PR description, clearly explain the problem you're solving and the changes you've made.
4.  Ensure all CI checks pass.

-----

## Testing

To run the test suite, navigate to the project root and run:

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
python -m pytest

# Run tests with coverage report
python -m pytest --cov=./ --cov-report=html
```

This will run all unit tests and generate an HTML coverage report in a `htmlcov/` directory.

-----

## Deployment / CI-CD

### Docker

This application is container-ready. You can build and run it using Docker.

1.  **Build the Docker Image**

    ```bash
    docker build -t document-processor:latest .
    ```

2.  **Run the Docker Container**
    Be sure to pass in your `.env` file for the container to access the API key.

    ```bash
    docker run -p 8000:8000 --env-file .env -d document-processor:latest
    ```

### CI/CD

A basic CI pipeline is configured using **GitHub Actions** (see `.github/workflows/ci.yml`). This pipeline triggers on every push to `main` and on all pull requests. It performs the following steps:

1.  Sets up a Python environment
2.  Installs dependencies
3.  Runs the linter (Flake8)
4.  Runs the formatter (Black --check)
5.  Runs the test suite (Pytest)

-----

## Roadmap / Future Work

This project has a strong foundation, but there's always room to grow. Planned enhancements include:

  * [ ] **Asynchronous Processing**: Convert the synchronous `/process-document/` endpoint to an async workflow using [Celery](https://docs.celeryq.dev/en/stable/) and [Redis](https://redis.io/) for long-running jobs.
  * [ ] **Frontend User Interface**: Build a [React](https://react.dev/) or [Vue.js](https://vuejs.org/) frontend to provide a user-friendly drag-and-drop interface.
  * [ ] **Support for More File Types**: Add `document_loader` modules for `.pdf` and `.txt` files.
  * [ ] **Support for Local LLMs**: Abstract the `LLMService` to support local models (e.g., via [Ollama](https://ollama.com/) or [HuggingFace](https://huggingface.co/)) to reduce API costs.
  * [ ] **Enhanced Caching**: Implement caching for processed chunks to speed up repeated requests.

-----

## License & Credits

This project is licensed under the **MIT License**. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

This project was built using the following outstanding open-source libraries:

  * [FastAPI](https://fastapi.tiangolo.com/)
  * [LangChain](https://www.langchain.com/)
  * [Pydantic](https://www.google.com/search?q=https://docs.pydantic.dev/)
  * [python-docx](https://python-docx.readthedocs.io/en/latest/)


# Execution Video


https://github.com/user-attachments/assets/a9198740-e7f9-428f-b7f1-67adcfb856e5
