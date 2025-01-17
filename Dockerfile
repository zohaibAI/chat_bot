FROM python:3.11-slim

# Set timezone
ENV TZ="UTC"

# Set work directory
WORKDIR /CHAT_BOT/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Update pyproject.toml to include Groq dependencies
RUN poetry add langchain-groq groq

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-root

# Copy the rest of the application files
COPY . .

# Set Python path
ENV PYTHONPATH=/CHAT_BOT

# Expose port
EXPOSE 9002

# Run the application
CMD ["poetry", "run", "uvicorn", "main:app", "--reload", "--reload-delay", "1", "--log-level", "debug", "--reload-dir", "/", "--host", "0.0.0.0", "--port", "9002"]