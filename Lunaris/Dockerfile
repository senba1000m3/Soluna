# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Copy the project files
COPY pyproject.toml .
COPY .python-version .

# Copy the rest of the application code
COPY . .

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application using uv
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
