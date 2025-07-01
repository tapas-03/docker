# Use Python base image
FROM python:3.12-slim

# Set working directory (Creates a seperate directory,All commands in the container will execute from /app)
WORKDIR /app

# Copy HTML and Python files
COPY index.html .
COPY server.py .

# Expose port 80
EXPOSE 80

# Run the server
CMD ["python", "server.py"]
