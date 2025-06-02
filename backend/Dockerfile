# Use an official Python runtime as a parent image
FROM python:3.11-slim
# Create a non-root user and group
USER root
# Set the working directory in the container
WORKDIR /code
RUN apt-get update && apt-get install -y python3-opencv

# Copy the requirements file into the container at /code
COPY ./requirements.txt /code/requirements.txt
ENV PYTHONPATH=/code
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the content of the local src directory to the working directory
USER root
COPY ./app /code/app
COPY ./realesrgan /code/realesrgan
COPY ./weights /code/weights

# Command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]