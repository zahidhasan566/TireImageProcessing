FROM teamacimisai/django-mssql-base-image-python3.12:latest

WORKDIR /app

COPY requirements.txt .
# install dependencies
RUN pip install pip --upgrade
RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

# copy project
COPY . .
EXPOSE 4000
