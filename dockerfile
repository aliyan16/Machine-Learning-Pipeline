FROM python:3.11.1
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirement.txt
EXPOSE 5000
ENV flaskApp=app.py
CMD ["flask","run","--host.0.0.0.0"]