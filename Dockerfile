FROM python:3.11.4

WORKDIR /dashboard

COPY predicter.py dashboard_docker.py stock_dict.json INFO.csv requirements.txt /dashboard/

RUN pip install --no-cache-dir -r /dashboard/requirements.txt

ENV PORT 3000

EXPOSE $PORT

CMD ["python", "dashboard_docker.py"]