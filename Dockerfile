FROM eoxa/eoxserver:latest

ADD requirements.txt .
RUN pip3 install -r requirements.txt && \
    apt-get update && \
    apt-get install -y python3-netcdf4

WORKDIR /home/ogc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_APP edc_ogc/app.py
COPY edc_ogc/. edc_ogc/.

ENTRYPOINT []
CMD ["flask", "run", "--host=0.0.0.0"]