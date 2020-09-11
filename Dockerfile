FROM eoxa/eoxserver:latest

ADD requirements.txt .
RUN pip3 install -r requirements.txt

RUN apt-get install unzip wget -y && \
    wget -q https://github.com/geopython/pygeoapi/archive/master.zip && \
    unzip master.zip && \
    cd pygeoapi-master && \
    pip3 install -r requirements.txt && \
    python3 setup.py install

ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /home/ogc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_APP edc_ogc/app.py
COPY edc_ogc/. edc_ogc/.

ENTRYPOINT []
CMD ["flask", "run", "--host=0.0.0.0"]