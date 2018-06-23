FROM python:2.7

RUN apt-get update && apt-get --yes install gfortran && rm -rf /var/lib/apt/lists/*

#RUN git clone https://github.com/emerrf/gym-wind-turbine.git /app/gym-wind-turbine
ADD . /app/gym-wind-turbine
WORKDIR /app/gym-wind-turbine

RUN pip install numpy==1.12.1
RUN pip install -r /app/gym-wind-turbine/requirements.txt

ENTRYPOINT ["gwt-run-real-control-test"]