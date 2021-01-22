FROM dolfinx/lab
LABEL description="DOLFIN-X Jupyter Lab for Binder"

USER root

RUN apt-get update && \
	apt-get install python3-pip -y
    
RUN pip3 install --no-cache --upgrade pip && \
    pip3 install --no-cache notebook

ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}
ENV PETSC_ARCH "linux-gnu-real-32"
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

WORKDIR ${HOME}
COPY . ${HOME}
USER ${USER}
