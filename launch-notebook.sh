#!/bin/bash
CONTAINER_ENGINE="docker"
${CONTAINER_ENGINE} run -ti -v $(pwd):/shared -w /shared -p 8888:8888 --init dolfinx/lab
