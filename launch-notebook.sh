#!/bin/bash
CONTAINER_ENGINE="docker"
docker pull dolfinx/lab
${CONTAINER_ENGINE} run -ti -v $(pwd):/shared -w /shared --publish-all --init dolfinx/lab
