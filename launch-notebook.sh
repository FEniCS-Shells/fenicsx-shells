#!/bin/bash
CONTAINER_ENGINE="docker"
${CONTAINER_ENGINE} run --rm -ti -v $(pwd):/shared -w /shared --publish-all --init jhale/fenics-shellsx:latest
