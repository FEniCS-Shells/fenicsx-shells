#!/bin/bash
docker pull dolfinx/lab:latest
docker buildx build --file Dockerfile.local -t jhale/fenics-shellsx:latest .
