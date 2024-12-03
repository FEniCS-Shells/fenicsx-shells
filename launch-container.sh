#!/bin/bash
CONTAINER_ENGINE="podman"
${CONTAINER_ENGINE} run -ti -v $(pwd):/shared -w /shared dolfinx/dolfinx:nightly
