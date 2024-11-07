#!/bin/bash

CONTAINER=$1

xhost +local:root

docker exec -it \
    -u $(id -u):$(id -g) \
    ${CONTAINER:=lio_sam} \
    /bin/bash
