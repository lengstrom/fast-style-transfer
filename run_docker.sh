#!/bin/sh

docker run -ti \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output \
    thhuang/fast_style_transfer bash
