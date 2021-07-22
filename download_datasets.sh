#!/bin/sh

echo 'Starting to download dataset' && \
cd causal_graphs && \
gdown https://drive.google.com/uc?id=1mJXJpvkG8Ol4w6QlbzW4EETjpXmHPlMX && \
unzip -q exported_graphs.zip && \
rm exported_graphs.zip && \
echo 'Download complete. Data can be found under causal_graphs/...'