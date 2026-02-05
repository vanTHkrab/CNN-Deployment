FROM ubuntu:latest
LABEL authors="vanthkrab"

ENTRYPOINT ["top", "-b"]