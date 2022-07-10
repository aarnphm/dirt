FROM debian:latest as build1
RUN echo "hello"
FROM alpine as build2

FROM base-image as test

FROM base

COPY --from=build1 /hello /hello
