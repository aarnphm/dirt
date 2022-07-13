FROM ghcr.io/railwayapp/nixpacks:debian-1657555817

WORKDIR /app/

# Setup
COPY environment.nix /app/
RUN nix-env -if environment.nix
RUN apt-get update && apt-get install -y libgmp-dev libnuma1 libnuma-dev libtinfo-dev libtinfo5 libc6-dev libtinfo6 llvm* clang ninja-build zlib1g-dev gcc binutils make




# Load environment variables


# Install
COPY . /app/
RUN  stack setup




# Build

RUN  stack build

# Start
COPY . /app/
CMD stack exec haskell-stack-exe
