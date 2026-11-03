FROM dolfinx/dolfinx:stable

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/

# Install into the image's existing env so dolfinx/gmsh/numpy stay visible.
ENV UV_PROJECT_ENVIRONMENT=/dolfinx-env \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --inexact --no-dev

WORKDIR /work
ENTRYPOINT ["fea-pipeline"]
