FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ───────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    mininet \
    openvswitch-switch \
    iperf \
    iperf3 \
    iproute2 \
    net-tools \
    iputils-ping \
    dnsutils \
    wget \
    tcpdump \
    traceroute \
    hping3 \
    python3 \
    python3-pip \
    git \
    curl \
    nodejs \
    npm \
    postgresql \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# ── Node.js global tools ─────────────────────────────────────
RUN npm install -g artillery

# ── Python packages (Ryu + ML) ───────────────────────────────
RUN pip3 install ryu \
    eventlet==0.33.3 \
    torch \
    pytorch-lightning \
    pandas \
    numpy \
    matplotlib \
    scikit-learn

# Fix Ryu/eventlet ALREADY_HANDLED incompatibility
RUN sed -i 's/from eventlet.wsgi import ALREADY_HANDLED/ALREADY_HANDLED = object()/' \
    /usr/local/lib/python3.10/dist-packages/ryu/app/wsgi.py

# ── LMS Backend: install dependencies ────────────────────────
WORKDIR /work/lms/backend
COPY lms/backend/package.json ./
RUN npm install --production

# ── LMS Frontend: install + build ─────────────────────────────
WORKDIR /work/lms/frontend
COPY lms/frontend/package.json ./
RUN npm install
COPY lms/frontend/ ./
RUN npx vite build

# ── Copy tất cả project files ────────────────────────────────
WORKDIR /work
COPY . .

# ── Entrypoint ────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
