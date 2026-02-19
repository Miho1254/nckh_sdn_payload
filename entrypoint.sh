#!/bin/bash
# Khởi động OVS + PostgreSQL trong container

# ── Dọn dẹp tiến trình cũ ──────────────────────────────────
killall -q ovsdb-server ovs-vswitchd 2>/dev/null || true
rm -f /var/run/openvswitch/ovsdb-server.pid
rm -f /var/run/openvswitch/ovs-vswitchd.pid
rm -f /var/run/openvswitch/db.sock
mn -c > /dev/null 2>&1 || true

# ── Khởi động OVS ──────────────────────────────────────────
mkdir -p /var/run/openvswitch /var/log/openvswitch

if [ ! -f /etc/openvswitch/conf.db ]; then
    ovsdb-tool create /etc/openvswitch/conf.db /usr/share/openvswitch/vswitch.ovsschema
fi

ovsdb-server --remote=punix:/var/run/openvswitch/db.sock \
    --remote=db:Open_vSwitch,Open_vSwitch,manager_options \
    --pidfile --detach --log-file

ovs-vsctl --no-wait init
ovs-vswitchd --pidfile --detach --log-file --no-mlockall 2>/dev/null || true

echo "✓ Open vSwitch started"

# ── Khởi động PostgreSQL ───────────────────────────────────
PG_DATA="/var/lib/postgresql/14/main"
PG_CONF="/etc/postgresql/14/main"

# Init cluster nếu chưa có
if [ ! -d "$PG_DATA" ]; then
    su - postgres -c "/usr/lib/postgresql/14/bin/initdb -D $PG_DATA"
fi

# Cho phép kết nối từ mọi IP (Mininet hosts)
if ! grep -q "0.0.0.0/0" "$PG_CONF/pg_hba.conf" 2>/dev/null; then
    echo "host all all 0.0.0.0/0 md5" >> "$PG_CONF/pg_hba.conf"
fi

# Listen trên tất cả interfaces
sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" "$PG_CONF/postgresql.conf" 2>/dev/null || true

# Start PostgreSQL
su - postgres -c "/usr/lib/postgresql/14/bin/pg_ctl -D $PG_DATA -l /var/log/postgresql/postgresql.log start" 2>/dev/null

# Tạo user + database
sleep 2
su - postgres -c "psql -tc \"SELECT 1 FROM pg_user WHERE usename='lms'\" | grep -q 1 || psql -c \"CREATE USER lms WITH PASSWORD 'lms123';\"" 2>/dev/null
su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='lms'\" | grep -q 1 || psql -c \"CREATE DATABASE lms OWNER lms;\"" 2>/dev/null

echo "✓ PostgreSQL started (user: lms, db: lms)"

exec "$@"
