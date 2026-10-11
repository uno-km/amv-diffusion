#!/data/data/com.termux/files/usr/bin/bash
termux-wake-lock
cd /data/data/com.termux/files/home
export PYTHONUNBUFFERED=1
nohup python3 -u /data/data/com.termux/files/home/run_korean_man_spring_3tiers.py --tier all > /data/data/com.termux/files/home/spring_man_benchmark.log 2>&1 &
PID=$!
echo $PID > /data/data/com.termux/files/home/spring_man_benchmark.pid
echo "LAUNCHED_PID: $PID"
