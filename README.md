# 1. Send Pi-side script (only file needed there)
export RPI_HOST=raspberrypi@raspberry.pi
ssh $RPI_HOST 'mkdir -p ~/bep/scripts ~/bep/queue/incoming ~/bep/queue/done ~/bep/results'
rsync -av scripts/pi_daemon.py $RPI_HOST:~/bep/scripts/

On Pi (in a dedicated terminal/tmux)

sudo cpupower frequency-set -g performance
taskset -c 3 python3 ~/bep/scripts/pi_daemon.py
# leaves running, prints "arch_<N>: 9 rows" per arch

Back on Mac (in another terminal)

# 2. test 5 archs
uv run python scripts/stream_to_pi.py --limit 5

# 3. Full run (6466 non-iso archs)
uv run python scripts/stream_to_pi.py

After run completes — on Mac

# 4. Pull CSV back
rsync -av $RPI_HOST:~/bep/results/latency_rpi.csv results/

# 5. Join accuracies (cifar100 from nas_201_api, ninapro/darcy from NB360 pickles)
uv run python scripts/join_accuracies.py
# -> results/hwnas_bench_360_v1.csv