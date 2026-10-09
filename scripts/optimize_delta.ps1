# GPU-ready optimization sweep for delta_context policy

echo "=========================================================="
echo "Starting DeltaContext Optimization Sweep"
echo "Sweeping: change_threshold, accumulator, landmark_fraction"
echo "Budgets: 512, 1024"
echo "=========================================================="

python benchmark.py -m policy=delta_context budget=512,1024 +policy.change_threshold=0.2,0.4,0.6,0.8 +policy.accumulator=sum,displacement policy.landmark_fraction=0.2,0.4

echo "=========================================================="
echo "Sweep Complete! Check the multirun or outputs directory for results."
echo "=========================================================="
