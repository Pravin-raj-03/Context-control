# GPU-ready comparison harness for ContextBench
# Sweeps across baseline policies and different token budgets.

# To use on GPU, ensure your environment is configured (e.g. PyTorch CUDA)
# and adjust the model config if needed.

echo "=========================================================="
echo "Starting ContextBench Sweep"
echo "Policies: sliding_window, summary, retrieval, delta_context"
echo "Budgets: 256, 512, 1024"
echo "=========================================================="

# We use Hydra's multirun flag (-m) to automatically iterate over the combinations.
# Adjust the budget list or policy list as needed.
python benchmark.py -m policy=retrieval,delta_context budget=256,512,1024

echo "=========================================================="
echo "Sweep Complete! Check the outputs directory for results."
echo "=========================================================="
