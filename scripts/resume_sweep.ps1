# GPU-ready comparison harness for ContextBench - RESUME SCRIPT
# Resumes the sweep from policy=summary, budget=1024

echo "=========================================================="
echo "Resuming ContextBench Sweep"
echo "Continuing from summary@1024, followed by retrieval and delta_context"
echo "=========================================================="

# 1. Finish the summary policy for the remaining 1024 budget
python benchmark.py -m policy=summary budget=1024

# 2. Run the remaining policies for all budgets
python benchmark.py -m policy=retrieval,delta_context budget=256,512,1024

echo "=========================================================="
echo "Resume Sweep Complete! Check the outputs directory for results."
echo "=========================================================="
