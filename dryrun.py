"""Dry-run harness: runs mrta_robotarium.main() in the Robotarium simulator
for one (n, seed) with overridden configuration. Usage:
    python3 dryrun.py N SEED [P_LOSS ...]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mrta_robotarium as M
M.N_ROBOTS = int(sys.argv[1])
M.SEEDS = [int(sys.argv[2])]
if len(sys.argv) > 3:
    M.P_LOSS_LEVELS = [float(v) for v in sys.argv[3:]]
if os.environ.get("METHODS"):
    M.METHODS = os.environ["METHODS"].split(",")
if os.environ.get("TMAX"):
    M.MAX_MISSION_TIME = float(os.environ["TMAX"])
M.SIMULATION = True
M.main()
