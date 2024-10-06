# PYTHONPATH=. PYTHONSAFEPATH=1 python examples/modular/launch_concordia_challenge_evaluation.py \
#   --agent=basic_agent \
#   --api_type=local \
#   --embedder=all-mpnet-base-v2 \
#   --num_repetitions_per_scenario=1 \

PYTHONPATH=. PYTHONSAFEPATH=1 python examples/modular/launch_one_scenario.py \
  --agent=basic_agent \
  --scenario=reality_show_circa_2003_stag_hunt_0 \
  --api_type=local \
  --embedder=all-mpnet-base-v2 \
  --num_repetitions_per_scenario=1
