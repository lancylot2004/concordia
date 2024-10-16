# PYTHONPATH=. PYTHONSAFEPATH=1 python examples/modular/launch_concordia_challenge_evaluation.py \
#   --agent=basic_agent \
#   --api_type=local \
#   --embedder=all-mpnet-base-v2 \
#   --num_repetitions_per_scenario=1 \

PYTHONPATH=. PYTHONSAFEPATH=1 python examples/modular/launch_one_scenario.py \
  --agent=defect_agent_1 \
  --scenario=labor_collective_action__fixed_rule_boss_0 \
  --api_type=ollama \
  --model=gemma2:9b-instruct-q8_0 \
  --embedder=all-mpnet-base-v2 \
  --num_repetitions_per_scenario=1
