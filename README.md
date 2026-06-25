# Jotham's LLM Proxy for vLLM

This is a simple uvicorn server that acts as a proxy/sleeper for vLLM.

vLLM offers a sleep mode but its serving layer does not have an auto sleeper
function: https://docs.vllm.ai/en/latest/features/sleep_mode/#rlhf-weight-updates.

So this uvicorn server simply wakes it up on demand and sleep when idle.
It also exposes a status get method to retrieve the output of nvidia-smi on the
gpu server. This exists to facilitate telegram bots.

Immediate TODOs:
- [ ] Status endpoint should return model name and idle or not
- [ ] Should provide endpoint to change model. Entails unloading and loading a new model.