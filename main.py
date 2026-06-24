import asyncio
import subprocess
import time
import httpx
from fastapi import FastAPI, Request, Response
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()
# Mandatory for .env to have
# HF_TOKEN
# VLLM_SERVER_DEV_MODE
# Optional for the rest

PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.getenv("PROXY_PORT", 42016))
VLLM_PORT = int(os.getenv("VLLM_PORT", 42017))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", 600))
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
GPU_MEMORY_UTIL = os.getenv("GPU_MEMORY_UTIL", "0.90")

app = FastAPI()
vllm_process = None
last_activity_time = time.time()
is_sleeping = False
client = httpx.AsyncClient(base_url=f"http://0.0.0.0:{VLLM_PORT}", timeout=None)

def start_vllm():
    global vllm_process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", GPU_MEMORY_UTIL,
        "--enable-sleep-mode"
    ]
    print(f"Starting vLLM on port {VLLM_PORT}...")
    vllm_process = subprocess.Popen(cmd, env=env)

async def wake_vllm():
    global is_sleeping
    if is_sleeping:
        print("Waking")
        try:
            await client.post("/wake_up")
            is_sleeping = False
        except Exception as e:
            print(f"[Manager] Error attempting to wake vLLM: {e}")

async def sleep_vllm():
    global is_sleeping
    if not is_sleeping:
        print("Sleep...")
        try:
            await client.post("/sleep?level=1")
            is_sleeping = True
        except Exception as e:
            print(f"[Manager] Error attempting to sleep vLLM: {e}")

@app.on_event("startup")
async def startup_event():
    start_vllm()
    asyncio.create_task(idle_monitor_loop())

async def idle_monitor_loop():
    global last_activity_time
    while True:
        await asyncio.sleep(10)
        if time.time() - last_activity_time > IDLE_TIMEOUT:
            await sleep_vllm()

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_catch_all(request: Request, path: str):
    global last_activity_time
    
    last_activity_time = time.time()
    await wake_vllm()

    url = f"/{path}"
    headers = dict(request.headers)
    headers.pop("host", None) 
    content = await request.body()

    try:
        req = client.build_request(
            method=request.method,
            url=url,
            headers=headers,
            content=content,
            params=request.query_params
        )
        vllm_response = await client.send(req)
        
        return Response(
            content=vllm_response.content,
            status_code=vllm_response.status_code,
            headers=dict(vllm_response.headers)
        )
    except httpx.RequestError as exc:
        return Response(content=f"Proxy error: vLLM backend unavailable ({exc})", status_code=503)

if __name__ == "__main__":
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT)
