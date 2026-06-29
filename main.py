import asyncio
import subprocess
import time
import httpx
from fastapi import FastAPI, Request, Response, HTTPException
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
        "--served-model-name", "jotham",
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", GPU_MEMORY_UTIL,
        "--enable-sleep-mode",
        # Locally deployed qwen3 agent
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
    ]
    print(f"Starting vLLM on port {VLLM_PORT}...")
    vllm_process = subprocess.Popen(cmd, env=env)
    
    
async def wait_for_vllm_ready():
    """
    Start the idle loop only after the server is done loading the model
    Readiness is defined as the vllm endpoint returning 200 for Status
    """
    poll_interval = 5
    while True:
        try:
            response = await client.get("/health")
            if response.status_code == 200:
                print("Done loading")
                return True
        except (httpx.ConnectError, httpx.HTTPError):
            pass
        await asyncio.sleep(poll_interval)


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
    global last_activity_time
    start_vllm()
    await wait_for_vllm_ready()
    last_activity_time = time.time()
    asyncio.create_task(idle_monitor_loop())


@app.on_event("shutdown")
async def shutdown_event():
    global vllm_process
    vllm_process.kill()


async def idle_monitor_loop():
    global last_activity_time
    while True:
        await asyncio.sleep(10)
        if time.time() - last_activity_time > IDLE_TIMEOUT:
            await sleep_vllm()


@app.get("/status")
async def status():
    """Call nvidia-smi on the machine and return the output in a CSV friendly manner"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            name, mem_used, mem_total, util, temp = [x.strip() for x in line.split(",")]
            gpus.append({
                "name": name,
                "memory_used": int(mem_used),
                "memory_total": int(mem_total),
                "utilization": int(util),
                "temperature": int(temp),   
            })
        return {"gpus": gpus}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr)


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
            params=request.query_params,
        )
        vllm_response = await client.send(req)

        return Response(
            content=vllm_response.content,
            status_code=vllm_response.status_code,
            headers=dict(vllm_response.headers),
        )
    except httpx.RequestError as exc:
        return Response(
            content=f"Proxy error: vLLM backend unavailable ({exc})", status_code=503
        )


if __name__ == "__main__":
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT)
