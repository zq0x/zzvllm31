from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import torch
import pynvml
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM
import gc
import logging
import argparse

LOG_PATH= './logs'
LOGFILE_CONTAINER = './logs/logfile_container_vllm.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] started logging in {LOGFILE_CONTAINER}')



def initialize_nvml():
    try:
        pynvml.nvmlInit()
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [initialize_nvml] NVML initialized successfully.')
    except Exception as e:
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [initialize_nvml] Failed to initialize NVML: {e}')

initialize_nvml()

# Check if CUDA is available
def cuda_support_bool():
    try:
        res_cuda_support = torch.cuda.is_available()
        if res_cuda_support:
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [cuda_support_bool] CUDA found')
        else:
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [cuda_support_bool] CUDA not supported!')
        return res_cuda_support
    except Exception as e:
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [cuda_support_bool] {e}')
        return False

# Return the number of CUDA GPUs available
def cuda_device_count():
    try:
        res_gpu_int = torch.cuda.device_count()
        res_gpu_int_arr = [i for i in range(res_gpu_int)]
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [cuda_support_bool] {str(res_gpu_int)}x GPUs found {res_gpu_int_arr}')
        return res_gpu_int_arr
    except Exception as e:
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [cuda_device_count] Failed to get device_count. Using default [0]. Error: {e}')
        return [0]

gpu_int_arr = cuda_device_count()

# FastAPI app
app = FastAPI()

# Global LLM instance
llm_instance = None

@app.get("/")
async def root():
    return f'Hello from vllm server!'

@app.post("/vllm")
async def vllmt(request: Request):
    global llm_instance
    try:
        req_data = await request.json()
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [/vllmt] [req_data] {req_data}')
        if req_data["req_type"] == "clear":
            try:
                print(f' @ ********* CLEARING GPU MEMORY 0/3 *********')
                torch.cuda.empty_cache()
                print(f' @ ********* CLEARING GPU MEMORY 1/3 EMPTIED CACHE *********')
                torch.cuda.reset_max_memory_allocated()
                print(f' @ ********* CLEARING GPU MEMORY 2/3 reset_max_memory_allocate *********')
                gc.collect()
                print(f' @ ********* CLEARING GPU MEMORY 3/3 gc.collect() *********')
                print(f' @ ********* FINISHED *********')
            except Exception as e:
                logging.error(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return JSONResponse({"result_status": 404, "result_data": str(e)})
        if req_data["req_type"] == "load":
            try:
                req_model = req_data.get("model", "facebook/opt-125m")
                req_tensor_parallel_size = req_data.get("tensor_parallel_size", 1)
                req_gpu_memory_utilization = req_data.get("gpu_memory_utilization", 0.91)
                req_max_model_len = req_data.get("max_model_len", 1024)
                req_cpu_offload_gb = req_data.get("cpu_offload_gb", 0)
                req_enforce_eager = req_data.get("enforce_eager", True)
                req_enable_prefix_caching = req_data.get("enable_prefix_caching", True)
                req_dtype = req_data.get("dtype", "auto")
                req_torch_dtype = req_data.get("torch_dtype", "auto")
                req_kv_cache_dtype = req_data.get("kv_cache_dtype", "auto")
                req_swap_space = req_data.get("swap_space", 4)
                req_enable_chunked_prefill = req_data.get("enable_chunked_prefill", True)
                req_trust_remote_code = req_data.get("trust_remote_code", True)
                req_model_storage = req_data.get("model_storage", "/models")
                req_model_path = f'{req_model_storage}/{req_model}'
                

                if args.model:
                    print(f' @@@ args.model: {args.model}')
                    req_model = args.model
                if args.tensor_parallel_size:
                    print(f' @@@ args.tensor_parallel_size: {args.tensor_parallel_size}')
                    req_tensor_parallel_size = args.tensor_parallel_size
                if args.gpu_memory_utilization:
                    print(f' @@@ args.gpu_memory_utilization: {args.gpu_memory_utilization}')
                    req_gpu_memory_utilization = args.gpu_memory_utilization
                if args.max_model_len:
                    print(f' @@@ args.max_model_len: {args.max_model_len}')
                    req_max_model_len = args.max_model_len


                                
                
                
                log_format = " >>>>>>>>>>>>>>>> {}: {}"
                print(log_format.format("req_type", req_data["req_type"]))
                print(log_format.format("req_model", "Qwen/Qwen2.5-1.5B-Instruct"))
                print(log_format.format("req_tensor_parallel_size", "1"))
                print(log_format.format("req_gpu_memory_utilization", "0.87"))
                print(log_format.format("req_max_model_len", "2048"))
                print(log_format.format("req_cpu_offload_gb", "0"))
                print(log_format.format("req_enforce_eager", "True"))
                print(log_format.format("req_enable_prefix_caching", "True"))
                print(log_format.format("req_dtype", "auto"))
                print(log_format.format("req_torch_dtype", "auto"))
                print(log_format.format("req_kv_cache_dtype", "auto"))
                print(log_format.format("req_swap_space", "4"))
                print(log_format.format("req_enable_chunked_prefill", "True"))
                print(log_format.format("req_trust_remote_code", "True"))
                print(log_format.format("req_model_storage", "/models"))
                print(log_format.format("req_model_path", "models/facebook/opt-125m"))

                log_format = "{} >>>>>>>>>>>>>>>> {}: {}"
                logging.info("")
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_type", req_data["req_type"]))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_model", "Qwen/Qwen2.5-1.5B-Instruct"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_tensor_parallel_size", "1"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_gpu_memory_utilization", "0.87"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_max_model_len", "2048"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_cpu_offload_gb", "0"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_enforce_eager", "True"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_enable_prefix_caching", "True"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_dtype", "auto"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_torch_dtype", "auto"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_kv_cache_dtype", "auto"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_swap_space", "4"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_enable_chunked_prefill", "True"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_trust_remote_code", "True"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_model_storage", "/models"))
                logging.info(log_format.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "req_model_path", "models/facebook/opt-125m"))
                logging.info("")

                
                
                logging.info(f' @@@ searching {req_model_storage} ...')
                models_found = []
                try:                   
                    if os.path.isdir(req_model_storage):
                        logging.info(f' @@@ found model storage path! {req_model_storage}')
                        logging.info(f' @@@ getting folder elements ...')                        
                        for m_entry in os.listdir(req_model_storage):
                            m_path = os.path.join(req_model_storage, m_entry)
                            if os.path.isdir(m_path):
                                for item_sub in os.listdir(m_path):
                                    sub_item_path = os.path.join(m_path, item_sub)
                                    models_found.append(sub_item_path)        
                        print(f' @@@ found models ({len(models_found)}): {models_found}')
                    else:
                        logging.info(f' @@@ ERR model path not found! {req_model_storage}')
                except Exception as e:
                    logging.info(f' @@@ ERR getting models in {req_model_storage}: {e}')

                
                logging.info(f' @@@ does requested model path match downloaded?')
                model_path = req_model
                if req_model_path in models_found:
                    print(f' @@@ FOUND MODELS ALREADY!!! {req_model} ist in {models_found}')
                    model_path = req_model_path
                else:
                    print(f' @@@ NUH UH DIDNT FIND MODEL YET!! {req_model} ist NAWT in {models_found}')
                
                # free_memory = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                # logging.info(f' @@@ Available GPU memory: {free_memory:.2f} GB')

                # model_pretrained = AutoModelForCausalLM.from_pretrained(req_model, torch_dtype="auto")
                # model_size = sum(p.numel() * p.element_size() for p in model_pretrained.parameters()) / (1024 ** 3)  # Size in GB
                # logging.info(f' @@@ Model size: {model_size:.2f} GB')
                
                logging.info(f' @@@ Using model path: {model_path}')

                llm_instance = LLM(
                    model=model_path,
                    tensor_parallel_size=req_tensor_parallel_size,
                    gpu_memory_utilization=req_gpu_memory_utilization,
                    max_model_len=req_max_model_len,
                    cpu_offload_gb=req_cpu_offload_gb,
                    enforce_eager=req_enforce_eager,
                    enable_prefix_caching=req_enable_prefix_caching,
                    dtype=req_dtype,
                    kv_cache_dtype=req_kv_cache_dtype,
                    swap_space=req_swap_space,
                    enable_chunked_prefill=req_enable_chunked_prefill,
                    trust_remote_code=req_trust_remote_code
                )
                logging.info(f' @@@ Model loaded successfully')
                return JSONResponse({"result_status": 200, "result_data": "Model loaded successfully"})

            except Exception as e:
                logging.error(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return JSONResponse({"result_status": 404, "result_data": str(e)})

        elif req_data["req_type"] == "generate":
            try:
                if llm_instance is None:
                    raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")

                prompt = req_data.get("prompt", "Follow the")
                temperature = req_data.get("temperature", 0.8)
                top_p = req_data.get("top_p", 0.95)
                max_tokens = req_data.get("max_tokens", 100)
                # print('')
                # log_format = " >>>>>>>>>>>>>>>> {}: {}"
                # print(log_format.format("prompt", prompt))
                # print(log_format.format("temperature", temperature))
                # print(log_format.format("top_p", top_p))
                # print(log_format.format("max_tokens", max_tokens))
                # print('')
                
                log_format = " >>>>>>>>>>>>>>>> {}: {}"
                logging.info("")
                logging.info(log_format.format("prompt", prompt))
                logging.info(log_format.format("temperature", temperature))
                logging.info(log_format.format("top_p", top_p))
                logging.info(log_format.format("max_tokens", max_tokens))
                logging.info("")
                
                
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )

                logging.info(f'Generating text for prompt: {prompt}')
                outputs = llm_instance.generate([prompt], sampling_params)
                generated_text = outputs[0].outputs[0].text
                logging.info(log_format.format("generated_text", generated_text[1:]))
            
                return JSONResponse({"result_status": 200, "result_data": generated_text[1:]})

            except Exception as e:
                logging.error(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return JSONResponse({"result_status": 500, "result_data": str(e)})

        else:
            raise HTTPException(status_code=400, detail="Invalid request type.")

    except Exception as e:
        logging.error(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": str(e)})

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=1370, required=True, help="Port to run the application on.")
    
    parser.add_argument("--model", type=str, help="ID of the model.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Amout of tensors.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.91, help="Max GPU memory.")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Max model length.")
    args = parser.parse_args()
    if args.model:
        print(f' @@@ args.model: {args.model}')
    if args.tensor_parallel_size:
        print(f' @@@ args.tensor_parallel_size: {args.tensor_parallel_size}')
    if args.gpu_memory_utilization:
        print(f' @@@ args.gpu_memory_utilization: {args.gpu_memory_utilization}')
    if args.max_model_len:
        print(f' @@@ args.max_model_len: {args.max_model_len}')
    if args.port:
        print(f' @@@ args.port: {args.port}')
    port = args.port
    print(f' @@@ starting unicorn on port: {port}')
    uvicorn.run(app, host="0.0.0.0", port=port)
    

