import aiohttp
import asyncio
import os
import json
import subprocess
from time import time
import numpy as np
import pandas as pd

TORCHSERVE_URL = 'http://localhost:8080/predictions/vit_b_16'
IMAGE_DIR = '/media/Extra/ImageNet-2012-Dataset/imagenet/val/'
MAX_IMAGES = 30000
current_lambda = 32 # Some value initially
LAMBDA_SCHEDULE = [(20, 200), (60, 50), (700, 10), (20, 200)]
FREQUENCY_SCHEDULE = [(210, 200), (705, 50), (1710, 100), (210, None)]
# MAX_IMAGES = 200
# current_lambda = 32 # Some value initially
# LAMBDA_SCHEDULE = [(20, 10)]
# FREQUENCY_SCHEDULE = [(210, 5), (705, None)]

def process_gpu_metrics(gpu_df):
    avg_temperature = gpu_df['Temperature (C)'].mean()
    avg_power_draw = gpu_df['Power Draw (W)'].mean()
    avg_gpu_util = gpu_df['GPU Utilization (%)'].mean()
    avg_mem_util = gpu_df['Memory Utilization (%)'].mean()

    return {
        'Avg Temperature (C)': round(avg_temperature, 2),
        'Avg Power Draw (W)': round(avg_power_draw, 2),
        'Avg GPU Utilization (%)': round(avg_gpu_util, 2),
        'Avg Memory Utilization (%)': round(avg_mem_util, 2)
    }

def get_image_paths(img_dir, num_images=MAX_IMAGES):
    image_paths = []
    for subdir, _, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith(('jpe', 'jpeg', 'png')):
                image_paths.append(os.path.join(subdir, file))
            if len(image_paths) >= num_images:
                return image_paths
    return image_paths

def poisson_process(lambda_rate):
    return -np.log(1.0 - np.random.rand()) / lambda_rate

async def update_gpu_frequency(is_freq_modulation_enabled):
    if is_freq_modulation_enabled is False:
        return
    for frequency, duration in FREQUENCY_SCHEDULE:
        subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-lgc', f'0,{frequency}'], check=True)
        if duration is None:
            return
        await asyncio.sleep(duration)

# async def update_gpu_frequency():
#     for frequency, duration in FREQUENCY_SCHEDULE:
#         subprocess.run(['./set_gpu_frequency.exp', 'Temporary@24', "set", str(frequency)], check=True)
#         if duration is None:
#             return
#         await asyncio.sleep(duration)

async def update_lambda(stop_event):
    for lambda_value, duration in LAMBDA_SCHEDULE:
        global current_lambda
        current_lambda = lambda_value
        await asyncio.sleep(duration)
    stop_event.set()

async def send_warmup_requests(session):
    warmup_image = '/media/Extra/ImageNet-2012-Dataset/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG'
    for _ in range(5):
        try:
            await session.post(
                TORCHSERVE_URL,
                json={"path": warmup_image, "request_id": "warmup"},
                headers={"Content-Type": "application/json"}
            )
        except Exception as e:
            print(f'Warmup error occurred: {e}')

async def send_request(session, image_path, unique_id, timestamp_A, lambda_value):
    try:
        async with session.post(
            TORCHSERVE_URL,
            json={"path": image_path, "request_id": unique_id},
            headers={"Content-Type": "application/json"}
        ) as response:
            response_txt = await response.text()
            response_json = json.loads(response_txt)
            timestamp_C = time()

            if response.status == 200:
                timestamp_B = response_json['timestamp_B']
                queuing_delay = (timestamp_B - timestamp_A)
                processing_time = response_json['processing_time']
                job_completion_time = (timestamp_C - timestamp_A)

                return {
                    'id': unique_id,
                    'lambda_value': lambda_value,
                    'qd': round(queuing_delay,3),
                    'pt': round(processing_time,3),
                    'jct': round(job_completion_time,3)
                }
            else:
                print(f"Request failed: {response.status}: {await response.text()}")
                return None

    except Exception as e:
        print(f'An error occurred: {e}')
        return None

async def send_requests(image_paths, stop_event):
    async with aiohttp.ClientSession() as session:
        print('Sending warmup requests')
        await send_warmup_requests(session)

        print('Sending real requests')
        tasks = []
        for idx, image_path in enumerate(image_paths):
            unique_id = str(idx)
            timestamp_A = time()
            lambda_value = current_lambda
            task = asyncio.create_task(send_request(session, image_path, unique_id, timestamp_A, lambda_value))
            tasks.append(task)
            await asyncio.sleep(poisson_process(lambda_value))
            if stop_event.is_set():
                break

        results = await asyncio.gather(*tasks)
        results = [result for result in results if result is not None]
        return results

async def collect_gpu_metrics(interval, stop_event):
    await asyncio.sleep(0.5) #Allowing GPU to warmup before collecting the metrics
    metrics = []
    while not stop_event.is_set():
        lambda_value = current_lambda
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,timestamp,power.draw,clocks.gr,clocks.mem,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        metrics_line = result.stdout.strip().split(', ')
        metrics_line.append(str(lambda_value))
        metrics.append(metrics_line)
        await asyncio.sleep(interval)
    pd.DataFrame(metrics, columns=["Index", "Timestamp", "Power Draw (W)", "GPU Clock (MHz)", "Memory Clock (MHz)", "Temperature (C)", "GPU Utilization (%)", "Memory Utilization (%)", "Total Memory (MiB)", "Used Memory (MiB)", "Lambda"]).to_csv(f'gpu_metrics.csv', index=False)

async def main():
    is_freq_modulation_enabled = True
    image_paths = get_image_paths(IMAGE_DIR)
    # print(f"Total Requests: {len(image_paths)}")

    stop_event = asyncio.Event()

    start_time = time()

    gpu_frequency_task = asyncio.create_task(update_gpu_frequency(is_freq_modulation_enabled))
    update_lambda_task = asyncio.create_task(update_lambda(stop_event))
    gpu_metrics_task = asyncio.create_task(collect_gpu_metrics(0.5, stop_event))

    latency_data = await send_requests(image_paths, stop_event)

    end_time = time()

    total_time = (end_time - start_time)
    
    await update_lambda_task
    await gpu_metrics_task
    await gpu_frequency_task

    # subprocess.run(['./set_gpu_frequency.exp', 'Temporary@24', "reset", '0'], check=True)
    if is_freq_modulation_enabled is True:
        subprocess.run(['sudo', 'nvidia-smi', '-i', '0', '-rgc'], check=True)

    gpu_metrics_df = pd.read_csv(f'gpu_metrics.csv')
    gpu_metrics_summary = process_gpu_metrics(gpu_metrics_df)

    latency_df = pd.DataFrame(latency_data)
    latency_df.to_csv(f'latencies.csv', index=False)

    avg_queuing_delay = round(latency_df['qd'].mean(),3)
    avg_processing_time = round(latency_df['pt'].mean(),3)
    avg_job_completion_time = round(latency_df['jct'].mean(),3)
    total_requests_processed = len(latency_data)
    throughput = total_requests_processed / total_time if total_time > 0 else 0

    print(f"Average Queuing Delay (s): {avg_queuing_delay}")
    print(f"Average Processing Time (s): {avg_processing_time}")
    print(f"Average Job Completion Time (s): {avg_job_completion_time}")

    average_metrics_file = 'overall_avg_metrics.csv'
    summary_df = pd.DataFrame([{
        'Frequency Modulation Enabled': is_freq_modulation_enabled,
        'Average Queuing Delay (s)': avg_queuing_delay,
        'Average Processing Time (s)': avg_processing_time,
        'Average Job Completion Time (s)': avg_job_completion_time,
        'Total Requests Processed': total_requests_processed,
        'Total Time (s)': round(total_time,3),
        'Throughput (requests/sec)': round(throughput,2),
        **gpu_metrics_summary
    }])
    
    if os.path.exists(average_metrics_file):
        summary_df.to_csv(average_metrics_file, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(average_metrics_file, index=False)

    print("Latency Metrics, GPU metrics collection and inferences have completed.")
    print(f"Total time taken (s): {round(total_time,3)}")

if __name__ == '__main__':
    asyncio.run(main())