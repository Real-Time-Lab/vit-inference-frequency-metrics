import subprocess
import time
import shutil

# Cannot serve multiple models at a time due to significant resource consumption
# models = ["vit_b_16","vit_b_32","vit_l_16"]
models = ["vit_b_32"]
# gpu_frequencies = [1710,1695,1680,1665,1650,1635,1620,1605,1590,1575,1560,1545,1530,1515,1500,1485,1470,1455,1440,1425,1410,1395,1380,1365,1350,1335,1320,1305,1290,1275,1260,1245,1230,1215,1200,1185,1170,1155,1140,1125,1110,1095,1080,1065,1050,1035,1020,1005,990,975,960,945,930,915,900,885,870,855,840,825,810,795,780,765,750,735,720,705,690,675,660,645,630,615,600,585,570,555,540,525,510,495,480,465,450,435,420,405,390,375,360,345,330,315,300,285,270]
gpu_frequencies = [270,285,300,315,330,345,360,375,390,405,420,435,450,465,480,495,510,525,540,555,570,585,600,615,630,645,660,675,690,705,720,735,750,765,780,795,810,825,840,855,870,885,900,915,930,945,960,975,990,1005,1020,1035,1050,1065,1080,1095,1110,1125,1140,1155,1170,1185,1200,1215,1230,1245,1260,1275,1290,1305,1320,1335,1350,1365,1380,1395,1410,1425,1440,1455,1470,1485,1500,1515,1530,1545,1560,1575,1590,1605,1620,1635,1650,1665,1680,1695,1710]
# gpu_frequencies = [1710,270]
lambda_values = [820,700,580,460,380,300,220,140,100,60,20]
# lambda_values = [580]
images_size = 8000
# images_size = 100
sleep_time = 10
# sleep_time = 1

inferences_file_path = 'inferences.py'
gpu_id = 0

def set_gpu_max_frequency(max_freq):
    """ Set the maximum GPU frequency. """
    command = ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-lgc', f'0,{max_freq}']
    subprocess.run(command, check=True)
    print(f"Maximum GPU frequency set to {max_freq} MHz.")

def run_inferences_script_with_model_lambda_and_freq(model, lambda_value, gpu_freq):
    subprocess.run(['python', inferences_file_path, '--model', model, '--lambda', str(lambda_value), '--gpu-freq', str(gpu_freq), '--images-size', str(images_size)], check=True)

def delete_directory(directory_path='../logs/'):
    shutil.rmtree(directory_path)

for model in models:
    print()
    print(f"*************************************** Running with Model = {model} ***************************************\n")
    time.sleep(sleep_time)
    for gpu_freq in gpu_frequencies:
        # delete_directory()
        print(f"\n****** Running with Frequency = {gpu_freq} ******\n")
        set_gpu_max_frequency(gpu_freq)
        print()
        time.sleep(sleep_time)
        for lambda_value in lambda_values:
            time.sleep(sleep_time)
            print(f"Running inferences.py with LAMBDA = {lambda_value}")
            run_inferences_script_with_model_lambda_and_freq(model, lambda_value, gpu_freq)
        
        print()
        subprocess.run(['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rgc'], check=True)