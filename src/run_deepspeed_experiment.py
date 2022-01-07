import subprocess
import os
os.environ['TRANSFORMERS_CACHE'] = "/brtx/601-nvme1/estengel/.cache"
config=os.getenv("CONFIG", None) 
p = subprocess.Popen(["python","-u", "run_experiment.py", "--cfg", config], stdout=subprocess.PIPE, stderr = subprocess.PIPE)
out, errs = p.communicate()

print(out.decode('utf-8')) 
print(errs.decode('utf-8')) 
