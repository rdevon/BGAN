# Control for Cluster
from __future__ import print_function
import subprocess
import itertools
import time
import os
import glob

pf_err = open("./control_log_err.txt",'w')
pf_out = open("./control_log_out.txt",'w')

def main(start_count=0,debug=False):
    process_count = 0
    config_files = glob.glob("/home/apjacob/config_files/celeba/*") ##Add location of all config files
    for config_file in config_files:
        filename = os.path.basename(config_file)
        args = "--gpu --queue=gpu_1 --env=THEANO_FLAGS='device=gpu, floatX=float32' --project=jvb-000-ag"
        cmd = "jobdispatch " + args + " python -u celeba.py -o output/celeba_output/" + filename + " -S data/celeba_64.hdf5 -c " + config_file
        print("Launching command: ", cmd)
        print ("The Controller program is now launching jobs: %d"%process_count,file=pf_out)
        if process_count < start_count:
            continue
        try:
            subprocess.call(cmd,shell=True,stderr=pf_err, stdout=pf_out)
        except:
            print("Failed. >_<. Current process count is: %d"%process_count, file=pf_out)
        else:
            print("%d subprocesses have finished, preparing next..."%process_count,file=pf_out)
            process_count += 1
        finally:
            time.sleep(2)
        if debug and process_count >= 500:
            return
    pf_err.close()
    pf_out.close()

if __name__ == '__main__':
    main()
