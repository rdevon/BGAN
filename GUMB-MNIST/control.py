# Control for Cluster
from __future__ import print_function
import subprocess
import itertools
import time

gumbel_hard_arr = [True, False]
optimGD_arr = ['adam', 'rmsprop', 'sgd']
learning_rate_arr = [1e-2, 1e-3, 1e-4, 1e-5]
anneal_rate_arr = [0.01, 0.001, 0.0001]
anneal_interval_arr = [200, 300, 500]

pf_err = open("./control_log_err.txt",'w')
pf_out = open("./control_log_out.txt",'w')

def main(start_count=0,debug=False):
    process_count = 0
    for gumbel_hard, optimGD, lr, anneal_rate, anneal_interval in \
            itertools.product(gumbel_hard_arr, optimGD_arr, learning_rate_arr, anneal_rate_arr, anneal_interval_arr):

        parameters = "%d %s %.6f %.5f %d" % (gumbel_hard, optimGD, lr, anneal_rate, anneal_interval)
        args = "--gpu --queue=gpu_1 --duree=30:00 --env=THEANO_FLAGS='device=gpu, floatX=float32' --project=jvb-000-ag"
        prefix = "{}_{}_{}_{}_{}".format(gumbel_hard,
                                         optimGD,
                                         lr,
                                         anneal_rate,
                                         anneal_interval)
        cmd = "jobdispatch " + args + " python -u disc_mnist.py " + parameters
        print("Launching command: ", cmd)
        print ("The Controller program is now launching jobs: %s"%prefix,file=pf_out)
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
    
