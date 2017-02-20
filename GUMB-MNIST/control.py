# Control for Cluster
from __future__ import print_function
import subprocess
import itertools
import time

nLayerG = [3, 4]
nLayerD = [3, 4]
sizeG = [800,1600,3200]
sizeD = [256,512]
dropoutD = [False]
optimG = ['sgd', 'adam']  
optimD = ['sgd', 'adam']
learning_rate = [1e-3,1e-4]
encoder = [0,1]

pf = open("./control_log.txt",'w')
def main(start_count=0,debug=False):
    process_count = 0
    for nG, nD, sG, sD, dD, oG, oD,lr,enc in itertools.product(nLayerG, nLayerD,
                              sizeG, sizeD, dropoutD, optimG, optimD,learning_rate, encoder):

        parameters = "%d %d %d %d %i %s %s %.4f %d" % (nG, nD, sG, sD, dD, oG, oD, lr, enc)
        args = "--gpu --queue=k20 --duree=10:00 --env=THEANO_FLAGS='device=gpu'      --project=jvb-000-aa"
        prefix = "%d_%d_%d_%d_%i_%s_%s_%.4f_%d" % (nG, nD, sG, sD, dD, oG, oD, lr, enc)
        cmd = "jobdispatch " + args + " python -u mnist_grid.py " + parameters
        print ("The Controller program is now launching jobs: %s"%prefix,file=pf)
        if process_count < start_count:
            continue
        try:
            subprocess.call(cmd,shell=True,stderr=pf)
        except:
            print("Failed. >_<. Current process count is: %d"%process_count, file=pf)
        else:
            print("%d subprocesses have finished, preparing next..."%process_count,file=pf)
            process_count += 1
        finally:
            time.sleep(0.9)
        if debug and process_count >= 500:
            return
    pf.close()

if __name__ == '__main__':
    main()
    
