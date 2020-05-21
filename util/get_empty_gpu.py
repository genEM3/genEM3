import numpy as np
import os
import socket
from subprocess import Popen, PIPE
import time

def get_empty_gpu(random_sleep=False):
    '''
    If this function is called on GABA (includes check for hostname, so a call of this function on your local machine just does nothing)
    before a tensorflow session object is created,
    it checks via the nvidia-smi bash program whether any GPUs are already occupied and assigns a (random – if there is a choice) free GPU.
    If multiple tensorflow jobs are submitted via SGE at the same time to the same compute node, it can happen, that a call to nvidia-smi returns
    "No processes running" for both jobs and this might result, that they both end up running on the same GPU and blocking each other.
    With random_sleep=True both processes will sleep for a random amount of time (between 0 and 180 seconds),
    such that one of them will very likely start a bit earlier and therefore the other process will see the first one via the nvidia-smi call and will therefore get
    the empty GPU.
    '''
    if socket.gethostname()[0:4] == 'gaba':

        # If random_sleep==True wait for some random number of seconds to ensure,
        # that parallel submitted processes to the same node do not accidentally use the same GPU
        # For an interactive session in which you don’t want to wait, it might be useful to set random_sleep=False
        if random_sleep:
            rand_wait = np.random.rand()*180
            time.sleep(rand_wait)

        num_gpus = 2

        p = Popen(['nvidia-smi'], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
        stdout, stderr = p.communicate()

        if b'No running processes found' in stdout:
            random_device = np.random.randint(low=0, high=num_gpus)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(random_device)
            print("No running processes on GPUs")
            print("Assigned GPU %d" % random_device)
        else:
            # stdout.replace(b' ', b'')
            # stdout.replace(b'\t', b'')
            stdout.replace(b'\t', b' ')
            stdout = stdout.split(b'Process name')[-1].split(b'\n')[2:-2]
            print("Found %d process(es) running on GPU" % len(stdout))
            gpu_ids = []
            pids = []
            pnames = []
            types = []
            mem = []
            for line in stdout:
                splitted_line = line.split(b' ')
                parsed_line = [s for s in splitted_line if s not in [b'', b' ', b'-', b'+', b'|']]
                if len(parsed_line) != 5:
                    print('Problem with parsing nvidia-smi output')
                    raise ValueError(
                        'Found more than 4 entities in nvidia-smi output lines of processes.'
                        'Maybe nvidia-smi was updated and the parsing procedure does not work anymore.')
                gpu_ids.append(int(parsed_line[0]))
                pids.append(int(parsed_line[1]))
                types.append(parsed_line[2])
                pnames.append(parsed_line[3])
                mem.append(parsed_line[4])
            print("Found the following processes: ")
            format_string = 'GPU ID: %d, PID: %d, Type: %s, Pname: %s, Mem: %s'
            for k in range(len(gpu_ids)):
                print(format_string % (gpu_ids[k], pids[k], types[k], pnames[k], mem[k]))
            gpu_ids_unique = []
            for id in gpu_ids:
                if id not in gpu_ids_unique:
                    gpu_ids_unique.append(id)

            if len(gpu_ids_unique) >= num_gpus:
                print('All GPUs occupied')
                raise EnvironmentError('Occupied GPUs: ' + str(gpu_ids_unique) + '. Code assumes ' +
                                       str(num_gpus) + ' # of GPUs. It seems like they are all occupied! ' +
                                       'Maybe check implementation of get_empty_gpu() function.')
            else:
                empty_gpu_ids = [id for id in range(num_gpus) if id not in gpu_ids_unique]
                rand_idx = np.random.randint(low=0, high=len(empty_gpu_ids))
                random_device = empty_gpu_ids[rand_idx]
                os.environ['CUDA_VISIBLE_DEVICES'] = str(random_device)
                print("Empty GPUs: " + str(empty_gpu_ids))
                print("Assigned GPU %d" % random_device)
