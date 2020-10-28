import os
import time
import logging
import subprocess
import psutil
from config import AlphaZeroConfig
import sys

TIMEOUT_MIN = 10
MAX_RECDEPTH = 3


def tryGameGen(recDepth):
    logging.info("Starting iteration num: %d" % iterations)
    logging.info("Trying to call GameGenerator..")
    process = subprocess.Popen(['python3', 'game_generator.py'], shell=False)
    p_name = psutil.Process(process.pid).name()
    logging.info("Process name: %s" % p_name)

    timeout = 0
    while True:
        if timeout == TIMEOUT_MIN:
            logging.info("Still couldn't find child processes after %d minutes.. breaking.." % timeout)
            break
        try:
            num_gg_processes = len(subprocess.check_output(["pidof", p_name]).split())-6
        except Exception as exc:
            logging.info("Error while trying to find processes with name %s" % p_name)
            logging.info(exc)

            if TIMEOUT_MIN / 2 < timeout:
                logging.info("Stop seaching after %d minutes" % timeout)
                break
        else:
            logging.info("Found %d processes named ´%s´ after %d minutes" % (num_gg_processes, p_name, timeout))

            if num_gg_processes > config.max_processes:
                logging.info("All processes of GameGenerator started working.. breaking..")
                break

        logging.info("Wait for Selfplay Child Processes to start..")
        time.sleep(60)
        timeout += 1

    children = []
    if psutil.pid_exists(process.pid):
        children = psutil.Process(process.pid).children()
        logging.info("Found the children. %d has %d Child Processes." % (process.pid, len(children)))
    else:
        logging.info("Parent Process terminated before we could save children")

        orphans = len(subprocess.check_output(["pidof", p_name]).split())

        if orphans > config.max_processes:
            logging.info("Yeah we fucked up. Couldn't catch the bug.")
            logging.info("Still %d orphan processes.." % orphans)
            logging.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            logging.info("EXIT EXIT EXIT EXIT EXIT EXIT EXIT EXIT EXI")
            logging.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            sys.exit()

    code = process.wait()

    if code != 0:
        logging.info("Failed to call GameGenerator with code: %d" % code)
        try:
            for child in children:
                child.kill()
            logging.info(len("Found %d pids of GameGenerator.. just killed them.." % len(children)))
        
        except Exception as exc:
            logging.info("Couldn't find any pids of GameGenerator")
            logging.info(exc)
        
        if psutil.pid_exists(process.pid):
            process.kill()
            logging.info("Killed Parent Process..")
        
        if recDepth != MAX_RECDEPTH:
            tryGameGen(recDepth + 1)
            logging.info("Tried to call GameGenerator %d times on iteration %d" % (recDepth+1, iterations))
    else:
        logging.info("Called GameGenerator successfully")

if __name__ == '__main__':
    iterations = 1
    while True:
        logging.basicConfig(filename="wrapper.log", level=logging.DEBUG, format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')

        config = AlphaZeroConfig()

        tryGameGen(0)
        time.sleep(120)
        os.system("python3 trainer.py")
        iterations += 1

