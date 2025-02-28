import sys


class Logger(object):

    def __init__(self, log_filedir='.'):
        # self.terminal = sys.stdout
        self.log = open(log_filedir + "/logfile.log", "w+")

    def write(self, message):
        # self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
