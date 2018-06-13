import threading
import time

class QsizeDebugger(threading.Thread):
    def __init__(self, name, queue, interval=1, template="Queue %s size is %d"):
        super(QsizeDebugger,self).__init__()
        self.name = name
        self.queue = queue
        self.interval = interval
        self.template = template
        self.stopped = False
        self.setDaemon(True)
    def run(self):
        while not self.stopped:
            print(self.template % (self.name, self.queue.qsize()))
            time.sleep(self.interval)
    def stop(self):
        self.stopped = True
