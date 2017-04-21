import sys
import threading

class Sender(object):

    def __init__(self):
        pass

    def send_update(self, frame, filename):
        print("BLENDER FRAME: %d" % frame)
        print("BLENDER UPDATE: %s" % filename)

    def send_finish(self):
        print("BLENDER FINISH")

class Receiver(object):

    def __init__(self, callback_dict):
        self.callbacks = callback_dict

    def receive(self):
        self.thread = threading.Thread(target=self._receive)
        self.thread.start()

    def stop(self):
        pass

    def _receive(self):
        for line in map(str.strip, sys.stdin):
            if line.startswith("BLENDER"):
                line = line[len("BLENDER "):]
                action = line.split(":")[0].strip().lower()
                if action in self.callbacks:
                    print("BLENDER ACTION --> ", line)
                    self.callbacks[action](line[len(action)+1:].strip())
                if action is "finish":
                    return
