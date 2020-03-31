import subprocess
from hashlib import sha1
import os
import numpy as np
import shutil
import sys

CACHEDIR = sys.path[0] + "/.mumax3_cache"

def clear_mumax3_cache():
    shutil.rmtree(CACHEDIR)

def remove_white_space(string):
    lines = string.split("\n")
    lines = [line.strip() for line in lines if line.strip() != ""]
    string = "\n".join(lines)
    return string

class Mumax3Simulation:
    def __init__(self, script):
        self._script = remove_white_space(script)
        if not os.path.exists(self.outputdir):
            self.run()

    @property
    def script(self):
        return self._script

    @property
    def hash(self):
        return sha1(self._script.encode()).hexdigest()

    @property
    def scriptfile(self):
        return CACHEDIR + "/" + self.hash + ".mx3"

    @property
    def outputdir(self):
        return CACHEDIR + "/" + self.hash + ".out"

    def run(self):
        if not os.path.exists(CACHEDIR):
            os.makedirs(CACHEDIR)
        if os.path.exists(self.outputdir):
            shutil.rmtree(self.outputdir)
        with open(self.scriptfile, 'w') as f:
            f.write(self.script)
        subprocess.run(["mumax3", self.scriptfile])
        subprocess.run(["mumax3-convert", "-numpy",
                        self.outputdir + "/*.ovf"])

    def get_field(self, fieldname):
        filename = self.outputdir + "/" + fieldname + ".npy"
        return np.load(filename)