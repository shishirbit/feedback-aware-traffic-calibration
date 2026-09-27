"""External SUMO binary availability checks."""
from functools import lru_cache
import shutil
import subprocess


@lru_cache(maxsize=1)
def sumo_available():
    if not shutil.which("netconvert") or not shutil.which("sumo"):
        return False,"SUMO executables are not on PATH"
    for executable in ("netconvert","sumo"):
        result=subprocess.run([executable,"--version"],capture_output=True,text=True)
        if result.returncode:
            unsigned=result.returncode & 0xffffffff
            return False,f"{executable} launch failed with Windows status 0x{unsigned:08x}"
    return True,"available"
