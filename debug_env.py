import sys, subprocess, os
import tensorflow as tf

print("Python:", sys.version.replace("\n"," "))
print("TensorFlow:", tf.__version__)

try:
    import av
    print("PyAV:", av.__version__)
except Exception as e:
    print("PyAV import failed:", e)

# Check ffmpeg CLI presence
def which(cmd):
    from shutil import which as _which
    return _which(cmd)

print("ffmpeg CLI:", which("ffmpeg"))
print("ldd av:", None)
# If av import works, show av build info (might crash if broken)
try:
    import av
    print("av.formats:", list(av.formats.audio.keys())[:5] if hasattr(av, "formats") else "no formats")
except Exception as e:
    print("av info error:", e)
