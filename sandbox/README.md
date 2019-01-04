# Moonlight Sandbox

This directory has the symlinks necessary to import Moonlight after building.
You can either add the directory to your PYTHONPATH, or run Python from this
directory.

    git clone https://github.com/tensorflow/moonlight
    cd moonlight
    # You may want to run this inside a virtualenv.
    pip install -r requirements.txt
    # Builds dependencies and sets up the symlinks that we point to.
    bazel build moonlight:omr

    cd sandbox
    python
    >>> from moonlight import engine
