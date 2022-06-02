<img align="center" width="400" height="94,358" src="https://user-images.githubusercontent.com/34600369/40580500-74088e4a-6137-11e8-9705-ecac1499b1ce.png">

# Moonlight Optical Music Recognition (OMR) [![Build Status](https://travis-ci.org/tensorflow/moonlight.svg?branch=master)](https://travis-ci.org/tensorflow/moonlight)

An experimental [optical music
recognition](https://en.wikipedia.org/wiki/Optical_music_recognition) engine.

Moonlight reads PNG image(s) containing sheet music and outputs
[MusicXML](https://www.musicxml.com/) or a
[NoteSequence message](https://github.com/tensorflow/magenta/blob/master/magenta/protobuf/music.proto).
MusicXML is a standard sheet music interchange format, and `NoteSequence` is
used by [Magenta](http://magenta.tensorflow.org) for training generative music
models.

Moonlight is not an officially supported Google product.

### Command-Line Usage

    git clone https://github.com/tensorflow/moonlight
    cd moonlight
    # You may want to run this inside a virtualenv.
    pip install -r requirements.txt
    # Build the OMR command-line tool.
    bazel build moonlight:omr
    # Prints a Score message.
    bazel-bin/moonlight/omr moonlight/testdata/IMSLP00747-000.png
    # Scans several pages and prints a NoteSequence message.
    bazel-bin/moonlight/omr --output_type=NoteSequence IMSLP00001-*.png
    # Writes MusicXML to ~/mozart.xml.
    bazel-bin/moonlight/omr --output_type=MusicXML --output=$HOME/mozart.xml \
        corpus/56/IMSLP56442-*.png

The `omr` CLI will print a [`Score`](moonlight/protobuf/musicscore.proto)
message by default, or [MusicXML](https://www.musicxml.com/) or a
`NoteSequence` message if specified.

Moonlight is intended to be run in bulk, and will not offer a full UI for
correcting the score. The main entry point will be an Apache Beam pipeline that
processes an entire corpus of images.

There is no release yet, and Moonlight is not ready for end users. To run
interactively or import the module, you can use the [sandbox
directory](sandbox/README.md). Moonlight will be used offline for digitizing
a scanned corpus (it can be installed on all Cloud Compute platforms, and OS
compatibility is not a priority).

### Dependencies

* Linux
  - Note: Our Google dep versions are fragile, and updating them or updating other OS may break directory structure in fragile ways.
* [Protobuf 3.6.1](https://pypi.org/project/protobuf/3.6.1/)
* [Bazel 0.20.0](https://github.com/bazelbuild/bazel/releases/tag/0.20.0). We
  encountered some errors using Bazel 0.21.0 to build Protobuf 3.6.1, which is
  the latest Protobuf release at the time of writing.
* Python version supported by TensorFlow (Python 3.5-3.7)
* Python dependencies specified in the [requirements](requirements.txt).

### Resources

[Forum](https://groups.google.com/forum/#!forum/moonlight-omr)
