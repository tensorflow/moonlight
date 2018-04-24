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

### Usage

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
