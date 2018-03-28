# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash
# Extracts PNGs from the IMSLP backup. The output PNGs are suitable for training
# and running OMR. See IMSLP for the latest details on the backup:
# http://imslp.org/wiki/IMSLP:Backups

INPUT_DIR="$1"
OUTPUT_DIR="$2"

if ! [[ -d "$INPUT_DIR" ]]; then
  echo "First argument must be a directory" > /dev/stderr
  exit -1
fi

if ! [[ -d "$OUTPUT_DIR" ]]; then
  mkdir -v "$OUTPUT_DIR"
fi

if ! [[ -x "$(which pdfimages)" ]]; then
  echo "pdfimages is required. Please install poppler-utils." > /dev/stderr
  exit -1
fi

if ! [[ -x "$(which parallel)" ]]; then
  echo "GNU parallel is required. Please install parallel." > /dev/stderr
  exit -1
fi

if ! [[ -x "$(which convert)" ]]; then
  echo "'convert' is required. Please install imagemagick." > /dev/stderr
  exit -1
fi

# For each pdf...
find "$INPUT_DIR" -name "IMSLP*.pdf" | \
    # Convert to "IMSLPnnnnn-nnn.ppm" or ".pgm" images in $OUTPUT_DIR.
    perl -ne 'chomp; /(IMSLP[0-9]+)/; print qq(pdfimages "$_" "'"$OUTPUT_DIR"'"/$1\n)' | \
    # Run all emitted commands in parallel.
    parallel -v

# Convert extracted "pbm", "pgm", and "ppm" images to PNG.
(for file in "$OUTPUT_DIR"/*.p[bgp]m; do
  echo "convert '$file' '${file%.*}.png' && rm -v '$file'"
done) | parallel -v
