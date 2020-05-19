#!/bin/bash

# Copies (rsyncs) wkw data from remote to local
#
# Usage: 
# >>> ./rsync_wkw.sh uname@host:/remote/path/to/dataset/ /local/path/to/dataset/
#
# Example:
# >>> ./rsync_wkw.sh gabaExt:/u/flod/data/scMS109_1to7199_v01_subset/ ../.data/
#
rsync -chavzP --sparse --stats $1 $2 
