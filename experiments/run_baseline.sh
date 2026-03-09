#!/bin/bash

#execute baseline sim
echo "initiating fudge-fl baseline experiments..."

#model retrains from scratch --> "perfect unlearning". Control method
python src/server.py --unlearning_method retraining

#sisa unlearning
python src/server.py --unlearning_method sisa

#pga method
python src/server.py --unlearning_method pga

echo "Auditing complete. Check output logs for privacy, utility, and security scores."