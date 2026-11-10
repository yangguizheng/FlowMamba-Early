"""Train one job given as a JSON string on the command line; worker.py calls this file.

    python src/train_job.py '{"model": "FME", "variant": "orig", "dataset": "iscxvpn", "seed": 20260924,
                              "n": null, "budget": "orig", "mask": null, "epoch_preds": true}'
"""
import json
import sys

import experiment

if __name__ == "__main__":
    experiment.train(json.loads(sys.argv[1]))
