"""Write the job list of the controlled comparison to OUTPUT_ROOT/experiments/jobs.jsonl.

Seeds 20260921-23 of the main comparison are trained with run_flowmamba_early.py and
run_flowmamba_single_n.py and imported with export_first_seeds.py, so they are not listed.

1. Main comparison, seeds 20260924-30: FME and Single-N at every exit with the common recipe.
2. Step-matched Single-N, seeds 20260921-30, at exits 2, 3 and 5 of ISCXVPN2016, CICIoT2022
   and USTC-TFC2016. On CipherSpectrum and CSTNET-TLS1.3 at least 99% of the training flows
   reach exit 5, so the step-matched budget equals the common one.
3. Eligibility-matched FME, seeds 20260921-25: every exit is trained on the flows that reach
   exit N, with the step-matched budget of Single-N.
4. Retraining with the combined mask on training, validation and test inputs: ten seeds on
   ISCXVPN2016 and CSTNET-TLS1.3, three seeds on the other datasets.
5. Masked step-matched Single-N on ISCXVPN2016, seeds 20260921-30.
6. Long budget, 50 epochs for both approaches: seeds 20260921-25 on ISCXVPN2016 and CSTNET-TLS1.3.

    python src/plan.py
    python src/worker.py w1
"""
import json
import os

import experiment

SEEDS = list(range(20260921, 20260931))
SMALL = ["iscxvpn", "ciciot2022", "ustctfc2016"]
ALL = experiment.DATASETS


def job(model, variant, dataset, seed, n=None, budget="orig", mask=None, epoch_preds=False):
    return {"model": model, "variant": variant, "dataset": dataset, "seed": seed, "n": n,
            "budget": budget, "mask": mask, "epoch_preds": epoch_preds}


def both(variant, dataset, seed, budget="orig", mask=None):
    """FME, with the predictions of every epoch, and Single-N at every exit."""
    jobs = [job("FME", variant, dataset, seed, budget=budget, mask=mask, epoch_preds=True)]
    return jobs + [job("Single", variant, dataset, seed, n, budget=budget, mask=mask) for n in experiment.EXITS]


def main():
    jobs = []
    for seed in SEEDS[3:]:                                    # 1. main comparison
        for d in ALL:
            jobs += both("orig", d, seed)
    for seed in SEEDS:                                        # 2. step-matched Single-N
        for d in SMALL:
            jobs += [job("Single", "stepmatch", d, seed, n, budget="stepmatch") for n in [2, 3, 5]]
    for seed in SEEDS[:5]:                                    # 3. eligibility-matched FME
        for d in SMALL:
            jobs += [job("FME", f"elig{n}", d, seed, n, budget="stepmatch", epoch_preds=True) for n in [2, 3, 5]]
    for d in ALL:                                             # 4. masked retraining
        for seed in (SEEDS if d in ("iscxvpn", "cstnet-tls1.3") else SEEDS[:3]):
            jobs += both("maskall", d, seed, mask="mask_all")
    for seed in SEEDS:                                        # 5. masked step-matched Single-N
        jobs += [job("Single", "maskall_stepmatch", "iscxvpn", seed, n, budget="stepmatch", mask="mask_all")
                 for n in [2, 3, 5]]
    for seed in SEEDS[:5]:                                    # 6. long budget
        for d in ["iscxvpn", "cstnet-tls1.3"]:
            jobs += both("long50", d, seed, budget="long50")
    os.makedirs(experiment.OUT, exist_ok=True)
    with open(f"{experiment.OUT}/jobs.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{len(jobs)} jobs")


if __name__ == "__main__":
    main()
