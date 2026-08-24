import os


def _fmt(value):
    f = float(value)
    return str(int(f)) if f == int(f) else str(f)


def descriptor(config, attack):
    options = config["finetune"]
    parts = []
    if attack == "badnets":
        parts.append("n" + str(int(options["num_poison_badnets"])))
    else:
        parts.append("n" + str(int(options["num_poison_lc"])))
        parts.append("eps" + str(int(options["eps"])))
    if attack == "lc":
        lc = config["lc_poison"]
        parts.append("pgd" + str(int(lc["pgd_steps"])))
        parts.append("alpha" + _fmt(lc["pgd_alpha"]))
    parts.append("tal" + _fmt(options["tal_weight"]))
    parts.append("rand" if options.getboolean("rand_loc") else "fixed")
    if options.getboolean("feature_extract"):
        parts.append("headonly")
    return "_".join(parts)


def for_run(experiment_id, attack, config):
    run = descriptor(config, attack)
    tail = os.path.join(experiment_id, attack, run)
    run_dir = os.path.join("runs", tail)
    return {
        "run": run,
        "run_dir": run_dir,
        "finetune_log": os.path.join(run_dir, "finetune.log"),
        "defense_log": os.path.join(run_dir, "defense.log"),
        "viz_dir": os.path.join(run_dir, "viz"),
        "ckpt_dir": os.path.join("checkpoints", tail),
    }


def make_dirs(*paths):
    for p in paths:
        if p and not os.path.exists(p):
            os.makedirs(p)
