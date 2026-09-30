import os


def _fmt(value):
    f = float(value)
    return str(int(f)) if f == int(f) else str(f)


def descriptor(config, attack):
    options = config["finetune"]
    parts = []
    if attack == "badnets":
        parts.append("n" + str(int(options["num_poison_badnets"])))
    elif attack == "htba":
        parts.append("n" + str(int(options.get("num_poison_htba", 0))))
        parts.append("eps" + str(int(options["eps"])))
    else:
        # a new attack needs its own poison-count part, or runs collide on disk
        raise ValueError("descriptor: unknown attack " + attack)
    if attack == "htba":
        htba = config["htba_poison"]
        parts.append("it" + str(int(htba["num_iter"])))
    decoy = (options.get("tal_decoy", "") or "").strip()
    if attack != "htba" or decoy:
        parts.append("tal" + _fmt(options["tal_weight"]))
    if decoy:
        parts.append("dec" + decoy.replace(",", "-"))
        parts.append("pois" if options.getboolean("tal_poison_only", fallback=True)
                     else "allimg")
    # only shows up when tal_layers is overridden; the auto default is implied by uf
    tal_l = (options.get("tal_layers", "") or "").strip()
    if tal_l:
        parts.append("tL" + tal_l.replace(",", "-"))
    unfreeze = int(options.get("unfreeze_blocks", 0))
    if unfreeze:
        parts.append("uf" + str(unfreeze))
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


def filelist(paths, name):
    """Per-run filelist. Sharing one copy across runs makes concurrent grids race,
    and leaves test_time_defense reading whatever run trained last."""
    return os.path.join(paths["run_dir"], name + "_filelist.txt")


def make_dirs(*paths):
    for p in paths:
        if p and not os.path.exists(p):
            os.makedirs(p)
