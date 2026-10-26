import glob
import os
import random

import attacks


# the trigger every run so far used. Naming it unconditionally would rename
# every existing run directory, so only a deviation shows up in the descriptor.
DEFAULT_TRIGGER = 10

DEFAULT_EDGE = 30


def _fmt(value):
    f = float(value)
    return str(int(f)) if f == int(f) else str(f)


def list_dir(experiment_id):
    return os.path.join("ImageNet_data_list", experiment_id)


def class_names(experiment_id, split="train"):
    pattern = os.path.join(list_dir(experiment_id), split, "*.txt")
    return [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(pattern))]


def select_classes(config, names=None):
    experiment_id = config["experiment"]["ID"]
    classes = config["classes"]
    num_source = int(classes["num_source"])
    if config["finetune"].getboolean("random_classes", fallback=False):
        if names is None:
            names = class_names(experiment_id)
        if len(names) < num_source + 1:
            raise ValueError("random_classes needs at least {} classes, found {}".format(
                num_source + 1, len(names)))
        seed = int(config["finetune"].get("seed", 0))
        picked = random.Random(seed).sample(sorted(names), num_source + 1)
        return picked[0], picked[1:]
    with open(classes["source_wnid_list"].format(experiment_id)) as f:
        sources = [s.strip() for s in f if s.strip()]
    target, sources = classes["target_wnid"], sources[:num_source]
    if names is None:
        names = class_names(experiment_id)
    # a target that is not a real class trains a run whose attack can never
    # succeed, and says nothing about it. imagenet30 names its classes rather
    # than using wnids, so a cfg copied between datasets lands here.
    unknown = [c for c in [target] + sources if c not in names]
    if unknown:
        raise ValueError(
            "{}: {} not among the {} classes of {} ({}, ...)".format(
                "source_wnid_list/target_wnid", ", ".join(unknown),
                len(names), experiment_id, ", ".join(sorted(names)[:3])))
    return target, sources


def descriptor(config, attack):
    options = config["finetune"]
    parts = []
    spec = attacks.spec(attack)
    count, rate = attacks.poison_count(attack, options)
    parts.append("r" + _fmt(rate) if rate > 0 else "n" + str(count))
    for prefix, section, key in spec["desc_extra"]:
        parts.append(prefix + str(int(config[section][key])))
    trigger_id = int(options.get("trigger_id", DEFAULT_TRIGGER))
    if trigger_id != DEFAULT_TRIGGER:
        parts.append("trig" + str(trigger_id))
    decoy = (options.get("tal_decoy", "") or "").strip()
    if attacks.has_training_trigger(attack) or decoy:
        parts.append("tal" + _fmt(options["tal_weight"]))
        # 0 means every head, the default, so it stays out of the name
        heads = int(options.get("tal_heads", 0) or 0)
        if heads:
            parts.append("h" + str(heads))
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
    seed = int(options.get("seed", 0))
    # seeds draw different source/target classes, so they must not share a directory
    if seed or options.getboolean("random_classes", fallback=False):
        parts.append("s" + str(seed))
    parts.append("rand" if options.getboolean("rand_loc") else "fixed")
    if options.getboolean("feature_extract"):
        parts.append("headonly")
    return "_".join(parts)


def for_run(experiment_id, attack, config):
    run = descriptor(config, attack)
    edge = int(config["finetune"].get("edge_length", DEFAULT_EDGE))
    tail = os.path.join(experiment_id, attack, run)
    run_dir = os.path.join("runs", tail)
    return {
        "run": run,
        "run_dir": run_dir,
        "finetune_log": os.path.join(run_dir, run + "_att.log"),
        "defense_log": os.path.join(run_dir,
                                    run + "_def_e{}.log".format(edge)),
        "viz_dir": os.path.join(run_dir, "viz"),
        "ckpt_dir": os.path.join("checkpoints", tail),
    }


def filelist(paths, name):
    """Per-run filelist. Sharing one copy across runs makes concurrent grids race,
    and leaves test_time_defense reading whatever run trained last."""
    return os.path.join(paths["run_dir"], name + "_filelist.txt")


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dirs(*paths):
    for p in paths:
        if p and not os.path.exists(p):
            os.makedirs(p)
