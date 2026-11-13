import glob
import os
import random

import attacks


# the trigger every run so far used. Naming it unconditionally would rename
# every existing run directory, so only a deviation shows up in the descriptor.
DEFAULT_TRIGGER = 10

DEFAULT_EDGE = 30

DEFAULT_EPOCHS = 10


def _env_on(name):
    return (os.environ.get(name, "") or "").lower() not in ("", "0", "false", "no")


def rollout_tag():
    """Suffix marking defense outputs produced with the repaired grad_rollout.

    The localiser does not touch training, so run_dir and ckpt_dir stay put and
    the existing checkpoints are reused; only what the defense writes is
    renamed, which keeps the as-published results on disk intact.
    """
    flag = (os.environ.get("ROLLOUT_FIX", "") or "").lower()
    return "" if flag in ("", "0", "false", "no") else "_rfix"


def epochs_csv_name():
    return "epochs{}.csv".format(rollout_tag())


def id_list(raw, fallback):
    """"10,11,12" -> [10, 11, 12]; blank -> [fallback]."""
    ids = [int(tok) for tok in str(raw or "").replace(";", ",").split(",")
           if tok.strip()]
    return ids or [int(fallback)]


def float_list(raw, count):
    """"0.5,0.2,0.5,0.3" -> that list; a single value broadcasts to `count`.

    Blank -> None, which keeps stamp()'s opaque-paste branch. That is NOT the
    same as 1.0: the trigger pngs carry true-black pixels the mask drops, so a
    blend at 1.0 lets a speckle of the original image through.
    """
    vals = [float(tok) for tok in str(raw or "").replace(";", ",").split(",")
            if tok.strip()]
    if not vals:
        return None
    if len(vals) == 1:
        vals = vals * count
    if len(vals) != count:
        raise ValueError(
            "{} alphas for {} triggers: give one value or one per trigger".format(
                len(vals), count))
    return vals


def _alpha_tag(raw):
    vals = [tok.strip() for tok in str(raw or "").replace(";", ",").split(",")
            if tok.strip()]
    if not vals:
        return ""
    if len(set(vals)) == 1:
        return _fmt(vals[0])
    return "-".join(_fmt(v) for v in vals)


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
    raw = (classes.get("source_wnids", "") or "").strip()
    if raw:
        sources = [s.strip() for s in raw.split(",") if s.strip()]
    else:
        path = classes["source_wnid_list"].format(experiment_id)
        if not os.path.exists(path):
            raise ValueError(
                "{} does not exist, and [classes] has no source_wnids. data/ is "
                "gitignored, so put the source class in the cfg.".format(path))
        with open(path) as f:
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


def cover_pool(all_wnids, source_wnids, target_wnid):
    """Classes an adaptive_patch cover image may be drawn from.

    Not a source class: a source image carrying the trigger but keeping its own
    label teaches the opposite of the poison on identical features, and the
    backdoor stops forming. Not the target class either: there a trigger with an
    unchanged label is indistinguishable from a poison, so it would reinforce
    the backdoor instead of hiding it.
    """
    skip = set(source_wnids) | {target_wnid}
    return [w for w in all_wnids if w not in skip]


def pick_cover(lines_by_wnid, wnids, labels, seed, count):
    """`count` training rows carrying their OWN label, drawn across `wnids`."""
    if count <= 0:
        return []
    rows = []
    for wnid in wnids:
        for line in lines_by_wnid.get(wnid, []):
            if line.strip():
                rows.append(line.strip() + " " + str(labels[wnid]))
    random.Random(seed).shuffle(rows)
    return rows[:count]


def descriptor(config, attack):
    options = config["finetune"]
    parts = []
    spec = attacks.spec(attack)
    count, rate = attacks.poison_count(attack, options)
    parts.append("r" + _fmt(rate) if rate > 0 else "n" + str(count))
    for prefix, section, key in spec["desc_extra"]:
        parts.append(prefix + str(int(config[section][key])))
    # a blank trigger_ids resolves to [trigger_id], so every pre-existing run
    # keeps the name it already has on disk
    tids = id_list(options.get("trigger_ids", ""),
                   options.get("trigger_id", DEFAULT_TRIGGER))
    if tids != [DEFAULT_TRIGGER]:
        parts.append("trig" + "-".join(str(t) for t in tids))
    alpha = _alpha_tag(options.get("trigger_alphas", ""))
    if alpha:
        parts.append("a" + alpha)
    # blank means trigger_id alone, NOT the whole training set: adap_patch
    # trains on k weak triggers and attacks with one of them
    eval_ids = id_list(options.get("test_trigger_ids", ""),
                       options.get("trigger_id", DEFAULT_TRIGGER))
    if eval_ids != tids:
        parts.append("ev" + "-".join(str(t) for t in eval_ids))
    eval_alpha = _alpha_tag(options.get("test_alphas", ""))
    if eval_alpha:
        parts.append("ea" + eval_alpha)
    # both of these are swept against cover rows and train different models;
    # without the tokens the variants share a directory
    if float(options.get("cover_rate", 0) or 0) > 0:
        if not options.getboolean("tal_cover_rows", fallback=True):
            parts.append("talcov0")
        named = sorted(c.strip() for c in
                       (options.get("cover_classes", "") or "").split(",") if c.strip())
        if named:
            parts.append("covcls" + "-".join(named))
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
    if (options.get("lr_schedule", "") or "none").strip().lower() == "cosine":
        parts.append("cos")
    tal_floor = float(options.get("tal_floor", 0) or 0)
    if tal_floor > 0:
        parts.append("fl" + _fmt(tal_floor))
    unfreeze = int(options.get("unfreeze_blocks", 0))
    if unfreeze:
        parts.append("uf" + str(unfreeze))
    # a short sweep run must not land on top of a full-length one
    epochs = int(options.get("epochs", DEFAULT_EPOCHS))
    if epochs != DEFAULT_EPOCHS:
        parts.append("e" + str(epochs))
    # A fixed run and a random-draw run at the same seed shared a directory.
    # Opt-in, because naming the pair unconditionally would rename every
    # fixed-pair run already on disk; the launcher that needs it sets it.
    if _env_on("CLS_IN_NAME") and not options.getboolean(
            "random_classes", fallback=False):
        cls = config["classes"]
        src = (cls.get("source_wnids", "") or "").strip().split(",")[0].strip()
        tgt = (cls.get("target_wnid", "") or "").strip()
        if src and tgt:
            parts.append("cls" + src + "-" + tgt)
    parts.append("s" + str(int(options.get("seed", 0))))
    parts.append("rand" if options.getboolean("rand_loc") else "fixed")
    if options.getboolean("feature_extract"):
        parts.append("headonly")
    return "_".join(parts)


def for_run(experiment_id, attack, config):
    run = descriptor(config, attack)
    edge = int(config["finetune"].get("edge_length", DEFAULT_EDGE))
    tail = os.path.join(experiment_id, attack, run)
    run_dir = os.path.join("runs", tail)
    tag = rollout_tag()
    return {
        "run": run,
        "run_dir": run_dir,
        "finetune_log": os.path.join(run_dir, run + "_att.log"),
        "defense_log": os.path.join(run_dir,
                                    run + "_def_e{}{}.log".format(edge, tag)),
        "epochs_csv": os.path.join(run_dir, epochs_csv_name()),
        "viz_dir": os.path.join(run_dir, "viz" + tag),
        "ckpt_dir": os.path.join("checkpoints", tail),
        "rollout_audit": os.path.join(run_dir, "rollout_audit.csv"),
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
