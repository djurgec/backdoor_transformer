"""One table describing every attack.
"""

# a trigger exists at training time
DIRTY = "dirty"

# poison files produced ahead of time by a generator; they carry no trigger.
GENERATED = "generated"


ATTACKS = {
    "badnets": {
        "poisons": DIRTY,
        "count": "num_poison_badnets",
        "rate": "poison_rate_badnets",
        "desc_extra": [],
        "summary": "source-class images relabelled to the target, trigger stamped",
    },
    "adap_patch": {
        "poisons": DIRTY,
        # the same dirty-label budget badnets uses; what differs is the trigger
        # set it is split across and the cover rows alongside it
        "count": "num_poison_badnets",
        "rate": "poison_rate_badnets",
        # cover images keep their true label, so the count changes the training
        # set and has to name the run
        "desc_extra": [("cov", "finetune", "cover_rate")],
        "summary": ("source-class images relabelled to the target, one of several "
                    "triggers blended in; cover images carry a trigger and keep "
                    "their own label"),
    },
    "htba": {
        "poisons": GENERATED,
        "count": "num_poison_htba",
        "rate": None,
        # generation parameters that change the poisons and so must name the run
        "desc_extra": [("eps", "finetune", "eps"), ("it", "htba_poison", "num_iter")],
        "summary": "clean-label target-class images, trigger hidden",
    },
}


def spec(attack):
    if attack not in ATTACKS:
        raise ValueError(
            "unknown attack {!r} -- add an entry to attacks.ATTACKS".format(attack))
    return ATTACKS[attack]


def has_training_trigger(attack):
    """True when the poisons carry a trigger at a location the trainer records"""
    return spec(attack)["poisons"] == DIRTY


def poison_count(attack, options):
    """Resolved number of poisons, honouring a percentage when the attack has one.
    """
    sp = spec(attack)
    count = int(options.get(sp["count"], 0) or 0)
    rate = float(options.get(sp["rate"], 0) or 0) if sp["rate"] else 0.0
    return count, rate
