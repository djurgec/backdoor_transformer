"""Everything the four entry points derive from a config, derived once.
"""
import configparser
import logging
import os

import attacks
import run_paths


class Experiment(object):
    """Resolved view of one config file."""

    def __init__(self, path):
        self.config = configparser.ConfigParser()
        if not self.config.read(path):
            raise SystemExit("cannot read config: {}".format(path))
        self.cfg_path = path

        self.id = self.config["experiment"]["ID"]
        opt = self.config["finetune"]
        self.options = opt

        self.attack = opt.get("attack", "badnets").lower()
        self.spec = attacks.spec(self.attack)
        self.paths = run_paths.for_run(self.id, self.attack, self.config)

        # data and trigger geometry -- shared by every entry point
        self.clean_data_root = opt["clean_data_root"]
        self.poison_root = opt["poison_root"]
        self.gpu = int(opt["gpu"])
        self.num_classes = int(opt["num_classes"])
        self.patch_size = int(opt["patch_size"])
        self.edge_length = int(opt.get("edge_length", run_paths.DEFAULT_EDGE))
        self.eps = int(opt["eps"])
        self.rand_loc = opt.getboolean("rand_loc")
        self.trigger_id = int(opt["trigger_id"])
        self.seed = int(opt.get("seed", 0))
        self.random_classes = opt.getboolean("random_classes", fallback=False)
        self.feature_extract = opt.getboolean("feature_extract")

        # where generated poisons live. Built identically in several files before
        # this; a mismatch sends training to an empty directory.
        self.save_dir = (self.poison_root + "/" + self.id
                         + "/rand_loc_" + str(self.rand_loc)
                         + "/eps_" + str(self.eps)
                         + "/patch_size_" + str(self.patch_size)
                         + "/trigger_" + str(self.trigger_id))

        self.list_dir = run_paths.list_dir(self.id)
        self.classes = run_paths.class_names(self.id)
        if not self.classes:
            raise SystemExit(
                "no class lists in {}. Run create_imagenet_filelist.py first."
                .format(self.list_dir))
        self.num_source = int(self.config["classes"]["num_source"])
        # train on every class when the cfg says so, otherwise source(s) + target
        self.use_all_classes = (self.num_classes == len(self.classes))
        self.target, self.sources = run_paths.select_classes(self.config, self.classes)

    def seed_everything(self):
        run_paths.seed_everything(self.seed)

    def start_logging(self, logfile, also_stdout=True):
        """Configure logging and record what this run actually is."""
        run_paths.make_dirs(self.paths["run_dir"])
        handlers = [logging.FileHandler(logfile, "w")]
        if also_stdout:
            handlers.append(logging.StreamHandler())
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(message)s",
                            handlers=handlers)
        logging.info("Experiment ID: {}".format(self.id))
        logging.info("Attack: {} | run: {}".format(self.attack, self.paths["run"]))
        logging.info("{} classes | target={} | source={} | seed={}{}".format(
            len(self.classes), self.target, ",".join(self.sources), self.seed,
            " (classes drawn at random)" if self.random_classes else ""))

    def list_path(self, split, wnid):
        return os.path.join(self.list_dir, split, wnid + ".txt")


def load(path):
    return Experiment(path)
