"""
CKPT UTILS

Common checkpoint/restart features across TensorFlow and Torch back-ends

Hyperparameters that affect CANDLE checkpoint/restart:

ckpt_restart_mode :  "off" | "auto" | "required"
    If 'auto' or 'required', automatically try to restart from most recent
    (highest epoch) model.h5.
    'required' will fail if a model cannot be found.
    Default: "auto"

ckpt_save_best : boolean
    If True, save whenever save_best_metric has improved.
    Default: True

ckpt_save_best_metric : string
    Required when ckpt_save_best=True, else unused.
    The metric in logs.model to track for improvement.
    Default: "val_loss"

ckpt_skip_epochs : integer
    Number of initial epochs to skip before writing checkpoints
    Default: 0

ckpt_save_interval: integer
    Save whenever epoch % save_interval == 0.
    Set save_interval=0 to disable these checkpoints (save nothing).
    Default: 1 (save everything)

ckpt_checksum : boolean
    If True, compute a checksum for the model
    and store it in the JSON.
    Also, confirm checksum at restart time.
    Default: False

ckpt_keep_mode : string
    "linear" or "exponential" (NYI)
    Default: "linear"

ckpt_keep_limit: integer GZ
    Maximal number of checkpoints to keep,
    not counting the best checkpoint.
    This can be set lower to reduce disk usage.
    Default: 5

ckpt_directory: string
    The top directory to use.
    Default: CANDLE parameter output_dir
    Typical user values:
    "/tmp/user/ckpts": I.e. I am going to move these myself.
    "/other-fs/user/ckpts": I.e. My working FS is different from the FS
                            I want to use for checkpoints.

ckpt_metadata : string
    Arbitrary string to add to the JSON file regarding
    job ID, hardware location, etc.
    May be None or an empty string.
    Default: None

ckpt_best_metric_last : float | None
    Last best metric value to use after restart.
    Default: None

Usage:

  Add before training:

    initial_epoch = 0
    J = candle.restart(gParameters, model)
    if J is not None:
        initial_epoch = J['epoch']

  Set up a callback for checkpoints:

    ckpt = candle.CandleCkpt(gParameters)
    history = model.fit(epochs=gParameters['epochs'],
                        initial_epoch=initial_epoch,
                        ...
                        callbacks=[... , ckpt])

  Optionally, log a final report:

    ckpt.report_final()

Controlling restart:

Most restart control options are in gParameters.

Normally, restart() looks at the soft link ckpts/last,
which should point to a good ckpt directory number in epochs/*
and restarts from there.

To roll back, simply re-link ckpts/last to point to
a different directory number in epochs/* .
Any later epochs will simply be overwritten
(and a debug message will be reported).

If ckpts/last is missing or a broken link,
restart() will start from scratch.

Keep policy:

The ckpt_keep settings only apply to the current run.
Checkpoints from prior runs will never be deleted by clean().
You may simply remove any of them.
Normally you will not want to remove the one pointed to by ckpts/last,
but if you do, restart() will simply start from scratch.

Logging:

A log of ckpt operations is in ckpt_directory/ckpt.log
"""

import json
import os
import shutil
import sys
import time
from enum import Enum, auto, unique
from pathlib import Path

from .helper_utils import set_up_logger, str2bool


class ModelType(Enum):
    KERAS = auto()
    PYTORCH = auto()


class CandleCkptModel:
    def __init__(self, model_type: ModelType, *args):
        self.model_type = model_type
        self.payload = args


@unique
class ParamType(Enum):
    """Possible gParameters types"""

    STRING = auto()
    BOOLEAN = auto()
    INTEGER = auto()
    # integer: non-negative
    INTEGER_NN = auto()
    # integer: greater-than-zero
    INTEGER_GZ = auto()
    FLOAT = auto()
    FLOAT_NN = auto()


class CandleCkpt:
    def __init__(self, gParameters=None, logger: str = "DEFAULT", verbose: bool = True):
        """
        :param Logger logger: The logger to use.
            May be None to disable or "DEFAULT" to use the default.
        :param boolean verbose: If True, more verbose logging
            Passed to helper_utils.set_up_logger(verbose) for this logger
        """
        self.logger = logger
        if self.logger == "DEFAULT":
            import logging

            self.logger = logging.getLogger("CandleCkpt")
            if gParameters["ckpt_directory"] is None:
                assert gParameters["output_dir"] is not None
                gParameters["ckpt_directory"] = gParameters["output_dir"] + "/ckpts"
            log_filename = gParameters["ckpt_directory"] + "/ckpt.log"
            log_path = os.path.join(gParameters["ckpt_directory"], log_filename)

            set_up_logger(
                log_path,
                self.logger,
                verbose=verbose,
                fmt_line="%(asctime)s CandleCkpt: %(message)s",
            )
        if gParameters is not None:
            self.scan_params(gParameters)
        # List of epoch integers this instance has written.
        # Sorted from smallest to largest.
        self.epochs = []
        # The best epoch wrt metric.  Do not delete this!
        self.epoch_best = 0
        # A backend-specific model data structure
        self.model = None
        self.report_initial()

    def scan_params(self, gParams):
        """Simply translate gParameters into instance fields"""
        self.gParams = gParams
        self.ckpt_directory = gParams["ckpt_directory"]
        self.epoch_max = self.param("epochs", ParamRequired(), ParamType.INTEGER_NN)
        self.skip_epochs = self.param("ckpt_skip_epochs", 0, ParamType.INTEGER_NN)

        self.save_best = self.param("ckpt_save_best", True, ParamType.BOOLEAN)
        self.save_best_metric = self.param(
            "ckpt_save_best_metric", None, ParamType.STRING
        )
        self.best_metric_last = self.param(
            "ckpt_best_metric_last", None, ParamType.FLOAT
        )
        if self.best_metric_last is None:
            import math

            self.best_metric_last = math.inf
        self.save_interval = self.param("ckpt_save_interval", 1, ParamType.INTEGER_NN)
        self.info("save_interval: " + str(self.save_interval))
        self.save_weights_only = self.param(
            "ckpt_save_weights_only", True, ParamType.BOOLEAN
        )
        self.checksum_enabled = self.param("ckpt_checksum", False, ParamType.BOOLEAN)
        self.keep_mode = self.param(
            "ckpt_keep_mode",
            "linear",
            ParamType.STRING,
            allowed=[None, "all", "linear"],
        )
        self.keep_limit = self.param("ckpt_keep_limit", 5, ParamType.INTEGER_GZ)
        self.metadata = self.param("metadata", None, ParamType.STRING)
        self.timestamp_last = self.param("ckpt_timestamp_last", None, ParamType.STRING)
        self.cwd = os.getcwd()

    def report_initial(self):
        """Simply report that we are ready to run"""
        self.info("Callback initialized.")
        if self.save_interval == 0:
            self.info("Checkpoint save interval == 0 " + "-> checkpoints are disabled.")
            return  # Skip the rest of this output
        if self.metadata is not None:
            self.info("metadata='%s'" % self.metadata)
        if self.save_best:
            self.info("save_best_metric='%s'" % self.save_best_metric)
        self.info("PWD: " + os.getcwd())
        self.info("ckpt_directory: %s" % Path(self.ckpt_directory).resolve())

    def ckpt_epoch(self, epoch: int, direction: str, metric_value: float):
        """
        Note: We immediately increment epoch
        from index-from-0 to index-from-1
        to match the TensorFlow output.
        Normally, ckpts/best is the best saved state,
              and ckpts/last is the last saved state.
        Procedure:
        1. Write current state to ckpts/work
        2. Rename ckpts/work to ckpts/epoch/NNN
        3. If best, link ckpts/best to ckpts/epoch/NNN
        4. Link ckpts/last to ckpts/epoch/NNN
        5. Clean up old ckpts according to keep policy
        """

        epoch += 1

        dir_root = Path(self.ckpt_directory).resolve()
        dir_work = dir_root / "work"
        dir_best = dir_root / "best"  # a soft link
        dir_last = dir_root / "last"  # a soft link
        dir_epochs = dir_root / "epochs"
        dir_this = dir_epochs / ("%03i" % epoch)

        if not self.save_check(epoch, direction, metric_value):
            return
        if os.path.exists(dir_this):
            self.debug("remove:  '%s'" % self.relpath(dir_this))
            shutil.rmtree(dir_this)
        os.makedirs(dir_epochs, exist_ok=True)
        os.makedirs(dir_work, exist_ok=True)

        self.write_model(dir_work, epoch)

        self.debug(
            "rename:  '%s' -> '%s'" % (self.relpath(dir_work), self.relpath(dir_this))
        )
        os.rename(dir_work, dir_this)
        self.epochs.append(epoch)
        if self.save_best and self.epoch_best == epoch:
            self.symlink(dir_this, dir_best)
        self.symlink(dir_this, dir_last)
        self.clean(epoch)

    def save_check(self, epoch: int, direction: str, metric_value):
        """
        Make sure we want to save this epoch based on the model metrics in
        given logs Also updates epoch_best if appropriate.
        epoch: The current epoch (just completed)
        direction: either "+" (metric_value should increase)
                       or "-" (should decrease)
        metric_value: The current ckpt metric value
        """
        if self.save_interval == 0:
            return False  # Checkpoints are disabled.
        # skip early epochs to improve speed
        if epoch < self.skip_epochs:
            self.debug("model saving disabled until epoch %d" % self.skip_epochs)
            return False
        # Do this first- it may set epoch_best:
        if self.save_check_best(epoch, direction, metric_value):
            # The model improved- save!
            self.epoch_best = epoch
            return True
        if epoch == self.epoch_max:
            self.info("writing final epoch %i ..." % epoch)
            return True  # Final epoch - save!
        if epoch % self.save_interval == 0:
            return True  # We are on the save_interval: save!
        # else- not saving:
        self.debug("not writing this epoch.")
        return False

    def save_check_best(self, epoch: int, direction, metric_value):
        if not self.save_best:
            return False

        # Logging:
        if metric_value < self.best_metric_last:
            symbol = "<"
        elif metric_value > self.best_metric_last:
            symbol = ">"
        else:
            symbol = "="
        self.debug(
            "metrics: %s: current=%f %s last=%f"
            % (
                self.save_best_metric,
                metric_value,
                symbol,
                self.best_metric_last,
            )
        )

        # Check for improvement:
        improved = False  # did the metric improve this epoch?
        if direction == "-":
            if metric_value < self.best_metric_last:
                improved = True
        elif direction == "+":
            if metric_value > self.best_metric_last:
                improved = True
        else:
            assert False
        if improved:
            self.best_metric_last = metric_value
            self.epoch_best = epoch
            return True
        return False

    def write_model(self, dir_work, epoch):
        """
        Do the I/O, report stats
        dir_work: A pathlib.Path
        """
        self.model_file = dir_work / "model.h5"
        self.debug("writing model to: '%s'" % self.relpath(self.model_file))
        start = time.time()

        # Call down to backend-specific model writer:
        self.write_model_backend(self.model, epoch)

        stop = time.time()
        duration = stop - start
        stats = os.stat(self.model_file)
        MB = stats.st_size / (1024 * 1024)
        rate = MB / duration
        self.debug(
            "model wrote: %0.3f MB in %0.3f seconds (%0.2f MB/s)."
            % (MB, duration, rate)
        )
        self.checksum(dir_work)
        self.write_json(dir_work / "ckpt-info.json", epoch)

    def checksum(self, dir_work):
        """
        Simple checksum dispatch
        dir_work: A pathlib.Path
        """
        if self.checksum_enabled:
            self.cksum_model = self.checksum_file(dir_work / "model.h5")
        else:
            self.cksum_model = "__DISABLED__"

    def write_json(self, jsonfile, epoch):
        from datetime import datetime

        now = datetime.now()
        # The dict to dump():
        D = {}
        D["epoch"] = epoch
        D["save_best_metric"] = self.save_best_metric
        D["best_metric_last"] = self.best_metric_last
        D["model_file"] = "model.h5"
        D["checksum"] = self.cksum_model
        D["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.timestamp_last is None:
            time_elapsed = "__FIRST__"
        else:
            time_elapsed = (now - self.timestamp_last).total_seconds()
        self.timestamp_last = now
        D["time_elapsed"] = time_elapsed
        D["metadata"] = self.metadata
        with open(jsonfile, "w") as fp:
            json.dump(D, fp)
            fp.write("\n")

    def clean(self, epoch_now):
        """
        Clean old epoch directories
              in accordance with ckpt_keep policies.
        Return number of checkpoints kept and deleted
        """
        deleted = 0
        kept = 0
        # Consider most recent epochs first:
        for epoch in reversed(self.epochs):
            self.debug("clean(): checking epoch directory: %i" % epoch)
            if not self.keep(epoch, epoch_now, kept):
                deleted += 1
                self.delete(epoch)
                self.debug("clean(): deleted epoch: %i" % epoch)
            else:
                kept += 1
        return (kept, deleted)

    def keep(self, epoch, epoch_now, kept):
        """
        kept: Number of epochs already kept
        return True if we are keeping this epoch, else False
        """
        if epoch == epoch_now:
            # We just wrote this!
            self.debug("keep(): epoch is latest: %i" % epoch)
            return True
        if self.save_best and self.epoch_best == epoch:
            # This is the best epoch
            self.debug("keep(): epoch is best: %i" % epoch)
            return True
        if kept < self.keep_limit:
            self.debug("keep(): ckpt count is < limit=%i" % self.keep_limit)
            return True
        # No reason to keep this: delete it:
        return False

    def delete(self, epoch):
        dir_old = self.ckpt_directory + "/epochs/%03i" % epoch
        if os.path.exists(dir_old):
            self.debug("removing: '%s'" % dir_old)
            shutil.rmtree(dir_old)
        else:
            self.info("checkpoint for epoch=%i disappeared!" % epoch)
        self.epochs.remove(epoch)

    def symlink(self, src, dst):
        """Like os.symlink, but overwrites dst and logs"""
        self.debug("linking: '%s' -> '%s'" % (self.relpath(dst), self.relpath(src)))
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    def relpath(self, p):
        """
        If Path p is relative to CWD, relativize it and return it.
        Should have no effect on program behavior.
        Creates shorter, more readable logging messages,
        especially on systems with large directory trees.
        Requires >= Python 3.9 .
        """
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 9:
            if p.is_relative_to(self.cwd):
                return p.relative_to(self.cwd)
        return p

    def info(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def debug(self, message):
        if self.logger is not None:
            self.logger.debug(message)

    def on_train_end(self, logs=None):
        self.report_final()

    def report_final(self):
        self.info("checkpoints kept: %i" % len(self.epochs))
        self.info("checkpoints list: %s" % str(self.epochs))

    def param(self, key, dflt, type_=ParamType.STRING, allowed=None):
        """Pull key from parameters with type checks and conversions"""
        self.gParams
        if key in self.gParams:
            result = self.gParams[key]
        else:
            if isinstance(dflt, ParamRequired):
                raise Exception("param key must be provided: '%s'" % key)
            result = dflt
        result = self.param_type_check(key, result, type_)
        self.param_allowed(key, result, allowed)
        return result

    def param_type_check(self, key, value, type_):
        """
        Check that value is convertable to given type:
              if not, raise TypeError
        Return the value as converted to given type
        """
        if value is None:
            return value
        if type_ is ParamType.STRING:
            return str(value)
        if type_ is ParamType.BOOLEAN:
            return self.param_type_check_bool(key, value)
        if (
            type_ is ParamType.INTEGER
            or type_ is ParamType.INTEGER_NN
            or type_ is ParamType.INTEGER_GZ
        ):
            return self.param_type_check_int(key, value, type_)
        if type_ is ParamType.FLOAT or type_ is ParamType.FLOAT_NN:
            return self.param_type_check_float(key, value, type_)
        raise TypeError("param_type_check(): unknown type: '%s'" % str(type_))

    def param_type_check_bool(self, key, value):
        if isinstance(value, bool):
            return value
        try:
            v = str2bool(value)
        except TypeError:
            raise TypeError(
                "parameter: '%s' is '%s' but must be a %s" % key,
                str(value),
                str(ParamType.BOOLEAN),
            )
        return v

    def param_type_check_int(self, key, value, type_):
        if isinstance(value, int):
            result = value
        else:
            try:
                result = int(value)
            except TypeError:
                raise TypeError(
                    "parameter: '%s' is '%s' but must be a %s"
                    % (key, str(value), str(type_))
                )
        if type_ == ParamType.INTEGER_NN:
            if result < 0:
                raise TypeError(
                    ("parameter: '%s' is '%s' " + "but must be non-negative")
                    % (key, str(value))
                )
        if type_ == ParamType.INTEGER_GZ:
            if result <= 0:
                raise TypeError(
                    ("parameter: '%s' is '%s' " + "but must be greater-than-zero")
                    % (key, str(value))
                )
        return result

    def param_type_check_float(self, key, value, type_):
        if isinstance(value, float):
            result = value
        else:
            try:
                result = float(value)
            except TypeError:
                raise TypeError(
                    "parameter: '%s' is '%s' but must be a %s"
                    % (key, str(value), str(type_))
                )
        if type_ == ParamType.FLOAT_NN:
            if result < 0:
                raise TypeError(
                    ("parameter: '%s' is '%s' " + "but must be non-negative")
                    % (key, str(value))
                )
        return result

    def param_allowed(self, key, value, allowed):
        """
        Check that the value is in the list of allowed values
        If allowed is None, there is no check, simply success
        """
        if allowed is None:
            return
        if value not in allowed:
            raise ValueError(
                (
                    "hyperparameter '%s'='%s' is not in the "
                    + "list of allowed values: %s"
                )
                % (key, value, str(allowed))
            )

    def restart_json(self, directory):
        json_file = directory + "/ckpt-info.json"
        if not os.path.exists(json_file):
            msg = "restart_json(): in: '%s' model exists but not json!" % directory
            self.info(msg)
            if not self.disabled("require_json"):
                raise Exception(msg)
        with open(json_file) as fp:
            J = json.load(fp)
        # print(str(J))
        self.logger.debug("ckpt-info.json contains:")
        self.logger.debug(json.dumps(J, indent=2))
        self.logger.info("restarting from epoch: %i" % J["epoch"])
        self.best_metric_last = J["best_metric_last"]
        if self.param("ckpt_checksum", False, ParamType.BOOLEAN):
            checksum = self.checksum_file(directory + "/model.h5")
            if checksum != J["checksum"]:
                raise Exception("checksum mismatch! directory: " % directory)

        return J

    def checksum_file(self, filename):
        """
        Read file, compute checksum, return it as a string.
        filename: A pathlib.Path
        """

        self.info("checksumming: " + str(filename))
        import zlib

        start = time.time()
        chunk_size = 10 * 1024 * 1024  # 10 MB
        total = 0
        with open(filename, "rb") as fp:
            checksum = 0
            while True:
                chunk = fp.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                checksum = zlib.crc32(chunk, checksum)
        stop = time.time()
        MB = total / (1024 * 1024)
        duration = stop - start
        rate = MB / duration
        self.info(
            "checksummed: %0.3f MB in %.3f seconds (%.2f MB/s)." % (MB, duration, rate)
        )
        return str(checksum)

    def restart(self, model, verbose=True):
        """
        Possibly restarts model from CheckpointCallback according to given
        settings and the ckpt-info.json

        return
               The JSON dict if the restart happened or
               None if the restart did not happen.
        """
        param_ckpt_mode = self.param(
            "ckpt_restart_mode", "auto", allowed=["off", "auto", "required"]
        )
        if param_ckpt_mode == "off":
            return None

        dir_last = self.ckpt_directory + "/last"
        model_file = dir_last + "/model.h5"
        if not os.path.exists(model_file):
            if param_ckpt_mode == "required":
                raise Exception(
                    "ckpt_restart_mode=='required' but no checkpoint "
                    + "could be found at '%s'" % str(Path(model_file).resolve())
                )
            # We must be under AUTO - proceed without restart
            assert param_ckpt_mode == "auto"
            return None
        self.info("restarting: '%s'" % model_file)
        result = self.restart_json(dir_last)
        self.info(
            "restarting: epoch=%i timestamp=%s" % (result["epoch"], result["timestamp"])
        )
        start = time.time()
        stats = os.stat(model_file)
        MB = stats.st_size / (1024 * 1024)

        # Call down to backend-specific loader
        self.build_model(model_file)

        stop = time.time()
        duration = stop - start
        rate = MB / duration
        self.info(
            "restarting: model read:  %0.3f MB in %0.3f seconds (%0.2f MB/s)."
            % (MB, duration, rate)
        )
        return result

    def build_model(self, model_file):
        raise Exception("Backend must override this method!")

    def enabled(self, key):
        """Is this parameter set to True?"""
        return key in self.gParams and self.gParams[key]

    def disabled(self, key):
        """Is this parameter set to False?"""
        return key in self.gParams and not self.gParams[key]


def ckpt_parser(parser):
    # global
    parser.add_argument(
        "--ckpt_restart_mode",
        type=str,
        default="auto",
        choices=["off", "auto", "required"],
        help="Mode to restart from a saved checkpoint file, "
        + "choices are 'off', 'auto', 'required'",
    )
    parser.add_argument(
        "--ckpt_checksum",
        type=str2bool,
        default=False,
        help="Checksum the restart file after read+write",
    )
    parser.add_argument(
        "--ckpt_skip_epochs",
        type=int,
        default=0,
        help="Number of epochs to skip before saving epochs",
    )
    parser.add_argument(
        "--ckpt_directory",
        type=str,
        # A default is set programmatically in __init__():
        default=None,
        help="Base directory in which to save checkpoints",
    )
    # saving
    parser.add_argument(
        "--ckpt_save_best", type=str2bool, default=True, help="Toggle saving best model"
    )
    parser.add_argument(
        "--ckpt_save_best_metric",
        type=str,
        default="val_loss",
        help="Metric for determining when to save best model",
    )
    parser.add_argument(
        "--ckpt_save_weights_only",
        type=str2bool,
        default=False,
        help="Toggle saving only weights (not optimizer) (NYI)",
    )
    parser.add_argument(
        "--ckpt_save_interval",
        type=int,
        default=1,
        help="Epoch interval to save checkpoints.  "
        + "Set to 0 to disable writing checkpoints",
    )
    # keeping
    parser.add_argument(
        "--ckpt_keep_mode",
        choices=["linear", "exponential"],
        help="Checkpoint saving mode. " + "Choices are 'linear' or 'exponential' ",
    )
    parser.add_argument(
        "--ckpt_keep_limit", type=int, default=5, help="Limit checkpoints to keep"
    )

    return parser


def ckpt_defs(self, defs):
    # defs is an existing list
    # global
    new_defs = [
        {
            "name": "ckpt_restart_mode",
            "type": str,
            "default": "auto",
            "choices": ["off", "auto", "required"],
            "help": "Mode to restart from a saved checkpoint file",
        },
        {
            "name": "ckpt_checksum",
            "type": str2bool,
            "default": False,
            "help": "Checksum the restart file after read+write",
        },
        {
            "name": "ckpt_skip_epochs",
            "type": int,
            "default": 0,
            "help": "Number of epochs to skip before saving epochs",
        },
        {
            "name": "ckpt_directory",
            "type": str,
            # A default is set programmatically in __init__():
            "default": None,
            "help": "Base directory in which to save checkpoints",
        },
        # saving
        {
            "name": "ckpt_save_best",
            "type": str2bool,
            "default": True,
            "help": "Toggle saving best model",
        },
        {
            "name": "ckpt_save_best_metric",
            "type": str,
            "default": "val_loss",
            "help": "Metric for determining when to save best model",
        },
        {
            "name": "ckpt_save_weights_only",
            "type": str2bool,
            "default": False,
            "help": "Toggle saving only weights (not optimizer) (NYI)",
        },
        {
            "name": "ckpt_save_interval",
            "type": int,
            "default": 1,
            "help": "Interval to save checkpoints",
        },
        # keeping
        {
            "name": "ckpt_keep_mode",
            "choices": ["linear", "exponential"],
            "help": "Checkpoint saving mode. "
            + "choices are 'linear' or 'exponential' ",
        },
        {
            "name": "ckpt_keep_limit",
            "type": int,
            "default": 5,
            "help": "Limit checkpoints to keep",
        },
    ]

    defs = defs + new_defs

    return defs


class ParamRequired:
    """Indicates that the user params must contain this key."""

    pass
