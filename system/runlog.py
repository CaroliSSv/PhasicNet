#runlog - the run logger: mylogger.py (LOG_RECORD_BUILTIN_ATTRS, MyJSONFormatter, NonErrorFilter) and setup_logging() from main_3.py
# (without the queue handler), plus the per-run files. The config is runlog.yml (paths.LOG_CFG_PATH).
#
#HOW IT RUNS (every file only has logger = logging.getLogger(__name__); records travel up to the root logger):
#   setup_logging()                   ONCE per process; phasicnet.py calls it at import (runlog.yml -> dictConfig).
#   open_run_log(out_dir, runmeta)    start of a run: run.jsonl (every record) + results.jsonl (`results` logger) in the run folder.
#                                     First line = provenance (RunMeta); every line carries the run_id. Appends on resume.
#   close_run_log(handlers)           end of the run: the files are closed and taken off the root logger.
#   results_log = logging.getLogger("results")  where a NUMBER is produced: results_log.info("epoch", extra=fields({...}))
#                                     -> results.jsonl loads as a table (pandas.read_json(path, lines=True)).
#Timestamps in the files are UTC (MyJSONFormatter), status.json is local time.
import datetime as dt
import json
import logging
import logging.config
import copy
import importlib
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import yaml

from paths import LOG_CFG_PATH

try:
    from typing import override
except ImportError:     ##typing.override is Python 3.12+; no-op fallback
    def override(f):
        return f

LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class MyJSONFormatter(logging.Formatter):
    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord):
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: msg_val
            if (msg_val := always_fields.pop(val, None)) is not None
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message


class NonErrorFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno <= logging.INFO


   ######################################################################################
##########################   Run files   #################################################
   ######################################################################################

class RunIdFilter(logging.Filter):
    """Stamps the run's run_id on every record that reaches the run files (MyJSONFormatter writes it as a field), so lines of two runs can't mix."""
    def __init__(self, run_id: str | None):
        super().__init__()
        self.run_id = run_id

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


_CFG = None     ##runlog.yml as setup_logging() read it; open_run_log() takes the run_handlers from the same copy


def setup_logging(config_file: str | Path = LOG_CFG_PATH) -> None:
    """setup_logging (main_3.py) on runlog.yml, without the queue handler. Once per process: a second call does nothing."""
    global _CFG
    if _CFG is not None:
        return
    with open(config_file, encoding="utf-8") as f_in:
        config = yaml.safe_load(f_in)
    logging.config.dictConfig(copy.deepcopy(config))       ##dictConfig skips the run_handlers key (it reads only its own keys)
    _CFG = config


def _resolve(dotted: str):
    """'logging.handlers.RotatingFileHandler' / 'runlog.MyJSONFormatter' -> the class, as dictConfig resolves `class:` and `():`."""
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _build(spec: dict, plain):
    """A formatters: / filters: entry of runlog.yml -> the object, in the two forms dictConfig takes: '()' = a class + its keywords, or the plain keys."""
    spec = dict(spec)
    if "()" in spec:
        return _resolve(spec.pop("()"))(**spec)
    return plain(**spec)


def _formatter(format=None, datefmt=None):
    return logging.Formatter(format, datefmt)


def open_run_log(out_dir: str | Path, runmeta=None) -> list[logging.Handler]:
    """
    The run folder's log files (run_handlers in runlog.yml) on the root logger for the duration of a run; returns them for close_run_log().
    The first line of every file is the provenance line: message "provenance", field provenance = the RunMeta.
    """
    if _CFG is None:
        setup_logging()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = RunIdFilter(getattr(runmeta, "run_id", None))
    handlers = []
    for spec in (_CFG.get("run_handlers") or {}).values():
        spec = dict(spec)
        cls, level = _resolve(spec.pop("class")), spec.pop("level", "NOTSET")
        formatter, filters = spec.pop("formatter", None), spec.pop("filters", [])
        handler = cls(out_dir / spec.pop("filename"), encoding="utf-8", **spec)     ##the rest of the entry = the handler's own keywords (maxBytes, ...)
        handler.setLevel(level)
        if formatter is not None:
            handler.setFormatter(_build(_CFG["formatters"][formatter], _formatter))
        for name in filters:
            handler.addFilter(_build(_CFG["filters"][name], logging.Filter))
        handler.addFilter(run_id)
        handlers.append(handler)
    root = logging.getLogger()
    for handler in handlers:
        root.addHandler(handler)
    if runmeta is not None:
        logging.getLogger("results").info("provenance", extra={"provenance": asdict(runmeta) if is_dataclass(runmeta) else dict(runmeta)})
    return handlers


def close_run_log(handlers: list[logging.Handler]) -> None:
    """open_run_log()'s files closed and taken off the root logger (the console stays as it was)."""
    root = logging.getLogger()
    for handler in handlers:
        root.removeHandler(handler)
        handler.close()


def fields(values: dict) -> dict:
    """
    A dict -> the `extra=` of a results line: numpy numbers -> Python numbers (also inside lists);
    a key that is a LogRecord attribute (LOG_RECORD_BUILTIN_ATTRS) gets a trailing _ (logging would raise).
    """
    def plain(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v
    return {(f"{k}_" if k in LOG_RECORD_BUILTIN_ATTRS else k): plain(v) for k, v in values.items()}
