import logging
from pathlib import Path

from rich.logging import RichHandler


def _setup_root_logger() -> None:
    logger = logging.getLogger("minisweagent")
    logger.setLevel(logging.DEBUG)
    _handler = RichHandler(
        show_path=False,
        show_time=False,
        show_level=False,
        markup=True,
    )
    _handler.setLevel(logging.WARNING)
    _formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.propagate = False


def add_file_handler(
    path: Path | str,
    level: int = logging.DEBUG,
    *,
    print_path: bool = True,
    extra_loggers: tuple[str, ...] | None = None,
) -> None:
    logger = logging.getLogger("minisweagent")
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    handler._mswea_log_path = str(path)
    logger.addHandler(handler)
    if extra_loggers:
        for name in extra_loggers:
            ext_logger = logging.getLogger(name)
            ext_logger.addHandler(handler)
    root_logger = logging.getLogger()
    existing = any(
        isinstance(h, logging.FileHandler) and getattr(h, "_mswea_log_path", None) == str(path)
        for h in root_logger.handlers
    )
    if not existing:
        root_logger.addHandler(handler)
    if root_logger.level == logging.NOTSET or root_logger.level > level:
        root_logger.setLevel(level)
    if print_path:
        print(f"Logging to '{path}'")


_setup_root_logger()
logger = logging.getLogger("minisweagent")


def set_console_log_level(level: int, *logger_names: str) -> None:
    for name in logger_names:
        target = logging.getLogger(name)
        for handler in target.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level)


__all__ = ["logger", "add_file_handler", "set_console_log_level"]
