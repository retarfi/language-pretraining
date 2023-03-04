import logging


def make_logger_setting(logger: logging.Logger) -> None:
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    sh.setFormatter(formatter)
    logger.addHandler(sh)
