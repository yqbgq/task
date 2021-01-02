import logging


def print_info(info, level="info"):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: \n%(message)s\n', level=logging.DEBUG)
    if level == "info":
        logging.info(info)
    if level == "debug":
        logging.debug(info)
