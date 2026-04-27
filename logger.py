import logging

logging.basicConfig(
    filename="system.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def log_event(message):
    logging.info(message)
