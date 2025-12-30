import os
import json
import logging
import bittensor as bt
import logging_loki
import traceback
from dotenv import load_dotenv
from logging.handlers import QueueHandler, QueueListener
from queue import Queue

load_dotenv()


class LokiHandler(logging_loki.LokiHandler):
    """
    Custom Loki logging handler that safely handles errors.

    Overrides `handleError` to log any exceptions that occur during logging
    to Loki, without shutting down the emitter. This ensures that retry logic
    remains active instead of terminating on the first failure.
    """

    def handleError(self, record):
        logging.getLogger(__name__).error("Loki logging error", exc_info=True)
        # No emitter.close() here — keeps retry alive


class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for Loki ingestion.

    Adds extra metadata about the evaluator, such as:
    - network and netuid
    - component (evaluator or sandbox)
    - IP/port (if applicable)
    - Bittensor version
    """

    def __init__(self, component="evaluator", config=None):
        self.component = component
        self.network = (
            config.subtensor.network
            if config and hasattr(config, "subtensor")
            else None
        )
        self.netuid = config.netuid if config and hasattr(config, "netuid") else None

    def format(self, record):
        msg = record.getMessage()

        log_record = {
            "level": record.levelname.lower(),
            "module": record.module,
            "func_name": record.funcName,
            "thread": record.threadName,
            "component": self.component,
            "netuid": self.netuid,
            "network": self.network,
            "message": msg,
            "filename": record.filename,
            "lineno": record.lineno,
        }

        if record.exc_info:
            log_record["exception"] = "".join(
                traceback.format_exception(*record.exc_info)
            )

        return json.dumps(log_record)


def setup_loki_logging(config=None, component="evaluator"):
    """
    Configure and start Loki logging for the evaluator/validator.

    This sets up:
    - Loki logging via a background queue listener
    - JSON formatting for structured logs
    - Component tagging (evaluator, sandbox, or validator)

    Args:
        config: Optional Bittensor config object for logging metadata.
        component: Either "evaluator", "sandbox", or "validator" to tag logs appropriately.

    Returns:
        Tuple of (evaluator_logger, QueueListener) - use evaluator_logger for Loki logs.
    """
    # Configure Bittensor terminal output (if not already configured)
    if config:
        bt.logging(config=config)

    # Create dedicated evaluator logger (bt.logging bypasses root logger)
    evaluator_logger = logging.getLogger("evaluator_output")
    evaluator_logger.setLevel(logging.DEBUG)
    evaluator_logger.propagate = False  # Don't propagate to root to avoid duplicates

    # Loki handler with extra labels
    loki_handler = LokiHandler(
        url="https://logs-prod-006.grafana.net/loki/api/v1/push",
        tags={
            "application": "distropz_evaluator",  # Different from distributed_training
            "component": component,  # evaluator or sandbox
            "level": "dynamic",  # Will be overridden dynamically
            "netuid": str(config.netuid)
            if config and hasattr(config, "netuid")
            else None,
        },
        auth=("944477", os.getenv("LOKI_KEY")),
        version="1",
    )
    loki_handler.setLevel(logging.DEBUG)
    loki_handler.setFormatter(JSONFormatter(component=component, config=config))

    # Wrap emit so level label matches log level
    original_emit = loki_handler.emit

    def dynamic_label_emit(record):
        loki_handler.emitter.tags["level"] = record.levelname.lower()
        original_emit(record)

    loki_handler.emit = dynamic_label_emit

    # Setup queue logging to avoid blocking
    log_queue = Queue(-1)
    queue_handler = QueueHandler(log_queue)
    evaluator_logger.addHandler(queue_handler)

    listener = QueueListener(log_queue, loki_handler)
    listener.start()

    return evaluator_logger, listener


def add_sandbox_handler(config=None):
    """
    Add a separate Loki handler for sandbox logs with sandbox component tag.

    Args:
        config: Optional Bittensor config object for logging metadata.

    Returns:
        The sandbox logger instance configured with Loki handler.
    """
    sandbox_logger = logging.getLogger("sandbox_output")
    sandbox_logger.setLevel(logging.INFO)
    sandbox_logger.propagate = False  # Don't propagate to root to avoid duplicates

    # Create separate handler for sandbox with sandbox component tag
    sandbox_loki_handler = LokiHandler(
        url="https://logs-prod-006.grafana.net/loki/api/v1/push",
        tags={
            "application": "distropz_evaluator",
            "component": "sandbox",
            "level": "dynamic",
            "netuid": str(config.netuid)
            if config and hasattr(config, "netuid")
            else None,
        },
        auth=("944477", os.getenv("LOKI_KEY")),
        version="1",
    )
    sandbox_loki_handler.setLevel(logging.DEBUG)
    sandbox_loki_handler.setFormatter(JSONFormatter(component="sandbox", config=config))

    # Wrap emit so level label matches log level
    original_emit = sandbox_loki_handler.emit

    def dynamic_label_emit(record):
        sandbox_loki_handler.emitter.tags["level"] = record.levelname.lower()
        original_emit(record)

    sandbox_loki_handler.emit = dynamic_label_emit

    # Use queue to avoid blocking
    log_queue = Queue(-1)
    queue_handler = QueueHandler(log_queue)
    sandbox_logger.addHandler(queue_handler)

    listener = QueueListener(log_queue, sandbox_loki_handler)
    listener.start()

    return sandbox_logger, listener
