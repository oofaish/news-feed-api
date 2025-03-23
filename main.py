import logging

from ai import main as ai_main
from feed_processor import main as feed_processor_main
from utils import SetupClient


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting news feed API processing")
    try:
        with SetupClient():
            logger.info("Starting feed processing")
            feed_processor_main()
            logger.info("Feed processing completed, starting AI processing")
            ai_main()
            logger.info("AI processing completed")
    except Exception as e:
        logger.exception(f"Error in main process: {e}")
    logger.info("News feed API processing completed")
