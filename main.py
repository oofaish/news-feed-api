from feed_processor import main as feed_processor_main
from filters import main as filters_main
from utils import SetupClient


if __name__ == "__main__":
    with SetupClient():
        feed_processor_main()
        filters_main()
