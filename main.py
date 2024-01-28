from ai import main as ai_main
from feed_processor import main as feed_processor_main
from utils import SetupClient


if __name__ == "__main__":
    with SetupClient():
        feed_processor_main()
        ai_main()
        # filters_main()
