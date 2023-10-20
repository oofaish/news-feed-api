nyt = [
    "https://rss.nytimes.com/services/xml/rss/nyt/Europe.xml"
    "https://rss.nytimes.com/services/xml/rss/nyt/MostShared.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/EnergyEnvironment.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/PersonalTech.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Climate.xml",
    # "https://rss.nytimes.com/services/xml/rss/nyt/Space.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Well.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml",
    # "https://rss.nytimes.com/services/xml/rss/nyt/ArtandDesign.xml",
    # "https://rss.nytimes.com/services/xml/rss/nyt/Theater.xml",
    # "https://rss.nytimes.com/services/xml/rss/nyt/FashionandStyle.xml",
    # "https://rss.nytimes.com/services/xml/rss/nyt/Weddings.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/DiningandWine.xml",
    # "https://rss.nytimes.com/services/xml/rss/nyt/Weddings.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/tmagazine.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/RealEstate.xml",
    # "https://feeds.a.dj.com/rss/RSSLifestyle.xml",
]

ft = [
    "https://www.ft.com/rss/home/uk",
    "https://www.ft.com/news-feed?format=rss&page=1",
    "https://www.ft.com/news-feed?format=rss&page=2",
    "https://www.ft.com/news-feed?format=rss&page=3",
    # "https://www.ft.com/news-feed?format=rss&page=4",
    # "https://www.ft.com/news-feed?format=rss&page=5",
    # "https://www.ft.com/news-feed?format=rss&page=6",
    # "https://www.ft.com/news-feed?format=rss&page=7",
    "https://www.ft.com/technology?format=rss&page=1",
    # "https://www.ft.com/technology?format=rss&page=2",
    # "https://www.ft.com/technology?format=rss&page=3",
    "https://www.ft.com/markets?format=rss&page=1",
    "https://www.ft.com/markets?format=rss&page=2",
    # "https://www.ft.com/markets?format=rss&page=3",
]

wsj = [
    "https://feeds.a.dj.com/rss/RSSWSJD.xml",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://feeds.a.dj.com/rss/RSSLifestyle.xml",
    "https://feeds.a.dj.com/rss/RSSOpinion.xml",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
]

guardian = [
    "https://www.theguardian.com/uk/rss",
    "https://www.theguardian.com/uk/lifeandstyle/rss",
]

FEEDS_BY_PUBLOCATION = {"Guardian": guardian, "FT": ft, "NY Times": nyt, "WSJ": wsj}
