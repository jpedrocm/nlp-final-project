import scrapy

class LyricsMusicItem(scrapy.Item):
    title = scrapy.Field()
    link = scrapy.Field()
    genre = scrapy.Field()
    lyrics = scrapy.Field()
