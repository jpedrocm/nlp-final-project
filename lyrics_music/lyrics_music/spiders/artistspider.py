from scrapy.spiders import Spider
from lyrics_music.items import ArtistMusicItem
from scrapy.http    import Request
import re

#scrapy crawl artistspider  --set CLOSESPIDER_PAGECOUNT=5 

class LyricsSpider(Spider):
	name = "artistspider"
	allowed_domains = ["letras.mus.br"]
	start_urls = ["https://www.letras.mus.br/mais-acessadas/axe/", "https://www.letras.mus.br/mais-acessadas/samba/",
	"https://www.letras.mus.br/mais-acessadas/sertanejo/", "https://www.letras.mus.br/mais-acessadas/mpb/", 
	"https://www.letras.mus.br/mais-acessadas/funk/"]

	def parse(self, response):
		links = response.xpath("//ol[@class='top-list_art']/li/a/@href").extract() #artist page
		items = []
		for link in links:
			link = "https://www.letras.mus.br" + link
			item = ArtistMusicItem()
			item["link"] = link	
			items.append(item)
		return items
